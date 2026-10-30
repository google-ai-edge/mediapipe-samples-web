/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import template from '../templates/audio-transcriber.html?raw';
import { ModelSelector, type ModelSelection } from '../components/model-selector';
import { ViewToggle } from '../components/view-toggle';
import { InferenceTimer } from '../components/inference-timer';
import { isModelCached } from '../components/model-cache';
import { AUDIO_TRANSCRIBER_MODELS, hostedModelDownloadUrl, type HostedModel } from '../components/model-registry';
import type { TranscriptionSegment } from '@mediapipe/tasks-audio';
import AudioTranscriberWorker from '../workers/audio-transcriber.worker.ts?worker';

function formatTimestampMs(ms: number): string {
  const totalSec = Math.max(0, ms / 1000);
  const mins = Math.floor(totalSec / 60);
  const secs = (totalSec % 60).toFixed(1).padStart(4, '0');
  return `${String(mins).padStart(2, '0')}:${secs}`;
}

function parsePcmWavFallback(buffer: ArrayBuffer): { samples: Float32Array; sampleRate: number } | null {
  if (buffer.byteLength <= 44) return null;
  const view = new DataView(buffer);
  const riff =
    String.fromCharCode(view.getUint8(0)) +
    String.fromCharCode(view.getUint8(1)) +
    String.fromCharCode(view.getUint8(2)) +
    String.fromCharCode(view.getUint8(3));
  if (riff !== 'RIFF') return null;
  const channels = view.getUint16(22, true) || 1;
  const sampleRate = view.getUint32(24, true) || 16000;
  const bitsPerSample = view.getUint16(34, true) || 16;
  if (bitsPerSample !== 16) return null;

  let dataOffset = 44;
  let dataLength = buffer.byteLength - 44;
  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const chunkId =
      String.fromCharCode(view.getUint8(offset)) +
      String.fromCharCode(view.getUint8(offset + 1)) +
      String.fromCharCode(view.getUint8(offset + 2)) +
      String.fromCharCode(view.getUint8(offset + 3));
    const chunkSize = view.getUint32(offset + 4, true);
    if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataLength = Math.min(chunkSize, buffer.byteLength - dataOffset);
      break;
    }
    offset += 8 + chunkSize + (chunkSize % 2);
  }

  const totalInt16 = Math.floor(dataLength / 2);
  const frameCount = Math.floor(totalInt16 / channels);
  const out = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const sample = view.getInt16(dataOffset + (i * channels + c) * 2, true);
      sum += sample / 32768.0;
    }
    out[i] = sum / channels;
  }
  return { samples: out, sampleRate };
}

class AudioTranscriberTask {
  private container: HTMLElement;
  private worker: Worker | null = null;
  private modelSelector: ModelSelector | null = null;
  private inferenceTimer = new InferenceTimer();

  private selectedModelId = AUDIO_TRANSCRIBER_MODELS[0].id;
  private loadingModelId: string | null = null;
  private loadedModelId: string | null = null;
  private activeInitToken = 0;
  private customModelFile: File | null = null;
  private isModelReady = false;
  private isLoadingModel = false;
  private isTranscribing = false;
  private isRecording = false;
  private delegate: 'CPU' | 'GPU' = 'GPU';
  private transcriptionMode: 'streaming' | 'unary' = 'streaming';
  private enableInterimResults = true;
  private sessionId = 0;
  private pendingActionAfterLoad: (() => void) | null = null;

  // Decoded clip state
  private clipSamples: Float32Array | null = null;
  private clipSampleRate = 16000;
  private clipLabel = 'English Greeting (7.2s)';

  // Live microphone state
  private micAudioCtx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private micProcessor: ScriptProcessorNode | null = null;
  private micAnalyser: AnalyserNode | null = null;
  private micVisualizerRaf: number | null = null;
  private micChunkBuffers: Float32Array[] = [];
  private micBufferedSamples = 0;
  private isMicChunkInFlight = false;

  // UI elements
  private modeSelect!: HTMLSelectElement;
  private interimSelect!: HTMLSelectElement;
  private delegateSelect!: HTMLSelectElement;
  private viewFile!: HTMLElement;
  private viewMic!: HTMLElement;
  private clipMetaBadge!: HTMLElement;
  private fileWaveformCanvas!: HTMLCanvasElement;
  private audioPlayer!: HTMLAudioElement;
  private audioUpload!: HTMLInputElement;
  private runFileBtn!: HTMLButtonElement;
  private runFileBtnLabel!: HTMLElement;
  private stopFileBtn!: HTMLButtonElement;
  private recordBtn!: HTMLButtonElement;
  private visualizerCanvas!: HTMLCanvasElement;
  private micPlaceholder!: HTMLElement;
  private micStatusPill!: HTMLElement;
  private transcriptMetricsBadge!: HTMLElement;
  private transcriptPlaceholder!: HTMLElement;
  private transcriptConfirmedEl!: HTMLElement;
  private transcriptInterimEl!: HTMLElement;
  private segmentsContainer!: HTMLElement;
  private segmentsList!: HTMLElement;
  private copyBtn!: HTMLButtonElement;
  private copyLabel!: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async init() {
    this.container.innerHTML = template;
    this.bindElements();
    this.inferenceTimer.mount();
    this.setupViewToggle();
    this.setupModelSelector();
    this.bindEvents();
    this.initWorker();

    // Load default bundled sample clip
    void this.loadSampleClip('sample_english_greeting.wav', 'English Greeting (7.2s)');
    void this.tryAutoLoadCachedModel();
  }

  private bindElements() {
    this.modeSelect = document.getElementById('transcription-mode-select') as HTMLSelectElement;
    this.interimSelect = document.getElementById('interim-results-select') as HTMLSelectElement;
    this.delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
    this.viewFile = document.getElementById('view-file') as HTMLElement;
    this.viewMic = document.getElementById('view-microphone') as HTMLElement;
    this.clipMetaBadge = document.getElementById('clip-meta-badge') as HTMLElement;
    this.fileWaveformCanvas = document.getElementById('file-waveform-canvas') as HTMLCanvasElement;
    this.audioPlayer = document.getElementById('audio-player') as HTMLAudioElement;
    this.audioUpload = document.getElementById('audio-upload') as HTMLInputElement;
    this.runFileBtn = document.getElementById('run-file-transcription') as HTMLButtonElement;
    this.runFileBtnLabel = document.getElementById('run-file-btn-label') as HTMLElement;
    this.stopFileBtn = document.getElementById('stop-file-transcription') as HTMLButtonElement;
    this.recordBtn = document.getElementById('recordButton') as HTMLButtonElement;
    this.visualizerCanvas = document.getElementById('visualizer') as HTMLCanvasElement;
    this.micPlaceholder = document.getElementById('mic-placeholder') as HTMLElement;
    this.micStatusPill = document.getElementById('mic-status-pill') as HTMLElement;
    this.transcriptMetricsBadge = document.getElementById('transcript-metrics-badge') as HTMLElement;
    this.transcriptPlaceholder = document.getElementById('transcript-placeholder') as HTMLElement;
    this.transcriptConfirmedEl = document.getElementById('transcript-confirmed-text') as HTMLElement;
    this.transcriptInterimEl = document.getElementById('transcript-interim-text') as HTMLElement;
    this.segmentsContainer = document.getElementById('segments-container') as HTMLElement;
    this.segmentsList = document.getElementById('segments-list') as HTMLElement;
    this.copyBtn = document.getElementById('copy-transcript-btn') as HTMLButtonElement;
    this.copyLabel = document.getElementById('copy-transcript-label') as HTMLElement;
  }

  private setupViewToggle() {
    new ViewToggle(
      'view-mode-toggle',
      [
        { label: 'Sample & File', value: 'file', icon: 'audio_file' },
        { label: 'Microphone', value: 'microphone', icon: 'mic' },
      ],
      'file',
      (mode) => this.switchView(mode as 'file' | 'microphone')
    );
  }

  private switchView(mode: 'file' | 'microphone') {
    if (this.isRecording && mode !== 'microphone') {
      void this.stopRecording();
    }
    this.viewFile.style.display = mode === 'file' ? 'block' : 'none';
    this.viewMic.style.display = mode === 'microphone' ? 'block' : 'none';
  }

  private findModelSpec(id: string): HostedModel | undefined {
    return AUDIO_TRANSCRIBER_MODELS.find((m) => m.id === id);
  }

  private resolveModelUrl = (value: string): string | undefined => {
    const spec = this.findModelSpec(value);
    return spec ? hostedModelDownloadUrl(spec) : undefined;
  };

  private setupModelSelector() {
    const options = AUDIO_TRANSCRIBER_MODELS.map((m, idx) => ({
      label: m.name,
      value: m.id,
      isDefault: idx === 0,
    }));

    this.modelSelector = new ModelSelector(
      'model-selector-container',
      options,
      (selection: ModelSelection) => {
        if (selection.type === 'custom') {
          this.customModelFile = selection.file;
        } else {
          this.customModelFile = null;
          this.selectedModelId = selection.value;
        }
        this.loadSelectedModel();
      },
      {
        autoLoad: false,
        accept: '.litertlm,.task,.tflite',
        uploadLabel: 'Choose .litertlm Model',
        resolveUrl: this.resolveModelUrl,
        preferCached: true,
      }
    );

    const stdSelect = document.getElementById('model-selector-container-standard-select') as HTMLSelectElement | null;
    stdSelect?.addEventListener('change', () => {
      const nextId = stdSelect.value;
      this.selectedModelId = nextId;
      this.customModelFile = null;
      this.loadSelectedModel();
    });
  }

  private async tryAutoLoadCachedModel() {
    for (const model of AUDIO_TRANSCRIBER_MODELS) {
      const url = hostedModelDownloadUrl(model);
      if ((await isModelCached(url)) && !this.isLoadingModel && !this.isModelReady) {
        this.selectedModelId = model.id;
        const stdSelect = document.getElementById(
          'model-selector-container-standard-select'
        ) as HTMLSelectElement | null;
        if (stdSelect) stdSelect.value = model.id;
        this.loadSelectedModel();
        return;
      }
    }
    this.setStatus('Select a model on the left to begin', 'idle');
  }

  private bindEvents() {
    this.modeSelect.addEventListener('change', () => {
      this.transcriptionMode = this.modeSelect.value as 'streaming' | 'unary';
    });

    this.interimSelect.addEventListener('change', () => {
      this.enableInterimResults = this.interimSelect.value === 'true';
      if (this.isModelReady && !this.isLoadingModel) {
        this.loadSelectedModel();
      }
    });

    this.delegateSelect.addEventListener('change', () => {
      this.delegate = this.delegateSelect.value as 'CPU' | 'GPU';
      this.inferenceTimer.resetRollingWindow();
      if (this.isModelReady && !this.isLoadingModel) {
        this.loadSelectedModel();
      }
    });

    const sampleBtns = document.querySelectorAll<HTMLButtonElement>('#sample-clips-container .sample-btn');
    sampleBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        sampleBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const sampleFile = btn.dataset.sample || 'sample_english_greeting.wav';
        const label = btn.dataset.label || sampleFile;
        void this.loadSampleClip(sampleFile, label);
      });
    });

    this.audioUpload.addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        sampleBtns.forEach((b) => b.classList.remove('active'));
        void this.handleFileUpload(file);
      }
    });

    const dropzone = this.container.querySelector('.upload-dropzone');
    if (dropzone) {
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
      dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
      });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const dragEvent = e as DragEvent;
        if (dragEvent.dataTransfer?.files && dragEvent.dataTransfer.files.length > 0) {
          const file = dragEvent.dataTransfer.files[0];
          if (file.type.startsWith('audio/') || file.name.endsWith('.wav')) {
            sampleBtns.forEach((b) => b.classList.remove('active'));
            void this.handleFileUpload(file);
          }
        }
      });
    }

    this.runFileBtn.addEventListener('click', () => {
      this.triggerOrRun(() => this.runFileTranscription());
    });

    this.stopFileBtn.addEventListener('click', () => {
      this.cancelActiveTranscription();
    });

    this.recordBtn.addEventListener('click', () => {
      if (this.isRecording) {
        void this.stopRecording();
      } else {
        this.triggerOrRun(() => {
          void this.startRecording();
        });
      }
    });

    this.copyBtn.addEventListener('click', () => {
      const text = [this.transcriptConfirmedEl.textContent, this.transcriptInterimEl.textContent]
        .map((s) => (s || '').trim())
        .filter(Boolean)
        .join(' ');
      if (!text) return;
      navigator.clipboard?.writeText(text).then(() => {
        this.copyLabel.textContent = 'Copied!';
        setTimeout(() => {
          this.copyLabel.textContent = 'Copy';
        }, 1500);
      });
    });
  }

  private triggerOrRun(action: () => void) {
    if (this.isModelReady && !this.isLoadingModel) {
      action();
      return;
    }
    this.pendingActionAfterLoad = action;
    if (!this.isLoadingModel) {
      this.loadSelectedModel();
    }
  }

  private initWorker() {
    if (this.worker) {
      try {
        this.worker.onmessage = null;
        this.worker.onerror = null;
        this.worker.terminate();
      } catch (_) {}
    }
    this.worker = new AudioTranscriberWorker();
    this.worker.onmessage = (event: MessageEvent) => this.handleWorkerMessage(event);
    this.worker.onerror = (err) => {
      console.error('AudioTranscriber Worker Error:', err);
      this.isLoadingModel = false;
      this.isTranscribing = false;
      this.modelSelector?.setBusy(false);
      this.setStatus(`Error: ${err.message || 'Worker error'}`, 'error');
      this.updateControlsState();
    };
  }

  private loadSelectedModel() {
    if (!this.customModelFile && this.isLoadingModel && this.selectedModelId === this.loadingModelId) {
      return;
    }
    if (this.isRecording) {
      void this.stopRecording();
    }
    if (
      !this.worker ||
      this.isLoadingModel ||
      this.isTranscribing ||
      (this.loadedModelId !== null && this.loadedModelId !== this.selectedModelId)
    ) {
      this.isTranscribing = false;
      this.sessionId++;
      this.initWorker();
    }

    const myInitToken = ++this.activeInitToken;
    this.isLoadingModel = true;
    this.isModelReady = false;
    this.loadingModelId = this.customModelFile ? null : this.selectedModelId;
    this.loadedModelId = null;
    this.modelSelector?.setLoaded(null);
    this.modelSelector?.setBusy(true);
    const stdSelect = document.getElementById('model-selector-container-standard-select') as HTMLSelectElement | null;
    if (stdSelect) {
      stdSelect.disabled = false;
    }

    const spec = this.findModelSpec(this.selectedModelId);
    const label = this.customModelFile ? this.customModelFile.name : spec?.name || this.selectedModelId;
    this.modelSelector?.setStatus(`Loading ${label}...`);
    this.setStatus(`Loading ${label}...`, 'loading');
    this.updateControlsState();

    const baseUrl = import.meta.env.BASE_URL;
    if (this.customModelFile) {
      this.worker!.postMessage({
        type: 'INIT',
        initToken: myInitToken,
        modelFile: this.customModelFile,
        modelName: this.customModelFile.name,
        delegate: this.delegate,
        enableInterimResults: this.enableInterimResults,
        baseUrl,
      });
    } else {
      const modelAssetPath = this.resolveModelUrl(this.selectedModelId);
      this.worker!.postMessage({
        type: 'INIT',
        initToken: myInitToken,
        modelAssetPath,
        modelName: this.selectedModelId,
        delegate: this.delegate,
        enableInterimResults: this.enableInterimResults,
        baseUrl,
      });
    }
  }

  private handleWorkerMessage(event: MessageEvent) {
    const data = event.data;
    const { type } = data;

    if (type === 'LOAD_PROGRESS') {
      const { loaded, total } = data;
      this.modelSelector?.showProgress(loaded, total);
      const mb = (loaded / (1024 * 1024)).toFixed(1);
      if (total > 0) {
        const pct = Math.min(100, Math.round((loaded / total) * 100));
        const totalMb = (total / (1024 * 1024)).toFixed(1);
        this.setStatus(`Downloading model... ${pct}% (${mb} / ${totalMb} MB)`, 'loading');
      } else {
        this.setStatus(`Downloading model... ${mb} MB`, 'loading');
      }
      return;
    }

    if (type === 'MODEL_CACHED') {
      void this.modelSelector?.refreshCacheState();
      return;
    }

    if (type === 'DELEGATE_FALLBACK') {
      this.delegate = 'CPU';
      this.delegateSelect.value = 'CPU';
      this.inferenceTimer.resetRollingWindow();
      return;
    }

    if (type === 'INIT_DONE') {
      if (data.initToken !== undefined && data.initToken !== this.activeInitToken) {
        return;
      }
      this.isLoadingModel = false;
      this.loadingModelId = null;
      this.isModelReady = true;
      this.loadedModelId = this.customModelFile ? null : this.selectedModelId;
      this.modelSelector?.hideProgress();
      this.modelSelector?.setBusy(false);
      const spec = this.findModelSpec(this.selectedModelId);
      const label = this.customModelFile ? this.customModelFile.name : spec?.name || this.selectedModelId;
      this.modelSelector?.setLoaded(this.customModelFile ? null : this.selectedModelId);
      this.modelSelector?.setStatus(`✓ Active: ${label}`);
      this.setStatus('Ready', 'ready');
      this.transcriptMetricsBadge.textContent = 'Model Ready';
      if (!this.transcriptConfirmedEl.textContent && !this.transcriptInterimEl.textContent) {
        this.transcriptPlaceholder.textContent =
          'Model initialized and ready. Click "Transcribe Audio" or switch to "Microphone" to begin.';
      }
      this.updateControlsState();

      if (this.pendingActionAfterLoad) {
        const nextAction = this.pendingActionAfterLoad;
        this.pendingActionAfterLoad = null;
        nextAction();
      }
      return;
    }

    if (type === 'ERROR') {
      this.isLoadingModel = false;
      this.isTranscribing = false;
      this.isMicChunkInFlight = false;
      this.pendingActionAfterLoad = null;
      this.modelSelector?.hideProgress();
      this.modelSelector?.setBusy(false);
      this.modelSelector?.setStatus(`Error: ${data.error}`);
      this.setStatus(`Error: ${data.error}`, 'error');
      this.transcriptMetricsBadge.textContent = 'Error';
      this.updateControlsState();
      return;
    }

    if (data.sessionId !== undefined && data.sessionId !== this.sessionId) {
      return;
    }

    if (type === 'TRANSCRIBE_STREAM_UPDATE') {
      const {
        confirmedText,
        interimText,
        segments,
        audioProcessedSec,
        audioDurationSec,
        chunkLatencyMs,
        didRunWindow,
      } = data;

      if (didRunWindow && chunkLatencyMs > 0) {
        this.inferenceTimer.record(chunkLatencyMs, this.delegate);
      }
      this.renderTranscript(confirmedText, interimText, segments);

      const progressFrac = audioDurationSec > 0 ? Math.min(1, audioProcessedSec / audioDurationSec) : 0;
      this.drawClipWaveform(this.clipSamples, progressFrac);
      const winMetric = chunkLatencyMs > 0 ? ` (${Math.round(chunkLatencyMs)} ms/win)` : '';
      this.transcriptMetricsBadge.textContent = `Streaming · ${audioProcessedSec.toFixed(1)}s / ${audioDurationSec.toFixed(1)}s${winMetric}`;
      this.setStatus(`Transcribing (${audioProcessedSec.toFixed(1)}s / ${audioDurationSec.toFixed(1)}s)...`, 'busy');
      return;
    }

    if (type === 'TRANSCRIBE_RESULT') {
      this.isTranscribing = false;
      const { result, inferenceTime, audioDurationSec } = data;
      this.inferenceTimer.record(inferenceTime, this.delegate);
      this.drawClipWaveform(this.clipSamples, 1);

      const finalText = (result?.text || '').trim();
      const segments: TranscriptionSegment[] = result?.segments || [];
      this.renderTranscript(finalText || '(No speech detected)', '', segments);

      const rtf = audioDurationSec > 0 ? inferenceTime / 1000 / audioDurationSec : 0;
      const speedX = rtf > 0 ? (1 / rtf).toFixed(1) : '-';
      this.transcriptMetricsBadge.textContent = `Done · ${Math.round(inferenceTime)} ms · RTF ${rtf.toFixed(2)}x (${speedX}x RT)`;
      this.setStatus('Ready', 'ready');
      this.updateControlsState();
      return;
    }

    if (type === 'MIC_STREAM_UPDATE') {
      this.isMicChunkInFlight = false;
      const { confirmedText, interimText, segments, chunkLatencyMs, didRunWindow, audioProcessedSec } = data;
      if (didRunWindow && chunkLatencyMs > 0) {
        this.inferenceTimer.record(chunkLatencyMs, this.delegate);
      }
      const displayInterim = interimText || (confirmedText ? '' : 'Listening...');
      this.renderTranscript(confirmedText, displayInterim, segments);
      const winMetric = chunkLatencyMs > 0 ? ` (${Math.round(chunkLatencyMs)} ms/win)` : '';
      this.micStatusPill.textContent = `Live · ${audioProcessedSec.toFixed(1)}s${winMetric}`;
      this.transcriptMetricsBadge.textContent = `Live Mic · ${audioProcessedSec.toFixed(1)}s`;
      return;
    }

    if (type === 'MIC_STREAM_FINAL') {
      this.isMicChunkInFlight = false;
      const { confirmedText, segments, flushLatencyMs, audioProcessedSec } = data;
      this.inferenceTimer.record(flushLatencyMs, this.delegate);
      this.renderTranscript(confirmedText || '(No speech detected)', '', segments);
      this.micStatusPill.textContent = '16 kHz PCM · 500 ms chunks';
      this.transcriptMetricsBadge.textContent = `Recorded ${audioProcessedSec.toFixed(1)}s`;
      this.setStatus('Ready', 'ready');
      this.updateControlsState();
      return;
    }
  }

  private renderTranscript(confirmedText: string, interimText: string, segments: TranscriptionSegment[]) {
    const hasConfirmed = Boolean(confirmedText && confirmedText.trim());
    const hasInterim = Boolean(interimText && interimText.trim());

    this.transcriptPlaceholder.style.display = hasConfirmed || hasInterim ? 'none' : 'inline';
    this.transcriptConfirmedEl.textContent = hasConfirmed ? confirmedText : '';
    this.transcriptInterimEl.textContent = hasInterim ? (hasConfirmed ? ` ${interimText}` : interimText) : '';
    this.copyBtn.disabled = !(hasConfirmed || hasInterim);

    if (segments && segments.length > 0) {
      this.segmentsContainer.style.display = 'block';
      this.segmentsList.innerHTML = '';
      segments.forEach((seg) => {
        if (!seg.transcript || !seg.transcript.trim()) return;
        const row = document.createElement('div');
        row.className = `asr-segment-row${seg.isFinal === false ? ' interim' : ''}`;

        const timeSpan = document.createElement('span');
        timeSpan.className = 'asr-segment-time';
        timeSpan.textContent = `[${formatTimestampMs(seg.fromTimestampMs || 0)} – ${formatTimestampMs(
          seg.toTimestampMs || 0
        )}]`;

        const textSpan = document.createElement('span');
        textSpan.className = 'asr-segment-text';
        textSpan.textContent = seg.transcript.trim();

        row.appendChild(timeSpan);
        row.appendChild(textSpan);
        this.segmentsList.appendChild(row);
      });
      this.segmentsList.scrollTop = this.segmentsList.scrollHeight;
    } else {
      this.segmentsContainer.style.display = 'none';
      this.segmentsList.innerHTML = '';
    }
  }

  private async decodeAudioArrayBuffer(
    arrayBuffer: ArrayBuffer
  ): Promise<{ samples: Float32Array; sampleRate: number }> {
    const wavFallback = parsePcmWavFallback(arrayBuffer);
    try {
      const AC = window.AudioContext || (window as any).webkitAudioContext;
      if (AC) {
        const ctx = new AC({ sampleRate: 16000 });
        try {
          const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
          const channel = decoded.getChannelData(0);
          const copy = new Float32Array(channel.length);
          copy.set(channel);
          return { samples: copy, sampleRate: decoded.sampleRate || 16000 };
        } finally {
          void ctx.close().catch(() => {});
        }
      }
    } catch (_) {
      if (wavFallback) return wavFallback;
    }
    if (wavFallback) return wavFallback;
    throw new Error('Failed to decode audio file.');
  }

  private async loadSampleClip(sampleFile: string, label: string) {
    try {
      const base = import.meta.env.BASE_URL.endsWith('/') ? import.meta.env.BASE_URL : `${import.meta.env.BASE_URL}/`;
      const url = `${base}audio/${sampleFile}`;
      this.audioPlayer.src = url;
      this.clipLabel = label;

      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const { samples, sampleRate } = await this.decodeAudioArrayBuffer(buf);
      this.clipSamples = samples;
      this.clipSampleRate = sampleRate;

      const durSec = (samples.length / sampleRate).toFixed(1);
      this.clipMetaBadge.textContent = `${durSec}s · ${(sampleRate / 1000).toFixed(0)} kHz`;
      this.drawClipWaveform(samples, 0);
      this.updateControlsState();
    } catch (err: any) {
      console.error('Error loading sample clip:', err);
    }
  }

  private async handleFileUpload(file: File) {
    try {
      const url = URL.createObjectURL(file);
      this.audioPlayer.src = url;
      this.clipLabel = file.name;

      const buf = await file.arrayBuffer();
      const { samples, sampleRate } = await this.decodeAudioArrayBuffer(buf);
      this.clipSamples = samples;
      this.clipSampleRate = sampleRate;

      const durSec = (samples.length / sampleRate).toFixed(1);
      this.clipMetaBadge.textContent = `${file.name} (${durSec}s)`;
      this.drawClipWaveform(samples, 0);
      this.updateControlsState();
    } catch (err: any) {
      this.setStatus(`Error decoding audio file: ${err.message}`, 'error');
    }
  }

  private drawClipWaveform(samples: Float32Array | null, progressFrac = 0) {
    const canvas = this.fileWaveformCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (progressFrac > 0) {
      ctx.fillStyle = 'rgba(0, 127, 139, 0.10)';
      ctx.fillRect(0, 0, Math.round(w * Math.min(1, progressFrac)), h);
    }

    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (!samples || samples.length === 0) return;

    const numBars = 140;
    const step = Math.max(1, Math.floor(samples.length / numBars));
    const barWidth = Math.max(2, w / numBars - 2);
    const progressBarIdx = Math.floor(Math.min(1, progressFrac) * numBars);

    const peaks = new Float32Array(numBars);
    let maxPeak = 0.02;
    for (let i = 0; i < numBars; i++) {
      let peak = 0;
      const start = i * step;
      for (let j = 0; j < step && start + j < samples.length; j++) {
        const abs = Math.abs(samples[start + j]);
        if (abs > peak) peak = abs;
      }
      peaks[i] = peak;
      if (peak > maxPeak) maxPeak = peak;
    }

    for (let i = 0; i < numBars; i++) {
      const normalized = peaks[i] / maxPeak;
      const barHeight = Math.max(4, Math.min(h - 10, normalized * (h - 12)));
      const x = i * (w / numBars);
      const y = (h - barHeight) / 2;

      ctx.fillStyle = progressFrac > 0 && i <= progressBarIdx ? '#007f8b' : 'rgba(0, 127, 139, 0.55)';
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, 2);
      ctx.fill();
    }
  }

  private runFileTranscription() {
    if (!this.worker || !this.isModelReady || !this.clipSamples || this.isTranscribing) return;

    this.sessionId++;
    this.isTranscribing = true;
    this.drawClipWaveform(this.clipSamples, 0);
    this.renderTranscript('', 'Transcribing...', []);
    this.transcriptMetricsBadge.textContent =
      this.transcriptionMode === 'streaming' ? 'Streaming (500 ms chunks)...' : 'Running Unary ASR...';
    this.setStatus(`Transcribing ${this.clipLabel}...`, 'busy');
    this.updateControlsState();

    const copy = new Float32Array(this.clipSamples);
    this.worker.postMessage(
      {
        type: 'TRANSCRIBE_CLIP',
        audioData: copy,
        sampleRate: this.clipSampleRate,
        streamingMode: this.transcriptionMode === 'streaming',
        chunkDurationMs: 500,
        sessionId: this.sessionId,
      },
      [copy.buffer]
    );
  }

  private cancelActiveTranscription() {
    this.sessionId++;
    this.isTranscribing = false;
    this.worker?.postMessage({ type: 'CANCEL', sessionId: this.sessionId });
    this.transcriptMetricsBadge.textContent = 'Stopped';
    this.setStatus('Ready', 'ready');
    this.updateControlsState();
  }

  private async startRecording() {
    if (this.isRecording || !this.worker || !this.isModelReady) return;
    try {
      this.sessionId++;
      const currentSession = this.sessionId;
      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const AC = window.AudioContext || (window as any).webkitAudioContext;
      this.micAudioCtx = new AC({ sampleRate: 16000 });
      await this.micAudioCtx.resume();
      const actualRate = this.micAudioCtx.sampleRate || 16000;

      this.micSource = this.micAudioCtx.createMediaStreamSource(this.micStream);
      this.micAnalyser = this.micAudioCtx.createAnalyser();
      this.micAnalyser.fftSize = 2048;
      this.micProcessor = this.micAudioCtx.createScriptProcessor(4096, 1, 1);

      this.micSource.connect(this.micAnalyser);
      this.micAnalyser.connect(this.micProcessor);
      this.micProcessor.connect(this.micAudioCtx.destination);

      this.micChunkBuffers = [];
      this.micBufferedSamples = 0;
      this.isMicChunkInFlight = false;
      this.isRecording = true;

      this.worker.postMessage({ type: 'START_MIC_STREAM', sessionId: currentSession });
      this.renderTranscript('', 'Listening... speak into your microphone.', []);
      this.micPlaceholder.style.display = 'none';
      this.recordBtn.innerHTML = '<span class="material-icons">stop</span><span>Stop Recording</span>';
      this.recordBtn.classList.add('recording');
      this.setStatus('Recording live audio...', 'busy');
      this.startMicVisualizer();

      const targetChunkSamples = Math.round(actualRate * 0.5);
      this.micProcessor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (!this.isRecording || this.sessionId !== currentSession) return;
        const input = e.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        this.micChunkBuffers.push(copy);
        this.micBufferedSamples += copy.length;

        if (this.micBufferedSamples >= targetChunkSamples && !this.isMicChunkInFlight) {
          const merged = new Float32Array(this.micBufferedSamples);
          let offset = 0;
          for (const part of this.micChunkBuffers) {
            merged.set(part, offset);
            offset += part.length;
          }
          this.micChunkBuffers = [];
          this.micBufferedSamples = 0;
          this.isMicChunkInFlight = true;
          this.worker?.postMessage(
            {
              type: 'TRANSCRIBE_MIC_CHUNK',
              audioData: merged,
              sampleRate: actualRate,
              sessionId: currentSession,
            },
            [merged.buffer]
          );
        }
      };
    } catch (err: any) {
      console.error('Microphone access error:', err);
      this.setStatus(`Microphone error: ${err.message}`, 'error');
    }
  }

  private async stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this.isMicChunkInFlight = false;
    const actualRate = this.micAudioCtx?.sampleRate || 16000;

    if (this.micVisualizerRaf) {
      cancelAnimationFrame(this.micVisualizerRaf);
      this.micVisualizerRaf = null;
    }

    try {
      this.micProcessor?.disconnect();
      this.micAnalyser?.disconnect();
      this.micSource?.disconnect();
      this.micStream?.getTracks().forEach((t) => t.stop());
      await this.micAudioCtx?.close();
    } catch (_) {}

    this.micProcessor = null;
    this.micAnalyser = null;
    this.micSource = null;
    this.micStream = null;
    this.micAudioCtx = null;

    let remaining: Float32Array | undefined;
    if (this.micBufferedSamples > 0) {
      remaining = new Float32Array(this.micBufferedSamples);
      let offset = 0;
      for (const part of this.micChunkBuffers) {
        remaining.set(part, offset);
        offset += part.length;
      }
    }
    this.micChunkBuffers = [];
    this.micBufferedSamples = 0;

    this.recordBtn.innerHTML = '<span class="material-icons">mic</span><span>Start Recording</span>';
    this.recordBtn.classList.remove('recording');
    if (this.visualizerCanvas) {
      const ctx = this.visualizerCanvas.getContext('2d');
      ctx?.clearRect(0, 0, this.visualizerCanvas.width, this.visualizerCanvas.height);
    }
    this.micPlaceholder.style.display = 'flex';

    this.worker?.postMessage({
      type: 'STOP_MIC_STREAM',
      audioData: remaining,
      sampleRate: actualRate,
      sessionId: this.sessionId,
    });
  }

  private startMicVisualizer() {
    const canvas = this.visualizerCanvas;
    if (!canvas || !this.micAnalyser) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bufferLength = this.micAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      if (!this.isRecording || !this.micAnalyser) return;
      this.micVisualizerRaf = requestAnimationFrame(draw);
      this.micAnalyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#007f8b';
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();
  }

  private updateControlsState() {
    const busy = this.isLoadingModel || this.isTranscribing;
    this.runFileBtn.disabled = busy || !this.clipSamples;
    this.runFileBtnLabel.textContent = this.isModelReady ? 'Transcribe Audio' : 'Load Model & Transcribe';
    this.stopFileBtn.style.display = this.isTranscribing ? 'inline-flex' : 'none';
    this.recordBtn.disabled = this.isLoadingModel;
  }

  private setStatus(msg: string, state?: string) {
    this.inferenceTimer.setStatus(state || msg, state ? msg : undefined);
  }

  async cleanup() {
    this.sessionId++;
    if (this.isRecording) {
      await this.stopRecording();
    }
    this.inferenceTimer.cleanup();
    if (this.worker) {
      this.worker.postMessage({ type: 'CLEANUP' });
      this.worker.terminate();
      this.worker = null;
    }
  }
}

let activeTask: AudioTranscriberTask | null = null;

export async function setupAudioTranscriber(container: HTMLElement) {
  activeTask = new AudioTranscriberTask(container);
  await activeTask.init();
}

export function cleanupAudioTranscriber() {
  if (activeTask) {
    void activeTask.cleanup();
    activeTask = null;
  }
}
