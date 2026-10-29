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

import type { TranscriptionSegment } from '@mediapipe/tasks-audio';
import { BaseAudioTask, BaseAudioTaskOptions } from '../components/base-audio-task';
import type { ModelSelectorConfig } from '../components/model-selector';
import { ASR_MODELS, hostedModelDownloadUrl } from '../components/model-registry';

// @ts-ignore
import template from '../templates/audio-transcriber.html?raw';

/** Sample rate the transcriber is configured for; audio is resampled to it. */
const SAMPLE_RATE_HZ = 16000;
/** Size of the chunks an uploaded file is fed in, to show progress. */
const FILE_CHUNK_SECONDS = 2;

const DEFAULT_MODEL = ASR_MODELS[0];

class AudioTranscriberTask extends BaseAudioTask {
  private enableInterimResults = true;

  // Transcript state
  private finalSegments: TranscriptionSegment[] = [];
  private interimSegment: TranscriptionSegment | undefined;
  private totalInferenceMs = 0;
  private totalAudioMs = 0;

  /**
   * Identifies the current recording or file transcription. Results of an
   * earlier session (e.g. a recording still being finalized when a file is
   * transcribed) are ignored.
   */
  private sessionId = 0;

  // Microphone streaming state
  private pendingFinish = false;
  /** Whether a microphone chunk is being transcribed by the worker. */
  private chunkInFlight = false;
  /** Microphone audio captured while the worker was busy. */
  private queuedAudio: Float32Array[] = [];

  // File transcription state
  private fileChunksTotal = 0;
  private fileChunksDone = 0;
  private fileStartMs = 0;

  // Waveform visualization (same look as the Audio Classifier)
  private WAVEFORM_HISTORY_SIZE = 8000;
  private waveformBuffer = new Float32Array(this.WAVEFORM_HISTORY_SIZE);
  private canvasElement!: HTMLCanvasElement;
  private canvasCtx!: CanvasRenderingContext2D;

  private transcriptEl!: HTMLElement;
  private copyBtn!: HTMLButtonElement;
  private clearBtn!: HTMLButtonElement;
  private rtfEl!: HTMLElement;

  constructor(options: BaseAudioTaskOptions) {
    super(options);
  }

  protected override getModelSelectorConfig(): ModelSelectorConfig {
    // The models are 100-700 MB: only download once the user asks for it.
    return {
      autoLoad: false,
      accept: '.litertlm',
      uploadLabel: 'Choose .litertlm File',
      resolveUrl: (value) => this.resolveModelUrl(value),
      preferCached: true,
    };
  }

  protected override onInitializeUI() {
    this.canvasElement = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    if (this.canvasElement) {
      this.canvasCtx = this.canvasElement.getContext('2d')!;
    }
    this.transcriptEl = document.getElementById('asr-transcript') as HTMLElement;
    this.copyBtn = document.getElementById('asr-copy-btn') as HTMLButtonElement;
    this.clearBtn = document.getElementById('asr-clear-btn') as HTMLButtonElement;
    this.rtfEl = document.getElementById('asr-rtf') as HTMLElement;

    this.modelSelector?.updateOptions(
      ASR_MODELS.map((m) => ({ label: m.name, value: m.id, isDefault: m.id === DEFAULT_MODEL.id }))
    );

    const interimCheckbox = document.getElementById('interim-results') as HTMLInputElement;
    interimCheckbox.addEventListener('change', () => {
      this.enableInterimResults = interimCheckbox.checked;
      // Interim results are an engine option: reload if a model is active.
      if (this.modelRequested) this.initializeTask();
    });

    this.copyBtn.addEventListener('click', () => {
      navigator.clipboard?.writeText(this.transcriptText());
    });
    this.clearBtn.addEventListener('click', () => this.clearTranscript());

    this.drawWaveform();
  }

  protected override resolveModelUrl(modelName: string): string | undefined {
    const model = ASR_MODELS.find((m) => m.id === modelName);
    return model ? hostedModelDownloadUrl(model) : undefined;
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      sampleRateHz: SAMPLE_RATE_HZ,
      enableInterimResults: this.enableInterimResults,
    };
  }

  protected override async initializeTask(): Promise<void> {
    if (this.isRecording) this.stopRecording();
    this.sessionId++;
    this.pendingFinish = false;
    this.fileChunksTotal = 0;
    this.setRecordingEnabled(false);
    this.setFileButtonEnabled(false);
    await super.initializeTask();
  }

  protected override handleInitDone(data?: any) {
    super.handleInitDone(data);
    if (!this.isWorkerReady) return;
    this.setRecordingEnabled(true);
    this.setFileButtonEnabled(true);
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const data = event.data;
    if (data.type === 'TRANSCRIPTION_RESULT') {
      this.handleResult(data);
    } else if (data.type === 'TRANSCRIPTION_ERROR') {
      if (data.sessionId !== this.sessionId) return;
      this.chunkInFlight = false;
      console.error('Transcription error:', data.error);
      this.updateStatus(`Error: ${data.error}`);
      this.fileChunksTotal = 0;
      this.setFileButtonEnabled(true);
    } else {
      super.handleWorkerMessage(event);
    }
  }

  private handleResult(data: {
    sessionId: number;
    segments: TranscriptionSegment[];
    final: boolean;
    inferenceTime: number;
    audioMs: number;
  }) {
    if (data.sessionId !== this.sessionId) return;
    this.totalInferenceMs += data.inferenceTime;
    this.totalAudioMs += data.audioMs;
    if (data.audioMs > 0) this.updateInferenceTime(data.inferenceTime);

    for (const segment of data.segments) {
      if (!segment.transcript?.trim()) continue;
      if (segment.isFinal) {
        this.finalSegments.push(segment);
        this.interimSegment = undefined;
      } else {
        this.interimSegment = segment;
      }
    }
    if (data.final) this.interimSegment = undefined;
    this.renderTranscript();

    if (this.totalAudioMs > 0) {
      this.rtfEl.textContent = `RTF ${(this.totalInferenceMs / this.totalAudioMs).toFixed(3)}`;
    }

    // File transcription progress.
    if (this.fileChunksTotal > 0) {
      this.fileChunksDone++;
      this.setFileProgress(this.fileChunksDone / this.fileChunksTotal);
      if (data.final) {
        const elapsed = performance.now() - this.fileStartMs;
        this.updateStatus(`Transcribed ${(this.totalAudioMs / 1000).toFixed(1)}s of audio in ${Math.round(elapsed)}ms`);
        this.fileChunksTotal = 0;
        this.setFileButtonEnabled(true);
      }
    } else if (data.final) {
      this.pendingFinish = false;
      this.updateStatus('Ready');
    } else {
      // Microphone: send the audio captured while this chunk was processed.
      this.chunkInFlight = false;
      this.flushQueuedAudio(/* final= */ false);
    }
  }

  /**
   * Sends all queued microphone audio as one chunk. Only one chunk is in
   * flight at a time, so a slow device transcribes larger chunks instead of
   * falling further and further behind.
   */
  private flushQueuedAudio(final: boolean) {
    if (!this.worker || this.chunkInFlight) return;
    if (this.queuedAudio.length === 0 && !final) return;
    const audioData = concat(this.queuedAudio);
    this.queuedAudio = [];
    this.chunkInFlight = !final;
    this.worker.postMessage({ type: 'STREAM_CHUNK', sessionId: this.sessionId, audioData, final }, [audioData.buffer]);
  }

  // --- Microphone -------------------------------------------------------

  protected override async startRecording() {
    this.clearTranscript();
    this.sessionId++;
    this.chunkInFlight = false;
    this.queuedAudio = [];
    await super.startRecording();
    if (this.isRecording) this.updateStatus('Listening...');
  }

  protected override stopRecording() {
    const wasRecording = this.isRecording;
    // BaseAudioTask.stopRecording() clears the results; keep the transcript.
    this.pendingFinish = wasRecording;
    super.stopRecording();
    if (wasRecording && this.isWorkerReady && this.worker) {
      this.updateStatus('Finalizing...');
      // Chunks are processed in order, so the final (flushing) chunk can be
      // sent right away even if another chunk is still in flight.
      this.chunkInFlight = false;
      this.flushQueuedAudio(/* final= */ true);
    }
  }

  protected override processAudioData(data: Float32Array, sampleRate: number): void {
    this.visualizeWaveform(data);
    if (!this.isWorkerReady || !this.worker) return;
    // The record AudioContext asks for 16 kHz, but not every browser honors it.
    const samples = sampleRate === SAMPLE_RATE_HZ ? new Float32Array(data) : resample(data, sampleRate, SAMPLE_RATE_HZ);
    this.queuedAudio.push(samples);
    this.flushQueuedAudio(/* final= */ false);
  }

  // --- File -------------------------------------------------------------

  protected override onAudioFileLoaded(file: File) {
    const runBtn = document.getElementById('run-file-transcription') as HTMLButtonElement;
    if (!runBtn) return;
    this.setFileButtonEnabled(this.isWorkerReady);
    runBtn.onclick = async () => {
      this.setFileButtonEnabled(false);
      this.updateStatus('Decoding audio...');
      try {
        await this.transcribeFile(file);
      } catch (err: any) {
        console.error(err);
        this.updateStatus(`File Error: ${err?.message ?? err}`);
        this.setFileButtonEnabled(true);
      }
    };
  }

  private async transcribeFile(file: File) {
    if (!this.worker || !this.isWorkerReady) return;
    const samples = await decodeToMono(file, SAMPLE_RATE_HZ);

    this.clearTranscript();
    this.sessionId++;
    const chunkSize = FILE_CHUNK_SECONDS * SAMPLE_RATE_HZ;
    this.fileChunksTotal = Math.max(1, Math.ceil(samples.length / chunkSize));
    this.fileChunksDone = 0;
    this.fileStartMs = performance.now();
    this.setFileProgress(0);
    this.updateStatus('Transcribing...');

    for (let i = 0; i < this.fileChunksTotal; i++) {
      const chunk = samples.slice(i * chunkSize, (i + 1) * chunkSize);
      const final = i === this.fileChunksTotal - 1;
      this.worker.postMessage({ type: 'STREAM_CHUNK', sessionId: this.sessionId, audioData: chunk, final }, [
        chunk.buffer,
      ]);
    }
  }

  private setFileProgress(fraction: number) {
    const container = document.getElementById('file-progress');
    const bar = document.getElementById('file-progress-bar');
    if (!container || !bar) return;
    container.style.display = fraction >= 1 ? 'none' : 'block';
    bar.style.width = `${Math.round(fraction * 100)}%`;
  }

  // --- Transcript -------------------------------------------------------

  protected override clearResults(): void {
    // Called on view switches and after recording stops. Keep the transcript
    // after a recording so it can be read and copied; clear it otherwise.
    if (this.pendingFinish) return;
    this.clearTranscript();
  }

  private clearTranscript() {
    this.finalSegments = [];
    this.interimSegment = undefined;
    this.totalInferenceMs = 0;
    this.totalAudioMs = 0;
    if (this.rtfEl) this.rtfEl.textContent = '';
    this.renderTranscript();
  }

  private transcriptText(): string {
    return this.finalSegments
      .map((s) => s.transcript.trim())
      .join(' ')
      .trim();
  }

  private renderTranscript() {
    if (!this.transcriptEl) return;
    this.transcriptEl.innerHTML = '';
    const segments = this.interimSegment ? [...this.finalSegments, this.interimSegment] : this.finalSegments;
    for (const segment of segments) {
      const row = document.createElement('div');
      row.className = `asr-segment${segment.isFinal ? '' : ' interim'}`;
      const time = document.createElement('span');
      time.className = 'asr-segment-time';
      time.textContent = formatTime(segment.fromTimestampMs);
      const text = document.createElement('span');
      text.className = 'asr-segment-text';
      text.textContent = segment.transcript.trim();
      row.append(time, text);
      this.transcriptEl.appendChild(row);
    }
    this.transcriptEl.scrollTop = this.transcriptEl.scrollHeight;
    const hasText = this.finalSegments.length > 0;
    if (this.copyBtn) this.copyBtn.disabled = !hasText;
    if (this.clearBtn) this.clearBtn.disabled = !hasText && !this.interimSegment;
  }

  // --- Controls ---------------------------------------------------------

  private setRecordingEnabled(enabled: boolean) {
    const recBtn = document.getElementById('recordButton') as HTMLButtonElement;
    if (recBtn) recBtn.disabled = !enabled;
  }

  private setFileButtonEnabled(enabled: boolean) {
    const runBtn = document.getElementById('run-file-transcription') as HTMLButtonElement;
    if (runBtn) runBtn.disabled = !enabled;
  }

  // --- Waveform ---------------------------------------------------------

  private visualizeWaveform(newData: Float32Array) {
    const next = new Float32Array(this.WAVEFORM_HISTORY_SIZE);
    const tail = newData.subarray(Math.max(0, newData.length - this.WAVEFORM_HISTORY_SIZE));
    next.set(this.waveformBuffer.subarray(tail.length), 0);
    next.set(tail, this.WAVEFORM_HISTORY_SIZE - tail.length);
    this.waveformBuffer = next;
    this.drawWaveform();
  }

  private drawWaveform() {
    if (!this.canvasCtx || !this.canvasElement) return;
    if (this.canvasElement.width !== this.canvasElement.clientWidth) {
      this.canvasElement.width = this.canvasElement.clientWidth;
      this.canvasElement.height = this.canvasElement.clientHeight;
    }
    const { width, height } = this.canvasElement;
    const ctx = this.canvasCtx;
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    if (width === 0) return;

    ctx.strokeStyle = 'rgba(0, 96, 100, 1.0)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const step = Math.ceil(this.waveformBuffer.length / width);
    for (let i = 0, x = 0; i < this.waveformBuffer.length; i += step, x++) {
      const y = (this.waveformBuffer[i] * height) / 1.5 + height / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/** Decodes an audio (or video) file to mono PCM at `sampleRate`. */
async function decodeToMono(file: File, sampleRate: number): Promise<Float32Array> {
  const arrayBuffer = await file.arrayBuffer();
  const decodeCtx = new AudioContext();
  try {
    const decoded = await decodeCtx.decodeAudioData(arrayBuffer);
    // Down-mix and resample with an OfflineAudioContext.
    const frames = Math.ceil(decoded.duration * sampleRate);
    const offline = new OfflineAudioContext(1, frames, sampleRate);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0).slice();
  } finally {
    await decodeCtx.close();
  }
}

function concat(chunks: Float32Array[]): Float32Array {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/** Linear-interpolation resampler for the live microphone stream. */
function resample(data: Float32Array, fromRate: number, toRate: number): Float32Array {
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(data.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, data.length - 1);
    const t = pos - i0;
    out[i] = data[i0] * (1 - t) + data[i1] * t;
  }
  return out;
}

function formatTime(ms: number): string {
  const totalSec = Math.max(0, Math.floor((ms ?? 0) / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

let activeTask: AudioTranscriberTask | null = null;

export async function setupAudioTranscriber(container: HTMLElement) {
  activeTask = new AudioTranscriberTask({
    container,
    template,
    defaultModelName: DEFAULT_MODEL.id,
    defaultModelUrl: hostedModelDownloadUrl(DEFAULT_MODEL),
    workerFactory: () =>
      new Worker(new URL('../workers/audio-transcriber.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'GPU',
  });

  await activeTask.initialize();
}

export function cleanupAudioTranscriber() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
