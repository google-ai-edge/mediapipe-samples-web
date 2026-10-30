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

import type { VoiceInfo, WordTimestamp } from '@mediapipe/tasks-audio';
import { BaseTask, BaseTaskOptions } from '../components/base-task';
import type { ModelSelectorConfig } from '../components/model-selector';
import { TTS_MODELS, hostedModelDownloadUrl } from '../components/model-registry';

// @ts-ignore
import template from '../templates/text-to-speech.html?raw';

const DEFAULT_MODEL = TTS_MODELS[0];
const PREFERRED_VOICE = 'af_heart';

const ENGLISH_SAMPLES: Record<string, string> = {
  welcome:
    'Welcome to MediaPipe Text to Speech. Every word you hear is generated right here in your browser, ' +
    'without sending any text to a server.',
  story:
    'The lighthouse keeper climbed the spiral stairs one last time. Outside, the storm had finally passed, ' +
    'and the first light of morning spilled across the calm, silver sea.',
  numbers:
    'Your flight to Zurich departs on March 3rd at 7:45 AM from gate B12. The fare is $249.99, ' +
    'and the trip takes about 1.5 hours.',
};

/** Samples keyed by language prefix of a voice locale (e.g. "es" for "es-ES"). */
const LOCALIZED_SAMPLES: Record<string, string> = {
  en: ENGLISH_SAMPLES.welcome,
  es: 'Bienvenido a MediaPipe. Esta voz se genera completamente en tu navegador, sin conexión a ningún servidor.',
  fr: 'Bienvenue dans MediaPipe. Cette voix est générée entièrement dans votre navigateur, sans aucun serveur.',
  de: 'Willkommen bei MediaPipe. Diese Stimme wird vollständig in Ihrem Browser erzeugt, ganz ohne Server.',
  it: 'Benvenuto in MediaPipe. Questa voce viene generata interamente nel tuo browser, senza alcun server.',
  pt: 'Bem-vindo ao MediaPipe. Esta voz é gerada inteiramente no seu navegador, sem nenhum servidor.',
  hi: 'मीडियापाइप में आपका स्वागत है। यह आवाज़ पूरी तरह से आपके ब्राउज़र में बनाई जाती है।',
  ja: 'MediaPipeへようこそ。この音声は、すべてお使いのブラウザの中で生成されています。',
  zh: '欢迎使用 MediaPipe。这个声音完全在您的浏览器中生成，无需任何服务器。',
};

interface ScheduledChunk {
  /** AudioContext time at which the chunk starts playing. */
  startAt: number;
  duration: number;
  source: AudioBufferSourceNode;
}

class TextToSpeechTask extends BaseTask {
  private voices: VoiceInfo[] = [];
  private speakingRate = 1.0;

  // Request state
  private requestId = 0;
  private isSynthesizing = false;
  private firstChunkMs: number | undefined;

  // Playback state
  private audioCtx: AudioContext | undefined;
  private scheduled: ScheduledChunk[] = [];
  /** AudioContext time at which the current utterance starts. */
  private utteranceStartAt = 0;
  /** AudioContext time at which the next chunk can start (gapless playback). */
  private nextStartAt = 0;
  private animationFrame: number | undefined;

  // Output of the last utterance (for WAV export and visualization)
  private outputChunks: Float32Array[] = [];
  private outputSampleRate = 24000;
  private wordTimestamps: WordTimestamp[] = [];
  private spokenText = '';

  // DOM
  private textInput!: HTMLTextAreaElement;
  private voiceSelect!: HTMLSelectElement;
  private speakBtn!: HTMLButtonElement;
  private streamBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private downloadBtn!: HTMLButtonElement;
  private transcriptEl!: HTMLElement;
  private wordSpans: HTMLSpanElement[] = [];
  private canvas!: HTMLCanvasElement;

  constructor(options: BaseTaskOptions) {
    super(options);
  }

  protected override getModelSelectorConfig(): ModelSelectorConfig {
    // The models are 80-700 MB: only download once the user asks for it.
    return {
      autoLoad: false,
      accept: '.litertlm',
      uploadLabel: 'Choose .litertlm File',
      resolveUrl: (value) => this.resolveModelUrl(value),
      preferCached: true,
    };
  }

  protected override onInitializeUI() {
    this.textInput = document.getElementById('tts-text-input') as HTMLTextAreaElement;
    this.voiceSelect = document.getElementById('voice-select') as HTMLSelectElement;
    this.speakBtn = document.getElementById('tts-speak-btn') as HTMLButtonElement;
    this.streamBtn = document.getElementById('tts-stream-btn') as HTMLButtonElement;
    this.stopBtn = document.getElementById('tts-stop-btn') as HTMLButtonElement;
    this.downloadBtn = document.getElementById('tts-download-btn') as HTMLButtonElement;
    this.transcriptEl = document.getElementById('tts-transcript') as HTMLElement;
    this.canvas = document.getElementById('tts-waveform') as HTMLCanvasElement;

    this.textInput.value = ENGLISH_SAMPLES.welcome;

    this.modelSelector?.updateOptions(
      TTS_MODELS.map((m) => ({ label: m.name, value: m.id, isDefault: m.id === DEFAULT_MODEL.id }))
    );

    document.querySelectorAll<HTMLButtonElement>('#tts-samples .sample-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const sample = chip.dataset.sample!;
        this.textInput.value =
          sample === 'voice-language' ? this.sampleForVoice(this.voiceSelect.value) : ENGLISH_SAMPLES[sample];
      });
    });

    this.voiceSelect.addEventListener('change', () => {
      // If the text is still one of our samples, follow the voice's language.
      const isSample =
        Object.values(ENGLISH_SAMPLES).includes(this.textInput.value) ||
        Object.values(LOCALIZED_SAMPLES).includes(this.textInput.value);
      if (isSample) this.textInput.value = this.sampleForVoice(this.voiceSelect.value);
    });

    const rateInput = document.getElementById('speaking-rate') as HTMLInputElement;
    const rateValue = document.getElementById('speaking-rate-value')!;
    rateInput.addEventListener('input', () => {
      rateValue.innerText = parseFloat(rateInput.value).toFixed(1);
    });
    rateInput.addEventListener('change', () => {
      this.speakingRate = parseFloat(rateInput.value);
      // The speaking rate is an engine option: reload if a model is active.
      if (this.modelRequested) this.initializeTask();
    });

    this.speakBtn.addEventListener('click', () => this.synthesize('SYNTHESIZE'));
    this.streamBtn.addEventListener('click', () => this.synthesize('STREAM'));
    this.stopBtn.addEventListener('click', () => this.stop());
    this.downloadBtn.addEventListener('click', () => this.downloadWav());

    this.drawWaveform();
  }

  protected override resolveModelUrl(modelName: string): string | undefined {
    const model = TTS_MODELS.find((m) => m.id === modelName);
    return model ? hostedModelDownloadUrl(model) : undefined;
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return { speakingRate: this.speakingRate };
  }

  protected override async initializeTask(): Promise<void> {
    this.stop();
    this.setActionsEnabled(false);
    this.voiceSelect.disabled = true;
    await super.initializeTask();
  }

  protected override handleInitDone(data?: any) {
    super.handleInitDone(data);
    if (!this.isWorkerReady) return;
    this.populateVoices(data?.voices ?? []);
    this.setActionsEnabled(true);
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const data = event.data;
    switch (data.type) {
      case 'AUDIO_CHUNK':
        if (data.requestId === this.requestId) this.handleAudioChunk(data);
        break;
      case 'SYNTHESIS_DONE':
        if (data.requestId === this.requestId) this.handleSynthesisDone(data);
        break;
      case 'SYNTHESIS_ERROR':
        if (data.requestId === this.requestId) {
          this.isSynthesizing = false;
          this.updateStatus(`Error: ${data.error}`);
          this.setActionsEnabled(true);
        }
        break;
      default:
        super.handleWorkerMessage(event);
    }
  }

  private populateVoices(voices: VoiceInfo[]) {
    this.voices = voices;
    const previous = this.voiceSelect.value;
    this.voiceSelect.innerHTML = '';

    // Group voices by locale for easier browsing.
    const byLocale = new Map<string, VoiceInfo[]>();
    for (const v of voices) {
      const key = v.locale || 'other';
      if (!byLocale.has(key)) byLocale.set(key, []);
      byLocale.get(key)!.push(v);
    }
    for (const [locale, list] of [...byLocale.entries()].sort(([a], [b]) => a.localeCompare(b))) {
      const group = document.createElement('optgroup');
      group.label = locale;
      for (const v of list) {
        const opt = document.createElement('option');
        opt.value = v.voiceId;
        opt.textContent = v.displayName && v.displayName !== v.voiceId ? `${v.displayName} (${v.voiceId})` : v.voiceId;
        group.appendChild(opt);
      }
      this.voiceSelect.appendChild(group);
    }

    if (voices.length === 0) {
      this.voiceSelect.innerHTML = '<option value="">Model default</option>';
    }

    const ids = voices.map((v) => v.voiceId);
    if (ids.includes(previous)) this.voiceSelect.value = previous;
    else if (ids.includes(PREFERRED_VOICE)) this.voiceSelect.value = PREFERRED_VOICE;
    this.voiceSelect.disabled = false;
  }

  private sampleForVoice(voiceId: string): string {
    const locale = this.voices.find((v) => v.voiceId === voiceId)?.locale ?? 'en';
    return LOCALIZED_SAMPLES[locale.split(/[-_]/)[0].toLowerCase()] ?? ENGLISH_SAMPLES.welcome;
  }

  private setActionsEnabled(ready: boolean) {
    const canSpeak = ready && this.isWorkerReady && !this.isSynthesizing;
    this.speakBtn.disabled = !canSpeak;
    this.streamBtn.disabled = !canSpeak;
    this.stopBtn.disabled = !(this.isSynthesizing || this.scheduled.length > 0);
    this.downloadBtn.disabled = this.isSynthesizing || this.outputChunks.length === 0;
  }

  private ensureAudioContext(): AudioContext {
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      this.audioCtx = new AudioContext();
    }
    // Created/resumed from a click handler, so autoplay policies are satisfied.
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    return this.audioCtx;
  }

  private synthesize(mode: 'SYNTHESIZE' | 'STREAM') {
    const text = this.textInput.value.trim();
    if (!text || !this.worker || !this.isWorkerReady) return;

    this.stopPlayback();
    const ctx = this.ensureAudioContext();

    this.requestId++;
    this.isSynthesizing = true;
    this.firstChunkMs = undefined;
    this.outputChunks = [];
    this.wordTimestamps = [];
    this.spokenText = text;
    // Small lead time so the first chunk is never scheduled in the past.
    this.utteranceStartAt = 0;
    this.nextStartAt = ctx.currentTime + 0.05;
    this.renderTranscript(text);
    this.resetMetrics();
    this.drawWaveform();

    this.updateStatus(mode === 'STREAM' ? 'Streaming...' : 'Synthesizing...');
    this.setActionsEnabled(true);

    this.worker.postMessage({
      type: mode,
      requestId: this.requestId,
      text,
      voiceId: this.voiceSelect.value,
    });
  }

  private handleAudioChunk(data: {
    samples: Float32Array;
    sampleRateHz: number;
    wordTimestamps: WordTimestamp[];
    elapsedMs: number;
  }) {
    const { samples, sampleRateHz, wordTimestamps, elapsedMs } = data;
    if (this.firstChunkMs === undefined) {
      this.firstChunkMs = elapsedMs;
      this.setMetric('tts-metric-first', `${Math.round(elapsedMs)} ms`);
      this.updateStatus('Playing...');
    }

    this.outputSampleRate = sampleRateHz;
    this.outputChunks.push(samples);
    // Word timestamps are relative to the start of the utterance.
    this.wordTimestamps.push(...wordTimestamps);

    if (samples.length > 0) {
      const ctx = this.ensureAudioContext();
      const buffer = ctx.createBuffer(1, samples.length, sampleRateHz);
      buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const startAt = Math.max(this.nextStartAt, ctx.currentTime + 0.02);
      if (this.scheduled.length === 0) this.utteranceStartAt = startAt;
      source.start(startAt);
      this.nextStartAt = startAt + buffer.duration;

      const entry = { startAt, duration: buffer.duration, source };
      this.scheduled.push(entry);
      source.onended = () => {
        this.scheduled = this.scheduled.filter((s) => s !== entry);
        if (this.scheduled.length === 0 && !this.isSynthesizing) this.onPlaybackFinished();
      };
    }

    this.drawWaveform();
    this.startAnimation();
  }

  private handleSynthesisDone(data: { totalMs: number; cancelled: boolean }) {
    this.isSynthesizing = false;
    const audioSec = this.totalSamples() / this.outputSampleRate;

    const avg = this.updateInferenceTime(data.totalMs);
    this.setMetric('tts-metric-total', `${Math.round(data.totalMs)} ms`);
    this.setMetric('tts-metric-audio', `${audioSec.toFixed(2)} s`);
    this.setMetric('tts-metric-rtf', audioSec > 0 ? (data.totalMs / 1000 / audioSec).toFixed(3) : '-');

    if (data.cancelled) {
      this.updateStatus('Stopped');
    } else if (this.scheduled.length === 0) {
      this.onPlaybackFinished();
    } else {
      this.updateStatus(`Synthesized in ${Math.round(avg)}ms · playing`);
    }
    this.setActionsEnabled(true);
  }

  private onPlaybackFinished() {
    this.stopAnimation();
    this.highlightWordAt(Number.POSITIVE_INFINITY);
    this.drawWaveform();
    if (!this.isSynthesizing) this.updateStatus('Ready');
    this.setActionsEnabled(true);
  }

  private stopPlayback() {
    for (const s of this.scheduled) {
      s.source.onended = null;
      try {
        s.source.stop();
      } catch (_) {}
    }
    this.scheduled = [];
    this.stopAnimation();
  }

  private stop() {
    if (this.isSynthesizing) {
      this.worker?.postMessage({ type: 'CANCEL' });
    }
    this.stopPlayback();
    if (this.speakBtn) this.setActionsEnabled(true);
  }

  // --- Visualization ----------------------------------------------------

  private startAnimation() {
    if (this.animationFrame !== undefined) return;
    const tick = () => {
      this.animationFrame = requestAnimationFrame(tick);
      if (!this.audioCtx) return;
      const elapsed = this.audioCtx.currentTime - this.utteranceStartAt;
      this.highlightWordAt(elapsed);
      this.drawWaveform(elapsed);
    };
    this.animationFrame = requestAnimationFrame(tick);
  }

  private stopAnimation() {
    if (this.animationFrame !== undefined) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = undefined;
  }

  private renderTranscript(text: string) {
    this.transcriptEl.innerHTML = '';
    this.wordSpans = [];
    // Render the text as-is; spans are attached lazily per word timestamp.
    this.transcriptEl.textContent = text;
  }

  /** Builds per-word spans from the character offsets in the word timestamps. */
  private buildWordSpans() {
    const text = this.spokenText;
    const words = [...this.wordTimestamps].sort((a, b) => a.startChar - b.startChar);
    this.transcriptEl.innerHTML = '';
    this.wordSpans = [];
    let cursor = 0;
    for (const w of words) {
      if (w.startChar < cursor || w.endChar > text.length || w.endChar <= w.startChar) continue;
      if (w.startChar > cursor) this.transcriptEl.append(text.slice(cursor, w.startChar));
      const span = document.createElement('span');
      span.className = 'tts-word';
      span.textContent = text.slice(w.startChar, w.endChar);
      span.dataset.start = String(w.startSec);
      span.dataset.end = String(w.endSec);
      this.transcriptEl.append(span);
      this.wordSpans.push(span);
      cursor = w.endChar;
    }
    if (cursor < text.length) this.transcriptEl.append(text.slice(cursor));
  }

  private highlightWordAt(elapsedSec: number) {
    if (this.wordTimestamps.length === 0) return;
    if (this.wordSpans.length !== this.wordTimestamps.length) this.buildWordSpans();
    for (const span of this.wordSpans) {
      const start = parseFloat(span.dataset.start!);
      const end = parseFloat(span.dataset.end!);
      span.classList.toggle('active', elapsedSec >= start && elapsedSec < end);
      span.classList.toggle('spoken', elapsedSec >= start);
    }
  }

  private totalSamples(): number {
    return this.outputChunks.reduce((n, c) => n + c.length, 0);
  }

  private drawWaveform(playheadSec?: number) {
    const canvas = this.canvas;
    if (!canvas) return;
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);

    const total = this.totalSamples();
    if (total === 0 || width === 0) {
      ctx.strokeStyle = 'rgba(0, 151, 167, 0.4)';
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    // While streaming the final length is unknown; scale to what we have so far.
    const samplesPerPx = Math.max(1, Math.ceil(total / width));
    const playheadX =
      playheadSec !== undefined ? ((playheadSec * this.outputSampleRate) / total) * width : Number.POSITIVE_INFINITY;

    let x = 0;
    let acc = 0;
    let min = 1;
    let max = -1;
    for (const chunk of this.outputChunks) {
      for (let i = 0; i < chunk.length; i++) {
        const v = chunk[i];
        if (v < min) min = v;
        if (v > max) max = v;
        if (++acc === samplesPerPx) {
          ctx.fillStyle = x <= playheadX ? 'rgba(0, 96, 100, 1.0)' : 'rgba(0, 151, 167, 0.45)';
          const y1 = ((1 - max) / 2) * height;
          const y2 = ((1 - min) / 2) * height;
          ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
          x++;
          acc = 0;
          min = 1;
          max = -1;
        }
      }
    }

    if (Number.isFinite(playheadX) && playheadX >= 0 && playheadX <= width) {
      ctx.fillStyle = '#d93025';
      ctx.fillRect(playheadX, 0, 2, height);
    }
  }

  private resetMetrics() {
    for (const id of ['tts-metric-first', 'tts-metric-total', 'tts-metric-audio', 'tts-metric-rtf']) {
      this.setMetric(id, '-');
    }
  }

  private setMetric(id: string, value: string) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  // --- WAV export -------------------------------------------------------

  private downloadWav() {
    const total = this.totalSamples();
    if (total === 0) return;
    const sampleRate = this.outputSampleRate;
    const buffer = new ArrayBuffer(44 + total * 2);
    const view = new DataView(buffer);
    const writeString = (offset: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + total * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, total * 2, true);
    let offset = 44;
    for (const chunk of this.outputChunks) {
      for (let i = 0; i < chunk.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    }
    const url = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `mediapipe-tts-${this.voiceSelect.value || 'speech'}.wav`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  public override cleanup() {
    this.stop();
    if (this.audioCtx) {
      this.audioCtx.close();
      this.audioCtx = undefined;
    }
    super.cleanup();
  }
}

let activeTask: TextToSpeechTask | null = null;

export async function setupTextToSpeech(container: HTMLElement) {
  activeTask = new TextToSpeechTask({
    container,
    template,
    defaultModelName: DEFAULT_MODEL.id,
    defaultModelUrl: hostedModelDownloadUrl(DEFAULT_MODEL),
    workerFactory: () =>
      new Worker(new URL('../workers/text-to-speech.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'GPU',
  });

  await activeTask.initialize();
}

export function cleanupTextToSpeech() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
