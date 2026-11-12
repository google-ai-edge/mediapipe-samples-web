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
  welcome: [
    'Welcome to MediaPipe Text to Speech. Every word you hear is generated right here in your browser, without sending any text to a server.',
    'By streaming clause chunks as they are synthesized, playback begins in milliseconds while the remainder of the passage generates in the background.',
    'Click on any sentence below to jump immediately to that line, and watch each word highlight in real time as it is spoken.',
  ].join('\n\n'),
  story: [
    'The lighthouse keeper climbed the spiral stairs one last time. Outside, the storm had finally passed, and the first light of morning spilled across the calm, silver sea.',
    'Far on the horizon, a solitary sailboat caught the golden breeze and turned toward the harbor. He smiled, closed the brass logbook, and brewed a fresh pot of coffee.',
    'After forty winters on the headland, the old lantern would guide ships by solar power, yet its warm beam across the water remained unchanged.',
  ].join('\n\n'),
  numbers: [
    'Your flight to Zurich departs on March 3rd at 7:45 AM from gate B12. The fare is $249.99, and the nonstop trip takes about 1.5 hours.',
    'Upon arrival at Terminal 2, Dr. Smith will meet you on platform 4 for the 9:30 express train—please keep your confirmation code, ZX-4082, ready for inspection!',
    'With 99.4% on-time reliability across 4,520 routes, tickets start at €45 or £39.50.',
  ].join('\n\n'),
  article: [
    'On-device speech synthesis brings natural, low-latency voice interfaces to web applications with complete user privacy.',
    'Because inference runs locally on WebGPU or WebAssembly, audio generation works offline and incurs zero cloud serving cost.',
    'Try clicking around different sentences in this passage, or switch between Kokoro, PocketTTS, and Phonon to compare voice quality and speed.',
  ].join('\n\n'),
};

/** Samples keyed by language prefix of a voice locale (e.g. "es" for "es-ES"). */
const LOCALIZED_SAMPLES: Record<string, string> = {
  en: ENGLISH_SAMPLES.welcome,
  es: [
    'Bienvenido a MediaPipe. Esta voz se genera completamente en tu navegador, sin conexión a ningún servidor.',
    'Haz clic en cualquier oración para escuchar su pronunciación al instante, con resaltado palabra por palabra sincronizado en tiempo real.',
  ].join('\n\n'),
  fr: [
    'Bienvenue dans MediaPipe. Cette voix est générée entièrement dans votre navigateur, sans aucun serveur.',
    'Cliquez sur une phrase pour commencer la lecture à tout moment et suivez chaque mot mis en évidence au fil de la narration.',
  ].join('\n\n'),
  de: [
    'Willkommen bei MediaPipe. Diese Stimme wird vollständig in Ihrem Browser erzeugt, ganz ohne Server.',
    'Klicken Sie auf einen beliebigen Satz, um die Wiedergabe sofort ab dieser Zeile zu starten und jedes gesprochene Wort live zu verfolgen.',
  ].join('\n\n'),
  it: [
    'Benvenuto in MediaPipe. Questa voce viene generata interamente nel tuo browser, senza alcun server.',
    'Tocca qualsiasi frase per avviare la lettura interattiva e osserva le parole illuminarsi in perfetto sincrono con la voce.',
  ].join('\n\n'),
  pt: [
    'Bem-vindo ao MediaPipe. Esta voz é gerada inteiramente no seu navegador, sem nenhum servidor.',
    'Clique em qualquer frase abaixo para ouvir uma narração fluida com alinhamento preciso de cada palavra falada.',
  ].join('\n\n'),
  nl: [
    'Welkom bij MediaPipe. Deze stem wordt volledig in je browser gegenereerd, zonder enige server.',
    'Klik op een willekeurige zin om de uitspraak direct te horen, terwijl elk woord in realtime wordt gemarkeerd.',
  ].join('\n\n'),
  hi: [
    'मीडियापाइप में आपका स्वागत है। यह आवाज़ पूरी तरह से आपके ब्राउज़र में बनाई जाती है।',
    'किसी भी वाक्य पर क्लिक करें और वास्तविक समय में प्रत्येक शब्द को हाइलाइट होते हुए सुनें।',
  ].join('\n\n'),
  ja: [
    'MediaPipeへようこそ。この音声は、すべてお使いのブラウザの中で生成されています。',
    '下の任意の文をクリックすると、その位置からすぐに読み上げが始まり、発話に合わせて単語がハイलाइटされます。',
  ].join('\n\n'),
  zh: [
    '欢迎使用 MediaPipe。这个声音完全在您的浏览器中生成，无需任何服务器。',
    '点击下方任意句子即可从该行开始实时朗读，并同步高亮显示每一个字词。',
  ].join('\n\n'),
};

interface ScheduledChunk {
  /** AudioContext time at which the chunk starts playing. */
  startAt: number;
  duration: number;
  /** Cumulative synthesized audio offset (seconds) at the start of this chunk. */
  audioOffsetSec: number;
  source: AudioBufferSourceNode;
}

interface SentenceSpanInfo {
  index: number;
  paragraphIndex: number;
  startChar: number;
  endChar: number;
  text: string;
  element: HTMLSpanElement;
}

class TextToSpeechTask extends BaseTask {
  private voices: VoiceInfo[] = [];
  private speakingRate = 1.0;

  // Request state
  private requestId = 0;
  private isSynthesizing = false;
  private firstChunkMs: number | undefined;
  private pendingSpeakSentenceIdx: number | undefined;

  // Playback state
  private audioCtx: AudioContext | undefined;
  private scheduled: ScheduledChunk[] = [];
  /** AudioContext time at which the current utterance starts. */
  private utteranceStartAt = 0;
  /** AudioContext time at which the next chunk can start (gapless playback). */
  private nextStartAt = 0;
  /** Cumulative synthesized audio duration (seconds) of chunks that have finished playing. */
  private playedAudioSec = 0;
  private animationFrame: number | undefined;

  // Output of the last utterance (for WAV export and visualization)
  private outputChunks: Float32Array[] = [];
  private outputSampleRate = 24000;
  private wordTimestamps: WordTimestamp[] = [];
  private spokenText = '';
  /** Character offset in `spokenText` where the current synthesis request began. */
  private utteranceCharOffset = 0;
  private startSentenceIdx = 0;
  private activeSentenceIdx = -1;
  private sentences: SentenceSpanInfo[] = [];
  private renderedWordCount = 0;

  // DOM
  private textInput!: HTMLTextAreaElement;
  private voiceSelect!: HTMLSelectElement;
  private speakBtn!: HTMLButtonElement;
  private streamBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private downloadBtn!: HTMLButtonElement;
  private prevSentBtn!: HTMLButtonElement;
  private nextSentBtn!: HTMLButtonElement;
  private sentenceBadgeEl!: HTMLElement;
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
    this.prevSentBtn = document.getElementById('tts-prev-sent-btn') as HTMLButtonElement;
    this.nextSentBtn = document.getElementById('tts-next-sent-btn') as HTMLButtonElement;
    this.sentenceBadgeEl = document.getElementById('tts-sentence-badge') as HTMLElement;
    this.transcriptEl = document.getElementById('tts-transcript') as HTMLElement;
    this.canvas = document.getElementById('tts-waveform') as HTMLCanvasElement;

    this.textInput.value = ENGLISH_SAMPLES.welcome;
    this.renderTranscript(this.textInput.value);

    this.modelSelector?.updateOptions(
      TTS_MODELS.map((m) => ({ label: m.name, value: m.id, isDefault: m.id === DEFAULT_MODEL.id }))
    );

    const sampleChips = document.querySelectorAll<HTMLButtonElement>('#tts-samples .sample-chip');
    sampleChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        sampleChips.forEach((c) => c.classList.toggle('active', c === chip));
        const sample = chip.dataset.sample!;
        this.textInput.value = ENGLISH_SAMPLES[sample];
        this.stop();
        this.renderTranscript(this.textInput.value);
      });
    });

    this.textInput.addEventListener('input', () => {
      if (!this.isSynthesizing && this.scheduled.length === 0) {
        this.renderTranscript(this.textInput.value);
      }
    });

    this.voiceSelect.addEventListener('change', () => {
      // If the text is still one of our samples, follow the voice's language.
      const isSample =
        Object.values(ENGLISH_SAMPLES).includes(this.textInput.value) ||
        Object.values(LOCALIZED_SAMPLES).includes(this.textInput.value);
      if (isSample) {
        this.textInput.value = this.sampleForVoice(this.voiceSelect.value);
        this.renderTranscript(this.textInput.value);
      }
      if (this.isWorkerReady && !this.isSynthesizing && this.voiceSelect.value) {
        this.worker?.postMessage({ type: 'WARMUP_VOICE', voiceId: this.voiceSelect.value });
      }
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

    this.speakBtn.addEventListener('click', () => this.synthesize('SYNTHESIZE', 0));
    this.streamBtn.addEventListener('click', () => this.synthesize('STREAM', 0));
    this.stopBtn.addEventListener('click', () => this.stop());
    this.downloadBtn.addEventListener('click', () => this.downloadWav());

    this.prevSentBtn.addEventListener('click', () => {
      const target = Math.max(0, (this.activeSentenceIdx >= 0 ? this.activeSentenceIdx : 0) - 1);
      this.speakFromSentence(target);
    });
    this.nextSentBtn.addEventListener('click', () => {
      const target = Math.min(
        Math.max(0, this.sentences.length - 1),
        (this.activeSentenceIdx >= 0 ? this.activeSentenceIdx : -1) + 1
      );
      this.speakFromSentence(target);
    });

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
    if (this.pendingSpeakSentenceIdx !== undefined) {
      const idx = this.pendingSpeakSentenceIdx;
      this.pendingSpeakSentenceIdx = undefined;
      this.synthesize('STREAM', idx);
    }
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
    else {
      // Default to an English voice to match the default sample text.
      const english = voices.find((v) => /^en\b/i.test(v.locale ?? ''));
      if (english) this.voiceSelect.value = english.voiceId;
    }
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
    this.updateSentenceNavButtons();
  }

  private updateSentenceNavButtons() {
    if (!this.prevSentBtn || !this.nextSentBtn) return;
    const count = this.sentences.length;
    const cur = this.activeSentenceIdx >= 0 ? this.activeSentenceIdx : 0;
    this.prevSentBtn.disabled = count <= 1 || cur <= 0;
    this.nextSentBtn.disabled = count <= 1 || cur >= count - 1;
    if (this.sentenceBadgeEl) {
      if (count > 0 && this.activeSentenceIdx >= 0) {
        this.sentenceBadgeEl.textContent = `Sentence ${this.activeSentenceIdx + 1} of ${count}`;
      } else if (count > 0) {
        this.sentenceBadgeEl.textContent = `${count} sentence${count === 1 ? '' : 's'}`;
      } else {
        this.sentenceBadgeEl.textContent = '';
      }
    }
  }

  private ensureAudioContext(): AudioContext {
    if (!this.audioCtx || this.audioCtx.state === 'closed') {
      this.audioCtx = new AudioContext();
    }
    // Created/resumed from a click handler, so autoplay policies are satisfied.
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();
    return this.audioCtx;
  }

  private speakFromSentence(sentenceIdx: number) {
    const text = this.textInput.value.trim();
    if (!text) return;
    this.ensureAudioContext();
    if (!this.isWorkerReady) {
      this.pendingSpeakSentenceIdx = sentenceIdx;
      if (!this.modelRequested) {
        this.initializeTask();
      }
      return;
    }

    const clampedSentIdx =
      this.sentences.length > 0 ? Math.max(0, Math.min(sentenceIdx, this.sentences.length - 1)) : 0;
    this.synthesize('STREAM', clampedSentIdx);
  }

  private synthesize(mode: 'SYNTHESIZE' | 'STREAM', startSentenceIdx = 0) {
    const fullText = this.textInput.value.trim();
    if (!fullText || !this.worker || !this.isWorkerReady) return;

    if (this.isSynthesizing) {
      this.worker.postMessage({ type: 'CANCEL' });
    }
    this.stopPlayback();
    this.ensureAudioContext();

    if (this.spokenText !== fullText || this.sentences.length === 0) {
      this.renderTranscript(fullText);
    } else {
      this.resetWordSpansInTranscript();
    }

    const clampedSentIdx =
      this.sentences.length > 0 ? Math.max(0, Math.min(startSentenceIdx, this.sentences.length - 1)) : 0;
    const charOffset = this.sentences[clampedSentIdx]?.startChar ?? 0;
    const textToSpeak = fullText.slice(charOffset);
    if (!textToSpeak.trim()) return;

    this.requestId++;
    this.isSynthesizing = true;
    this.firstChunkMs = undefined;
    this.outputChunks = [];
    this.wordTimestamps = [];
    this.renderedWordCount = 0;
    this.spokenText = fullText;
    this.utteranceCharOffset = charOffset;
    this.startSentenceIdx = clampedSentIdx;
    this.activeSentenceIdx = clampedSentIdx;

    this.utteranceStartAt = 0;
    this.nextStartAt = 0;
    this.playedAudioSec = 0;

    this.highlightSentenceSpan(clampedSentIdx, /*synthesizing=*/ true);
    this.resetMetrics();
    this.drawWaveform();

    this.updateStatus(
      mode === 'STREAM'
        ? clampedSentIdx > 0
          ? `Streaming from sentence ${clampedSentIdx + 1}...`
          : 'Streaming...'
        : 'Synthesizing full text...'
    );
    this.setActionsEnabled(true);

    this.worker.postMessage({
      type: mode,
      requestId: this.requestId,
      text: textToSpeak,
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

    const audioOffsetSec = sampleRateHz > 0 ? this.totalSamples() / sampleRateHz : 0;
    this.outputSampleRate = sampleRateHz;
    this.outputChunks.push(samples);

    const audioSec = this.outputSampleRate > 0 ? this.totalSamples() / this.outputSampleRate : 0;
    this.setMetric('tts-metric-total', `${Math.round(elapsedMs)} ms`);
    this.setMetric('tts-metric-audio', `${audioSec.toFixed(2)} s`);
    this.setMetric('tts-metric-rtf', audioSec > 0 ? (elapsedMs / 1000 / audioSec).toFixed(3) : '-');

    const utteranceText = this.spokenText.slice(this.utteranceCharOffset);
    // Newer @mediapipe/tasks-audio builds convert C++ UTF-8 byte offsets to
    // UTF-16 code-unit indices in the WASM binding; older builds (1.2.2) return
    // raw UTF-8 byte offsets. Detect which one we received so both work.
    for (const w of wordTimestamps) {
      const isUtf16Range =
        w.startChar >= 0 && w.endChar <= utteranceText.length && utteranceText.slice(w.startChar, w.endChar) === w.word;
      const charRange = isUtf16Range
        ? { startChar: w.startChar, endChar: w.endChar }
        : this.utf8ByteRangeToCharRange(utteranceText, w.startChar, w.endChar);
      this.wordTimestamps.push({
        ...w,
        startChar: charRange.startChar + this.utteranceCharOffset,
        endChar: charRange.endChar + this.utteranceCharOffset,
      });
    }

    if (samples.length > 0) {
      const ctx = this.ensureAudioContext();
      const buffer = ctx.createBuffer(1, samples.length, sampleRateHz);
      buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const startAt = Math.max(this.nextStartAt, ctx.currentTime + 0.01);
      if (this.utteranceStartAt === 0) this.utteranceStartAt = startAt;
      source.start(startAt);
      this.nextStartAt = startAt + buffer.duration;

      const entry: ScheduledChunk = { startAt, duration: buffer.duration, audioOffsetSec, source };
      this.scheduled.push(entry);
      source.onended = () => {
        this.playedAudioSec = Math.max(this.playedAudioSec, entry.audioOffsetSec + entry.duration);
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
    this.highlightSentenceSpan(-1, false);
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
    this.pendingSpeakSentenceIdx = undefined;
    if (this.isSynthesizing) {
      this.worker?.postMessage({ type: 'CANCEL' });
      // Ignore chunks of the cancelled request that are already on their way.
      this.requestId++;
      this.isSynthesizing = false;
      this.updateStatus('Stopped');
    }
    this.stopPlayback();
    this.highlightSentenceSpan(-1, false);
    if (this.speakBtn) this.setActionsEnabled(true);
  }

  /**
   * Computes the current playhead in synthesized audio seconds (`0 .. totalAudioSec`),
   * immune to inter-chunk scheduling gaps or underruns.
   */
  private getPlayheadSec(): number {
    if (!this.audioCtx || this.utteranceStartAt === 0) return 0;
    // Compensate ~35ms for browser Web Audio output buffer latency so the
    // visual word highlight lands right on the audible word onset.
    const now = this.audioCtx.currentTime + 0.035;
    if (now < this.utteranceStartAt) return 0;

    for (const chunk of this.scheduled) {
      if (now >= chunk.startAt && now < chunk.startAt + chunk.duration) {
        return chunk.audioOffsetSec + (now - chunk.startAt);
      }
    }

    let playhead = this.playedAudioSec;
    for (const chunk of this.scheduled) {
      if (now >= chunk.startAt + chunk.duration) {
        playhead = Math.max(playhead, chunk.audioOffsetSec + chunk.duration);
      }
    }
    return playhead;
  }

  private utf8ByteRangeToCharRange(
    text: string,
    startByte: number,
    endByte: number
  ): { startChar: number; endChar: number } {
    if (startByte <= 0 && endByte <= 0) return { startChar: 0, endChar: 0 };
    let bytePos = 0;
    let charPos = 0;
    let startChar = -1;
    let endChar = -1;
    while (charPos < text.length) {
      if (startChar < 0 && bytePos >= startByte) {
        startChar = charPos;
      }
      const cp = text.codePointAt(charPos)!;
      const charLen = cp > 0xffff ? 2 : 1;
      const utf8Len = cp <= 0x7f ? 1 : cp <= 0x7ff ? 2 : cp <= 0xffff ? 3 : 4;
      bytePos += utf8Len;
      charPos += charLen;
      if (bytePos >= endByte) {
        endChar = charPos;
        break;
      }
    }
    if (startChar < 0) startChar = Math.min(startByte, text.length);
    if (endChar < 0) endChar = text.length;
    return { startChar, endChar };
  }

  // --- Visualization & Interactive Sentence Reader ----------------------

  private startAnimation() {
    if (this.animationFrame !== undefined) return;
    const tick = () => {
      this.animationFrame = requestAnimationFrame(tick);
      if (!this.audioCtx) return;
      const elapsed = this.getPlayheadSec();
      this.highlightWordAt(elapsed);
      this.drawWaveform(elapsed);
    };
    this.animationFrame = requestAnimationFrame(tick);
  }

  private stopAnimation() {
    if (this.animationFrame !== undefined) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = undefined;
  }

  /** Splits `text` into paragraphs and clickable sentences with exact character offsets. */
  private parseSentences(text: string): Array<Omit<SentenceSpanInfo, 'element'>> {
    const out: Array<Omit<SentenceSpanInfo, 'element'>> = [];
    const n = text.length;
    let pIdx = 0;
    let i = 0;

    const isSentenceTerminator = (ch: string) =>
      ch === '.' || ch === '!' || ch === '?' || ch === '。' || ch === '！' || ch === '？' || ch === '।' || ch === '॥';
    const isClosingQuoteOrBracket = (ch: string) =>
      ch === '"' || ch === "'" || ch === '”' || ch === '’' || ch === ')' || ch === ']' || ch === '」' || ch === '』';
    const nonSplittingAbbrs = new Set([
      'dr',
      'mr',
      'mrs',
      'ms',
      'st',
      'jr',
      'sr',
      'prof',
      'vs',
      'etc',
      'approx',
      'dept',
      'fig',
      'inc',
      'ltd',
      'no',
    ]);
    const isNonSplittingPeriod = (periodIdx: number, pStart: number, pEnd: number): boolean => {
      if (periodIdx + 1 < pEnd && text[periodIdx + 1] === '.') return true;
      if (
        periodIdx > pStart &&
        periodIdx + 1 < pEnd &&
        /\d/.test(text[periodIdx - 1]) &&
        /\d/.test(text[periodIdx + 1])
      ) {
        return true;
      }
      let afterIdx = periodIdx + 1;
      while (afterIdx < pEnd && isClosingQuoteOrBracket(text[afterIdx])) afterIdx++;
      if (afterIdx >= pEnd) return false;
      if (!/\s/.test(text[afterIdx])) return true;
      let nextWordIdx = afterIdx;
      while (nextWordIdx < pEnd && /\s/.test(text[nextWordIdx])) nextWordIdx++;
      if (nextWordIdx < pEnd && /^[a-z,;:]/.test(text[nextWordIdx])) return true;
      let tokStart = periodIdx;
      while (tokStart > pStart && /[a-zA-Z0-9]/.test(text[tokStart - 1])) tokStart--;
      const token = text.slice(tokStart, periodIdx);
      if (!token) return false;
      if (tokStart === pStart && /^\d+$/.test(token)) return true;
      if (token.length === 1 && /[a-zA-Z]/.test(token)) return true;
      if (tokStart > pStart && text[tokStart - 1] === '.') return true;
      if (nonSplittingAbbrs.has(token.toLowerCase())) return true;
      return false;
    };

    while (i < n) {
      while (i < n && /\s/.test(text[i])) i++;
      if (i >= n) break;

      // Find the end of the current paragraph (delimited by one or more newlines).
      let pEnd = i;
      while (pEnd < n && text[pEnd] !== '\n' && text[pEnd] !== '\r') pEnd++;

      let sStart = i;
      let j = i;
      while (j < pEnd) {
        if (isSentenceTerminator(text[j])) {
          if (text[j] === '.' && isNonSplittingPeriod(j, i, pEnd)) {
            j++;
            continue;
          }
          let k = j + 1;
          while (k < pEnd && isClosingQuoteOrBracket(text[k])) k++;
          // Split if at end of paragraph, CJK/Indic punctuation, or followed by whitespace.
          const isCjkOrIndic = text.charCodeAt(j) > 0x7f;
          if (k === pEnd || isCjkOrIndic || /\s/.test(text[k])) {
            const sentText = text.slice(sStart, k);
            if (sentText.trim().length > 0) {
              out.push({
                index: out.length,
                paragraphIndex: pIdx,
                startChar: sStart,
                endChar: k,
                text: sentText,
              });
            }
            while (k < pEnd && /\s/.test(text[k])) k++;
            sStart = k;
            j = k;
            continue;
          }
        }
        j++;
      }

      if (sStart < pEnd) {
        let sEnd = pEnd;
        while (sEnd > sStart && /\s/.test(text[sEnd - 1])) sEnd--;
        if (sEnd > sStart) {
          out.push({
            index: out.length,
            paragraphIndex: pIdx,
            startChar: sStart,
            endChar: sEnd,
            text: text.slice(sStart, sEnd),
          });
        }
      }

      pIdx++;
      i = pEnd;
      while (i < n && (text[i] === '\n' || text[i] === '\r')) i++;
    }

    return out;
  }

  private renderTranscript(text: string) {
    const trimmed = text.trim();
    this.spokenText = trimmed;
    this.transcriptEl.innerHTML = '';
    this.wordSpans = [];
    this.renderedWordCount = 0;
    this.sentences = [];
    this.activeSentenceIdx = -1;

    if (!trimmed) {
      this.updateSentenceNavButtons();
      return;
    }

    const parsed = this.parseSentences(trimmed);
    const byParagraph = new Map<number, Array<Omit<SentenceSpanInfo, 'element'>>>();
    for (const s of parsed) {
      if (!byParagraph.has(s.paragraphIndex)) byParagraph.set(s.paragraphIndex, []);
      byParagraph.get(s.paragraphIndex)!.push(s);
    }

    for (const [, list] of byParagraph) {
      const pEl = document.createElement('p');
      pEl.className = 'tts-paragraph';
      list.forEach((s, idxWithinPara) => {
        const span = document.createElement('span');
        span.className = 'tts-sentence';
        span.dataset.sentIdx = String(s.index);
        span.title = 'Click to stream from this sentence';
        span.textContent = s.text;
        span.addEventListener('click', () => {
          this.speakFromSentence(s.index);
        });
        pEl.appendChild(span);
        if (idxWithinPara < list.length - 1) {
          pEl.appendChild(document.createTextNode(' '));
        }
        this.sentences.push({ ...s, element: span });
      });
      this.transcriptEl.appendChild(pEl);
    }

    this.updateSentenceNavButtons();
  }

  private resetWordSpansInTranscript() {
    this.wordSpans = [];
    this.renderedWordCount = 0;
    for (const s of this.sentences) {
      s.element.classList.remove('active', 'synthesizing', 'spoken');
      s.element.textContent = s.text;
    }
  }

  private highlightSentenceSpan(sentIdx: number, synthesizing: boolean) {
    this.activeSentenceIdx = sentIdx;
    for (const s of this.sentences) {
      const isCurrent = s.index === sentIdx;
      s.element.classList.toggle('active', isCurrent && !synthesizing);
      s.element.classList.toggle('synthesizing', isCurrent && synthesizing);
      s.element.classList.toggle('spoken', sentIdx >= 0 && s.index >= this.startSentenceIdx && s.index < sentIdx);
    }
    this.updateSentenceNavButtons();
  }

  /** Builds per-word spans inside each clickable `.tts-sentence` element. */
  private buildWordSpans() {
    const text = this.spokenText;
    const words = [...this.wordTimestamps].sort((a, b) => a.startChar - b.startChar);
    this.wordSpans = [];
    this.renderedWordCount = this.wordTimestamps.length;

    for (const s of this.sentences) {
      const sentWords = words.filter((w) => w.startChar < s.endChar && w.endChar > s.startChar);
      s.element.innerHTML = '';
      if (sentWords.length === 0) {
        s.element.textContent = s.text;
        continue;
      }

      let cursor = s.startChar;
      for (const w of sentWords) {
        const wStart = Math.max(s.startChar, w.startChar);
        const wEnd = Math.min(s.endChar, w.endChar);
        if (wStart < cursor || wEnd > text.length || wEnd <= wStart) continue;
        if (wStart > cursor) {
          s.element.append(text.slice(cursor, wStart));
        }
        const span = document.createElement('span');
        span.className = 'tts-word';
        span.textContent = text.slice(wStart, wEnd);
        span.dataset.start = String(w.startSec);
        span.dataset.end = String(w.endSec);
        span.dataset.sentIdx = String(s.index);
        s.element.append(span);
        this.wordSpans.push(span);
        cursor = wEnd;
      }
      if (cursor < s.endChar) {
        s.element.append(text.slice(cursor, s.endChar));
      }
    }
  }

  private highlightWordAt(elapsedSec: number) {
    if (this.wordTimestamps.length === 0) return;
    if (this.renderedWordCount !== this.wordTimestamps.length) this.buildWordSpans();

    let currentSentIdx = -1;
    let latestSpokenSentIdx = -1;
    let activeAssigned = false;
    for (let i = 0; i < this.wordSpans.length; i++) {
      const span = this.wordSpans[i];
      const start = parseFloat(span.dataset.start!);
      const end = parseFloat(span.dataset.end!);
      const nextSpan = i + 1 < this.wordSpans.length ? this.wordSpans[i + 1] : undefined;
      const nextStart = nextSpan ? parseFloat(nextSpan.dataset.start!) : end + 0.15;
      const onsetSec = start - 0.015;
      const nextOnsetSec = nextStart - 0.015;
      const holdUntil = Math.min(nextOnsetSec, Math.max(end, start + 0.08) + 0.22);
      const isFirstWordLeadIn = i === 0 && Number.isFinite(elapsedSec) && elapsedSec >= 0 && elapsedSec < onsetSec;
      const isActive =
        !activeAssigned &&
        (isFirstWordLeadIn ||
          (elapsedSec >= onsetSec && elapsedSec < Math.max(end, holdUntil) && elapsedSec < nextOnsetSec));
      if (isActive) activeAssigned = true;
      const isSpoken = isFirstWordLeadIn || elapsedSec >= onsetSec;
      span.classList.toggle('active', isActive);
      span.classList.toggle('spoken', isSpoken);
      const sentIdx = Number(span.dataset.sentIdx);
      if (isActive) currentSentIdx = sentIdx;
      if (isSpoken) latestSpokenSentIdx = sentIdx;
    }

    if (Number.isFinite(elapsedSec)) {
      const targetSent = currentSentIdx >= 0 ? currentSentIdx : latestSpokenSentIdx;
      if (targetSent >= 0) {
        this.highlightSentenceSpan(targetSent, false);
      }
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
