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

import template from '../templates/text-to-speech.html?raw';
import { ModelSelector, ModelSelection } from '../components/model-selector';
import { isModelCached } from '../components/model-cache';
import { TEXT_TO_SPEECH_MODELS, HostedModel, hostedModelDownloadUrl } from '../components/model-registry';
import { ViewToggle } from '../components/view-toggle';
import { InferenceTimer } from '../components/inference-timer';
import TextToSpeechWorker from '../workers/text-to-speech.worker.ts?worker';
import type { WordTimestamp, VoiceInfo } from '@mediapipe/tasks-audio';

interface VoiceOptionMeta {
  voiceId: string;
  label: string;
  locale: string;
}

interface PassageSentence {
  index: number;
  paragraphIndex: number;
  text: string;
  locale?: string;
  voiceId?: string;
}

interface PassagePreset {
  id: string;
  locale: string;
  paragraphs: string[];
  polyglotSegments?: Array<{ text: string; locale: string; voiceId: string }>;
}

const VOICE_CATALOG: Record<string, VoiceOptionMeta[]> = {
  kokoro_82m: [
    { voiceId: 'af_heart', label: 'Heart — US Female', locale: 'en-US' },
    { voiceId: 'af_bella', label: 'Bella — US Female', locale: 'en-US' },
    { voiceId: 'am_adam', label: 'Adam — US Male', locale: 'en-US' },
    { voiceId: 'am_michael', label: 'Michael — US Male', locale: 'en-US' },
    { voiceId: 'bf_emma', label: 'Emma — UK Female', locale: 'en-GB' },
    { voiceId: 'bm_george', label: 'George — UK Male', locale: 'en-GB' },
    { voiceId: 'ef_dora', label: 'Dora — Spanish Female', locale: 'es-ES' },
    { voiceId: 'ff_siwis', label: 'Siwis — French Female', locale: 'fr-FR' },
    { voiceId: 'if_sara', label: 'Sara — Italian Female', locale: 'it-IT' },
    { voiceId: 'pf_dora', label: 'Dora — Portuguese Female', locale: 'pt-BR' },
    { voiceId: 'hf_alpha', label: 'Alpha — Hindi Female', locale: 'hi-IN' },
    { voiceId: 'jf_alpha', label: 'Alpha — Japanese Female', locale: 'ja-JP' },
    { voiceId: 'zf_xiaobei', label: 'Xiaobei — Mandarin Female', locale: 'zh-CN' },
  ],
  phonon_31m: [
    { voiceId: 'freya', label: 'Freya — US Female', locale: 'en-US' },
    { voiceId: 'claire', label: 'Claire — US Female', locale: 'en-US' },
    { voiceId: 'lukas', label: 'Lukas — US Male', locale: 'en-US' },
    { voiceId: 'sofia', label: 'Sofia — US Female', locale: 'en-US' },
    { voiceId: 'lucas', label: 'Lucas — US Male', locale: 'en-US' },
  ],
  pocket_tts_100m_en: [
    { voiceId: 'alba', label: 'Alba — English Female', locale: 'en-US' },
    { voiceId: 'jean', label: 'Jean — English Male', locale: 'en-US' },
  ],
  pocket_tts_100m_multi: [
    { voiceId: 'alba', label: 'Alba — English Female', locale: 'en-US' },
    { voiceId: 'jean', label: 'Jean — English Male', locale: 'en-US' },
    { voiceId: 'fantine', label: 'Fantine — French Female', locale: 'fr-FR' },
    { voiceId: 'estelle', label: 'Estelle — Spanish Female', locale: 'es-ES' },
    { voiceId: 'giovanni', label: 'Giovanni — Italian Male', locale: 'it-IT' },
    { voiceId: 'rafael', label: 'Rafael — Portuguese Male', locale: 'pt-BR' },
    { voiceId: 'daan', label: 'Daan — Dutch Male', locale: 'nl-NL' },
  ],
};

const PASSAGE_PRESETS: Record<string, PassagePreset> = {
  'en-US': {
    id: 'en-US',
    locale: 'en-US',
    paragraphs: [
      'MediaPipe Text-to-Speech runs high-fidelity neural speech synthesis directly inside your web browser. By streaming clause chunks as they are generated, playback starts in milliseconds while the remainder of the passage synthesizes in the background.',
      'Click on any sentence in this reader to jump immediately to that position. Notice how each word highlights in real time as it is spoken, respecting natural pauses at commas, dashes, and sentence boundaries.',
    ],
  },
  'en-GB': {
    id: 'en-GB',
    locale: 'en-GB',
    paragraphs: [
      'Good afternoon from London! On-device speech synthesis brings expressive, natural narration to web applications without sending a single byte of text to a remote server.',
      'Whether you are building an accessible document reader, a conversational assistant, or an interactive language tutor, local inference guarantees privacy and consistent responsiveness.',
    ],
  },
  'es-ES': {
    id: 'es-ES',
    locale: 'es-ES',
    paragraphs: [
      '¡Hola y bienvenidos a la demostración de síntesis de voz de MediaPipe! Este modelo convierte texto en audio natural directamente en tu navegador web.',
      'Haz clic en cualquier oración para escuchar su pronunciación al instante, con resaltado palabra por palabra sincronizado en tiempo real.',
    ],
  },
  'fr-FR': {
    id: 'fr-FR',
    locale: 'fr-FR',
    paragraphs: [
      'Bonjour et bienvenue dans la démonstration de synthèse vocale MediaPipe. La génération audio en continu permet une écoute fluide et naturelle directement dans votre navigateur.',
      'Cliquez sur une phrase pour commencer la lecture à tout moment et suivez chaque mot mis en évidence au fil de la narration.',
    ],
  },
  'it-IT': {
    id: 'it-IT',
    locale: 'it-IT',
    paragraphs: [
      'Benvenuti nella dimostrazione di sintesi vocale di MediaPipe! Il modello genera una voce chiara ed espressiva direttamente sul tuo dispositivo.',
      'Tocca qualsiasi frase per avviare la lettura interattiva e osserva le parole illuminarsi in perfetto sincrono con la voce.',
    ],
  },
  'pt-BR': {
    id: 'pt-BR',
    locale: 'pt-BR',
    paragraphs: [
      'Olá! Bem-vindo à demonstração de conversão de texto em fala do MediaPipe executada inteiramente no navegador.',
      'Clique em qualquer frase abaixo para ouvir uma narração fluida com alinhamento preciso de cada palavra falada.',
    ],
  },
  'nl-NL': {
    id: 'nl-NL',
    locale: 'nl-NL',
    paragraphs: [
      'Welkom bij de spraaksynthese-demonstratie van MediaPipe! Dit model zet tekst rechtstreeks in uw webbrowser om in natuurlijke audio.',
      'Klik op een willekeurige zin om de uitspraak direct te horen, terwijl elk woord in realtime wordt gemarkeerd.',
    ],
  },
  'hi-IN': {
    id: 'hi-IN',
    locale: 'hi-IN',
    paragraphs: [
      'नमस्ते! मीडियापाइप टेक्स्ट-टू-स्पीच डेमो में आपका स्वागत है। यह मॉडल सीधे आपके वेब ब्राउज़र में प्राकृतिक और स्पष्ट आवाज़ उत्पन्न करता है।',
      'किसी भी वाक्य पर क्लिक करें और वास्तविक समय में प्रत्येक शब्द को हाइलाइट होते हुए सुनें।',
    ],
  },
  'ja-JP': {
    id: 'ja-JP',
    locale: 'ja-JP',
    paragraphs: [
      'こんにちは！MediaPipeの音声合成デモへようこそ。ブラウザ上で直接ニューラル音声合成モデルを実行し、低遅延で自然な音声をストリーミング再生します。',
      '任意の文をクリックすると、その場所からすぐに読み上げが始まり、発話に合わせて単語がリアルタイムでハイライトされます。',
    ],
  },
  'zh-CN': {
    id: 'zh-CN',
    locale: 'zh-CN',
    paragraphs: [
      '您好，欢迎体验 MediaPipe 网页端语音合成演示！该模型完全在浏览器本地运行，支持低延迟流式音频播放与逐词同步高亮。',
      '点击下方任意句子即可从该处开始朗读，感受自然流畅的多语言语音合成效果。',
    ],
  },
  polyglot: {
    id: 'polyglot',
    locale: 'en-US',
    paragraphs: [],
    polyglotSegments: [
      {
        text: 'Welcome to our multilingual world tour powered by MediaPipe Text-to-Speech!',
        locale: 'en-US',
        voiceId: 'af_heart',
      },
      {
        text: 'Across the Atlantic in London, we can switch seamlessly to a British accent.',
        locale: 'en-GB',
        voiceId: 'bf_emma',
      },
      {
        text: 'En Madrid, el mismo modelo habla español con total naturalidad.',
        locale: 'es-ES',
        voiceId: 'ef_dora',
      },
      {
        text: 'À Paris, la synthèse vocale continue en français sans aucune interruption.',
        locale: 'fr-FR',
        voiceId: 'ff_siwis',
      },
      {
        text: 'A Roma, possiamo ascoltare una pronuncia italiana chiara ed espressiva.',
        locale: 'it-IT',
        voiceId: 'if_sara',
      },
      {
        text: 'और नई दिल्ली में यह मॉडल हिंदी में भी उतनी ही सहजता से बात करता है।',
        locale: 'hi-IN',
        voiceId: 'hf_alpha',
      },
      {
        text: '東京では、自然な日本語のイントネーションで文章を読み上げます。',
        locale: 'ja-JP',
        voiceId: 'jf_alpha',
      },
      {
        text: '最后在北京，我们可以用流利的中文普通话完成这次多语言之旅。',
        locale: 'zh-CN',
        voiceId: 'zf_xiaobei',
      },
    ],
  },
};

let worker: Worker | null = null;
let isWorkerReady = false;
let isLoadingModel = false;
let modelSelector: ModelSelector | null = null;
let inferenceTimer: InferenceTimer | null = null;
let currentModelId = 'phonon_31m';
let userPreferredModelId = 'phonon_31m';
let loadingModelId: string | null = null;
let loadedModelId: string | null = null;
let currentModelUrl = '';
let activePresetId = 'en-US';
let activeInitToken = 0;
let currentViewMode: 'reader' | 'editor' = 'reader';
let pendingActionAfterLoad: (() => void) | null = null;

interface UtteranceTimeline {
  sentenceIdx: number;
  targetWordContainer: HTMLElement;
  wordSpans: HTMLElement[];
  startTime: number;
  endTime: number;
  sampleRateHz: number;
  chunks: Float32Array[];
  combinedSamples: Float32Array | null;
  wordTimestamps: WordTimestamp[];
  ttfaMs: number | null;
  inferenceTimeMs: number | null;
  isDone: boolean;
  hasActivatedUi: boolean;
}

// Audio playback & scheduling state
let audioCtx: AudioContext | null = null;
let activeSources: AudioBufferSourceNode[] = [];
let nextScheduleTime = 0;
let activeSampleRateHz = 24000;
let utteranceTimelines: UtteranceTimeline[] = [];
let activeUtteranceTimeline: UtteranceTimeline | null = null;
let currentAudioObjectUrl: string | null = null;
let rafHighlightId: number | null = null;

// Reader state
let sentences: PassageSentence[] = [];
let currentSentenceIdx = 0;
let isNarrating = false;
let narrateSequenceToken = 0;
let pendingStreamDoneResolver: (() => void) | null = null;

export async function setupTextToSpeech(container: HTMLElement) {
  container.innerHTML = template;

  // Reset state
  if (worker) {
    try {
      worker.onmessage = null;
      worker.onerror = null;
      worker.terminate();
    } catch (_) {}
    worker = null;
  }
  isWorkerReady = false;
  isLoadingModel = false;
  currentModelId = 'phonon_31m';
  userPreferredModelId = 'phonon_31m';
  loadingModelId = null;
  loadedModelId = null;
  currentModelUrl = '';
  activePresetId = 'en-US';
  activeInitToken = 0;
  currentViewMode = 'reader';
  pendingActionAfterLoad = null;
  sentences = [];
  currentSentenceIdx = 0;
  isNarrating = false;
  narrateSequenceToken = 0;
  utteranceTimelines = [];
  activeUtteranceTimeline = null;

  inferenceTimer = new InferenceTimer();
  inferenceTimer.mount();

  // DOM Elements
  const statusMessage = document.getElementById('status-message')!;
  const localeSelect = document.getElementById('tts-locale-select') as HTMLSelectElement;
  const voiceSelect = document.getElementById('tts-voice-select') as HTMLSelectElement;
  const modeSelect = document.getElementById('tts-mode-select') as HTMLSelectElement;
  const speedSelect = document.getElementById('tts-speed-select') as HTMLSelectElement;
  const wordHighlightSelect = document.getElementById('tts-word-highlight-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;

  const viewReader = document.getElementById('view-reader')!;
  const viewEditor = document.getElementById('view-editor')!;
  const presetsBar = document.getElementById('tts-presets-bar')!;

  const readerProgressBadge = document.getElementById('reader-progress-badge')!;
  const readerWordBadge = document.getElementById('reader-word-badge')!;
  const readerPrevBtn = document.getElementById('reader-prev-btn') as HTMLButtonElement;
  const readerNarrateBtn = document.getElementById('reader-narrate-btn') as HTMLButtonElement;
  const readerNarrateLabel = document.getElementById('reader-narrate-label')!;
  const readerNextBtn = document.getElementById('reader-next-btn') as HTMLButtonElement;
  const readerStopBtn = document.getElementById('reader-stop-btn') as HTMLButtonElement;
  const readerBody = document.getElementById('reader-body')!;

  const ttsTextInput = document.getElementById('tts-text-input') as HTMLTextAreaElement;
  const runTtsBtn = document.getElementById('run-tts-btn') as HTMLButtonElement;
  const runTtsBtnLabel = document.getElementById('run-tts-btn-label')!;
  const stopTtsBtn = document.getElementById('stop-tts-btn') as HTMLButtonElement;
  const editorReadalongBox = document.getElementById('editor-readalong-box')!;
  const editorReadalongPreview = document.getElementById('editor-readalong-preview')!;

  const ttsMetricsBadge = document.getElementById('tts-metrics-badge')!;
  const waveformCanvas = document.getElementById('tts-waveform-canvas') as HTMLCanvasElement;
  const audioPlayer = document.getElementById('tts-audio-player') as HTMLAudioElement;

  // Populate initial voice select & passage
  populateVoiceSelect(currentModelId, localeSelect.value);
  loadPresetPassage('en-US', true);
  drawSynthesizedWaveform(waveformCanvas, new Float32Array(0), 0);

  // Initialize ViewToggle
  new ViewToggle(
    'view-mode-toggle',
    [
      { label: 'Interactive Reader', value: 'reader', icon: 'menu_book' },
      { label: 'Custom Text', value: 'editor', icon: 'edit_note' },
    ],
    'reader',
    (mode) => {
      stopAllSynthesis();
      currentViewMode = mode as 'reader' | 'editor';
      if (currentViewMode === 'reader') {
        viewReader.style.display = 'flex';
        viewEditor.style.display = 'none';
      } else {
        viewReader.style.display = 'none';
        viewEditor.style.display = 'flex';
      }
    }
  );

  function updateControlsState() {
    readerNarrateBtn.disabled = isLoadingModel;
    readerPrevBtn.disabled = !isWorkerReady || currentSentenceIdx <= 0;
    readerNextBtn.disabled = !isWorkerReady || currentSentenceIdx >= sentences.length - 1;
    runTtsBtn.disabled = isLoadingModel;

    if (isLoadingModel) {
      readerNarrateLabel.textContent = 'Loading Model...';
      runTtsBtnLabel.textContent = 'Loading Model...';
    } else if (isNarrating) {
      readerNarrateLabel.textContent = 'Restart';
      runTtsBtnLabel.textContent = 'Synthesizing...';
    } else {
      readerNarrateLabel.textContent = isWorkerReady ? 'Narrate All' : 'Load Model & Narrate';
      runTtsBtnLabel.textContent = isWorkerReady ? 'Synthesize & Speak' : 'Load Model & Speak';
    }
  }

  function populateVoiceSelect(modelId: string, preferredLocale?: string, runtimeVoices?: VoiceInfo[]) {
    const knownCatalog = VOICE_CATALOG[modelId];
    const catalog = knownCatalog || VOICE_CATALOG.kokoro_82m;
    let options: VoiceOptionMeta[] = [...catalog];

    if (!knownCatalog && runtimeVoices && runtimeVoices.length > 0) {
      const catalogById = new Map(catalog.map((v) => [v.voiceId, v]));
      options = runtimeVoices.map((rv) => {
        const existing = catalogById.get(rv.voiceId);
        if (existing) return existing;
        return {
          voiceId: rv.voiceId,
          label: `${rv.displayName || rv.voiceId} (${rv.locale || 'en-US'})`,
          locale: rv.locale || 'en-US',
        };
      });
    }

    const previousVoice = voiceSelect.value;
    voiceSelect.innerHTML = '';

    for (const opt of options) {
      const optionEl = document.createElement('option');
      optionEl.value = opt.voiceId;
      optionEl.textContent = opt.label;
      optionEl.dataset.locale = opt.locale;
      voiceSelect.appendChild(optionEl);
    }

    if (preferredLocale) {
      const langPrefix = preferredLocale.split('-')[0].toLowerCase();
      const exactMatch = options.find((o) => o.locale.toLowerCase() === preferredLocale.toLowerCase());
      const prefixMatch = options.find((o) => o.locale.toLowerCase().startsWith(langPrefix));
      if (exactMatch) {
        voiceSelect.value = exactMatch.voiceId;
        return;
      } else if (prefixMatch) {
        voiceSelect.value = prefixMatch.voiceId;
        return;
      }
    }

    if (options.some((o) => o.voiceId === previousVoice)) {
      voiceSelect.value = previousVoice;
    } else if (options.length > 0) {
      voiceSelect.value = options[0].voiceId;
    }
  }

  function splitIntoSentences(text: string): string[] {
    const matches = text.match(/[^.!?।。！？]+[.!?।。！？]+[\])'"`’”]*|[^.!?।。！？]+$/g);
    if (!matches) return [text.trim()].filter(Boolean);
    return matches.map((s) => s.trim()).filter(Boolean);
  }

  function isLocaleSupportedByModel(modelId: string, presetOrLocale: string): boolean {
    if (modelId === 'kokoro_82m') return presetOrLocale !== 'nl-NL';
    const isEnglish = presetOrLocale === 'en-US' || presetOrLocale === 'en-GB' || presetOrLocale.startsWith('en');
    if (modelId === 'phonon_31m' || modelId === 'pocket_tts_100m_en') {
      return isEnglish && presetOrLocale !== 'polyglot';
    }
    if (modelId === 'pocket_tts_100m_multi') {
      const isCjkOrIndicOrPolyglot =
        presetOrLocale === 'hi-IN' ||
        presetOrLocale === 'ja-JP' ||
        presetOrLocale === 'zh-CN' ||
        presetOrLocale === 'polyglot';
      return !isCjkOrIndicOrPolyglot;
    }
    return true;
  }

  function selectCompatibleModelForPreset(presetId: string): string {
    const isCjkOrIndicOrPolyglot =
      presetId === 'hi-IN' || presetId === 'ja-JP' || presetId === 'zh-CN' || presetId === 'polyglot';
    if (isCjkOrIndicOrPolyglot) {
      return 'kokoro_82m';
    }
    if (presetId === 'nl-NL') {
      return 'pocket_tts_100m_multi';
    }
    const isEuropeanNonEn =
      presetId === 'es-ES' || presetId === 'fr-FR' || presetId === 'it-IT' || presetId === 'pt-BR';
    if (isEuropeanNonEn) {
      if (userPreferredModelId === 'kokoro_82m' || userPreferredModelId === 'pocket_tts_100m_multi') {
        return userPreferredModelId;
      }
      return loadedModelId === 'kokoro_82m' && !isLoadingModel ? 'kokoro_82m' : 'pocket_tts_100m_multi';
    }
    return userPreferredModelId || 'phonon_31m';
  }

  function renderWordTokensIntoElement(el: HTMLElement, sentenceText: string) {
    el.innerHTML = '';
    // Match CJK characters individually (with optional trailing punctuation) and non-CJK tokens by whitespace,
    // matching C++ ComputeWordTimestamps tokenization in text_to_speech.cc.
    const tokenRegex =
      /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff][,.;:!?،、。！？，；：）》」』】"'”’)]*|[^\s\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+/g;
    let lastIdx = 0;
    let wordIdx = 0;
    let match: RegExpExecArray | null;
    while ((match = tokenRegex.exec(sentenceText)) !== null) {
      const startChar = match.index;
      const tokenStr = match[0];
      const endChar = startChar + tokenStr.length;

      if (startChar > lastIdx) {
        el.appendChild(document.createTextNode(sentenceText.slice(lastIdx, startChar)));
      }

      const wordSpan = document.createElement('span');
      wordSpan.className = 'tts-word';
      wordSpan.dataset.startChar = String(startChar);
      wordSpan.dataset.endChar = String(endChar);
      wordSpan.dataset.wordIdx = String(wordIdx++);
      wordSpan.textContent = tokenStr;
      el.appendChild(wordSpan);

      lastIdx = endChar;
    }

    if (lastIdx < sentenceText.length) {
      el.appendChild(document.createTextNode(sentenceText.slice(lastIdx)));
    }
  }

  function loadPresetPassage(presetId: string, skipModelAutoSwitch = false) {
    const preset = PASSAGE_PRESETS[presetId] || PASSAGE_PRESETS['en-US'];
    activePresetId = preset.id;

    presetsBar.querySelectorAll('.sample-btn').forEach((btn) => {
      const b = btn as HTMLButtonElement;
      b.classList.toggle('active', b.dataset.preset === preset.id);
    });

    const targetModelId = skipModelAutoSwitch ? currentModelId : selectCompatibleModelForPreset(preset.id);
    currentModelId = targetModelId;

    if (preset.locale) {
      localeSelect.value = preset.locale;
      populateVoiceSelect(currentModelId, preset.locale);
    }

    sentences = [];
    readerBody.innerHTML = '';

    if (preset.polyglotSegments && preset.polyglotSegments.length > 0) {
      const pEl = document.createElement('p');
      pEl.className = 'tts-paragraph';

      preset.polyglotSegments.forEach((seg, idx) => {
        const sentIdx = sentences.length;
        sentences.push({
          index: sentIdx,
          paragraphIndex: 0,
          text: seg.text,
          locale: seg.locale,
          voiceId: seg.voiceId,
        });

        const sentSpan = document.createElement('span');
        sentSpan.className = 'tts-sentence';
        sentSpan.dataset.sentIdx = String(sentIdx);
        renderWordTokensIntoElement(sentSpan, seg.text);
        sentSpan.addEventListener('click', () => {
          triggerReaderOrLoad(sentIdx);
        });
        pEl.appendChild(sentSpan);
        if (idx < preset.polyglotSegments!.length - 1) {
          pEl.appendChild(document.createTextNode(' '));
        }
      });

      readerBody.appendChild(pEl);
      ttsTextInput.value = preset.polyglotSegments.map((s) => s.text).join(' ');
    } else {
      preset.paragraphs.forEach((paraText, pIdx) => {
        const pEl = document.createElement('p');
        pEl.className = 'tts-paragraph';
        const rawSentences = splitIntoSentences(paraText);

        rawSentences.forEach((sText, sIdx) => {
          const sentIdx = sentences.length;
          sentences.push({
            index: sentIdx,
            paragraphIndex: pIdx,
            text: sText,
          });

          const sentSpan = document.createElement('span');
          sentSpan.className = 'tts-sentence';
          sentSpan.dataset.sentIdx = String(sentIdx);
          renderWordTokensIntoElement(sentSpan, sText);
          sentSpan.addEventListener('click', () => {
            triggerReaderOrLoad(sentIdx);
          });
          pEl.appendChild(sentSpan);
          if (sIdx < rawSentences.length - 1) {
            pEl.appendChild(document.createTextNode(' '));
          }
        });

        readerBody.appendChild(pEl);
      });
      ttsTextInput.value = preset.paragraphs.join('\n\n');
    }

    currentSentenceIdx = 0;
    highlightActiveSentence(-1);
    readerProgressBadge.textContent = `Ready · ${sentences.length} sentences`;
    readerWordBadge.style.display = 'none';
    if (!skipModelAutoSwitch) {
      switchToModel(targetModelId);
    } else {
      updateControlsState();
    }
  }

  function highlightActiveSentence(sentIdx: number, isSynthesizing = false) {
    readerBody.querySelectorAll('.tts-sentence').forEach((el) => {
      const span = el as HTMLElement;
      const idx = Number(span.dataset.sentIdx);
      span.classList.toggle('active', idx === sentIdx && !isSynthesizing);
      span.classList.toggle('synthesizing', idx === sentIdx && isSynthesizing);
      if (idx !== sentIdx) {
        span.querySelectorAll('.tts-word').forEach((w) => {
          w.classList.remove('active', 'spoken');
        });
      }
    });
  }

  function clearWordHighlights(rootEl: HTMLElement) {
    rootEl.querySelectorAll('.tts-word').forEach((w) => {
      w.classList.remove('active', 'spoken');
    });
  }

  function ensureAudioContext(sampleRateHz: number): AudioContext {
    if (!audioCtx || audioCtx.state === 'closed') {
      const AudioCtx =
        window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioCtx = new AudioCtx({ sampleRate: sampleRateHz });
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }
    return audioCtx;
  }

  function formatTimelineMetricsBadge(entry: UtteranceTimeline): string {
    const sr = entry.sampleRateHz || 24000;
    const combined = entry.combinedSamples || concatFloat32Arrays(entry.chunks);
    const audioDurSec = combined.length / sr;
    const ttfaStr = entry.ttfaMs !== null ? `TTFA ${entry.ttfaMs.toFixed(0)} ms · ` : '';
    if (!entry.isDone) {
      return `${ttfaStr}Streaming...`;
    }
    const rtf = audioDurSec > 0 && entry.inferenceTimeMs !== null ? entry.inferenceTimeMs / 1000 / audioDurSec : 0;
    return `${ttfaStr}${audioDurSec.toFixed(1)}s audio · RTF ${rtf.toFixed(2)}x (${(sr / 1000).toFixed(1)} kHz)`;
  }

  function scheduleChunkPlayback(samples: Float32Array, sampleRateHz: number, entry?: UtteranceTimeline | null) {
    if (!samples || samples.length === 0) return;
    activeSampleRateHz = sampleRateHz || 24000;
    const ctx = ensureAudioContext(activeSampleRateHz);

    const buffer = ctx.createBuffer(1, samples.length, activeSampleRateHz);
    buffer.getChannelData(0).set(samples);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    if (activeSources.length === 0 && nextScheduleTime <= now) {
      nextScheduleTime = now + 0.02;
    } else if (entry && entry.chunks.length === 1 && utteranceTimelines.length > 1 && nextScheduleTime > now) {
      // Natural breath pause between consecutive sentences scheduled on the timeline.
      nextScheduleTime += 0.1;
    }

    const startAt = Math.max(now + 0.005, nextScheduleTime);
    source.start(startAt);
    nextScheduleTime = startAt + buffer.duration;
    activeSources.push(source);

    if (entry) {
      if (entry.startTime === 0) {
        entry.startTime = startAt;
      }
      entry.endTime = nextScheduleTime;
    }

    source.onended = () => {
      const idx = activeSources.indexOf(source);
      if (idx !== -1) activeSources.splice(idx, 1);
    };
  }

  function ensureTimelinePlaybackLoop() {
    if (rafHighlightId !== null) return;

    const step = () => {
      if (!isNarrating) {
        rafHighlightId = null;
        return;
      }
      if (!audioCtx || utteranceTimelines.length === 0) {
        rafHighlightId = requestAnimationFrame(step);
        return;
      }

      const now = audioCtx.currentTime;
      let activeEntry: UtteranceTimeline | null = null;
      for (let i = 0; i < utteranceTimelines.length; i++) {
        const cand = utteranceTimelines[i];
        if (cand.chunks.length === 0) continue;
        if (now >= cand.startTime - 0.03 && now <= cand.endTime + 0.08) {
          activeEntry = cand;
          break;
        }
        if (now >= cand.startTime - 0.03) {
          activeEntry = cand;
        }
      }
      if (!activeEntry) {
        activeEntry = utteranceTimelines.find((u) => u.chunks.length > 0) || null;
      }

      if (activeEntry) {
        if (!activeEntry.hasActivatedUi && now >= activeEntry.startTime - 0.03) {
          activeEntry.hasActivatedUi = true;
          statusMessage.textContent = 'Playing audio...';
          if (currentViewMode === 'reader' && activeEntry.sentenceIdx >= 0) {
            currentSentenceIdx = activeEntry.sentenceIdx;
            updateControlsState();
            highlightActiveSentence(activeEntry.sentenceIdx, false);
            activeEntry.targetWordContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            readerProgressBadge.textContent = `Sentence ${activeEntry.sentenceIdx + 1} of ${sentences.length}`;
            const sentObj = sentences[activeEntry.sentenceIdx];
            if (sentObj?.locale) {
              localeSelect.value = sentObj.locale;
            }
            if (sentObj?.voiceId && currentModelId === 'kokoro_82m') {
              voiceSelect.value = sentObj.voiceId;
            }
          }
          ttsMetricsBadge.textContent = formatTimelineMetricsBadge(activeEntry);
          const combined = activeEntry.combinedSamples || concatFloat32Arrays(activeEntry.chunks);
          if (activeEntry.isDone && combined.length > 0) {
            updateWavDownloadPlayer(audioPlayer, combined, activeEntry.sampleRateHz);
          }
        }

        const elapsedSec = Math.max(0, now - activeEntry.startTime);
        const enableWordHighlight = wordHighlightSelect.value === 'true';

        if (enableWordHighlight && activeEntry.wordTimestamps.length > 0 && activeEntry.wordSpans.length > 0) {
          let activeWt: WordTimestamp | null = null;
          let activeWtIdx = -1;
          for (let i = 0; i < activeEntry.wordTimestamps.length; i++) {
            const wt = activeEntry.wordTimestamps[i];
            if (elapsedSec >= wt.startSec && elapsedSec <= wt.endSec + 0.04) {
              activeWt = wt;
              activeWtIdx = i;
              break;
            }
          }

          activeEntry.wordSpans.forEach((span, spanIdx) => {
            const startChar = Number(span.dataset.startChar);
            const endChar = Number(span.dataset.endChar);
            const isExactCharMatch =
              activeWt !== null &&
              ((startChar === activeWt.startChar && endChar === activeWt.endChar) ||
                (startChar < activeWt.endChar &&
                  endChar > activeWt.startChar &&
                  Math.abs(startChar - activeWt.startChar) <= 1));
            const isFallbackIdxMatch =
              activeWt !== null &&
              !isExactCharMatch &&
              activeEntry!.wordTimestamps.length === activeEntry!.wordSpans.length &&
              spanIdx === activeWtIdx;

            if (isExactCharMatch || isFallbackIdxMatch) {
              span.classList.add('active');
              span.classList.remove('spoken');
            } else {
              span.classList.remove('active');
              if (activeWt !== null) {
                if (
                  endChar <= activeWt.startChar ||
                  (activeEntry!.wordTimestamps.length === activeEntry!.wordSpans.length && spanIdx < activeWtIdx)
                ) {
                  span.classList.add('spoken');
                } else {
                  span.classList.remove('spoken');
                }
              } else {
                const matchingWt = activeEntry!.wordTimestamps.find(
                  (wt) => wt.startChar === startChar || Math.abs(startChar - wt.startChar) <= 1
                );
                if (matchingWt && elapsedSec > matchingWt.endSec) {
                  span.classList.add('spoken');
                } else {
                  span.classList.remove('spoken');
                }
              }
            }
          });

          if (activeWt) {
            readerWordBadge.style.display = 'inline-flex';
            readerWordBadge.textContent = `"${activeWt.word}" · ${activeWt.startSec.toFixed(2)}s`;
          }
        }

        const combined = activeEntry.combinedSamples || concatFloat32Arrays(activeEntry.chunks);
        const totalDurSec = combined.length / (activeEntry.sampleRateHz || 24000);
        const progressRatio = totalDurSec > 0 ? Math.min(1, elapsedSec / totalDurSec) : 0;
        drawSynthesizedWaveform(waveformCanvas, combined, progressRatio);
      }

      rafHighlightId = requestAnimationFrame(step);
    };

    rafHighlightId = requestAnimationFrame(step);
  }

  function stopAudioPlaybackOnly() {
    if (rafHighlightId !== null) {
      cancelAnimationFrame(rafHighlightId);
      rafHighlightId = null;
    }
    for (const src of activeSources) {
      try {
        src.onended = null;
        src.stop();
        src.disconnect();
      } catch {
        // Ignore already stopped sources
      }
    }
    activeSources = [];
    nextScheduleTime = 0;
    utteranceTimelines = [];
    activeUtteranceTimeline = null;
  }

  function stopAllSynthesis() {
    const wasNarrating = isNarrating;
    narrateSequenceToken++;
    isNarrating = false;
    stopAudioPlaybackOnly();
    if (wasNarrating) {
      worker?.postMessage({ type: 'CANCEL', sessionId: narrateSequenceToken });
      if (isWorkerReady && !isLoadingModel) {
        statusMessage.textContent = 'Ready';
      }
    }

    if (pendingStreamDoneResolver) {
      const resolve = pendingStreamDoneResolver;
      pendingStreamDoneResolver = null;
      resolve();
    }

    readerStopBtn.style.display = 'none';
    stopTtsBtn.style.display = 'none';
    updateControlsState();
  }

  async function synthesizeUtteranceToTimeline(
    text: string,
    targetWordContainer: HTMLElement,
    sentenceIdx: number,
    overrideVoiceId?: string,
    overrideLocale?: string
  ): Promise<void> {
    if (!worker || !isWorkerReady) return;

    clearWordHighlights(targetWordContainer);

    const speed = parseFloat(speedSelect.value) || 1.0;
    const voiceId = overrideVoiceId && currentModelId === 'kokoro_82m' ? overrideVoiceId : voiceSelect.value;
    const locale = overrideLocale || localeSelect.value;
    const mode = modeSelect.value;
    const currentSessionId = narrateSequenceToken;

    const entry: UtteranceTimeline = {
      sentenceIdx,
      targetWordContainer,
      wordSpans: Array.from(targetWordContainer.querySelectorAll('.tts-word')) as HTMLElement[],
      startTime: 0,
      endTime: 0,
      sampleRateHz: activeSampleRateHz || 24000,
      chunks: [],
      combinedSamples: null,
      wordTimestamps: [],
      ttfaMs: null,
      inferenceTimeMs: null,
      isDone: false,
      hasActivatedUi: false,
    };
    utteranceTimelines.push(entry);
    activeUtteranceTimeline = entry;

    if (utteranceTimelines.length === 1) {
      statusMessage.textContent = 'Synthesizing speech...';
      ttsMetricsBadge.textContent = 'Synthesizing audio...';
    }

    ensureTimelinePlaybackLoop();

    await new Promise<void>((resolve) => {
      pendingStreamDoneResolver = resolve;
      worker!.postMessage({
        type: mode === 'unary' ? 'SYNTHESIZE_UNARY' : 'SYNTHESIZE_STREAM',
        text,
        sentenceIdx,
        sessionId: currentSessionId,
        options: {
          voiceId,
          locale,
          speed,
        },
      });
    });
  }

  function triggerReaderOrLoad(startIdx: number) {
    if (!isWorkerReady) {
      pendingActionAfterLoad = () => {
        void startReaderFromSentence(startIdx);
      };
      if (!isLoadingModel) {
        const url = resolveModelUrl(currentModelId);
        if (url) {
          currentModelUrl = url;
          initWorkerModel(url);
        }
      }
      return;
    }
    void startReaderFromSentence(startIdx);
  }

  async function startReaderFromSentence(startIdx: number) {
    if (!isWorkerReady || sentences.length === 0) return;

    stopAllSynthesis();
    const myToken = ++narrateSequenceToken;
    isNarrating = true;

    readerStopBtn.style.display = 'inline-flex';
    currentSentenceIdx = startIdx;
    updateControlsState();

    for (let idx = startIdx; idx < sentences.length; idx++) {
      if (myToken !== narrateSequenceToken || !isNarrating) break;

      // Throttle background prefetching only if > 4.5s of audio is already buffered ahead.
      if (idx > startIdx) {
        while (
          myToken === narrateSequenceToken &&
          isNarrating &&
          audioCtx &&
          nextScheduleTime - audioCtx.currentTime > 4.5
        ) {
          await new Promise<void>((r) => setTimeout(r, 80));
        }
        if (myToken !== narrateSequenceToken || !isNarrating) break;
      }

      const sent = sentences[idx];
      const sentSpan = readerBody.querySelector(`.tts-sentence[data-sent-idx="${idx}"]`) as HTMLElement | null;
      if (!sentSpan) continue;

      if (idx === startIdx) {
        highlightActiveSentence(idx, true);
        sentSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        readerProgressBadge.textContent = `Synthesizing sentence ${idx + 1} of ${sentences.length}...`;
        if (sent.locale) {
          localeSelect.value = sent.locale;
        }
        if (sent.voiceId && currentModelId === 'kokoro_82m') {
          voiceSelect.value = sent.voiceId;
        }
      }

      await synthesizeUtteranceToTimeline(sent.text, sentSpan, idx, sent.voiceId, sent.locale);
    }

    // Wait for scheduled audio playback of all sentences to finish on AudioContext.
    while (
      myToken === narrateSequenceToken &&
      isNarrating &&
      audioCtx &&
      nextScheduleTime > audioCtx.currentTime + 0.04
    ) {
      await new Promise<void>((r) => setTimeout(r, 60));
    }

    if (myToken === narrateSequenceToken) {
      isNarrating = false;
      if (rafHighlightId !== null) {
        cancelAnimationFrame(rafHighlightId);
        rafHighlightId = null;
      }
      readerStopBtn.style.display = 'none';
      readerProgressBadge.textContent = `Finished · ${sentences.length} sentences`;
      if (isWorkerReady && !isLoadingModel) {
        statusMessage.textContent = 'Ready';
      }
      updateControlsState();
    }
  }

  function triggerEditorOrLoad() {
    if (!isWorkerReady) {
      pendingActionAfterLoad = () => {
        void synthesizeCustomEditorText();
      };
      if (!isLoadingModel) {
        const url = resolveModelUrl(currentModelId);
        if (url) {
          currentModelUrl = url;
          initWorkerModel(url);
        }
      }
      return;
    }
    void synthesizeCustomEditorText();
  }

  async function synthesizeCustomEditorText() {
    const text = ttsTextInput.value.trim();
    if (!text || !isWorkerReady) return;

    stopAllSynthesis();
    const myToken = ++narrateSequenceToken;
    isNarrating = true;

    stopTtsBtn.style.display = 'inline-flex';
    editorReadalongBox.style.display = 'block';
    renderWordTokensIntoElement(editorReadalongPreview, text);
    updateControlsState();

    await synthesizeUtteranceToTimeline(text, editorReadalongPreview, -1);

    while (
      myToken === narrateSequenceToken &&
      isNarrating &&
      audioCtx &&
      nextScheduleTime > audioCtx.currentTime + 0.04
    ) {
      await new Promise<void>((r) => setTimeout(r, 60));
    }

    if (myToken === narrateSequenceToken) {
      isNarrating = false;
      if (rafHighlightId !== null) {
        cancelAnimationFrame(rafHighlightId);
        rafHighlightId = null;
      }
      stopTtsBtn.style.display = 'none';
      if (isWorkerReady && !isLoadingModel) {
        statusMessage.textContent = 'Ready';
      }
      updateControlsState();
    }
  }

  function createAndBindWorker(): Worker {
    const w = new TextToSpeechWorker();
    w.onmessage = (e) => {
      const data = e.data;

      switch (data.type) {
        case 'STATUS':
          statusMessage.textContent = data.message;
          if (modelSelector) {
            modelSelector.setStatus(data.message);
          }
          break;

        case 'LOAD_PROGRESS':
          if (modelSelector) {
            modelSelector.showProgress(data.loaded, data.total);
          }
          break;

        case 'DELEGATE_FALLBACK':
          delegateSelect.value = 'CPU';
          inferenceTimer?.resetRollingWindow();
          break;

        case 'INIT_DONE': {
          if (data.initToken !== undefined && data.initToken !== activeInitToken) {
            break;
          }
          isWorkerReady = true;
          loadingModelId = null;
          loadedModelId = currentModelId;
          isLoadingModel = false;
          statusMessage.textContent = 'Ready';
          const spec = TEXT_TO_SPEECH_MODELS.find((m: HostedModel) => m.id === currentModelId);
          const label = spec?.name || currentModelId;
          if (modelSelector) {
            modelSelector.hideProgress();
            modelSelector.setBusy(false);
            modelSelector.setLoaded(currentModelId);
            modelSelector.setStatus(`✓ Active: ${label}`);
            void modelSelector.refreshCacheState();
          }
          if (data.voices && Array.isArray(data.voices)) {
            populateVoiceSelect(currentModelId, localeSelect.value, data.voices);
          }
          updateControlsState();
          if (pendingActionAfterLoad) {
            const action = pendingActionAfterLoad;
            pendingActionAfterLoad = null;
            action();
          }
          break;
        }

        case 'TTS_CHUNK': {
          if (!isNarrating) break;
          if (data.sessionId !== undefined && data.sessionId !== narrateSequenceToken) break;
          const chunkSamples = data.samples as Float32Array;
          const sr = (data.sampleRateHz as number) || 24000;
          const entry = activeUtteranceTimeline;
          if (!entry) break;

          entry.sampleRateHz = sr;
          if (entry.chunks.length === 0 && data.ttfaMs !== null && data.ttfaMs !== undefined) {
            const entryIdx = utteranceTimelines.indexOf(entry);
            const prevEntry = entryIdx > 0 ? utteranceTimelines[entryIdx - 1] : null;
            if (prevEntry && audioCtx && prevEntry.endTime > 0) {
              // Effective wait time after the previous sentence finishes playing on AudioContext.
              const waitAfterPrevMs = Math.max(0, (audioCtx.currentTime - prevEntry.endTime) * 1000);
              entry.ttfaMs = waitAfterPrevMs > 0 ? waitAfterPrevMs : Math.min(data.ttfaMs as number, 120);
            } else {
              entry.ttfaMs = data.ttfaMs as number;
            }
          }

          entry.chunks.push(chunkSamples);
          entry.combinedSamples = concatFloat32Arrays(entry.chunks);
          if (Array.isArray(data.wordTimestamps) && data.wordTimestamps.length > 0) {
            entry.wordTimestamps = [...entry.wordTimestamps, ...data.wordTimestamps];
          }

          scheduleChunkPlayback(chunkSamples, sr, entry);

          if (utteranceTimelines.length === 1 || entry.hasActivatedUi) {
            statusMessage.textContent = 'Playing audio...';
            if (currentViewMode === 'reader' && entry.sentenceIdx >= 0 && !entry.hasActivatedUi) {
              entry.hasActivatedUi = true;
              currentSentenceIdx = entry.sentenceIdx;
              highlightActiveSentence(entry.sentenceIdx, false);
              readerProgressBadge.textContent = `Sentence ${entry.sentenceIdx + 1} of ${sentences.length}`;
            }
            ttsMetricsBadge.textContent = formatTimelineMetricsBadge(entry);
          }
          break;
        }

        case 'TTS_STREAM_DONE': {
          if (data.sessionId !== undefined && data.sessionId !== narrateSequenceToken) break;
          inferenceTimer?.record(data.inferenceTimeMs, delegateSelect.value as 'CPU' | 'GPU');
          const entry = activeUtteranceTimeline;
          if (entry) {
            entry.isDone = true;
            entry.inferenceTimeMs = data.inferenceTimeMs;
            if (entry.ttfaMs === null && data.ttfaMs !== null && data.ttfaMs !== undefined) {
              entry.ttfaMs = data.ttfaMs;
            }
            if (Array.isArray(data.wordTimestamps) && data.wordTimestamps.length > 0) {
              entry.wordTimestamps = data.wordTimestamps;
            }
            entry.combinedSamples = concatFloat32Arrays(entry.chunks);
            if (entry.hasActivatedUi || utteranceTimelines.length === 1) {
              ttsMetricsBadge.textContent = formatTimelineMetricsBadge(entry);
              updateWavDownloadPlayer(audioPlayer, entry.combinedSamples, entry.sampleRateHz);
            }
          }

          if (pendingStreamDoneResolver) {
            const resolve = pendingStreamDoneResolver;
            pendingStreamDoneResolver = null;
            resolve();
          }
          break;
        }

        case 'TTS_RESULT': {
          if (!isNarrating) break;
          if (data.sessionId !== undefined && data.sessionId !== narrateSequenceToken) break;
          const samples = data.samples as Float32Array;
          const sr = (data.sampleRateHz as number) || 24000;
          const entry = activeUtteranceTimeline;

          if (entry) {
            entry.sampleRateHz = sr;
            entry.chunks = [samples];
            entry.combinedSamples = samples;
            entry.wordTimestamps = data.wordTimestamps || [];
            entry.inferenceTimeMs = data.inferenceTimeMs;
            entry.isDone = true;
            scheduleChunkPlayback(samples, sr, entry);
            if (utteranceTimelines.length === 1 || entry.hasActivatedUi) {
              entry.hasActivatedUi = true;
              statusMessage.textContent = 'Playing audio...';
              if (currentViewMode === 'reader' && entry.sentenceIdx >= 0) {
                currentSentenceIdx = entry.sentenceIdx;
                highlightActiveSentence(entry.sentenceIdx, false);
                readerProgressBadge.textContent = `Sentence ${entry.sentenceIdx + 1} of ${sentences.length}`;
              }
              const audioDurSec = samples.length / sr;
              const rtf = audioDurSec > 0 ? data.inferenceTimeMs / 1000 / audioDurSec : 0;
              ttsMetricsBadge.textContent = `Unary · ${audioDurSec.toFixed(1)}s audio · RTF ${rtf.toFixed(2)}x (${(sr / 1000).toFixed(1)} kHz)`;
              updateWavDownloadPlayer(audioPlayer, samples, sr);
            }
          }

          inferenceTimer?.record(data.inferenceTimeMs, delegateSelect.value as 'CPU' | 'GPU');

          if (pendingStreamDoneResolver) {
            const resolve = pendingStreamDoneResolver;
            pendingStreamDoneResolver = null;
            resolve();
          }
          break;
        }

        case 'CANCELLED':
          break;

        case 'ERROR':
          console.error('TextToSpeech worker error:', data.error);
          isLoadingModel = false;
          loadingModelId = null;
          statusMessage.textContent = `Error: ${data.error}`;
          if (modelSelector) {
            modelSelector.hideProgress();
            modelSelector.setBusy(false);
            modelSelector.setStatus(`Error: ${data.error}`);
          }
          updateControlsState();
          if (pendingStreamDoneResolver) {
            const resolve = pendingStreamDoneResolver;
            pendingStreamDoneResolver = null;
            resolve();
          }
          break;
      }
    };
    w.onerror = (err) => {
      console.error('TextToSpeech worker uncaught error:', err);
      isLoadingModel = false;
      loadingModelId = null;
      statusMessage.textContent = `Error: ${err.message || 'Worker error'}`;
      if (modelSelector) {
        modelSelector.hideProgress();
        modelSelector.setBusy(false);
        modelSelector.setStatus(`Error: ${err.message || 'Worker error'}`);
      }
      updateControlsState();
      if (pendingStreamDoneResolver) {
        const resolve = pendingStreamDoneResolver;
        pendingStreamDoneResolver = null;
        resolve();
      }
    };
    return w;
  }

  function recreateWorker() {
    if (worker) {
      try {
        worker.onmessage = null;
        worker.onerror = null;
        worker.terminate();
      } catch (_) {}
    }
    worker = createAndBindWorker();
    loadedModelId = null;
    isWorkerReady = false;
  }

  // Initialize Worker
  worker = createAndBindWorker();

  function initWorkerModel(modelUrl: string, modelFile?: File) {
    const targetId = modelFile ? null : currentModelId;
    if (!worker || isLoadingModel || isNarrating || (loadedModelId !== null && loadedModelId !== targetId)) {
      stopAudioPlaybackOnly();
      isNarrating = false;
      narrateSequenceToken++;
      if (pendingStreamDoneResolver) {
        const resolve = pendingStreamDoneResolver;
        pendingStreamDoneResolver = null;
        resolve();
      }
      readerStopBtn.style.display = 'none';
      stopTtsBtn.style.display = 'none';
      recreateWorker();
    } else {
      stopAllSynthesis();
    }

    const myInitToken = ++activeInitToken;
    isWorkerReady = false;
    isLoadingModel = true;
    loadingModelId = targetId;
    loadedModelId = null;
    updateControlsState();

    const spec = TEXT_TO_SPEECH_MODELS.find((m: HostedModel) => m.id === currentModelId);
    const label = modelFile ? modelFile.name : spec?.name || currentModelId;
    statusMessage.textContent = `Loading ${label}...`;
    if (modelSelector) {
      modelSelector.setLoaded(null);
      modelSelector.setBusy(true);
      modelSelector.setStatus(`Loading ${label}...`);
    }
    const selectEl = document.getElementById('model-selector-container-standard-select') as HTMLSelectElement | null;
    if (selectEl) {
      selectEl.disabled = false;
    }

    const baseUrl = import.meta.env.BASE_URL || '/';

    worker!.postMessage({
      type: 'INIT',
      initToken: myInitToken,
      modelName: modelFile ? modelFile.name : currentModelId,
      modelAssetPath: modelUrl,
      modelFile,
      delegate: delegateSelect.value as 'CPU' | 'GPU',
      defaultVoiceId: voiceSelect.value || '',
      defaultLocale: localeSelect.value || 'en-US',
      speakingRate: parseFloat(speedSelect.value) || 1.0,
      baseUrl,
    });
  }

  function resolveModelUrl(id: string): string | undefined {
    const spec = TEXT_TO_SPEECH_MODELS.find((m: HostedModel) => m.id === id);
    return spec ? hostedModelDownloadUrl(spec) : undefined;
  }

  function switchToModel(targetModelId: string) {
    if (isLoadingModel && loadingModelId === targetModelId) {
      return;
    }
    currentModelId = targetModelId;
    pendingActionAfterLoad = null;

    if (targetModelId === 'kokoro_82m') {
      delegateSelect.value = 'GPU';
    }

    if (modelSelector && modelSelector.getSelectedValue() !== targetModelId) {
      modelSelector.setSelectedValue(targetModelId);
    }
    populateVoiceSelect(currentModelId, localeSelect.value);

    if (loadedModelId === targetModelId && !isLoadingModel && worker) {
      stopAllSynthesis();
      isWorkerReady = true;
      const spec = TEXT_TO_SPEECH_MODELS.find((m: HostedModel) => m.id === targetModelId);
      const label = spec?.name || targetModelId;
      statusMessage.textContent = 'Ready';
      if (modelSelector) {
        modelSelector.hideProgress();
        modelSelector.setBusy(false);
        modelSelector.setLoaded(targetModelId);
        modelSelector.setStatus(`✓ Active: ${label}`);
      }
      updateControlsState();
      return;
    }

    const url = resolveModelUrl(targetModelId);
    if (url) {
      currentModelUrl = url;
      initWorkerModel(url);
    }
  }

  // Initialize ModelSelector
  const modelOptions = TEXT_TO_SPEECH_MODELS.map((m: HostedModel, idx: number) => ({
    label: m.name,
    value: m.id,
    isDefault: idx === 0,
  }));

  modelSelector = new ModelSelector(
    'model-selector-container',
    modelOptions,
    (selection: ModelSelection) => {
      if (selection.type === 'standard') {
        switchToModel(selection.value);
      } else if (selection.type === 'custom' && selection.file) {
        const url = URL.createObjectURL(selection.file);
        currentModelUrl = url;
        initWorkerModel(url, selection.file);
      }
    },
    {
      autoLoad: false,
      accept: '.litertlm,.task,.tflite',
      uploadLabel: 'Choose .litertlm Model',
      resolveUrl: resolveModelUrl,
      preferCached: false,
    }
  );

  const stdSelect = document.getElementById('model-selector-container-standard-select') as HTMLSelectElement | null;
  stdSelect?.addEventListener('change', () => {
    const chosenId = stdSelect.value;
    userPreferredModelId = chosenId;
    if (!isLocaleSupportedByModel(chosenId, activePresetId)) {
      currentModelId = chosenId;
      loadPresetPassage('en-US', true);
    } else {
      populateVoiceSelect(chosenId, localeSelect.value);
    }
    switchToModel(chosenId);
  });

  // Event Listeners
  presetsBar.querySelectorAll('.sample-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const presetId = (e.currentTarget as HTMLButtonElement).dataset.preset || 'en-US';
      stopAllSynthesis();
      loadPresetPassage(presetId);
    });
  });

  localeSelect.addEventListener('change', () => {
    const loc = localeSelect.value;
    if (PASSAGE_PRESETS[loc]) {
      stopAllSynthesis();
      loadPresetPassage(loc);
    } else {
      populateVoiceSelect(currentModelId, loc);
    }
  });

  delegateSelect.addEventListener('change', () => {
    if ((isWorkerReady || isLoadingModel) && currentModelUrl) {
      initWorkerModel(currentModelUrl);
    }
  });

  readerNarrateBtn.addEventListener('click', () => {
    triggerReaderOrLoad(0);
  });

  readerPrevBtn.addEventListener('click', () => {
    if (currentSentenceIdx > 0) {
      triggerReaderOrLoad(currentSentenceIdx - 1);
    }
  });

  readerNextBtn.addEventListener('click', () => {
    if (currentSentenceIdx < sentences.length - 1) {
      triggerReaderOrLoad(currentSentenceIdx + 1);
    }
  });

  readerStopBtn.addEventListener('click', () => {
    stopAllSynthesis();
  });

  runTtsBtn.addEventListener('click', () => {
    triggerEditorOrLoad();
  });

  stopTtsBtn.addEventListener('click', () => {
    stopAllSynthesis();
  });

  // Auto-load if the default TTS model is already cached in IndexedDB
  void (async () => {
    const defaultModel = TEXT_TO_SPEECH_MODELS[0];
    if (defaultModel) {
      const url = hostedModelDownloadUrl(defaultModel);
      if ((await isModelCached(url)) && !isLoadingModel && !isWorkerReady && currentModelId === defaultModel.id) {
        currentModelUrl = url;
        populateVoiceSelect(currentModelId, localeSelect.value);
        if (stdSelect) stdSelect.value = defaultModel.id;
        initWorkerModel(url);
      }
    }
  })();
}

function concatFloat32Arrays(chunks: Float32Array[]): Float32Array {
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const out = new Float32Array(totalLen);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function encodeMonoWavBlob(samples: Float32Array, sampleRateHz: number): Blob {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRateHz, true);
  view.setUint32(28, sampleRateHz * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function updateWavDownloadPlayer(audioPlayer: HTMLAudioElement, samples: Float32Array, sampleRateHz: number) {
  if (!samples || samples.length === 0) return;
  if (currentAudioObjectUrl) {
    URL.revokeObjectURL(currentAudioObjectUrl);
  }
  const wavBlob = encodeMonoWavBlob(samples, sampleRateHz);
  currentAudioObjectUrl = URL.createObjectURL(wavBlob);
  audioPlayer.src = currentAudioObjectUrl;
}

function drawSynthesizedWaveform(canvas: HTMLCanvasElement, samples: Float32Array, progressRatio = 0) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  if (!samples || samples.length === 0) {
    ctx.strokeStyle = 'rgba(0, 127, 139, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    return;
  }

  const numBars = 140;
  const step = Math.max(1, Math.floor(samples.length / numBars));
  const barWidth = Math.max(2, width / numBars - 2);
  const progressBarIdx = Math.floor(progressRatio * numBars);

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
    const barHeight = Math.max(4, Math.min(height - 10, normalized * (height - 12)));
    const x = i * (width / numBars);
    const y = (height - barHeight) / 2;

    ctx.fillStyle = i <= progressBarIdx ? '#007f8b' : 'rgba(0, 127, 139, 0.28)';
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barHeight, 2);
    ctx.fill();
  }
}

export function cleanupTextToSpeech() {
  narrateSequenceToken++;
  isNarrating = false;

  if (rafHighlightId !== null) {
    cancelAnimationFrame(rafHighlightId);
    rafHighlightId = null;
  }

  for (const src of activeSources) {
    try {
      src.onended = null;
      src.stop();
      src.disconnect();
    } catch {
      // Ignore
    }
  }
  activeSources = [];

  if (audioCtx && audioCtx.state !== 'closed') {
    audioCtx.close().catch(() => {});
  }
  audioCtx = null;

  if (currentAudioObjectUrl) {
    URL.revokeObjectURL(currentAudioObjectUrl);
    currentAudioObjectUrl = null;
  }

  if (worker) {
    worker.terminate();
    worker = null;
  }
}
