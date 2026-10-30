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

import {
  FilesetResolver,
  TextToSpeech,
  type TextToSpeechStream,
  type VoiceInfo,
  type WordTimestamp,
} from '@mediapipe/tasks-audio';
import audioWasmLoaderSource from '../../public/wasm/audio_wasm_module_internal.js?raw';
import { openModelStream } from '../components/model-cache';
import { BaseWorker } from './base-worker';

let cachedModuleFactory: ((config?: Record<string, any>) => Promise<any>) | null = null;
let cachedWasmModulePromise: Promise<WebAssembly.Module> | null = null;

async function ensureAudioModuleFactory(fallbackUrl?: string) {
  if (!cachedModuleFactory) {
    const blob = new Blob([audioWasmLoaderSource], { type: 'text/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    try {
      const mod = await import(/* @vite-ignore */ blobUrl);
      cachedModuleFactory = mod.default || (self as any).ModuleFactory;
    } catch {
      if (fallbackUrl) {
        const mod = await import(/* @vite-ignore */ fallbackUrl);
        cachedModuleFactory = mod.default || (self as any).ModuleFactory;
      }
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  }
  (self as any).ModuleFactory = cachedModuleFactory;
  return cachedModuleFactory;
}

const WASM_CACHE_VERSION = 'v3';

function getCompiledWasmModule(wasmUrl: string): Promise<WebAssembly.Module> {
  if (!cachedWasmModulePromise) {
    cachedWasmModulePromise = (async () => {
      const versionedUrl = wasmUrl.includes('?')
        ? `${wasmUrl}&v=${WASM_CACHE_VERSION}`
        : `${wasmUrl}?v=${WASM_CACHE_VERSION}`;
      const { stream } = await openModelStream(versionedUrl);
      const bytes = await new Response(stream).arrayBuffer();
      return WebAssembly.compile(bytes);
    })().catch((err) => {
      cachedWasmModulePromise = null;
      throw err;
    });
  }
  return cachedWasmModulePromise;
}

(self as any).import = async (url: string) => {
  const factory = await ensureAudioModuleFactory(url);
  return { default: factory };
};

function shouldSuppressCppLog(args: unknown[]): boolean {
  const first = typeof args[0] === 'string' ? args[0].trimStart() : '';
  return (
    first.startsWith('INFO:') ||
    first.startsWith('VERBOSE:') ||
    first.startsWith('WARNING:') ||
    first.startsWith('I0000 ') ||
    first.startsWith('W0000 ')
  );
}

const origConsoleError = console.error.bind(console);
const origConsoleWarn = console.warn.bind(console);
const origConsoleLog = console.log.bind(console);
console.error = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleError(...args);
};
console.warn = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleWarn(...args);
};
console.log = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleLog(...args);
};

let cachedWebGpuDevicePromise: Promise<any | null> | null = null;

async function getHardwareWebGpuDevice(): Promise<any | null> {
  if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
    return null;
  }
  if (!cachedWebGpuDevicePromise) {
    cachedWebGpuDevicePromise = (async () => {
      try {
        const gpu = (navigator as any).gpu;
        const adapter =
          (await gpu.requestAdapter({ powerPreference: 'high-performance' })) || (await gpu.requestAdapter());
        if (!adapter) return null;
        const adapterInfo = adapter.info;
        const isFallback = Boolean(adapter.isFallbackAdapter);
        const isSoftwareAdapter =
          isFallback ||
          /swiftshader|llvmpipe|software|lavapipe/i.test(
            `${adapterInfo?.vendor ?? ''} ${adapterInfo?.architecture ?? ''} ${adapterInfo?.description ?? ''}`
          );
        if (isSoftwareAdapter) return null;

        const requiredFeatures: string[] = [];
        for (const feat of ['shader-f16', 'subgroups']) {
          if (adapter.features?.has?.(feat)) {
            requiredFeatures.push(feat);
          }
        }
        const requiredLimits: Record<string, number> = {};
        for (const limitKey of [
          'maxBufferSize',
          'maxStorageBufferBindingSize',
          'maxStorageBuffersPerShaderStage',
          'maxComputeWorkgroupStorageSize',
          'maxTextureDimension2D',
        ]) {
          const val = adapter.limits?.[limitKey];
          if (typeof val === 'number') {
            requiredLimits[limitKey] = val;
          }
        }

        let device: any = null;
        try {
          device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits,
          });
        } catch {
          try {
            const fallbackFeatures = adapter.features?.has?.('shader-f16') ? ['shader-f16'] : [];
            device = await adapter.requestDevice({
              requiredFeatures: fallbackFeatures,
            });
          } catch {
            device = await adapter.requestDevice();
          }
        }
        if (!device) return null;

        if (!device.adapterInfo && adapterInfo) {
          try {
            Object.defineProperty(device, 'adapterInfo', {
              value: adapterInfo,
              writable: true,
              configurable: true,
            });
          } catch {
            try {
              device.adapterInfo = adapterInfo;
            } catch {}
          }
        }
        return device;
      } catch {
        return null;
      }
    })();
  }
  return cachedWebGpuDevicePromise;
}

interface KokoroClauseSlice {
  text: string;
  charOffset: number;
}

/**
 * Builds a lookup table mapping UTF-8 byte offsets (used by C++ `std::string` /
 * `absl::string_view` in `ComputeWordTimestamps`) to JavaScript UTF-16 code unit
 * indices (`0 .. str.length`).
 */
function buildUtf8ByteToUtf16Map(str: string): number[] {
  const map: number[] = [];
  for (let u16 = 0; u16 < str.length; u16++) {
    const cp = str.codePointAt(u16)!;
    const utf8Len = cp <= 0x7f ? 1 : cp <= 0x7ff ? 2 : cp <= 0xffff ? 3 : 4;
    const u16Step = cp > 0xffff ? 2 : 1;
    for (let b = 0; b < utf8Len; b++) {
      map.push(u16);
    }
    if (u16Step === 2) {
      u16++;
    }
  }
  map.push(str.length);
  return map;
}

/**
 * Converts C++ `WordTimestamp` offsets (UTF-8 byte offsets within `clauseText`)
 * to exact JavaScript UTF-16 code unit indices in the full utterance string.
 */
function normalizeWordTimestampsToUtf16(
  clauseText: string,
  clauseCharOffsetUtf16: number,
  timeOffsetSec: number,
  rawWords: WordTimestamp[],
  byteToUtf16: number[],
  searchCursorRef: { pos: number }
): WordTimestamp[] {
  if (!rawWords || rawWords.length === 0) return [];
  const maxByte = byteToUtf16.length - 1;

  return rawWords.map((wt) => {
    const rawStart = Math.max(0, Math.min(maxByte, wt.startChar ?? 0));
    const rawEnd = Math.max(rawStart, Math.min(maxByte, wt.endChar ?? rawStart));
    let startUtf16 = byteToUtf16[rawStart] ?? 0;
    let endUtf16 = byteToUtf16[rawEnd] ?? startUtf16 + (wt.word?.length || 0);

    if (wt.word && clauseText.slice(startUtf16, endUtf16) !== wt.word) {
      const foundIdx = clauseText.indexOf(wt.word, searchCursorRef.pos);
      if (foundIdx !== -1) {
        startUtf16 = foundIdx;
        endUtf16 = foundIdx + wt.word.length;
      }
    }
    searchCursorRef.pos = Math.max(searchCursorRef.pos, endUtf16);

    return {
      word: wt.word,
      startChar: startUtf16 + clauseCharOffsetUtf16,
      endChar: endUtf16 + clauseCharOffsetUtf16,
      startSec: wt.startSec + timeOffsetSec,
      endSec: wt.endSec + timeOffsetSec,
    };
  });
}

function estimateKokoroPhonemeTokens(str: string): number {
  let tokens = 0;
  for (let i = 0; i < str.length; i++) {
    const code = str.charCodeAt(i);
    if (
      (code >= 0x3040 && code <= 0x30ff) ||
      (code >= 0x3400 && code <= 0x9fff) ||
      (code >= 0xf900 && code <= 0xfaff)
    ) {
      tokens += 2.3;
    } else if (code >= 0x0900 && code <= 0x097f) {
      tokens += 1.1;
    } else if (code <= 0x20) {
      tokens += 0.45;
    } else {
      tokens += 0.92;
    }
  }
  return tokens;
}

/**
 * Splits a sentence into single-slice clauses (`<= 74` estimated phoneme tokens)
 * for Kokoro so `KokoroAcousticStage::ScheduleInternal()` never blocks on multiple
 * 85-token acoustic slices before running `KokoroVocoderStage` on the opening clause,
 * while avoiding extra slices when clauses can fit in one 85-token pass.
 */
function splitTextIntoKokoroClauses(rawText: string): KokoroClauseSlice[] {
  if (!rawText || rawText.trim().length === 0) {
    return [];
  }
  const MAX_SLICE_TOKENS = 74;
  if (estimateKokoroPhonemeTokens(rawText) <= MAX_SLICE_TOKENS) {
    return [{ text: rawText, charOffset: 0 }];
  }

  // Step 1: Split on natural clause punctuation (comma, semicolon, colon, em-dash, CJK comma).
  const initialClauses: KokoroClauseSlice[] = [];
  const punctRegex = /[^,;:\u2014\u3001\uff0c]+[,;:\u2014\u3001\uff0c]+|[^,;:\u2014\u3001\uff0c]+$/g;
  let match: RegExpExecArray | null;
  while ((match = punctRegex.exec(rawText)) !== null) {
    const seg = match[0];
    if (seg.trim().length > 0) {
      initialClauses.push({ text: seg, charOffset: match.index });
    }
  }
  if (initialClauses.length === 0) {
    initialClauses.push({ text: rawText, charOffset: 0 });
  }

  // Step 2: Merge adjacent fragments as long as the combined clause fits within a single slice,
  // keeping the opening clause separated if it is already >= 32 tokens so TTFA stays fast.
  const mergedClauses: KokoroClauseSlice[] = [];
  for (const clause of initialClauses) {
    if (mergedClauses.length > 0) {
      const prev = mergedClauses[mergedClauses.length - 1];
      const prevTokens = estimateKokoroPhonemeTokens(prev.text);
      const combinedEnd = clause.charOffset + clause.text.length;
      const combinedText = rawText.slice(prev.charOffset, combinedEnd);
      const combinedTokens = estimateKokoroPhonemeTokens(combinedText);
      const isOpeningAlreadySubstantial = mergedClauses.length === 1 && prevTokens >= 32;
      if (combinedTokens <= MAX_SLICE_TOKENS && !isOpeningAlreadySubstantial) {
        prev.text = combinedText;
        continue;
      }
    }
    mergedClauses.push({ ...clause });
  }

  // Step 3: Split any clause exceeding MAX_SLICE_TOKENS at a natural phrase boundary so every
  // clause fits inside a single 85-phoneme-token slice in KokoroAcousticStage.
  const finalClauses: KokoroClauseSlice[] = [];
  const splitWordsSet = new Set([
    'while',
    'where',
    'when',
    'which',
    'that',
    'because',
    'although',
    'though',
    'before',
    'after',
    'without',
    'through',
    'between',
    'directly',
    'inside',
    'across',
    'during',
    'and',
    'or',
    'but',
    'as',
    'for',
    'with',
    'from',
    'into',
  ]);

  const splitLongClause = (clause: KokoroClauseSlice) => {
    if (estimateKokoroPhonemeTokens(clause.text) <= MAX_SLICE_TOKENS) {
      finalClauses.push(clause);
      return;
    }

    const words: Array<{ word: string; start: number; end: number }> = [];
    const wordRe = /\S+/g;
    let wm: RegExpExecArray | null;
    while ((wm = wordRe.exec(clause.text)) !== null) {
      words.push({ word: wm[0], start: wm.index, end: wm.index + wm[0].length });
    }

    if (words.length <= 4) {
      const midChar = Math.floor(clause.text.length / 2);
      const leftText = clause.text.slice(0, midChar);
      const rightText = clause.text.slice(midChar);
      if (leftText.trim().length > 0 && rightText.trim().length > 0) {
        finalClauses.push({ text: leftText, charOffset: clause.charOffset });
        splitLongClause({ text: rightText, charOffset: clause.charOffset + midChar });
      } else {
        finalClauses.push(clause);
      }
      return;
    }

    const targetIdx = Math.max(2, Math.min(words.length - 2, Math.floor(words.length * 0.5)));
    let bestSplitWordIdx = targetIdx;
    let bestDistance = 999;

    for (let i = 2; i <= words.length - 2; i++) {
      const leftCandidate = clause.text.slice(0, words[i].start);
      if (estimateKokoroPhonemeTokens(leftCandidate) > MAX_SLICE_TOKENS) {
        break;
      }
      const clean = words[i].word.toLowerCase().replace(/[^a-z]/g, '');
      if (splitWordsSet.has(clean)) {
        const dist = Math.abs(i - targetIdx);
        if (dist < bestDistance) {
          bestDistance = dist;
          bestSplitWordIdx = i;
        }
      }
    }

    while (
      bestSplitWordIdx > 2 &&
      estimateKokoroPhonemeTokens(clause.text.slice(0, words[bestSplitWordIdx].start)) > MAX_SLICE_TOKENS
    ) {
      bestSplitWordIdx--;
    }

    const splitPosInClause = words[bestSplitWordIdx].start;
    const leftText = clause.text.slice(0, splitPosInClause);
    const rightText = clause.text.slice(splitPosInClause);

    if (leftText.trim().length > 0 && rightText.trim().length > 0) {
      finalClauses.push({ text: leftText, charOffset: clause.charOffset });
      splitLongClause({ text: rightText, charOffset: clause.charOffset + splitPosInClause });
    } else {
      finalClauses.push(clause);
    }
  };

  for (const clause of mergedClauses) {
    splitLongClause(clause);
  }

  return finalClauses;
}

class TextToSpeechWorker extends BaseWorker<TextToSpeech> {
  private voices: VoiceInfo[] = [];
  private activeStream: TextToSpeechStream | null = null;
  private cancelRequested = false;
  private activeVirtualFileName = 'kokoro_model.litertlm';
  private prevVirtualFileName = '';

  protected override async handleMessage(event: MessageEvent) {
    if (event.data?.type === 'CANCEL') {
      this.cancelRequested = true;
      try {
        this.activeStream?.cancel();
      } catch (_) {}
      try {
        this.taskInstance?.cancel();
      } catch (_) {}
      self.postMessage({ type: 'CANCELLED', sessionId: event.data?.sessionId });
      return;
    }
    return super.handleMessage(event);
  }

  protected async initializeTask(): Promise<void> {
    let effectiveDelegate: 'CPU' | 'GPU' = this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU';
    let webGpuDevice: any = null;
    if (effectiveDelegate === 'GPU') {
      webGpuDevice = await getHardwareWebGpuDevice();
      if (!webGpuDevice) {
        effectiveDelegate = 'CPU';
        this.currentOptions.delegate = 'CPU';
        self.postMessage({
          type: 'DELEGATE_FALLBACK',
          reason: 'WebGPU hardware adapter unavailable',
          advice: 'Using CPU (XNNPACK) delegate.',
        });
      }
    }

    const fileset = await FilesetResolver.forAudioTasks(this.getWasmPath(), true);
    const wasmBinaryUrl = fileset.wasmBinaryPath;
    await ensureAudioModuleFactory(fileset.wasmLoaderPath);
    const compiledWasmPromise = getCompiledWasmModule(wasmBinaryUrl);

    const { stream, size } = await this.openModelStream();
    const rawSourceName =
      this.currentOptions.modelFile?.name ||
      String(this.currentOptions.modelAssetPath || '')
        .split('?')[0]
        .split('/')
        .pop() ||
      'kokoro_model.litertlm';
    const virtualFileName = rawSourceName.endsWith('.litertlm') ? rawSourceName : `${rawSourceName}.litertlm`;
    this.prevVirtualFileName = this.activeVirtualFileName;
    this.activeVirtualFileName = virtualFileName;
    const workerSelf = this;

    const moduleConfig: Record<string, any> = {
      ...(webGpuDevice ? { preinitializedWebGPUDevice: webGpuDevice } : {}),
      instantiateWasm(imports: WebAssembly.Imports, successCallback: (instance: WebAssembly.Instance) => void) {
        compiledWasmPromise
          .then((wasmMod) => WebAssembly.instantiate(wasmMod, imports))
          .then((instance) => successCallback(instance))
          .catch((err) => origConsoleError('Failed to instantiate audio WASM:', err));
        return {};
      },
      onRuntimeInitialized() {
        if (webGpuDevice) {
          moduleConfig.preinitializedWebGPUDevice = webGpuDevice;
        }
        delete moduleConfig.createAudioTranscriberFromBuffer;
        delete moduleConfig.createTextToSpeechFromBuffer;

        if (typeof moduleConfig.FS_createDataFile === 'function') {
          const origCreateDataFile = moduleConfig.FS_createDataFile.bind(moduleConfig);
          moduleConfig.FS_createDataFile = (
            parent: string,
            name: string,
            data: Uint8Array,
            canRead: boolean,
            canWrite: boolean,
            canOwn: boolean
          ) => {
            const targetName = name === 'model.litertlm' ? workerSelf.activeVirtualFileName : name;
            if (workerSelf.prevVirtualFileName && workerSelf.prevVirtualFileName !== targetName) {
              try {
                moduleConfig.FS_unlink?.(`/${workerSelf.prevVirtualFileName}`);
              } catch (_) {}
            }
            try {
              moduleConfig.FS_unlink?.(`/${targetName}`);
            } catch (_) {}
            return origCreateDataFile(parent, targetName, data, canRead, canWrite, canOwn);
          };
        }

        if (typeof moduleConfig.createTextToSpeech === 'function') {
          const origCreateTts = moduleConfig.createTextToSpeech.bind(moduleConfig);
          moduleConfig.createTextToSpeech = (modelPath: string, ...args: any[]) => {
            const resolvedPath = modelPath === '/model.litertlm' ? `/${workerSelf.activeVirtualFileName}` : modelPath;
            return origCreateTts(resolvedPath, ...args);
          };
        }
      },
    };
    (self as unknown as { Module: unknown }).Module = moduleConfig;

    this.taskInstance = await TextToSpeech.createFromOptions(fileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: effectiveDelegate,
      },
      ...(size ? { modelAssetSize: size } : {}),
      defaultVoiceId: this.currentOptions.defaultVoiceId || '',
      defaultLocale: this.currentOptions.defaultLocale || 'en-US',
      speakingRate: Number(this.currentOptions.speakingRate) || 1.0,
      cacheDirectory: ':nocache',
    });

    try {
      this.voices = await this.taskInstance.listVoices();
    } catch (_) {
      this.voices = [];
    }

    if (
      virtualFileName.startsWith('phonon') ||
      virtualFileName.startsWith('pocket_tts') ||
      virtualFileName.startsWith('kokoro')
    ) {
      try {
        const warmupStream = await this.taskInstance.synthesizeStream('Hi.', {
          voiceId: this.currentOptions.defaultVoiceId || undefined,
          locale: this.currentOptions.defaultLocale || undefined,
        });
        try {
          await warmupStream.readNext();
          warmupStream.cancel();
        } finally {
          warmupStream.close();
        }
      } catch (_) {}
    }
  }

  protected override getInitPayload(): any {
    return {
      voices: this.voices,
      initToken: this.currentOptions.initToken,
      modelId: this.currentOptions.modelName,
    };
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (!this.taskInstance) {
      throw new Error('TextToSpeech is not initialized.');
    }

    const { type, text, sentenceIdx, sessionId } = data;
    const voiceId = data.voiceId ?? data.options?.voiceId;
    const locale = data.locale ?? data.options?.locale;

    if (type === 'SYNTHESIZE_UNARY') {
      this.cancelRequested = false;
      const rawText = String(text || '');
      const t0 = performance.now();
      const result = await this.taskInstance.synthesize(rawText, {
        voiceId: voiceId || undefined,
        locale: locale || undefined,
      });
      const inferenceTime = performance.now() - t0;
      if (this.cancelRequested) return;

      const samples =
        result.samples instanceof Float32Array
          ? new Float32Array(result.samples)
          : new Float32Array(result.samples || []);

      const byteToUtf16 = buildUtf8ByteToUtf16Map(rawText);
      const searchCursorRef = { pos: 0 };
      const normalizedWords = normalizeWordTimestampsToUtf16(
        rawText,
        0,
        0,
        result.wordTimestamps || [],
        byteToUtf16,
        searchCursorRef
      );

      self.postMessage(
        {
          type: 'TTS_RESULT',
          samples,
          sampleRateHz: result.sampleRateHz || 24000,
          wordTimestamps: normalizedWords,
          inferenceTime,
          inferenceTimeMs: inferenceTime,
          sentenceIdx,
          sessionId,
        },
        { transfer: [samples.buffer] }
      );
    } else if (type === 'SYNTHESIZE_STREAM') {
      this.cancelRequested = false;
      const tStart = performance.now();
      let ttfaMs: number | null = null;
      let totalSamples = 0;
      let sampleRateHz = 24000;
      let globalChunkIndex = 0;
      const allWordTimestamps: WordTimestamp[] = [];

      const rawText = String(text || '');
      const isKokoro = this.activeVirtualFileName.startsWith('kokoro');
      const clauses: KokoroClauseSlice[] = isKokoro
        ? splitTextIntoKokoroClauses(rawText)
        : [{ text: rawText, charOffset: 0 }];

      for (let cIdx = 0; cIdx < clauses.length; cIdx++) {
        if (this.cancelRequested) break;
        const clause = clauses[cIdx];
        const isLastClause = cIdx === clauses.length - 1;
        const clauseBaseAudioSec = totalSamples / (sampleRateHz || 24000);
        const clauseByteToUtf16 = buildUtf8ByteToUtf16Map(clause.text);
        const clauseSearchCursorRef = { pos: 0 };

        const stream = await this.taskInstance.synthesizeStream(
          clause.text,
          {
            voiceId: voiceId || undefined,
            locale: locale || undefined,
          },
          true
        );
        this.activeStream = stream;

        try {
          while (!this.cancelRequested) {
            const chunk = await stream.readNext();
            if (!chunk || this.cancelRequested) break;

            const now = performance.now();
            const elapsedMs = now - tStart;
            const samples =
              chunk.samples instanceof Float32Array
                ? new Float32Array(chunk.samples)
                : new Float32Array(chunk.samples || []);

            if (samples.length > 0 && ttfaMs === null) {
              ttfaMs = elapsedMs;
            }
            if (chunk.sampleRateHz) {
              sampleRateHz = chunk.sampleRateHz;
            }
            totalSamples += samples.length;

            const chunkWords = normalizeWordTimestampsToUtf16(
              clause.text,
              clause.charOffset,
              clauseBaseAudioSec,
              chunk.wordTimestamps || [],
              clauseByteToUtf16,
              clauseSearchCursorRef
            );
            if (chunkWords.length > 0) {
              allWordTimestamps.push(...chunkWords);
            }

            const isFinalOverall = isLastClause && Boolean(chunk.isFinal);

            self.postMessage(
              {
                type: 'TTS_CHUNK',
                samples,
                sampleRateHz,
                chunkIndex: globalChunkIndex++,
                isFinal: isFinalOverall,
                wordTimestamps: chunkWords,
                ttfaMs: ttfaMs ?? elapsedMs,
                elapsedMs,
                sentenceIdx,
                sessionId,
              },
              { transfer: [samples.buffer] }
            );

            if (chunk.isFinal) break;
            await new Promise((r) => setTimeout(r, 0));
          }
        } finally {
          try {
            stream.close();
          } catch (_) {}
          if (this.activeStream === stream) {
            this.activeStream = null;
          }
        }

        if (!isLastClause && !this.cancelRequested) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      if (this.cancelRequested) return;

      const totalInferenceMs = performance.now() - tStart;
      self.postMessage({
        type: 'TTS_STREAM_DONE',
        totalInferenceMs,
        inferenceTimeMs: totalInferenceMs,
        ttfaMs: ttfaMs ?? totalInferenceMs,
        totalSamples,
        sampleRateHz,
        wordTimestamps: allWordTimestamps,
        sentenceIdx,
        sessionId,
      });
    }
  }
}

const defaultWasmUrl = new URL(
  `${import.meta.env.BASE_URL || '/'}wasm/audio_wasm_module_internal.wasm`,
  self.location.origin
).href;
void ensureAudioModuleFactory();
void getCompiledWasmModule(defaultWasmUrl).catch(() => {});

new TextToSpeechWorker();
