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

import { AudioTranscriber, FilesetResolver, type TranscriptionSegment } from '@mediapipe/tasks-audio';
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

const TARGET_SAMPLE_RATE = 16000;

let cachedWebGpuDevicePromise: Promise<any | null> | null = null;

async function getHardwareWebGpuDevice(): Promise<any | null> {
  if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
    return null;
  }
  if (!cachedWebGpuDevicePromise) {
    cachedWebGpuDevicePromise = (async () => {
      try {
        const gpu = (navigator as any).gpu;
        const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
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
        const device = await adapter.requestDevice({
          requiredFeatures,
          requiredLimits,
        });
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

function resampleToTarget(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (!samples || samples.length === 0) return new Float32Array(0);
  if (fromRate === toRate || fromRate <= 0) return samples;
  const targetLen = Math.max(1, Math.round((samples.length * toRate) / fromRate));
  const out = new Float32Array(targetLen);
  const ratio = (samples.length - 1) / Math.max(1, targetLen - 1);
  for (let i = 0; i < targetLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = samples[idx] || 0;
    const b = samples[Math.min(samples.length - 1, idx + 1)] ?? a;
    out[i] = a + frac * (b - a);
  }
  return out;
}

function canonToken(token: string): string {
  return token.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
}

function isSuffixFragmentOverlap(prevTok: string, currTok: string): boolean {
  const cPrev = canonToken(prevTok);
  const cCurr = canonToken(currTok);
  return cPrev.length >= 5 && cCurr.length >= 3 && cPrev !== cCurr && cPrev.endsWith(cCurr);
}

function extractTrailingPunct(token: string): string {
  const m = token.match(/([.!?…,:;]+)$/);
  return m ? m[1] : '';
}

function deduplicateTokensWithinSegment(rawTokens: string[]): string[] {
  const tokens = rawTokens.slice();
  let i = 0;
  while (i < tokens.length) {
    let matchedK = 0;
    const maxK = Math.min(6, Math.floor((tokens.length - i) / 2));
    for (let k = maxK; k >= 2; k--) {
      let allMatch = true;
      for (let j = 0; j < k; j++) {
        const cA = canonToken(tokens[i + j]);
        const cB = canonToken(tokens[i + k + j]);
        if (!cA || cA !== cB) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        matchedK = k;
        break;
      }
    }
    if (matchedK >= 2) {
      const firstLastIdx = i + matchedK - 1;
      const secondLastIdx = i + 2 * matchedK - 1;
      const firstPunct = extractTrailingPunct(tokens[firstLastIdx]);
      const secondPunct = extractTrailingPunct(tokens[secondLastIdx]);
      if (!firstPunct && secondPunct) {
        tokens[firstLastIdx] += secondPunct;
      }
      const nextTokAfterRepeat = tokens[i + 2 * matchedK];
      if (nextTokAfterRepeat && /^[a-z]/.test(nextTokAfterRepeat) && tokens[firstLastIdx].endsWith('.')) {
        tokens[firstLastIdx] = tokens[firstLastIdx].replace(/\.+$/, '');
      }
      tokens.splice(i + matchedK, matchedK);
      continue;
    }

    if (i + 1 < tokens.length) {
      const cA = canonToken(tokens[i]);
      const cB = canonToken(tokens[i + 1]);
      const pA = extractTrailingPunct(tokens[i]);
      const isPunctuatedSingleRepeat = Boolean(cA && cA === cB && pA.length > 0);
      if (isPunctuatedSingleRepeat || isSuffixFragmentOverlap(tokens[i], tokens[i + 1])) {
        if (cA === cB) {
          const pB = extractTrailingPunct(tokens[i + 1]);
          if (!pA && pB) {
            tokens[i] += pB;
          }
          const nextTok = tokens[i + 2];
          if (nextTok && /^[a-z]/.test(nextTok) && tokens[i].endsWith('.')) {
            tokens[i] = tokens[i].replace(/\.+$/, '');
          }
        }
        tokens.splice(i + 1, 1);
        continue;
      }
    }
    i++;
  }
  return tokens;
}

function getTailTokens(segments: TranscriptionSegment[], maxTokens = 8): string[] {
  const tail: string[] = [];
  for (let i = segments.length - 1; i >= 0 && tail.length < maxTokens; i--) {
    const words = (segments[i].transcript || '').trim().split(/\s+/).filter(Boolean);
    for (let j = words.length - 1; j >= 0 && tail.length < maxTokens; j--) {
      tail.unshift(words[j]);
    }
  }
  return tail;
}

function findBoundaryOverlapStripCount(
  prevTokens: string[],
  currTokens: string[]
): {
  stripCount: number;
  dropLastPrevToken: boolean;
} {
  if (prevTokens.length === 0 || currTokens.length === 0) {
    return { stripCount: 0, dropLastPrevToken: false };
  }
  const maxK = Math.min(8, prevTokens.length, currTokens.length);
  for (let k = maxK; k >= 1; k--) {
    let match = true;
    for (let j = 0; j < k; j++) {
      const cPrev = canonToken(prevTokens[prevTokens.length - k + j]);
      const cCurr = canonToken(currTokens[j]);
      if (!cPrev || cPrev !== cCurr) {
        match = false;
        break;
      }
    }
    if (match) {
      return { stripCount: k, dropLastPrevToken: false };
    }
  }

  if (isSuffixFragmentOverlap(prevTokens[prevTokens.length - 1], currTokens[0])) {
    return { stripCount: 1, dropLastPrevToken: false };
  }

  // Handle 1-token overshoot at the end of the previous window (e.g. "...trying out" + "trying ...")
  if (prevTokens.length >= 2) {
    const prevWithoutLast = prevTokens.slice(0, prevTokens.length - 1);
    const maxK2 = Math.min(4, prevWithoutLast.length, currTokens.length);
    for (let k = maxK2; k >= 1; k--) {
      if (k === 1 && canonToken(currTokens[0]).length < 4) continue;
      let match = true;
      for (let j = 0; j < k; j++) {
        const cPrev = canonToken(prevWithoutLast[prevWithoutLast.length - k + j]);
        const cCurr = canonToken(currTokens[j]);
        if (!cPrev || cPrev !== cCurr) {
          match = false;
          break;
        }
      }
      if (match) {
        return { stripCount: k, dropLastPrevToken: true };
      }
    }
  }

  return { stripCount: 0, dropLastPrevToken: false };
}

function appendDeduplicatedSegment(confirmedList: TranscriptionSegment[], nextSeg: TranscriptionSegment): void {
  const rawTokens = (nextSeg.transcript || '').trim().split(/\s+/).filter(Boolean);
  if (rawTokens.length === 0) return;
  let currTokens = deduplicateTokensWithinSegment(rawTokens);
  if (currTokens.length === 0) return;

  const prevTokens = getTailTokens(confirmedList, 8);
  const { stripCount, dropLastPrevToken } = findBoundaryOverlapStripCount(prevTokens, currTokens);

  if (stripCount > 0 && confirmedList.length > 0) {
    const lastSeg = confirmedList[confirmedList.length - 1];
    const lastSegTokens = (lastSeg.transcript || '').trim().split(/\s+/).filter(Boolean);
    if (dropLastPrevToken && lastSegTokens.length > 1) {
      lastSegTokens.pop();
    }
    if (lastSegTokens.length > 0) {
      const lastIdx = lastSegTokens.length - 1;
      const strippedLastTok = currTokens[stripCount - 1];
      const prevPunct = extractTrailingPunct(lastSegTokens[lastIdx]);
      const currPunct = extractTrailingPunct(strippedLastTok);
      if (!prevPunct && currPunct) {
        lastSegTokens[lastIdx] += currPunct;
      }
      const firstRemaining = currTokens[stripCount];
      if (firstRemaining && /^[a-z]/.test(firstRemaining) && lastSegTokens[lastIdx].endsWith('.')) {
        lastSegTokens[lastIdx] = lastSegTokens[lastIdx].replace(/\.+$/, '');
      }
      lastSeg.transcript = lastSegTokens.join(' ');
    }
    currTokens = currTokens.slice(stripCount);
  }

  if (currTokens.length === 0) {
    if (confirmedList.length > 0) {
      const lastSeg = confirmedList[confirmedList.length - 1];
      lastSeg.toTimestampMs = Math.max(lastSeg.toTimestampMs || 0, nextSeg.toTimestampMs || 0);
    }
    return;
  }

  confirmedList.push({
    ...nextSeg,
    transcript: currTokens.join(' '),
  });
}

function deduplicateInterimSegments(
  confirmedList: TranscriptionSegment[],
  interimSegments: TranscriptionSegment[]
): TranscriptionSegment[] {
  const out: TranscriptionSegment[] = [];
  const virtualConfirmed = confirmedList.map((s) => ({ ...s }));
  for (const seg of interimSegments) {
    const beforeLen = virtualConfirmed.length;
    appendDeduplicatedSegment(virtualConfirmed, { ...seg, isFinal: false });
    if (virtualConfirmed.length > beforeLen) {
      out.push({ ...virtualConfirmed[virtualConfirmed.length - 1], isFinal: false });
    }
  }
  return out;
}

function normalizeSegmentTimestamps(segments: TranscriptionSegment[], totalDurationMs: number): TranscriptionSegment[] {
  if (segments.length === 0 || totalDurationMs <= 0) return segments;
  const lastToMs = segments[segments.length - 1].toTimestampMs || 0;
  if (lastToMs <= 0 || lastToMs >= totalDurationMs * 0.85) {
    return segments;
  }
  const scale = totalDurationMs / lastToMs;
  let prevEnd = 0;
  return segments.map((seg, idx) => {
    const fromMs = idx === 0 ? 0 : prevEnd;
    const rawTo = Math.round((seg.toTimestampMs || 0) * scale);
    const toMs =
      idx === segments.length - 1 ? totalDurationMs : Math.max(fromMs + 100, Math.min(totalDurationMs, rawTo));
    prevEnd = toMs;
    return {
      ...seg,
      fromTimestampMs: fromMs,
      toTimestampMs: toMs,
    };
  });
}

class AudioTranscriberWorker extends BaseWorker<AudioTranscriber> {
  private cancelRequested = false;
  private micConfirmedSegments: TranscriptionSegment[] = [];
  private micInterimSegments: TranscriptionSegment[] = [];
  private micSamplesProcessed = 0;
  private lastMicWindowLatencyMs = 0;
  private activeVirtualFileName = 'moonshine_tiny.litertlm';
  private prevVirtualFileName = '';

  protected override async handleMessage(event: MessageEvent) {
    if (event.data?.type === 'CANCEL') {
      this.cancelRequested = true;
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
      'moonshine_tiny.litertlm';
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

        if (typeof moduleConfig.createAudioTranscriber === 'function') {
          const origCreateAsr = moduleConfig.createAudioTranscriber.bind(moduleConfig);
          moduleConfig.createAudioTranscriber = (modelPath: string, ...args: any[]) => {
            const resolvedPath = modelPath === '/model.litertlm' ? `/${workerSelf.activeVirtualFileName}` : modelPath;
            return origCreateAsr(resolvedPath, ...args);
          };
        }
      },
    };
    (self as unknown as { Module: unknown }).Module = moduleConfig;

    this.taskInstance = await AudioTranscriber.createFromOptions(fileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: effectiveDelegate,
      },
      ...(size ? { modelAssetSize: size } : {}),
      sampleRateHz: TARGET_SAMPLE_RATE,
      enableInterimResults: this.currentOptions.enableInterimResults ?? true,
      cacheDir: ':nocache',
    });

    try {
      await this.taskInstance.prewarm();
    } catch (_) {
      // Prewarm is optional across model architectures
    }
  }

  protected override getInitPayload(): any {
    return {
      initToken: this.currentOptions.initToken,
      modelId: this.currentOptions.modelName,
    };
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (!this.taskInstance) {
      throw new Error('AudioTranscriber is not initialized.');
    }

    const { type, sessionId } = data;

    if (type === 'TRANSCRIBE_CLIP') {
      this.cancelRequested = false;
      const rawSamples: Float32Array =
        data.audioData instanceof Float32Array ? data.audioData : new Float32Array(data.audioData || []);
      const inputRate = Number(data.sampleRate) || TARGET_SAMPLE_RATE;
      const streamingMode = Boolean(data.streamingMode);
      const chunkDurationMs = Number(data.chunkDurationMs) || 500;

      const samples16k = resampleToTarget(rawSamples, inputRate, TARGET_SAMPLE_RATE);
      const audioDurationSec = samples16k.length / TARGET_SAMPLE_RATE;

      // Flush any residual stream state before starting a new clip
      try {
        await this.taskInstance.finishStream();
      } catch (_) {}

      if (!streamingMode) {
        const startTime = performance.now();
        const rawResult = await this.taskInstance.transcribe(samples16k, true);
        const inferenceTime = performance.now() - startTime;
        const dedupedSegments: TranscriptionSegment[] = [];
        for (const seg of rawResult.segments || []) {
          if (seg.transcript && seg.transcript.trim()) {
            appendDeduplicatedSegment(dedupedSegments, { ...seg, isFinal: true });
          }
        }
        if (dedupedSegments.length === 0 && rawResult.text?.trim()) {
          appendDeduplicatedSegment(dedupedSegments, {
            transcript: rawResult.text.trim(),
            isFinal: true,
            fromTimestampMs: 0,
            toTimestampMs: Math.round(audioDurationSec * 1000),
          });
        }
        const normalizedSegments = normalizeSegmentTimestamps(dedupedSegments, Math.round(audioDurationSec * 1000));
        const finalText = normalizedSegments
          .map((s) => s.transcript.trim())
          .filter(Boolean)
          .join(' ');
        self.postMessage({
          type: 'TRANSCRIBE_RESULT',
          result: {
            text: finalText,
            segments: normalizedSegments,
          },
          inferenceTime,
          audioDurationSec,
          isFinal: true,
          sessionId,
        });
        return;
      }

      // Streaming mode: feed audio in fixed chunks and emit progressive updates
      const chunkSamples = Math.max(1600, Math.round(TARGET_SAMPLE_RATE * (chunkDurationMs / 1000)));
      const totalChunks = Math.max(1, Math.ceil(samples16k.length / chunkSamples));
      const confirmedSegments: TranscriptionSegment[] = [];
      let latestInterimSegments: TranscriptionSegment[] = [];
      let totalInferenceMs = 0;
      let lastWindowLatencyMs = 0;

      for (let i = 0; i < totalChunks; i++) {
        if (this.cancelRequested) {
          try {
            await this.taskInstance.finishStream();
          } catch (_) {}
          return;
        }
        const startIdx = i * chunkSamples;
        const endIdx = Math.min(samples16k.length, startIdx + chunkSamples);
        const chunk = samples16k.subarray(startIdx, endIdx);

        const t0 = performance.now();
        const stepResult = await this.taskInstance.transcribeStream(chunk);
        const chunkLatencyMs = performance.now() - t0;
        totalInferenceMs += chunkLatencyMs;

        const hasStepSegments =
          Boolean(stepResult.segments && stepResult.segments.length > 0) || Boolean(stepResult.text?.trim());
        if (hasStepSegments) {
          lastWindowLatencyMs = chunkLatencyMs;
          const rawInterim: TranscriptionSegment[] = [];
          for (const seg of stepResult.segments || []) {
            if (seg.isFinal !== false) {
              if (seg.transcript && seg.transcript.trim()) {
                appendDeduplicatedSegment(confirmedSegments, seg);
              }
            } else if (seg.transcript && seg.transcript.trim()) {
              rawInterim.push(seg);
            }
          }
          if ((!stepResult.segments || stepResult.segments.length === 0) && stepResult.text?.trim()) {
            rawInterim.push({
              transcript: stepResult.text.trim(),
              isFinal: false,
              fromTimestampMs: Math.round((startIdx / TARGET_SAMPLE_RATE) * 1000),
              toTimestampMs: Math.round((endIdx / TARGET_SAMPLE_RATE) * 1000),
            });
          }
          latestInterimSegments = deduplicateInterimSegments(confirmedSegments, rawInterim);
        }

        const confirmedText = confirmedSegments
          .map((s) => s.transcript.trim())
          .filter(Boolean)
          .join(' ');
        const interimText = latestInterimSegments
          .map((s) => s.transcript.trim())
          .filter(Boolean)
          .join(' ');

        self.postMessage({
          type: 'TRANSCRIBE_STREAM_UPDATE',
          confirmedText,
          interimText,
          segments: [...confirmedSegments, ...latestInterimSegments],
          chunkIndex: i,
          totalChunks,
          audioProcessedSec: endIdx / TARGET_SAMPLE_RATE,
          audioDurationSec,
          chunkLatencyMs: lastWindowLatencyMs,
          didRunWindow: hasStepSegments,
          totalInferenceMs,
          sessionId,
        });

        await new Promise((r) => setTimeout(r, 0));
      }

      if (this.cancelRequested) return;

      const tFlush = performance.now();
      const flushResult = await this.taskInstance.finishStream();
      const flushMs = performance.now() - tFlush;
      totalInferenceMs += flushMs;

      for (const seg of flushResult.segments || []) {
        if (seg.transcript && seg.transcript.trim()) {
          appendDeduplicatedSegment(confirmedSegments, { ...seg, isFinal: true });
        }
      }
      if (confirmedSegments.length === 0 && flushResult.text?.trim()) {
        appendDeduplicatedSegment(confirmedSegments, {
          transcript: flushResult.text.trim(),
          isFinal: true,
          fromTimestampMs: 0,
          toTimestampMs: Math.round(audioDurationSec * 1000),
        });
      }

      const normalizedSegments = normalizeSegmentTimestamps(confirmedSegments, Math.round(audioDurationSec * 1000));
      const finalText = normalizedSegments
        .map((s) => s.transcript.trim())
        .filter(Boolean)
        .join(' ');

      self.postMessage({
        type: 'TRANSCRIBE_RESULT',
        result: {
          text: finalText,
          segments: normalizedSegments,
        },
        inferenceTime: totalInferenceMs,
        audioDurationSec,
        isFinal: true,
        sessionId,
      });
    } else if (type === 'START_MIC_STREAM') {
      this.cancelRequested = false;
      this.micConfirmedSegments = [];
      this.micInterimSegments = [];
      this.micSamplesProcessed = 0;
      this.lastMicWindowLatencyMs = 0;
      try {
        await this.taskInstance.finishStream();
      } catch (_) {}
      self.postMessage({ type: 'MIC_STREAM_STARTED', sessionId });
    } else if (type === 'TRANSCRIBE_MIC_CHUNK') {
      if (this.cancelRequested) return;
      const rawSamples: Float32Array =
        data.audioData instanceof Float32Array ? data.audioData : new Float32Array(data.audioData || []);
      const inputRate = Number(data.sampleRate) || TARGET_SAMPLE_RATE;
      const samples16k = resampleToTarget(rawSamples, inputRate, TARGET_SAMPLE_RATE);
      if (samples16k.length === 0) return;

      const startSample = this.micSamplesProcessed;
      this.micSamplesProcessed += samples16k.length;
      const endSample = this.micSamplesProcessed;

      const t0 = performance.now();
      const stepResult = await this.taskInstance.transcribeStream(samples16k);
      const chunkLatencyMs = performance.now() - t0;

      const hasStepSegments =
        Boolean(stepResult.segments && stepResult.segments.length > 0) || Boolean(stepResult.text?.trim());
      if (hasStepSegments) {
        this.lastMicWindowLatencyMs = chunkLatencyMs;
        const rawInterim: TranscriptionSegment[] = [];
        for (const seg of stepResult.segments || []) {
          if (seg.isFinal !== false) {
            if (seg.transcript && seg.transcript.trim()) {
              appendDeduplicatedSegment(this.micConfirmedSegments, seg);
            }
          } else if (seg.transcript && seg.transcript.trim()) {
            rawInterim.push(seg);
          }
        }
        if ((!stepResult.segments || stepResult.segments.length === 0) && stepResult.text?.trim()) {
          rawInterim.push({
            transcript: stepResult.text.trim(),
            isFinal: false,
            fromTimestampMs: Math.round((startSample / TARGET_SAMPLE_RATE) * 1000),
            toTimestampMs: Math.round((endSample / TARGET_SAMPLE_RATE) * 1000),
          });
        }
        this.micInterimSegments = deduplicateInterimSegments(this.micConfirmedSegments, rawInterim);
      }

      const confirmedText = this.micConfirmedSegments
        .map((s) => s.transcript.trim())
        .filter(Boolean)
        .join(' ');
      const interimText = this.micInterimSegments
        .map((s) => s.transcript.trim())
        .filter(Boolean)
        .join(' ');

      self.postMessage({
        type: 'MIC_STREAM_UPDATE',
        confirmedText,
        interimText,
        segments: [...this.micConfirmedSegments, ...this.micInterimSegments],
        chunkLatencyMs: this.lastMicWindowLatencyMs,
        didRunWindow: hasStepSegments,
        audioProcessedSec: this.micSamplesProcessed / TARGET_SAMPLE_RATE,
        sessionId,
      });
    } else if (type === 'STOP_MIC_STREAM') {
      if (data.audioData) {
        const rawSamples: Float32Array =
          data.audioData instanceof Float32Array ? data.audioData : new Float32Array(data.audioData || []);
        const inputRate = Number(data.sampleRate) || TARGET_SAMPLE_RATE;
        const samples16k = resampleToTarget(rawSamples, inputRate, TARGET_SAMPLE_RATE);
        if (samples16k.length > 0) {
          this.micSamplesProcessed += samples16k.length;
          try {
            const stepResult = await this.taskInstance.transcribeStream(samples16k);
            for (const seg of stepResult.segments || []) {
              if (seg.isFinal !== false && seg.transcript?.trim()) {
                appendDeduplicatedSegment(this.micConfirmedSegments, seg);
              }
            }
          } catch (_) {}
        }
      }

      const t0 = performance.now();
      const flushResult = await this.taskInstance.finishStream();
      const flushLatencyMs = performance.now() - t0;

      for (const seg of flushResult.segments || []) {
        if (seg.transcript && seg.transcript.trim()) {
          appendDeduplicatedSegment(this.micConfirmedSegments, { ...seg, isFinal: true });
        }
      }
      if (this.micConfirmedSegments.length === 0 && flushResult.text?.trim()) {
        appendDeduplicatedSegment(this.micConfirmedSegments, {
          transcript: flushResult.text.trim(),
          isFinal: true,
          fromTimestampMs: 0,
          toTimestampMs: Math.round((this.micSamplesProcessed / TARGET_SAMPLE_RATE) * 1000),
        });
      }

      const normalizedSegments = normalizeSegmentTimestamps(
        this.micConfirmedSegments,
        Math.round((this.micSamplesProcessed / TARGET_SAMPLE_RATE) * 1000)
      );
      this.micConfirmedSegments = normalizedSegments;
      const confirmedText = this.micConfirmedSegments
        .map((s) => s.transcript.trim())
        .filter(Boolean)
        .join(' ');

      self.postMessage({
        type: 'MIC_STREAM_FINAL',
        confirmedText,
        segments: this.micConfirmedSegments,
        flushLatencyMs,
        audioProcessedSec: this.micSamplesProcessed / TARGET_SAMPLE_RATE,
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

new AudioTranscriberWorker();
