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
        const result = await this.taskInstance.transcribe(samples16k, true);
        const inferenceTime = performance.now() - startTime;
        self.postMessage({
          type: 'TRANSCRIBE_RESULT',
          result,
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
          latestInterimSegments = [];
          for (const seg of stepResult.segments || []) {
            if (seg.isFinal !== false) {
              if (seg.transcript && seg.transcript.trim()) {
                confirmedSegments.push(seg);
              }
            } else if (seg.transcript && seg.transcript.trim()) {
              latestInterimSegments.push(seg);
            }
          }
          if ((!stepResult.segments || stepResult.segments.length === 0) && stepResult.text?.trim()) {
            latestInterimSegments.push({
              transcript: stepResult.text.trim(),
              isFinal: false,
              fromTimestampMs: Math.round((startIdx / TARGET_SAMPLE_RATE) * 1000),
              toTimestampMs: Math.round((endIdx / TARGET_SAMPLE_RATE) * 1000),
            });
          }
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
          confirmedSegments.push({ ...seg, isFinal: true });
        }
      }
      if (confirmedSegments.length === 0 && flushResult.text?.trim()) {
        confirmedSegments.push({
          transcript: flushResult.text.trim(),
          isFinal: true,
          fromTimestampMs: 0,
          toTimestampMs: Math.round(audioDurationSec * 1000),
        });
      }

      const finalText = confirmedSegments
        .map((s) => s.transcript.trim())
        .filter(Boolean)
        .join(' ');

      self.postMessage({
        type: 'TRANSCRIBE_RESULT',
        result: {
          text: finalText,
          segments: confirmedSegments,
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
        this.micInterimSegments = [];
        for (const seg of stepResult.segments || []) {
          if (seg.isFinal !== false) {
            if (seg.transcript && seg.transcript.trim()) {
              this.micConfirmedSegments.push(seg);
            }
          } else if (seg.transcript && seg.transcript.trim()) {
            this.micInterimSegments.push(seg);
          }
        }
        if ((!stepResult.segments || stepResult.segments.length === 0) && stepResult.text?.trim()) {
          this.micInterimSegments.push({
            transcript: stepResult.text.trim(),
            isFinal: false,
            fromTimestampMs: Math.round((startSample / TARGET_SAMPLE_RATE) * 1000),
            toTimestampMs: Math.round((endSample / TARGET_SAMPLE_RATE) * 1000),
          });
        }
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
                this.micConfirmedSegments.push(seg);
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
          this.micConfirmedSegments.push({ ...seg, isFinal: true });
        }
      }
      if (this.micConfirmedSegments.length === 0 && flushResult.text?.trim()) {
        this.micConfirmedSegments.push({
          transcript: flushResult.text.trim(),
          isFinal: true,
          fromTimestampMs: 0,
          toTimestampMs: Math.round((this.micSamplesProcessed / TARGET_SAMPLE_RATE) * 1000),
        });
      }

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
