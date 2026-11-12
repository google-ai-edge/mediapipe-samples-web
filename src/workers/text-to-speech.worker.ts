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

import { TextToSpeech, TextToSpeechStream, VoiceInfo } from '@mediapipe/tasks-audio';
import { BaseWorker } from './base-worker';
import { installLitertLmFileNameWorkaround, litertLmFileNameFor } from './litertlm-file-name-workaround';

/**
 * Runs the MediaPipe TextToSpeech task off the main thread.
 *
 * Page -> worker:
 *   INIT        { modelAssetPath | modelFile, delegate, speakingRate }
 *   SYNTHESIZE  { requestId, text, voiceId }   full-utterance synthesis
 *   STREAM      { requestId, text, voiceId }   clause-by-clause streaming
 *   CANCEL      {}                             stops the in-flight request
 *
 * Worker -> page:
 *   INIT_DONE       { voices }
 *   AUDIO_CHUNK     { requestId, samples, sampleRateHz, wordTimestamps, chunkIndex, isFinal, elapsedMs }
 *   SYNTHESIS_DONE  { requestId, totalMs, cancelled }
 *   SYNTHESIS_ERROR { requestId, error }
 */

class TextToSpeechWorker extends BaseWorker<TextToSpeech> {
  private voices: VoiceInfo[] = [];
  private activeStream: TextToSpeechStream | undefined;
  private cancelled = false;
  private latestRequestId = 0;
  private cancelWaitResolver: (() => void) | undefined;
  private pendingSynthesisEvent: MessageEvent | undefined;
  private yieldChannel = new MessageChannel();

  constructor() {
    super();
    // Keep Kokoro's file name (e.g. "kokoro_int8_gpu_fp16.litertlm") in the
    // Wasm file system so the native task detects the model architecture. Other
    // models are detected correctly from their contents.
    installLitertLmFileNameWorkaround(() => {
      const name = litertLmFileNameFor(this.currentOptions.modelFile?.name ?? this.currentOptions.modelAssetPath);
      return name && /kokoro/i.test(name) ? name : undefined;
    });
    // BaseWorker handles one message at a time. CANCEL and superseding
    // STREAM/SYNTHESIZE requests must immediately abort any in-flight stream
    // and coalesce rapid clicks to the latest request.
    const queued = self.onmessage!;
    self.onmessage = (event: MessageEvent) => {
      const msgType = event.data?.type;
      if (msgType === 'CANCEL') {
        this.pendingSynthesisEvent = undefined;
        this.cancel();
        return;
      }
      if (msgType === 'STREAM' || msgType === 'SYNTHESIZE') {
        if (typeof event.data.requestId === 'number') {
          this.latestRequestId = event.data.requestId;
        }
        this.cancel();
        if (this.isProcessing) {
          this.pendingSynthesisEvent = event;
          return;
        }
      }
      if (msgType === 'WARMUP_VOICE') {
        if (this.isProcessing) return;
      }
      queued.call(self, event);
    };
  }

  protected async initializeTask(): Promise<void> {
    const audioFileset = await this.getAudioFileset();
    if (audioFileset.wasmBinaryPath && !String(audioFileset.wasmBinaryPath).includes('?')) {
      audioFileset.wasmBinaryPath = `${audioFileset.wasmBinaryPath}?cb=${Date.now()}` as any;
    }
    // Stream the (large) model straight into the Wasm virtual filesystem.
    const { stream, size } = await this.openModelStream();

    this.taskInstance = await TextToSpeech.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      modelAssetSize: size,
      speakingRate: this.currentOptions.speakingRate ?? 1.0,
    });

    this.voices = await this.taskInstance.listVoices();
    const preferred =
      this.voices.find((v) => v.voiceId === 'af_heart') ??
      this.voices.find((v) => /^en\b/i.test(v.locale ?? '')) ??
      this.voices[0];
    await this.warmupVoice(preferred?.voiceId);
  }

  private async warmupVoice(voiceId?: string): Promise<void> {
    if (!this.taskInstance) return;
    this.cancelled = false;
    try {
      const stream = await this.taskInstance.synthesizeStream('Hi.', voiceId ? { voiceId } : undefined);
      this.activeStream = stream;
      try {
        for await (const _ of stream) {
          break;
        }
      } finally {
        stream.close();
        if (this.activeStream === stream) {
          this.activeStream = undefined;
        }
      }
    } catch (_) {
      // Ignore warm-up errors; actual synthesis will report any real failure.
    }
  }

  protected override getInitPayload(): any {
    return { voices: this.voices };
  }

  private cancel() {
    this.cancelled = true;
    if (this.cancelWaitResolver) {
      const resolve = this.cancelWaitResolver;
      this.cancelWaitResolver = undefined;
      resolve();
    }
    try {
      this.activeStream?.cancel();
    } catch (_) {}
    try {
      this.taskInstance?.cancel();
    } catch (_) {}
  }

  /**
   * Yields to the worker macrotask event loop via MessageChannel (without
   * Chrome's 4ms nested setTimeout clamping) so incoming CANCEL / STREAM
   * messages on `self.onmessage` are processed between every single chunk.
   */
  private yieldToEventLoopOrCancel(): Promise<void> {
    if (this.cancelled) return Promise.resolve();
    return new Promise((resolve) => {
      const finish = () => {
        this.yieldChannel.port1.onmessage = null;
        if (this.cancelWaitResolver === finish) {
          this.cancelWaitResolver = undefined;
        }
        resolve();
      };
      this.cancelWaitResolver = finish;
      this.yieldChannel.port1.onmessage = finish;
      this.yieldChannel.port2.postMessage(null);
    });
  }

  private postChunk(requestId: number, chunk: any, elapsedMs: number) {
    // Copy into a standalone buffer: the samples may be a view on the Wasm heap.
    const samples = new Float32Array(chunk.samples);
    self.postMessage(
      {
        type: 'AUDIO_CHUNK',
        requestId,
        samples,
        sampleRateHz: chunk.sampleRateHz,
        wordTimestamps: chunk.wordTimestamps ?? [],
        chunkIndex: chunk.chunkIndex ?? 0,
        isFinal: chunk.isFinal ?? true,
        elapsedMs,
      },
      { transfer: [samples.buffer] }
    );
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    await this.runSynthesisMessage(data);
    while (this.pendingSynthesisEvent) {
      const nextEvent = this.pendingSynthesisEvent;
      this.pendingSynthesisEvent = undefined;
      await this.runSynthesisMessage(nextEvent.data);
    }
  }

  private async runSynthesisMessage(data: any): Promise<void> {
    if (data.type === 'WARMUP_VOICE') {
      await this.warmupVoice(data.voiceId);
      return;
    }
    if (data.type !== 'SYNTHESIZE' && data.type !== 'STREAM') return;

    const { requestId, text, voiceId } = data;
    // Skip stale requests that were superseded by a newer click while queued.
    if (typeof requestId === 'number' && requestId < this.latestRequestId) {
      return;
    }
    if (!this.taskInstance) {
      self.postMessage({ type: 'SYNTHESIS_ERROR', requestId, error: 'Not initialized' });
      return;
    }

    this.cancelled = false;
    const startMs = performance.now();
    let computeMs = 0;
    const options = voiceId ? { voiceId } : undefined;

    try {
      if (data.type === 'SYNTHESIZE') {
        const t0 = performance.now();
        const result = await this.taskInstance.synthesize(text, options);
        computeMs = performance.now() - t0;
        if (!this.cancelled && requestId === this.latestRequestId) {
          this.postChunk(requestId, { ...result, chunkIndex: 0, isFinal: true }, computeMs);
        }
      } else {
        let stepStartMs = performance.now();
        this.activeStream = await this.taskInstance.synthesizeStream(text, options);
        try {
          for await (const chunk of this.activeStream) {
            const now = performance.now();
            computeMs += Math.max(0, now - stepStartMs);
            if (this.cancelled || requestId !== this.latestRequestId) break;

            this.postChunk(requestId, chunk, computeMs);
            if (chunk.isFinal) break;

            // Yield to the worker macrotask loop between chunks so incoming
            // CANCEL or superseding sentence clicks are processed immediately.
            await this.yieldToEventLoopOrCancel();
            if (this.cancelled || requestId !== this.latestRequestId) break;
            stepStartMs = performance.now();
          }
        } finally {
          this.activeStream.close();
          this.activeStream = undefined;
        }
      }
      self.postMessage({
        type: 'SYNTHESIS_DONE',
        requestId,
        totalMs: computeMs > 0 ? computeMs : performance.now() - startMs,
        cancelled: this.cancelled || requestId !== this.latestRequestId,
      });
    } catch (e: any) {
      if (this.cancelled || requestId !== this.latestRequestId) {
        self.postMessage({
          type: 'SYNTHESIS_DONE',
          requestId,
          totalMs: computeMs > 0 ? computeMs : performance.now() - startMs,
          cancelled: true,
        });
        return;
      }
      console.error('Worker synthesis error:', e);
      self.postMessage({ type: 'SYNTHESIS_ERROR', requestId, error: e?.message || 'Synthesis failed' });
    }
  }
}

new TextToSpeechWorker();
