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

  constructor() {
    super();
    // BaseWorker handles one message at a time. CANCEL must not wait behind the
    // synthesis it is meant to stop, so it is handled before the queue.
    const queued = self.onmessage!;
    self.onmessage = (event: MessageEvent) => {
      if (event.data?.type === 'CANCEL') {
        this.cancel();
        return;
      }
      queued.call(self, event);
    };
  }

  protected async initializeTask(): Promise<void> {
    const audioFileset = await this.getAudioFileset();
    // Stream the (large) model straight into the Wasm virtual filesystem.
    const { stream, size } = await this.openModelStream();

    this.taskInstance = await TextToSpeech.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      modelAssetSize: size,
      speakingRate: this.currentOptions.speakingRate ?? 1.0,
      // The XNNPack weight cache can't be built in the Wasm file system and
      // makes inference fail with "Failed to allocate tensors".
      cacheDirectory: ':nocache',
    });

    this.voices = await this.taskInstance.listVoices();
  }

  protected override getInitPayload(): any {
    return { voices: this.voices };
  }

  private cancel() {
    this.cancelled = true;
    try {
      this.activeStream?.cancel();
    } catch (_) {}
    try {
      this.taskInstance?.cancel();
    } catch (_) {}
  }

  private postChunk(requestId: number, chunk: any, startMs: number) {
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
        elapsedMs: performance.now() - startMs,
      },
      { transfer: [samples.buffer] }
    );
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type !== 'SYNTHESIZE' && data.type !== 'STREAM') return;

    const { requestId, text, voiceId } = data;
    if (!this.taskInstance) {
      self.postMessage({ type: 'SYNTHESIS_ERROR', requestId, error: 'Not initialized' });
      return;
    }

    this.cancelled = false;
    const startMs = performance.now();
    const options = voiceId ? { voiceId } : undefined;

    try {
      if (data.type === 'SYNTHESIZE') {
        const result = await this.taskInstance.synthesize(text, options);
        if (!this.cancelled) {
          this.postChunk(requestId, { ...result, chunkIndex: 0, isFinal: true }, startMs);
        }
      } else {
        this.activeStream = await this.taskInstance.synthesizeStream(text, options);
        try {
          for await (const chunk of this.activeStream) {
            if (this.cancelled) break;
            this.postChunk(requestId, chunk, startMs);
          }
        } finally {
          this.activeStream.close();
          this.activeStream = undefined;
        }
      }
      self.postMessage({
        type: 'SYNTHESIS_DONE',
        requestId,
        totalMs: performance.now() - startMs,
        cancelled: this.cancelled,
      });
    } catch (e: any) {
      if (this.cancelled) {
        self.postMessage({ type: 'SYNTHESIS_DONE', requestId, totalMs: performance.now() - startMs, cancelled: true });
        return;
      }
      console.error('Worker synthesis error:', e);
      self.postMessage({ type: 'SYNTHESIS_ERROR', requestId, error: e?.message || 'Synthesis failed' });
    }
  }
}

new TextToSpeechWorker();
