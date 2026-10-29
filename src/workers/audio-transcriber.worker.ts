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

import { AudioTranscriber, AudioTranscriberResult } from '@mediapipe/tasks-audio';
import { BaseWorker } from './base-worker';

/**
 * Runs the MediaPipe AudioTranscriber task off the main thread.
 *
 * Page -> worker:
 *   INIT          { modelAssetPath | modelFile, delegate, sampleRateHz, enableInterimResults }
 *   STREAM_CHUNK  { sessionId, audioData, final }  feeds 16 kHz mono PCM; `final` flushes the stream
 *
 * Worker -> page:
 *   TRANSCRIPTION_RESULT { sessionId, segments, final, inferenceTime, audioMs }
 *   TRANSCRIPTION_ERROR  { sessionId, error }
 */
class AudioTranscriberWorker extends BaseWorker<AudioTranscriber> {
  protected async initializeTask(): Promise<void> {
    const audioFileset = await this.getAudioFileset();
    // Stream the model straight into the Wasm heap (models are ~100-700 MB).
    const { stream, size } = await this.openModelStream();

    this.taskInstance = await AudioTranscriber.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      modelAssetSize: size,
      sampleRateHz: this.currentOptions.sampleRateHz ?? 16000,
      enableInterimResults: this.currentOptions.enableInterimResults ?? true,
      modelArchitecture: 'AUTO',
      // Compiled-model caches don't work in the Wasm file system (see the
      // TextToSpeech worker).
      cacheDir: ':nocache',
    });

    // Reduces the latency of the first chunk.
    await this.taskInstance.prewarm();
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type !== 'STREAM_CHUNK') return;

    if (!this.taskInstance) {
      self.postMessage({ type: 'TRANSCRIPTION_ERROR', error: 'Not initialized' });
      return;
    }

    const audioData: Float32Array = data.audioData ?? new Float32Array(0);
    const sampleRateHz = this.currentOptions.sampleRateHz ?? 16000;
    const startTimeMs = performance.now();
    let result: AudioTranscriberResult;
    try {
      if (data.final) {
        // Feed the remaining audio, then flush whatever is still buffered.
        const segments = [];
        if (audioData.length > 0) {
          segments.push(...(await this.taskInstance.transcribeStream(audioData)).segments);
        }
        segments.push(...(await this.taskInstance.finishStream()).segments);
        result = { segments, text: '' };
      } else {
        result = await this.taskInstance.transcribeStream(audioData);
      }
    } catch (e: any) {
      console.error('Worker transcription error:', e);
      self.postMessage({
        type: 'TRANSCRIPTION_ERROR',
        sessionId: data.sessionId,
        error: e?.message || 'Transcription failed',
      });
      return;
    }

    self.postMessage({
      type: 'TRANSCRIPTION_RESULT',
      sessionId: data.sessionId,
      segments: result.segments,
      final: !!data.final,
      inferenceTime: performance.now() - startTimeMs,
      audioMs: (audioData.length / sampleRateHz) * 1000,
    });
  }
}

new AudioTranscriberWorker();
