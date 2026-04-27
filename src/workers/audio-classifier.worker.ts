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

import { AudioClassifier, AudioClassifierResult } from '@mediapipe/tasks-audio';
import { BaseWorker } from './base-worker';

class AudioClassifierWorker extends BaseWorker<AudioClassifier> {
  protected async initializeTask(): Promise<void> {
    const audioFileset = await this.getAudioFileset();
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await AudioClassifier.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: this.currentOptions.maxResults,
      scoreThreshold: this.currentOptions.scoreThreshold,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        maxResults: this.currentOptions.maxResults,
        scoreThreshold: this.currentOptions.scoreThreshold,
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'CLASSIFY') {
      const { audioData, sampleRate } = data;
      if (!this.taskInstance) {
        self.postMessage({ type: 'CLASSIFY_ERROR', error: 'Not initialized' });
        return;
      }

      const startTimeMs = performance.now();
      let results: AudioClassifierResult[] = [];

      try {
        results = this.taskInstance.classify(audioData, sampleRate);
      } catch (e: any) {
        console.error('Worker classification error:', e);
        self.postMessage({ type: 'CLASSIFY_ERROR', error: e.message || 'Classification failed' });
        return;
      }

      self.postMessage({
        type: 'CLASSIFY_RESULT',
        results: results,
        inferenceTime: performance.now() - startTimeMs,
      });
    }
  }
}

new AudioClassifierWorker();
