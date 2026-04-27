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

import { TextClassifier } from '@mediapipe/tasks-text';
import { BaseWorker } from './base-worker';

class TextClassifierWorker extends BaseWorker<TextClassifier> {
  protected async initializeTask(): Promise<void> {
    const text = await this.getTextFileset();
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await TextClassifier.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: this.currentOptions.maxResults || 3,
      scoreThreshold: this.currentOptions.scoreThreshold || 0,
    });
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'CLASSIFY') {
      if (!this.taskInstance || !data.text) {
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing text' });
        return;
      }

      try {
        const result = this.taskInstance.classify(data.text);
        self.postMessage({
          type: 'CLASSIFY_RESULT',
          result,
          timestampMs: data.timestampMs,
        });
      } catch (error: any) {
        console.error('Worker classify error:', error);
        self.postMessage({ type: 'ERROR', error: error.message || 'Classification failed' });
      }
    }
  }
}

new TextClassifierWorker();
