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

import { TextEmbedder, TextEmbedderResult } from '@mediapipe/tasks-text';
import { BaseWorker } from './base-worker';

class TextEmbedderWorker extends BaseWorker<TextEmbedder> {
  protected async initializeTask(): Promise<void> {
    const text = await this.getTextFileset();
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await TextEmbedder.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
    });
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'EMBED') {
      if (!this.taskInstance || !data.text1) {
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing text' });
        return;
      }

      try {
        const result1 = this.taskInstance.embed(data.text1);
        let result2: TextEmbedderResult | undefined;
        let similarity: number | undefined;

        if (data.text2) {
          result2 = this.taskInstance.embed(data.text2);
          const embedding1 = result1.embeddings[0];
          const embedding2 = result2.embeddings[0];
          similarity = TextEmbedder.cosineSimilarity(embedding1, embedding2);
        }

        self.postMessage({
          type: 'EMBED_RESULT',
          result1,
          result2,
          similarity,
          timestampMs: data.timestampMs,
        });
      } catch (error: any) {
        console.error('Worker embed error:', error);
        self.postMessage({ type: 'ERROR', error: error.message || 'Embed failed' });
      }
    }
  }
}

new TextEmbedderWorker();
