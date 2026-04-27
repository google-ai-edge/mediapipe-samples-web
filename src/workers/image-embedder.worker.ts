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

import { ImageEmbedder, ImageEmbedderResult } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class ImageEmbedderWorker extends BaseWorker<ImageEmbedder> {
  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await ImageEmbedder.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
    });
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'EMBED') {
      if (!this.taskInstance || !data.image1) {
        data.image1?.close();
        data.image2?.close();
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing primary image' });
        return;
      }

      try {
        const result1 = this.taskInstance.embed(data.image1);
        data.image1.close();

        let result2: ImageEmbedderResult | undefined;
        let similarity: number | undefined;

        if (data.image2) {
          result2 = this.taskInstance.embed(data.image2);
          data.image2.close();

          const embedding1 = result1.embeddings[0];
          const embedding2 = result2.embeddings[0];
          similarity = ImageEmbedder.cosineSimilarity(embedding1, embedding2);
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

new ImageEmbedderWorker();
