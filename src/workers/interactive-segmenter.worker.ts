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

/// <reference types="vite/client" />
import { InteractiveSegmenter, DrawingUtils, RGBAColor } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class InteractiveSegmenterWorker extends BaseWorker<InteractiveSegmenter> {
  private renderCanvas?: OffscreenCanvas;

  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();

    if (!this.renderCanvas) {
      this.renderCanvas = new OffscreenCanvas(1, 1);
    }

    this.taskInstance = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.currentOptions.modelAssetPath,
        delegate: this.currentOptions.delegate || 'GPU',
      },
      canvas: this.renderCanvas,
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  }

  protected async updateOptions(_: any): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        runningMode: 'IMAGE',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    const { type, ...rest } = data;

    if (type === 'SEGMENT' && this.taskInstance) {
      try {
        const { bitmap, pt } = rest;
        const timestampMs = performance.now();

        const result = this.taskInstance.segment(bitmap, {
          keypoint: { x: pt.x, y: pt.y },
        });

        const categoryMask = result.categoryMask;
        let maskBitmap: ImageBitmap | null = null;
        let width = 0;
        let height = 0;

        if (categoryMask) {
          width = categoryMask.width;
          height = categoryMask.height;

          if (this.renderCanvas) {
            this.renderCanvas.width = width;
            this.renderCanvas.height = height;

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
            if (glCtx) {
              const drawingUtils = new DrawingUtils(glCtx);
              const transparent: RGBAColor = [0, 0, 0, /* alpha= */ 0];

              // Target (category === 0) gets semi-transparent blue, everything else gets transparent
              const colors: RGBAColor[] = [];
              for (let i = 0; i < 256; i++) {
                colors.push(i === 0 ? [0, 0, 255, /* alpha= */ 128] : transparent);
              }

              drawingUtils.drawCategoryMask(categoryMask, colors, transparent);
              maskBitmap = this.renderCanvas.transferToImageBitmap();
            }
          }
          categoryMask.close();
        }

        (self as any).postMessage(
          {
            type: 'SEGMENT_RESULT',
            maskBitmap,
            width,
            height,
            inferenceTime: performance.now() - timestampMs,
          },
          maskBitmap ? [maskBitmap] : []
        );
      } catch (error: any) {
        console.error('Segmentation Error:', error);
        self.postMessage({ type: 'ERROR', error: error.message });
      }
    }
  }
}

new InteractiveSegmenterWorker();
