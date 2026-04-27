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

import { ImageSegmenter, ImageSegmenterResult, DrawingUtils, RGBAColor } from '@mediapipe/tasks-vision';

import { BaseWorker } from './base-worker';

class ImageSegmentationWorker extends BaseWorker<ImageSegmenter> {
  private renderCanvas?: OffscreenCanvas;

  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();
    const modelBuffer = await this.loadModelAsset();

    if (!this.renderCanvas) {
      this.renderCanvas = new OffscreenCanvas(1, 1);
    }

    this.taskInstance = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      canvas: this.renderCanvas,
      runningMode: this.currentOptions.runningMode,
      outputCategoryMask: true,
      outputConfidenceMasks: true,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        runningMode: this.currentOptions.runningMode,
      });
    }
  }

  protected override getInitPayload(): any {
    return {
      labels: this.taskInstance ? this.taskInstance.getLabels() : [],
    };
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'SEGMENT_IMAGE' || data.type === 'SEGMENT_VIDEO') {
      const { bitmap, timestampMs, colors } = data;
      const requiredMode = data.type === 'SEGMENT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        console.warn('ImageSegmenter not initialized.');
        bitmap.close();
        self.postMessage({ type: 'SEGMENT_ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();

      const callback = async (result: ImageSegmenterResult) => {
        try {
          const inferenceTime = performance.now() - startTimeMs;
          bitmap.close();

          let maskBitmap: ImageBitmap | null = null;
          let width = 0;
          let height = 0;

          if (result.categoryMask && colors) {
            width = result.categoryMask.width;
            height = result.categoryMask.height;

            if (!this.renderCanvas || this.renderCanvas.width !== width || this.renderCanvas.height !== height) {
              this.renderCanvas = new OffscreenCanvas(width, height);
            }

            this.renderCanvas.width = width;
            this.renderCanvas.height = height;

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
            if (glCtx) {
              const drawingUtils = new DrawingUtils(glCtx);
              const transparent: RGBAColor = [0, 0, 0, 0];
              drawingUtils.drawCategoryMask(result.categoryMask, colors, transparent);
              maskBitmap = this.renderCanvas.transferToImageBitmap();
            }
            result.categoryMask.close();
          }

          if (result.confidenceMasks) {
            result.confidenceMasks.forEach((m: any) => m.close());
          }

          (self as any).postMessage(
            {
              type: 'SEGMENT_RESULT',
              mode: requiredMode,
              maskBitmap,
              width,
              height,
              inferenceTime,
            },
            maskBitmap ? [maskBitmap] : []
          );
        } catch (e: any) {
          console.error('Worker callback error:', e);
          self.postMessage({ type: 'SEGMENT_ERROR', error: e.message || 'Processing failed' });
        }
      };

      try {
        if (requiredMode === 'VIDEO') {
          this.taskInstance.segmentForVideo(bitmap, timestampMs, callback);
        } else {
          const result = this.taskInstance.segment(bitmap);
          callback(result);
        }
      } catch (e: any) {
        console.error('Worker segmentation error:', e);
        bitmap.close();
        self.postMessage({ type: 'SEGMENT_ERROR', error: e.message || 'Segmentation failed' });
      }
    }
  }
}

new ImageSegmentationWorker();
