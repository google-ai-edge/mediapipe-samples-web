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

import { ImageSegmenter, ImageSegmenterResult, DrawingUtils, MPMask, RGBAColor } from '@mediapipe/tasks-vision';

import { BaseWorker } from './base-worker';

/**
 * Summarizes a segmentation result for automated validation (see
 * test-hooks.ts): the share and centre of every category in the category mask,
 * and the value range of one confidence mask (`confidenceIndex`, or -1 for none).
 */
function summarizeSegmentation(result: ImageSegmenterResult, confidenceIndex: number): Record<string, unknown> {
  const stats: Record<string, unknown> = {
    confidenceMaskCount: result.confidenceMasks?.length ?? 0,
    categories: [],
    confidence: null,
  };

  const categoryMask: MPMask | undefined = result.categoryMask;
  if (categoryMask) {
    const { width, height } = categoryMask;
    const values = categoryMask.getAsUint8Array();
    const counts = new Float64Array(256);
    const sumX = new Float64Array(256);
    const sumY = new Float64Array(256);
    for (let i = 0; i < values.length; i++) {
      const category = values[i];
      const x = i % width;
      counts[category]++;
      sumX[category] += x;
      sumY[category] += (i - x) / width;
    }
    const categories = [];
    for (let category = 0; category < 256; category++) {
      if (counts[category] === 0) continue;
      categories.push({
        index: category,
        fraction: counts[category] / values.length,
        // Normalized centre of the pixels of this category.
        centroid: { x: sumX[category] / counts[category] / width, y: sumY[category] / counts[category] / height },
      });
    }
    stats.width = width;
    stats.height = height;
    stats.categories = categories.sort((a, b) => b.fraction - a.fraction);
  }

  const confidenceMask = confidenceIndex >= 0 ? result.confidenceMasks?.[confidenceIndex] : undefined;
  if (confidenceMask) {
    const values = confidenceMask.getAsFloat32Array();
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let aboveHalf = 0;
    let nonFinite = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) {
        nonFinite++;
        continue;
      }
      if (v < min) min = v;
      if (v > max) max = v;
      if (v > 0.5) aboveHalf++;
      sum += v;
    }
    const finite = values.length - nonFinite;
    stats.confidence = {
      index: confidenceIndex,
      width: confidenceMask.width,
      height: confidenceMask.height,
      min: finite > 0 ? min : null,
      max: finite > 0 ? max : null,
      mean: finite > 0 ? sum / finite : null,
      aboveHalfFraction: values.length > 0 ? aboveHalf / values.length : 0,
      nonFiniteCount: nonFinite,
    };
  }
  return stats;
}

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
      const { bitmap, timestampMs, colors, collectMaskStats, confidenceIndex } = data;
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
          let maskStats: Record<string, unknown> | null = null;
          let width = 0;
          let height = 0;

          if (result.categoryMask && colors) {
            width = result.categoryMask.width;
            height = result.categoryMask.height;

            if (!this.renderCanvas || this.renderCanvas.width !== width || this.renderCanvas.height !== height) {
              this.renderCanvas = new OffscreenCanvas(width, height);
            }

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
            if (glCtx) {
              const drawingUtils = new DrawingUtils(glCtx);
              const transparent: RGBAColor = [0, 0, 0, 0];
              drawingUtils.drawCategoryMask(result.categoryMask, colors, transparent);
              maskBitmap = this.renderCanvas.transferToImageBitmap();
            }
          }

          if (collectMaskStats) {
            try {
              maskStats = summarizeSegmentation(result, confidenceIndex ?? -1);
            } catch (e: any) {
              maskStats = { error: e?.message || String(e) };
            }
          }

          if (result.categoryMask) {
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
              maskStats,
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
