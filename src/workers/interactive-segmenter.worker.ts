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
import { InteractiveSegmenter, DrawingUtils, MPMask, Stroke } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

/** Confidence above which a mask pixel counts as foreground. */
const FOREGROUND_THRESHOLD = 0.5;

/**
 * Summarizes a confidence mask for automated validation: how much of the image
 * is foreground, where the foreground lies (normalized bounding box and
 * centroid), and how much of each stroke lies on the foreground.
 */
function summarizeMask(mask: MPMask, strokes: readonly Stroke[]): Record<string, unknown> {
  const values = mask.getAsFloat32Array();
  const { width, height } = mask;
  let foreground = 0;
  let nonFinite = 0;
  let sum = 0;
  let max = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      nonFinite++;
      continue;
    }
    if (v > FOREGROUND_THRESHOLD) {
      const x = i % width;
      const y = (i - x) / width;
      foreground++;
      sumX += x;
      sumY += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (v > max) max = v;
    sum += v;
  }

  const strokeStats = strokes.map((stroke) => {
    let onForeground = 0;
    for (const p of stroke.point) {
      const x = Math.min(width - 1, Math.max(0, Math.round(p.x * (width - 1))));
      const y = Math.min(height - 1, Math.max(0, Math.round(p.y * (height - 1))));
      if (values[y * width + x] > FOREGROUND_THRESHOLD) onForeground++;
    }
    return {
      brushMode: stroke.brushMode,
      pointCount: stroke.point.length,
      foregroundFraction: stroke.point.length > 0 ? onForeground / stroke.point.length : 0,
    };
  });

  return {
    width,
    height,
    foregroundFraction: values.length > 0 ? foreground / values.length : 0,
    // Normalized to [0, 1]; null when the mask has no foreground.
    foregroundBox:
      foreground > 0
        ? { minX: minX / width, minY: minY / height, maxX: (maxX + 1) / width, maxY: (maxY + 1) / height }
        : null,
    foregroundCentroid: foreground > 0 ? { x: sumX / foreground / width, y: sumY / foreground / height } : null,
    mean: values.length > 0 ? sum / values.length : 0,
    max,
    nonFiniteCount: nonFinite,
    strokes: strokeStats,
  };
}

class InteractiveSegmenterWorker extends BaseWorker<InteractiveSegmenter> {
  private renderCanvas?: OffscreenCanvas;
  private drawingUtils?: DrawingUtils;

  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();

    if (!this.renderCanvas) {
      this.renderCanvas = new OffscreenCanvas(1, 1);
    }

    // Try to get WebGL2 context safely (do not fail if creation returns null)
    const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext | null;

    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate || 'GPU',
      },
      canvas: this.renderCanvas,
    });

    if (glCtx) {
      try {
        this.drawingUtils = new DrawingUtils(glCtx);
      } catch (e) {
        console.warn('Failed to initialize DrawingUtils with WebGL context:', e);
      }
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    const { type, ...rest } = data;

    if (type === 'CLEAR') {
      (self as any).postMessage({
        type: 'SEGMENT_RESULT',
        maskBitmap: null,
        width: 0,
        height: 0,
        inferenceTime: 0,
        reqId: rest.reqId,
      });
      return;
    }

    if (type === 'SET_IMAGE' && this.taskInstance) {
      const { bitmap } = rest;
      if (bitmap) {
        if (this.renderCanvas) {
          const sizeChanged = this.renderCanvas.width !== bitmap.width || this.renderCanvas.height !== bitmap.height;

          if (sizeChanged) {
            this.renderCanvas.width = bitmap.width;
            this.renderCanvas.height = bitmap.height;

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
            if (glCtx && !glCtx.isContextLost()) {
              try {
                glCtx.viewport(0, 0, bitmap.width, bitmap.height);
                this.drawingUtils = new DrawingUtils(glCtx);
              } catch (e) {
                console.warn('Failed to update WebGL viewport or DrawingUtils on SET_IMAGE:', e);
              }
            }
          }
        }

        this.taskInstance.setImage(bitmap);
        bitmap.close();
      }
      return;
    }

    if (type === 'SEGMENT' && this.taskInstance) {
      try {
        const { strokes } = rest;
        const strokeList = strokes && strokes.length > 0 ? strokes : [];

        // Only segment if there are strokes
        if (strokeList.length === 0) {
          (self as any).postMessage({
            type: 'SEGMENT_RESULT',
            maskBitmap: null,
            width: 0,
            height: 0,
            inferenceTime: 0,
            reqId: rest.reqId,
          });
          return;
        }

        const startTimeMs = performance.now();
        const mask = this.taskInstance.segment(strokeList);
        const inferenceTime = performance.now() - startTimeMs;

        let maskBitmap: ImageBitmap | null = null;
        let maskStats: Record<string, unknown> | null = null;
        let width = 0;
        let height = 0;

        if (mask) {
          width = mask.width;
          height = mask.height;

          if (this.renderCanvas && this.drawingUtils) {
            let sizeChanged = false;
            if (this.renderCanvas.width !== width) {
              this.renderCanvas.width = width;
              sizeChanged = true;
            }
            if (this.renderCanvas.height !== height) {
              this.renderCanvas.height = height;
              sizeChanged = true;
            }

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
            if (glCtx && !glCtx.isContextLost() && sizeChanged) {
              glCtx.viewport(0, 0, width, height);
              // Recreate DrawingUtils after resizing the canvas to avoid dimension mismatch errors on CPU
              try {
                this.drawingUtils = new DrawingUtils(glCtx);
              } catch (e) {
                console.warn('Failed to recreate DrawingUtils:', e);
              }
            }

            try {
              // Use bright magenta to ensure it stands out on almost any background, especially blue.
              this.drawingUtils.drawConfidenceMask(
                mask,
                [0, 0, 0, 0], // Background -> Transparent
                [255, 0, 255, 180] // Foreground -> Bright magenta
              );
              maskBitmap = this.renderCanvas.transferToImageBitmap();
            } catch (e) {
              console.warn('DrawingUtils drawConfidenceMask failed:', e);
            }
          }
          if (rest.collectMaskStats) {
            // Summarize the raw mask for automated validation (see test-hooks.ts).
            try {
              maskStats = summarizeMask(mask, strokeList);
            } catch (e: any) {
              maskStats = { error: e?.message || String(e) };
            }
          }
          mask.close();
        }

        (self as any).postMessage(
          {
            type: 'SEGMENT_RESULT',
            maskBitmap,
            width,
            height,
            inferenceTime,
            reqId: rest.reqId,
            maskStats,
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
