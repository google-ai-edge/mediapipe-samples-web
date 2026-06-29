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
import { InteractiveSegmenter, DrawingUtils } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class InteractiveSegmenterWorker extends BaseWorker<InteractiveSegmenter> {
  private renderCanvas?: OffscreenCanvas;
  private currentImagePixelPtr = 0;

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
    });
  }

  protected async updateOptions(_: any): Promise<void> {
    // InteractiveSegmenter (New API) options only support canvas and baseOptions.
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
      });
      return;
    }

    if (type === 'SEGMENT' && this.taskInstance) {
      try {
        const { bitmap, pt, strokes, brushMode } = rest;
        const timestampMs = performance.now();

        const wasmModule = (this.taskInstance as any).i;
        const handle = (this.taskInstance as any).h;

        if (!wasmModule || !handle) {
          throw new Error('WASM module or handle not accessible.');
        }

        // Free old pixel data
        if (this.currentImagePixelPtr !== 0) {
          wasmModule._free(this.currentImagePixelPtr);
          this.currentImagePixelPtr = 0;
        }

        // Extract pixels
        const bitmapWidth = bitmap.width;
        const bitmapHeight = bitmap.height;
        const canvas = new OffscreenCanvas(bitmapWidth, bitmapHeight);
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(bitmap, 0, 0);
        const pixels = ctx.getImageData(0, 0, bitmapWidth, bitmapHeight).data;
        bitmap.close();

        // Allocate and copy
        const pixelPtr = wasmModule._malloc(pixels.length);
        wasmModule.HEAPU8.set(pixels, pixelPtr);
        this.currentImagePixelPtr = pixelPtr;

        // Set image
        const success = wasmModule._interactive_segmenter_set_image(
          handle,
          pixelPtr,
          bitmapWidth,
          bitmapHeight,
          4 // numChannels
        );

        if (!success) {
          throw new Error('Failed to set image on native engine.');
        }

        // Segment with provided strokes list or construct single point stroke
        const strokeList =
          strokes && strokes.length > 0
            ? strokes
            : [
                {
                  isCompleted: true,
                  brushMode: brushMode ?? 1, // Default to POSITIVE (1)
                  point: pt ? [{ x: pt.x, y: pt.y }] : [],
                },
              ];

        const mask = this.taskInstance.segment(strokeList);

        let maskBitmap: ImageBitmap | null = null;
        let width = 0;
        let height = 0;

        if (mask) {
          width = mask.width;
          height = mask.height;

          if (this.renderCanvas) {
            this.renderCanvas.width = width;
            this.renderCanvas.height = height;

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
            if (glCtx) {
              const drawingUtils = new DrawingUtils(glCtx);

              // Using drawConfidenceMask as the new API returns a confidence-like mask.
              // We color Foreground semi-transparent blue, and Background transparent.
              drawingUtils.drawConfidenceMask(
                mask,
                [0, 0, 0, 0], // Background -> Transparent
                [0, 0, 255, 128] // Foreground -> Semi-transparent blue
              );
              maskBitmap = this.renderCanvas.transferToImageBitmap();
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
