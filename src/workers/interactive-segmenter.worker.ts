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
  private drawingUtils?: DrawingUtils;

  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();

    if (!this.renderCanvas) {
      this.renderCanvas = new OffscreenCanvas(1, 1);
    }

    // Try to get WebGL2 context safely (do not fail if creation returns null)
    const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext | null;

    this.taskInstance = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.currentOptions.modelAssetPath,
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
        const timestampMs = performance.now();
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
        const mask = this.taskInstance.segment(strokeList);

        let maskBitmap: ImageBitmap | null = null;
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
          mask.close();
        }

        (self as any).postMessage(
          {
            type: 'SEGMENT_RESULT',
            maskBitmap,
            width,
            height,
            inferenceTime: performance.now() - timestampMs,
            reqId: rest.reqId,
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
