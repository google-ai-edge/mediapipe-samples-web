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

    // Initialize WebGL context BEFORE passing to MediaPipe.
    const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;

    this.taskInstance = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.currentOptions.modelAssetPath,
        delegate: this.currentOptions.delegate || 'GPU',
      },
      canvas: this.renderCanvas,
    });

    if (glCtx) {
      this.drawingUtils = new DrawingUtils(glCtx);
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
      });
      return;
    }

    if (type === 'SET_IMAGE' && this.taskInstance) {
      const { bitmap } = rest;
      if (bitmap) {
        if (this.renderCanvas) {
          let sizeChanged = false;
          if (this.renderCanvas.width !== bitmap.width) {
            this.renderCanvas.width = bitmap.width;
            sizeChanged = true;
          }
          if (this.renderCanvas.height !== bitmap.height) {
            this.renderCanvas.height = bitmap.height;
            sizeChanged = true;
          }

          const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
          if (glCtx && sizeChanged) {
            glCtx.viewport(0, 0, bitmap.width, bitmap.height);
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

            const glCtx = this.renderCanvas.getContext('webgl2') as WebGL2RenderingContext;
            if (glCtx && sizeChanged) {
              glCtx.viewport(0, 0, width, height);
            }

            // Using drawConfidenceMask as the new API returns a confidence-like mask.
            // We color Foreground semi-transparent blue, and Background transparent.
            this.drawingUtils.drawConfidenceMask(
              mask,
              [0, 0, 0, 0], // Background -> Transparent
              [0, 0, 255, 128] // Foreground -> Semi-transparent blue
            );
            maskBitmap = this.renderCanvas.transferToImageBitmap();
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
