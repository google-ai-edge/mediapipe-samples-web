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

import { FaceDetectorResult, DrawingUtils } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';

// @ts-ignore
import template from '../templates/face-detector.html?raw';
// @ts-ignore

class FaceDetectorTask extends BaseVisionTask {
  private minDetectionConfidence = 0.5;
  private minSuppressionThreshold = 0.3;

  protected override onInitializeUI() {
    const minDetectionConfidenceInput = document.getElementById('min-detection-confidence') as HTMLInputElement;
    const minDetectionConfidenceValue = document.getElementById('min-detection-confidence-value')!;
    if (minDetectionConfidenceInput && minDetectionConfidenceValue) {
      minDetectionConfidenceInput.addEventListener('input', () => {
        this.minDetectionConfidence = parseFloat(minDetectionConfidenceInput.value);
        minDetectionConfidenceValue.innerText = this.minDetectionConfidence.toString();
        this.worker?.postMessage({ type: 'SET_OPTIONS', minDetectionConfidence: this.minDetectionConfidence });

        if (this.runningMode === 'IMAGE') {
          const testImage = document.getElementById('test-image') as HTMLImageElement;
          if (testImage && testImage.src && testImage.naturalWidth > 0 && this.isWorkerReady) {
            this.detectImage(testImage);
          }
        }
      });
    }

    const minSuppressionThresholdInput = document.getElementById('min-suppression-threshold') as HTMLInputElement;
    const minSuppressionThresholdValue = document.getElementById('min-suppression-threshold-value')!;
    if (minSuppressionThresholdInput && minSuppressionThresholdValue) {
      minSuppressionThresholdInput.addEventListener('input', () => {
        this.minSuppressionThreshold = parseFloat(minSuppressionThresholdInput.value);
        minSuppressionThresholdValue.innerText = this.minSuppressionThreshold.toString();
        this.worker?.postMessage({ type: 'SET_OPTIONS', minSuppressionThreshold: this.minSuppressionThreshold });

        if (this.runningMode === 'IMAGE') {
          const testImage = document.getElementById('test-image') as HTMLImageElement;
          if (testImage && testImage.src && testImage.naturalWidth > 0 && this.isWorkerReady) {
            this.detectImage(testImage);
          }
        }
      });
    }

    this.models = {
      blaze_face_short_range:
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
      blaze_face_full_range:
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'BlazeFace (Short Range)', value: 'blaze_face_short_range', isDefault: true },
        { label: 'BlazeFace (Full Range)', value: 'blaze_face_full_range' },
      ]);
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      minDetectionConfidence: this.minDetectionConfidence,
      minSuppressionThreshold: this.minSuppressionThreshold,
    };
  }

  protected override readonly testTaskName = 'face_detector';

  protected override summarizeTestResult(result: FaceDetectorResult) {
    return {
      detections: (result.detections ?? []).map((detection) => ({
        score: detection.categories[0]?.score ?? 0,
        box: detection.boundingBox
          ? {
              x: detection.boundingBox.originX,
              y: detection.boundingBox.originY,
              width: detection.boundingBox.width,
              height: detection.boundingBox.height,
            }
          : null,
        // Normalized to the image size.
        keypoints: (detection.keypoints ?? []).map((k) => ({ x: k.x, y: k.y })),
      })),
    };
  }

  protected override displayImageResult(result: FaceDetectorResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (!imageCanvas) return;
    const ctx = imageCanvas.getContext('2d')!;
    const testImage = document.getElementById('test-image') as HTMLImageElement;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

    if (result.detections) {
      for (let detection of result.detections) {
        this.drawDetection(ctx, detection, false);
      }
    }
  }

  protected override displayVideoResult(result: FaceDetectorResult) {
    if (!this.canvasElement || !this.video) return;

    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    if (result.detections) {
      for (let detection of result.detections) {
        this.drawDetection(this.canvasCtx, detection, true); // mirror = true
      }
    }
  }

  private drawDetection(ctx: CanvasRenderingContext2D, detection: any, mirror: boolean) {
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawBoundingBox(detection.boundingBox!, {
      color: '#007f8b',
      lineWidth: 4,
      fillColor: 'transparent',
    });

    const { originX, originY } = detection.boundingBox!;
    let x = originX;

    ctx.fillStyle = '#007f8b';
    ctx.font = '16px sans-serif';

    const category = detection.categories[0];
    const score = category.score ? Math.round(category.score * 100) : 0;
    const labelText = `Face - ${score}%`;

    const textWidth = ctx.measureText(labelText).width;

    if (mirror) {
      ctx.save();
      const centerX = x + (textWidth + 10) / 2;
      const centerY = originY + 12.5;

      ctx.translate(centerX, centerY);
      ctx.scale(-1, 1);
      ctx.translate(-centerX, -centerY);

      ctx.fillRect(x, originY, textWidth + 10, 25);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, originY + 18);
      ctx.restore();
    } else {
      ctx.fillRect(x, originY, textWidth + 10, 25);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, originY + 18);
    }
  }
}

let activeTask: FaceDetectorTask | null = null;

export async function setupFaceDetector(container: HTMLElement) {
  activeTask = new FaceDetectorTask({
    container,
    template,
    defaultModelName: 'blaze_face_short_range',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
    defaultDelegate: 'GPU',
    workerFactory: () => new Worker(new URL('../workers/face-detector.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupFaceDetector() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
