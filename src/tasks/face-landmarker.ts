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

import { FaceLandmarkerResult, DrawingUtils, FaceLandmarker } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';
import { summarizeLandmarks } from '../components/test-hooks';

// @ts-ignore
import template from '../templates/face-landmarker.html?raw';
// @ts-ignore

class FaceLandmarkerTask extends BaseVisionTask {
  private drawingUtils: DrawingUtils | undefined;

  private numFaces = 1;
  private minFaceDetectionConfidence = 0.5;
  private minFacePresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;

  protected override onInitializeUI() {
    // Confidence Sliders
    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener('input', () => {
          const val = parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider('min-face-detection-confidence', (val) => {
      this.minFaceDetectionConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minFaceDetectionConfidence: this.minFaceDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-face-presence-confidence', (val) => {
      this.minFacePresenceConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minFacePresenceConfidence: this.minFacePresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-tracking-confidence', (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    setupSlider('num-faces', (val) => {
      this.numFaces = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', numFaces: this.numFaces });
      this.triggerRedetection();
    });

    // Custom model options for Face Landmarker
    this.models = {
      face_landmarker:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: 'Face Landmarker', value: 'face_landmarker', isDefault: true }]);
    }
  }

  private triggerRedetection() {
    if (this.runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src) {
        this.detectImage(testImage);
      }
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      numFaces: this.numFaces,
      minFaceDetectionConfidence: this.minFaceDetectionConfidence,
      minFacePresenceConfidence: this.minFacePresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    };
  }

  protected override readonly testTaskName = 'face_landmarker';

  protected override summarizeTestResult(result: FaceLandmarkerResult) {
    return {
      // Face mesh indices: 1 nose tip, 152 chin, 10 forehead, 33 / 263 outer eye corners.
      faces: (result.faceLandmarks ?? []).map((landmarks) =>
        summarizeLandmarks(landmarks, { noseTip: 1, chin: 152, forehead: 10, rightEyeOuter: 33, leftEyeOuter: 263 })
      ),
      blendshapeCounts: (result.faceBlendshapes ?? []).map((b) => b.categories.length),
      matrixCount: (result.facialTransformationMatrixes ?? []).length,
    };
  }

  protected override displayImageResult(result: FaceLandmarkerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.beginPath();
    ctx.rect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.clip();

    if (result.faceLandmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx);

      for (const landmarks of result.faceLandmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  protected override displayVideoResult(result: FaceLandmarkerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    this.canvasCtx.beginPath();
    this.canvasCtx.rect(0, 0, this.canvasElement.width, this.canvasElement.height);
    this.canvasCtx.clip();

    if (result.faceLandmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(this.canvasCtx);
      else this.drawingUtils = new DrawingUtils(this.canvasCtx);

      for (const landmarks of result.faceLandmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  private drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: '#C0C0C070',
      lineWidth: 1,
    });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF3030' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#FF3030' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#30FF30' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#30FF30' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#E0E0E0' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#E0E0E0' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: '#FF3030' });
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: '#30FF30' });
  }
}

// Singleton instance to support modular cleanup
let activeTask: FaceLandmarkerTask | null = null;

export async function setupFaceLandmarker(container: HTMLElement) {
  activeTask = new FaceLandmarkerTask({
    container,
    template,
    defaultModelName: 'face_landmarker',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    workerFactory: () =>
      new Worker(new URL('../workers/face-landmarker.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupFaceLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
