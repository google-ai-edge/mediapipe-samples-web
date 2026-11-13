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

import {
  HolisticLandmarkerResult,
  DrawingUtils,
  FaceLandmarker,
  PoseLandmarker,
  HandLandmarker,
} from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';
import { summarizeLandmarks } from '../components/test-hooks';

// @ts-ignore
import template from '../templates/holistic-landmarker.html?raw';
// @ts-ignore

class HolisticLandmarkerTask extends BaseVisionTask {
  protected override getWorkerInitParams(): Record<string, any> {
    return {};
  }

  protected override readonly testTaskName = 'holistic_landmarker';

  protected override summarizeTestResult(result: HolisticLandmarkerResult) {
    return {
      face: summarizeLandmarks(result.faceLandmarks?.[0], { noseTip: 1, chin: 152, forehead: 10 }),
      pose: summarizeLandmarks(result.poseLandmarks?.[0], {
        nose: 0,
        leftShoulder: 11,
        rightShoulder: 12,
        leftHip: 23,
        rightHip: 24,
        leftAnkle: 27,
        rightAnkle: 28,
      }),
      leftHand: summarizeLandmarks(result.leftHandLandmarks?.[0], { wrist: 0 }),
      rightHand: summarizeLandmarks(result.rightHandLandmarks?.[0], { wrist: 0 }),
    };
  }

  protected override displayImageResult(result: HolisticLandmarkerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (!imageCanvas) return;
    const ctx = imageCanvas.getContext('2d')!;
    const testImage = document.getElementById('test-image') as HTMLImageElement;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

    this.drawResults(ctx, result);
  }

  protected override displayVideoResult(result: HolisticLandmarkerResult) {
    if (!this.canvasElement || !this.video) return;

    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.save();
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    this.drawResults(this.canvasCtx, result);
    this.canvasCtx.restore();
  }

  private drawResults(ctx: CanvasRenderingContext2D, result: HolisticLandmarkerResult) {
    const drawingUtils = new DrawingUtils(ctx);

    // Face Landmarks
    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
      for (const landmarks of result.faceLandmarks) {
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
          color: '#C0C0C070',
          lineWidth: 1,
        });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF3030' });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#FF3030' });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#30FF30' });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#30FF30' });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#E0E0E0' });
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#E0E0E0' });
      }
    }

    // Pose Landmarks
    if (result.poseLandmarks && result.poseLandmarks.length > 0) {
      for (const landmarks of result.poseLandmarks) {
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#FFFFFF' });
        drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 1 });
      }
    }

    // Hand Landmarks
    if (result.leftHandLandmarks && result.leftHandLandmarks.length > 0) {
      for (const landmarks of result.leftHandLandmarks) {
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 5 });
        drawingUtils.drawLandmarks(landmarks, { color: '#00FF00', lineWidth: 2 });
      }
    }

    if (result.rightHandLandmarks && result.rightHandLandmarks.length > 0) {
      for (const landmarks of result.rightHandLandmarks) {
        drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 5 });
        drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', lineWidth: 2 });
      }
    }
  }
}

let activeTask: HolisticLandmarkerTask | null = null;

export async function setupHolisticLandmarker(container: HTMLElement) {
  activeTask = new HolisticLandmarkerTask({
    container,
    template,
    defaultModelName: 'holistic_landmarker_lite',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task',
    defaultDelegate: 'GPU',
    workerFactory: () =>
      new Worker(new URL('../workers/holistic-landmarker.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupHolisticLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
