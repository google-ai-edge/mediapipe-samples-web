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

import { GestureRecognizerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';
import { summarizeLandmarks } from '../components/test-hooks';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';

// @ts-ignore
import template from '../templates/gesture-recognizer.html?raw';
// @ts-ignore

class GestureRecognizerTask extends BaseVisionTask {
  private drawingUtils: DrawingUtils | undefined;
  private classificationResultUI: ClassificationResult | undefined;

  private numHands = 2;
  private minHandDetectionConfidence = 0.5;
  private minHandPresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;

  protected override onInitializeUI() {
    this.classificationResultUI = new ClassificationResult('classification-results');

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

    setupSlider('num-hands', (val) => {
      this.numHands = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', numHands: this.numHands });
      this.triggerRedetection();
    });

    setupSlider('min-hand-detection-confidence', (val) => {
      this.minHandDetectionConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: this.minHandDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-hand-presence-confidence', (val) => {
      this.minHandPresenceConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: this.minHandPresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-tracking-confidence', (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    // Custom model options
    this.models = {
      gesture_recognizer:
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: 'Gesture Recognizer', value: 'gesture_recognizer', isDefault: true }]);
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
      numHands: this.numHands,
      minHandDetectionConfidence: this.minHandDetectionConfidence,
      minHandPresenceConfidence: this.minHandPresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
    };
  }

  protected override readonly testTaskName = 'gesture_recognizer';

  protected override summarizeTestResult(result: GestureRecognizerResult) {
    return {
      hands: (result.gestures ?? []).map((gestures, i) => ({
        gesture: gestures[0]?.categoryName ?? '',
        score: gestures[0]?.score ?? 0,
        handedness: result.handedness?.[i]?.[0]?.categoryName ?? '',
        handednessDisplayName: result.handedness?.[i]?.[0]?.displayName ?? '',
        ...summarizeLandmarks(result.landmarks?.[i], { wrist: 0, thumbTip: 4, indexTip: 8 }),
      })),
    };
  }

  protected override displayImageResult(result: GestureRecognizerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;
    imageCanvas.style.width = '100%';
    imageCanvas.style.height = 'auto';

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx);

      for (const landmark of result.landmarks) {
        this.drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        this.drawingUtils.drawLandmarks(landmark, { color: '#FF0000', lineWidth: 2 });
      }
    }

    this.displayGestureText(result);
  }

  protected override displayVideoResult(result: GestureRecognizerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;

    this.canvasCtx.save();
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(this.canvasCtx);
      else this.drawingUtils = new DrawingUtils(this.canvasCtx);

      for (const landmark of result.landmarks) {
        this.drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        this.drawingUtils.drawLandmarks(landmark, { color: '#FF0000', lineWidth: 2 });
      }
    }

    this.canvasCtx.restore();
    this.displayGestureText(result);
  }

  private displayGestureText(result: GestureRecognizerResult) {
    if (!this.classificationResultUI) return;

    if (result.gestures && result.gestures.length > 0) {
      const items: ClassificationItem[] = [];
      result.gestures.forEach((gestures, index) => {
        const handedness =
          result.handedness && result.handedness[index] ? result.handedness[index][0].displayName : `Hand ${index + 1}`;
        const topGesture = gestures[0];
        if (topGesture && topGesture.categoryName !== 'None') {
          items.push({
            label: `${handedness}: ${topGesture.categoryName}`,
            score: topGesture.score,
          });
        }
      });

      // If we filtered out all the 'None' gestures and have a real valid gesture, update.
      if (items.length > 0) {
        this.classificationResultUI.updateResults(items);
      } else {
        // If the only gesture found was 'None', explicitly print 0% 'No results'.
        this.classificationResultUI.updateResults([]);
      }
    } else {
      // If the Mediapipe graph found no hands at all, explicitly print 0% 'No results'.
      this.classificationResultUI.updateResults([]);
    }
  }
}

// Singleton instance
let activeTask: GestureRecognizerTask | null = null;

export async function setupGestureRecognizer(container: HTMLElement) {
  activeTask = new GestureRecognizerTask({
    container,
    template,
    defaultModelName: 'gesture_recognizer',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
    workerFactory: () =>
      new Worker(new URL('../workers/gesture-recognizer.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupGestureRecognizer() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
