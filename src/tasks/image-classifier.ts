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

import { ImageClassifierResult } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';

// @ts-ignore
import template from '../templates/image-classifier.html?raw';
// @ts-ignore

class ImageClassifierTask extends BaseVisionTask {
  private classificationResultUI: ClassificationResult | undefined;

  private maxResults = 3;
  private scoreThreshold = 0.5;

  protected override onInitializeUI() {
    this.classificationResultUI = new ClassificationResult('classification-results');

    const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
    const maxResultsValue = document.getElementById('max-results-value')!;
    if (maxResultsInput && maxResultsValue) {
      maxResultsInput.addEventListener('input', () => {
        this.maxResults = parseInt(maxResultsInput.value);
        maxResultsValue.innerText = this.maxResults.toString();
        this.worker?.postMessage({ type: 'SET_OPTIONS', maxResults: this.maxResults });
        this.triggerRedetection();
      });
    }

    const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
    const scoreThresholdValue = document.getElementById('score-threshold-value')!;
    if (scoreThresholdInput && scoreThresholdValue) {
      scoreThresholdInput.addEventListener('input', () => {
        this.scoreThreshold = parseInt(scoreThresholdInput.value) / 100;
        scoreThresholdValue.innerText = `${parseInt(scoreThresholdInput.value)}%`;
        this.worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold: this.scoreThreshold });
        this.triggerRedetection();
      });
    }

    // Custom model options
    this.models = {
      efficientnet_lite0:
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
      efficientnet_lite2:
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'EfficientNet-Lite0', value: 'efficientnet_lite0', isDefault: true },
        { label: 'EfficientNet-Lite2', value: 'efficientnet_lite2' },
      ]);
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
      maxResults: this.maxResults,
      scoreThreshold: this.scoreThreshold,
    };
  }

  protected override displayImageResult(result: ImageClassifierResult) {
    this.displayResult(result);
  }

  protected override displayVideoResult(result: ImageClassifierResult) {
    this.displayResult(result);
  }

  private displayResult(result: ImageClassifierResult) {
    if (!this.classificationResultUI) return;

    if (result.classifications && result.classifications.length > 0) {
      const categories = result.classifications[0].categories;
      const items: ClassificationItem[] = categories.map((c) => ({
        label: c.categoryName,
        score: c.score,
      }));
      this.classificationResultUI.updateResults(items);
    } else {
      this.classificationResultUI.updateResults([]);
    }
  }
}

// Singleton instance
let activeTask: ImageClassifierTask | null = null;

export async function setupImageClassifier(container: HTMLElement) {
  activeTask = new ImageClassifierTask({
    container,
    template,
    defaultModelName: 'efficientnet_lite0',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
    models: {
      efficientnet_lite2:
        'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite',
    },
    workerFactory: () =>
      new Worker(new URL('../workers/image-classifier.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'GPU', // Image classifier defaults to GPU in original code
  });

  await activeTask.initialize();
}

export function cleanupImageClassifier() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
