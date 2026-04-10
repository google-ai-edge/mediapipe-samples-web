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

// @ts-ignore
import template from '../templates/text-classifier.html?raw';

import { BaseTextTask } from '../components/base-text-task';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';

// @ts-ignore

class TextClassifierTask extends BaseTextTask {
  private classificationResultUI: ClassificationResult | undefined;
  private textInput!: HTMLTextAreaElement;
  private classifyBtn!: HTMLButtonElement;

  private maxResults = 3;

  protected override onInitializeUI() {
    this.classificationResultUI = new ClassificationResult('classification-results');

    this.classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;
    this.textInput = document.getElementById('text-input') as HTMLTextAreaElement;

    // Event Listeners
    if (this.classifyBtn) {
      this.classifyBtn.addEventListener('click', () => {
        if (this.textInput.value.trim()) {
          this.classifyText(this.textInput.value);
        }
      });
    }

    // Sample Buttons
    const sampleBtns = this.container.querySelectorAll('.sample-btn');
    sampleBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const text = (e.currentTarget as HTMLElement).dataset.text;
        if (text) {
          this.textInput.value = text;
          this.classifyText(text);
        }
      });
    });

    const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
    const maxResultsValue = document.getElementById('max-results-value');
    if (maxResultsInput && maxResultsValue) {
      maxResultsInput.addEventListener('input', async (e) => {
        this.maxResults = parseInt((e.target as HTMLInputElement).value);
        maxResultsValue.innerText = this.maxResults.toString();
        await this.initializeTask();
      });
    }

    this.models = {
      bert_classifier:
        'https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite',
      average_word_classifier:
        'https://storage.googleapis.com/mediapipe-models/text_classifier/average_word_classifier/float32/1/average_word_classifier.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'BERT Classifier', value: 'bert_classifier', isDefault: true },
        { label: 'Average Word Classifier', value: 'average_word_classifier' },
      ]);
    }
  }

  protected override async initializeTask() {
    if (this.classifyBtn) {
      this.classifyBtn.disabled = true;
    }
    await super.initializeTask();
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      maxResults: this.maxResults,
    };
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case 'CLASSIFY_RESULT':
        const { result, timestampMs } = event.data;
        const duration = performance.now() - timestampMs;
        this.updateInferenceTime(duration);
        this.displayResults(result);
        if (this.classifyBtn) this.classifyBtn.disabled = false;
        this.updateStatus('Done');
        break;
      case 'ERROR':
        if (this.classifyBtn) {
          this.classifyBtn.disabled = false;
          this.classifyBtn.innerText = 'Retry';
        }
        super.handleWorkerMessage(event);
        break;
      default:
        super.handleWorkerMessage(event);
        break;
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();
    if (this.classifyBtn) {
      this.classifyBtn.disabled = false;
      this.classifyBtn.innerText = 'Classify';
    }
  }

  private classifyText(text: string) {
    if (!this.worker || !this.isWorkerReady) return;

    if (this.classifyBtn) this.classifyBtn.disabled = true;
    this.updateStatus('Classifying...');

    this.worker.postMessage({
      type: 'CLASSIFY',
      text: text,
      timestampMs: performance.now(),
    });
  }

  private displayResults(result: any) {
    if (!result || !result.classifications || result.classifications.length === 0 || !this.classificationResultUI)
      return;

    const categories = result.classifications[0].categories;
    categories.sort((a: any, b: any) => b.score - a.score);

    const items: ClassificationItem[] = categories.map((c: any) => ({
      label: c.categoryName,
      score: c.score,
    }));

    this.classificationResultUI.updateResults(items);
  }
}

let activeTask: TextClassifierTask | null = null;

export async function setupTextClassifier(container: HTMLElement) {
  activeTask = new TextClassifierTask({
    container,
    template,
    defaultModelName: 'bert_classifier',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite',
    workerFactory: () =>
      new Worker(new URL('../workers/text-classifier.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'CPU',
  });

  await activeTask.initialize();
}

export function cleanupTextClassifier() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
