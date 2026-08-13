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
import template from '../templates/language-detector.html?raw';

import { BaseTextTask } from '../components/base-text-task';

// @ts-ignore

class LanguageDetectorTask extends BaseTextTask {
  private textInput!: HTMLTextAreaElement;
  private detectBtn!: HTMLButtonElement;

  private maxResults = 3;
  private scoreThreshold = 0.0;

  protected override onInitializeUI() {
    this.detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;
    this.textInput = document.getElementById('text-input') as HTMLTextAreaElement;

    // Event Listeners
    if (this.detectBtn) {
      this.detectBtn.addEventListener('click', () => {
        if (this.textInput.value.trim()) {
          this.detectLanguage(this.textInput.value);
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
          this.detectLanguage(text);
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

    const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
    const scoreThresholdValue = document.getElementById('score-threshold-value');
    if (scoreThresholdInput && scoreThresholdValue) {
      scoreThresholdInput.addEventListener('input', async (e) => {
        this.scoreThreshold = parseFloat((e.target as HTMLInputElement).value);
        scoreThresholdValue.innerText = this.scoreThreshold.toString();
        await this.initializeTask();
      });
    }

    this.models = {
      language_detector:
        'https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: 'Language Detector', value: 'language_detector', isDefault: true }]);
    }
  }

  protected override async initializeTask() {
    if (this.detectBtn) {
      this.detectBtn.disabled = true;
    }
    await super.initializeTask();
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      maxResults: this.maxResults,
      scoreThreshold: this.scoreThreshold,
    };
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case 'DETECT_RESULT':
        const { result, inferenceTime } = event.data;
        this.updateInferenceTime(inferenceTime);
        this.displayResults(result);
        if (this.detectBtn) this.detectBtn.disabled = false;
        this.updateStatus('Done');
        break;
      case 'ERROR':
        if (this.detectBtn) {
          this.detectBtn.disabled = false;
          this.detectBtn.innerText = 'Retry';
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
    if (this.detectBtn) {
      this.detectBtn.disabled = false;
      this.detectBtn.innerText = 'Detect Language';
    }
  }

  private detectLanguage(text: string) {
    if (!this.worker || !this.isWorkerReady) return;

    if (this.detectBtn) this.detectBtn.disabled = true;
    this.updateStatus('Detecting...');

    this.worker.postMessage({
      type: 'DETECT',
      text: text,
    });
  }

  private displayResults(result: any) {
    const container = document.getElementById('detection-results');
    if (!container || !result.languages || result.languages.length === 0) return;

    container.innerHTML = '';
    const languages = result.languages;

    languages.sort((a: any, b: any) => b.probability - a.probability);

    languages.forEach((lang: any) => {
      const item = document.createElement('div');
      item.className = 'classification-item';

      const scorePct = Math.round(lang.probability * 100);

      item.innerHTML = `
        <div class="class-name">${lang.languageCode}</div>
        <div class="class-bar-container">
          <div class="class-bar" style="width: ${scorePct}%"></div>
        </div>
        <div class="class-score">${scorePct}%</div>
      `;
      container.appendChild(item);
    });
  }
}

let activeTask: LanguageDetectorTask | null = null;

export async function setupLanguageDetector(container: HTMLElement) {
  activeTask = new LanguageDetectorTask({
    container,
    template,
    defaultModelName: 'language_detector',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite',
    workerFactory: () =>
      new Worker(new URL('../workers/language-detector.worker', import.meta.url), { type: 'module' }),
    defaultDelegate: 'CPU',
  });

  await activeTask.initialize();
}

export function cleanupLanguageDetector() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
