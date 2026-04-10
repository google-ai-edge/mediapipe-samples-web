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
import template from '../templates/text-embedder.html?raw';

import { BaseTextTask } from '../components/base-text-task';

// @ts-ignore

class TextEmbedderTask extends BaseTextTask {
  private embedBtn!: HTMLButtonElement;
  private textInput1!: HTMLTextAreaElement;
  private textInput2!: HTMLTextAreaElement;

  protected override onInitializeUI() {
    this.embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
    this.textInput1 = document.getElementById('text-input-1') as HTMLTextAreaElement;
    this.textInput2 = document.getElementById('text-input-2') as HTMLTextAreaElement;

    // Event Listeners
    if (this.embedBtn) {
      this.embedBtn.addEventListener('click', () => {
        if (this.textInput1.value.trim() && this.textInput2.value.trim()) {
          this.computeSimilarity(this.textInput1.value, this.textInput2.value);
        }
      });
    }

    // Sample Buttons (Pairs)
    const sampleBtns = this.container.querySelectorAll('.sample-btn');
    sampleBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const el = e.currentTarget as HTMLElement;
        const t1 = el.dataset.text1;
        const t2 = el.dataset.text2;
        if (t1 && t2) {
          this.textInput1.value = t1;
          this.textInput2.value = t2;
          this.computeSimilarity(t1, t2);
        }
      });
    });

    // Individual Sample Chips
    const sampleChips = this.container.querySelectorAll('.sample-chip');
    sampleChips.forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const el = e.currentTarget as HTMLElement;
        const targetId = el.dataset.target;
        const text = el.dataset.text;
        if (targetId && text) {
          const targetInput = document.getElementById(targetId) as HTMLTextAreaElement;
          if (targetInput) {
            targetInput.value = text;
            // Trigger computation if both have values
            if (this.textInput1.value.trim() && this.textInput2.value.trim()) {
              this.computeSimilarity(this.textInput1.value, this.textInput2.value);
            }
          }
        }
      });
    });

    this.models = {
      universal_sentence_encoder:
        'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'Universal Sentence Encoder', value: 'universal_sentence_encoder', isDefault: true },
      ]);
    }
  }

  protected override async initializeTask() {
    if (this.embedBtn) {
      this.embedBtn.disabled = true;
    }
    await super.initializeTask();
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {};
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case 'EMBED_RESULT':
        const { similarity, timestampMs } = event.data;
        const duration = performance.now() - timestampMs;
        this.updateInferenceTime(duration);
        this.displayResults(similarity);
        if (this.embedBtn) this.embedBtn.disabled = false;
        this.updateStatus('Done');
        break;
      case 'ERROR':
        if (this.embedBtn) {
          this.embedBtn.disabled = false;
          this.embedBtn.innerText = 'Retry';
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
    if (this.embedBtn) {
      this.embedBtn.disabled = false;
      this.embedBtn.innerText = 'Compute Similarity';
    }
  }

  private computeSimilarity(text1: string, text2: string) {
    if (!this.worker || !this.isWorkerReady) return;

    if (this.embedBtn) this.embedBtn.disabled = true;
    this.updateStatus('Computing...');

    this.worker.postMessage({
      type: 'EMBED',
      text1: text1,
      text2: text2,
      timestampMs: performance.now(),
    });
  }

  private displayResults(similarity: number) {
    const container = document.getElementById('embedding-results');
    const valueEl = document.getElementById('similarity-value');

    if (container && valueEl) {
      container.style.display = 'block';
      valueEl.innerText = similarity.toFixed(4);
    }
  }
}

let activeTask: TextEmbedderTask | null = null;

export async function setupTextEmbedder(container: HTMLElement) {
  activeTask = new TextEmbedderTask({
    container,
    template,
    defaultModelName: 'universal_sentence_encoder',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite',
    workerFactory: () => new Worker(new URL('../workers/text-embedder.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'CPU',
  });

  await activeTask.initialize();
}

export function cleanupTextEmbedder() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
