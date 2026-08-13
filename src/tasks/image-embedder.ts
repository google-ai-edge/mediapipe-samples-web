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

import { BaseVisionTask } from '../components/base-vision-task';

// @ts-ignore
import template from '../templates/image-embedder.html?raw';
// @ts-ignore

class ImageEmbedderTask extends BaseVisionTask {
  private image1: HTMLImageElement | null = null;
  private image2: HTMLImageElement | null = null;

  protected override onInitializeUI() {
    this.image1 = document.getElementById('image-1') as HTMLImageElement;
    this.image2 = document.getElementById('image-2') as HTMLImageElement;
    const display1 = document.getElementById('display-area-1')!;
    const display2 = document.getElementById('display-area-2')!;

    const setImage = (img: HTMLImageElement, display: HTMLElement, src: string) => {
      img.src = src;
      img.style.display = 'block';
      display.classList.add('has-image');
      const placeholder = display.querySelector('.placeholder-text') as HTMLElement;
      if (placeholder) placeholder.style.display = 'none';
      this.checkEnableButton();
    };

    const setupUpload = (inputElementId: string, img: HTMLImageElement, display: HTMLElement) => {
      const input = document.getElementById(inputElementId) as HTMLInputElement;
      if (!input) return;

      input.addEventListener('change', (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (evt) => {
            setImage(img, display, evt.target?.result as string);
          };
          reader.readAsDataURL(file);
        }
      });

      return () => {
        input.value = '';
        input.click();
      };
    };

    const triggerUpload1 = setupUpload('image-upload-1', this.image1!, display1);
    const triggerUpload2 = setupUpload('image-upload-2', this.image2!, display2);

    // Attach listeners to samples
    document.querySelectorAll('.sample-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = (btn as HTMLElement).dataset.target;
        const src = (btn as HTMLElement).dataset.src;
        if (targetId === '1' && src && this.image1) setImage(this.image1, display1, src);
        if (targetId === '2' && src && this.image2) setImage(this.image2, display2, src);
      });
    });

    // Attach listeners to upload buttons
    document.querySelectorAll('.upload-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = (btn as HTMLElement).dataset.target;
        if (targetId === '1' && triggerUpload1) triggerUpload1();
        if (targetId === '2' && triggerUpload2) triggerUpload2();
      });
    });

    // Make display areas clickable for upload
    if (display1 && triggerUpload1) {
      display1.addEventListener('click', triggerUpload1);
    }
    if (display2 && triggerUpload2) {
      display2.addEventListener('click', triggerUpload2);
    }

    if (this.image1) this.image1.onload = () => this.checkEnableButton();
    if (this.image2) this.image2.onload = () => this.checkEnableButton();

    this.models = {
      mobilenet_v3_small:
        'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
      mobilenet_v3_large:
        'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'MobileNet V3 Small', value: 'mobilenet_v3_small', isDefault: true },
        { label: 'MobileNet V3 Large', value: 'mobilenet_v3_large' },
      ]);
    }
  }

  private checkEnableButton() {
    if (this.image1 && this.image2 && this.image1.src && this.image2.src && this.isWorkerReady) {
      this.computeSimilarity(this.image1, this.image2);
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();

    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) loadingOverlay.style.display = 'none';

    this.checkEnableButton();
  }

  protected override async initializeTask() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
      loadingOverlay.style.display = 'flex';
      const loadingText = loadingOverlay.querySelector('.loading-text');
      if (loadingText) loadingText.textContent = 'Loading Model...';
    }

    await super.initializeTask();
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    if (type === 'EMBED_RESULT') {
      const { similarity, inferenceTime } = event.data;
      this.updateInferenceTime(inferenceTime);
      this.displayResults(similarity);
      this.updateStatus('Done');
    } else {
      super.handleWorkerMessage(event);
    }
  }

  private async computeSimilarity(img1: HTMLImageElement, img2: HTMLImageElement) {
    if (!this.worker || !this.isWorkerReady) return;

    if (!img1 || !img2 || img1.naturalWidth === 0 || img2.naturalWidth === 0) {
      this.updateStatus('Select two images');
      return;
    }

    this.updateStatus('Computing...');

    const bitmap1 = await createImageBitmap(img1);

    const bitmap2 = await createImageBitmap(img2);

    this.worker.postMessage(
      {
        type: 'EMBED',
        image1: bitmap1,
        image2: bitmap2,
      },
      [bitmap1, bitmap2]
    );
  }

  private displayResults(similarity: number) {
    const valueEl = document.getElementById('similarity-value');
    if (valueEl && similarity !== undefined) {
      valueEl.innerText = similarity.toFixed(4);
    } else if (valueEl) {
      valueEl.innerText = '--';
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {};
  }

  protected override displayImageResult() {}
  protected override displayVideoResult() {}
}

let activeTask: ImageEmbedderTask | null = null;

export async function setupImageEmbedder(container: HTMLElement) {
  activeTask = new ImageEmbedderTask({
    container,
    template,
    defaultModelName: 'mobilenet_v3_small',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
    workerFactory: () =>
      new Worker(new URL('../workers/image-embedder.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'CPU',
  });

  await activeTask.initialize();
}

export function cleanupImageEmbedder() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
