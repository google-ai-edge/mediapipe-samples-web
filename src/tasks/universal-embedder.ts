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
import template from '../templates/universal-embedder.html?raw';
// @ts-ignore
import { UniversalEmbedder } from '@mediapipe/tasks-retrieval';
import {
  mountModelSelector,
  retrievalRuntime,
  SAMPLE_IMAGES,
  type SampleImageItem,
} from '../components/retrieval-runtime';
import { InferenceTimer } from '../components/inference-timer';
import type { ModelSelector } from '../components/model-selector';

function formatSigned(val: number): string {
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(3)}`;
}

const PREVIEW_MAX_ELEMENTS = 32;

function renderVectorPreviewRow(slotName: string, floatEmbedding: number[]): string {
  const head = floatEmbedding.slice(0, PREVIEW_MAX_ELEMENTS).map(formatSigned).join(', ');
  const tail = floatEmbedding.length > PREVIEW_MAX_ELEMENTS ? ', …' : '';
  return (
    `<div class="vector-preview-row">` +
    `<span class="vector-preview-values">${slotName}: [${head}${tail}</span>` +
    `<span class="vector-preview-suffix">] ${floatEmbedding.length}d</span>` +
    `</div>`
  );
}

interface CustomImageSlot {
  label: string;
  bytes: Uint8Array;
  objectUrl: string;
}

class UniversalEmbedderTask {
  private container: HTMLElement;
  private modeA: 'text' | 'image' = 'text';
  private modeB: 'text' | 'image' = 'image';
  private sampleA: SampleImageItem = SAMPLE_IMAGES[0];
  private sampleB: SampleImageItem = SAMPLE_IMAGES[1];
  private customImageA: CustomImageSlot | null = null;
  private customImageB: CustomImageSlot | null = null;
  private isComparing = false;
  private pendingCompare = false;
  private timer = new InferenceTimer({
    mode: 'dual',
    labelA: 'A',
    labelB: 'B',
  });
  private modelSelector: ModelSelector | null = null;

  private l2Select!: HTMLSelectElement;

  private modeTextA!: HTMLButtonElement;
  private modeImageA!: HTMLButtonElement;
  private panelTextA!: HTMLElement;
  private panelImageA!: HTMLElement;
  private inputTextA!: HTMLTextAreaElement;
  private previewImgA!: HTMLImageElement;
  private customInputA!: HTMLInputElement;
  private galleryStripA!: HTMLElement;
  private metaA!: HTMLElement;

  private modeTextB!: HTMLButtonElement;
  private modeImageB!: HTMLButtonElement;
  private panelTextB!: HTMLElement;
  private panelImageB!: HTMLElement;
  private inputTextB!: HTMLTextAreaElement;
  private previewImgB!: HTMLImageElement;
  private customInputB!: HTMLInputElement;
  private galleryStripB!: HTMLElement;
  private metaB!: HTMLElement;

  private embedBtn!: HTMLButtonElement;
  private embedBtnLabel!: HTMLElement;
  private resultsCard!: HTMLElement;
  private similarityValueEl!: HTMLElement;
  private similarityBarFill!: HTMLElement;
  private vectorPreviewEl!: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async initialize() {
    this.container.innerHTML = template;

    this.bindElements();
    this.timer.mount();
    this.renderImageGalleries();
    this.syncSlotUI('A');
    this.syncSlotUI('B');
    this.bindEvents();

    this.modelSelector = mountModelSelector('model-selector-container', {
      getL2Normalize: () => this.l2Select.value === 'true',
      onLoadStart: (msg) => {
        this.setStatus('busy', msg);
        this.updateControlsState();
      },
      onModelReady: async () => {
        this.setStatus('ready', 'Ready · Click "Compute Similarity"');
        this.updateControlsState();
      },
      onError: (err) => {
        console.error(err);
        this.setStatus('error', err?.message || String(err));
        this.updateControlsState();
      },
    });

    if (retrievalRuntime.isReady()) {
      this.l2Select.value = String(retrievalRuntime.activeL2Normalize);
      this.setStatus('ready', 'Ready · Click "Compute Similarity"');
    } else if (!(navigator as any).gpu) {
      this.setStatus('error', 'WebGPU is not available in this browser.');
    } else {
      this.setStatus('idle', 'Select a .litertlm model on the left to begin');
    }

    this.updateControlsState();
  }

  private bindElements() {
    this.l2Select = document.getElementById('l2-normalize-select') as HTMLSelectElement;

    this.modeTextA = document.getElementById('mode-text-a') as HTMLButtonElement;
    this.modeImageA = document.getElementById('mode-image-a') as HTMLButtonElement;
    this.panelTextA = document.getElementById('panel-text-a')!;
    this.panelImageA = document.getElementById('panel-image-a')!;
    this.inputTextA = document.getElementById('input-text-a') as HTMLTextAreaElement;
    this.previewImgA = document.getElementById('preview-img-a') as HTMLImageElement;
    this.customInputA = document.getElementById('custom-image-a') as HTMLInputElement;
    this.galleryStripA = document.getElementById('gallery-strip-a')!;
    this.metaA = document.getElementById('meta-a')!;

    this.modeTextB = document.getElementById('mode-text-b') as HTMLButtonElement;
    this.modeImageB = document.getElementById('mode-image-b') as HTMLButtonElement;
    this.panelTextB = document.getElementById('panel-text-b')!;
    this.panelImageB = document.getElementById('panel-image-b')!;
    this.inputTextB = document.getElementById('input-text-b') as HTMLTextAreaElement;
    this.previewImgB = document.getElementById('preview-img-b') as HTMLImageElement;
    this.customInputB = document.getElementById('custom-image-b') as HTMLInputElement;
    this.galleryStripB = document.getElementById('gallery-strip-b')!;
    this.metaB = document.getElementById('meta-b')!;

    this.embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
    this.embedBtnLabel = document.getElementById('embed-btn-label')!;
    this.resultsCard = document.getElementById('embedding-results')!;
    this.similarityValueEl = document.getElementById('similarity-value')!;
    this.similarityBarFill = document.getElementById('similarity-bar-fill')!;
    this.vectorPreviewEl = document.getElementById('vector-preview')!;
  }

  private renderImageGalleries() {
    const buildGallery = (slot: 'A' | 'B', container: HTMLElement, activeSampleId: string, hasCustom: boolean) => {
      container.innerHTML = '';
      SAMPLE_IMAGES.forEach((sample) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `sample-thumb-btn${!hasCustom && sample.id === activeSampleId ? ' selected' : ''}`;
        btn.title = sample.label;
        const imgUrl = retrievalRuntime.getImageUrl(sample.fileName);
        btn.innerHTML = `
          <img src="${imgUrl}" alt="${sample.label}" loading="lazy" />
        `;
        btn.addEventListener('click', () => {
          if (slot === 'A') {
            this.sampleA = sample;
            this.customImageA = null;
          } else {
            this.sampleB = sample;
            this.customImageB = null;
          }
          this.syncSlotUI(slot);
          this.invalidateResult();
          this.triggerAutoCompare();
        });
        container.appendChild(btn);
      });
    };

    buildGallery('A', this.galleryStripA, this.sampleA.id, Boolean(this.customImageA));
    buildGallery('B', this.galleryStripB, this.sampleB.id, Boolean(this.customImageB));
  }

  private syncSlotUI(slot: 'A' | 'B') {
    if (slot === 'A') {
      const isText = this.modeA === 'text';
      this.modeTextA.classList.toggle('active', isText);
      this.modeImageA.classList.toggle('active', !isText);
      this.panelTextA.style.display = isText ? 'block' : 'none';
      this.panelImageA.style.display = isText ? 'none' : 'flex';

      if (this.customImageA) {
        this.previewImgA.src = this.customImageA.objectUrl;
        this.previewImgA.title = this.customImageA.label;
      } else {
        this.previewImgA.src = retrievalRuntime.getImageUrl(this.sampleA.fileName);
        this.previewImgA.title = this.sampleA.label;
      }
    } else {
      const isText = this.modeB === 'text';
      this.modeTextB.classList.toggle('active', isText);
      this.modeImageB.classList.toggle('active', !isText);
      this.panelTextB.style.display = isText ? 'block' : 'none';
      this.panelImageB.style.display = isText ? 'none' : 'flex';

      if (this.customImageB) {
        this.previewImgB.src = this.customImageB.objectUrl;
        this.previewImgB.title = this.customImageB.label;
      } else {
        this.previewImgB.src = retrievalRuntime.getImageUrl(this.sampleB.fileName);
        this.previewImgB.title = this.sampleB.label;
      }
    }
    this.renderImageGalleries();
  }

  private bindEvents() {
    this.modeTextA.addEventListener('click', () => {
      this.modeA = 'text';
      this.syncSlotUI('A');
      this.invalidateResult();
      this.triggerAutoCompare();
    });
    this.modeImageA.addEventListener('click', () => {
      this.modeA = 'image';
      this.syncSlotUI('A');
      this.invalidateResult();
      this.triggerAutoCompare();
    });
    this.modeTextB.addEventListener('click', () => {
      this.modeB = 'text';
      this.syncSlotUI('B');
      this.invalidateResult();
      this.triggerAutoCompare();
    });
    this.modeImageB.addEventListener('click', () => {
      this.modeB = 'image';
      this.syncSlotUI('B');
      this.invalidateResult();
      this.triggerAutoCompare();
    });

    this.inputTextA.addEventListener('input', () => this.invalidateResult());
    this.inputTextB.addEventListener('input', () => this.invalidateResult());

    this.container.querySelectorAll('.slot-text-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const el = chip as HTMLElement;
        const slot = el.dataset.slot;
        const text = el.dataset.text || '';
        if (slot === 'A') {
          this.inputTextA.value = text;
        } else {
          this.inputTextB.value = text;
        }
        this.invalidateResult();
        this.triggerAutoCompare();
      });
    });

    this.container.querySelectorAll('.preset-pair-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const el = btn as HTMLElement;
        this.modeA = (el.dataset.modeA as 'text' | 'image') || 'text';
        this.modeB = (el.dataset.modeB as 'text' | 'image') || 'image';

        if (el.dataset.textA) {
          this.inputTextA.value = el.dataset.textA;
        }
        if (el.dataset.imageA) {
          const foundA = SAMPLE_IMAGES.find((s) => s.id === el.dataset.imageA);
          if (foundA) {
            this.sampleA = foundA;
            this.customImageA = null;
          }
        }
        if (el.dataset.textB) {
          this.inputTextB.value = el.dataset.textB;
        }
        if (el.dataset.imageB) {
          const foundB = SAMPLE_IMAGES.find((s) => s.id === el.dataset.imageB);
          if (foundB) {
            this.sampleB = foundB;
            this.customImageB = null;
          }
        }

        this.syncSlotUI('A');
        this.syncSlotUI('B');
        this.invalidateResult();
        this.triggerAutoCompare();
      });
    });

    this.customInputA.addEventListener('change', async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      const objectUrl = URL.createObjectURL(file);
      this.customImageA = { label: file.name, bytes, objectUrl };
      this.syncSlotUI('A');
      this.invalidateResult();
      this.triggerAutoCompare();
    });

    this.customInputB.addEventListener('change', async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      const objectUrl = URL.createObjectURL(file);
      this.customImageB = { label: file.name, bytes, objectUrl };
      this.syncSlotUI('B');
      this.invalidateResult();
      this.triggerAutoCompare();
    });

    this.embedBtn.addEventListener('click', () => {
      this.compareEmbeddings();
    });

    this.l2Select.addEventListener('change', async () => {
      const l2 = this.l2Select.value === 'true';
      if (retrievalRuntime.activeSource) {
        this.setStatus('busy', `Re-initializing model (l2Normalize=${l2})...`);
        this.modelSelector?.setBusy(true);
        this.updateControlsState();
        try {
          // Standard models are re-read from the local model cache, so this is quick.
          await retrievalRuntime.reinitialize(l2, (loaded, total) => this.modelSelector?.showProgress(loaded, total));
          this.setStatus('ready', 'Ready · Click "Compute Similarity"');
          this.invalidateResult();
        } catch (err: any) {
          this.setStatus('error', err?.message || String(err));
        }
        this.modelSelector?.hideProgress();
        this.modelSelector?.setBusy(false);
        this.updateControlsState();
      }
    });
  }

  private invalidateResult() {
    if (this.resultsCard) {
      this.resultsCard.style.display = 'none';
    }
    if (this.metaA) this.metaA.textContent = '';
    if (this.metaB) this.metaB.textContent = '';
  }

  private canAutoCompare(): boolean {
    if (!retrievalRuntime.isReady()) return false;
    if (this.modeA === 'text' && !this.inputTextA.value.trim()) return false;
    if (this.modeB === 'text' && !this.inputTextB.value.trim()) return false;
    return true;
  }

  private triggerAutoCompare() {
    if (this.canAutoCompare()) {
      this.compareEmbeddings();
    }
  }

  private setStatus(state: 'idle' | 'ready' | 'busy' | 'error', message: string) {
    this.timer.setStatus(state, message);
  }

  private updateControlsState() {
    const ready = retrievalRuntime.isReady() && !this.isComparing;
    this.embedBtn.disabled = !ready;
    this.embedBtnLabel.textContent = this.isComparing ? 'Computing…' : 'Compute Similarity';
  }

  private async embedSlot(slot: 'A' | 'B') {
    const embedder = retrievalRuntime.embedder;
    const isSlotA = slot === 'A';
    const mode = isSlotA ? this.modeA : this.modeB;

    if (mode === 'text') {
      const inputEl = isSlotA ? this.inputTextA : this.inputTextB;
      const text = inputEl.value.trim();
      if (!text) {
        throw new Error(`Input ${slot} text cannot be empty.`);
      }
      const t0 = performance.now();
      const result = await embedder.embedText(text);
      const elapsedMs = performance.now() - t0;
      const shortText = text.length > 28 ? `${text.slice(0, 28)}…` : text;
      return {
        embedding: result.embeddings[0],
        label: 'text',
        summaryLabel: `"${shortText}"`,
        elapsedMs,
      };
    } else {
      const customImg = isSlotA ? this.customImageA : this.customImageB;
      const sample = isSlotA ? this.sampleA : this.sampleB;
      const imageBytes = customImg ? customImg.bytes : await retrievalRuntime.fetchSampleImageBytes(sample);
      const imageId = customImg ? customImg.label : sample.id;

      const t0 = performance.now();
      const result = await embedder.embedImage(imageBytes);
      const elapsedMs = performance.now() - t0;
      return {
        embedding: result.embeddings[0],
        label: imageId,
        summaryLabel: `[Image: ${customImg ? customImg.label : sample.label}]`,
        elapsedMs,
      };
    }
  }

  private async compareEmbeddings() {
    if (!retrievalRuntime.isReady()) return;
    if (this.isComparing) {
      this.pendingCompare = true;
      return;
    }

    this.isComparing = true;
    this.pendingCompare = false;
    this.updateControlsState();
    this.setStatus('busy', 'Computing…');

    try {
      const resA = await this.embedSlot('A');
      const resB = await this.embedSlot('B');

      const dimA = resA.embedding.floatEmbedding.length;
      const dimB = resB.embedding.floatEmbedding.length;

      this.metaA.textContent = `${resA.label} · ${dimA}d · ${Math.round(resA.elapsedMs)}ms`;
      this.metaB.textContent = `${resB.label} · ${dimB}d · ${Math.round(resB.elapsedMs)}ms`;

      const similarity = UniversalEmbedder.cosineSimilarity(resA.embedding, resB.embedding);
      const pct = Math.max(0, Math.min(100, Math.round(((similarity + 1) / 2) * 100)));

      this.similarityValueEl.textContent = similarity.toFixed(4);
      this.similarityBarFill.style.width = `${pct}%`;
      this.vectorPreviewEl.innerHTML =
        renderVectorPreviewRow('A', resA.embedding.floatEmbedding) +
        renderVectorPreviewRow('B', resB.embedding.floatEmbedding);

      this.resultsCard.style.display = 'block';
      this.timer.recordDual(resA.elapsedMs, resB.elapsedMs);
      this.setStatus('ready', 'Done · Cosine similarity computed');
    } catch (err: any) {
      console.error(err);
      this.setStatus('error', err?.message || String(err));
    } finally {
      this.isComparing = false;
      this.updateControlsState();
      if (this.pendingCompare) {
        this.pendingCompare = false;
        this.compareEmbeddings();
      }
    }
  }

  cleanup() {
    this.pendingCompare = false;
    this.timer.cleanup();
  }
}

let activeTask: UniversalEmbedderTask | null = null;

export async function setupUniversalEmbedder(container: HTMLElement) {
  activeTask = new UniversalEmbedderTask(container);
  await activeTask.initialize();
}

export function cleanupUniversalEmbedder() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
