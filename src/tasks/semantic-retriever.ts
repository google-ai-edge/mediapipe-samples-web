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
import {
  mountModelSelector,
  retrievalRuntime,
  SAMPLE_IMAGES,
  type SampleImageItem,
} from '../components/retrieval-runtime';
import { InferenceTimer } from '../components/inference-timer';
import { publishTestResult } from '../components/test-hooks';
import template from '../templates/semantic-retriever.html?raw';

function capitalizeFirst(str: string): string {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

interface CorpusItem extends SampleImageItem {
  isCustom: boolean;
  indexed: boolean;
  bytes?: Uint8Array;
  objectUrl?: string;
}

class SemanticRetrieverTask {
  private container: HTMLElement;
  private retriever: any = null;
  private isIndexing = false;
  private isSearching = false;
  private isIndexed = false;
  private corpusItems: CorpusItem[] = SAMPLE_IMAGES.map((item) => ({
    ...item,
    isCustom: false,
    indexed: false,
  }));

  private timer = new InferenceTimer({
    mode: 'single',
    labelA: 'Query',
  });

  private btnIndex!: HTMLButtonElement;
  private btnIndexLabel!: HTMLElement;
  private addCustomImageInput!: HTMLInputElement;
  private galleryGrid!: HTMLElement;
  private indexProgressWrap!: HTMLElement;
  private indexProgressBar!: HTMLElement;

  private queryInput!: HTMLInputElement;
  private btnSearch!: HTMLButtonElement;
  private searchSummaryMeta!: HTMLElement;
  private resultsList!: HTMLElement;
  private suggestionChips: HTMLButtonElement[] = [];

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async initialize() {
    this.container.innerHTML = template;

    this.bindElements();
    this.timer.mount();
    this.renderGalleryStrip();
    this.bindEvents();

    mountModelSelector('model-selector-container', {
      getL2Normalize: () => true,
      onLoadStart: (msg) => {
        this.setStatus('busy', msg);
        this.isIndexed = false;
        this.corpusItems.forEach((item) => {
          item.indexed = false;
        });
        this.renderGalleryStrip();
        this.updateControlsState();
      },
      onModelReady: async () => {
        await this.setupRetrieverInstance();
      },
      onError: (err) => {
        console.error(err);
        this.setStatus('error', err?.message || String(err));
        this.updateControlsState();
      },
    });

    if (retrievalRuntime.isReady()) {
      await this.setupRetrieverInstance();
    } else if (!(navigator as any).gpu) {
      this.setStatus('error', 'WebGPU is not available in this browser.');
    } else {
      this.setStatus('idle', 'Select a .litertlm model on the left to begin');
    }

    this.updateControlsState();
  }

  private async setupRetrieverInstance() {
    try {
      if (this.retriever) {
        try {
          this.retriever.close();
        } catch {
          // Ignore close errors
        }
      }
      this.retriever = await retrievalRuntime.createSemanticRetriever(/*chunkSize*/ 512, /*chunkOverlap*/ 100);
      this.isIndexed = false;
      this.setStatus('ready', 'Model ready · Click "Index 10 sample images"');
      this.updateControlsState();
    } catch (err: any) {
      console.error(err);
      this.setStatus('error', err?.message || String(err));
    }
  }

  private bindElements() {
    this.btnIndex = document.getElementById('btn-index') as HTMLButtonElement;
    this.btnIndexLabel = document.getElementById('btn-index-label')!;
    this.addCustomImageInput = document.getElementById('add-custom-image-input') as HTMLInputElement;
    this.galleryGrid = document.getElementById('index-gallery-grid')!;
    this.indexProgressWrap = document.getElementById('index-progress-wrap')!;
    this.indexProgressBar = document.getElementById('index-progress-bar')!;

    this.queryInput = document.getElementById('query-input') as HTMLInputElement;
    this.btnSearch = document.getElementById('btn-search') as HTMLButtonElement;
    this.searchSummaryMeta = document.getElementById('search-summary-meta')!;
    this.resultsList = document.getElementById('results-list')!;
    this.suggestionChips = Array.from(this.container.querySelectorAll('.suggestion-chip'));
  }

  private renderGalleryStrip(activeIndex = -1) {
    this.galleryGrid.innerHTML = '';
    this.corpusItems.forEach((item, idx) => {
      const card = document.createElement('div');
      let cls = 'index-gallery-item';
      if (idx === activeIndex) cls += ' embedding-active';
      else if (item.indexed) cls += ' indexed';
      card.className = cls;
      const imgUrl = item.isCustom ? item.objectUrl! : retrievalRuntime.getImageUrl(item.fileName);
      card.innerHTML = `
        <img src="${imgUrl}" alt="${item.label}" loading="lazy" />
        <div class="index-badge" title="Indexed in VectorStore">
          <span class="material-icons">check</span>
        </div>
      `;
      this.galleryGrid.appendChild(card);
    });
  }

  private bindEvents() {
    this.btnIndex.addEventListener('click', () => {
      this.indexSampleImages();
    });

    this.addCustomImageInput.addEventListener('change', async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      const objectUrl = URL.createObjectURL(file);
      const cleanName = file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
      const id = `custom_${Date.now()}_${cleanName.replace(/\s+/g, '_')}`;
      const newItem: CorpusItem = {
        id,
        label: cleanName,
        fileName: file.name,
        isCustom: true,
        bytes,
        objectUrl,
        indexed: false,
      };
      this.corpusItems.push(newItem);
      this.renderGalleryStrip();

      if (this.isIndexed && this.retriever && !this.isIndexing) {
        this.isIndexing = true;
        this.updateControlsState();
        this.setStatus('busy', `Indexing custom image "${cleanName}"…`);
        try {
          await this.retriever.insertImage(newItem.id, newItem.bytes);
          newItem.indexed = true;
          this.renderGalleryStrip();
          this.setStatus('ready', `Indexed ${this.corpusItems.length} images · Ready to search`);
        } catch (err: any) {
          this.setStatus('error', err?.message || String(err));
        } finally {
          this.isIndexing = false;
          this.updateControlsState();
        }
      } else {
        this.btnIndexLabel.textContent = `Index ${this.corpusItems.length} images`;
      }
    });

    this.btnSearch.addEventListener('click', () => {
      this.runSearch();
    });

    this.queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.runSearch();
      }
    });

    this.suggestionChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const q = chip.dataset.query || '';
        this.queryInput.value = q;
        this.runSearch();
      });
    });
  }

  private setStatus(state: 'idle' | 'ready' | 'busy' | 'error', message: string) {
    this.timer.setStatus(state, message);
  }

  private updateControlsState() {
    const modelReady = retrievalRuntime.isReady() && Boolean(this.retriever);
    const busy = this.isIndexing || this.isSearching;

    this.btnIndex.disabled = !modelReady || busy;
    const count = this.corpusItems.length;
    this.btnIndexLabel.textContent = this.isIndexing
      ? 'Indexing…'
      : this.isIndexed
        ? `Re-index ${count} sample images`
        : `Index ${count} sample images`;

    const searchEnabled = modelReady && this.isIndexed && !busy;
    this.queryInput.disabled = !searchEnabled;
    this.btnSearch.disabled = !searchEnabled;
    this.suggestionChips.forEach((chip) => {
      chip.disabled = !searchEnabled;
    });
  }

  private async indexSampleImages() {
    if (!this.retriever || this.isIndexing) return;

    this.isIndexing = true;
    this.isIndexed = false;
    this.corpusItems.forEach((item) => {
      item.indexed = false;
    });
    this.updateControlsState();

    this.indexProgressWrap.style.display = 'block';
    this.indexProgressBar.style.width = '0%';

    const total = this.corpusItems.length;
    const tStart = performance.now();

    try {
      await this.retriever.deleteAll();

      for (let i = 0; i < total; i++) {
        const item = this.corpusItems[i];
        this.renderGalleryStrip(i);
        const stepMsg = `Embedding ${i + 1}/${total}: ${item.label}`;
        this.setStatus('busy', stepMsg);

        const bytes = item.isCustom ? item.bytes! : await retrievalRuntime.fetchSampleImageBytes(item);

        await this.retriever.insertImage(item.id, bytes);

        item.indexed = true;
        const pct = Math.round(((i + 1) / total) * 100);
        this.indexProgressBar.style.width = `${pct}%`;
      }

      this.renderGalleryStrip(-1);
      const totalSec = ((performance.now() - tStart) / 1000).toFixed(1);
      this.isIndexed = true;
      this.setStatus('ready', `Indexed ${total} images (${totalSec}s) · Enter a query below`);

      if (this.queryInput.value.trim()) {
        await this.runSearch();
      } else {
        this.resultsList.innerHTML = `
          <div class="no-results">
            ✓ ${total} images indexed in <code>MemoryVectorStore</code>. Click a suggestion chip above or type any natural-language description!
          </div>
        `;
      }
    } catch (err: any) {
      console.error(err);
      this.setStatus('error', err?.message || String(err));
    } finally {
      this.isIndexing = false;
      this.updateControlsState();
    }
  }

  private async runSearch() {
    if (!this.retriever || !this.isIndexed || this.isSearching) return;
    const query = this.queryInput.value.trim();
    if (!query) return;

    const limit = 5;
    this.isSearching = true;
    this.updateControlsState();
    this.setStatus('busy', `Retrieving top ${limit} matches for "${query}"…`);

    try {
      const t0 = performance.now();
      const results = await this.retriever.retrieve(query, { limit });
      const elapsedMs = Math.round(performance.now() - t0);

      this.renderSearchResults(query, results);
      publishTestResult('semantic_retriever', () => ({
        ...retrievalRuntime.testResultOrigin(),
        query,
        indexedCount: this.corpusItems.filter((item) => item.indexed).length,
        results: (results ?? []).map((r: { id: string; score: number }) => ({ id: r.id, score: r.score })),
      }));
      this.searchSummaryMeta.textContent = `Top ${results.length} results in ${elapsedMs} ms`;
      this.timer.recordSingle(elapsedMs);
      this.setStatus('ready', `Done · ${results.length} matches found`);
    } catch (err: any) {
      console.error(err);
      this.setStatus('error', err?.message || String(err));
    } finally {
      this.isSearching = false;
      this.updateControlsState();
    }
  }

  private renderSearchResults(query: string, results: Array<{ id: string; score: number }>) {
    if (!results || results.length === 0) {
      this.resultsList.innerHTML = `<div class="no-results">No matching images found for "${query}".</div>`;
      return;
    }

    this.resultsList.innerHTML = '';
    results.forEach((result, index) => {
      const itemMeta = this.corpusItems.find((c) => c.id === result.id);
      const displayTitle = itemMeta ? capitalizeFirst(itemMeta.label) : capitalizeFirst(result.id.replace(/_/g, ' '));
      const imgUrl = itemMeta
        ? itemMeta.isCustom
          ? itemMeta.objectUrl!
          : retrievalRuntime.getImageUrl(itemMeta.fileName)
        : retrievalRuntime.getImageUrl(`${result.id}.jpg`);

      const pct = Math.max(0, Math.min(100, Math.round(((result.score + 1) / 2) * 100)));

      const card = document.createElement('div');
      card.className = 'search-result-card';
      card.innerHTML = `
        <div class="result-rank-badge">${index + 1}</div>
        <img class="result-thumb" src="${imgUrl}" alt="${displayTitle}" />
        <div class="result-body">
          <div class="result-title-row">
            <span class="result-title">${displayTitle}</span>
            <span class="result-score-label">Similarity ${result.score.toFixed(4)}</span>
          </div>
          <div class="result-bar-track">
            <div class="result-bar-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
      this.resultsList.appendChild(card);
    });
  }

  cleanup() {
    this.timer.cleanup();
    if (this.retriever) {
      try {
        this.retriever.close();
      } catch {
        // Ignore close errors
      }
      this.retriever = null;
    }
  }
}

let activeTask: SemanticRetrieverTask | null = null;

export async function setupSemanticRetriever(container: HTMLElement) {
  activeTask = new SemanticRetrieverTask(container);
  await activeTask.initialize();
}

export function cleanupSemanticRetriever() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
