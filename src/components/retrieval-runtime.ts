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
  DefaultTextChunker,
  FilesetResolver,
  MemoryVectorStore,
  SemanticRetriever,
  SemanticRetrieverComponents,
  UniversalEmbedder,
} from '@mediapipe/tasks-retrieval';

import {
  formatMegabytes,
  openModelStream,
  resolveModelDownloadUrl,
  trackProgress,
  type ModelProgressCallback,
} from './model-cache';
import {
  EMBEDDING_GEMMA_2_740M,
  EMBEDDING_GEMMA_2_TEXT_VISION_440M,
  hostedModelDownloadUrl,
  type HostedModel,
} from './model-registry';
import { ModelSelector } from './model-selector';

export interface SampleImageItem {
  id: string;
  label: string;
  fileName: string;
}

export const SAMPLE_IMAGES: SampleImageItem[] = [
  { id: 'red_apple', label: 'red apple', fileName: 'red_apple.jpg' },
  { id: 'yellow_banana', label: 'yellow banana', fileName: 'yellow_banana.jpg' },
  { id: 'cute_cat', label: 'cute cat', fileName: 'cute_cat.jpg' },
  { id: 'fast_car', label: 'fast car', fileName: 'fast_car.jpg' },
  { id: 'green_tree', label: 'green tree', fileName: 'green_tree.jpg' },
  { id: 'blue_sky', label: 'blue sky', fileName: 'blue_sky.jpg' },
  { id: 'coffee_mug', label: 'coffee mug', fileName: 'coffee_mug.jpg' },
  { id: 'open_book', label: 'open book', fileName: 'open_book.jpg' },
  { id: 'sunny_beach', label: 'sunny beach', fileName: 'sunny_beach.jpg' },
  {
    id: 'snowy_mountain',
    label: 'snowy mountain',
    fileName: 'snowy_mountain.jpg',
  },
];

/** Metadata configuration for standard pre-quantized retrieval models. */
export type StandardRetrievalModel = HostedModel;

/**
 * Standard pre-configured multimodal retrieval models available in the demo.
 * Shared with the Decision Maker (see model-registry.ts), so a model downloaded
 * here is reused there without another download.
 */
export const STANDARD_RETRIEVAL_MODELS: StandardRetrievalModel[] = [
  EMBEDDING_GEMMA_2_TEXT_VISION_440M,
  EMBEDDING_GEMMA_2_740M,
];

/** Direct download URL for a standard model (or any model page URL). */
export function retrievalModelDownloadUrl(url: string): string {
  const matched = STANDARD_RETRIEVAL_MODELS.find((m) => m.url === url);
  return matched ? hostedModelDownloadUrl(matched) : resolveModelDownloadUrl(url);
}

/**
 * WASM binaries location.
 * Local: './wasm' (populated by copy-wasm.js to public/wasm/).
 * TODO(gkarpiak): Post-release CDN: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-retrieval@1.1.0/wasm'
 */
export const RETRIEVAL_WASM_PATH = './wasm';

/** Base path for sample images located in public/images/ */
export const SAMPLE_IMAGES_PATH = './images';

/** Where the active embedder's model came from, so it can be re-created with new options. */
export type RetrievalModelSource = { type: 'url'; url: string } | { type: 'file'; file: File };

class RetrievalRuntimeManager {
  public wasmFileset: any = null;
  public embedder: any = null;
  public activeModelLabel: string | null = null;
  public activeL2Normalize = true;
  public lastSelectedFile: File | null = null;
  public lastSelectedUrl = STANDARD_RETRIEVAL_MODELS[0].url;
  /** Source of the currently loaded model (null until one has been loaded). */
  public activeSource: RetrievalModelSource | null = null;
  public isInitializing = false;
  private imageBytesCache = new Map<string, Uint8Array>();

  getImageUrl(fileName: string): string {
    return `${SAMPLE_IMAGES_PATH}/${fileName}`;
  }

  async fetchSampleImageBytes(sample: SampleImageItem): Promise<Uint8Array> {
    if (this.imageBytesCache.has(sample.id)) {
      return this.imageBytesCache.get(sample.id)!;
    }
    const url = this.getImageUrl(sample.fileName);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch sample image ${url} (${response.status})`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    this.imageBytesCache.set(sample.id, bytes);
    return bytes;
  }

  async resolveWasmFileset(): Promise<any> {
    if (this.wasmFileset) return this.wasmFileset;
    this.wasmFileset = await (FilesetResolver as any).forRetrievalTasks(RETRIEVAL_WASM_PATH);
    return this.wasmFileset;
  }

  isReady(): boolean {
    return this.embedder !== null && !this.isInitializing;
  }

  private closeEmbedder() {
    if (this.embedder) {
      try {
        this.embedder.close();
      } catch {
        // Ignore close errors
      }
      this.embedder = null;
    }
  }

  async initializeFromFile(file: File, l2Normalize = true, onProgress?: ModelProgressCallback | null): Promise<any> {
    this.lastSelectedFile = file;
    this.isInitializing = true;

    try {
      const wasmFileset = await this.resolveWasmFileset();
      this.closeEmbedder();

      const reader = trackProgress(file.stream(), file.size, onProgress).getReader();

      this.embedder = await UniversalEmbedder.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetBuffer: reader,
        },
        l2Normalize,
      });

      this.activeModelLabel = `${file.name} (${formatMegabytes(file.size)})`;
      this.activeL2Normalize = l2Normalize;
      this.activeSource = { type: 'file', file };
      return this.embedder;
    } finally {
      this.isInitializing = false;
    }
  }

  async initializeFromUrl(
    url: string,
    l2Normalize = true,
    onProgress?: ModelProgressCallback | null,
    onCached?: () => void
  ): Promise<any> {
    this.lastSelectedUrl = url;
    this.isInitializing = true;

    try {
      const wasmFileset = await this.resolveWasmFileset();

      // Served from the shared model cache after the first download, so the
      // same file is reused by the Universal Embedder, Semantic Retriever and
      // Decision Maker demos (and across page reloads).
      const { stream, cached } = await openModelStream(retrievalModelDownloadUrl(url), onProgress);
      if (onCached) cached.then((ok) => ok && onCached());
      this.closeEmbedder();

      this.embedder = await UniversalEmbedder.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetBuffer: stream.getReader(),
        },
        l2Normalize,
      });

      const matched = STANDARD_RETRIEVAL_MODELS.find((m) => m.url === url);
      this.activeModelLabel = matched ? matched.name : url.split('/').pop() || url;
      this.activeL2Normalize = l2Normalize;
      this.activeSource = { type: 'url', url };
      return this.embedder;
    } finally {
      this.isInitializing = false;
    }
  }

  /** Re-creates the embedder from the last loaded source with new options. */
  async reinitialize(l2Normalize: boolean, onProgress?: ModelProgressCallback | null): Promise<any> {
    const source = this.activeSource;
    if (!source) throw new Error('No model has been loaded yet.');
    return source.type === 'file'
      ? this.initializeFromFile(source.file, l2Normalize, onProgress)
      : this.initializeFromUrl(source.url, l2Normalize, onProgress);
  }

  async createSemanticRetriever(chunkSize = 512, chunkOverlap = 100): Promise<any> {
    if (!this.embedder) {
      throw new Error('UniversalEmbedder is not initialized yet.');
    }
    const wasmFileset = await this.resolveWasmFileset();
    let textChunker;
    if (this.embedder.wasmModule && typeof DefaultTextChunker.createFromModule === 'function') {
      textChunker = DefaultTextChunker.createFromModule(this.embedder.wasmModule, chunkSize, chunkOverlap);
    } else {
      textChunker = await DefaultTextChunker.create(wasmFileset, chunkSize, chunkOverlap);
    }

    const vectorStore = new MemoryVectorStore();
    const components = new SemanticRetrieverComponents()
      .addProvider(this.embedder.getProvider())
      .setVectorStore(vectorStore)
      .setTextChunker(textChunker);

    return SemanticRetriever.createFromComponents(components);
  }
}

export const retrievalRuntime = new RetrievalRuntimeManager();

export interface ModelSelectorCallbacks {
  getL2Normalize?: () => boolean;
  onLoadStart?: (message: string) => void;
  onModelReady?: () => Promise<void> | void;
  onError?: (err: any) => void;
}

/**
 * Mounts the shared model selector for the retrieval demos. Standard models
 * are only downloaded once the user presses "Initialize Task"; the resulting
 * embedder lives in `retrievalRuntime` and is shared by both retrieval tasks.
 */
export function mountModelSelector(containerId: string, callbacks: ModelSelectorCallbacks): ModelSelector | null {
  if (!document.getElementById(containerId)) return null;

  const selector = new ModelSelector(
    containerId,
    STANDARD_RETRIEVAL_MODELS.map((model, idx) => ({ label: model.name, value: model.url, isDefault: idx === 0 })),
    async (selection) => {
      const l2Normalize = callbacks.getL2Normalize ? callbacks.getL2Normalize() : true;
      const name =
        selection.type === 'custom'
          ? selection.file.name
          : (STANDARD_RETRIEVAL_MODELS.find((m) => m.url === selection.value)?.name ?? selection.value);

      selector.setBusy(true);
      selector.setStatus(`Loading ${name}...`);
      callbacks.onLoadStart?.(`Loading ${name}...`);

      try {
        if (selection.type === 'custom') {
          await retrievalRuntime.initializeFromFile(selection.file, l2Normalize, (l, t) => selector.showProgress(l, t));
        } else {
          await retrievalRuntime.initializeFromUrl(
            selection.value,
            l2Normalize,
            (l, t) => selector.showProgress(l, t),
            () => selector.refreshCacheState()
          );
        }
        selector.hideProgress();
        selector.setBusy(false);
        selector.setLoaded(selection.type === 'standard' ? selection.value : null);
        selector.setStatus(`✓ Active: ${retrievalRuntime.activeModelLabel}`);
        await callbacks.onModelReady?.();
      } catch (err: any) {
        selector.hideProgress();
        selector.setBusy(false);
        selector.setStatus(`Failed to load ${name}`);
        callbacks.onError?.(err);
      }
    },
    {
      autoLoad: false,
      accept: '.litertlm,.bin,.task,.tflite',
      uploadLabel: 'Choose .litertlm Model',
      resolveUrl: retrievalModelDownloadUrl,
      preferCached: true,
    }
  );

  // A model loaded by the other retrieval demo is still active: reflect it.
  if (retrievalRuntime.isReady()) {
    const source = retrievalRuntime.activeSource;
    if (source?.type === 'url') selector.setSelectedValue(source.url);
    selector.setLoaded(source?.type === 'url' ? source.url : null);
    selector.setStatus(`✓ Active: ${retrievalRuntime.activeModelLabel}`);
  }

  return selector;
}
