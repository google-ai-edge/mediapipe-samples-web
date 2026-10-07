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

/**
 * Shared DecisionMaker runtime for the Decision Maker and Dino Game pages.
 *
 * Like retrieval-runtime.ts for the retrieval demos, the worker and its loaded
 * model live here rather than in a page, so a model loaded on one page is still
 * loaded after navigating to the other. Pages subscribe to events and
 * unsubscribe on cleanup; they never terminate the worker.
 */

import type { QuestionKind } from '../tasks/decision-maker-json';
import { resolveModelDownloadUrl } from './model-cache';
import { EMBEDDING_GEMMA_2_TEXT_270M, EMBEDDING_GEMMA_2_TEXT_VISION_440M } from './model-registry';

/**
 * Built-in models. `url` models are downloaded directly (through the shared
 * model cache, so one already fetched by another demo is reused); the others are
 * served by the dev server from LOCAL_MODELS_DIR (see vite.config.ts).
 */
export const DECISION_MODELS: Record<string, { label: string; file: string; url?: string; unsupported?: boolean }> = {
  embeddinggemma2_270m: {
    label: EMBEDDING_GEMMA_2_TEXT_270M.name,
    file: EMBEDDING_GEMMA_2_TEXT_270M.fileName,
    url: EMBEDDING_GEMMA_2_TEXT_270M.url,
  },
  // Same model the Universal Embedder / Semantic Retriever demos use: if it was
  // loaded there it is already on disk and loads here without a download.
  embeddinggemma2_text_vision_440m: {
    label: EMBEDDING_GEMMA_2_TEXT_VISION_440M.name,
    file: EMBEDDING_GEMMA_2_TEXT_VISION_440M.fileName,
    url: EMBEDDING_GEMMA_2_TEXT_VISION_440M.url,
  },
  laya_s256: {
    label: 'Laya S256',
    file: 'laya_s256.task',
    url: 'https://storage.googleapis.com/mediapipe-models/decision_maker/laya/float32/laya_s256/latest/laya_s256.task',
  },
};

/** Direct download URL of a built-in model (remote, or the dev server's local-models route). */
export function decisionModelDownloadUrl(name: string): string | undefined {
  const model = DECISION_MODELS[name];
  if (!model) return undefined;
  if (model.url) return resolveModelDownloadUrl(model.url, model.file);
  return new URL(`local-models/${model.file}`, new URL(import.meta.env.BASE_URL, window.location.origin)).href;
}

/** `schema` evaluates a whole ClassifierSchema in one request. */
export type DecisionKind = QuestionKind | 'schema';
export type Delegate = 'GPU' | 'CPU';

export type RuntimeEvent =
  /** Load started (pages should stop using the old model). */
  | { type: 'loading' }
  /** Bytes streamed so far (total is 0 when unknown). */
  | { type: 'progress'; loaded: number; total: number }
  /** The model finished loading. */
  | { type: 'ready' }
  /** Loading or a model call failed. */
  | { type: 'error'; error: string }
  /** A line for the page's status message. */
  | { type: 'status'; text: string }
  /** loading / ready / selection / delegate changed: re-read the runtime state. */
  | { type: 'state' }
  | { type: 'cached' };

class DecisionRuntimeManager {
  modelName = 'embeddinggemma2_270m';
  /** Set when the user uploads a model file instead of picking a standard one. */
  customModel: File | undefined;
  delegate: Delegate = 'GPU';

  ready = false;
  /** True from load() until the worker reports INIT_DONE / ERROR. */
  loading = false;
  /** True if the last load failed. */
  failed = false;
  /** Label of the loaded model (null = none yet). */
  loadedLabel: string | null = null;
  /** Value of the loaded model (null = none or custom file). */
  loadedValue: string | null = null;
  /** Label of the model being loaded (the selection may change while it loads). */
  loadingLabel = '';
  loadingValue: string | null = null;

  private worker: Worker | undefined;
  private requestId = 0;
  private resolvers = new Map<number, (msg: any) => void>();
  /** Bumped on every load so stale async steps are ignored. */
  private epoch = 0;
  private listeners = new Set<(event: RuntimeEvent) => void>();

  /** Subscribes to runtime events; returns the unsubscribe function. */
  subscribe(listener: (event: RuntimeEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: RuntimeEvent) {
    for (const listener of [...this.listeners]) listener(event);
  }

  /** Display name of the currently selected model (upload or standard). */
  currentLabel(): string {
    return this.customModel ? this.customModel.name : DECISION_MODELS[this.modelName].label;
  }

  /** Selects a standard model (loaded when the user presses Load Model). */
  select(modelName: string) {
    this.customModel = undefined;
    this.modelName = modelName;
    this.emit({ type: 'state' });
  }

  /** Uploading a file loads it right away, like the retrieval demos. */
  selectFile(file: File) {
    this.customModel = file;
    this.load();
  }

  /** Only reloads if a model is already loaded (or loading); otherwise waits for Load Model. */
  setDelegate(delegate: Delegate) {
    this.delegate = delegate;
    if (this.loadedLabel || this.loading) this.load();
    else this.emit({ type: 'state' });
  }

  async load() {
    this.worker?.terminate();
    this.worker = undefined;
    this.ready = false;
    this.failed = false;
    // Settle calls to the old worker so awaiting code unwinds.
    for (const resolve of this.resolvers.values()) resolve({ type: 'ERROR', error: 'Model reloaded' });
    this.resolvers.clear();
    const epoch = ++this.epoch;
    this.loading = true;
    this.loadedLabel = null;
    this.loadingLabel = this.currentLabel();
    this.loadingValue = this.customModel ? null : this.modelName;
    this.emit({ type: 'loading' });
    this.emit({ type: 'state' });

    const model = DECISION_MODELS[this.modelName];
    this.emit({ type: 'status', text: `Loading ${this.customModel?.name ?? model.file}...` });

    const baseUrl = import.meta.env.BASE_URL;
    // Downloaded through the shared model cache in the worker, so a model
    // already fetched by another demo is served from disk.
    const modelUrl = this.customModel ? undefined : decisionModelDownloadUrl(this.modelName);

    // Local models are served from LOCAL_MODELS_DIR by the dev server. Check
    // it's there first; otherwise the wasm gets an error page instead of a model.
    if (modelUrl && !model.url && !(await this.isModelAvailable(modelUrl))) {
      if (epoch !== this.epoch) return;
      this.fail(
        `${model.file} not found. Upload a model, or set LOCAL_MODELS_DIR in .env.local and restart the dev server.`
      );
      return;
    }
    if (epoch !== this.epoch) return;

    this.worker = new Worker(new URL('../workers/decision-maker.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event) => this.onWorkerMessage(event.data);
    this.worker.postMessage({
      type: 'INIT',
      modelAssetPath: modelUrl,
      modelFile: this.customModel,
      delegate: this.delegate,
      baseUrl,
    });
  }

  /** Runs one evaluation in the worker; resolves with the raw worker message. */
  ask(text: string, kind: DecisionKind, question: object): Promise<any> {
    if (!this.worker || !this.ready) return Promise.resolve({ type: 'ERROR', error: 'Model not ready' });
    const id = ++this.requestId;
    return new Promise((resolve) => {
      this.resolvers.set(id, resolve);
      this.worker!.postMessage({ type: 'DECIDE', id, kind, text, question });
    });
  }

  private async isModelAvailable(url: string): Promise<boolean> {
    try {
      const res = await fetch(url, { method: 'HEAD' });
      return res.ok && !(res.headers.get('Content-Type') ?? '').includes('text/html');
    } catch {
      return false;
    }
  }

  private fail(error: string) {
    this.loading = false;
    this.failed = true;
    this.emit({ type: 'state' });
    this.emit({ type: 'error', error });
  }

  private onWorkerMessage(msg: any) {
    switch (msg.type) {
      case 'LOAD_PROGRESS':
        if (this.loading) this.emit({ type: 'progress', loaded: msg.loaded, total: msg.total });
        break;
      case 'MODEL_CACHED':
        this.emit({ type: 'cached' });
        break;
      case 'INIT_DONE':
        this.ready = true;
        this.loading = false;
        this.loadedLabel = this.loadingLabel;
        this.loadedValue = this.loadingValue;
        this.emit({ type: 'state' });
        this.emit({ type: 'status', text: 'Model ready' });
        this.emit({ type: 'ready' });
        break;
      case 'DELEGATE_FALLBACK':
        this.delegate = 'CPU';
        this.emit({ type: 'state' });
        break;
      case 'DECIDE_RESULT':
        this.resolvers.get(msg.id)?.(msg);
        this.resolvers.delete(msg.id);
        break;
      case 'ERROR':
        console.error('Decision Maker error:', msg.error);
        for (const resolve of this.resolvers.values()) resolve(msg);
        this.resolvers.clear();
        if (this.loading) this.fail(msg.error);
        else this.emit({ type: 'error', error: msg.error });
        break;
    }
  }
}

export const decisionRuntime = new DecisionRuntimeManager();
