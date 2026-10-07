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

/**
 * Built-in models. `url` models are downloaded directly; the others are served
 * by the dev server from LOCAL_MODELS_DIR (see vite.config.ts).
 */
export const DECISION_MODELS: Record<string, { label: string; file: string; url?: string; unsupported?: boolean }> = {
  embeddinggemma2_270m: {
    label: 'EmbeddingGemma-2 Text 270M',
    file: 'embeddinggemma-2-text-270m.litertlm',
    url: 'https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm',
  },
  laya_s256: {
    label: 'Laya S256',
    file: 'laya_s256.task',
    url: 'https://storage.googleapis.com/mediapipe-models/decision_maker/laya/float32/laya_s256/latest/laya_s256.task',
  },
};

/**
 * Resolves model page URLs (e.g. Hugging Face repository or tree URLs) to their
 * direct binary download endpoints suitable for HTTP fetching and streaming.
 */
export function resolveModelDownloadUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  const hfRepoMatch = trimmed.match(
    /^https?:\/\/huggingface\.co\/([^/]+)\/([^/]+)(?:\/(?:tree|blob|resolve)\/([^/]+)(?:\/(.+))?)?$/
  );
  if (hfRepoMatch) {
    const [, org, repo, branchOrType, filePath] = hfRepoMatch;
    // 1. Direct download endpoint already specified
    if (branchOrType === 'resolve' && filePath) {
      return trimmed;
    }
    // 2. Look up the specific model filename for known repository versions
    const matched = Object.values(DECISION_MODELS).find((m) => m.url?.includes(`${org}/${repo}`));
    const fileName = filePath || (matched ? matched.file : `${repo}.litertlm`);
    // 3. Preserve custom git branch/tag/revision if specified, otherwise default to 'main'
    const branch = branchOrType && branchOrType !== 'tree' && branchOrType !== 'blob' ? branchOrType : 'main';
    return `https://huggingface.co/${org}/${repo}/resolve/${branch}/${fileName}`;
  }
  // Passthrough for non-Hugging Face URLs (GCS, local dev server, direct CDNs)
  return trimmed;
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
    const localUrl = new URL(`local-models/${model.file}`, new URL(baseUrl, window.location.origin)).href;
    const modelUrl = this.customModel ? undefined : model.url ? resolveModelDownloadUrl(model.url) : localUrl;
    if (model.url && !this.customModel) {
      this.emit({ type: 'status', text: `Downloading ${model.file} (first load may take a while)...` });
    }

    // Local models are served from LOCAL_MODELS_DIR by the dev server. Check
    // it's there first; otherwise the wasm gets an error page instead of a model.
    if (modelUrl === localUrl && !(await this.isModelAvailable(modelUrl))) {
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
