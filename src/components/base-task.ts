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

import { ModelSelector } from './model-selector';

export interface BaseTaskOptions {
  container: HTMLElement;
  template: string;
  defaultModelName: string;
  defaultModelUrl: string;
  workerFactory: () => Worker;
  defaultDelegate?: 'CPU' | 'GPU';
}

export abstract class BaseTask {
  protected container: HTMLElement;
  protected worker: Worker | undefined;

  protected currentModel: string;
  protected models: Record<string, string> = {};
  protected modelSelector!: ModelSelector;
  protected currentDelegate: 'CPU' | 'GPU' = 'GPU';

  protected isWorkerReady = false;

  constructor(protected options: BaseTaskOptions) {
    this.container = options.container;
    this.currentModel = options.defaultModelName;
    this.models[options.defaultModelName] = options.defaultModelUrl;
    if (options.defaultDelegate) {
      this.currentDelegate = options.defaultDelegate;
    }
  }

  public async initialize() {
    this.container.innerHTML = this.options.template;

    this.initWorker();
    this.setupUI();

    // Child class hook
    this.onInitializeUI();
    this.setupDelegateSelect();

    await this.initializeTask();
  }

  protected initWorker() {
    if (!this.worker) {
      this.worker = this.options.workerFactory();
    }
    if (this.worker) {
      this.worker.onmessage = this.handleWorkerMessage.bind(this);
    }
  }

  protected hadDelegateFallback = false;

  protected handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case 'LOAD_PROGRESS':
        this.handleLoadProgress(event.data);
        break;

      case 'INIT_DONE':
        this.handleInitDone();
        break;

      case 'DELEGATE_FALLBACK':
        const { reason, advice } = event.data;
        const msg = advice ? `${reason} (${advice})` : 'GPU unavailable.';
        console.warn(`Worker fell back to CPU delegate: ${msg}`);
        this.currentDelegate = 'CPU';
        this.hadDelegateFallback = true;
        const delSelect = document.getElementById('delegate-select') as HTMLSelectElement;
        if (delSelect) delSelect.value = 'CPU';
        this.renderFallbackWarning(msg);
        break;

      case 'ERROR':
      case 'DETECT_ERROR':
      case 'CLASSIFY_ERROR':
        console.error('Worker error:', event.data.error);
        this.updateStatus(`Error: ${event.data.error}`);
        break;
    }
  }

  protected handleLoadProgress(data: any) {
    const { progress, loaded, total } = data;
    if (progress !== undefined) {
      this.modelSelector?.showProgress(progress * 100, 100);
      if (progress >= 1) setTimeout(() => this.modelSelector?.hideProgress(), 500);
    } else if (loaded !== undefined && total !== undefined) {
      this.modelSelector?.showProgress(loaded, total);
      if (loaded >= total) setTimeout(() => this.modelSelector?.hideProgress(), 500);
    }
  }

  protected handleInitDone() {
    this.modelSelector?.hideProgress();
    document.querySelector('.viewport')?.classList.remove('loading-model');
    this.isWorkerReady = true;
    if (this.hadDelegateFallback) {
      this.updateStatus('GPU unavailable. Using CPU delegate (Ready).');
      this.hadDelegateFallback = false;
    } else {
      this.updateStatus('Ready');
    }
  }

  protected setupDelegateSelect() {
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
    if (delegateSelect) {
      delegateSelect.addEventListener('change', async () => {
        this.currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
        await this.initializeTask();
      });
      delegateSelect.value = this.currentDelegate;
    }
  }

  protected setupUI() {
    this.modelSelector = new ModelSelector(
      'model-selector-container',
      [{ label: this.options.defaultModelName, value: this.options.defaultModelName, isDefault: true }],
      async (selection) => {
        if (selection.type === 'standard') {
          this.currentModel = selection.value;
        } else if (selection.type === 'custom') {
          this.models['custom'] = URL.createObjectURL(selection.file);
          this.currentModel = 'custom';
        }
        await this.initializeTask();
      }
    );
  }

  protected async initializeTask(): Promise<void> {
    document.querySelector('.viewport')?.classList.add('loading-model');
    this.isWorkerReady = false;
    this.updateStatus('Loading Model...');

    // @ts-ignore
    const baseUrl = import.meta.env.BASE_URL;
    let modelPath = this.models[this.currentModel];

    if (this.currentModel === 'custom' && this.models['custom']) {
      modelPath = this.models['custom'];
    } else if (!modelPath.startsWith('http')) {
      modelPath = new URL(modelPath, new URL(baseUrl, window.location.origin)).href;
    }

    const initParams = this.getWorkerInitParamsInner();

    this.worker?.postMessage({
      type: 'INIT',
      modelAssetPath: modelPath,
      delegate: this.currentDelegate,
      baseUrl,
      ...initParams,
    });
  }

  protected getWorkerInitParamsInner(): Record<string, any> {
    return this.getWorkerInitParams();
  }

  protected updateStatus(msg: string) {
    const el = document.getElementById('status-message');
    if (el) el.innerText = msg;
  }

  protected updateInferenceTime(time: number) {
    const el = document.getElementById('inference-time');
    if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
  }

  public cleanup() {
    if (this.worker) {
      this.worker.postMessage({ type: 'CLEANUP' });
      this.worker.terminate();
      this.worker = undefined;
    }

    this.isWorkerReady = false;
  }

  protected renderFallbackWarning(msg: string) {
    let warningEl = document.getElementById('fallback-warning');
    if (!warningEl) {
      const statusGroup = document.querySelector('.status-group');
      if (statusGroup) {
        warningEl = document.createElement('div');
        warningEl.id = 'fallback-warning';
        warningEl.className = 'fallback-warning';
        warningEl.style.cssText = 'color: #d97706; font-size: 0.85rem; margin-top: 4px;' + ' font-weight: 500;';
        statusGroup.appendChild(warningEl);
      }
    }
    if (warningEl) {
      warningEl.innerText = `⚠️ GPU Unavailable (${msg}). Switched to CPU delegate.`;
    }
  }

  protected onInitializeUI(): void {}
  protected abstract getWorkerInitParams(): Record<string, any>;
}
