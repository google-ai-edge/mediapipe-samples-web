
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
 *limitations under the License.
 */

/**
 * MediaPipe ESM Web Worker Polyfills
 */
if (typeof (self as any).import === 'undefined') {
  (self as any).import = async (url: string) => {
    const module = await import(/* @vite-ignore */ url);
    if (module && module.default) {
      (self as any).ModuleFactory = module.default;
    }
    return module;
  };
}

if (typeof (self as any).custom_dbg === 'undefined') {
  (self as any).custom_dbg = function (text: string) {
    console.log('[MediaPipe Debug]:', text);
  };
}


export abstract class BaseWorker<T> {
  protected taskInstance: T | undefined;
  protected isInitializing = false;
  protected currentOptions: any = {};
  protected basePath = '/';
  protected isProcessing = false;

  protected static async loadWasmModule(basePath: string, fileName: string): Promise<any> {
    const url = `${basePath}/${fileName}`;

    const module = await import(/* @vite-ignore */ url);
    const ModuleFactory = module.default;

    const wasmModule = await ModuleFactory({
      print: (text: string) => console.log('[MediaPipe Debug]:', text),
      printErr: (text: string) => console.error('[MediaPipe Error]:', text),
      custom_dbg: (text: string) => console.log('[MediaPipe Debug]:', text),
    });

    return wasmModule;
  }

  constructor() {
    self.onmessage = this.handleMessage.bind(this);
  }

  protected async handleMessage(event: MessageEvent) {
    const { type } = event.data;

    while (this.isProcessing) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    this.isProcessing = true;

    try {
      if (type === 'INIT') {
        const { modelAssetPath, delegate, baseUrl, ...rest } = event.data;
        this.basePath = baseUrl || '/';
        this.currentOptions = { modelAssetPath, delegate, ...rest };

        await this.initializeBase(event.data);

        const payload = this.getInitPayload();
        self.postMessage({ type: 'INIT_DONE', ...payload });

      } else if (type === 'SET_OPTIONS') {
        const { type: _type, ...optionsToUpdate } = event.data;
        Object.assign(this.currentOptions, optionsToUpdate);
        await this.updateOptions(optionsToUpdate);
        self.postMessage({ type: 'OPTIONS_UPDATED' });

      } else if (type === 'CLEANUP') {
        if (this.taskInstance) {
          (this.taskInstance as any).close?.();
          this.taskInstance = undefined;
        }
        self.postMessage({ type: 'CLEANUP_DONE' });
      } else {
        await this.handleCustomMessage(event.data);
      }
    } catch (error: any) {
      console.error("Worker Error:", error);
      self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
    } finally {
      this.isProcessing = false;
    }
  }

  private async initializeBase(data: any) {
    if (this.isInitializing) return;
    this.isInitializing = true;

    try {
      if (this.taskInstance) {
        (this.taskInstance as any).close?.();
        this.taskInstance = undefined;
      }
      await this.initializeTask(data);
    } finally {
      this.isInitializing = false;
    }
  }

  protected async loadModelAsset(): Promise<ArrayBuffer> {
    const response = await fetch(this.currentOptions.modelAssetPath);
    if (!response.ok) {
      throw new Error(`Failed to load model: ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    const reader = response.body?.getReader();
    if (!reader) {
      return response.arrayBuffer();
    }

    let receivedLength = 0;
    const chunks: Uint8Array[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      if (total > 0) {
        self.postMessage({ type: 'LOAD_PROGRESS', loaded: receivedLength, total });
      }
    }

    const chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      chunksAll.set(chunk, position);
      position += chunk.length;
    }

    return chunksAll.buffer;
  }

  protected getWasmPath(): string {
    const formattedBasePath = this.basePath.endsWith('/') ? this.basePath : `${this.basePath}/`;
    return new URL(`${formattedBasePath}wasm`, self.location.origin).href.replace(/\/$/, '');
  }

  protected updateOptions(_?: any): Promise<void> { return Promise.resolve(); }

  protected abstract initializeTask(data?: any): Promise<void>;
  protected abstract handleCustomMessage(data: any): Promise<void>;

  protected getInitPayload(): any {
    return {};
  }
}
