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

import { FilesetResolver } from '@mediapipe/tasks-vision';

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
      await new Promise((resolve) => setTimeout(resolve, 10));
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
      console.error('Worker Error:', error);
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
    } catch (error: any) {
      if (this.currentOptions.delegate === 'GPU') {
        const diagnostics = this.diagnoseWebGLFailure();
        console.warn('Worker GPU delegate initialization failed, falling back to CPU:', error, diagnostics);
        this.currentOptions.delegate = 'CPU';
        self.postMessage({ type: 'DELEGATE_FALLBACK', ...diagnostics });
        if (this.taskInstance) {
          try {
            (this.taskInstance as any).close?.();
          } catch (_) {}
          this.taskInstance = undefined;
        }
        await this.initializeTask(data);
      } else {
        throw error;
      }
    } finally {
      this.isInitializing = false;
    }
  }

  private diagnoseWebGLFailure(): { reason: string; advice: string } {
    try {
      if (typeof OffscreenCanvas === 'undefined') {
        return {
          reason: 'OffscreenCanvas unsupported',
          advice: 'OffscreenCanvas is unsupported in this browser environment.',
        };
      }
      const testCanvas = new OffscreenCanvas(1, 1);
      const gl2 = testCanvas.getContext('webgl2') as WebGL2RenderingContext | null;
      if (!gl2) {
        const gl1 = testCanvas.getContext('webgl');
        if (!gl1) {
          return {
            reason: 'WebGL disabled',
            advice: 'WebGL is disabled or unsupported in browser settings.',
          };
        }
        return {
          reason: 'WebGL 2.0 unsupported',
          advice: 'Device supports WebGL 1.0, but MediaPipe GPU requires WebGL 2.0.',
        };
      }

      if (gl2.isContextLost()) {
        return {
          reason: 'WebGL context lost',
          advice:
            'Maximum active WebGL contexts limit exceeded for this domain.' + ' Please close other tabs or refresh.',
        };
      }

      const debugInfo = gl2.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        const rawParam = gl2.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '';
        const renderer = rawParam.toString().toLowerCase();
        if (renderer.includes('swiftshader') || renderer.includes('software')) {
          return {
            reason: 'Software WebGL renderer',
            advice: 'Hardware acceleration disabled in browser.',
          };
        }
      }
    } catch (_) {}

    return {
      reason: 'GPU graph initialization failed',
      advice: 'GPU delegate initialization failed in WebAssembly.',
    };
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

  protected async getVisionFileset() {
    const wasmPath = this.getWasmPath();
    const fileset = await FilesetResolver.forVisionTasks(wasmPath, true);
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}`; // Force reload
    return fileset;
  }

  protected async getAudioFileset() {
    const wasmPath = this.getWasmPath();
    const fileset = await FilesetResolver.forAudioTasks(wasmPath, true);
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}`; // Force reload
    return fileset;
  }

  protected async getTextFileset() {
    const wasmPath = this.getWasmPath();
    const fileset = await FilesetResolver.forTextTasks(wasmPath, true);
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}`; // Force reload
    return fileset;
  }

  protected updateOptions(_?: any): Promise<void> {
    return Promise.resolve();
  }

  protected abstract initializeTask(data?: any): Promise<void>;
  protected abstract handleCustomMessage(data: any): Promise<void>;

  protected getInitPayload(): any {
    return {};
  }
}
