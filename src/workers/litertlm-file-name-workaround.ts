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
 * Workaround for @mediapipe/tasks-audio <= 1.2.2 TextToSpeech.
 *
 * The Web task always writes the model to "/model.litertlm" in the Wasm file
 * system. The native task picks the TTS architecture from the model file name
 * ("kokoro", "qwen3", ...) and only probes the bundle contents when the name
 * gives no hint. That probe treats any bundle with an "af_heart" section as a
 * Phonon model, and Kokoro bundles ship an "af_heart" voice, so creation fails
 * with "No tokenizer found in the model for model type: tf_lite_prefill_decode".
 *
 * This keeps the original model file name in the Wasm file system, so the
 * native name check picks the right architecture. It wraps the Emscripten
 * module produced by the MediaPipe Wasm loader, which publishes its factory as
 * `globalThis.ModuleFactory`.
 *
 * Remove once the native architecture detection no longer misclassifies
 * Kokoro bundles.
 */

const DEFAULT_MODEL_FILE = 'model.litertlm';

/** Returns a Wasm-FS-safe file name that keeps the model's original name. */
export function litertLmFileNameFor(source: string | undefined): string | undefined {
  if (!source) return undefined;
  let name = source;
  try {
    name = new URL(source, self.location.href).pathname;
  } catch {
    // Not a URL (e.g. an uploaded file name); use as-is.
  }
  name = name.split('/').pop() ?? '';
  name = name.replace(/[^A-Za-z0-9._-]/g, '_');
  if (!name || name === DEFAULT_MODEL_FILE) return undefined;
  return name.endsWith('.litertlm') ? name : `${name}.litertlm`;
}

/**
 * Installs the workaround. `getModelFileName` is consulted whenever a task
 * writes or opens its model, and returns the file name to use instead of
 * "model.litertlm", or undefined to keep the default. Must be called before the
 * task's Wasm loader runs.
 */
export function installLitertLmFileNameWorkaround(getModelFileName: () => string | undefined) {
  const g = globalThis as any;
  let factory: any = g.ModuleFactory;
  const wrap = (f: any) =>
    typeof f !== 'function' || f.__litertLmFileNameWrapped
      ? f
      : Object.assign(async (...args: any[]) => patchModule(await f(...args), getModelFileName), {
          __litertLmFileNameWrapped: true,
        });
  factory = wrap(factory);
  Object.defineProperty(g, 'ModuleFactory', {
    configurable: true,
    get: () => factory,
    set: (f) => {
      factory = wrap(f);
    },
  });
}

function patchModule(module: any, getModelFileName: () => string | undefined): any {
  if (!module || typeof module.FS_createDataFile !== 'function') return module;
  // Resolved per call rather than per module: the loaded model can change
  // while the same module instance is reused.
  const remap = (path: unknown) => {
    const fileName = getModelFileName();
    return fileName && path === `/${DEFAULT_MODEL_FILE}` ? `/${fileName}` : path;
  };

  const createDataFile = module.FS_createDataFile;
  module.FS_createDataFile = (parent: string, name: string, ...rest: unknown[]) => {
    const fileName = getModelFileName();
    const target = fileName && parent === '/' && name === DEFAULT_MODEL_FILE ? fileName : name;
    return createDataFile.call(module, parent, target, ...rest);
  };

  const unlink = module.FS_unlink;
  if (typeof unlink === 'function') {
    module.FS_unlink = (path: string) => unlink.call(module, remap(path));
  }

  const createTextToSpeech = module.createTextToSpeech;
  if (typeof createTextToSpeech === 'function') {
    module.createTextToSpeech = (path: string, ...rest: unknown[]) =>
      createTextToSpeech.call(module, remap(path), ...rest);
  }
  return module;
}
