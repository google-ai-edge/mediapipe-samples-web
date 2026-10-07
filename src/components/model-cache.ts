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
 * Shared model download + cache layer.
 *
 * Every task, whether it runs its model in a worker (vision, audio, text,
 * decision) or on the main thread (retrieval), gets its model bytes through
 * `openModelStream()` / `fetchModelBytes()`. Downloads are persisted with the
 * Cache API, keyed by URL, so a model only has to be downloaded once per
 * origin: a later page load, a different task that uses the same file (e.g.
 * Universal Embedder and Semantic Retriever both using EmbeddingGemma), a
 * delegate switch or a GPU→CPU fallback all read it back from disk.
 *
 * The Cache API was chosen over OPFS / IndexedDB because it is available on
 * both the main thread and in workers, stores responses as streams (a model of
 * several hundred MB never has to be held in JS memory to persist it) and needs
 * no schema or bookkeeping: the URL is the key. When it is unavailable (e.g.
 * insecure contexts, some private-browsing modes) or the write fails (quota),
 * the model is simply streamed straight from the network as before.
 *
 * This module must stay free of DOM access so it can be imported by workers.
 */

const MODEL_CACHE_NAME = 'mediapipe-models-v1';

export type ModelProgressCallback = (loaded: number, total: number) => void;

export interface ModelStream {
  /** The model bytes. */
  stream: ReadableStream<Uint8Array>;
  /** Total size in bytes, or 0 if unknown. */
  size: number;
  /** True if the bytes come from the local cache rather than the network. */
  fromCache: boolean;
  /**
   * Resolves to true once the model is in the shared cache (right away for a
   * cache hit, after the background write for a download), or false if it could
   * not be cached.
   */
  cached: Promise<boolean>;
}

/**
 * Resolves model page URLs (e.g. Hugging Face repository or tree URLs) to their
 * direct binary download endpoints suitable for HTTP fetching and streaming.
 *
 * Browsers require direct raw binary streams (HTTP 200 with binary content) to
 * initialize LiteRT via WebAssembly / WebGPU. When users copy links from Hugging
 * Face or select pre-defined models, the input URL can come in multiple
 * web-facing formats rather than a direct raw download URL.
 *
 * @param rawUrl The input URL from UI selection, input field, or configuration.
 * @param defaultFileName File to download when `rawUrl` is a bare repository
 *     URL (defaults to `<repo>.litertlm`).
 * @returns The direct HTTP download URL that resolves to raw model binary bytes.
 */
export function resolveModelDownloadUrl(rawUrl: string, defaultFileName?: string): string {
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
    // 2. Use the specific model filename for the repository when known
    const fileName = filePath || defaultFileName || `${repo}.litertlm`;
    // 3. Preserve custom git branch/tag/revision if specified, otherwise default to 'main'
    const branch = branchOrType && branchOrType !== 'tree' && branchOrType !== 'blob' ? branchOrType : 'main';
    return `https://huggingface.co/${org}/${repo}/resolve/${branch}/${fileName}`;
  }
  // Passthrough for non-Hugging Face URLs (GCS, local dev server, direct CDNs)
  return trimmed;
}

/** Only http(s) URLs are cached; blob:/data: URLs (uploads) are transient by nature. */
export function isCacheableModelUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

/** Formats a byte count as "12.3 MB" (or "0.8 MB" for small files). */
export function formatMegabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Pipes `stream` through a counter that reports cumulative progress.
 * Returns the input unchanged when there is nobody to report to.
 */
export function trackProgress(
  stream: ReadableStream<Uint8Array>,
  total: number,
  onProgress?: ModelProgressCallback | null
): ReadableStream<Uint8Array> {
  if (!onProgress) return stream;
  let loaded = 0;
  return stream.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        loaded += chunk.byteLength;
        onProgress(loaded, total);
        controller.enqueue(chunk);
      },
    })
  );
}

async function openModelCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return null;
  try {
    return await caches.open(MODEL_CACHE_NAME);
  } catch (err) {
    // SecurityError in some private-browsing modes; treat as "no cache".
    console.warn('Model cache unavailable:', err);
    return null;
  }
}

function contentLength(response: Response): number {
  const value = Number(response.headers.get('content-length') || 0);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

async function fetchFromNetwork(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`Failed to download model (HTTP ${response.status}) from ${url}`);
  }
  // Guard against SPA fallbacks / error pages served with a 200 status: the
  // wasm would otherwise try to parse an HTML document as a model.
  if ((response.headers.get('content-type') || '').includes('text/html')) {
    throw new Error(`Expected a model file but received an HTML page from ${url}`);
  }
  return response;
}

/**
 * Cache writes still in flight in this JS context, so that a second request for
 * the same URL (e.g. a GPU→CPU re-initialization) waits for the entry instead of
 * downloading the model a second time. Each resolves to whether the write
 * succeeded.
 */
const pendingWrites = new Map<string, Promise<boolean>>();

/** Whether `url` has already been downloaded into the shared model cache. */
export async function isModelCached(url: string): Promise<boolean> {
  if (!isCacheableModelUrl(url)) return false;
  const cache = await openModelCache();
  if (!cache) return false;
  await pendingWrites.get(url);
  return (await cache.match(url)) !== undefined;
}

/** Removes a single model from the shared cache. Returns true if it was there. */
export async function removeCachedModel(url: string): Promise<boolean> {
  const cache = await openModelCache();
  if (!cache) return false;
  await pendingWrites.get(url);
  return cache.delete(url);
}

/** Deletes every cached model. */
export async function clearModelCache(): Promise<void> {
  if (typeof caches === 'undefined') return;
  await Promise.all(pendingWrites.values());
  await caches.delete(MODEL_CACHE_NAME);
}

/**
 * Opens a byte stream for the model at `url`.
 *
 * Served from the shared cache when possible; otherwise downloaded, and – while
 * the caller consumes it – written to the cache in the background so the next
 * load (by any task) is local. `onProgress` is reported for the bytes handed to
 * the caller in either case.
 */
export async function openModelStream(url: string, onProgress?: ModelProgressCallback | null): Promise<ModelStream> {
  const cache = isCacheableModelUrl(url) ? await openModelCache() : null;

  if (cache) {
    await pendingWrites.get(url);
    const cached = await cache.match(url);
    if (cached?.body) {
      const size = contentLength(cached);
      return {
        stream: trackProgress(cached.body, size, onProgress),
        size,
        fromCache: true,
        cached: Promise.resolve(true),
      };
    }
  }

  const response = await fetchFromNetwork(url);
  const size = contentLength(response);
  let body = response.body!;
  let written: Promise<boolean> = Promise.resolve(false);

  if (cache) {
    // One branch goes to the caller, the other to disk. If the write fails
    // (quota exceeded, storage disabled) the caller's branch is unaffected.
    const [toCache, toCaller] = body.tee();
    body = toCaller;
    const headers = new Headers({
      'content-type': response.headers.get('content-type') || 'application/octet-stream',
    });
    if (size) headers.set('content-length', String(size));
    written = cache
      .put(url, new Response(toCache, { status: 200, headers }))
      .then(() => true)
      .catch((err) => {
        console.warn(`Model could not be cached (${url}):`, err);
        return false;
      })
      .finally(() => pendingWrites.delete(url));
    pendingWrites.set(url, written);
  }

  return { stream: trackProgress(body, size, onProgress), size, fromCache: false, cached: written };
}

/** Convenience wrapper for tasks that need the whole model in memory. */
export async function fetchModelBytes(url: string, onProgress?: ModelProgressCallback | null): Promise<ArrayBuffer> {
  const { stream } = await openModelStream(url, onProgress);
  return new Response(stream).arrayBuffer();
}
