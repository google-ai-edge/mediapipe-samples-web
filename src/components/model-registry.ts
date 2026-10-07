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

import { resolveModelDownloadUrl } from './model-cache';

/**
 * Hosted LiteRT-LM models that more than one demo can use. Defining them once
 * guarantees every demo downloads the exact same URL, which is what the shared
 * model cache keys on: a model loaded for the Universal Embedder is then
 * instantly available to the Semantic Retriever and the Decision Maker.
 */
export interface HostedModel {
  id: string;
  name: string;
  /** Model page URL; resolved to the direct download via `hostedModelDownloadUrl`. */
  url: string;
  fileName: string;
  description?: string;
}

export const EMBEDDING_GEMMA_2_TEXT_270M: HostedModel = {
  id: 'embeddinggemma-2-text-270m',
  name: 'EmbeddingGemma-2 Text 270M',
  url: 'https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm',
  fileName: 'embeddinggemma-2-text-270m.litertlm',
  description: 'Text',
};

export const EMBEDDING_GEMMA_2_TEXT_VISION_440M: HostedModel = {
  id: 'embeddinggemma-2-text-vision-440m',
  name: 'EmbeddingGemma-2 Text-Vision 440M',
  url: 'https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm',
  fileName: 'embeddinggemma-2-text-vision-440m.litertlm',
  description: 'Multimodal (Text & Image)',
};

export const EMBEDDING_GEMMA_2_740M: HostedModel = {
  id: 'embeddinggemma-2-740m',
  name: 'EmbeddingGemma-2 740M',
  url: 'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm',
  fileName: 'embeddinggemma-2-740m.litertlm',
  description: 'Multimodal (Text & Image & Audio)',
};

/** Direct download URL of a hosted model (the shared cache key). */
export function hostedModelDownloadUrl(model: HostedModel): string {
  return resolveModelDownloadUrl(model.url, model.fileName);
}
