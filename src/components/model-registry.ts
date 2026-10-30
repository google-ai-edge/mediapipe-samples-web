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

export const PARAKEET_TDT_0_6B_V3_ASR: HostedModel = {
  id: 'parakeet_tdt_0_6b_v3',
  name: 'Parakeet TDT 0.6B v3',
  url: 'https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3',
  fileName: 'parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm',
  description: '25 Languages · Stateful TDT',
};

export const KOKORO_82M_TTS: HostedModel = {
  id: 'kokoro_82m',
  name: 'Kokoro 82M',
  url: 'https://huggingface.co/litert-community/Kokoro-82M',
  fileName: 'kokoro_int8_gpu_fp16.litertlm',
  description: '9 Languages · 13 Voices',
};

export const PHONON_31M_TTS: HostedModel = {
  id: 'phonon_31m',
  name: 'Phonon 31M',
  url: 'https://huggingface.co/litert-community/phonon-31m-en',
  fileName: 'phonon_int8.litertlm',
  description: 'English · Ultra-compact 31M',
};

export const POCKET_TTS_100M_EN: HostedModel = {
  id: 'pocket_tts_100m_en',
  name: 'PocketTTS 100M (English)',
  url: 'https://huggingface.co/litert-community/pocket-tts',
  fileName: 'pocket_tts_en_int8.litertlm',
  description: 'English · 3 Voices',
};

export const POCKET_TTS_100M_MULTI: HostedModel = {
  id: 'pocket_tts_100m_multi',
  name: 'PocketTTS 100M (Multilingual)',
  url: 'https://huggingface.co/litert-community/pocket-tts',
  fileName: 'pocket_tts_int8.litertlm',
  description: '7 Languages · 8 Voices',
};

export const AUDIO_TRANSCRIBER_MODELS: HostedModel[] = [PARAKEET_TDT_0_6B_V3_ASR];

export const TEXT_TO_SPEECH_MODELS: HostedModel[] = [
  PHONON_31M_TTS,
  POCKET_TTS_100M_EN,
  POCKET_TTS_100M_MULTI,
  KOKORO_82M_TTS,
];

/** Direct download URL of a hosted model (the shared cache key). */
export function hostedModelDownloadUrl(model: HostedModel): string {
  return resolveModelDownloadUrl(model.url, model.fileName);
}
