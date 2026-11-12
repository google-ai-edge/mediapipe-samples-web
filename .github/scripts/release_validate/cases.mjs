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

const RETRIEVAL_440M_URL = 'https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm';
const RETRIEVAL_740M_URL = 'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm';

/**
 * Complete validation inventory covering every task, model, mode, and delegate
 * exposed in mediapipe-samples-web.
 */
export const TASK_DEFINITIONS = [
  // --- Vision (11 tasks) ---
  {
    domain: 'vision',
    task: 'object_detector',
    route: '#/vision/object_detector',
    heading: 'Object Detection',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [
      { id: 'efficientdet_lite0', selectValue: 'efficientdet_lite0', large: false },
      { id: 'efficientdet_lite2', selectValue: 'efficientdet_lite2', large: false },
      { id: 'ssd_mobilenet_v2', selectValue: 'ssd_mobilenet_v2', large: false },
    ],
  },
  {
    domain: 'vision',
    task: 'face_detector',
    route: '#/vision/face_detector',
    heading: 'Face Detection',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [
      { id: 'blaze_face_short_range', selectValue: 'blaze_face_short_range', large: false },
      { id: 'blaze_face_full_range', selectValue: 'blaze_face_full_range', large: false },
    ],
  },
  {
    domain: 'vision',
    task: 'face_landmarker',
    route: '#/vision/face_landmarker',
    heading: 'Face Landmarker',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [{ id: 'face_landmarker', selectValue: 'face_landmarker', large: false }],
  },
  {
    domain: 'vision',
    task: 'hand_landmarker',
    route: '#/vision/hand_landmarker',
    heading: 'Hand Landmarker',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [{ id: 'hand_landmarker', selectValue: 'hand_landmarker', large: false }],
  },
  {
    domain: 'vision',
    task: 'pose_landmarker',
    route: '#/vision/pose_landmarker',
    heading: 'Pose Landmarker',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [
      { id: 'pose_landmarker_lite', selectValue: 'pose_landmarker_lite', large: false },
      { id: 'pose_landmarker_full', selectValue: 'pose_landmarker_full', large: false },
      { id: 'pose_landmarker_heavy', selectValue: 'pose_landmarker_heavy', large: false },
    ],
  },
  {
    domain: 'vision',
    task: 'holistic_landmarker',
    route: '#/vision/holistic_landmarker',
    heading: 'Holistic Landmarker',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [{ id: 'holistic_landmarker_lite', selectValue: 'holistic_landmarker_lite', large: false }],
  },
  {
    domain: 'vision',
    task: 'image_classifier',
    route: '#/vision/image_classifier',
    heading: 'Image Classifier',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [
      { id: 'efficientnet_lite0', selectValue: 'efficientnet_lite0', large: false },
      { id: 'efficientnet_lite2', selectValue: 'efficientnet_lite2', large: false },
    ],
  },
  {
    domain: 'vision',
    task: 'gesture_recognizer',
    route: '#/vision/gesture_recognizer',
    heading: 'Gesture Recognizer',
    kind: 'vision',
    delegates: ['CPU', 'GPU'],
    modes: ['image', 'video'],
    models: [{ id: 'gesture_recognizer', selectValue: 'gesture_recognizer', large: false }],
  },
  {
    domain: 'vision',
    task: 'interactive_segmenter',
    route: '#/vision/interactive_segmenter',
    heading: 'Interactive Segmenter',
    kind: 'interactive_segmenter',
    delegates: ['CPU', 'GPU'],
    modes: ['image-positive', 'image-negative', 'image-lasso', 'video-freeze'],
    models: [{ id: 'interactive_segmentation', selectValue: 'interactive_segmentation', large: false }],
  },
  {
    domain: 'vision',
    task: 'image_segmenter',
    route: '#/vision/image_segmenter',
    heading: 'Image Segmentation',
    kind: 'image_segmenter',
    delegates: ['CPU', 'GPU'],
    modes: ['image-category', 'image-confidence', 'video'],
    models: [
      { id: 'deeplab_v3', selectValue: 'deeplab_v3', large: false },
      { id: 'hair_segmenter', selectValue: 'hair_segmenter', large: false },
      { id: 'selfie_segmenter', selectValue: 'selfie_segmenter', large: false },
      { id: 'selfie_segmenter_landscape', selectValue: 'selfie_segmenter_landscape', large: false },
      { id: 'selfie_multiclass', selectValue: 'selfie_multiclass', large: false },
    ],
  },
  {
    domain: 'vision',
    task: 'image_embedder',
    route: '#/vision/image_embedder',
    heading: 'Image Embedding',
    kind: 'image_embedder',
    delegates: ['CPU', 'GPU'],
    modes: ['dual-image'],
    models: [
      { id: 'mobilenet_v3_small', selectValue: 'mobilenet_v3_small', large: false },
      { id: 'mobilenet_v3_large', selectValue: 'mobilenet_v3_large', large: false },
    ],
  },

  // --- Audio (3 tasks) ---
  {
    domain: 'audio',
    task: 'audio_classifier',
    route: '#/audio/audio_classifier',
    heading: 'Audio Classifier',
    kind: 'audio_classifier',
    delegates: ['CPU', 'GPU'],
    modes: ['mic', 'file'],
    models: [{ id: 'yamnet', selectValue: 'yamnet', large: false }],
  },
  {
    domain: 'audio',
    task: 'audio_transcriber',
    route: '#/audio/audio_transcriber',
    heading: 'Audio Transcriber',
    kind: 'audio_transcriber',
    delegates: ['CPU', 'GPU'],
    modes: ['mic', 'file'],
    models: [{ id: 'parakeet_tdt_v3', selectValue: 'parakeet_tdt_v3', large: true }],
  },
  {
    domain: 'audio',
    task: 'text_to_speech',
    route: '#/audio/text_to_speech',
    heading: 'Text to Speech',
    kind: 'text_to_speech',
    delegates: ['CPU', 'GPU'],
    modes: ['speak', 'stream'],
    models: [
      { id: 'kokoro_82m', selectValue: 'kokoro_82m', large: true },
      { id: 'kokoro_82m_cpu', selectValue: 'kokoro_82m_cpu', large: true },
      { id: 'pocket_tts_en', selectValue: 'pocket_tts_en', large: true },
      { id: 'pocket_tts', selectValue: 'pocket_tts', large: true },
      { id: 'phonon_31m_en', selectValue: 'phonon_31m_en', large: true },
    ],
  },

  // --- Text (3 tasks) ---
  {
    domain: 'text',
    task: 'text_classifier',
    route: '#/text/text_classifier',
    heading: 'Text Classification',
    kind: 'text_classifier',
    delegates: ['CPU', 'GPU'],
    modes: ['classify'],
    models: [
      { id: 'bert_classifier', selectValue: 'bert_classifier', large: false },
      { id: 'average_word_classifier', selectValue: 'average_word_classifier', large: false },
    ],
  },
  {
    domain: 'text',
    task: 'language_detector',
    route: '#/text/language_detector',
    heading: 'Language Detection',
    kind: 'language_detector',
    delegates: ['CPU', 'GPU'],
    modes: ['detect'],
    models: [{ id: 'language_detector', selectValue: 'language_detector', large: false }],
  },
  {
    domain: 'text',
    task: 'text_embedder',
    route: '#/text/text_embedder',
    heading: 'Text Embedding',
    kind: 'text_embedder',
    delegates: ['CPU', 'GPU'],
    modes: ['embed'],
    models: [
      { id: 'universal_sentence_encoder', selectValue: 'universal_sentence_encoder', large: false },
      { id: 'embedding_gemma_v1', selectValue: 'embedding_gemma_v1', large: true },
    ],
  },

  // --- Retrieval (2 tasks, WebGPU) ---
  {
    domain: 'retrieval',
    task: 'universal_embedder',
    route: '#/retrieval/universal_embedder',
    heading: 'Universal Embedder',
    kind: 'universal_embedder',
    delegates: ['GPU'],
    modes: ['text-image-l2true', 'text-text-l2true', 'image-image-l2false'],
    models: [
      { id: 'embeddinggemma2_text_vision_440m', selectValue: RETRIEVAL_440M_URL, large: true },
      { id: 'embeddinggemma2_740m', selectValue: RETRIEVAL_740M_URL, large: true },
    ],
  },
  {
    domain: 'retrieval',
    task: 'semantic_retriever',
    route: '#/retrieval/semantic_retriever',
    heading: 'Semantic Retriever',
    kind: 'semantic_retriever',
    delegates: ['GPU'],
    modes: ['index-and-search'],
    models: [
      { id: 'embeddinggemma2_text_vision_440m', selectValue: RETRIEVAL_440M_URL, large: true },
      { id: 'embeddinggemma2_740m', selectValue: RETRIEVAL_740M_URL, large: true },
    ],
  },

  // --- Decision (3 tasks, WebGPU) ---
  {
    domain: 'decision',
    task: 'decision_maker',
    route: '#/decision/decision_maker',
    heading: 'Decision Maker',
    kind: 'decision_maker',
    delegates: ['GPU'],
    modes: ['polymorphic', 'boolean', 'choice', 'score', 'json'],
    models: [
      { id: 'embeddinggemma2_270m', selectValue: 'embeddinggemma2_270m', large: true },
      { id: 'embeddinggemma2_text_vision_440m', selectValue: 'embeddinggemma2_text_vision_440m', large: true },
      { id: 'embeddinggemma2_740m', selectValue: 'embeddinggemma2_740m', large: true },
    ],
  },
  {
    domain: 'decision',
    task: 'id_photo',
    route: '#/decision/id_photo',
    heading: 'Decision Maker - ID Photo',
    kind: 'id_photo',
    delegates: ['GPU'],
    modes: ['preset-passport', 'rule-pack-badge'],
    models: [
      { id: 'embeddinggemma2_text_vision_440m', selectValue: 'embeddinggemma2_text_vision_440m', large: true },
      { id: 'embeddinggemma2_740m', selectValue: 'embeddinggemma2_740m', large: true },
    ],
  },
  {
    domain: 'decision',
    task: 'dino_game',
    route: '#/decision/dino_game',
    heading: 'Decision Maker - Dino Game',
    kind: 'dino_game',
    delegates: ['GPU'],
    modes: ['text-tap', 'text-versus', 'image-tap'],
    models: [
      { id: 'embeddinggemma2_270m', selectValue: 'embeddinggemma2_270m', large: true, supportsVision: false },
      {
        id: 'embeddinggemma2_text_vision_440m',
        selectValue: 'embeddinggemma2_text_vision_440m',
        large: true,
        supportsVision: true,
      },
      { id: 'embeddinggemma2_740m', selectValue: 'embeddinggemma2_740m', large: true, supportsVision: true },
    ],
  },
];

/**
 * Expands TASK_DEFINITIONS into individual runnable matrix test cases, ordered
 * so modes for the same (task, model, delegate) execute consecutively and
 * share the cached model cleanly.
 */
export function buildValidationMatrix({ domain = '', taskFilter = '', includeLargeModels = true } = {}) {
  const filters = taskFilter
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);

  const cases = [];
  for (const def of TASK_DEFINITIONS) {
    if (domain && def.domain !== domain) continue;
    const taskSlug = `${def.domain}/${def.task}`.toLowerCase();
    if (filters.length > 0 && !filters.some((f) => taskSlug.includes(f) || def.domain === f)) {
      continue;
    }

    for (const model of def.models) {
      if (!includeLargeModels && model.large) continue;
      for (const delegate of def.delegates) {
        for (const mode of def.modes) {
          if (def.kind === 'dino_game' && mode === 'image-tap' && !model.supportsVision) {
            continue;
          }
          const id = `${def.domain}__${def.task}__${model.id}__${mode}__${delegate}`;
          cases.push({
            id,
            domain: def.domain,
            task: def.task,
            route: def.route,
            heading: def.heading,
            kind: def.kind,
            modelId: model.id,
            modelSelectValue: model.selectValue,
            largeModel: Boolean(model.large),
            delegate,
            mode,
          });
        }
      }
    }
  }
  return cases;
}
