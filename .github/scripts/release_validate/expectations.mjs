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
 * Ground truths for output verification.
 *
 * Each expectation states what is true about the input (a dog is in the photo,
 * the sentence is negative, the clip is speech), with tolerances wide enough to
 * hold for any correct release rather than for one package version. Positions
 * were marked by eye on the sample images, as fractions of the image size.
 * Where a value is a baseline measured from a release instead of an
 * independent truth, the comment says so.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_DIR = path.resolve(SCRIPT_DIR, '..', '..', '..');
const FIXTURE_DIR = path.join(SCRIPT_DIR, 'fixtures');
const publicFile = (name) => path.join(REPO_DIR, 'public', name);

/** MediaPipe's own speech test clip: 4.27 s, 16 kHz mono (see fixtures/README.md). */
export const SPEECH_FIXTURE = path.join(FIXTURE_DIR, 'speech_16000_hz_mono.wav');

/**
 * What is said in the speech clip. There is no published transcript for it:
 * the sentence was first taken from what the Audio Transcriber of
 * @mediapipe/tasks-audio 1.2.2 produced (the same from the uploaded file and
 * through the microphone) and was then confirmed by a person listening to the
 * clip. Set it to null to report the transcriber cases as unverified.
 */
export const SPEECH_TRANSCRIPT = "Send a text to Mike, I'll be home late tomorrow.";

/** Sentence typed into Text to Speech. At a normal speaking rate its 14 words take about 4 to 7 seconds. */
export const TTS_SENTENCE = 'The quick brown fox jumps over the lazy dog near the quiet river bank.';

/** Sentences typed into the Text Classifier, with their true sentiment. */
export const TEXT_CLASSIFIER_INPUTS = [
  { text: 'This MediaPipe release validation test is working wonderfully!', label: 'positive' },
  { text: 'This is a terrible, broken and disappointing release. I hate it.', label: 'negative' },
];

/** Sentences typed into the Language Detector, with the language they are written in. */
export const LANGUAGE_DETECTOR_INPUTS = [
  { text: 'Bonjour, ceci est un test de validation automatique.', label: 'fr' },
  { text: 'This is an automatic validation test for the new release.', label: 'en' },
  { text: 'Hola, esta es una prueba de validación automática para la nueva versión.', label: 'es' },
  { text: 'Hallo, dies ist ein automatischer Validierungstest für die neue Version.', label: 'de' },
  { text: 'これは新しいリリースの自動検証テストです。', label: 'ja' },
];

const BEACH_SENTENCE = 'I love watching sunny sunsets at the beach.';

/**
 * Sentence pairs typed into the Text Embedder. An identical pair must score
 * about 1, and the pair about the same subject must score clearly higher than
 * the unrelated pair.
 */
export const TEXT_EMBEDDER_PAIRS = [
  { key: [BEACH_SENTENCE, BEACH_SENTENCE], name: 'identical sentences', range: [0.99, 1.0001] },
  { key: [BEACH_SENTENCE, 'Warm ocean evenings are wonderful.'], name: 'related sentences', range: [-1, 1] },
  { key: [BEACH_SENTENCE, 'The quarterly tax filing deadline is next Tuesday.'], name: 'unrelated sentences', range: [-1, 1] },
];

/**
 * Sample-image pairs chosen in the Image Embedder. The same photo twice must
 * score about 1; photos of different animals must score clearly lower.
 */
export const IMAGE_EMBEDDER_PAIRS = [
  { key: ['dog.jpg', 'dog.jpg'], name: 'the same photo twice', range: [0.99, 1.0001] },
  { key: ['dog.jpg', 'cat.png'], name: 'a dog and a cat', range: [-1, 0.8] },
  { key: ['cat.png', 'elephant.png'], name: 'a cat and an elephant', range: [-1, 0.8] },
];

const asText = (value) => ({ mode: 'text', value });
const asImage = (value) => ({ mode: 'image', value });
const CAT_TEXT = 'a cute furry pet cat';
const BANANA_TEXT = 'a yellow banana';

/**
 * Inputs set in the two slots of the Universal Embedder, per mode. Images are
 * the page's sample photos, chosen from its gallery (public/images/<id>.jpg).
 * A description must score higher against the photo it describes than against
 * a photo of something else; identical inputs must score about 1. With L2
 * normalization on, every embedding has length 1.
 */
export const UNIVERSAL_EMBEDDER_PLANS = {
  'text-text-l2true': {
    l2Normalize: true,
    runs: [
      { name: 'identical sentences', key: [asText(CAT_TEXT), asText(CAT_TEXT)], range: [0.99, 1.0001] },
      { name: 'related sentences', key: [asText(CAT_TEXT), asText('an adorable fluffy kitten')], range: [-1, 1] },
      { name: 'unrelated sentences', key: [asText(CAT_TEXT), asText('the quarterly tax filing deadline')], range: [-1, 1] },
    ],
    ordered: [{ higher: 1, lower: 2, by: 0.1 }],
  },
  'text-image-l2true': {
    l2Normalize: true,
    runs: [
      { name: 'cat description and cat photo', key: [asText(CAT_TEXT), asImage('cute_cat')], range: [-1, 1] },
      { name: 'cat description and banana photo', key: [asText(CAT_TEXT), asImage('yellow_banana')], range: [-1, 1] },
      { name: 'banana description and banana photo', key: [asText(BANANA_TEXT), asImage('yellow_banana')], range: [-1, 1] },
      { name: 'banana description and cat photo', key: [asText(BANANA_TEXT), asImage('cute_cat')], range: [-1, 1] },
    ],
    ordered: [
      { higher: 0, lower: 1, by: 0.02 },
      { higher: 2, lower: 3, by: 0.02 },
    ],
  },
  'image-image-l2false': {
    l2Normalize: false,
    runs: [
      { name: 'the same photo twice', key: [asImage('cute_cat'), asImage('cute_cat')], range: [0.99, 1.0001] },
      { name: 'a cat photo and a banana photo', key: [asImage('cute_cat'), asImage('yellow_banana')], range: [-1, 0.9] },
    ],
  },
};

/**
 * Queries typed into the Semantic Retriever after its ten sample photos are
 * indexed, with the photo each one describes (public/images/<id>.jpg).
 */
export const SEMANTIC_RETRIEVER_QUERIES = [
  { query: 'cute cat', topId: 'cute_cat', topTitle: 'Cute cat' },
  { query: 'a ripe yellow banana', topId: 'yellow_banana', topTitle: 'Yellow banana' },
  { query: 'a fast red sports car', topId: 'fast_car', topTitle: 'Fast car' },
  { query: 'snow on a mountain peak', topId: 'snowy_mountain', topTitle: 'Snowy mountain' },
];
export const SEMANTIC_RETRIEVER_CORPUS_SIZE = 10;

/**
 * Support tickets typed into the Decision Maker, which triages them with its
 * built-in "Support Ticket Triage" questions: is it an urgent outage, which
 * department owns it, and how severe is it (1 to 5). The first describes a
 * production outage that locks customers out; the second is a routine billing
 * request. The outage is plainly the more severe of the two.
 */
export const DECISION_TICKETS = [
  {
    name: 'outage ticket',
    text: 'Urgent: our production database pipeline crashes with a fatal segfault and customers cannot sign in!',
    urgent: true,
    category: 'bug',
  },
  {
    name: 'billing ticket',
    text: 'We were charged twice on invoice #4821 this month, can you refund the duplicate?',
    urgent: false,
    category: 'billing',
  },
];

/** The request kind the Decision Maker sends for each tab of the page. */
const DECISION_KIND_BY_TAB = { polymorphic: 'schema', json: 'schema', boolean: 'boolean', choice: 'choice', score: 'score' };

/**
 * Sample photos checked on the ID Photo page, per rule pack. The first is a
 * compliant passport photo (public/images/id_photos): a woman facing the
 * camera with a neutral expression, eyes open, no glasses, shoulders visible,
 * on a plain white background. The others break one rule in an obvious way.
 */
export const ID_PHOTO_PLANS = {
  'preset-passport': {
    pack: 'passport_id',
    packName: 'Passport / ID',
    rules: ['full_face', 'eyes_open', 'shoulders_visible', 'no_glasses', 'framing_chest_up', 'occupancy_70_percent'],
    photos: [
      { preset: 'id_fail_glasses_smile.jpg', name: 'photo with sunglasses', compliant: false, failingRule: 'no_glasses' },
      { preset: 'passport_compliant_female.jpg', name: 'compliant photo', compliant: true },
    ],
  },
  'rule-pack-badge': {
    pack: 'badge_profile',
    packName: 'Badge / Profile',
    rules: ['full_face', 'eyes_open', 'shoulders_visible', 'framing_chest_up', 'occupancy_70_percent'],
    photos: [
      { preset: 'id_fail_eyes_closed_hair.jpg', name: 'photo with eyes closed', compliant: false, failingRule: 'eyes_open' },
      { preset: 'passport_compliant_female.jpg', name: 'compliant photo', compliant: true },
    ],
  },
};

/**
 * The Dino Game asks the model what the dino should do about what is in front
 * of it. The right answer follows from the scene alone, as the game's own
 * prompt states it: jump a ground obstacle, duck a flying one, hold the pace
 * for a distant one, sprint on a clear track. Every game starts with cacti on
 * the track; the runner advances the game until the dino has reached a cactus
 * and a bird (dropping a bird with the page's button where it offers one).
 */
export const DINO_SCENE_ACTIONS = [
  { scene: 'a clear track', pattern: /^The path ahead is clear/, action: 'SPRINT' },
  { scene: 'a ground obstacle right ahead', pattern: /^Ground obstacle right ahead/, action: 'JUMP' },
  { scene: 'a low flying obstacle right ahead', pattern: /^Low flying obstacle right ahead/, action: 'DUCK' },
  { scene: 'an obstacle in the distance', pattern: /^Obstacle spotted in the distance/, action: 'WAIT' },
];
// The same rule holds for text and image input: with image input the model sees a picture of
// the scene instead of its description, but the right action for the scene is the same.
/** Actions every Dino Game case must have decided at least once; the others are checked when they occur. */
export const DINO_REQUIRED_ACTIONS = ['JUMP', 'DUCK'];

/** Tasks whose app code publishes a result hook (see src/components/test-hooks.ts). */
export const TASKS_WITH_RESULT_HOOK = new Set([
  'object_detector',
  'face_detector',
  'face_landmarker',
  'hand_landmarker',
  'pose_landmarker',
  'holistic_landmarker',
  'image_classifier',
  'gesture_recognizer',
  'interactive_segmenter',
  'image_segmenter',
  'image_embedder',
  'text_classifier',
  'language_detector',
  'text_embedder',
  'audio_classifier',
  'audio_transcriber',
  'text_to_speech',
  'universal_embedder',
  'semantic_retriever',
  'decision_maker',
  'id_photo',
  'dino_game',
]);

const near = (x, y, tolerance) => ({ x, y, tolerance });

/**
 * public/dog.jpg (640x640): one golden retriever, sitting, facing the camera.
 * The box was marked by eye on the photo (ears to the bottom edge); 0.5 IoU is
 * the usual detection criterion.
 */
const DOG_PHOTO = {
  type: 'detections',
  objects: [{ label: 'dog', minScore: 0.5, box: { x: 0.03, y: 0.14, width: 0.76, height: 0.86 }, minIoU: 0.5 }],
};

/**
 * public/face_model.png (640x640): one woman facing the camera. Her right eye
 * is on the left of the image.
 */
const FACE = {
  noseTip: near(0.505, 0.42, 0.05),
  rightEye: near(0.425, 0.353, 0.05),
  leftEye: near(0.575, 0.347, 0.05),
  rightEyeOuter: near(0.394, 0.356, 0.05),
  leftEyeOuter: near(0.606, 0.347, 0.05),
  chin: near(0.5, 0.595, 0.05),
};

/**
 * public/pose_model.png (640x640): one woman in tree pose, facing the camera.
 * She stands on her right leg (left of the image); her left foot rests on her
 * right thigh, and her hands are pressed together in front of her chest.
 */
const TREE_POSE = {
  nose: near(0.505, 0.2, 0.05),
  leftShoulder: near(0.59, 0.3, 0.06),
  rightShoulder: near(0.42, 0.3, 0.06),
  leftAnkle: near(0.52, 0.6, 0.07),
  rightAnkle: near(0.51, 0.85, 0.06),
};
const PRAYER_HANDS_WRIST = near(0.5, 0.36, 0.08);

/**
 * public/cat.png (640x640): a cat's face and chest fill roughly two thirds of
 * the frame, centred slightly right of and below the middle; the rest is a
 * window and wall. Selecting the cat should therefore cover about two thirds of
 * the image.
 */
const CAT_SELECTED = {
  foregroundFraction: [0.45, 0.85],
  centroidWithin: { minX: 0.35, maxX: 0.7, minY: 0.4, maxY: 0.8 },
};

/**
 * Strokes drawn on the Interactive Segmenter image with the mouse, per mode,
 * and what must be true of the mask after each stroke. Coordinates are
 * fractions of the canvas. `onMask` is the share of a stroke's points that lie
 * on the selected region.
 */
export const INTERACTIVE_SEGMENTER_PLANS = {
  // A positive stroke across the eyes selects the cat.
  'image-positive': [
    {
      brush: 1,
      gesture: { line: [0.3, 0.5, 0.65, 0.5] },
      expect: { ...CAT_SELECTED, strokes: [{ brushMode: 1, onMask: [0.9, 1] }] },
    },
  ],
  // A negative stroke across the chest then removes that part of the selection.
  'image-negative': [
    {
      brush: 1,
      gesture: { line: [0.3, 0.5, 0.65, 0.5] },
      expect: { ...CAT_SELECTED, strokes: [{ brushMode: 1, onMask: [0.9, 1] }] },
    },
    {
      brush: 2,
      gesture: { line: [0.25, 0.9, 0.75, 0.9] },
      expect: {
        foregroundFraction: [0.05, 0.85],
        shrinksBy: 0.1,
        strokes: [
          { brushMode: 1, onMask: [0.9, 1] },
          { brushMode: 2, onMask: [0, 0.1] },
        ],
      },
    },
  ],
  // The cat photo is fed to the fake camera and frozen; a positive stroke across the eyes selects the cat.
  'video-freeze': [
    {
      brush: 1,
      gesture: { line: [0.35, 0.5, 0.65, 0.5] },
      expect: { ...CAT_SELECTED, strokes: [{ brushMode: 1, onMask: [0.9, 1] }] },
    },
  ],
  // A lasso drawn around the cat selects the cat.
  'image-lasso': [
    {
      brush: 3,
      gesture: { circle: [0.5, 0.55, 0.42] },
      expect: { ...CAT_SELECTED, strokes: [{ brushMode: 3 }] },
    },
  ],
};

const LOWER_CENTRE = { minX: 0.4, maxX: 0.6, minY: 0.5, maxY: 0.75 };

/**
 * What each Image Segmenter model must find, and in which photo.
 *
 * DeepLab runs on the page's built-in sample photo: a golden retriever sitting
 * in front of a plain wall, covering roughly a third of the frame, centred
 * below the middle.
 *
 * The other models are trained on people, so public/face_model.png is uploaded
 * for them: a woman's head and shoulders cover a little over half of the frame.
 * Her hair is around the upper centre (roughly a sixth of the frame), her face
 * and neck are in the middle, and her T-shirt fills the bottom quarter.
 *
 * `confidence` names the class whose confidence mask is chosen in the UI in
 * confidence mode; it must cover the same share as in the category mask.
 */
export const IMAGE_SEGMENTER_PLANS = {
  deeplab_v3: {
    upload: null,
    classes: [
      { label: 'dog', fraction: [0.2, 0.45], centroidWithin: LOWER_CENTRE },
      { label: 'background', fraction: [0.55, 0.8] },
    ],
    confidence: { label: 'dog', fraction: [0.2, 0.45], minPeak: 0.8 },
  },
  hair_segmenter: {
    upload: 'face_model.png',
    classes: [
      { label: 'hair', fraction: [0.08, 0.3], centroidWithin: { minX: 0.4, maxX: 0.6, minY: 0.2, maxY: 0.5 } },
      { label: 'background', fraction: [0.7, 0.92] },
    ],
    confidence: { label: 'hair', fraction: [0.08, 0.3], minPeak: 0.8 },
  },
  selfie_segmenter: {
    upload: 'face_model.png',
    classes: [{ label: 'selfie', fraction: [0.45, 0.7], centroidWithin: LOWER_CENTRE }],
    confidence: { label: 'selfie', fraction: [0.45, 0.7], minPeak: 0.8 },
  },
  selfie_segmenter_landscape: {
    upload: 'face_model.png',
    classes: [{ label: 'selfie', fraction: [0.45, 0.7], centroidWithin: LOWER_CENTRE }],
    confidence: { label: 'selfie', fraction: [0.45, 0.7], minPeak: 0.8 },
  },
  selfie_multiclass: {
    upload: 'face_model.png',
    classes: [
      { label: 'background', fraction: [0.3, 0.55] },
      { label: 'hair', fraction: [0.08, 0.3], centroidWithin: { minX: 0.4, maxX: 0.6, minY: 0.2, maxY: 0.5 } },
      { label: 'face-skin', fraction: [0.05, 0.2], centroidWithin: { minX: 0.4, maxX: 0.6, minY: 0.3, maxY: 0.55 } },
      { label: 'clothes', fraction: [0.15, 0.4], centroidWithin: { minX: 0.4, maxX: 0.6, minY: 0.7, maxY: 0.95 } },
      { label: 'body-skin', fraction: [0, 0.12] },
    ],
    confidence: { label: 'clothes', fraction: [0.15, 0.4], minPeak: 0.8 },
  },
};

/**
 * The webcam cases feed the task's sample photo to Chrome's fake camera (see
 * `getFakeMedia`), so they must produce what the image cases produce.
 */
const imageAndWebcam = (expectation) => (testCase) => ({ ...expectation, mode: testCase.mode === 'video' ? 'VIDEO' : 'IMAGE' });

const EXPECTATIONS = {
  object_detector: imageAndWebcam(DOG_PHOTO),

  // BlazeFace keypoints: 0 right eye, 1 left eye, 2 nose tip. The face is about a third of the image wide.
  face_detector: imageAndWebcam({
    type: 'faces',
    count: 1,
    minScore: 0.5,
    keypoints: [
      { index: 0, name: 'right eye', ...FACE.rightEye },
      { index: 1, name: 'left eye', ...FACE.leftEye },
      { index: 2, name: 'nose tip', ...FACE.noseTip },
    ],
    boxWidth: [0.25, 0.45],
    boxContains: FACE.noseTip,
  }),

  // The Face Landmarker model outputs 478 landmarks and 52 blendshapes per face.
  face_landmarker: imageAndWebcam({
    type: 'landmarks',
    groups: [
      {
        key: 'faces',
        name: 'face',
        instances: 1,
        landmarkCount: 478,
        points: { noseTip: FACE.noseTip, chin: FACE.chin, rightEyeOuter: FACE.rightEyeOuter, leftEyeOuter: FACE.leftEyeOuter },
      },
    ],
    equals: [
      { key: 'blendshapeCounts', value: [52] },
      { key: 'matrixCount', value: 1 },
    ],
  }),

  // public/hand_model.png: one hand pointing left with the index finger; the wrist is at the lower right.
  hand_landmarker: imageAndWebcam({
    type: 'landmarks',
    groups: [
      {
        key: 'hands',
        name: 'hand',
        instances: 1,
        landmarkCount: 21,
        points: { indexTip: near(0.15, 0.33, 0.06), wrist: near(0.83, 0.66, 0.1) },
      },
    ],
  }),

  pose_landmarker: imageAndWebcam({
    type: 'landmarks',
    groups: [{ key: 'poses', name: 'pose', instances: 1, landmarkCount: 33, points: TREE_POSE }],
  }),

  // Same photo as the Pose Landmarker; the face landmarks must sit on her head and the hands at her chest.
  holistic_landmarker: imageAndWebcam({
    type: 'landmarks',
    groups: [
      { key: 'pose', name: 'pose', instances: 1, landmarkCount: 33, points: TREE_POSE },
      { key: 'face', name: 'face', instances: 1, landmarkCount: 478, points: { noseTip: TREE_POSE.nose } },
      { key: 'leftHand', name: 'left hand', instances: 1, landmarkCount: 21, points: { wrist: PRAYER_HANDS_WRIST } },
      { key: 'rightHand', name: 'right hand', instances: 1, landmarkCount: 21, points: { wrist: PRAYER_HANDS_WRIST } },
    ],
  }),

  // public/dog.jpg is a golden retriever, which is an ImageNet class.
  image_classifier: imageAndWebcam({ type: 'imageClassification', label: 'golden retriever', minScore: 0.3 }),

  // public/thumbs_up.png: one hand giving a thumbs up; the thumb tip is at the top centre.
  gesture_recognizer: imageAndWebcam({
    type: 'landmarks',
    groups: [
      {
        key: 'hands',
        name: 'hand',
        instances: 1,
        landmarkCount: 21,
        fields: { gesture: 'Thumb_Up' },
        minScore: 0.5,
        points: { thumbTip: near(0.49, 0.16, 0.06) },
      },
    ],
    gestureRowInUi: true,
  }),

  interactive_segmenter: (testCase) => {
    const plan = INTERACTIVE_SEGMENTER_PLANS[testCase.mode];
    return { type: 'maskSteps', steps: plan.map((step) => step.expect) };
  },

  image_segmenter: (testCase) => {
    const plan = IMAGE_SEGMENTER_PLANS[testCase.modelId];
    if (!plan) return null;
    const confidenceMode = testCase.mode === 'image-confidence';
    return {
      type: 'segmentation',
      mode: testCase.mode === 'video' ? 'VIDEO' : 'IMAGE',
      outputType: confidenceMode ? 'CONFIDENCE_MASKS' : 'CATEGORY_MASK',
      classes: plan.classes,
      maxOther: 0.05,
      confidence: confidenceMode ? plan.confidence : null,
    };
  },

  image_embedder: () => ({ type: 'similarity', keyField: 'images', runs: IMAGE_EMBEDDER_PAIRS, sameDimensions: true }),

  text_classifier: () => ({ type: 'textLabels', inputs: TEXT_CLASSIFIER_INPUTS }),

  language_detector: () => ({ type: 'textLabels', inputs: LANGUAGE_DETECTOR_INPUTS }),

  // "related" (index 1) must score at least 0.1 above "unrelated" (index 2).
  text_embedder: () => ({
    type: 'similarity',
    keyField: 'texts',
    runs: TEXT_EMBEDDER_PAIRS,
    ordered: [{ higher: 1, lower: 2, by: 0.1 }],
  }),

  audio_classifier: (testCase) => {
    if (testCase.mode === 'file') {
      // YAMNet classifies 0.975 s windows, so the 4.27 s clip has 4 full windows of speech.
      return { type: 'audioClassification', mode: 'AUDIO_CLIPS', label: 'Speech', minScore: 0.5, fullWindows: 4 };
    }
    // The speech clip is fed to Chrome's fake microphone on a loop. The page classifies short
    // chunks as they arrive; chunks that fall in a pause may be inconclusive, but nothing may
    // be labelled as anything other than speech.
    return { type: 'audioStream', mode: 'AUDIO_STREAM', label: 'Speech', minScore: 0.5, minMatches: 5, minFraction: 0.9 };
  },
};

EXPECTATIONS.audio_transcriber = (testCase) => {
  if (!SPEECH_TRANSCRIPT) return { unverified: 'the reference transcript of the speech clip has not been confirmed yet' };
  // An uploaded file is transcribed completely, so nearly every word must match. Through the
  // microphone the clip plays on a loop and recording starts and stops mid-sentence, so most of
  // its words must be heard, in order.
  return testCase.mode === 'file'
    ? { type: 'transcript', mode: 'AUDIO_CLIPS', reference: SPEECH_TRANSCRIPT, maxWordErrorRate: 0.2 }
    : { type: 'transcript', mode: 'AUDIO_STREAM', reference: SPEECH_TRANSCRIPT, minWordRecall: 0.6 };
};

// Synthesized speech is checked as audio: a plausible length for the sentence (generously, 2.5 to
// 12 seconds), audible, not clipped, and free of invalid samples.
EXPECTATIONS.text_to_speech = (testCase) => ({
  type: 'speech',
  mode: testCase.mode === 'stream' ? 'STREAM' : 'SYNTHESIZE',
  text: TTS_SENTENCE,
  durationSec: [2.5, 12],
  sampleRateHz: [8000, 48000],
  minRms: 0.005,
  maxPeak: 1.5,
});

EXPECTATIONS.universal_embedder = (testCase) => {
  const plan = UNIVERSAL_EMBEDDER_PLANS[testCase.mode];
  return {
    type: 'similarity',
    keyField: 'inputs',
    runs: plan.runs,
    ordered: plan.ordered || [],
    sameDimensions: true,
    l2Normalize: plan.l2Normalize,
  };
};

EXPECTATIONS.semantic_retriever = () => ({
  type: 'retrieval',
  queries: SEMANTIC_RETRIEVER_QUERIES,
  corpusSize: SEMANTIC_RETRIEVER_CORPUS_SIZE,
});

// On the 1-to-5 severity scale the outage must score at least one point above the billing request.
EXPECTATIONS.decision_maker = (testCase) => ({
  type: 'decisions',
  kind: DECISION_KIND_BY_TAB[testCase.mode],
  tickets: DECISION_TICKETS,
  minSeverityGap: 1,
});

EXPECTATIONS.id_photo = (testCase) => ({ type: 'photoCompliance', ...ID_PHOTO_PLANS[testCase.mode] });

EXPECTATIONS.dino_game = (testCase) => ({
  type: 'gameDecisions',
  inputMode: testCase.mode.startsWith('image') ? 'image' : 'text',
  sceneActions: DINO_SCENE_ACTIONS,
  requiredActions: DINO_REQUIRED_ACTIONS,
});

/** The expectation for a matrix case, or null when none is defined yet. */
export function getExpectation(testCase) {
  const build = EXPECTATIONS[testCase.task];
  return build ? build(testCase) : null;
}

const image = (name, label) => ({ kind: 'image', label: label || name, path: publicFile(name) });
const text = (value, label) => ({ kind: 'text', label, value });
const MIC_INPUT = { kind: 'audio', label: "Speech clip, played on a loop by Chrome's fake microphone", path: SPEECH_FIXTURE };

/** The sample photo built into the Image Segmenter page as a data URL, for the report. */
function segmenterSamplePhoto() {
  const label = 'Sample photo built into the page: a golden retriever';
  const template = fs.readFileSync(path.join(REPO_DIR, 'src', 'templates', 'image-segmenter.html'), 'utf8');
  const match = template.match(/src="data:image\/jpeg;base64,([^"]+)"/);
  if (!match) return text('The sample photo built into the page', 'Sample photo');
  return { kind: 'image', label, name: 'image-segmenter-sample.jpg', bytes: Buffer.from(match[1], 'base64') };
}

/**
 * What Chrome's fake camera and microphone play for a case, so that the webcam
 * and microphone modes see a known input: the photo the task's image mode
 * uses, and the speech clip. `video` is `{ name, path }` or `{ name, bytes }`.
 */
export function getFakeMedia(testCase) {
  if (VISION_SAMPLES[testCase.task]) {
    const file = VISION_SAMPLES[testCase.task][0];
    return { video: { name: file, path: publicFile(file) }, audio: null };
  }
  if (testCase.task === 'interactive_segmenter') return { video: { name: 'cat.png', path: publicFile('cat.png') }, audio: null };
  if (testCase.task === 'image_segmenter') {
    const upload = IMAGE_SEGMENTER_PLANS[testCase.modelId]?.upload;
    if (upload) return { video: { name: upload, path: publicFile(upload) }, audio: null };
    const sample = segmenterSamplePhoto();
    return { video: sample.bytes ? { name: sample.name, bytes: sample.bytes } : null, audio: null };
  }
  if (testCase.domain === 'audio') return { video: null, audio: SPEECH_FIXTURE };
  return { video: null, audio: null };
}

/** The photo fed to the fake camera, as an input of the report's review mode. */
function webcamInput(testCase) {
  const video = getFakeMedia(testCase).video;
  const label = "Photo played by Chrome's fake camera";
  if (!video) return text("Chrome's built-in fake camera (a moving test pattern)", 'Webcam');
  return video.bytes ? { kind: 'image', label, name: video.name, bytes: video.bytes } : { kind: 'image', label, path: video.path };
}

const transcriptInput = () => (SPEECH_TRANSCRIPT ? [text(SPEECH_TRANSCRIPT, 'Reference transcript')] : []);

/** The sample image a vision task shows by default, and what it depicts. */
const VISION_SAMPLES = {
  object_detector: ['dog.jpg', 'a golden retriever', 'Object Detector finds the dog and draws a box around it'],
  face_detector: ['face_model.png', 'a woman facing the camera', 'Face Detector finds her face and its eyes and nose'],
  face_landmarker: ['face_model.png', 'a woman facing the camera', 'Face Landmarker places its 478-point mesh on her face'],
  hand_landmarker: ['hand_model.png', 'a hand pointing left', 'Hand Landmarker places its 21 landmarks on the hand'],
  pose_landmarker: ['pose_model.png', 'a woman in tree pose', 'Pose Landmarker places its 33 landmarks on her body'],
  holistic_landmarker: ['pose_model.png', 'a woman in tree pose', 'Holistic Landmarker finds her pose, face and both hands'],
  image_classifier: ['dog.jpg', 'a golden retriever', 'Image Classifier labels the photo as a golden retriever'],
  gesture_recognizer: ['thumbs_up.png', 'a thumbs-up hand', 'Gesture Recognizer recognizes the thumbs up'],
};

const INTERACTIVE_STEPS = {
  'image-positive': 'A positive stroke is drawn across the cat\'s eyes with the mouse. The cat must be selected.',
  'image-negative':
    'A positive stroke is drawn across the cat\'s eyes, then a negative stroke across its chest. The chest must be removed from the selection.',
  'image-lasso': 'A lasso is drawn around the cat with the mouse. The cat must be selected.',
  'video-freeze':
    "The webcam is started while Chrome's fake camera plays the cat photo, the frame is frozen, and a positive stroke is drawn across the cat's eyes. The cat must be selected.",
};

/**
 * A plain-language description of a case and the inputs it uses, for the review
 * mode of the report. Input files are given as absolute paths; the runner
 * copies them into the report.
 */
export function describeCase(testCase) {
  const setup = `Model "${testCase.modelId}" is selected with the ${testCase.delegate} delegate through the page controls.`;
  const webcam = testCase.mode === 'video';

  if (VISION_SAMPLES[testCase.task]) {
    const [file, subject, outcome] = VISION_SAMPLES[testCase.task];
    if (webcam) {
      return {
        description: `${setup} The view is switched to Webcam and the camera is enabled. Chrome's fake camera plays the photo of ${subject}, so the task must find in the live frames what it finds in the photo.`,
        inputs: [webcamInput(testCase)],
      };
    }
    return {
      description: `${setup} The page runs the task on its sample photo of ${subject}. ${outcome}.`,
      inputs: [image(file, `Sample photo: ${subject}`)],
    };
  }

  switch (testCase.task) {
    case 'interactive_segmenter':
      return {
        description: `${setup} ${INTERACTIVE_STEPS[testCase.mode]}`,
        inputs: [testCase.mode === 'video-freeze' ? webcamInput(testCase) : image('cat.png', 'Sample photo: a cat')],
      };
    case 'image_segmenter': {
      const plan = IMAGE_SEGMENTER_PLANS[testCase.modelId];
      if (testCase.mode === 'video' || !plan) {
        return {
          description: `${setup} The view is switched to Webcam and the camera is enabled. Chrome's fake camera plays the photo, so the mask of the live frames must contain ${(plan?.classes || []).map((c) => `"${c.label}"`).join(', ')} in the share each really occupies.`,
          inputs: [webcamInput(testCase)],
        };
      }
      const photo = plan.upload
        ? `The photo of a woman's head and shoulders is uploaded through the image upload control.`
        : `The page's built-in sample photo of a golden retriever is used.`;
      const output =
        testCase.mode === 'image-confidence'
          ? `The output type is switched to Confidence Masks and the class "${plan.confidence.label}" is chosen. Its confidence mask must cover the share of the photo that class really occupies.`
          : `The category mask must contain ${plan.classes.map((c) => `"${c.label}"`).join(', ')} in the share of the photo each really occupies, and no other class.`;
      return {
        description: `${setup} ${photo} ${output}`,
        inputs: plan.upload
          ? [image(plan.upload, 'Uploaded photo: a woman facing the camera')]
          : [segmenterSamplePhoto()],
      };
    }
    case 'image_embedder':
      return {
        description: `${setup} Pairs of sample photos are chosen with the sample buttons, and the cosine similarity of each pair is read. ${IMAGE_EMBEDDER_PAIRS.map((p) => p.name).join('; ')}.`,
        inputs: [image('dog.jpg', 'Dog'), image('cat.png', 'Cat'), image('elephant.png', 'Elephant')],
      };
    case 'text_classifier':
      return {
        description: `${setup} Each sentence is typed into the text box and Classify is clicked. The sentiment of each must be correct.`,
        inputs: TEXT_CLASSIFIER_INPUTS.map((input) => text(input.text, `Expected: ${input.label}`)),
      };
    case 'language_detector':
      return {
        description: `${setup} Each sentence is typed into the text box and Detect Language is clicked. The language of each must be correct.`,
        inputs: LANGUAGE_DETECTOR_INPUTS.map((input) => text(input.text, `Expected: ${input.label}`)),
      };
    case 'text_embedder':
      return {
        description: `${setup} Each pair of sentences is typed into the two text boxes and Compute Similarity is clicked. Identical sentences must score about 1, and related sentences must score higher than unrelated ones.`,
        inputs: TEXT_EMBEDDER_PAIRS.map((pair) => text(`${pair.key[0]}  |  ${pair.key[1]}`, pair.name)),
      };
    case 'audio_classifier':
      if (testCase.mode === 'file') {
        return {
          description: `${setup} The view is switched to Audio File, the speech clip is uploaded and Run Classification is clicked. Every full window must be labelled Speech.`,
          inputs: [{ kind: 'audio', label: 'Speech clip (4.27 s)', path: SPEECH_FIXTURE }],
        };
      }
      return {
        description: `${setup} Recording is started and runs for a few seconds while Chrome's fake microphone plays the speech clip, then it is stopped. The live results must be Speech.`,
        inputs: [MIC_INPUT],
      };
    case 'universal_embedder': {
      const plan = UNIVERSAL_EMBEDDER_PLANS[testCase.mode];
      const describeInput = (input) => (input.mode === 'text' ? `"${input.value}"` : `photo ${input.value}`);
      const photos = [...new Set(plan.runs.flatMap((run) => run.key).filter((i) => i.mode === 'image').map((i) => i.value))];
      return {
        description: `${setup} L2 normalization is set to ${plan.l2Normalize}. For each pair, the two slots are set to text or to a sample photo and Compare is clicked. Identical inputs must score about 1, and a description must score higher against the photo or sentence it matches.`,
        inputs: [
          ...plan.runs.map((run) => text(`${describeInput(run.key[0])}  |  ${describeInput(run.key[1])}`, run.name)),
          ...photos.map((id) => ({ kind: 'image', label: `Sample photo ${id}`, path: publicFile(`images/${id}.jpg`) })),
        ],
      };
    }
    case 'semantic_retriever':
      return {
        description: `${setup} The ten sample photos are indexed with the Index button, then each query is typed and searched. The top result must be the photo the query describes.`,
        inputs: [
          ...SEMANTIC_RETRIEVER_QUERIES.map((q) => text(q.query, `Expected top result: ${q.topTitle}`)),
          ...SEMANTIC_RETRIEVER_QUERIES.map((q) => ({ kind: 'image', label: q.topTitle, path: publicFile(`images/${q.topId}.jpg`) })),
        ],
      };
    case 'decision_maker':
      return {
        description: `${setup} The "${testCase.mode}" tab is opened. Each support ticket is typed into the input box and Evaluate is clicked. The outage ticket must be judged urgent, a bug and more severe; the billing ticket not urgent, billing and less severe (as far as this tab reports them).`,
        inputs: DECISION_TICKETS.map((ticket) => text(ticket.text, `${ticket.name}: urgent=${ticket.urgent}, department=${ticket.category}`)),
      };
    case 'id_photo': {
      const plan = ID_PHOTO_PLANS[testCase.mode];
      return {
        description: `${setup} The "${plan.packName}" rule pack is chosen, then each sample photo is selected from the photo strip. The compliant photo must pass every rule of the pack; the other must fail the rule it obviously breaks.`,
        inputs: plan.photos.map((photo) => ({
          kind: 'image',
          label: photo.compliant ? 'Compliant photo: must pass every rule' : `${photo.name}: must fail "${photo.failingRule}"`,
          path: publicFile(`images/id_photos/${photo.preset}`),
        })),
      };
    }
    case 'dino_game':
      return {
        description: `${setup} The game is set to ${testCase.mode.startsWith('image') ? 'image' : 'text'} input${testCase.mode.endsWith('versus') ? ' and versus play' : ''}. The game is advanced with the Next Step button until the dino has reached a cactus and a bird${testCase.mode.endsWith('versus') ? '' : ' (a bird is dropped with the page\'s Bird button)'}. Every decision must fit its scene: jump a ground obstacle, duck a flying one, hold the pace for a distant one, sprint on a clear track.`,
        inputs: DINO_SCENE_ACTIONS.map((entry) => text(`${entry.scene}`, `Expected: ${entry.action}`)),
      };
    case 'audio_transcriber':
      if (testCase.mode === 'file') {
        return {
          description: `${setup} The view is switched to Audio File, the speech clip is uploaded and Transcribe Audio is clicked. The transcript must match what is said in the clip.`,
          inputs: [{ kind: 'audio', label: 'Speech clip (4.27 s)', path: SPEECH_FIXTURE }, ...transcriptInput()],
        };
      }
      return {
        description: `${setup} Recording is started and runs for a few seconds while Chrome's fake microphone plays the speech clip, then it is stopped. The transcript must contain most of what is said in the clip.`,
        inputs: [MIC_INPUT, ...transcriptInput()],
      };
    case 'text_to_speech':
      return {
        description: `${setup} The sentence is typed into the text box and ${testCase.mode === 'stream' ? 'Stream' : 'Speak'} is clicked. The synthesized audio must have a plausible length for the sentence and be audible, unclipped and free of invalid samples.`,
        inputs: [text(TTS_SENTENCE, 'Sentence to speak')],
      };
    default:
      return {
        description: `${setup} Mode "${testCase.mode}" is exercised through the page controls.`,
        inputs: [],
      };
  }
}
