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
 * Vision tab for the Decision Maker task: an ID & Passport Photo Compliance
 * Checker that evaluates biometric and visual rules on preset portraits,
 * uploaded photos, or a live webcam stream, with use-case rule packs
 * (Full Biometric, Passport / ID, Badge / Profile, Custom), plain-English
 * custom requirement authoring, live auto-snap, JSON schema inspection, and
 * interactive positive/negative reference examples per rule.
 */

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import template from '../templates/decision-maker-vision.html?raw';
import { DECISION_MODELS } from '../components/decision-runtime';
import { fetchModelBytes } from '../components/model-cache';

export interface PassportRuleOption {
  label: string;
  description: string;
}

export interface PassportRule {
  id: string;
  num: number;
  icon: string;
  shortTitle: string;
  title: string;
  statement?: string;
  normalizePrior?: boolean;
  prompt: string;
  options: PassportRuleOption[];
  optionDisplay: Record<string, string>;
  enabled?: boolean;
  isCustom?: boolean;
}

export interface PhotoRulePack {
  id: string;
  name: string;
  subtitle: string;
  enabledRuleIds: string[];
  isUserSaved?: boolean;
}

export type RuleStatusTier = 'pass' | 'borderline' | 'fail' | 'skipped';

export interface FaceBiometricInfo {
  hasFace: boolean;
  minX?: number;
  minY?: number;
  maxX?: number;
  maxY?: number;
  centerX?: number;
  centerY?: number;
  faceW?: number;
  faceH?: number;
  yawAsymmetry?: number;
  eyeRoll?: number;
  gazeHoriz?: number;
  blink?: number;
  smile?: number;
  jawOpen?: number;
  foreheadDarkDiff?: number;
  bridgeContrast?: number;
}

export interface PhotoBiometrics {
  bgLum: number;
  bgVar: number;
  bgSat?: number;
  isPlainWhiteBg: boolean;
  fgRatio: number;
  bottomShoulderWidthRatio: number;
  faceInfo: FaceBiometricInfo;
}

export interface RawRuleVisionScore {
  passProb: number;
  topLabel: string;
  failLabel: string;
}

export interface PhotoPreset {
  id: string;
  title: string;
  sub: string;
  isCompliant: boolean | null;
  evaluated?: boolean;
  url: string;
  groundTruth?: Record<string, boolean>;
  biometrics: PhotoBiometrics;
  rawEg2Vision: Record<string, RawRuleVisionScore>;
}

export interface ExemplarEntry {
  id: string;
  name: string;
  url: string;
  rawPassProb: number;
  logitVal: number;
  logitVec: number[];
  enabled: boolean;
  isDefault?: boolean;
}

export interface RuleEvalResult {
  pending?: boolean;
  enabled: boolean;
  status: RuleStatusTier;
  passed: boolean;
  prob: number;
  rawPassProb: number;
  calPassProb: number;
  chosenOptionLabel: string;
  reasonText: string;
}

export type VisionAskFn = (
  input: { text?: string; imageBytes?: Uint8Array },
  kind: 'schema',
  question: object
) => Promise<any>;

export interface VisionModelState {
  ready: boolean;
  loading: boolean;
  label: string;
  loadedValue?: string | null;
  hasCustomModel?: boolean;
  error?: string;
}

/**
 * Synthesizes positive and negative visual prototype descriptions from a
 * plain-English requirement statement, including automatic negation handling
 * (e.g., "No headphones or earbuds" -> bare ears vs wearing headphones).
 */
export function buildAnchorsForStatement(statement: string): {
  prompt: string;
  positiveAnchor: string;
  negativeAnchor: string;
  passDisplay: string;
  failDisplay: string;
} {
  const cleaned = statement.trim().replace(/[.?!]+$/, '');
  if (!cleaned) {
    return {
      prompt: 'Does the photo satisfy this requirement?',
      positiveAnchor: 'Portrait photo that clearly satisfies the requirement.',
      negativeAnchor: 'Portrait photo that fails or violates the requirement.',
      passDisplay: 'Requirement met',
      failDisplay: 'Requirement not met',
    };
  }
  const negMatch = cleaned.match(/^(?:no|not|without|zero|free\s+of)\s+(.+)$/i);
  if (negMatch) {
    const forbidden = negMatch[1].trim();
    return {
      prompt: `Is the person free of ${forbidden}?`,
      positiveAnchor: `Clear natural portrait or selfie of a person with no ${forbidden} visible anywhere.`,
      negativeAnchor: `Portrait of a person with ${forbidden} clearly visible on the face, head, or background.`,
      passDisplay: `No ${forbidden}`,
      failDisplay: `${forbidden.charAt(0).toUpperCase() + forbidden.slice(1)} detected`,
    };
  }
  return {
    prompt: `Does the photo show: ${cleaned}?`,
    positiveAnchor: `Portrait or selfie clearly showing: ${cleaned}.`,
    negativeAnchor: `Portrait that does NOT show ${cleaned}, or shows the opposite condition.`,
    passDisplay: cleaned.length <= 36 ? cleaned : `${cleaned.slice(0, 33)}…`,
    failDisplay: `Missing: ${cleaned.length <= 28 ? cleaned : `${cleaned.slice(0, 25)}…`}`,
  };
}

export const QUICK_RULE_IDEAS: Array<{ shortTitle: string; statement: string }> = [
  {
    shortTitle: 'No Headphones',
    statement: 'No headphones, earbuds, or headset worn on the head',
  },
  {
    shortTitle: 'Even Lighting',
    statement: 'Bright, even lighting across the face with no harsh shadows',
  },
  {
    shortTitle: 'Collared Attire',
    statement: 'Person wearing a collared shirt, jacket, or formal top',
  },
  {
    shortTitle: 'Single Person',
    statement: 'No other people or faces visible in the background',
  },
];

export const PASSPORT_RULES: PassportRule[] = [
  {
    id: 'full_face',
    num: 1,
    icon: 'face',
    shortTitle: 'Head in Frame',
    title: '1. Head Inside Frame',
    statement: 'Whole head, eyes, and chin visible in frame',
    normalizePrior: false,
    prompt: "Can you see the top of the person's hair and their shirt below the chin?",
    options: [
      {
        label: 'pass_frontal',
        description:
          'Portrait or webcam selfie showing the top of the head with hair, forehead, both eyes, chin, and shirt or shoulders.',
      },
      {
        label: 'fail_cropped',
        description: 'Tightly cropped macro close-up of only skin, eyes, nose, and lips filling the entire image.',
      },
      {
        label: 'fail_top_cut',
        description:
          'Half-face crop cut off at the nose with only the lower face, beard or mouth, chin, and shirt visible and no eyes.',
      },
    ],
    optionDisplay: {
      pass_frontal: 'Full head & chin in frame',
      fail_cropped: 'Head/chin clipped at edge',
      fail_top_cut: 'Upper head / eyes cut off at top',
      fail_bottom_cut: 'Face too low / chin cut off',
    },
    enabled: true,
  },
  {
    id: 'eyes_open',
    num: 2,
    icon: 'visibility',
    shortTitle: 'Eyes Open',
    title: '2. Eyes Open & Direct',
    statement: 'Eyes open and looking at camera',
    normalizePrior: false,
    prompt: "Are the person's eyes open and looking straight ahead?",
    options: [
      {
        label: 'pass_open',
        description: 'Front-facing portrait of a person with open eyes looking straight ahead at the camera.',
      },
      {
        label: 'fail_looking_away',
        description: 'Person with eyes cast downward looking down and away from the camera.',
      },
      {
        label: 'fail_closed',
        description: 'Woman with both eyes closed shut, sleeping or blinking with eyelids closed.',
      },
      {
        label: 'fail_sunglasses',
        description: 'Person wearing dark black sunglasses covering the eyes.',
      },
    ],
    optionDisplay: {
      pass_open: 'Open & looking at camera',
      fail_looking_away: 'Looking away from camera',
      fail_closed: 'Eyes closed / blinking',
      fail_sunglasses: 'Hidden behind dark lenses',
      fail_top_cut: 'Eyes cut off / not in frame',
      fail_bottom_cut: 'Face too low / not in frame',
    },
    enabled: true,
  },
  {
    id: 'neutral_expression',
    num: 3,
    icon: 'sentiment_neutral',
    shortTitle: 'Neutral Face',
    title: '3. Neutral Expression',
    statement: 'Natural facial expression with mouth closed',
    normalizePrior: false,
    prompt: 'Does the person have a calm, neutral expression with mouth closed?',
    options: [
      {
        label: 'pass_neutral',
        description: 'Calm, serious, neutral facial expression with lips and mouth closed.',
      },
      {
        label: 'fail_smiling',
        description: 'Happy person grinning and smiling broadly showing white teeth.',
      },
    ],
    optionDisplay: {
      pass_neutral: 'Neutral expression, mouth closed',
      fail_smiling: 'Smiling / teeth visible',
      fail_bottom_cut: 'Mouth & chin cut off at bottom',
    },
    enabled: true,
  },
  {
    id: 'neutral_background',
    num: 4,
    icon: 'crop_portrait',
    shortTitle: 'Plain BG',
    title: '4. Plain Neutral Background',
    statement: 'Plain neutral wall or backdrop',
    normalizePrior: false,
    prompt: 'Is the background a plain, simple wall or backdrop?',
    options: [
      {
        label: 'pass_white_bg',
        description: 'Indoor portrait against a plain wall, simple neutral wall, or solid white studio background.',
      },
      {
        label: 'fail_outdoor_street',
        description: 'Outdoor street scene with green trees, autumn leaves, buildings, and sky in the background.',
      },
      {
        label: 'fail_indoor_clutter',
        description: 'Cluttered room background filled with bookshelves, books, and household items.',
      },
    ],
    optionDisplay: {
      pass_white_bg: 'Plain wall / backdrop',
      fail_indoor_clutter: 'Cluttered room background',
      fail_outdoor_street: 'Outdoor street background',
    },
    enabled: true,
  },
  {
    id: 'shoulders_visible',
    num: 5,
    icon: 'center_focus_strong',
    shortTitle: 'Straight Pose',
    title: '5. Straight Frontal Pose & Framing',
    statement: 'Head straight and upright with upper shoulders visible',
    normalizePrior: false,
    prompt: 'How is the person positioned and framed in the photo?',
    options: [
      {
        label: 'pass_shoulders',
        description: 'Level frontal portrait looking straight ahead with neck and upper chest or shoulders visible.',
      },
      {
        label: 'fail_angled_selfie',
        description: 'Tilted or high-angle selfie with the head angled and camera held above or to the side.',
      },
      {
        label: 'fail_chin_up',
        description: 'Person tilting their head back with chin raised high and looking down at the camera.',
      },
      {
        label: 'fail_turned_away',
        description: 'Person with their head turned sideways or looking away from the camera.',
      },
      {
        label: 'fail_bottom_cutoff',
        description: 'Tight face crop cut off right under the chin with no neck, collar, or shoulders visible.',
      },
    ],
    optionDisplay: {
      pass_shoulders: 'Upright & level head pose',
      fail_angled_selfie: 'Level the camera (head tilted)',
      fail_chin_up: 'Lower your chin',
      fail_turned_away: 'Face straight ahead',
      fail_bottom_cutoff: 'Show upper shoulders',
    },
    enabled: true,
  },
  {
    id: 'hair_clear',
    num: 6,
    icon: 'content_cut',
    shortTitle: 'Hair Clear',
    title: '6. Hair Clear of Eyes',
    statement: 'Eyes and eyebrows clear of hair bangs',
    normalizePrior: false,
    prompt: 'Are the eyes clear of hair bangs?',
    options: [
      {
        label: 'pass_hair_clear',
        description: "Portrait where the person's eyes and face are clearly visible and not covered by bangs.",
      },
      {
        label: 'fail_bangs_over_eyes',
        description: 'Long messy hair bangs hanging down across the forehead and covering the eyes.',
      },
    ],
    optionDisplay: {
      pass_hair_clear: 'Eyes & brow unobstructed',
      fail_bangs_over_eyes: 'Bangs covering eyes/brow',
    },
    enabled: true,
  },
  {
    id: 'no_glasses',
    num: 7,
    icon: 'visibility_off',
    shortTitle: 'No Glasses',
    title: '7. No Eyeglasses or Sunglasses',
    statement: 'No eyeglasses or dark sunglasses',
    normalizePrior: false,
    prompt: 'Are the eyes bare and unobstructed by lenses?',
    options: [
      {
        label: 'pass_bare_eyes',
        description: 'Close-up portrait of a person with bare eyes, eyebrows, and nose.',
      },
      {
        label: 'fail_eyeglasses',
        description: 'Eyeglass frames and corrective spectacle lenses worn over the eyes.',
      },
      {
        label: 'fail_sunglasses',
        description: 'Dark black sunglasses with opaque tinted lenses worn over the eyes.',
      },
    ],
    optionDisplay: {
      pass_bare_eyes: 'No glasses or frames',
      fail_eyeglasses: 'Eyeglasses worn',
      fail_sunglasses: 'Dark sunglasses worn',
    },
    enabled: true,
  },
  {
    id: 'framing_chest_up',
    num: 8,
    icon: 'masks',
    shortTitle: 'No Mask/Cap',
    title: '8. Uncovered Face & Head',
    statement: 'No mask, cap, or hat covering face or head',
    normalizePrior: false,
    prompt: 'Is the lower face uncovered (no mask or cap)?',
    options: [
      {
        label: 'pass_chest_up',
        description: 'Person with visible nose, mouth, lips, and chin, and bare hair with no cap or mask.',
      },
      {
        label: 'fail_mask_hat',
        description: 'Face hidden behind a protective face mask covering nose and mouth and wearing a cap.',
      },
    ],
    optionDisplay: {
      pass_chest_up: 'No mask, cap, or hat',
      fail_mask_hat: 'Mask or cap covering face/head',
      fail_bottom_cut: 'Lower face cut off at bottom',
    },
    enabled: true,
  },
  {
    id: 'occupancy_70_percent',
    num: 9,
    icon: 'photo_size_select_large',
    shortTitle: 'Good Distance',
    title: '9. Proper Portrait Distance',
    statement: 'Subject close enough to fill portrait frame',
    normalizePrior: false,
    prompt: 'Is the subject close enough to the camera (not standing far away)?',
    options: [
      {
        label: 'pass_70_pct',
        description: "Close-up headshot portrait of a person's face filling the frame.",
      },
      {
        label: 'fail_too_small',
        description: 'Distant person standing far away outdoors on a street with trees and buildings.',
      },
    ],
    optionDisplay: {
      pass_70_pct: 'Good close-up portrait scale',
      fail_too_small: 'Subject too far from camera',
      fail_bottom_peek: 'Subject not centered in frame',
    },
    enabled: true,
  },
];

export const BUILT_IN_RULE_PACKS: PhotoRulePack[] = [
  {
    id: 'full_biometric',
    name: 'Full Biometric (9)',
    subtitle: 'All 9 strict studio & biometric photo requirements',
    enabledRuleIds: PASSPORT_RULES.map((r) => r.id),
  },
  {
    id: 'passport_id',
    name: 'Passport / ID (6)',
    subtitle: 'Core official ID & passport checklist tuned for indoor/outdoor camera lighting',
    enabledRuleIds: [
      'full_face',
      'eyes_open',
      'shoulders_visible',
      'no_glasses',
      'framing_chest_up',
      'occupancy_70_percent',
    ],
  },
  {
    id: 'badge_profile',
    name: 'Badge / Profile (5)',
    subtitle: 'Employee badge & profile photo checklist (eyeglasses allowed)',
    enabledRuleIds: ['full_face', 'eyes_open', 'shoulders_visible', 'framing_chest_up', 'occupancy_70_percent'],
  },
];

function photoAssetUrl(fileName: string): string {
  const base = import.meta.env.BASE_URL || '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  return `${normalizedBase}images/id_photos/${fileName}`;
}

function isBuiltInPhotoId(id: string): boolean {
  return id.endsWith('.jpg') && !id.includes('_1');
}

function presetThumbSrc(p: PhotoPreset): string {
  return isBuiltInPhotoId(p.id) ? photoAssetUrl(`thumbs/${p.id}`) : p.url;
}

function presetThumbSrcSet(p: PhotoPreset): string {
  return isBuiltInPhotoId(p.id)
    ? `${photoAssetUrl(`thumbs_1x/${p.id}`)} 1x, ${photoAssetUrl(`thumbs/${p.id}`)} 2x`
    : '';
}

function exemplarThumbSrc(item: ExemplarEntry): string {
  return isBuiltInPhotoId(item.id) ? photoAssetUrl(`thumbs_1x/${item.id}`) : item.url;
}

function exemplarThumbSrcSet(item: ExemplarEntry): string {
  return isBuiltInPhotoId(item.id)
    ? `${photoAssetUrl(`thumbs_xs/${item.id}`)} 1x, ${photoAssetUrl(`thumbs_1x/${item.id}`)} 2x`
    : '';
}

export const PRESET_PHOTOS: PhotoPreset[] = [
  {
    id: 'passport_compliant_female.jpg',
    title: 'Compliant #1',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('passport_compliant_female.jpg'),
    biometrics: {
      bgLum: 251,
      bgVar: 2,
      isPlainWhiteBg: true,
      fgRatio: 0.68,
      bottomShoulderWidthRatio: 0.78,
      faceInfo: {
        hasFace: true,
        minX: 0.284,
        minY: 0.231,
        maxX: 0.713,
        maxY: 0.616,
        faceW: 0.429,
        faceH: 0.385,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'passport_compliant_male.jpg',
    title: 'Compliant #2',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('passport_compliant_male.jpg'),
    biometrics: {
      bgLum: 249,
      bgVar: 3,
      isPlainWhiteBg: true,
      fgRatio: 0.71,
      bottomShoulderWidthRatio: 0.91,
      faceInfo: {
        hasFace: true,
        minX: 0.294,
        minY: 0.215,
        maxX: 0.708,
        maxY: 0.594,
        faceW: 0.414,
        faceH: 0.379,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_glasses_smile.jpg',
    title: 'Sunglasses',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_glasses_smile.jpg'),
    biometrics: {
      bgLum: 242,
      bgVar: 3,
      isPlainWhiteBg: true,
      fgRatio: 0.69,
      bottomShoulderWidthRatio: 1.0,
      faceInfo: {
        hasFace: true,
        minX: 0.3,
        minY: 0.222,
        maxX: 0.72,
        maxY: 0.608,
        faceW: 0.42,
        faceH: 0.386,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_eyes_closed_hair.jpg',
    title: 'Eyes Closed',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_eyes_closed_hair.jpg'),
    biometrics: {
      bgLum: 232,
      bgVar: 10,
      isPlainWhiteBg: true,
      fgRatio: 0.67,
      bottomShoulderWidthRatio: 0.92,
      faceInfo: {
        hasFace: true,
        minX: 0.341,
        minY: 0.272,
        maxX: 0.674,
        maxY: 0.555,
        faceW: 0.333,
        faceH: 0.283,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_extreme_zoom_cropped.jpg',
    title: 'Too Close',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_extreme_zoom_cropped.jpg'),
    biometrics: {
      bgLum: 105,
      bgVar: 83,
      isPlainWhiteBg: false,
      fgRatio: 0.95,
      bottomShoulderWidthRatio: 0.97,
      faceInfo: {
        hasFace: true,
        minX: 0.067,
        minY: 0.157,
        maxX: 0.926,
        maxY: 1.046,
        faceW: 0.859,
        faceH: 0.888,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_distant_waist_up.jpg',
    title: 'Too Far',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_distant_waist_up.jpg'),
    biometrics: {
      bgLum: 169,
      bgVar: 93,
      isPlainWhiteBg: false,
      fgRatio: 0.26,
      bottomShoulderWidthRatio: 0.8,
      faceInfo: { hasFace: false },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_tilted_glasses_selfie.jpg',
    title: 'Tilted Selfie',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_tilted_glasses_selfie.jpg'),
    biometrics: {
      bgLum: 188,
      bgVar: 55,
      bgSat: 63,
      isPlainWhiteBg: false,
      fgRatio: 0.89,
      bottomShoulderWidthRatio: 0.95,
      faceInfo: {
        hasFace: true,
        minX: 0.164,
        minY: 0.271,
        maxX: 0.553,
        maxY: 0.58,
        centerX: 0.359,
        centerY: 0.426,
        faceW: 0.389,
        faceH: 0.31,
        yawAsymmetry: 0.497,
        eyeRoll: 0.521,
      },
    },
    rawEg2Vision: {},
  },
  {
    id: 'id_fail_mask_hat.jpg',
    title: 'Mask & Cap',
    sub: 'Pending…',
    isCompliant: null,
    evaluated: false,
    url: photoAssetUrl('id_fail_mask_hat.jpg'),
    biometrics: {
      bgLum: 180,
      bgVar: 10,
      isPlainWhiteBg: false,
      fgRatio: 0.31,
      bottomShoulderWidthRatio: 1.0,
      faceInfo: { hasFace: false },
    },
    rawEg2Vision: {},
  },
];

const logit = (p: number): number => {
  const c = Math.max(1e-4, Math.min(1 - 1e-4, p));
  return Math.log(c / (1 - c));
};

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

const STORAGE_CUSTOM_PACKS_KEY = 'mp_photocheck_custom_packs_v1';
const STORAGE_CUSTOM_RULES_KEY = 'mp_photocheck_custom_rules_v1';

export const SUBJECT_PRESENT_KEY = 'subject_present';
export const SUBJECT_PRESENT_PASS_LABEL = 'pass_human_face';
export const SUBJECT_PRESENT_FAIL_EMPTY = 'fail_empty_scene';
export const SUBJECT_PRESENT_FAIL_OBJECT = 'fail_object_animal';
export const SUBJECT_PRESENT_FAIL_HAND = 'fail_hand_only';
export const SUBJECT_PRESENT_FAIL_PARTIAL = 'fail_partial_peek';
export const SUBJECT_PRESENT_THRESHOLD = 0.4;
export const DEFAULT_NO_FACE_REASON = 'No face detected in frame';
export const DEFAULT_BOTTOM_CUTOFF_REASON = 'Face too low / chin cut off';

const SELFIE_MULTICLASS_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';
const SELFIE_SEGMENTER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

export class DecisionVisionPlayground {
  private el: Record<string, HTMLElement> = {};
  private rules: PassportRule[] = PASSPORT_RULES.map((r) => ({
    ...r,
    options: r.options.map((o) => ({ ...o })),
    optionDisplay: { ...r.optionDisplay },
    enabled: true,
  }));
  private activePackId = 'full_biometric';
  private savedCustomPacks: PhotoRulePack[] = [];
  private showIssuesOnly = false;
  private editingRuleId: string | null = null;
  private lastInferenceMs: number | undefined;

  private currentPhotoObj: PhotoPreset = PRESET_PHOTOS[0];
  private selectedRuleId = 'full_face';
  private showBiometricOverlay = true;
  private ruleExemplarBank: Record<string, { positives: ExemplarEntry[]; negatives: ExemplarEntry[] }> = {};
  private modelState: VisionModelState = { ready: false, loading: false, label: '' };

  private webcamActive = false;
  private webcamStream: MediaStream | null = null;
  private liveLoopTimer: ReturnType<typeof setTimeout> | null = null;
  private liveFrameCanvas = document.createElement('canvas');
  private encodeCanvas = document.createElement('canvas');
  private autoSnapOnPass = true;
  private autoSnapStreak = 0;
  private readonly autoSnapRequiredStreak = 2;
  private bestStreakFrameDataUrl: string | null = null;
  private latestLiveFrameDataUrl: string | null = null;
  private imageSegmenter: ImageSegmenter | null = null;
  private imageSegmenterPromise: Promise<ImageSegmenter | null> | null = null;
  private isMulticlassSegmenter = true;

  constructor(
    private root: HTMLElement,
    private ask?: VisionAskFn,
    private onStatus?: (text: string, inferenceTime?: number) => void,
    private ensureVisionModel?: () => Promise<boolean>
  ) {
    this.liveFrameCanvas.width = 360;
    this.liveFrameCanvas.height = 460;
    this.encodeCanvas.width = 360;
    this.encodeCanvas.height = 460;
    this.loadSavedCustomizations();
    this.initExemplarBank();
  }

  private loadSavedCustomizations() {
    try {
      const rawRules = localStorage.getItem(STORAGE_CUSTOM_RULES_KEY);
      if (rawRules) {
        const parsed = JSON.parse(rawRules) as PassportRule[];
        if (Array.isArray(parsed)) {
          for (const cr of parsed) {
            if (cr?.id && !this.rules.some((r) => r.id === cr.id)) {
              this.rules.push({ ...cr, isCustom: true, enabled: cr.enabled ?? true });
            }
          }
        }
      }
      const rawPacks = localStorage.getItem(STORAGE_CUSTOM_PACKS_KEY);
      if (rawPacks) {
        const parsedPacks = JSON.parse(rawPacks) as PhotoRulePack[];
        if (Array.isArray(parsedPacks)) {
          this.savedCustomPacks = parsedPacks.filter((p) => p?.id && p?.name);
        }
      }
    } catch {
      // Ignore storage access errors in restricted contexts
    }
  }

  private persistCustomizations() {
    try {
      const customRules = this.rules.filter((r) => r.isCustom);
      localStorage.setItem(STORAGE_CUSTOM_RULES_KEY, JSON.stringify(customRules));
      localStorage.setItem(STORAGE_CUSTOM_PACKS_KEY, JSON.stringify(this.savedCustomPacks));
    } catch {
      // Ignore storage access errors
    }
  }

  private computeLogitVector(rawVision: Record<string, RawRuleVisionScore>): number[] {
    return this.rules.map((r) => logit(rawVision?.[r.id]?.passProb ?? 0.5));
  }

  private l2DistSq(a: number[], b: number[]): number {
    let s = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  private ensureRuleBank(ruleId: string) {
    if (this.ruleExemplarBank[ruleId]) return;
    const defaultFails: Record<string, string[]> = {
      'passport_compliant_female.jpg': [],
      'passport_compliant_male.jpg': [],
      'id_fail_glasses_smile.jpg': ['eyes_open', 'neutral_expression', 'no_glasses'],
      'id_fail_eyes_closed_hair.jpg': ['eyes_open', 'hair_clear'],
      'id_fail_extreme_zoom_cropped.jpg': ['full_face', 'occupancy_70_percent'],
      'id_fail_distant_waist_up.jpg': ['neutral_background', 'occupancy_70_percent'],
      'id_fail_tilted_glasses_selfie.jpg': [
        'neutral_expression',
        'neutral_background',
        'shoulders_visible',
        'no_glasses',
      ],
      'id_fail_mask_hat.jpg': ['neutral_background', 'framing_chest_up'],
    };
    this.ruleExemplarBank[ruleId] = { positives: [], negatives: [] };
    for (const p of PRESET_PHOTOS) {
      const failedRules = defaultFails[p.id] ?? [];
      const isPos = !failedRules.includes(ruleId);
      const defaultProb = isPos ? 0.82 : 0.18;
      const entry: ExemplarEntry = {
        id: p.id,
        name: p.title,
        url: p.url,
        rawPassProb: defaultProb,
        logitVal: logit(defaultProb),
        logitVec: this.rules.map((r) => logit(!(defaultFails[p.id] ?? []).includes(r.id) ? 0.82 : 0.18)),
        enabled: true,
        isDefault: true,
      };
      if (isPos) this.ruleExemplarBank[ruleId].positives.push(entry);
      else this.ruleExemplarBank[ruleId].negatives.push(entry);
    }
  }

  private initExemplarBank() {
    for (const rule of this.rules) {
      this.ensureRuleBank(rule.id);
    }
  }

  private syncExemplarBankWithLiveScore(photoObj: PhotoPreset) {
    if (!photoObj?.rawEg2Vision) return;
    const vec = this.computeLogitVector(photoObj.rawEg2Vision);
    for (const rule of this.rules) {
      this.ensureRuleBank(rule.id);
      const bank = this.ruleExemplarBank[rule.id];
      if (!bank) continue;
      const rawProb = photoObj.rawEg2Vision[rule.id]?.passProb ?? 0.5;
      for (const list of [bank.positives, bank.negatives]) {
        for (const item of list) {
          if (item.id === photoObj.id) {
            item.rawPassProb = rawProb;
            item.logitVal = logit(rawProb);
            item.logitVec = vec;
          }
        }
      }
    }
  }

  public setLoadProgress(loaded: number, total: number) {
    const pct = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
    const loadedMB = (loaded / 1048576).toFixed(0);
    const totalMB = total > 0 ? (total / 1048576).toFixed(0) : '?';
    const label = `⏳ Loading 435M-VL: ${pct}% (${loadedMB}/${totalMB} MB)`;
    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = label;
    }
    if (!this.currentPhotoObj.evaluated && this.el['dv-verdict-sub']) {
      this.el['dv-verdict-sub'].textContent = label;
    }
  }

  public setModelState(state: VisionModelState) {
    const becameReady = !this.modelState.ready && state.ready;
    this.modelState = state;
    if (state.loading && this.el['dv-result-meta'] && !this.el['dv-result-meta'].textContent?.includes('%')) {
      this.el['dv-result-meta'].textContent = '⏳ Loading 435M-VL: 0%…';
    } else if (becameReady && this.isVisionModelReady()) {
      if (this.el['dv-result-meta'] && !this.webcamActive) {
        this.el['dv-result-meta'].textContent = '✓ 435M-VL Ready · Evaluating…';
      }
      void this.evaluateCurrentPhotoWithDecisionMaker();
    }
    this.renderAllPanels();
  }

  private isVisionModelReady(): boolean {
    if (!this.modelState.ready || !this.ask) return false;
    return (
      Boolean(this.modelState.loadedValue && DECISION_MODELS[this.modelState.loadedValue]?.supportsVision) ||
      Boolean(this.modelState.hasCustomModel)
    );
  }

  private getEnabledRules(): PassportRule[] {
    const enabled = this.rules.filter((r) => r.enabled !== false);
    return enabled.length > 0 ? enabled : [this.rules[0]];
  }

  private buildPassportSchema() {
    return {
      context: '',
      questions: [
        {
          id: SUBJECT_PRESENT_KEY,
          type: 'categorical',
          prompt: "Is a human person's face or head visible in the photo?",
          normalizePrior: false,
          options: [
            {
              label: SUBJECT_PRESENT_PASS_LABEL,
              description: "Portrait, selfie, headshot, or photo showing a human person's face or head.",
            },
            {
              label: SUBJECT_PRESENT_FAIL_EMPTY,
              description:
                "Empty room, wall, curtain, furniture, desk, ceiling, floor, or background with no person's face.",
            },
            {
              label: SUBJECT_PRESENT_FAIL_OBJECT,
              description: 'Object, animal, pet, cat, dog, food, text, screen, or item with no human face.',
            },
            {
              label: SUBJECT_PRESENT_FAIL_HAND,
              description: 'Close-up of only a hand, fingers, fist, or thumb with no face visible.',
            },
          ],
        },
        ...this.rules.map((rule) => ({
          id: rule.id,
          type: 'categorical',
          prompt: rule.prompt,
          normalizePrior: Boolean(rule.normalizePrior),
          options: rule.options.map((opt) => ({
            label: opt.label,
            description: opt.description,
          })),
        })),
      ],
    };
  }

  /**
   * Computes a fast spatial framing check on `this.encodeCanvas` to catch
   * frames where the upper/center head guide zone (`y = 0.18..0.62`) is empty
   * background/curtain while either no subject is present or only the top of a
   * head peeks in at the bottom edge (`y = 0.72..0.98`).
   */
  public analyzeSpatialFraming(source?: HTMLImageElement | HTMLCanvasElement): {
    mode: 'normal' | 'empty_scene' | 'bottom_cutoff';
    upperDy: number;
    lowerDy: number;
    upperCd: number;
    lowerCd: number;
  } {
    const W = 90;
    const H = 115;
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const ctx = c.getContext('2d');
    const target = source ?? this.encodeCanvas;
    if (!ctx || !target) {
      return { mode: 'normal', upperDy: 15, lowerDy: 8, upperCd: 110, lowerCd: 150 };
    }
    ctx.drawImage(target, 0, 0, W, H);
    const d = ctx.getImageData(0, 0, W, H).data;

    const x0 = Math.floor(W * 0.22);
    const x1 = Math.floor(W * 0.78);
    const colBg: Array<[number, number, number]> = [];
    const bgY0 = Math.floor(H * 0.04);
    const bgY1 = Math.floor(H * 0.11);
    for (let x = 0; x < W; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let y = bgY0; y < bgY1; y++) {
        const i = (y * W + x) * 4;
        r += d[i];
        g += d[i + 1];
        b += d[i + 2];
        n++;
      }
      colBg.push(n > 0 ? [r / n, g / n, b / n] : [200, 200, 200]);
    }

    const avgBand = (yStart: number, yEnd: number) => {
      let dySum = 0;
      let cdSum = 0;
      let n = 0;
      const y0 = Math.floor(H * yStart);
      const y1 = Math.floor(H * yEnd);
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * W + x) * 4;
          const iu = (Math.max(0, y - 1) * W + x) * 4;
          const idn = (Math.min(H - 1, y + 1) * W + x) * 4;
          const dy =
            (Math.abs(d[idn] - d[iu]) + Math.abs(d[idn + 1] - d[iu + 1]) + Math.abs(d[idn + 2] - d[iu + 2])) / 3;
          const [br, bg, bb] = colBg[x];
          const cd = Math.hypot(d[i] - br, d[i + 1] - bg, d[i + 2] - bb);
          dySum += dy;
          cdSum += cd;
          n++;
        }
      }
      return {
        dy: n > 0 ? dySum / n : 0,
        cd: n > 0 ? cdSum / n : 0,
      };
    };

    const upper = avgBand(0.18, 0.62);
    const lower = avgBand(0.72, 0.98);

    if (upper.dy < 4.8 && lower.dy < 4.8 && upper.cd < 85 && lower.cd < 85) {
      return { mode: 'empty_scene', upperDy: upper.dy, lowerDy: lower.dy, upperCd: upper.cd, lowerCd: lower.cd };
    }
    if (upper.dy < 5.5 && lower.dy > 6.0 && lower.dy > upper.dy * 1.8 && lower.cd > upper.cd * 1.25) {
      return { mode: 'bottom_cutoff', upperDy: upper.dy, lowerDy: lower.dy, upperCd: upper.cd, lowerCd: lower.cd };
    }
    return { mode: 'normal', upperDy: upper.dy, lowerDy: lower.dy, upperCd: upper.cd, lowerCd: lower.cd };
  }

  private applySpatialFramingToRawVision(
    rawVision: Record<string, RawRuleVisionScore>,
    spatial: { mode: 'normal' | 'empty_scene' | 'bottom_cutoff' }
  ): Record<string, RawRuleVisionScore> {
    if (spatial.mode === 'empty_scene') {
      const prev = rawVision[SUBJECT_PRESENT_KEY]?.passProb ?? 1.0;
      rawVision[SUBJECT_PRESENT_KEY] = {
        passProb: Math.min(prev, 0.12),
        topLabel: SUBJECT_PRESENT_FAIL_EMPTY,
        failLabel: SUBJECT_PRESENT_FAIL_EMPTY,
      };
    } else if (spatial.mode === 'bottom_cutoff') {
      const prevSubj = rawVision[SUBJECT_PRESENT_KEY]?.passProb ?? 1.0;
      rawVision[SUBJECT_PRESENT_KEY] = {
        passProb: Math.min(prevSubj, 0.18),
        topLabel: SUBJECT_PRESENT_FAIL_PARTIAL,
        failLabel: SUBJECT_PRESENT_FAIL_PARTIAL,
      };
      if (rawVision['full_face']) {
        rawVision['full_face'] = {
          passProb: Math.min(rawVision['full_face'].passProb, 0.18),
          topLabel: 'fail_bottom_cut',
          failLabel: 'fail_bottom_cut',
        };
      }
      if (rawVision['shoulders_visible']) {
        rawVision['shoulders_visible'] = {
          passProb: Math.min(rawVision['shoulders_visible'].passProb, 0.18),
          topLabel: 'fail_bottom_cutoff',
          failLabel: 'fail_bottom_cutoff',
        };
      }
      if (rawVision['occupancy_70_percent']) {
        rawVision['occupancy_70_percent'] = {
          passProb: Math.min(rawVision['occupancy_70_percent'].passProb, 0.2),
          topLabel: 'fail_bottom_peek',
          failLabel: 'fail_bottom_peek',
        };
      }
    }
    return rawVision;
  }

  private async sourceToImageBytes(source: HTMLImageElement | HTMLCanvasElement): Promise<Uint8Array> {
    const ctx = this.encodeCanvas.getContext('2d')!;
    ctx.drawImage(source, 0, 0, this.encodeCanvas.width, this.encodeCanvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      this.encodeCanvas.toBlob((b) => resolve(b), 'image/jpeg', 0.92)
    );
    if (!blob) return new Uint8Array(0);
    const buf = await blob.arrayBuffer();
    return new Uint8Array(buf);
  }

  private parseSchemaResultToRawVision(
    schemaRes: Record<string, any>,
    fallback?: Record<string, RawRuleVisionScore>
  ): Record<string, RawRuleVisionScore> {
    const out: Record<string, RawRuleVisionScore> = {};
    const subjDec = schemaRes?.[SUBJECT_PRESENT_KEY];
    const subjProbs: Array<{ label: string; probability: number }> = subjDec?.probabilities ?? [];
    if (subjProbs.length > 0) {
      const passEntry = subjProbs.find((p) => p.label === SUBJECT_PRESENT_PASS_LABEL);
      const pPass = passEntry ? passEntry.probability : 0;
      let bestFailLabel = SUBJECT_PRESENT_FAIL_EMPTY;
      let bestFailProb = -1;
      for (const p of subjProbs) {
        if (p.label !== SUBJECT_PRESENT_PASS_LABEL && p.probability > bestFailProb) {
          bestFailProb = p.probability;
          bestFailLabel = p.label;
        }
      }
      if (bestFailProb < 0) bestFailProb = Math.max(0, 1 - pPass);
      const subjectPresentProb = pPass + bestFailProb > 0 ? pPass / (pPass + bestFailProb) : 1.0;
      out[SUBJECT_PRESENT_KEY] = {
        passProb: subjectPresentProb,
        topLabel:
          subjDec.label ||
          (subjectPresentProb >= SUBJECT_PRESENT_THRESHOLD ? SUBJECT_PRESENT_PASS_LABEL : bestFailLabel),
        failLabel: bestFailLabel,
      };
    } else if (fallback?.[SUBJECT_PRESENT_KEY]) {
      out[SUBJECT_PRESENT_KEY] = fallback[SUBJECT_PRESENT_KEY];
    } else {
      out[SUBJECT_PRESENT_KEY] = {
        passProb: 1.0,
        topLabel: SUBJECT_PRESENT_PASS_LABEL,
        failLabel: SUBJECT_PRESENT_FAIL_EMPTY,
      };
    }

    for (const rule of this.rules) {
      const dec = schemaRes?.[rule.id];
      const probsArr: Array<{ label: string; probability: number }> = dec?.probabilities ?? [];
      if (probsArr.length > 0) {
        const passLabel = rule.options[0].label;
        const passEntry = probsArr.find((p) => p.label === passLabel);
        const pPass = passEntry ? passEntry.probability : 0;
        let bestFailLabel = rule.options[1]?.label ?? passLabel;
        let bestFailProb = -1;
        for (const p of probsArr) {
          if (p.label !== passLabel && p.probability > bestFailProb) {
            bestFailProb = p.probability;
            bestFailLabel = p.label;
          }
        }
        if (bestFailProb < 0) bestFailProb = Math.max(0, 1 - pPass);
        const binaryPassProb = pPass + bestFailProb > 0 ? pPass / (pPass + bestFailProb) : 0.5;
        out[rule.id] = {
          passProb: binaryPassProb,
          topLabel: dec.label || (binaryPassProb >= 0.5 ? passLabel : bestFailLabel),
          failLabel: bestFailLabel,
        };
      } else if (fallback?.[rule.id]) {
        out[rule.id] = fallback[rule.id];
      } else {
        out[rule.id] = {
          passProb: 0.5,
          topLabel: rule.options[0].label,
          failLabel: rule.options[1]?.label ?? rule.options[0].label,
        };
      }
    }
    return out;
  }

  private async evaluateSourceWithDecisionMaker(
    source: HTMLImageElement | HTMLCanvasElement,
    fallback?: Record<string, RawRuleVisionScore>
  ): Promise<{ rawEg2Vision: Record<string, RawRuleVisionScore>; inferenceTime?: number } | null> {
    if (!this.isVisionModelReady() && this.ensureVisionModel) {
      await this.ensureVisionModel();
    }
    if (!this.isVisionModelReady() || !this.ask) return null;
    const run = async () => {
      try {
        const imageBytes = await this.sourceToImageBytes(source);
        if (imageBytes.byteLength === 0) return null;
        const spatial = this.analyzeSpatialFraming(this.encodeCanvas);
        const msg = await this.ask!({ imageBytes }, 'schema', this.buildPassportSchema());
        if (msg?.type === 'DECIDE_RESULT' && msg.result) {
          const parsed = this.parseSchemaResultToRawVision(msg.result, fallback);
          return {
            rawEg2Vision: this.applySpatialFramingToRawVision(parsed, spatial),
            inferenceTime: msg.inferenceTime,
          };
        }
      } catch (e) {
        console.warn('DecisionMaker multimodal evaluation fallback:', e);
      }
      return null;
    };
    const next = (this.askQueue ?? Promise.resolve()).then(run, run);
    this.askQueue = next.then(
      () => {},
      () => {}
    );
    return next;
  }
  private askQueue: Promise<any> = Promise.resolve();

  private updatePhotoComplianceBadge(targetPhoto: PhotoPreset) {
    if (!targetPhoto.evaluated) return;
    const evalRes = this.evaluatePhotoRules(targetPhoto);
    const activeRules = this.getEnabledRules();
    let passCount = 0;
    for (const r of activeRules) {
      if (evalRes[r.id]?.passed) passCount++;
    }
    const failCount = activeRules.length - passCount;
    targetPhoto.isCompliant = failCount === 0;
    targetPhoto.sub =
      failCount === 0
        ? `${activeRules.length}/${activeRules.length} Pass`
        : `${failCount} Fail${failCount > 1 ? 's' : ''}`;
  }

  private evaluatingPhotoIds = new Set<string>();

  private async evaluateCurrentPhotoWithDecisionMaker() {
    if (!this.isVisionModelReady() || this.webcamActive) return;
    const targetPhoto = this.currentPhotoObj;
    if (this.evaluatingPhotoIds.has(targetPhoto.id)) return;
    this.evaluatingPhotoIds.add(targetPhoto.id);
    try {
      const sourceImg = await new Promise<HTMLImageElement | undefined>((resolve) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () => resolve(undefined);
        el.src = targetPhoto.url;
      });
      if (!sourceImg) return;
      const dmEval = await this.evaluateSourceWithDecisionMaker(sourceImg);
      if (!dmEval) return;
      this.lastInferenceMs = dmEval.inferenceTime;
      targetPhoto.rawEg2Vision = dmEval.rawEg2Vision;
      targetPhoto.evaluated = true;
      this.syncExemplarBankWithLiveScore(targetPhoto);
      this.updatePhotoComplianceBadge(targetPhoto);
      this.renderPresetStrip();
      this.renderAllPanels();
      if (dmEval.inferenceTime !== undefined) {
        const activeCount = this.getEnabledRules().length;
        if (this.el['dv-result-meta']) {
          this.el['dv-result-meta'].textContent =
            `${activeCount}/${this.rules.length} rules · ${Math.round(dmEval.inferenceTime)} ms`;
        }
        this.onStatus?.('Evaluated photo compliance', dmEval.inferenceTime);
      }
    } finally {
      this.evaluatingPhotoIds.delete(targetPhoto.id);
    }
  }

  init() {
    this.root.innerHTML = template;
    this.root.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));

    this.el['dv-guide-btn']?.addEventListener('click', () => this.toggleBiometricOverlay());
    this.el['dv-webcam-btn']?.addEventListener('click', () => this.toggleWebcamBooth());
    this.el['dv-autosnap-btn']?.addEventListener('click', () => this.toggleAutoSnap());
    this.el['dv-snap-btn']?.addEventListener('click', () => this.captureWebcamPhoto(false));
    this.el['dv-retake-btn']?.addEventListener('click', () => this.toggleWebcamBooth());
    this.el['dv-save-photo-btn']?.addEventListener('click', () => this.saveCurrentPhoto());
    this.el['dv-upload-input']?.addEventListener('change', (e) => this.handleCustomPhotoUpload(e));

    this.el['dv-filter-issues-btn']?.addEventListener('click', () => this.toggleIssuesFilter());
    this.el['dv-add-rule-btn']?.addEventListener('click', () => this.openRuleEditor(null));
    this.el['dv-cancel-rule-btn']?.addEventListener('click', () => this.closeRuleEditor());
    this.el['dv-save-rule-btn']?.addEventListener('click', () => this.saveRuleFromEditor());
    this.el['dv-save-pack-btn']?.addEventListener('click', () => this.saveCurrentPackToLibrary());

    this.el['dv-rule-stmt-input']?.addEventListener('input', () => this.syncEditorAnchorsFromStatement());

    this.el['dv-json-btn']?.addEventListener('click', () => this.toggleJsonDrawer());
    this.el['dv-close-json-btn']?.addEventListener('click', () => this.toggleJsonDrawer(false));
    this.el['dv-copy-json-btn']?.addEventListener('click', () => this.copyJsonToClipboard());

    this.el['dv-edit-active-rule-btn']?.addEventListener('click', () => this.openRuleEditor(this.selectedRuleId));
    this.el['dv-delete-active-rule-btn']?.addEventListener('click', () => this.deleteRule(this.selectedRuleId));

    this.el['dv-add-pos-current']?.addEventListener('click', () => this.addCurrentPhotoAsExemplar(true));
    this.el['dv-add-neg-current']?.addEventListener('click', () => this.addCurrentPhotoAsExemplar(false));
    this.el['dv-upload-pos-input']?.addEventListener('change', (e) => this.handleExemplarFileUpload(e, true));
    this.el['dv-upload-neg-input']?.addEventListener('change', (e) => this.handleExemplarFileUpload(e, false));

    this.renderQuickIdeas();
    this.renderPackChips();
    this.renderPresetStrip();
    this.renderAllPanels();
  }

  cleanup() {
    this.stopWebcamBooth();
    if (this.imageSegmenter) {
      try {
        this.imageSegmenter.close();
      } catch {
        // Ignore close errors
      }
      this.imageSegmenter = null;
      this.imageSegmenterPromise = null;
    }
  }

  private getVisionWasmPath(): string {
    const baseUrl = import.meta.env.BASE_URL || '/';
    const formattedBasePath = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    return new URL(`${formattedBasePath}wasm`, window.location.origin).href.replace(/\/$/, '');
  }

  private async ensureImageSegmenter(): Promise<ImageSegmenter | null> {
    if (this.imageSegmenter) return this.imageSegmenter;
    if (this.imageSegmenterPromise) return this.imageSegmenterPromise;

    this.imageSegmenterPromise = (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(this.getVisionWasmPath());
        let modelBytes: ArrayBuffer;
        try {
          modelBytes = await fetchModelBytes(SELFIE_MULTICLASS_MODEL_URL);
          this.isMulticlassSegmenter = true;
        } catch {
          modelBytes = await fetchModelBytes(SELFIE_SEGMENTER_MODEL_URL);
          this.isMulticlassSegmenter = false;
        }

        const createWithDelegate = async (delegate: 'GPU' | 'CPU') =>
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
              modelAssetBuffer: new Uint8Array(modelBytes.slice(0)),
              delegate,
            },
            runningMode: 'IMAGE',
            outputConfidenceMasks: true,
            outputCategoryMask: false,
          });

        try {
          this.imageSegmenter = await createWithDelegate('GPU');
        } catch {
          this.imageSegmenter = await createWithDelegate('CPU');
        }
        return this.imageSegmenter;
      } catch (err) {
        console.warn('MediaPipe ImageSegmenter initialization failed:', err);
        this.imageSegmenterPromise = null;
        return null;
      }
    })();

    return this.imageSegmenterPromise;
  }

  /**
   * Runs MediaPipe Selfie Segmentation on the snapped portrait and replaces
   * the background with pure studio white (`#ffffff`) using a soft-feathered
   * alpha mask when a foreground subject is present (`fgRatio >= 0.02`).
   */
  public async replaceBackgroundWithWhite(
    source: HTMLImageElement | HTMLCanvasElement
  ): Promise<{ dataUrl: string; replaced: boolean; fgRatio: number }> {
    const w = 360;
    const h = 460;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return { dataUrl: '', replaced: false, fgRatio: 0 };
    }
    ctx.drawImage(source, 0, 0, w, h);

    const segmenter = await this.ensureImageSegmenter();
    if (!segmenter) {
      return { dataUrl: canvas.toDataURL('image/jpeg', 0.92), replaced: false, fgRatio: 0 };
    }

    try {
      const result = segmenter.segment(canvas);
      try {
        const masks = result.confidenceMasks;
        if (!masks || masks.length === 0) {
          return { dataUrl: canvas.toDataURL('image/jpeg', 0.92), replaced: false, fgRatio: 0 };
        }

        const nPixels = w * h;
        const rawAlpha = new Float32Array(nPixels);
        let fgCount = 0;

        if (this.isMulticlassSegmenter && masks.length > 1) {
          // Class 0 is background; classes 1..5 are hair, body-skin, face-skin, clothes, accessories
          const bg = masks[0].getAsFloat32Array();
          for (let i = 0; i < nPixels; i++) {
            const fg = 1.0 - bg[i];
            if (fg > 0.5) fgCount++;
            const t = Math.max(0, Math.min(1, (fg - 0.45) / 0.32));
            rawAlpha[i] = t * t * (3 - 2 * t);
          }
        } else {
          const fgArr = masks[0].getAsFloat32Array();
          for (let i = 0; i < nPixels; i++) {
            const fg = fgArr[i];
            if (fg > 0.5) fgCount++;
            const t = Math.max(0, Math.min(1, (fg - 0.42) / 0.32));
            rawAlpha[i] = t * t * (3 - 2 * t);
          }
        }

        const fgRatio = fgCount / nPixels;
        if (fgRatio < 0.02) {
          return { dataUrl: canvas.toDataURL('image/jpeg', 0.92), replaced: false, fgRatio };
        }

        // 3x3 box blur on the alpha mask for a clean, anti-aliased studio edge
        const blurredAlpha = new Float32Array(nPixels);
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            let sum = 0;
            let count = 0;
            for (let dy = -1; dy <= 1; dy++) {
              const ny = y + dy;
              if (ny < 0 || ny >= h) continue;
              for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx;
                if (nx < 0 || nx >= w) continue;
                sum += rawAlpha[ny * w + nx];
                count++;
              }
            }
            blurredAlpha[y * w + x] = sum / count;
          }
        }

        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;
        for (let i = 0; i < nPixels; i++) {
          const alpha = blurredAlpha[i];
          const p = i * 4;
          data[p] = Math.round(data[p] * alpha + 255 * (1 - alpha));
          data[p + 1] = Math.round(data[p + 1] * alpha + 255 * (1 - alpha));
          data[p + 2] = Math.round(data[p + 2] * alpha + 255 * (1 - alpha));
          data[p + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        return { dataUrl: canvas.toDataURL('image/jpeg', 0.94), replaced: true, fgRatio };
      } finally {
        result.close();
      }
    } catch (err) {
      console.warn('MediaPipe background segmentation failed:', err);
      return { dataUrl: canvas.toDataURL('image/jpeg', 0.92), replaced: false, fgRatio: 0 };
    }
  }

  public selectPresetPhoto(photoId: string) {
    if (this.webcamActive) this.stopWebcamBooth();
    this.hideAutoSnapBanner();
    const found = PRESET_PHOTOS.find((x) => x.id === photoId);
    if (!found) return;
    const hasAllCachedRules = Boolean(found.evaluated) && this.rules.every((r) => Boolean(found.rawEg2Vision?.[r.id]));
    if (!hasAllCachedRules) {
      found.evaluated = false;
      found.rawEg2Vision = {};
      found.isCompliant = null;
      found.sub = this.isVisionModelReady() ? 'Evaluating…' : 'Pending…';
    } else {
      this.updatePhotoComplianceBadge(found);
    }
    this.currentPhotoObj = found;
    this.renderPresetStrip();
    this.renderAllPanels();
    if (!hasAllCachedRules && this.isVisionModelReady()) {
      void this.evaluateCurrentPhotoWithDecisionMaker();
    }
  }

  public selectRulePack(packId: string) {
    const allPacks = [...BUILT_IN_RULE_PACKS, ...this.savedCustomPacks];
    const pack = allPacks.find((p) => p.id === packId);
    if (!pack) return;
    this.activePackId = pack.id;
    const idSet = new Set(pack.enabledRuleIds);
    for (const r of this.rules) {
      r.enabled = idSet.has(r.id);
    }
    this.autoSnapStreak = 0;
    this.updatePhotoComplianceBadge(this.currentPhotoObj);
    this.renderPackChips();
    this.renderPresetStrip();
    this.renderAllPanels();
  }

  public toggleRuleEnabled(ruleId: string, enabled?: boolean) {
    const rule = this.rules.find((r) => r.id === ruleId);
    if (!rule) return;
    const nextEnabled = enabled !== undefined ? enabled : rule.enabled === false;
    // Keep at least 1 rule enabled
    if (!nextEnabled && this.getEnabledRules().length <= 1 && rule.enabled !== false) {
      return;
    }
    rule.enabled = nextEnabled;
    this.activePackId = 'custom';
    this.autoSnapStreak = 0;
    this.updatePhotoComplianceBadge(this.currentPhotoObj);
    this.renderPackChips();
    this.renderPresetStrip();
    this.renderAllPanels();
  }

  private saveCurrentPackToLibrary() {
    const enabledIds = this.getEnabledRules().map((r) => r.id);
    const defaultName = `Custom (${enabledIds.length})`;
    const packId = `user_pack_${Date.now()}`;
    const newPack: PhotoRulePack = {
      id: packId,
      name: defaultName,
      subtitle: `${enabledIds.length} active requirements`,
      enabledRuleIds: enabledIds,
      isUserSaved: true,
    };
    this.savedCustomPacks.push(newPack);
    this.activePackId = packId;
    this.persistCustomizations();
    this.renderPackChips();
  }

  private deleteSavedPack(packId: string) {
    this.savedCustomPacks = this.savedCustomPacks.filter((p) => p.id !== packId);
    if (this.activePackId === packId) {
      this.activePackId = 'custom';
    }
    this.persistCustomizations();
    this.renderPackChips();
  }

  private renderPackChips() {
    const container = this.el['dv-pack-chips'];
    if (!container) return;
    container.innerHTML = '';
    const allPacks = [...BUILT_IN_RULE_PACKS, ...this.savedCustomPacks];
    for (const pack of allPacks) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `dv-pack-chip${this.activePackId === pack.id ? ' active' : ''}`;
      btn.dataset.packId = pack.id;
      btn.title = pack.subtitle;
      const span = document.createElement('span');
      span.textContent = pack.name;
      btn.appendChild(span);
      if (pack.isUserSaved) {
        const del = document.createElement('span');
        del.className = 'material-icons dv-pack-del';
        del.textContent = 'close';
        del.title = 'Remove saved preset';
        del.addEventListener('click', (e) => {
          e.stopPropagation();
          this.deleteSavedPack(pack.id);
        });
        btn.appendChild(del);
      }
      btn.addEventListener('click', () => this.selectRulePack(pack.id));
      container.appendChild(btn);
    }

    if (this.activePackId === 'custom') {
      const customChip = document.createElement('span');
      customChip.className = 'dv-pack-chip active';
      customChip.dataset.packId = 'custom';
      customChip.textContent = `Custom (${this.getEnabledRules().length}/${this.rules.length})`;
      container.appendChild(customChip);
    }

    const savePackBtn = this.el['dv-save-pack-btn'];
    if (savePackBtn) {
      savePackBtn.style.display = this.activePackId === 'custom' ? 'inline-flex' : 'none';
    }
  }

  private renderQuickIdeas() {
    const container = this.el['dv-quick-ideas'];
    if (!container) return;
    container.innerHTML = '';
    for (const idea of QUICK_RULE_IDEAS) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'dv-idea-chip';
      chip.textContent = `+ ${idea.shortTitle}`;
      chip.addEventListener('click', () => {
        const labelInput = this.el['dv-rule-label-input'] as HTMLInputElement | undefined;
        const stmtInput = this.el['dv-rule-stmt-input'] as HTMLInputElement | undefined;
        if (labelInput) labelInput.value = idea.shortTitle;
        if (stmtInput) stmtInput.value = idea.statement;
        this.syncEditorAnchorsFromStatement();
      });
      container.appendChild(chip);
    }
  }

  private syncEditorAnchorsFromStatement() {
    const stmtInput = this.el['dv-rule-stmt-input'] as HTMLInputElement | undefined;
    const passInput = this.el['dv-rule-pass-input'] as HTMLInputElement | undefined;
    const failInput = this.el['dv-rule-fail-input'] as HTMLInputElement | undefined;
    if (!stmtInput || !passInput || !failInput) return;
    const synth = buildAnchorsForStatement(stmtInput.value);
    passInput.value = synth.positiveAnchor;
    failInput.value = synth.negativeAnchor;
  }

  public openRuleEditor(ruleId: string | null) {
    const panel = this.el['dv-custom-rule-panel'];
    if (!panel) return;
    this.editingRuleId = ruleId;
    const heading = this.el['dv-editor-heading'];
    const saveLabel = this.el['dv-save-rule-btn-label'];
    const quickRow = this.el['dv-quick-ideas-row'];
    const labelInput = this.el['dv-rule-label-input'] as HTMLInputElement | undefined;
    const stmtInput = this.el['dv-rule-stmt-input'] as HTMLInputElement | undefined;
    const passInput = this.el['dv-rule-pass-input'] as HTMLInputElement | undefined;
    const failInput = this.el['dv-rule-fail-input'] as HTMLInputElement | undefined;

    if (ruleId) {
      const rule = this.rules.find((r) => r.id === ruleId);
      if (!rule) return;
      if (heading) heading.textContent = `Edit Rule: ${rule.title}`;
      if (saveLabel) saveLabel.textContent = 'Update & Re-evaluate Rule';
      if (quickRow) quickRow.style.display = 'none';
      if (labelInput) labelInput.value = rule.shortTitle;
      if (stmtInput) stmtInput.value = rule.statement || rule.prompt;
      if (passInput) passInput.value = rule.options[0]?.description || '';
      if (failInput) failInput.value = rule.options[1]?.description || '';
    } else {
      if (heading) heading.textContent = 'Add Plain-English Photo Requirement';
      if (saveLabel) saveLabel.textContent = 'Add & Evaluate Rule';
      if (quickRow) quickRow.style.display = 'flex';
      if (labelInput) labelInput.value = '';
      if (stmtInput) stmtInput.value = '';
      if (passInput) passInput.value = '';
      if (failInput) failInput.value = '';
    }

    panel.style.display = 'flex';
    labelInput?.focus();
  }

  public closeRuleEditor() {
    const panel = this.el['dv-custom-rule-panel'];
    if (panel) panel.style.display = 'none';
    this.editingRuleId = null;
  }

  private async saveRuleFromEditor() {
    const labelInput = this.el['dv-rule-label-input'] as HTMLInputElement | undefined;
    const stmtInput = this.el['dv-rule-stmt-input'] as HTMLInputElement | undefined;
    const passInput = this.el['dv-rule-pass-input'] as HTMLInputElement | undefined;
    const failInput = this.el['dv-rule-fail-input'] as HTMLInputElement | undefined;

    const rawStatement = (stmtInput?.value ?? '').trim();
    const rawShort = (labelInput?.value ?? '').trim() || rawStatement.slice(0, 22) || 'Custom Rule';
    if (!rawStatement && !rawShort) return;

    const synth = buildAnchorsForStatement(rawStatement || rawShort);
    const posDesc = (passInput?.value ?? '').trim() || synth.positiveAnchor;
    const negDesc = (failInput?.value ?? '').trim() || synth.negativeAnchor;

    if (this.editingRuleId) {
      const rule = this.rules.find((r) => r.id === this.editingRuleId);
      if (rule) {
        rule.shortTitle = rawShort;
        rule.title = `${rule.num}. ${rawShort}`;
        rule.statement = rawStatement || rawShort;
        rule.prompt = synth.prompt;
        if (rule.options.length >= 2) {
          rule.options[0].description = posDesc;
          rule.options[1].description = negDesc;
        }
        rule.optionDisplay[rule.options[0].label] = synth.passDisplay;
        if (rule.options[1]) {
          rule.optionDisplay[rule.options[1].label] = synth.failDisplay;
        }
      }
    } else {
      const num = this.rules.length + 1;
      const newId = `custom_rule_${Date.now()}`;
      const newRule: PassportRule = {
        id: newId,
        num,
        icon: 'verified_user',
        shortTitle: rawShort,
        title: `${num}. ${rawShort}`,
        statement: rawStatement || rawShort,
        normalizePrior: false,
        prompt: synth.prompt,
        options: [
          { label: 'pass_custom', description: posDesc },
          { label: 'fail_custom', description: negDesc },
        ],
        optionDisplay: {
          pass_custom: synth.passDisplay,
          fail_custom: synth.failDisplay,
        },
        enabled: true,
        isCustom: true,
      };
      this.rules.push(newRule);
      this.ensureRuleBank(newId);
      this.selectedRuleId = newId;
      this.activePackId = 'custom';
    }

    for (const p of PRESET_PHOTOS) {
      p.evaluated = false;
    }
    this.currentPhotoObj.evaluated = false;
    this.persistCustomizations();
    this.closeRuleEditor();
    this.renderPackChips();
    this.renderAllPanels();
    if (this.isVisionModelReady() && !this.webcamActive) {
      await this.evaluateCurrentPhotoWithDecisionMaker();
    }
  }

  private deleteRule(ruleId: string) {
    const idx = this.rules.findIndex((r) => r.id === ruleId && r.isCustom);
    if (idx < 0) return;
    this.rules.splice(idx, 1);
    delete this.ruleExemplarBank[ruleId];
    this.rules.forEach((r, i) => {
      r.num = i + 1;
    });
    this.selectedRuleId = this.rules[0]?.id ?? 'full_face';
    this.persistCustomizations();
    this.updatePhotoComplianceBadge(this.currentPhotoObj);
    this.renderPackChips();
    this.renderPresetStrip();
    this.renderAllPanels();
  }

  public toggleIssuesFilter() {
    this.showIssuesOnly = !this.showIssuesOnly;
    const btn = this.el['dv-filter-issues-btn'];
    const label = this.el['dv-filter-issues-label'];
    if (btn) btn.classList.toggle('active', this.showIssuesOnly);
    if (label) label.textContent = this.showIssuesOnly ? 'Issues Only' : 'All Rules';
    this.renderAllPanels();
  }

  public toggleAutoSnap() {
    this.autoSnapOnPass = !this.autoSnapOnPass;
    this.autoSnapStreak = 0;
    const btn = this.el['dv-autosnap-btn'];
    if (btn) {
      btn.classList.toggle('active', this.autoSnapOnPass);
      btn.innerHTML = `<span class="material-icons">hdr_auto</span> Auto-Snap: ${this.autoSnapOnPass ? 'ON' : 'OFF'}`;
    }
  }

  private showAutoSnapBanner(text: string) {
    const banner = this.el['dv-autosnap-banner'];
    const textEl = this.el['dv-autosnap-banner-text'];
    if (textEl) textEl.textContent = text;
    if (banner) banner.style.display = 'flex';
  }

  private hideAutoSnapBanner() {
    const banner = this.el['dv-autosnap-banner'];
    if (banner) banner.style.display = 'none';
  }

  public toggleJsonDrawer(forceState?: boolean) {
    const panel = this.el['dv-json-panel'];
    const btn = this.el['dv-json-btn'];
    if (!panel) return;
    const nextOpen = forceState !== undefined ? forceState : panel.style.display === 'none';
    panel.style.display = nextOpen ? 'flex' : 'none';
    btn?.classList.toggle('active', nextOpen);
    if (nextOpen) {
      this.renderJsonDump();
    }
  }

  private buildJsonDumpObject() {
    const evalRes = this.evaluatePhotoRules(this.currentPhotoObj);
    const activeRules = this.getEnabledRules();
    const passedCount = activeRules.filter((r) => evalRes[r.id]?.passed).length;
    const readinessPct =
      activeRules.length > 0
        ? Math.round((activeRules.reduce((s, r) => s + (evalRes[r.id]?.prob ?? 0), 0) / activeRules.length) * 100)
        : 0;
    const subjectEntry = this.currentPhotoObj.rawEg2Vision?.[SUBJECT_PRESENT_KEY];
    const subjectPresentProb = subjectEntry !== undefined ? subjectEntry.passProb : 1.0;
    const subjectDetected = subjectPresentProb >= SUBJECT_PRESENT_THRESHOLD;
    return {
      photoId: this.currentPhotoObj.id,
      activeRulePack: this.activePackId,
      subjectDetected,
      subjectPresentProbability: Number(subjectPresentProb.toFixed(3)),
      overallCompliant:
        subjectDetected && passedCount === activeRules.length && Boolean(this.currentPhotoObj.evaluated),
      passedRules: `${passedCount}/${activeRules.length}`,
      readinessPercent: readinessPct,
      latencyMs: this.lastInferenceMs !== undefined ? Math.round(this.lastInferenceMs) : null,
      requestSchema: this.buildPassportSchema(),
      results: this.rules.map((r) => {
        const res = evalRes[r.id];
        return {
          id: r.id,
          title: r.shortTitle,
          enabled: r.enabled !== false,
          status: res?.status?.toUpperCase() ?? 'WAIT',
          passProbability: Number((res?.prob ?? 0).toFixed(3)),
          selectedPrototype: res?.chosenOptionLabel ?? '',
          diagnosis: res?.reasonText ?? '',
        };
      }),
    };
  }

  private renderJsonDump() {
    const pre = this.el['dv-json-pre'];
    if (!pre || this.el['dv-json-panel']?.style.display === 'none') return;
    pre.textContent = JSON.stringify(this.buildJsonDumpObject(), null, 2);
  }

  private async copyJsonToClipboard() {
    const text = JSON.stringify(this.buildJsonDumpObject(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      const label = this.el['dv-copy-json-label'];
      if (label) {
        label.textContent = 'Copied!';
        setTimeout(() => {
          label.textContent = 'Copy JSON';
        }, 1500);
      }
    } catch {
      // Clipboard may be unavailable in non-secure test contexts
    }
  }

  public toggleBiometricOverlay() {
    this.showBiometricOverlay = !this.showBiometricOverlay;
    const btn = this.el['dv-guide-btn'];
    if (btn) {
      btn.classList.toggle('active', this.showBiometricOverlay);
      btn.innerHTML = `<span class="material-icons">grid_on</span> Guide: ${this.showBiometricOverlay ? 'ON' : 'OFF'}`;
    }
    this.renderActivePhotoView();
  }

  /**
   * Evaluates all rules purely from `DecisionMaker` schema probabilities
   * (`photoObj.rawEg2Vision`) calibrated against the enabled positive and
   * negative Few-Shot Reference Examples in `this.ruleExemplarBank` — with
   * zero hard-coded biometric or pixel heuristics.
   */
  public evaluatePhotoRules(photoObj: PhotoPreset): Record<string, RuleEvalResult> {
    const results: Record<string, RuleEvalResult> = {};
    const isEvaluated = Boolean(photoObj?.evaluated && Object.keys(photoObj.rawEg2Vision ?? {}).length > 0);
    if (!isEvaluated) {
      for (const rule of this.rules) {
        const isEnabled = rule.enabled !== false;
        results[rule.id] = {
          pending: true,
          enabled: isEnabled,
          status: isEnabled ? 'fail' : 'skipped',
          passed: false,
          prob: 0,
          rawPassProb: 0,
          calPassProb: 0,
          chosenOptionLabel: rule.options[0].label,
          reasonText: !isEnabled
            ? 'Skipped in current pack'
            : this.isVisionModelReady()
              ? 'Evaluating with 435M-VL…'
              : 'Waiting for 435M-VL model…',
        };
      }
      return results;
    }

    const queryLogitVec = this.computeLogitVector(photoObj.rawEg2Vision);
    const subjectEntry = photoObj.rawEg2Vision?.[SUBJECT_PRESENT_KEY];
    const subjectPresentProb = subjectEntry !== undefined ? subjectEntry.passProb : 1.0;
    const isNoFaceScene = subjectPresentProb < SUBJECT_PRESENT_THRESHOLD;
    const isBottomCutoffScene = isNoFaceScene && subjectEntry?.failLabel === SUBJECT_PRESENT_FAIL_PARTIAL;
    const bottomCapProb = Math.min(subjectPresentProb, 0.25);

    for (let rIdx = 0; rIdx < this.rules.length; rIdx++) {
      const rule = this.rules[rIdx];
      const isEnabled = rule.enabled !== false;
      const rawEntry = photoObj.rawEg2Vision[rule.id] || {
        passProb: 0.5,
        topLabel: rule.options[0].label,
        failLabel: rule.options[1]?.label ?? rule.options[0].label,
      };
      const rawPassProb = rawEntry.passProb;
      const rawL = queryLogitVec[rIdx];

      const bank = this.ruleExemplarBank[rule.id];
      const hasExemplarCustomization = Boolean(
        bank &&
        (bank.positives.some((x) => !x.isDefault || !x.enabled) ||
          bank.negatives.some((x) => !x.isDefault || !x.enabled))
      );
      const activePos =
        hasExemplarCustomization && bank ? bank.positives.filter((x) => x.enabled && x.id !== photoObj.id) : [];
      const activeNeg =
        hasExemplarCustomization && bank ? bank.negatives.filter((x) => x.enabled && x.id !== photoObj.id) : [];

      let calPassProb = rawPassProb;
      if (activePos.length > 0 && activeNeg.length > 0) {
        const muPos = activePos.reduce((s, e) => s + e.logitVal, 0) / activePos.length;
        const muNeg = activeNeg.reduce((s, e) => s + e.logitVal, 0) / activeNeg.length;
        const mid = 0.5 * (muPos + muNeg);
        const scale = 2.8 / Math.max(0.35, Math.abs(muPos - muNeg));
        const linearLogit = (rawL - mid) * scale;

        let posWeight = 0;
        for (const e of activePos) {
          const d2 = 0.35 * this.l2DistSq(queryLogitVec, e.logitVec) + 3.5 * (rawL - e.logitVal) ** 2;
          posWeight += Math.exp(-d2 / 2.5);
        }
        let negWeight = 0;
        for (const e of activeNeg) {
          const d2 = 0.35 * this.l2DistSq(queryLogitVec, e.logitVec) + 3.5 * (rawL - e.logitVal) ** 2;
          negWeight += Math.exp(-d2 / 2.5);
        }
        const linProb = sigmoid(linearLogit);
        const knnProb = (posWeight + 0.05 * linProb) / (posWeight + negWeight + 0.05);
        calPassProb = 0.45 * linProb + 0.55 * knnProb;
      } else if (activePos.length > 0) {
        const muPos = activePos.reduce((s, e) => s + e.logitVal, 0) / activePos.length;
        calPassProb = sigmoid((rawL - (muPos - 0.8)) * 2.0);
      } else if (activeNeg.length > 0) {
        const muNeg = activeNeg.reduce((s, e) => s + e.logitVal, 0) / activeNeg.length;
        calPassProb = sigmoid((rawL - (muNeg + 0.8)) * 2.0);
      }

      let finalPassProb = calPassProb;
      let specificFailLabel = rawEntry.failLabel || rule.options[1]?.label || rule.options[0].label;
      let overrideReason: string | undefined;

      // Cross-rule top cutoff guard
      if (
        rule.id === 'eyes_open' &&
        results['full_face'] &&
        !results['full_face'].passed &&
        results['full_face'].chosenOptionLabel === 'fail_top_cut' &&
        results['full_face'].prob < 0.45
      ) {
        finalPassProb = Math.min(finalPassProb, results['full_face'].prob);
        specificFailLabel = 'fail_top_cut';
      }

      // Cross-rule bottom cutoff guard: when the head is cut off at the bottom edge with
      // the mouth, chin, and shoulders below the frame, dependent rules must also fail.
      if (isBottomCutoffScene) {
        if (rule.id === 'full_face') {
          finalPassProb = Math.min(finalPassProb, bottomCapProb);
          specificFailLabel = 'fail_bottom_cut';
        } else if (rule.id === 'shoulders_visible') {
          finalPassProb = Math.min(finalPassProb, bottomCapProb);
          specificFailLabel = 'fail_bottom_cutoff';
        } else if (rule.id === 'neutral_expression' || rule.id === 'framing_chest_up') {
          finalPassProb = Math.min(finalPassProb, bottomCapProb);
          specificFailLabel = 'fail_bottom_cut';
        } else if (rule.id === 'occupancy_70_percent') {
          finalPassProb = Math.min(finalPassProb, bottomCapProb);
          specificFailLabel = 'fail_bottom_peek';
        }
      }

      // Subject Presence Guard: prevent false positives when no human face (or only a partial bottom peek) is in frame
      if (isNoFaceScene) {
        finalPassProb = Math.min(finalPassProb, subjectPresentProb);
        if (subjectEntry?.failLabel === SUBJECT_PRESENT_FAIL_PARTIAL) {
          specificFailLabel =
            rule.optionDisplay[specificFailLabel] && specificFailLabel.startsWith('fail_bottom')
              ? specificFailLabel
              : 'fail_bottom_cut';
          overrideReason =
            rule.optionDisplay[specificFailLabel] ||
            rule.optionDisplay['fail_bottom_cutoff'] ||
            rule.optionDisplay['fail_bottom_peek'] ||
            DEFAULT_BOTTOM_CUTOFF_REASON;
        } else {
          specificFailLabel = 'fail_no_face';
          overrideReason = DEFAULT_NO_FACE_REASON;
        }
      }

      const passed = !isNoFaceScene && finalPassProb >= 0.5;
      const status: RuleStatusTier = !isEnabled ? 'skipped' : passed ? 'pass' : 'fail';
      const chosenOptionLabel = passed ? rule.options[0].label : specificFailLabel;
      const baseReason = overrideReason || rule.optionDisplay[chosenOptionLabel] || chosenOptionLabel;
      const reasonText = !isEnabled ? `Off (${baseReason})` : baseReason;

      results[rule.id] = {
        pending: false,
        enabled: isEnabled,
        status,
        passed,
        prob: finalPassProb,
        rawPassProb,
        calPassProb,
        chosenOptionLabel,
        reasonText,
      };
    }

    return results;
  }

  private renderPresetStrip() {
    const strip = this.el['dv-preset-strip'];
    if (!strip) return;
    strip.innerHTML = '';
    for (const p of PRESET_PHOTOS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'dv-preset-btn' + (!this.webcamActive && this.currentPhotoObj.id === p.id ? ' active' : '');
      btn.dataset.presetId = p.id;
      btn.addEventListener('click', () => this.selectPresetPhoto(p.id));
      const tagClass = p.isCompliant === null ? '' : p.isCompliant ? 'pass' : 'fail';
      const thumbSrc = presetThumbSrc(p);
      const thumbSet = presetThumbSrcSet(p);
      btn.innerHTML = `
        <img class="dv-preset-thumb" src="${thumbSrc}"${thumbSet ? ` srcset="${thumbSet}"` : ''} alt="${p.title}" />
        <div class="dv-preset-title">${p.title}</div>
        <span class="dv-preset-tag ${tagClass}">${p.sub}</span>
      `;
      strip.appendChild(btn);
    }
  }

  private isLiveCapturedPhoto(): boolean {
    return (
      !this.webcamActive &&
      Boolean(this.currentPhotoObj?.id?.startsWith('auto_snap_') || this.currentPhotoObj?.id?.startsWith('live_snap_'))
    );
  }

  public saveCurrentPhoto() {
    if (this.webcamActive || !this.currentPhotoObj?.url) return;
    let href = this.currentPhotoObj.url;
    const imgEl = this.el['dv-main-img'] as HTMLImageElement | undefined;
    if (!href.startsWith('data:') && imgEl && imgEl.complete && imgEl.naturalWidth > 0) {
      try {
        const c = document.createElement('canvas');
        c.width = imgEl.naturalWidth;
        c.height = imgEl.naturalHeight;
        const ctx = c.getContext('2d');
        if (ctx) {
          ctx.drawImage(imgEl, 0, 0);
          href = c.toDataURL('image/jpeg', 0.95);
        }
      } catch {
        // Fallback to direct URL if canvas tainting occurs
      }
    }
    const a = document.createElement('a');
    a.href = href;
    a.download = `photo_id_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    const iconEl = this.el['dv-save-photo-icon'];
    const labelEl = this.el['dv-save-photo-label'];
    if (iconEl && labelEl) {
      iconEl.textContent = 'check';
      labelEl.textContent = 'Saved!';
      setTimeout(() => {
        iconEl.textContent = 'download';
        labelEl.textContent = 'Save Photo';
      }, 1500);
    }
    this.onStatus?.('Saved photo');
  }

  private renderActivePhotoView() {
    const imgEl = this.el['dv-main-img'] as HTMLImageElement | undefined;
    const ambientEl = this.el['dv-ambient-bg'] as HTMLImageElement | undefined;
    const svgEl = this.el['dv-overlay-svg'];
    if (!this.webcamActive && imgEl) {
      imgEl.style.display = 'block';
      if (imgEl.getAttribute('src') !== this.currentPhotoObj.url) {
        imgEl.src = this.currentPhotoObj.url;
      }
      if (ambientEl && ambientEl.getAttribute('src') !== this.currentPhotoObj.url) {
        ambientEl.src = this.currentPhotoObj.url;
      }
    }

    const stageActions = this.el['dv-stage-actions'];
    const retakeBtn = this.el['dv-retake-btn'];
    const savePhotoBtn = this.el['dv-save-photo-btn'];
    const webcamBtnIcon = this.el['dv-webcam-btn-icon'];
    const webcamBtnLabel = this.el['dv-webcam-btn-label'];
    const isLiveCaptured = this.isLiveCapturedPhoto();
    if (stageActions) {
      stageActions.style.display = this.webcamActive ? 'none' : 'flex';
    }
    if (retakeBtn) {
      retakeBtn.style.display = isLiveCaptured ? 'inline-flex' : 'none';
    }
    if (savePhotoBtn) {
      savePhotoBtn.style.display = !this.webcamActive && this.currentPhotoObj?.url ? 'inline-flex' : 'none';
    }
    if (webcamBtnIcon && webcamBtnLabel) {
      if (this.webcamActive) {
        webcamBtnIcon.textContent = 'stop_circle';
        webcamBtnLabel.textContent = 'Stop Webcam';
      } else if (isLiveCaptured) {
        webcamBtnIcon.textContent = 'refresh';
        webcamBtnLabel.textContent = 'Retake';
      } else {
        webcamBtnIcon.textContent = 'videocam';
        webcamBtnLabel.textContent = 'Webcam';
      }
    }

    if (!svgEl) return;
    if (!this.showBiometricOverlay) {
      svgEl.innerHTML = '';
      return;
    }

    const cw = 360;
    const ch = 460;
    const fi = this.currentPhotoObj.biometrics?.faceInfo;
    const subjectEntry = this.currentPhotoObj.rawEg2Vision?.[SUBJECT_PRESENT_KEY];
    const subjectPresentProb = subjectEntry !== undefined ? subjectEntry.passProb : 1.0;
    const isNoFaceScene = subjectPresentProb < SUBJECT_PRESENT_THRESHOLD;
    let overlayMarkup = `
      <ellipse cx="${cw * 0.5}" cy="${ch * 0.42}" rx="${cw * 0.33}" ry="${ch * 0.3}"
        fill="none" stroke="rgba(0, 127, 139, 0.85)" stroke-width="2.2" stroke-dasharray="7 5" />
      <line x1="${cw * 0.18}" y1="${ch * 0.4}" x2="${cw * 0.82}" y2="${ch * 0.4}"
        stroke="rgba(0, 127, 139, 0.45)" stroke-width="1.5" stroke-dasharray="4 4" />
      <path d="M ${cw * 0.05} ${ch * 0.96} Q ${cw * 0.14} ${ch * 0.76} ${cw * 0.34} ${ch * 0.72}"
        fill="none" stroke="rgba(0, 127, 139, 0.55)" stroke-width="2" stroke-dasharray="6 4" />
      <path d="M ${cw * 0.66} ${ch * 0.72} Q ${cw * 0.86} ${ch * 0.76} ${cw * 0.95} ${ch * 0.96}"
        fill="none" stroke="rgba(0, 127, 139, 0.55)" stroke-width="2" stroke-dasharray="6 4" />
    `;

    if (
      this.currentPhotoObj.evaluated &&
      !isNoFaceScene &&
      fi &&
      fi.hasFace &&
      fi.minX !== undefined &&
      fi.minY !== undefined &&
      fi.maxX !== undefined &&
      fi.maxY !== undefined
    ) {
      const bx = fi.minX * cw;
      const by = fi.minY * ch;
      const bw = (fi.maxX - fi.minX) * cw;
      const bh = Math.min(ch - by - 4, (fi.maxY - fi.minY) * ch);
      const evalRes = this.evaluatePhotoRules(this.currentPhotoObj);
      const isGoodBox = Boolean(evalRes['full_face']?.passed && evalRes['framing_chest_up']?.passed);
      const boxColor = isGoodBox ? '#137333' : '#c5221f';
      overlayMarkup += `
        <rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="6"
          fill="none" stroke="${boxColor}" stroke-width="2.5" />
      `;
    }
    svgEl.innerHTML = overlayMarkup;
  }

  private renderAllPanels() {
    this.renderActivePhotoView();

    const evalRes = this.evaluatePhotoRules(this.currentPhotoObj);
    const isPending = Boolean(!this.currentPhotoObj.evaluated);
    const activeRules = this.getEnabledRules();
    let passCount = 0;
    let borderlineCount = 0;
    let sumProb = 0;
    for (const r of activeRules) {
      const res = evalRes[r.id];
      if (res?.passed) passCount++;
      else if (res?.status === 'borderline') borderlineCount++;
      sumProb += res?.prob ?? 0;
    }
    const readinessPct = !isPending && activeRules.length > 0 ? Math.round((sumProb / activeRules.length) * 100) : 0;
    const allPassed = !isPending && passCount === activeRules.length;

    const readinessLabel = this.el['dv-readiness-label'];
    const readinessFill = this.el['dv-readiness-fill'];
    if (readinessLabel && readinessFill) {
      if (isPending) {
        readinessLabel.textContent = `Readiness: -- (${activeRules.length} active)`;
        readinessFill.style.width = '0%';
        readinessFill.className = 'dv-readiness-fill';
      } else {
        readinessLabel.textContent = `${passCount}/${activeRules.length} Passed · ${readinessPct}% Ready`;
        readinessFill.style.width = `${readinessPct}%`;
        readinessFill.className = `dv-readiness-fill${allPassed ? '' : readinessPct >= 65 ? ' warn' : ' fail'}`;
      }
    }

    const banner = this.el['dv-verdict-banner'];
    const iconEl = this.el['dv-verdict-icon'];
    const titleEl = this.el['dv-verdict-title'];
    const subEl = this.el['dv-verdict-sub'];
    if (banner && iconEl && titleEl && subEl) {
      if (isPending) {
        banner.className = 'dv-verdict-banner';
        iconEl.textContent = 'hourglass_top';
        titleEl.textContent = this.isVisionModelReady() ? 'EVALUATING…' : 'LOADING MODEL…';
        subEl.textContent = this.isVisionModelReady() ? `${activeRules.length} Rules` : 'Please wait';
      } else {
        const failCount = activeRules.length - passCount;
        const onlyBorderline = failCount > 0 && borderlineCount === failCount;
        banner.className = 'dv-verdict-banner ' + (allPassed ? 'pass' : onlyBorderline ? 'warn' : 'fail');
        iconEl.textContent = allPassed ? 'check_circle' : onlyBorderline ? 'warning_amber' : 'error';
        titleEl.textContent = allPassed
          ? 'COMPLIANT'
          : onlyBorderline
            ? `${borderlineCount} BORDERLINE`
            : `${failCount} ISSUE${failCount > 1 ? 'S' : ''}`;
        subEl.textContent = `${passCount}/${activeRules.length} Passed · ${readinessPct}%`;
      }
    }

    const listEl = this.el['dv-rules-list'];
    if (listEl) {
      if (listEl.children.length !== this.rules.length) {
        listEl.innerHTML = '';
        for (const rule of this.rules) {
          const card = document.createElement('div');
          card.dataset.ruleId = rule.id;
          card.addEventListener('click', () => {
            this.selectedRuleId = rule.id;
            this.renderAllPanels();
          });
          card.innerHTML = `
            <div class="dv-rule-top">
              <div class="dv-rule-name">
                <span class="material-icons">${rule.icon}</span>
                <span class="dv-rule-short-title">${rule.shortTitle}</span>
              </div>
              <div class="dv-rule-right">
                <input type="checkbox" class="dv-rule-toggle" title="Toggle rule in active pack" />
                <span class="dv-rule-badge"></span>
              </div>
            </div>
            <div class="dv-rule-reason"></div>
            <div class="dv-rule-bar-track">
              <div class="dv-rule-bar-fill"></div>
            </div>
          `;
          const toggleEl = card.querySelector('.dv-rule-toggle') as HTMLInputElement;
          toggleEl.addEventListener('click', (e) => e.stopPropagation());
          toggleEl.addEventListener('change', () => {
            this.toggleRuleEnabled(rule.id, toggleEl.checked);
          });
          listEl.appendChild(card);
        }
      }
      for (let i = 0; i < this.rules.length; i++) {
        const rule = this.rules[i];
        const r = evalRes[rule.id];
        const pct = Math.round(r.prob * 100);
        const card = listEl.children[i] as HTMLElement;
        const shortTitleEl = card.querySelector('.dv-rule-short-title') as HTMLElement;
        const toggleEl = card.querySelector('.dv-rule-toggle') as HTMLInputElement;
        const badgeEl = card.querySelector('.dv-rule-badge') as HTMLElement;
        const reasonEl = card.querySelector('.dv-rule-reason') as HTMLElement;
        const fillEl = card.querySelector('.dv-rule-bar-fill') as HTMLElement;

        if (shortTitleEl) shortTitleEl.textContent = rule.shortTitle;
        if (toggleEl) toggleEl.checked = rule.enabled !== false;

        const hideForIssuesFilter =
          this.showIssuesOnly && !r.pending && (r.status === 'pass' || r.status === 'skipped');
        card.style.display = hideForIssuesFilter ? 'none' : 'flex';

        if (!r.enabled) {
          card.className = `dv-rule-card skipped${this.selectedRuleId === rule.id ? ' selected' : ''}`;
          badgeEl.className = 'dv-rule-badge skipped';
          badgeEl.textContent = 'SKIPPED';
          reasonEl.textContent = r.reasonText;
          fillEl.className = 'dv-rule-bar-fill skipped';
          fillEl.style.width = `${pct}%`;
        } else if (r.pending) {
          card.className = `dv-rule-card${this.selectedRuleId === rule.id ? ' selected' : ''}`;
          badgeEl.className = 'dv-rule-badge';
          badgeEl.textContent = 'WAIT';
          reasonEl.textContent = r.reasonText;
          fillEl.className = 'dv-rule-bar-fill';
          fillEl.style.width = '0%';
        } else {
          const tierClass = r.status;
          card.className = `dv-rule-card ${tierClass}${this.selectedRuleId === rule.id ? ' selected' : ''}`;
          badgeEl.className = `dv-rule-badge ${tierClass}`;
          badgeEl.textContent = r.status === 'pass' ? 'PASS' : r.status === 'borderline' ? 'BORDERLINE' : 'FAIL';
          reasonEl.textContent = r.reasonText;
          fillEl.className = `dv-rule-bar-fill ${tierClass}`;
          fillEl.style.width = `${pct}%`;
        }
      }
    }

    this.renderExemplarStudioPanel(evalRes);
    this.renderJsonDump();
  }

  private renderExemplarStudioPanel(evalRes: Record<string, RuleEvalResult>) {
    const rule = this.rules.find((r) => r.id === this.selectedRuleId) || this.rules[0];
    this.ensureRuleBank(rule.id);
    const rRes = evalRes[rule.id];
    const bank = this.ruleExemplarBank[rule.id];

    if (this.el['dv-active-rule-title']) this.el['dv-active-rule-title'].textContent = rule.title;
    if (this.el['dv-active-rule-prompt']) this.el['dv-active-rule-prompt'].textContent = rule.prompt;

    const delBtn = this.el['dv-delete-active-rule-btn'];
    if (delBtn) delBtn.style.display = rule.isCustom ? 'inline-flex' : 'none';

    const anchorsEl = this.el['dv-active-rule-anchors'];
    if (anchorsEl) {
      anchorsEl.innerHTML = '';
      rule.options.forEach((opt, idx) => {
        const pill = document.createElement('div');
        pill.className = `dv-anchor-pill ${idx === 0 ? 'pass' : 'fail'}`;
        const tag = idx === 0 ? '✓ Pass Anchor:' : `✗ Fail Anchor ${rule.options.length > 2 ? `#${idx}` : ''}:`;
        pill.innerHTML = `<strong>${tag}</strong> <span>${opt.description}</span>`;
        anchorsEl.appendChild(pill);
      });
    }

    const statusBadge = this.el['dv-active-rule-badge'];
    if (statusBadge) {
      if (!rRes.enabled) {
        statusBadge.className = 'dv-rule-badge skipped';
        statusBadge.textContent = `SKIPPED (${Math.round(rRes.prob * 100)}%)`;
      } else if (rRes.pending) {
        statusBadge.className = 'dv-rule-badge';
        statusBadge.textContent = 'WAIT';
      } else {
        statusBadge.className = `dv-rule-badge ${rRes.status}`;
        const label = rRes.status === 'pass' ? 'PASS' : rRes.status === 'borderline' ? 'BORDERLINE' : 'FAIL';
        statusBadge.textContent = `${label} (${Math.round(rRes.prob * 100)}%)`;
      }
    }

    const renderList = (container: HTMLElement | undefined, items: ExemplarEntry[], isPositive: boolean) => {
      if (!container) return;
      container.innerHTML = '';
      items.forEach((item) => {
        const label = document.createElement('label');
        label.className = 'dv-exemplar-item' + (item.enabled ? '' : ' off');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = item.enabled;
        checkbox.addEventListener('change', () => {
          item.enabled = checkbox.checked;
          this.renderAllPanels();
        });
        const img = document.createElement('img');
        img.className = 'dv-exemplar-thumb';
        img.src = exemplarThumbSrc(item);
        const srcSet = exemplarThumbSrcSet(item);
        if (srcSet) img.srcset = srcSet;
        img.alt = item.name;
        const span = document.createElement('span');
        span.textContent = item.name;
        label.append(checkbox, img, span);
        container.appendChild(label);
      });
      const badgeEl = this.el[isPositive ? 'dv-pos-count' : 'dv-neg-count'];
      if (badgeEl) badgeEl.textContent = String(items.filter((x) => x.enabled).length);
    };

    renderList(this.el['dv-pos-list'], bank.positives, true);
    renderList(this.el['dv-neg-list'], bank.negatives, false);
  }

  private addCurrentPhotoAsExemplar(isPositive: boolean) {
    if (!this.currentPhotoObj.evaluated) return;
    const ruleId = this.selectedRuleId;
    this.ensureRuleBank(ruleId);
    const rawProb = this.currentPhotoObj.rawEg2Vision[ruleId]?.passProb ?? 0.5;
    const entry: ExemplarEntry = {
      id: `${this.currentPhotoObj.id}_${Date.now()}`,
      name: this.currentPhotoObj.title,
      url: this.currentPhotoObj.url,
      rawPassProb: rawProb,
      logitVal: logit(rawProb),
      logitVec: this.computeLogitVector(this.currentPhotoObj.rawEg2Vision),
      enabled: true,
    };
    if (isPositive) this.ruleExemplarBank[ruleId].positives.unshift(entry);
    else this.ruleExemplarBank[ruleId].negatives.unshift(entry);
    this.renderAllPanels();
  }

  private async handleExemplarFileUpload(event: Event, isPositive: boolean) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const dataUrl = await new Promise<string>((res) => {
      const r = new FileReader();
      r.onload = () => res(String(r.result));
      r.readAsDataURL(file);
    });
    const ruleId = this.selectedRuleId;
    this.ensureRuleBank(ruleId);
    let rawEg2Vision: Record<string, RawRuleVisionScore> = {};

    const img = await new Promise<HTMLImageElement>((res) => {
      const el = new Image();
      el.onload = () => res(el);
      el.src = dataUrl;
    });
    const dmEval = await this.evaluateSourceWithDecisionMaker(img);
    if (dmEval) rawEg2Vision = dmEval.rawEg2Vision;

    const rawProb = rawEg2Vision[ruleId]?.passProb ?? (isPositive ? 0.8 : 0.2);
    const entry: ExemplarEntry = {
      id: `upload_${Date.now()}`,
      name: file.name.slice(0, 14),
      url: dataUrl,
      rawPassProb: rawProb,
      logitVal: logit(rawProb),
      logitVec: this.computeLogitVector(rawEg2Vision),
      enabled: true,
    };
    if (isPositive) this.ruleExemplarBank[ruleId].positives.unshift(entry);
    else this.ruleExemplarBank[ruleId].negatives.unshift(entry);
    input.value = '';
    this.renderAllPanels();
  }

  private async evaluateCustomImageElement(
    img: HTMLImageElement,
    titleLabel: string,
    wasAutoSnapped = false,
    idPrefix = 'custom',
    bgReplacedToWhite = false
  ) {
    const t0 = performance.now();
    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = 'Evaluating photo…';
    }
    const pendingObj: PhotoPreset = {
      id: `${idPrefix}_${Date.now()}`,
      title: titleLabel,
      sub: wasAutoSnapped ? 'Auto-Snapped' : 'Evaluating…',
      isCompliant: null,
      evaluated: false,
      url: img.src,
      biometrics: {
        bgLum: bgReplacedToWhite ? 255 : 240,
        bgVar: bgReplacedToWhite ? 0 : 5,
        isPlainWhiteBg: true,
        fgRatio: 0.68,
        bottomShoulderWidthRatio: 0.85,
        faceInfo: { hasFace: false },
      },
      rawEg2Vision: {},
    };
    this.currentPhotoObj = pendingObj;
    this.renderPresetStrip();
    this.renderAllPanels();

    const dmEval = await this.evaluateSourceWithDecisionMaker(img);
    const rawEg2Vision: Record<string, RawRuleVisionScore> = dmEval ? dmEval.rawEg2Vision : {};
    if (bgReplacedToWhite && (rawEg2Vision[SUBJECT_PRESENT_KEY]?.passProb ?? 1.0) >= SUBJECT_PRESENT_THRESHOLD) {
      rawEg2Vision['neutral_background'] = {
        passProb: Math.max(rawEg2Vision['neutral_background']?.passProb ?? 0.92, 0.92),
        topLabel: 'pass_plain',
        failLabel: 'fail_cluttered',
      };
    }
    if (dmEval?.inferenceTime !== undefined) {
      this.lastInferenceMs = dmEval.inferenceTime;
    }

    pendingObj.rawEg2Vision = rawEg2Vision;
    pendingObj.evaluated = Boolean(dmEval);
    this.updatePhotoComplianceBadge(pendingObj);
    this.renderPresetStrip();
    this.renderAllPanels();

    if (dmEval) {
      const elapsed = dmEval.inferenceTime ?? performance.now() - t0;
      const activeCount = this.getEnabledRules().length;
      if (this.el['dv-result-meta']) {
        this.el['dv-result-meta'].textContent = `${activeCount}/${this.rules.length} rules · ${Math.round(elapsed)} ms`;
      }
      this.onStatus?.('Evaluated photo compliance', elapsed);
    }
  }

  private async handleCustomPhotoUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (this.webcamActive) this.stopWebcamBooth();
    this.hideAutoSnapBanner();
    const dataUrl = await new Promise<string>((res) => {
      const r = new FileReader();
      r.onload = () => res(String(r.result));
      r.readAsDataURL(file);
    });
    const img = new Image();
    img.onload = () => this.evaluateCustomImageElement(img, file.name.slice(0, 12));
    img.src = dataUrl;
    input.value = '';
  }

  private drawMirroredVideoToCanvas(video: HTMLVideoElement, targetCanvas: HTMLCanvasElement) {
    const sctx = targetCanvas.getContext('2d')!;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const targetAspect = targetCanvas.width / targetCanvas.height;
    const videoAspect = vw / vh;
    let sx = 0;
    let sy = 0;
    let sw = vw;
    let sh = vh;
    if (videoAspect > targetAspect) {
      sw = vh * targetAspect;
      sx = (vw - sw) * 0.5;
    } else {
      sh = vw / targetAspect;
      sy = (vh - sh) * 0.5;
    }
    sctx.save();
    sctx.translate(targetCanvas.width, 0);
    sctx.scale(-1, 1);
    sctx.drawImage(video, sx, sy, sw, sh, 0, 0, targetCanvas.width, targetCanvas.height);
    sctx.restore();
  }

  private async runLiveWebcamLoop() {
    if (!this.webcamActive) return;
    const video = this.el['dv-webcam-video'] as HTMLVideoElement | undefined;
    if (video && video.readyState >= 2) {
      const t0 = performance.now();
      this.drawMirroredVideoToCanvas(video, this.liveFrameCanvas);
      const frameDataUrl = this.liveFrameCanvas.toDataURL('image/jpeg', 0.92);
      this.latestLiveFrameDataUrl = frameDataUrl;

      const dmEval = await this.evaluateSourceWithDecisionMaker(
        this.liveFrameCanvas,
        this.currentPhotoObj.rawEg2Vision
      );
      if (!this.webcamActive) return;

      if (dmEval) {
        this.lastInferenceMs = dmEval.inferenceTime;
        this.currentPhotoObj.rawEg2Vision = dmEval.rawEg2Vision;
        this.currentPhotoObj.evaluated = true;
        this.renderAllPanels();

        const evalRes = this.evaluatePhotoRules(this.currentPhotoObj);
        const activeRules = this.getEnabledRules();
        // Since snapping replaces the background with plain white via MediaPipe Segmentation,
        // allow auto-snap to trigger as soon as all non-background requirements pass.
        const allActivePassed =
          activeRules.length > 0 && activeRules.every((r) => r.id === 'neutral_background' || evalRes[r.id]?.passed);

        if (this.autoSnapOnPass && allActivePassed) {
          this.bestStreakFrameDataUrl = frameDataUrl;
          this.autoSnapStreak++;
          if (this.autoSnapStreak >= this.autoSnapRequiredStreak) {
            await this.captureWebcamPhoto(true);
            return;
          }
        } else {
          this.autoSnapStreak = 0;
          this.bestStreakFrameDataUrl = null;
        }

        const elapsed = Math.max(8, Math.round(dmEval.inferenceTime ?? performance.now() - t0));
        if (this.el['dv-result-meta']) {
          const streakHint =
            this.autoSnapOnPass && this.autoSnapStreak > 0
              ? ` · Holding pass (${this.autoSnapStreak}/${this.autoSnapRequiredStreak})`
              : '';
          this.el['dv-result-meta'].textContent = `Live · ${elapsed} ms${streakHint}`;
        }
      }
    }
    if (this.webcamActive) {
      this.liveLoopTimer = setTimeout(() => this.runLiveWebcamLoop(), 140);
    }
  }

  public async toggleWebcamBooth() {
    if (this.webcamActive) {
      this.stopWebcamBooth();
      this.currentPhotoObj = PRESET_PHOTOS[0];
      this.renderPresetStrip();
      this.renderAllPanels();
      if (this.isVisionModelReady() && !this.currentPhotoObj.evaluated) {
        void this.evaluateCurrentPhotoWithDecisionMaker();
      }
      return;
    }
    try {
      this.hideAutoSnapBanner();
      const video = this.el['dv-webcam-video'] as HTMLVideoElement;
      const imgEl = this.el['dv-main-img'] as HTMLImageElement;
      this.webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      video.srcObject = this.webcamStream;
      await video.play();
      this.webcamActive = true;
      this.autoSnapStreak = 0;
      this.bestStreakFrameDataUrl = null;
      this.latestLiveFrameDataUrl = null;

      imgEl.style.display = 'none';
      video.style.display = 'block';
      if (this.el['dv-webcam-btn-icon']) this.el['dv-webcam-btn-icon'].textContent = 'stop_circle';
      if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Stop Webcam';
      this.el['dv-webcam-btn']?.classList.add('active');
      if (this.el['dv-autosnap-btn']) this.el['dv-autosnap-btn'].style.display = 'inline-flex';
      if (this.el['dv-snap-btn']) this.el['dv-snap-btn'].style.display = 'inline-flex';
      if (this.el['dv-result-meta']) this.el['dv-result-meta'].textContent = 'Evaluating live frame…';

      this.currentPhotoObj = {
        id: 'live_webcam',
        title: 'Live Camera',
        sub: 'Live',
        isCompliant: null,
        evaluated: false,
        url: this.currentPhotoObj.url,
        biometrics: {
          bgLum: 220,
          bgVar: 10,
          isPlainWhiteBg: false,
          fgRatio: 0.65,
          bottomShoulderWidthRatio: 0.85,
          faceInfo: { hasFace: false },
        },
        rawEg2Vision: {},
      };
      this.renderPresetStrip();
      this.renderAllPanels();
      // Pre-warm MediaPipe Selfie Segmenter while the webcam is running so background
      // replacement is instantaneous when the photo is snapped.
      void this.ensureImageSegmenter();
      this.runLiveWebcamLoop();
    } catch (err: any) {
      this.onStatus?.(`Webcam error: ${err?.message || err}`);
    }
  }

  public stopWebcamBooth() {
    this.webcamActive = false;
    this.autoSnapStreak = 0;
    if (this.liveLoopTimer) {
      clearTimeout(this.liveLoopTimer);
      this.liveLoopTimer = null;
    }
    if (this.webcamStream) {
      this.webcamStream.getTracks().forEach((t) => t.stop());
      this.webcamStream = null;
    }
    const video = this.el['dv-webcam-video'] as HTMLVideoElement | undefined;
    const imgEl = this.el['dv-main-img'] as HTMLImageElement | undefined;
    if (video) video.style.display = 'none';
    if (imgEl) imgEl.style.display = 'block';
    if (this.el['dv-webcam-btn-icon']) this.el['dv-webcam-btn-icon'].textContent = 'videocam';
    if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Webcam';
    this.el['dv-webcam-btn']?.classList.remove('active');
    if (this.el['dv-autosnap-btn']) this.el['dv-autosnap-btn'].style.display = 'none';
    if (this.el['dv-snap-btn']) this.el['dv-snap-btn'].style.display = 'none';
    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = `${this.getEnabledRules().length}/${this.rules.length} rules`;
    }
  }

  private async captureWebcamPhoto(wasAutoSnapped = false) {
    if (!this.webcamActive) return;
    const bestStreakUrl = wasAutoSnapped ? this.bestStreakFrameDataUrl : null;
    const recentLiveUrl = !wasAutoSnapped ? this.latestLiveFrameDataUrl : null;
    const video = this.el['dv-webcam-video'] as HTMLVideoElement;
    this.drawMirroredVideoToCanvas(video, this.liveFrameCanvas);
    this.stopWebcamBooth();

    const rawDataUrl =
      (wasAutoSnapped ? bestStreakUrl : recentLiveUrl) || this.liveFrameCanvas.toDataURL('image/jpeg', 0.92);
    this.bestStreakFrameDataUrl = null;
    this.latestLiveFrameDataUrl = null;

    const rawImg = await new Promise<HTMLImageElement>((resolve) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => resolve(el);
      el.src = rawDataUrl;
    });

    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = 'Segmenting background (MediaPipe)…';
    }
    this.onStatus?.('Segmenting background to plain white…');

    const segResult = await this.replaceBackgroundWithWhite(rawImg);
    if (segResult.replaced) {
      this.showAutoSnapBanner(
        wasAutoSnapped
          ? 'Auto-snapped & background replaced with plain white (MP Segmentation)!'
          : 'Photo snapped — background replaced with plain white (MP Segmentation)'
      );
      const whiteBgImg = await new Promise<HTMLImageElement>((resolve) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () => resolve(rawImg);
        el.src = segResult.dataUrl;
      });
      await this.evaluateCustomImageElement(
        whiteBgImg,
        wasAutoSnapped ? 'Auto-Snapped Photo' : 'Camera Photo',
        wasAutoSnapped,
        wasAutoSnapped ? 'auto_snap' : 'live_snap',
        true
      );
    } else {
      if (wasAutoSnapped) {
        this.showAutoSnapBanner('Auto-snapped — all active requirements passed!');
      }
      await this.evaluateCustomImageElement(
        rawImg,
        wasAutoSnapped ? 'Auto-Snapped Photo' : 'Camera Photo',
        wasAutoSnapped,
        wasAutoSnapped ? 'auto_snap' : 'live_snap',
        false
      );
    }
  }
}
