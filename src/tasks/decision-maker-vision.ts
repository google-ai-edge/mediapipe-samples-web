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
 * Checker that evaluates 9 biometric and visual rules on preset portraits,
 * uploaded photos, or a live webcam stream, with interactive positive and
 * negative reference examples per rule.
 */

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import template from '../templates/decision-maker-vision.html?raw';

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
  prompt: string;
  options: PassportRuleOption[];
  optionDisplay: Record<string, string>;
}

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
  isCompliant: boolean;
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
  enabled: boolean;
}

export interface RuleEvalResult {
  passed: boolean;
  prob: number;
  rawPassProb: number;
  calPassProb: number;
  chosenOptionLabel: string;
  reasonText: string;
}

const FACE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export const PASSPORT_RULES: PassportRule[] = [
  {
    id: 'full_face',
    num: 1,
    icon: 'face',
    shortTitle: 'Full Face',
    title: '1. Full Face Present',
    prompt: 'Is a complete, unobstructed, centered frontal face present inside the frame?',
    options: [
      {
        label: 'pass_frontal',
        description:
          'Complete unobstructed frontal portrait face centered in frame with full forehead, chin, nose, and mouth clearly visible.',
      },
      {
        label: 'fail_cropped',
        description: 'Extreme close-up face with forehead or chin cut off at the edges of the image.',
      },
      {
        label: 'fail_masked',
        description: 'Face covered by a black face mask, respirator, cloth covering, or hand.',
      },
      {
        label: 'fail_distant',
        description: 'Distant person standing far away so the face is tiny.',
      },
    ],
    optionDisplay: {
      pass_frontal: 'Centered & straight frontal',
      fail_off_center_and_tilted: 'Off-center & head tilted',
      fail_off_center: 'Off-center in frame',
      fail_turned_tilted: 'Head turned or tilted',
      fail_cropped: 'Head/chin clipped at edge',
      fail_masked: 'Face covered by mask',
      fail_distant: 'Subject too far away',
    },
  },
  {
    id: 'eyes_open',
    num: 2,
    icon: 'visibility',
    shortTitle: 'Eyes Open',
    title: '2. Eyes Open & Direct',
    prompt: 'Are both eyes clearly open, visible, and looking straight at the camera?',
    options: [
      {
        label: 'pass_open',
        description: 'Both eyes are wide open with visible pupils and irises looking directly at the camera lens.',
      },
      {
        label: 'fail_closed',
        description: 'Both eyes are shut closed, blinking, or sleeping with eyelids closed.',
      },
      {
        label: 'fail_sunglasses',
        description: 'Eyes are hidden behind dark tinted sunglasses or dark lenses.',
      },
    ],
    optionDisplay: {
      pass_open: 'Open & looking at camera',
      fail_looking_away: 'Not looking straight at camera',
      fail_closed: 'Eyes closed / blinking',
      fail_sunglasses: 'Hidden behind dark lenses',
    },
  },
  {
    id: 'neutral_expression',
    num: 3,
    icon: 'sentiment_neutral',
    shortTitle: 'Neutral Face',
    title: '3. Neutral Expression',
    prompt: 'Does the person have a calm, neutral facial expression with mouth closed?',
    options: [
      {
        label: 'pass_neutral',
        description: 'Calm, serious, neutral facial expression with lips and mouth closed.',
      },
      {
        label: 'fail_smiling',
        description: 'Smiling broadly, grinning, laughing, or showing teeth with an open mouth.',
      },
    ],
    optionDisplay: {
      pass_neutral: 'Neutral expression, mouth closed',
      fail_smiling: 'Smiling / teeth visible',
    },
  },
  {
    id: 'neutral_background',
    num: 4,
    icon: 'crop_portrait',
    shortTitle: 'White BG',
    title: '4. Neutral White Background',
    prompt: 'Is the background a plain, uniform white or off-white studio backdrop?',
    options: [
      {
        label: 'pass_white_bg',
        description: 'Plain, uniform, solid white studio background backdrop.',
      },
      {
        label: 'fail_indoor_clutter',
        description: 'Busy indoor room background showing bookshelves, doors, furniture, or walls.',
      },
      {
        label: 'fail_outdoor_street',
        description: 'Busy outdoor street scene with trees, buildings, sky, bicycles, or pedestrians.',
      },
    ],
    optionDisplay: {
      pass_white_bg: 'Plain uniform white backdrop',
      fail_indoor_clutter: 'Indoor room background',
      fail_outdoor_street: 'Outdoor street background',
    },
  },
  {
    id: 'shoulders_visible',
    num: 5,
    icon: 'checkroom',
    shortTitle: 'Shoulders',
    title: '5. Shoulders Visible & Square',
    prompt: 'Are both upper shoulders clearly visible and symmetrical at the bottom?',
    options: [
      {
        label: 'pass_shoulders',
        description:
          'Both left and right upper shoulders in suit jacket or shirt are symmetrical and visible at the bottom of the portrait.',
      },
      {
        label: 'fail_no_shoulders',
        description: 'Extreme face-only macro close-up cropped at the chin with zero neck and zero shoulders visible.',
      },
      {
        label: 'fail_angled_selfie',
        description: 'Angled one-handed selfie posture with one arm extended toward the camera.',
      },
    ],
    optionDisplay: {
      pass_shoulders: 'Both shoulders visible',
      fail_no_shoulders: 'Cropped at chin (no shoulders)',
      fail_angled_selfie: 'Angled selfie posture',
    },
  },
  {
    id: 'hair_clear',
    num: 6,
    icon: 'content_cut',
    shortTitle: 'Hair Clear',
    title: '6. Hair Not Covering Face',
    prompt: 'Is the hair neat and clear of the face, eyebrows, and eyes, with no hat?',
    options: [
      {
        label: 'pass_hair_clear',
        description:
          'Neat hair tucked behind ears or short hair with forehead, both eyebrows, and both eyes completely clear and visible.',
      },
      {
        label: 'fail_bangs_over_eyes',
        description: 'Long messy hair bangs hanging down covering the forehead, eyebrows, and eyes.',
      },
      {
        label: 'fail_hat_cap',
        description: 'Wearing a hat, cap, or headgear covering the head.',
      },
    ],
    optionDisplay: {
      pass_hair_clear: 'Face & eyebrows unobstructed',
      fail_bangs_over_eyes: 'Bangs covering eyes/brow',
      fail_hat_cap: 'Hat / cap worn',
    },
  },
  {
    id: 'no_glasses',
    num: 7,
    icon: 'visibility_off',
    shortTitle: 'No Glasses',
    title: '7. No Eyeglasses or Sunglasses',
    prompt: 'Are the eyes bare and free of eyeglasses or sunglasses?',
    options: [
      {
        label: 'pass_bare_eyes',
        description: 'Bare natural face and eyes with clear skin around the eyes and bridge of the nose.',
      },
      {
        label: 'fail_eyeglasses',
        description: 'Wearing prescription eyeglasses, reading glasses, or optical frames over the eyes.',
      },
      {
        label: 'fail_sunglasses',
        description: 'Wearing dark tinted sunglasses with black frames covering the eyes.',
      },
    ],
    optionDisplay: {
      pass_bare_eyes: 'No glasses or frames',
      fail_eyeglasses: 'Eyeglasses worn',
      fail_sunglasses: 'Dark sunglasses worn',
    },
  },
  {
    id: 'framing_chest_up',
    num: 8,
    icon: 'aspect_ratio',
    shortTitle: 'Chest-Up',
    title: '8. Close-Up to Upper Chest',
    prompt: 'Is the photo framed as a head-and-shoulders close-up down to the upper chest?',
    options: [
      {
        label: 'pass_chest_up',
        description: 'Standard passport head-and-shoulders bust portrait cropped at the upper chest and collarbone.',
      },
      {
        label: 'fail_full_body',
        description: 'Wide full-body shot showing waist, stomach, jeans, legs, or arms far below the chest.',
      },
      {
        label: 'fail_macro_face',
        description: 'Extreme macro close-up of the face only, cropped at the chin without neck or chest.',
      },
    ],
    optionDisplay: {
      pass_chest_up: 'Head & upper chest framing',
      fail_off_center: 'Not centered in guide oval',
      fail_full_body: 'Shows waist/body below chest',
      fail_macro_face: 'Zoomed too close (face only)',
    },
  },
  {
    id: 'occupancy_70_percent',
    num: 9,
    icon: 'photo_size_select_large',
    shortTitle: '~70% Frame',
    title: '9. Occupies ~70% of Photo',
    prompt: 'Does the person occupy approximately 70% of the photo frame?',
    options: [
      {
        label: 'pass_70_pct',
        description:
          'Well-proportioned passport bust portrait where head and shoulders fill 70% of the frame with white margin above the head.',
      },
      {
        label: 'fail_too_small',
        description: 'Distant small person occupying less than 35% of the frame.',
      },
      {
        label: 'fail_too_large',
        description: 'Zoomed-in giant face filling 98% of the frame with cut-off edges.',
      },
    ],
    optionDisplay: {
      pass_70_pct: 'Subject fills ~70% of frame',
      fail_too_small: 'Too small in frame (<35%)',
      fail_too_large: 'Overfills frame (>90%)',
    },
  },
];

function photoAssetUrl(fileName: string): string {
  const base = import.meta.env.BASE_URL || '/';
  const normalizedBase = base.endsWith('/') ? base : `${base}/`;
  return `${normalizedBase}images/id_photos/${fileName}`;
}

export const PRESET_PHOTOS: PhotoPreset[] = [
  {
    id: 'passport_compliant_female.jpg',
    title: 'Compliant #1',
    sub: '9/9 Pass',
    isCompliant: true,
    url: photoAssetUrl('passport_compliant_female.jpg'),
    groundTruth: {
      full_face: true,
      eyes_open: true,
      neutral_expression: true,
      neutral_background: true,
      shoulders_visible: true,
      hair_clear: true,
      no_glasses: true,
      framing_chest_up: true,
      occupancy_70_percent: true,
    },
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
        blink: 0.031,
        smile: 0.003,
        jawOpen: 0.003,
        foreheadDarkDiff: 26,
        bridgeContrast: 92,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.72, topLabel: 'pass_frontal', failLabel: 'fail_distant' },
      eyes_open: { passProb: 0.2, topLabel: 'fail_closed', failLabel: 'fail_closed' },
      neutral_expression: { passProb: 0.83, topLabel: 'pass_neutral', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.99, topLabel: 'pass_white_bg', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.8, topLabel: 'pass_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.68, topLabel: 'pass_hair_clear', failLabel: 'fail_bangs_over_eyes' },
      no_glasses: { passProb: 0.69, topLabel: 'pass_bare_eyes', failLabel: 'fail_sunglasses' },
      framing_chest_up: { passProb: 0.41, topLabel: 'pass_chest_up', failLabel: 'fail_macro_face' },
      occupancy_70_percent: { passProb: 0.65, topLabel: 'pass_70_pct', failLabel: 'fail_too_small' },
    },
  },
  {
    id: 'passport_compliant_male.jpg',
    title: 'Compliant #2',
    sub: '9/9 Pass',
    isCompliant: true,
    url: photoAssetUrl('passport_compliant_male.jpg'),
    groundTruth: {
      full_face: true,
      eyes_open: true,
      neutral_expression: true,
      neutral_background: true,
      shoulders_visible: true,
      hair_clear: true,
      no_glasses: true,
      framing_chest_up: true,
      occupancy_70_percent: true,
    },
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
        blink: 0.136,
        smile: 0.02,
        jawOpen: 0.005,
        foreheadDarkDiff: 26,
        bridgeContrast: 88,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.66, topLabel: 'pass_frontal', failLabel: 'fail_distant' },
      eyes_open: { passProb: 0.19, topLabel: 'fail_closed', failLabel: 'fail_closed' },
      neutral_expression: { passProb: 0.77, topLabel: 'pass_neutral', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.96, topLabel: 'pass_white_bg', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.74, topLabel: 'pass_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.65, topLabel: 'pass_hair_clear', failLabel: 'fail_hat_cap' },
      no_glasses: { passProb: 0.63, topLabel: 'pass_bare_eyes', failLabel: 'fail_sunglasses' },
      framing_chest_up: { passProb: 0.49, topLabel: 'pass_chest_up', failLabel: 'fail_macro_face' },
      occupancy_70_percent: { passProb: 0.46, topLabel: 'pass_70_pct', failLabel: 'fail_too_small' },
    },
  },
  {
    id: 'id_fail_glasses_smile.jpg',
    title: 'Sunglasses',
    sub: '3 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_glasses_smile.jpg'),
    groundTruth: {
      full_face: true,
      eyes_open: false,
      neutral_expression: false,
      neutral_background: true,
      shoulders_visible: true,
      hair_clear: true,
      no_glasses: false,
      framing_chest_up: true,
      occupancy_70_percent: true,
    },
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
        blink: 0.056,
        smile: 0.931,
        jawOpen: 0.146,
        foreheadDarkDiff: 76,
        bridgeContrast: 124,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.31, topLabel: 'fail_distant', failLabel: 'fail_distant' },
      eyes_open: { passProb: 0.006, topLabel: 'fail_sunglasses', failLabel: 'fail_sunglasses' },
      neutral_expression: { passProb: 0.03, topLabel: 'fail_smiling', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.79, topLabel: 'pass_white_bg', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.1, topLabel: 'fail_no_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.63, topLabel: 'pass_hair_clear', failLabel: 'fail_hat_cap' },
      no_glasses: { passProb: 0.011, topLabel: 'fail_sunglasses', failLabel: 'fail_sunglasses' },
      framing_chest_up: { passProb: 0.45, topLabel: 'pass_chest_up', failLabel: 'fail_macro_face' },
      occupancy_70_percent: { passProb: 0.31, topLabel: 'fail_too_large', failLabel: 'fail_too_large' },
    },
  },
  {
    id: 'id_fail_eyes_closed_hair.jpg',
    title: 'Eyes Closed',
    sub: '2 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_eyes_closed_hair.jpg'),
    groundTruth: {
      full_face: true,
      eyes_open: false,
      neutral_expression: true,
      neutral_background: true,
      shoulders_visible: true,
      hair_clear: false,
      no_glasses: true,
      framing_chest_up: true,
      occupancy_70_percent: true,
    },
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
        blink: 0.524,
        smile: 0.001,
        jawOpen: 0.002,
        foreheadDarkDiff: 244,
        bridgeContrast: 116,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.32, topLabel: 'fail_distant', failLabel: 'fail_distant' },
      eyes_open: { passProb: 0.01, topLabel: 'fail_closed', failLabel: 'fail_closed' },
      neutral_expression: { passProb: 0.66, topLabel: 'pass_neutral', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.7, topLabel: 'pass_white_bg', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.2, topLabel: 'fail_no_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.014, topLabel: 'fail_bangs_over_eyes', failLabel: 'fail_bangs_over_eyes' },
      no_glasses: { passProb: 0.56, topLabel: 'pass_bare_eyes', failLabel: 'fail_sunglasses' },
      framing_chest_up: { passProb: 0.2, topLabel: 'fail_full_body', failLabel: 'fail_full_body' },
      occupancy_70_percent: { passProb: 0.25, topLabel: 'fail_too_small', failLabel: 'fail_too_small' },
    },
  },
  {
    id: 'id_fail_extreme_zoom_cropped.jpg',
    title: 'Too Close',
    sub: '5 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_extreme_zoom_cropped.jpg'),
    groundTruth: {
      full_face: false,
      eyes_open: true,
      neutral_expression: true,
      neutral_background: false,
      shoulders_visible: false,
      hair_clear: true,
      no_glasses: true,
      framing_chest_up: false,
      occupancy_70_percent: false,
    },
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
        blink: 0.036,
        smile: 0.001,
        jawOpen: 0.062,
        foreheadDarkDiff: 51,
        bridgeContrast: 116,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.35, topLabel: 'fail_cropped', failLabel: 'fail_cropped' },
      eyes_open: { passProb: 0.43, topLabel: 'pass_open', failLabel: 'fail_closed' },
      neutral_expression: { passProb: 0.53, topLabel: 'pass_neutral', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.09, topLabel: 'fail_indoor_clutter', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.06, topLabel: 'fail_no_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.16, topLabel: 'fail_bangs_over_eyes', failLabel: 'fail_bangs_over_eyes' },
      no_glasses: { passProb: 0.84, topLabel: 'pass_bare_eyes', failLabel: 'fail_eyeglasses' },
      framing_chest_up: { passProb: 0.05, topLabel: 'fail_macro_face', failLabel: 'fail_macro_face' },
      occupancy_70_percent: { passProb: 0.2, topLabel: 'fail_too_large', failLabel: 'fail_too_large' },
    },
  },
  {
    id: 'id_fail_distant_waist_up.jpg',
    title: 'Too Far',
    sub: '4 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_distant_waist_up.jpg'),
    groundTruth: {
      full_face: false,
      eyes_open: true,
      neutral_expression: true,
      neutral_background: false,
      shoulders_visible: true,
      hair_clear: true,
      no_glasses: true,
      framing_chest_up: false,
      occupancy_70_percent: false,
    },
    biometrics: {
      bgLum: 169,
      bgVar: 93,
      isPlainWhiteBg: false,
      fgRatio: 0.26,
      bottomShoulderWidthRatio: 0.8,
      faceInfo: { hasFace: false },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.27, topLabel: 'fail_distant', failLabel: 'fail_distant' },
      eyes_open: { passProb: 0.28, topLabel: 'fail_sunglasses', failLabel: 'fail_sunglasses' },
      neutral_expression: { passProb: 0.52, topLabel: 'pass_neutral', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.001, topLabel: 'fail_outdoor_street', failLabel: 'fail_outdoor_street' },
      shoulders_visible: { passProb: 0.29, topLabel: 'fail_no_shoulders', failLabel: 'fail_no_shoulders' },
      hair_clear: { passProb: 0.05, topLabel: 'fail_bangs_over_eyes', failLabel: 'fail_bangs_over_eyes' },
      no_glasses: { passProb: 0.08, topLabel: 'fail_eyeglasses', failLabel: 'fail_eyeglasses' },
      framing_chest_up: { passProb: 0.08, topLabel: 'fail_full_body', failLabel: 'fail_full_body' },
      occupancy_70_percent: { passProb: 0.39, topLabel: 'fail_too_small', failLabel: 'fail_too_small' },
    },
  },
  {
    id: 'id_fail_tilted_glasses_selfie.jpg',
    title: 'Tilted Selfie',
    sub: '8 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_tilted_glasses_selfie.jpg'),
    groundTruth: {
      full_face: false,
      eyes_open: false,
      neutral_expression: false,
      neutral_background: false,
      shoulders_visible: false,
      hair_clear: true,
      no_glasses: false,
      framing_chest_up: false,
      occupancy_70_percent: false,
    },
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
        gazeHoriz: 0.27,
        blink: 0.178,
        smile: 0.927,
        jawOpen: 0.049,
        foreheadDarkDiff: 24,
        bridgeContrast: 147,
      },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.22, topLabel: 'fail_turned_tilted', failLabel: 'fail_turned_tilted' },
      eyes_open: { passProb: 0.16, topLabel: 'fail_looking_away', failLabel: 'fail_looking_away' },
      neutral_expression: { passProb: 0.08, topLabel: 'fail_smiling', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.07, topLabel: 'fail_indoor_clutter', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.03, topLabel: 'fail_angled_selfie', failLabel: 'fail_angled_selfie' },
      hair_clear: { passProb: 0.72, topLabel: 'pass_hair_clear', failLabel: 'fail_bangs_over_eyes' },
      no_glasses: { passProb: 0.015, topLabel: 'fail_eyeglasses', failLabel: 'fail_eyeglasses' },
      framing_chest_up: { passProb: 0.18, topLabel: 'fail_off_center', failLabel: 'fail_off_center' },
      occupancy_70_percent: { passProb: 0.28, topLabel: 'fail_too_large', failLabel: 'fail_too_large' },
    },
  },
  {
    id: 'id_fail_mask_hat.jpg',
    title: 'Mask & Cap',
    sub: '6 Fails',
    isCompliant: false,
    url: photoAssetUrl('id_fail_mask_hat.jpg'),
    groundTruth: {
      full_face: false,
      eyes_open: true,
      neutral_expression: false,
      neutral_background: false,
      shoulders_visible: true,
      hair_clear: false,
      no_glasses: true,
      framing_chest_up: false,
      occupancy_70_percent: false,
    },
    biometrics: {
      bgLum: 180,
      bgVar: 10,
      isPlainWhiteBg: false,
      fgRatio: 0.31,
      bottomShoulderWidthRatio: 1.0,
      faceInfo: { hasFace: false },
    },
    rawEg2Vision: {
      full_face: { passProb: 0.09, topLabel: 'fail_masked', failLabel: 'fail_masked' },
      eyes_open: { passProb: 0.4, topLabel: 'pass_open', failLabel: 'fail_closed' },
      neutral_expression: { passProb: 0.43, topLabel: 'fail_smiling', failLabel: 'fail_smiling' },
      neutral_background: { passProb: 0.62, topLabel: 'fail_indoor_clutter', failLabel: 'fail_indoor_clutter' },
      shoulders_visible: { passProb: 0.16, topLabel: 'fail_angled_selfie', failLabel: 'fail_angled_selfie' },
      hair_clear: { passProb: 0.18, topLabel: 'fail_hat_cap', failLabel: 'fail_hat_cap' },
      no_glasses: { passProb: 0.22, topLabel: 'fail_sunglasses', failLabel: 'fail_sunglasses' },
      framing_chest_up: { passProb: 0.06, topLabel: 'fail_full_body', failLabel: 'fail_full_body' },
      occupancy_70_percent: { passProb: 0.32, topLabel: 'fail_too_small', failLabel: 'fail_too_small' },
    },
  },
];

const logit = (p: number): number => {
  const c = Math.max(1e-4, Math.min(1 - 1e-4, p));
  return Math.log(c / (1 - c));
};

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

export class DecisionVisionPlayground {
  private el: Record<string, HTMLElement> = {};
  private currentPhotoObj: PhotoPreset = PRESET_PHOTOS[0];
  private selectedRuleId = 'full_face';
  private showBiometricOverlay = true;
  private ruleExemplarBank: Record<string, { positives: ExemplarEntry[]; negatives: ExemplarEntry[] }> = {};

  private faceLandmarker: FaceLandmarker | null = null;
  private faceLandmarkerPromise: Promise<FaceLandmarker> | null = null;
  private webcamActive = false;
  private webcamStream: MediaStream | null = null;
  private liveLoopTimer: ReturnType<typeof setTimeout> | null = null;
  private liveFrameCanvas = document.createElement('canvas');

  constructor(
    private root: HTMLElement,
    private onStatus?: (text: string, inferenceTime?: number) => void
  ) {
    this.liveFrameCanvas.width = 360;
    this.liveFrameCanvas.height = 460;
    this.initExemplarBank();
  }

  private initExemplarBank() {
    for (const rule of PASSPORT_RULES) {
      this.ruleExemplarBank[rule.id] = { positives: [], negatives: [] };
      for (const p of PRESET_PHOTOS) {
        const isPos = Boolean(p.groundTruth?.[rule.id]);
        const rawProb = p.rawEg2Vision[rule.id].passProb;
        const entry: ExemplarEntry = {
          id: p.id,
          name: p.title,
          url: p.url,
          rawPassProb: rawProb,
          logitVal: logit(rawProb),
          enabled: isPos ? p.isCompliant : true,
        };
        if (isPos) this.ruleExemplarBank[rule.id].positives.push(entry);
        else this.ruleExemplarBank[rule.id].negatives.push(entry);
      }
    }
  }

  init() {
    this.root.innerHTML = template;
    this.root.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));

    this.el['dv-guide-btn']?.addEventListener('click', () => this.toggleBiometricOverlay());
    this.el['dv-webcam-btn']?.addEventListener('click', () => this.toggleWebcamBooth());
    this.el['dv-snap-btn']?.addEventListener('click', () => this.captureWebcamPhoto());
    this.el['dv-upload-input']?.addEventListener('change', (e) => this.handleCustomPhotoUpload(e));

    this.el['dv-add-pos-current']?.addEventListener('click', () => this.addCurrentPhotoAsExemplar(true));
    this.el['dv-add-neg-current']?.addEventListener('click', () => this.addCurrentPhotoAsExemplar(false));
    this.el['dv-upload-pos-input']?.addEventListener('change', (e) => this.handleExemplarFileUpload(e, true));
    this.el['dv-upload-neg-input']?.addEventListener('change', (e) => this.handleExemplarFileUpload(e, false));

    this.renderPresetStrip();
    this.renderAllPanels();
  }

  cleanup() {
    this.stopWebcamBooth();
    if (this.faceLandmarker) {
      try {
        this.faceLandmarker.close();
      } catch (_) {}
      this.faceLandmarker = null;
      this.faceLandmarkerPromise = null;
    }
  }

  public selectPresetPhoto(photoId: string) {
    if (this.webcamActive) this.stopWebcamBooth();
    const found = PRESET_PHOTOS.find((x) => x.id === photoId);
    if (!found) return;
    this.currentPhotoObj = found;
    this.renderPresetStrip();
    this.renderAllPanels();
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

  private getCalibrationParamsForRule(ruleId: string) {
    const bank = this.ruleExemplarBank[ruleId];
    const activePos = bank.positives.filter((x) => x.enabled);
    const activeNeg = bank.negatives.filter((x) => x.enabled);
    const muPos = activePos.length > 0 ? activePos.reduce((acc, x) => acc + x.logitVal, 0) / activePos.length : 0.0;
    const muNeg = activeNeg.length > 0 ? activeNeg.reduce((acc, x) => acc + x.logitVal, 0) / activeNeg.length : -2.0;
    const mid = (muPos + muNeg) * 0.5;
    const gap = Math.max(0.45, muPos - muNeg);
    const scale = 3.0 / gap;
    return { muPos, muNeg, mid, gap, scale };
  }

  public evaluatePhotoRules(photoObj: PhotoPreset): Record<string, RuleEvalResult> {
    const results: Record<string, RuleEvalResult> = {};
    const fi: FaceBiometricInfo = photoObj.biometrics.faceInfo || { hasFace: false };
    const bioData = photoObj.biometrics;
    const isAngledSelfie =
      (photoObj.rawEg2Vision?.shoulders_visible?.passProb ?? 0.5) < 0.06 &&
      photoObj.rawEg2Vision?.shoulders_visible?.topLabel === 'fail_angled_selfie' &&
      !bioData.isPlainWhiteBg;

    const minX = fi.minX ?? 0;
    const maxX = fi.maxX ?? 1;
    const minY = fi.minY ?? 0;
    const maxY = fi.maxY ?? 1;
    const faceH = fi.faceH ?? 0;

    const centerX = fi.hasFace ? (fi.centerX ?? (minX + maxX) * 0.5) : 0.5;
    const centerY = fi.hasFace ? (fi.centerY ?? (minY + maxY) * 0.5) : 0.45;
    const yawAsymmetry = fi.yawAsymmetry ?? 0;
    const eyeRoll = fi.eyeRoll ?? 0;
    const gazeHoriz = fi.gazeHoriz ?? 0;

    const isCentered = Math.abs(centerX - 0.5) <= 0.11 && centerY >= 0.32 && centerY <= 0.56;
    const isHeadStraight = yawAsymmetry <= 0.22 && eyeRoll <= 0.13;
    const isLookingDirect = gazeHoriz <= 0.42 && yawAsymmetry <= 0.24;

    const eyesCalParams = this.getCalibrationParamsForRule('eyes_open');
    const glassesCalParams = this.getCalibrationParamsForRule('no_glasses');
    const vEyesCal = sigmoid(
      (logit(photoObj.rawEg2Vision.eyes_open.passProb) - eyesCalParams.mid) * eyesCalParams.scale
    );
    const vGlassesCal = sigmoid(
      (logit(photoObj.rawEg2Vision.no_glasses.passProb) - glassesCalParams.mid) * glassesCalParams.scale
    );

    for (const rule of PASSPORT_RULES) {
      const rawEntry = photoObj.rawEg2Vision[rule.id] || {
        passProb: 0.5,
        topLabel: rule.options[0].label,
        failLabel: rule.options[1].label,
      };
      const rawPassProb = rawEntry.passProb;
      const cal = this.getCalibrationParamsForRule(rule.id);
      const calPassProb = sigmoid((logit(rawPassProb) - cal.mid) * cal.scale);

      let bioProb = calPassProb;
      let specificFailLabel = rawEntry.failLabel || rule.options[1].label;

      if (rule.id === 'full_face') {
        const faceUnclipped = fi.hasFace && minY >= 0.08 && maxY <= 0.91 && faceH >= 0.24 && faceH <= 0.64;
        const fullFacePass = faceUnclipped && isCentered && isHeadStraight;
        bioProb = fullFacePass ? 0.95 : 0.04;
        if (!fullFacePass && fi.hasFace) {
          if (minY < 0.08 || maxY > 0.91 || faceH > 0.64) specificFailLabel = 'fail_cropped';
          else if (faceH < 0.24) specificFailLabel = 'fail_distant';
          else if (!isCentered && !isHeadStraight) specificFailLabel = 'fail_off_center_and_tilted';
          else if (!isCentered) specificFailLabel = 'fail_off_center';
          else if (!isHeadStraight) specificFailLabel = 'fail_turned_tilted';
        }
      } else if (rule.id === 'eyes_open') {
        const hasDarkSunglasses =
          vEyesCal < 0.22 && vGlassesCal < 0.22 && fi.hasFace && (fi.bridgeContrast ?? 0) >= 120;
        if (!fi.hasFace) {
          bioProb = rawPassProb >= 0.15 ? 0.88 : calPassProb;
        } else if ((fi.blink ?? 0) >= 0.38) {
          bioProb = 0.04;
          specificFailLabel = 'fail_closed';
        } else if (hasDarkSunglasses) {
          bioProb = 0.05;
          specificFailLabel = 'fail_sunglasses';
        } else if (!isLookingDirect) {
          bioProb = 0.04;
          specificFailLabel = 'fail_looking_away';
        } else {
          bioProb = 0.94;
        }
      } else if (rule.id === 'neutral_expression') {
        if (!fi.hasFace) {
          bioProb = calPassProb < 0.5 || rawPassProb < 0.48 ? 0.16 : 0.88;
        } else if ((fi.smile ?? 0) >= 0.24 || (fi.jawOpen ?? 0) >= 0.11 || calPassProb < 0.25) {
          bioProb = 0.04;
          specificFailLabel = 'fail_smiling';
        } else {
          bioProb = 0.94;
        }
      } else if (rule.id === 'neutral_background') {
        bioProb = bioData.isPlainWhiteBg ? 0.96 : 0.04;
      } else if (rule.id === 'shoulders_visible') {
        if (fi.hasFace && (faceH > 0.65 || maxY > 0.87)) {
          bioProb = 0.04;
          specificFailLabel = 'fail_no_shoulders';
        } else if (isAngledSelfie || (fi.hasFace && (!isCentered || !isHeadStraight))) {
          bioProb = 0.05;
          specificFailLabel = 'fail_angled_selfie';
        } else {
          bioProb = 0.93;
        }
      } else if (rule.id === 'hair_clear') {
        if (fi.hasFace && (fi.foreheadDarkDiff ?? 0) >= 140) {
          bioProb = 0.04;
          specificFailLabel = 'fail_bangs_over_eyes';
        } else if (!fi.hasFace && photoObj.rawEg2Vision.hair_clear.topLabel === 'fail_hat_cap') {
          bioProb = 0.06;
          specificFailLabel = 'fail_hat_cap';
        } else {
          bioProb = 0.94;
        }
      } else if (rule.id === 'no_glasses') {
        if (fi.hasFace && (fi.bridgeContrast ?? 0) >= 120 && vGlassesCal < 0.35) bioProb = 0.04;
        else bioProb = 0.94;
      } else if (rule.id === 'framing_chest_up') {
        if (
          fi.hasFace &&
          faceH >= 0.24 &&
          faceH <= 0.6 &&
          maxY >= 0.48 &&
          maxY <= 0.85 &&
          isCentered &&
          !isAngledSelfie
        ) {
          bioProb = 0.95;
        } else {
          bioProb = 0.05;
          if (fi.hasFace && (faceH > 0.6 || maxY > 0.85 || isAngledSelfie)) {
            specificFailLabel = 'fail_macro_face';
          } else if (fi.hasFace && !isCentered) {
            specificFailLabel = 'fail_off_center';
          } else {
            specificFailLabel = 'fail_full_body';
          }
        }
      } else if (rule.id === 'occupancy_70_percent') {
        if (fi.hasFace && faceH >= 0.25 && faceH <= 0.58 && minY >= 0.04 && maxY <= 0.86 && !isAngledSelfie) {
          bioProb = 0.95;
        } else {
          bioProb = 0.05;
          if (fi.hasFace && (faceH > 0.58 || maxY > 0.86 || minY < 0.04 || isAngledSelfie)) {
            specificFailLabel = 'fail_too_large';
          } else {
            specificFailLabel = 'fail_too_small';
          }
        }
      }

      const effectiveCalProb =
        rule.id === 'neutral_background' && bioData.isPlainWhiteBg ? Math.max(calPassProb, 0.9) : calPassProb;
      const finalPassProb = 0.35 * effectiveCalProb + 0.65 * bioProb;
      const passed = finalPassProb >= 0.5;
      const chosenOptionLabel = passed ? rule.options[0].label : specificFailLabel;
      const reasonText = rule.optionDisplay[chosenOptionLabel] || chosenOptionLabel;

      results[rule.id] = {
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
      btn.innerHTML = `
        <img class="dv-preset-thumb" src="${p.url}" alt="${p.title}" />
        <div class="dv-preset-title">${p.title}</div>
        <span class="dv-preset-tag ${p.isCompliant ? 'pass' : 'fail'}">${p.sub}</span>
      `;
      strip.appendChild(btn);
    }
  }

  private renderActivePhotoView() {
    const imgEl = this.el['dv-main-img'] as HTMLImageElement | undefined;
    const svgEl = this.el['dv-overlay-svg'];
    if (!this.webcamActive && imgEl) {
      imgEl.style.display = 'block';
      if (imgEl.getAttribute('src') !== this.currentPhotoObj.url) {
        imgEl.src = this.currentPhotoObj.url;
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
      const centerX = fi.centerX ?? (fi.minX + fi.maxX) * 0.5;
      const centerY = fi.centerY ?? (fi.minY + fi.maxY) * 0.5;
      const faceH = fi.faceH ?? fi.maxY - fi.minY;
      const isCentered = Math.abs(centerX - 0.5) <= 0.11 && centerY >= 0.32 && centerY <= 0.56;
      const isHeadStraight = (fi.yawAsymmetry ?? 0) <= 0.22 && (fi.eyeRoll ?? 0) <= 0.13;
      const isGoodBox =
        fi.minY >= 0.08 && fi.maxY <= 0.86 && faceH >= 0.24 && faceH <= 0.6 && isCentered && isHeadStraight;
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
    let passCount = 0;
    for (const r of PASSPORT_RULES) {
      if (evalRes[r.id].passed) passCount++;
    }
    const allPassed = passCount === PASSPORT_RULES.length;

    const banner = this.el['dv-verdict-banner'];
    const iconEl = this.el['dv-verdict-icon'];
    const titleEl = this.el['dv-verdict-title'];
    const subEl = this.el['dv-verdict-sub'];
    if (banner && iconEl && titleEl && subEl) {
      banner.className = 'dv-verdict-banner ' + (allPassed ? 'pass' : 'fail');
      iconEl.textContent = allPassed ? 'check_circle' : 'error';
      const failCount = PASSPORT_RULES.length - passCount;
      titleEl.textContent = allPassed ? 'COMPLIANT' : `${failCount} ISSUE${failCount > 1 ? 'S' : ''} DETECTED`;
      subEl.textContent = `${passCount} / ${PASSPORT_RULES.length} Passed`;
    }

    const listEl = this.el['dv-rules-list'];
    if (listEl) {
      if (listEl.children.length !== PASSPORT_RULES.length) {
        listEl.innerHTML = '';
        for (const rule of PASSPORT_RULES) {
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
                <span>${rule.shortTitle}</span>
              </div>
              <span class="dv-rule-badge"></span>
            </div>
            <div class="dv-rule-reason"></div>
            <div class="dv-rule-bar-track">
              <div class="dv-rule-bar-fill"></div>
            </div>
          `;
          listEl.appendChild(card);
        }
      }
      for (let i = 0; i < PASSPORT_RULES.length; i++) {
        const rule = PASSPORT_RULES[i];
        const r = evalRes[rule.id];
        const pct = Math.round(r.prob * 100);
        const card = listEl.children[i] as HTMLElement;
        card.className = `dv-rule-card ${r.passed ? 'pass' : 'fail'}${this.selectedRuleId === rule.id ? ' selected' : ''}`;
        const badgeEl = card.querySelector('.dv-rule-badge') as HTMLElement;
        badgeEl.className = `dv-rule-badge ${r.passed ? 'pass' : 'fail'}`;
        badgeEl.textContent = r.passed ? 'PASS' : 'FAIL';
        const reasonEl = card.querySelector('.dv-rule-reason') as HTMLElement;
        reasonEl.textContent = r.reasonText;
        const fillEl = card.querySelector('.dv-rule-bar-fill') as HTMLElement;
        fillEl.className = `dv-rule-bar-fill ${r.passed ? 'pass' : 'fail'}`;
        fillEl.style.width = `${pct}%`;
      }
    }

    this.renderExemplarStudioPanel(evalRes);
  }

  private renderExemplarStudioPanel(evalRes: Record<string, RuleEvalResult>) {
    const rule = PASSPORT_RULES.find((r) => r.id === this.selectedRuleId) || PASSPORT_RULES[0];
    const rRes = evalRes[rule.id];
    const bank = this.ruleExemplarBank[rule.id];

    if (this.el['dv-active-rule-title']) this.el['dv-active-rule-title'].textContent = rule.title;
    if (this.el['dv-active-rule-prompt']) this.el['dv-active-rule-prompt'].textContent = rule.prompt;

    const statusBadge = this.el['dv-active-rule-badge'];
    if (statusBadge) {
      statusBadge.className = 'dv-rule-badge ' + (rRes.passed ? 'pass' : 'fail');
      statusBadge.textContent = `${rRes.passed ? 'PASS' : 'FAIL'} (${Math.round(rRes.prob * 100)}%)`;
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
        img.src = item.url;
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
    const ruleId = this.selectedRuleId;
    const rawProb = this.currentPhotoObj.rawEg2Vision[ruleId]?.passProb ?? 0.5;
    const entry: ExemplarEntry = {
      id: `${this.currentPhotoObj.id}_${Date.now()}`,
      name: this.currentPhotoObj.title,
      url: this.currentPhotoObj.url,
      rawPassProb: rawProb,
      logitVal: logit(rawProb),
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
    const provisionalProb = isPositive ? 0.68 : 0.04;
    const entry: ExemplarEntry = {
      id: `upload_${Date.now()}`,
      name: file.name.slice(0, 14),
      url: dataUrl,
      rawPassProb: provisionalProb,
      logitVal: logit(provisionalProb),
      enabled: true,
    };
    if (isPositive) this.ruleExemplarBank[ruleId].positives.unshift(entry);
    else this.ruleExemplarBank[ruleId].negatives.unshift(entry);
    input.value = '';
    this.renderAllPanels();
  }

  private async ensureFaceLandmarker(): Promise<FaceLandmarker> {
    if (this.faceLandmarker) return this.faceLandmarker;
    if (this.faceLandmarkerPromise) return this.faceLandmarkerPromise;

    this.faceLandmarkerPromise = (async () => {
      const baseUrl = import.meta.env.BASE_URL || '/';
      const formattedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
      const wasmPath = new URL(`${formattedBase}wasm`, window.location.origin).href.replace(/\/$/, '');
      const resolver = await FilesetResolver.forVisionTasks(wasmPath, true);
      try {
        this.faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath: FACE_LANDMARKER_MODEL_URL,
            delegate: 'GPU',
          },
          runningMode: 'IMAGE',
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
      } catch (_) {
        this.faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
          baseOptions: {
            modelAssetPath: FACE_LANDMARKER_MODEL_URL,
            delegate: 'CPU',
          },
          runningMode: 'IMAGE',
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
      }
      return this.faceLandmarker;
    })();

    return this.faceLandmarkerPromise;
  }

  private async extractBiometricsFromSource(source: HTMLImageElement | HTMLCanvasElement): Promise<PhotoBiometrics> {
    const cw = 160;
    const ch = 200;
    const ca = document.createElement('canvas');
    ca.width = cw;
    ca.height = ch;
    const ctxA = ca.getContext('2d')!;
    ctxA.drawImage(source, 0, 0, cw, ch);
    const imgData = ctxA.getImageData(0, 0, cw, ch).data;

    const rawBgSamples: { r: number; g: number; b: number; lum: number }[] = [];
    for (let y = 3; y < Math.floor(ch * 0.34); y += 3) {
      for (const x of [4, 9, cw - 10, cw - 5]) {
        const idx = (y * cw + x) * 4;
        const r = imgData[idx];
        const g = imgData[idx + 1];
        const b = imgData[idx + 2];
        rawBgSamples.push({ r, g, b, lum: 0.299 * r + 0.587 * g + 0.114 * b });
      }
    }
    rawBgSamples.sort((a, b) => b.lum - a.lum);
    const bgSamples = rawBgSamples.slice(0, Math.max(8, Math.floor(rawBgSamples.length * 0.85)));

    let meanR = 0;
    let meanG = 0;
    let meanB = 0;
    for (const s of bgSamples) {
      meanR += s.r;
      meanG += s.g;
      meanB += s.b;
    }
    meanR /= bgSamples.length;
    meanG /= bgSamples.length;
    meanB /= bgSamples.length;

    let bgVar = 0;
    let bgSat = 0;
    for (const s of bgSamples) {
      bgVar += Math.hypot(s.r - meanR, s.g - meanG, s.b - meanB);
      bgSat += Math.max(s.r, s.g, s.b) - Math.min(s.r, s.g, s.b);
    }
    bgVar /= bgSamples.length;
    bgSat /= bgSamples.length;
    const bgLum = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;
    const isPlainWhiteBg = bgLum >= 165 && bgVar <= 30 && bgSat <= 70;

    let fgPixels = 0;
    let bottomRowFg = 0;
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const idx = (y * cw + x) * 4;
        const d = Math.hypot(imgData[idx] - meanR, imgData[idx + 1] - meanG, imgData[idx + 2] - meanB);
        if (d > 28) {
          fgPixels++;
          if (y >= ch - 12) bottomRowFg++;
        }
      }
    }
    const fgRatio = fgPixels / (cw * ch);
    const bottomShoulderWidthRatio = bottomRowFg / (cw * 12);

    let faceInfo: FaceBiometricInfo = { hasFace: false };
    try {
      const fl = await this.ensureFaceLandmarker();
      const flRes = fl.detect(source);
      if (flRes.faceLandmarks && flRes.faceLandmarks.length > 0) {
        const lm = flRes.faceLandmarks[0];
        let minX = 1;
        let minY = 1;
        let maxX = 0;
        let maxY = 0;
        for (const p of lm) {
          if (p.x < minX) minX = p.x;
          if (p.y < minY) minY = p.y;
          if (p.x > maxX) maxX = p.x;
          if (p.y > maxY) maxY = p.y;
        }
        const noseTip = lm[4];
        const noseIdx =
          (Math.min(ch - 1, Math.max(0, Math.floor(noseTip.y * ch))) * cw +
            Math.min(cw - 1, Math.max(0, Math.floor(noseTip.x * cw)))) *
          4;
        const skinR = imgData[noseIdx];
        const skinG = imgData[noseIdx + 1];
        const skinB = imgData[noseIdx + 2];

        let foreheadDarkDiff = 0;
        let foreheadSamples = 0;
        for (const ptIdx of [10, 151, 9, 107, 336, 67, 297]) {
          const pt = lm[ptIdx];
          if (pt) {
            const px = Math.min(cw - 1, Math.max(0, Math.floor(pt.x * cw)));
            const py = Math.min(ch - 1, Math.max(0, Math.floor(pt.y * ch)));
            const pIdx = (py * cw + px) * 4;
            foreheadDarkDiff += Math.hypot(imgData[pIdx] - skinR, imgData[pIdx + 1] - skinG, imgData[pIdx + 2] - skinB);
            foreheadSamples++;
          }
        }
        foreheadDarkDiff = foreheadSamples > 0 ? foreheadDarkDiff / foreheadSamples : 0;

        let bridgeContrast = 0;
        for (const ptIdx of [168, 6, 122, 351, 33, 263]) {
          const pt = lm[ptIdx];
          if (pt) {
            const px = Math.min(cw - 1, Math.max(0, Math.floor(pt.x * cw)));
            const py = Math.min(ch - 1, Math.max(0, Math.floor(pt.y * ch)));
            const pIdx = (py * cw + px) * 4;
            bridgeContrast += Math.hypot(imgData[pIdx] - skinR, imgData[pIdx + 1] - skinG, imgData[pIdx + 2] - skinB);
          }
        }
        bridgeContrast /= 6;

        const bs: Record<string, number> = {};
        if (flRes.faceBlendshapes && flRes.faceBlendshapes[0]) {
          for (const cat of flRes.faceBlendshapes[0].categories) {
            bs[cat.categoryName] = cat.score;
          }
        }
        const leftCheekDist = Math.abs(lm[4].x - lm[234].x);
        const rightCheekDist = Math.abs(lm[454].x - lm[4].x);
        const yawAsymmetry = Math.abs(leftCheekDist - rightCheekDist) / Math.max(1e-4, leftCheekDist + rightCheekDist);
        const eyeRoll = Math.abs(lm[263].y - lm[33].y) / Math.max(1e-4, Math.abs(lm[263].x - lm[33].x));
        const gazeHoriz = Math.max(
          ((bs.eyeLookInLeft || 0) + (bs.eyeLookOutRight || 0)) / 2,
          ((bs.eyeLookOutLeft || 0) + (bs.eyeLookInRight || 0)) / 2
        );
        faceInfo = {
          hasFace: true,
          minX,
          minY,
          maxX,
          maxY,
          centerX: Number(((minX + maxX) * 0.5).toFixed(3)),
          centerY: Number(((minY + maxY) * 0.5).toFixed(3)),
          faceW: maxX - minX,
          faceH: maxY - minY,
          yawAsymmetry: Number(yawAsymmetry.toFixed(3)),
          eyeRoll: Number(eyeRoll.toFixed(3)),
          gazeHoriz: Number(gazeHoriz.toFixed(3)),
          blink: ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2,
          smile: ((bs.mouthSmileLeft || 0) + (bs.mouthSmileRight || 0)) / 2,
          jawOpen: bs.jawOpen || 0,
          foreheadDarkDiff: Math.round(foreheadDarkDiff),
          bridgeContrast: Math.round(bridgeContrast),
        };
      }
    } catch (e) {
      console.warn('FaceLandmarker error:', e);
    }

    return {
      bgLum: Math.round(bgLum),
      bgVar: Math.round(bgVar),
      bgSat: Math.round(bgSat),
      isPlainWhiteBg,
      fgRatio: Number(fgRatio.toFixed(2)),
      bottomShoulderWidthRatio: Number(bottomShoulderWidthRatio.toFixed(2)),
      faceInfo,
    };
  }

  private async evaluateCustomImageElement(img: HTMLImageElement, titleLabel: string) {
    const t0 = performance.now();
    const biometrics = await this.extractBiometricsFromSource(img);
    const rawEg2Vision: Record<string, RawRuleVisionScore> = {};
    for (const r of PASSPORT_RULES) {
      rawEg2Vision[r.id] = { passProb: 0.45, topLabel: r.options[0].label, failLabel: r.options[1].label };
    }

    this.currentPhotoObj = {
      id: `custom_${Date.now()}`,
      title: titleLabel,
      sub: 'Custom',
      isCompliant: false,
      url: img.src,
      biometrics,
      rawEg2Vision,
    };
    this.renderPresetStrip();
    this.renderAllPanels();

    const elapsed = performance.now() - t0;
    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = `9/9 rules · ${Math.round(elapsed)} ms`;
    }
    this.onStatus?.('Evaluated photo compliance', elapsed);
  }

  private async handleCustomPhotoUpload(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    if (this.webcamActive) this.stopWebcamBooth();
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
    const sw = vh * (360 / 460);
    const sx = (vw - sw) * 0.5;
    sctx.save();
    sctx.translate(targetCanvas.width, 0);
    sctx.scale(-1, 1);
    sctx.drawImage(video, sx, 0, sw, vh, 0, 0, targetCanvas.width, targetCanvas.height);
    sctx.restore();
  }

  private async runLiveWebcamLoop() {
    if (!this.webcamActive) return;
    const video = this.el['dv-webcam-video'] as HTMLVideoElement | undefined;
    if (video && video.readyState >= 2) {
      const t0 = performance.now();
      this.drawMirroredVideoToCanvas(video, this.liveFrameCanvas);
      const biometrics = await this.extractBiometricsFromSource(this.liveFrameCanvas);
      if (!this.webcamActive) return;

      this.currentPhotoObj.biometrics = biometrics;
      this.renderAllPanels();

      const elapsed = Math.max(8, Math.round(performance.now() - t0));
      if (this.el['dv-result-meta']) {
        this.el['dv-result-meta'].textContent = `Live · ${elapsed} ms`;
      }
    }
    if (this.webcamActive) {
      this.liveLoopTimer = setTimeout(() => this.runLiveWebcamLoop(), 95);
    }
  }

  public async toggleWebcamBooth() {
    if (this.webcamActive) {
      this.stopWebcamBooth();
      this.renderPresetStrip();
      this.renderAllPanels();
      return;
    }
    try {
      const video = this.el['dv-webcam-video'] as HTMLVideoElement;
      const imgEl = this.el['dv-main-img'] as HTMLImageElement;
      this.webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      video.srcObject = this.webcamStream;
      await video.play();
      this.webcamActive = true;
      imgEl.style.display = 'none';
      video.style.display = 'block';
      if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Stop Live';
      this.el['dv-webcam-btn']?.classList.add('active');
      if (this.el['dv-snap-btn']) this.el['dv-snap-btn'].style.display = 'inline-flex';

      const rawEg2Vision: Record<string, RawRuleVisionScore> = {};
      for (const r of PASSPORT_RULES) {
        rawEg2Vision[r.id] = { passProb: 0.45, topLabel: r.options[0].label, failLabel: r.options[1].label };
      }
      this.currentPhotoObj = {
        id: 'live_webcam',
        title: 'Live Camera',
        sub: 'Live',
        isCompliant: false,
        url: this.currentPhotoObj.url,
        biometrics: {
          bgLum: 220,
          bgVar: 10,
          isPlainWhiteBg: false,
          fgRatio: 0.65,
          bottomShoulderWidthRatio: 0.85,
          faceInfo: { hasFace: false },
        },
        rawEg2Vision,
      };
      this.renderPresetStrip();
      this.runLiveWebcamLoop();
    } catch (err: any) {
      this.onStatus?.(`Webcam error: ${err?.message || err}`);
    }
  }

  public stopWebcamBooth() {
    this.webcamActive = false;
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
    if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Live Check';
    this.el['dv-webcam-btn']?.classList.remove('active');
    if (this.el['dv-snap-btn']) this.el['dv-snap-btn'].style.display = 'none';
    if (this.el['dv-result-meta']) this.el['dv-result-meta'].textContent = '9/9 rules';
  }

  private async captureWebcamPhoto() {
    if (!this.webcamActive) return;
    const video = this.el['dv-webcam-video'] as HTMLVideoElement;
    this.drawMirroredVideoToCanvas(video, this.liveFrameCanvas);
    this.stopWebcamBooth();

    const dataUrl = this.liveFrameCanvas.toDataURL('image/jpeg', 0.92);
    const img = new Image();
    img.onload = () => this.evaluateCustomImageElement(img, 'Camera Photo');
    img.src = dataUrl;
  }
}
