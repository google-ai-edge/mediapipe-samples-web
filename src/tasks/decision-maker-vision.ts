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

import template from '../templates/decision-maker-vision.html?raw';
import { DECISION_MODELS } from '../components/decision-runtime';

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
  normalizePrior?: boolean;
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

export const PASSPORT_RULES: PassportRule[] = [
  {
    id: 'full_face',
    num: 1,
    icon: 'face',
    shortTitle: 'Head in Frame',
    title: '1. Head Inside Frame',
    normalizePrior: false,
    prompt: "Can you see the top of the person's hair and their shirt below the chin?",
    options: [
      {
        label: 'pass_frontal',
        description: 'Portrait showing the top of the head with hair, the chin, and the shirt or shoulders.',
      },
      {
        label: 'fail_cropped',
        description: 'Tightly cropped macro close-up of only skin, eyes, nose, and lips filling the entire image.',
      },
    ],
    optionDisplay: {
      pass_frontal: 'Full head & chin in frame',
      fail_cropped: 'Head/chin clipped at edge',
    },
  },
  {
    id: 'eyes_open',
    num: 2,
    icon: 'visibility',
    shortTitle: 'Eyes Open',
    title: '2. Eyes Open & Direct',
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
    },
  },
  {
    id: 'neutral_expression',
    num: 3,
    icon: 'sentiment_neutral',
    shortTitle: 'Neutral Face',
    title: '3. Neutral Expression',
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
    },
  },
  {
    id: 'neutral_background',
    num: 4,
    icon: 'crop_portrait',
    shortTitle: 'Plain BG',
    title: '4. Plain Neutral Background',
    normalizePrior: true,
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
  },
  {
    id: 'shoulders_visible',
    num: 5,
    icon: 'center_focus_strong',
    shortTitle: 'Straight Pose',
    title: '5. Straight Frontal Pose',
    normalizePrior: false,
    prompt: 'Is the head held straight and upright facing the camera?',
    options: [
      {
        label: 'pass_shoulders',
        description: 'Upright frontal portrait of a person with head straight and level.',
      },
      {
        label: 'fail_angled_selfie',
        description: 'Angled selfie with head tilted sideways and leaning to one side.',
      },
    ],
    optionDisplay: {
      pass_shoulders: 'Upright & level head pose',
      fail_angled_selfie: 'Head tilted sideways',
    },
  },
  {
    id: 'hair_clear',
    num: 6,
    icon: 'content_cut',
    shortTitle: 'Hair Clear',
    title: '6. Hair Clear of Eyes',
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
  },
  {
    id: 'no_glasses',
    num: 7,
    icon: 'visibility_off',
    shortTitle: 'No Glasses',
    title: '7. No Eyeglasses or Sunglasses',
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
  },
  {
    id: 'framing_chest_up',
    num: 8,
    icon: 'masks',
    shortTitle: 'No Mask/Cap',
    title: '8. Uncovered Face & Head',
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
    },
  },
  {
    id: 'occupancy_70_percent',
    num: 9,
    icon: 'photo_size_select_large',
    shortTitle: 'Good Distance',
    title: '9. Proper Portrait Distance',
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
    },
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

export class DecisionVisionPlayground {
  private el: Record<string, HTMLElement> = {};
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
    this.initExemplarBank();
  }

  private computeLogitVector(rawVision: Record<string, RawRuleVisionScore>): number[] {
    return PASSPORT_RULES.map((r) => logit(rawVision?.[r.id]?.passProb ?? 0.5));
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

  private initExemplarBank() {
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
    for (const rule of PASSPORT_RULES) {
      this.ruleExemplarBank[rule.id] = { positives: [], negatives: [] };
      for (const p of PRESET_PHOTOS) {
        const failedRules = defaultFails[p.id] ?? [];
        const isPos = !failedRules.includes(rule.id);
        const defaultProb = isPos ? 0.82 : 0.18;
        const entry: ExemplarEntry = {
          id: p.id,
          name: p.title,
          url: p.url,
          rawPassProb: defaultProb,
          logitVal: logit(defaultProb),
          logitVec: PASSPORT_RULES.map((r) => logit(!(defaultFails[p.id] ?? []).includes(r.id) ? 0.82 : 0.18)),
          enabled: true,
          isDefault: true,
        };
        if (isPos) this.ruleExemplarBank[rule.id].positives.push(entry);
        else this.ruleExemplarBank[rule.id].negatives.push(entry);
      }
    }
  }

  private syncExemplarBankWithLiveScore(photoObj: PhotoPreset) {
    if (!photoObj?.rawEg2Vision) return;
    const vec = this.computeLogitVector(photoObj.rawEg2Vision);
    for (const rule of PASSPORT_RULES) {
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

  private buildPassportSchema() {
    return {
      context: '',
      questions: PASSPORT_RULES.map((rule) => ({
        id: rule.id,
        type: 'categorical',
        prompt: rule.prompt,
        normalizePrior: Boolean(rule.normalizePrior),
        options: rule.options.map((opt) => ({
          label: opt.label,
          description: opt.description,
        })),
      })),
    };
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
    for (const rule of PASSPORT_RULES) {
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
    try {
      const imageBytes = await this.sourceToImageBytes(source);
      if (imageBytes.byteLength === 0) return null;
      const msg = await this.ask({ imageBytes }, 'schema', this.buildPassportSchema());
      if (msg?.type === 'DECIDE_RESULT' && msg.result) {
        return {
          rawEg2Vision: this.parseSchemaResultToRawVision(msg.result, fallback),
          inferenceTime: msg.inferenceTime,
        };
      }
    } catch (e) {
      console.warn('DecisionMaker multimodal evaluation fallback:', e);
    }
    return null;
  }

  private async evaluateCurrentPhotoWithDecisionMaker() {
    if (!this.isVisionModelReady() || this.webcamActive) return;
    const targetPhoto = this.currentPhotoObj;
    const imgEl = this.el['dv-main-img'] as HTMLImageElement | undefined;
    let sourceImg: HTMLImageElement | undefined = imgEl;
    if (!sourceImg || !sourceImg.complete || sourceImg.naturalWidth === 0) {
      sourceImg = await new Promise<HTMLImageElement | undefined>((resolve) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () => resolve(undefined);
        el.src = targetPhoto.url;
      });
    }
    if (!sourceImg) return;
    const dmEval = await this.evaluateSourceWithDecisionMaker(sourceImg);
    if (!dmEval) return;
    targetPhoto.rawEg2Vision = dmEval.rawEg2Vision;
    targetPhoto.evaluated = true;
    this.syncExemplarBankWithLiveScore(targetPhoto);
    const evalRes = this.evaluatePhotoRules(targetPhoto);
    let passCount = 0;
    for (const r of PASSPORT_RULES) {
      if (evalRes[r.id].passed) passCount++;
    }
    const failCount = PASSPORT_RULES.length - passCount;
    targetPhoto.isCompliant = failCount === 0;
    targetPhoto.sub = failCount === 0 ? '9/9 Pass' : `${failCount} Fail${failCount > 1 ? 's' : ''}`;
    this.renderPresetStrip();
    this.renderAllPanels();
    if (dmEval.inferenceTime !== undefined) {
      if (this.el['dv-result-meta']) {
        this.el['dv-result-meta'].textContent = `9/9 rules · ${Math.round(dmEval.inferenceTime)} ms`;
      }
      this.onStatus?.('Evaluated photo compliance', dmEval.inferenceTime);
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
  }

  public selectPresetPhoto(photoId: string) {
    if (this.webcamActive) this.stopWebcamBooth();
    const found = PRESET_PHOTOS.find((x) => x.id === photoId);
    if (!found) return;
    found.evaluated = false;
    found.rawEg2Vision = {};
    found.isCompliant = null;
    found.sub = this.isVisionModelReady() ? 'Evaluating…' : 'Pending…';
    this.currentPhotoObj = found;
    this.renderPresetStrip();
    this.renderAllPanels();
    if (this.isVisionModelReady()) {
      void this.evaluateCurrentPhotoWithDecisionMaker();
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
   * Evaluates all 9 rules purely from `DecisionMaker` schema probabilities
   * (`photoObj.rawEg2Vision`) calibrated against the enabled positive and
   * negative Few-Shot Reference Examples in `this.ruleExemplarBank` — with
   * zero hard-coded biometric or pixel heuristics.
   */
  public evaluatePhotoRules(photoObj: PhotoPreset): Record<string, RuleEvalResult> {
    const results: Record<string, RuleEvalResult> = {};
    const isEvaluated = Boolean(photoObj?.evaluated && Object.keys(photoObj.rawEg2Vision ?? {}).length > 0);
    if (!isEvaluated) {
      for (const rule of PASSPORT_RULES) {
        results[rule.id] = {
          pending: true,
          passed: false,
          prob: 0,
          rawPassProb: 0,
          calPassProb: 0,
          chosenOptionLabel: rule.options[0].label,
          reasonText: this.isVisionModelReady() ? 'Evaluating with 435M-VL…' : 'Waiting for 435M-VL model…',
        };
      }
      return results;
    }

    const queryLogitVec = this.computeLogitVector(photoObj.rawEg2Vision);

    for (let rIdx = 0; rIdx < PASSPORT_RULES.length; rIdx++) {
      const rule = PASSPORT_RULES[rIdx];
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

      const finalPassProb = calPassProb;
      const passed = finalPassProb >= 0.5;
      const specificFailLabel = rawEntry.failLabel || rule.options[1]?.label || rule.options[0].label;
      const chosenOptionLabel = passed ? rule.options[0].label : specificFailLabel;
      const reasonText = rule.optionDisplay[chosenOptionLabel] || chosenOptionLabel;

      results[rule.id] = {
        pending: false,
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
      this.currentPhotoObj.evaluated &&
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
    let passCount = 0;
    for (const r of PASSPORT_RULES) {
      if (evalRes[r.id].passed) passCount++;
    }
    const allPassed = !isPending && passCount === PASSPORT_RULES.length;

    const banner = this.el['dv-verdict-banner'];
    const iconEl = this.el['dv-verdict-icon'];
    const titleEl = this.el['dv-verdict-title'];
    const subEl = this.el['dv-verdict-sub'];
    if (banner && iconEl && titleEl && subEl) {
      if (isPending) {
        banner.className = 'dv-verdict-banner';
        iconEl.textContent = 'hourglass_top';
        titleEl.textContent = this.isVisionModelReady() ? 'EVALUATING…' : 'LOADING MODEL…';
        subEl.textContent = this.isVisionModelReady() ? '9 Rules' : 'Please wait';
      } else {
        banner.className = 'dv-verdict-banner ' + (allPassed ? 'pass' : 'fail');
        iconEl.textContent = allPassed ? 'check_circle' : 'error';
        const failCount = PASSPORT_RULES.length - passCount;
        titleEl.textContent = allPassed ? 'COMPLIANT' : `${failCount} ISSUE${failCount > 1 ? 'S' : ''}`;
        subEl.textContent = `${passCount}/${PASSPORT_RULES.length} Passed`;
      }
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
        const badgeEl = card.querySelector('.dv-rule-badge') as HTMLElement;
        const reasonEl = card.querySelector('.dv-rule-reason') as HTMLElement;
        const fillEl = card.querySelector('.dv-rule-bar-fill') as HTMLElement;
        if (r.pending) {
          card.className = `dv-rule-card${this.selectedRuleId === rule.id ? ' selected' : ''}`;
          badgeEl.className = 'dv-rule-badge';
          badgeEl.textContent = 'WAIT';
          reasonEl.textContent = r.reasonText;
          fillEl.className = 'dv-rule-bar-fill';
          fillEl.style.width = '0%';
        } else {
          card.className = `dv-rule-card ${r.passed ? 'pass' : 'fail'}${this.selectedRuleId === rule.id ? ' selected' : ''}`;
          badgeEl.className = `dv-rule-badge ${r.passed ? 'pass' : 'fail'}`;
          badgeEl.textContent = r.passed ? 'PASS' : 'FAIL';
          reasonEl.textContent = r.reasonText;
          fillEl.className = `dv-rule-bar-fill ${r.passed ? 'pass' : 'fail'}`;
          fillEl.style.width = `${pct}%`;
        }
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
      if (rRes.pending) {
        statusBadge.className = 'dv-rule-badge';
        statusBadge.textContent = 'WAIT';
      } else {
        statusBadge.className = 'dv-rule-badge ' + (rRes.passed ? 'pass' : 'fail');
        statusBadge.textContent = `${rRes.passed ? 'PASS' : 'FAIL'} (${Math.round(rRes.prob * 100)}%)`;
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

  private async evaluateCustomImageElement(img: HTMLImageElement, titleLabel: string) {
    const t0 = performance.now();
    if (this.el['dv-result-meta']) {
      this.el['dv-result-meta'].textContent = 'Evaluating photo…';
    }
    const dmEval = await this.evaluateSourceWithDecisionMaker(img);
    const rawEg2Vision: Record<string, RawRuleVisionScore> = dmEval ? dmEval.rawEg2Vision : {};

    this.currentPhotoObj = {
      id: `custom_${Date.now()}`,
      title: titleLabel,
      sub: 'Custom',
      isCompliant: null,
      evaluated: Boolean(dmEval),
      url: img.src,
      biometrics: {
        bgLum: 240,
        bgVar: 5,
        isPlainWhiteBg: true,
        fgRatio: 0.68,
        bottomShoulderWidthRatio: 0.85,
        faceInfo: { hasFace: false },
      },
      rawEg2Vision,
    };
    this.renderPresetStrip();
    this.renderAllPanels();

    if (dmEval) {
      const elapsed = dmEval.inferenceTime ?? performance.now() - t0;
      if (this.el['dv-result-meta']) {
        this.el['dv-result-meta'].textContent = `9/9 rules · ${Math.round(elapsed)} ms`;
      }
      this.onStatus?.('Evaluated photo compliance', elapsed);
    }
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
      const dmEval = await this.evaluateSourceWithDecisionMaker(
        this.liveFrameCanvas,
        this.currentPhotoObj.rawEg2Vision
      );
      if (!this.webcamActive) return;

      if (dmEval) {
        this.currentPhotoObj.rawEg2Vision = dmEval.rawEg2Vision;
        this.currentPhotoObj.evaluated = true;
        this.renderAllPanels();

        const elapsed = Math.max(8, Math.round(dmEval.inferenceTime ?? performance.now() - t0));
        if (this.el['dv-result-meta']) {
          this.el['dv-result-meta'].textContent = `Live · ${elapsed} ms`;
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
      if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Stop Webcam';
      this.el['dv-webcam-btn']?.classList.add('active');
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
    if (this.el['dv-webcam-btn-label']) this.el['dv-webcam-btn-label'].textContent = 'Webcam';
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
