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

import { DecisionMaker, FilesetResolver } from '@mediapipe/tasks-decision';
import { BaseWorker } from './base-worker';

const DEFAULT_TRUE_DESC = 'yes, the statement holds';
const DEFAULT_FALSE_DESC = 'no, the statement does not hold';

/**
 * Contrastive negative anchors evaluated alongside task options on bi-encoder
 * models so that bare keyword demands without context (e.g. "urgent", "refund",
 * "this is urgent help me asap", "I want a refund right now") and unrelated /
 * off-topic inputs (e.g. "hello how are you", "what is the weather", "I like pizza")
 * do not trigger false-positive decisions.
 */
const NEGATIVE_GUARD_ANCHORS: Record<string, string> = {
  __guard_bare_claim__:
    'Vague demand, bare keyword, or unsubstantiated claim without concrete details, such as just saying urgent, emergency, help me asap, I want a refund, give me my money back, it is broken, fix this now, yes, or true.',
  __guard_offtopic__:
    'Unrelated off-topic message, casual greeting, test string, trivia question, weather inquiry, food preference, or general conversation unrelated to the task.',
};

const FALLBACK_LABEL_REGEX =
  /^(deny|reject|false|no|none|benign|legitimate|safe|normal|primary|allow|backlog|standard_exchange|on_device|unactionable|needs_clarification|insufficient_info|general|other|irrelevant|unrelated)/i;

/**
 * Detects empty strings, whitespace, punctuation-only strings (e.g. ";;", "..."),
 * keyboard mash / non-lexical gibberish (e.g. "asdfghjkl", "qwerty"), and
 * context-free 1-2 word bare keywords (e.g. "urgent", "refund", "damaged",
 * "critical bug") that lack substantive context for a reliable decision.
 */
export function isLowSignalOrGibberish(rawText: string): boolean {
  const text = (rawText ?? '').trim();
  if (!text) return true;

  // Must contain at least 2 Unicode letters.
  const letters = (text.match(/\p{L}/gu) ?? []).length;
  if (letters < 2) return true;

  // If the text contains non-Latin script characters (e.g. CJK), require at least
  // a short phrase (>= 5 non-space characters) rather than a bare 2-char keyword.
  const cjkOrNonLatin = (text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) ?? [])
    .length;
  if (cjkOrNonLatin >= 2) {
    return text.replace(/\s+/g, '').length < 5;
  }

  // Extract ASCII / Latin alphabetic tokens.
  const tokens = text
    .toLowerCase()
    .split(/[^a-z]+/)
    .filter((w) => w.length > 0);
  if (tokens.length === 0) return true;

  // Require at least 3 alphabetic words so bare 1-2 word trigger keywords
  // (e.g. "urgent", "refund", "refund please", "damaged", "critical bug")
  // without any descriptive context are rejected as insufficient context.
  if (tokens.length < 3) return true;

  // Known valid short words (2-3 letters) and common technical acronyms.
  const knownShortWords = new Set([
    'a',
    'an',
    'am',
    'as',
    'at',
    'be',
    'by',
    'do',
    'go',
    'he',
    'hi',
    'id',
    'if',
    'in',
    'is',
    'it',
    'me',
    'my',
    'no',
    'of',
    'ok',
    'on',
    'or',
    'so',
    'to',
    'up',
    'us',
    'we',
    'all',
    'and',
    'api',
    'app',
    'are',
    'ask',
    'aws',
    'bad',
    'ban',
    'big',
    'bot',
    'box',
    'bug',
    'bus',
    'but',
    'buy',
    'can',
    'car',
    'cat',
    'ceo',
    'cli',
    'cpu',
    'csv',
    'cut',
    'day',
    'dev',
    'did',
    'dns',
    'doc',
    'dog',
    'due',
    'end',
    'env',
    'err',
    'etc',
    'eye',
    'far',
    'fee',
    'few',
    'fix',
    'for',
    'fun',
    'gcp',
    'get',
    'git',
    'got',
    'gpu',
    'gui',
    'guy',
    'had',
    'has',
    'her',
    'him',
    'his',
    'hit',
    'hot',
    'how',
    'iam',
    'ice',
    'inc',
    'ios',
    'ip',
    'its',
    'job',
    'jwt',
    'key',
    'kid',
    'kms',
    'lan',
    'law',
    'let',
    'llm',
    'log',
    'lot',
    'low',
    'mac',
    'man',
    'map',
    'max',
    'may',
    'mem',
    'men',
    'met',
    'min',
    'mix',
    'mfa',
    'mod',
    'mom',
    'msg',
    'net',
    'new',
    'nil',
    'non',
    'nor',
    'not',
    'now',
    'oak',
    'odd',
    'off',
    'oil',
    'okr',
    'old',
    'one',
    'oom',
    'opt',
    'org',
    'our',
    'out',
    'own',
    'pay',
    'pdf',
    'per',
    'pin',
    'pod',
    'pop',
    'pr',
    'pro',
    'put',
    'p99',
    'qa',
    'ram',
    'ran',
    'raw',
    'red',
    'ref',
    'req',
    'res',
    'rip',
    'row',
    'rpc',
    'run',
    'sad',
    'saw',
    'say',
    'sdk',
    'see',
    'set',
    'sha',
    'she',
    'sit',
    'six',
    'sku',
    'sla',
    'slo',
    'sms',
    'sql',
    'sre',
    'ssh',
    'ssl',
    'sso',
    'sub',
    'sum',
    'sun',
    'tag',
    'tap',
    'tax',
    'tea',
    'ten',
    'the',
    'tie',
    'tip',
    'tls',
    'tmp',
    'tok',
    'too',
    'top',
    'try',
    'two',
    'txt',
    'ui',
    'uri',
    'url',
    'use',
    'utc',
    'ux',
    'var',
    'via',
    'vip',
    'vm',
    'vpn',
    'vpc',
    'war',
    'was',
    'way',
    'web',
    'wet',
    'who',
    'why',
    'win',
    'won',
    'xml',
    'yet',
    'you',
    'zip',
  ]);

  // Common keyboard-row mash substrings.
  const mashPatterns = /asdf|sdfg|dfgh|fghj|ghjk|hjkl|qwerty|werty|ertyu|zxcv|xcvb|cvbn/i;

  let validWords = 0;
  for (const w of tokens) {
    if (/(.)\1{2,}/.test(w)) continue;
    if (mashPatterns.test(w)) continue;

    if (w.length <= 3 && knownShortWords.has(w)) {
      validWords++;
      continue;
    }

    if (w.length >= 4) {
      const vowels = (w.match(/[aeiouy]/g) ?? []).length;
      const ratio = vowels / w.length;
      if (vowels >= 1 && ratio >= 0.15 && ratio <= 0.85 && !/[bcdfghjklmnpqrstvwxz]{5,}/.test(w)) {
        validWords++;
      }
    }
  }

  return validWords === 0 || validWords / tokens.length < 0.5;
}

class DecisionMakerWorker extends BaseWorker<DecisionMaker> {
  private isEmbeddingGemma = false;

  protected async initializeTask(): Promise<void> {
    const fileset = await FilesetResolver.forDecisionTasks(this.getWasmPath(), true);
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}`; // Force reload

    const file: File | undefined = this.currentOptions.modelFile;
    const url: string = this.currentOptions.modelAssetPath ?? '';
    const modelNameHint = `${file?.name ?? ''} ${url}`.toLowerCase();
    this.isEmbeddingGemma = modelNameHint.includes('embeddinggemma') || modelNameHint.endsWith('.litertlm');

    // Stream the model (cached download or uploaded file) so it isn't copied
    // into one big JS buffer; the size lets the wasm allocate it in one pass.
    const { stream, size } = await this.openModelStream();
    this.taskInstance = await DecisionMaker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetBuffer: stream.getReader(),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      ...(size ? { modelAssetSize: size } : {}),
      // Same as the Android sample; the default (256) is too small for longer inputs.
      maxNumTokens: 4096,
    });

    // In @mediapipe/tasks-decision@1.1.0, synchronous embind execution (when using
    // the CPU delegate) returns a direct value instead of a Promise, while
    // decision_bundle.mjs calls `.then()` on the return value. Wrap the underlying
    // Wasm methods with Promise.resolve() so both GPU and CPU delegates work seamlessly.
    const wasmMod = (this.taskInstance as any)?.i?.h;
    if (wasmMod) {
      for (const fn of ['evaluateBoolean', 'evaluateChoice', 'evaluateScore', 'evaluateSchema', 'prewarmSchema']) {
        if (typeof wasmMod[fn] === 'function' && !wasmMod[fn]._promised) {
          const orig = wasmMod[fn].bind(wasmMod);
          wasmMod[fn] = (...args: any[]) => Promise.resolve(orig(...args));
          wasmMod[fn]._promised = true;
        }
      }
    }
  }

  /**
   * Evaluates a binary/boolean question with:
   * 1. Low-signal / gibberish / bare-keyword protection (returns No with P(yes) = 0.00).
   * 2. Contrastive multi-anchor evaluation on bi-encoder models (EmbeddingGemma-2)
   *    so unsubstantiated demands and off-topic text are absorbed by negative guard anchors.
   */
  private async evaluateBooleanSmart(
    text: string,
    question: {
      condition: string;
      threshold?: number;
      context?: string;
      options?: Array<{ label: string; description: string }>;
    }
  ): Promise<{ value: boolean; probabilityTrue: number; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    const threshold = typeof question.threshold === 'number' ? question.threshold : 0.5;

    if (isLowSignalOrGibberish(text)) {
      return { value: false, probabilityTrue: 0, lowSignal: true };
    }

    const opts = question.options ?? [];
    const customTrue = opts.find((o) => o.label === 'true')?.description?.trim() ?? '';
    const customFalse = opts.find((o) => o.label === 'false')?.description?.trim() ?? '';
    const hasCustomOptions =
      (customTrue !== '' && customTrue !== DEFAULT_TRUE_DESC) ||
      (customFalse !== '' && customFalse !== DEFAULT_FALSE_DESC);

    if (this.isEmbeddingGemma || hasCustomOptions) {
      const cond = question.condition.trim();
      const trueDesc =
        customTrue && customTrue !== DEFAULT_TRUE_DESC
          ? customTrue
          : `Yes, the input text clearly satisfies this condition with concrete details: ${cond}`;
      const falseDesc =
        customFalse && customFalse !== DEFAULT_FALSE_DESC
          ? customFalse
          : `No, the input text does NOT satisfy this condition, is opposite, or lacks concrete evidence: ${cond}`;

      const choiceRes = await dm.evaluateChoice(text, {
        instructions: this.isEmbeddingGemma ? '' : cond,
        criteria: {
          opt_false: falseDesc,
          opt_true: trueDesc,
          ...(this.isEmbeddingGemma ? NEGATIVE_GUARD_ANCHORS : {}),
        },
        ...(!this.isEmbeddingGemma && question.context ? { context: question.context } : {}),
      });
      const rawProbs = choiceRes.probabilities ?? {};
      const pTrue = rawProbs['opt_true'] ?? 0;
      const guardMass = (rawProbs['__guard_bare_claim__'] ?? 0) + (rawProbs['__guard_offtopic__'] ?? 0);
      const guardTriggered =
        choiceRes.selectedKey === '__guard_bare_claim__' ||
        choiceRes.selectedKey === '__guard_offtopic__' ||
        guardMass >= 0.25;
      return {
        value: !guardTriggered && pTrue >= threshold,
        probabilityTrue: guardTriggered ? 0 : pTrue,
        ...(guardTriggered ? { lowSignal: true } : {}),
      };
    }

    return dm.evaluateBoolean(text, question as any);
  }

  /**
   * Evaluates a categorical/choice question with low-signal and guard-anchor protection.
   */
  private async evaluateChoiceSmart(
    text: string,
    question: {
      criteria: Record<string, string>;
      instructions?: string;
      context?: string;
    }
  ): Promise<{ selectedKey: string; probabilities: Record<string, number>; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    const keys = Object.keys(question.criteria ?? {});
    if (keys.length === 0) {
      return { selectedKey: '', probabilities: {} };
    }
    const fallbackKey = keys.find((k) => FALLBACK_LABEL_REGEX.test(k));

    if (isLowSignalOrGibberish(text)) {
      if (fallbackKey) {
        const probs: Record<string, number> = {};
        for (const k of keys) probs[k] = k === fallbackKey ? 1 : 0;
        return { selectedKey: fallbackKey, probabilities: probs, lowSignal: true };
      }
      const uniform = 1 / keys.length;
      const probs: Record<string, number> = {};
      for (const k of keys) probs[k] = uniform;
      return { selectedKey: keys[0], probabilities: probs, lowSignal: true };
    }

    if (this.isEmbeddingGemma) {
      const raw = await dm.evaluateChoice(text, {
        instructions: '',
        criteria: {
          ...question.criteria,
          ...NEGATIVE_GUARD_ANCHORS,
        },
      });
      const rawProbs = raw.probabilities ?? {};
      const guardMass = (rawProbs['__guard_bare_claim__'] ?? 0) + (rawProbs['__guard_offtopic__'] ?? 0);
      const guardTriggered =
        raw.selectedKey === '__guard_bare_claim__' || raw.selectedKey === '__guard_offtopic__' || guardMass >= 0.25;

      if (guardTriggered && fallbackKey) {
        const probs: Record<string, number> = {};
        for (const k of keys) probs[k] = k === fallbackKey ? 1 : 0;
        return {
          selectedKey: fallbackKey,
          probabilities: probs,
          lowSignal: true,
        };
      }

      const probs: Record<string, number> = {};
      for (const k of keys) probs[k] = rawProbs[k] ?? 0;

      if (fallbackKey) {
        probs[fallbackKey] = (probs[fallbackKey] ?? 0) + guardMass;
      } else {
        const sum = keys.reduce((s, k) => s + probs[k], 0);
        if (sum > 0) {
          for (const k of keys) probs[k] = probs[k] / sum;
        }
      }

      let bestKey = keys[0];
      for (const k of keys) {
        if ((probs[k] ?? 0) > (probs[bestKey] ?? 0)) bestKey = k;
      }
      return {
        selectedKey: bestKey,
        probabilities: probs,
        ...(guardTriggered ? { lowSignal: true } : {}),
      };
    }

    return dm.evaluateChoice(text, question as any);
  }

  /**
   * Evaluates an ordinal/score question with low-signal and guard-anchor protection
   * so bare keywords ("urgent") or off-topic inputs fall back to the lowest/baseline level.
   */
  private async evaluateScoreSmart(
    text: string,
    question: {
      rubric: string[];
      instructions?: string;
      context?: string;
    }
  ): Promise<{ selectedIndex: number; expectedScore: number; probabilities: number[]; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    const n = question.rubric?.length ?? 0;
    if (n === 0) {
      return { selectedIndex: 0, expectedScore: 0, probabilities: [] };
    }

    // Identify baseline index: if a level is explicitly labeled "Neutral" (e.g. Customer satisfaction),
    // use that index; otherwise use index 0 (Level 1: minimal/none/clean).
    const neutralIdx = question.rubric.findIndex((r) => /^neutral\b/i.test(r.trim()));
    const baselineIdx = neutralIdx >= 0 ? neutralIdx : 0;

    if (isLowSignalOrGibberish(text)) {
      const probs = question.rubric.map((_, i) => (i === baselineIdx ? 1 : 0));
      return {
        selectedIndex: baselineIdx,
        expectedScore: baselineIdx,
        probabilities: probs,
        lowSignal: true,
      };
    }

    if (this.isEmbeddingGemma) {
      const criteria: Record<string, string> = {};
      question.rubric.forEach((entry, i) => {
        const sep = entry.indexOf(': ');
        criteria[`__lvl_${i}__`] = sep > 0 ? entry.slice(sep + 2) : entry;
      });
      const choiceRes = await dm.evaluateChoice(text, {
        instructions: '',
        criteria: {
          ...criteria,
          ...NEGATIVE_GUARD_ANCHORS,
        },
      });
      const rawProbs = choiceRes.probabilities ?? {};
      const guardMass = (rawProbs['__guard_bare_claim__'] ?? 0) + (rawProbs['__guard_offtopic__'] ?? 0);
      const guardTriggered =
        choiceRes.selectedKey === '__guard_bare_claim__' ||
        choiceRes.selectedKey === '__guard_offtopic__' ||
        guardMass >= 0.25;

      if (guardTriggered) {
        const probs = question.rubric.map((_, i) => (i === baselineIdx ? 1 : 0));
        return {
          selectedIndex: baselineIdx,
          expectedScore: baselineIdx,
          probabilities: probs,
          lowSignal: true,
        };
      }

      const probs = question.rubric.map((_, i) => rawProbs[`__lvl_${i}__`] ?? 0);
      probs[baselineIdx] = (probs[baselineIdx] ?? 0) + guardMass;

      let best = baselineIdx;
      for (let i = 0; i < probs.length; i++) {
        if (probs[i] > probs[best]) best = i;
      }
      const expected0 = probs.reduce((sum, p, i) => sum + i * p, 0);
      return {
        selectedIndex: best,
        expectedScore: expected0,
        probabilities: probs,
      };
    }

    const scoreRes = await dm.evaluateScore(text, question as any);
    let best = 0;
    for (let i = 1; i < scoreRes.probabilities.length; i++) {
      if (scoreRes.probabilities[i] > scoreRes.probabilities[best]) best = i;
    }
    return {
      selectedIndex: best,
      expectedScore: scoreRes.expectedScore,
      probabilities: scoreRes.probabilities,
    };
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type !== 'DECIDE') return;
    if (!this.taskInstance) {
      self.postMessage({ type: 'ERROR', error: 'Decision Maker not initialized' });
      return;
    }

    const startTimeMs = performance.now();
    // 'schema' / 'polymorphic': a whole ClassifierSchema, all questions evaluated on the same input.
    const result =
      data.kind === 'schema' || data.kind === 'polymorphic'
        ? await this.evaluateSchema(data.text, data.question)
        : data.kind === 'choice'
          ? await this.evaluateChoiceSmart(data.text, data.question)
          : data.kind === 'score'
            ? await this.evaluateScoreSmart(data.text, data.question)
            : await this.evaluateBooleanSmart(data.text, data.question);
    const inferenceTime = performance.now() - startTimeMs;
    self.postMessage({ type: 'DECIDE_RESULT', id: data.id, result, inferenceTime });
  }

  /**
   * Evaluates a ClassifierSchema using the smart contrastive and low-signal-guarded
   * evaluators for each binary, categorical, and ordinal question.
   */
  private async evaluateSchema(text: string, schema: any): Promise<Record<string, any>> {
    const context: string | undefined = schema.context || undefined;
    const result: Record<string, any> = {};
    for (const q of schema.questions ?? []) {
      const type = String(q.type).toLowerCase();
      const options: { label: string; description?: string }[] = q.options ?? [];
      if (type === 'binary' || type === 'boolean') {
        const threshold = typeof q.threshold === 'number' ? q.threshold : 0.5;
        const r = await this.evaluateBooleanSmart(text, {
          condition: q.prompt,
          context,
          threshold,
          options: options.map((o) => ({ label: o.label, description: o.description ?? '' })),
        });
        const p = r.probabilityTrue;
        result[q.id] = {
          id: q.id,
          label: r.value ? 'true' : 'false',
          confidence: r.value ? p : 1 - p,
          probability: p,
          lowSignal: r.lowSignal,
          probabilities: [
            { label: 'true', probability: p },
            { label: 'false', probability: 1 - p },
          ],
        };
      } else if (type === 'categorical' || type === 'choice') {
        const criteria = Object.fromEntries(options.map((o) => [o.label, o.description ?? '']));
        const r = await this.evaluateChoiceSmart(text, { criteria, instructions: q.prompt, context });
        const probs: Record<string, number> = r.probabilities ?? {};
        result[q.id] = {
          id: q.id,
          label: r.selectedKey,
          confidence: probs[r.selectedKey] ?? 0,
          lowSignal: r.lowSignal,
          probabilities: options.map((o) => ({ label: o.label, probability: probs[o.label] ?? 0 })),
        };
      } else {
        const rubric = options.map((o) => (o.description ? `${o.label}: ${o.description}` : o.label));
        const r = await this.evaluateScoreSmart(text, { rubric, instructions: q.prompt, context });
        const probs: number[] = r.probabilities ?? [];
        const best = r.selectedIndex ?? probs.indexOf(Math.max(...probs));
        result[q.id] = {
          id: q.id,
          label: options[best]?.label ?? String(best + 1),
          confidence: probs[best] ?? 0,
          expectedScore: r.expectedScore, // 0-based, converted to 1..N in UI
          lowSignal: r.lowSignal,
          probabilities: options.map((o, i) => ({ label: o.label, probability: probs[i] ?? 0 })),
        };
      }
    }
    return result;
  }
}

new DecisionMakerWorker();
