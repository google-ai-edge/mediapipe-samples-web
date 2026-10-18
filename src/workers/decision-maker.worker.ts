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

function shouldSuppressCppLog(args: unknown[]): boolean {
  const first = typeof args[0] === 'string' ? args[0].trimStart() : '';
  return (
    first.startsWith('INFO:') ||
    first.startsWith('VERBOSE:') ||
    first.startsWith('WARNING:') ||
    first.startsWith('I0000 ') ||
    first.startsWith('W0000 ')
  );
}

const origConsoleError = console.error.bind(console);
const origConsoleWarn = console.warn.bind(console);
const origConsoleLog = console.log.bind(console);
console.error = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleError(...args);
};
console.warn = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleWarn(...args);
};
console.log = (...args: unknown[]) => {
  if (!shouldSuppressCppLog(args)) origConsoleLog(...args);
};

const DEFAULT_TRUE_DESC = 'yes, the statement holds';
const DEFAULT_FALSE_DESC = 'no, the statement does not hold';

/**
 * Simple minimum-length check: requires at least 3 letters (or 2 CJK characters)
 * so empty whitespace or bare punctuation (e.g. "", ";;") falls back to the
 * question's prior rather than producing arbitrary embeddings.
 */
function isTooShort(rawText: string): boolean {
  const text = (rawText ?? '').trim();
  if (text.length < 2) return true;
  const cjk = (text.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) ?? []).length;
  if (cjk >= 2) return false;
  const letters = (text.match(/\p{L}/gu) ?? []).length;
  return letters < 3;
}

class DecisionMakerWorker extends BaseWorker<DecisionMaker> {
  private isEmbeddingGemma = false;

  protected async initializeTask(): Promise<void> {
    const fileset = await FilesetResolver.forDecisionTasks(this.getWasmPath(), true);
    const cb = Date.now();
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${cb}`;
    fileset.wasmBinaryPath = `${fileset.wasmBinaryPath}?cb=${cb}`;

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
      // With @mediapipe/tasks-decision 1.1.1, larger budgets (e.g. 4096) make LiteRT-LM
      // EmbeddingEngine creation fail on WebGPU for the EmbeddingGemma-2 models.
      maxNumTokens: 256,
    });

    // Serialize all asyncify Wasm calls through a per-module promise queue so
    // concurrent evaluate/prewarm calls never re-enter a suspended Wasm stack
    // while holding absl::Mutex.
    const wasmMod =
      (this.taskInstance as any)?.j?.h ?? (this.taskInstance as any)?.o?.h ?? (this.taskInstance as any)?.i?.h;
    if (wasmMod) {
      if (!wasmMod._callQueue) wasmMod._callQueue = Promise.resolve();
      for (const fn of [
        'evaluateBoolean',
        'evaluateChoice',
        'evaluateScore',
        'evaluateSchema',
        'prewarmSchema',
        'evaluateContextBoolean',
        'evaluateContextChoice',
        'evaluateContextScore',
        'evaluateContextSchema',
      ]) {
        if (typeof wasmMod[fn] === 'function' && !wasmMod[fn]._promised) {
          const orig = wasmMod[fn].bind(wasmMod);
          wasmMod[fn] = (...args: any[]) => {
            const next = wasmMod._callQueue.then(() => Promise.resolve(orig(...args)));
            wasmMod._callQueue = next.then(
              () => {},
              () => {}
            );
            return next;
          };
          wasmMod[fn]._promised = true;
        }
      }
    }
  }

  /**
   * Evaluates a binary/boolean question.
   * - If the input is empty or too short (< 3 letters, e.g. ";;"), errs on the
   *   side indicated by `threshold` (`true` when `threshold < 0.5` such as spam
   *   detection, `false` when `threshold >= 0.5` such as urgent outage or refund).
   * - Otherwise, evaluates directly with the model using the custom Yes/No option
   *   descriptions so the model naturally errs toward whichever option describes
   *   the default/fallback case.
   */
  private async evaluateBooleanSmart(
    text: string,
    question: {
      condition: string;
      threshold?: number;
      context?: string;
      options?: Array<{ label: string; description: string }>;
    },
    imageBytes?: Uint8Array
  ): Promise<{ value: boolean; probabilityTrue: number; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    const threshold = typeof question.threshold === 'number' ? question.threshold : 0.5;
    const hasImage = Boolean(imageBytes && imageBytes.byteLength > 0);

    if (!hasImage && isTooShort(text)) {
      const errOnTrue = threshold < 0.5;
      return {
        value: errOnTrue,
        probabilityTrue: errOnTrue ? 1 : 0,
        lowSignal: true,
      };
    }

    const opts = question.options ?? [];
    const customTrue = opts.find((o) => o.label === 'true')?.description?.trim() ?? '';
    const customFalse = opts.find((o) => o.label === 'false')?.description?.trim() ?? '';
    const hasCustomOptions =
      (customTrue !== '' && customTrue !== DEFAULT_TRUE_DESC) ||
      (customFalse !== '' && customFalse !== DEFAULT_FALSE_DESC);

    if (this.isEmbeddingGemma || hasCustomOptions || hasImage) {
      const cond = question.condition.trim();
      const trueDesc =
        customTrue && customTrue !== DEFAULT_TRUE_DESC
          ? customTrue
          : `Yes, the input text clearly satisfies this condition with concrete details: ${cond}`;
      const falseDesc =
        customFalse && customFalse !== DEFAULT_FALSE_DESC
          ? customFalse
          : `No, the input text does NOT satisfy this condition, is opposite, or lacks concrete evidence: ${cond}`;

      const choiceRes = await this.evaluateChoiceSmart(
        text,
        {
          instructions: this.isEmbeddingGemma ? '' : cond,
          criteria: {
            false: falseDesc,
            true: trueDesc,
          },
          ...(!this.isEmbeddingGemma && question.context ? { context: question.context } : {}),
        },
        imageBytes
      );
      const rawProbs = choiceRes.probabilities ?? {};
      const pTrue = rawProbs['true'] ?? 0;
      return {
        value: pTrue >= threshold,
        probabilityTrue: pTrue,
      };
    }

    return dm.evaluateBoolean(text, question as any);
  }

  private async evaluateChoiceSmart(
    text: string,
    question: {
      criteria?: Record<string, string>;
      /** Alternative to `criteria`, as accepted by ChoiceQuestion (used e.g. by the Dino game). */
      options?: Array<{ label: string; description?: string; imageBytes?: Uint8Array }>;
      instructions?: string;
      context?: string;
      temperature?: number;
      normalizePrior?: boolean;
    },
    imageBytes?: Uint8Array
  ): Promise<{ selectedKey: string; probabilities: Record<string, number>; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    // Normalize `options: [{label, description}]` into `criteria: {label: description}`.
    const criteria: Record<string, string> = {
      ...Object.fromEntries((question.options ?? []).map((o) => [o.label, o.description ?? ''])),
      ...(question.criteria ?? {}),
    };
    const keys = Object.keys(criteria);
    if (keys.length === 0) {
      return { selectedKey: '', probabilities: {} };
    }

    const hasImage = Boolean(imageBytes && imageBytes.byteLength > 0);
    const hasOptionMedia = (question.options ?? []).some((o) => o.imageBytes && o.imageBytes.byteLength > 0);
    if (!hasImage && !hasOptionMedia && isTooShort(text)) {
      const fallbackKey = keys[keys.length - 1];
      const probs: Record<string, number> = {};
      for (const k of keys) probs[k] = k === fallbackKey ? 1 : 0;
      return { selectedKey: fallbackKey, probabilities: probs, lowSignal: true };
    }

    const input = hasImage ? { text: text ?? '', imageBytes } : text;

    if (this.isEmbeddingGemma) {
      return dm.evaluateChoice(input, {
        instructions: hasOptionMedia ? (question.instructions ?? '') : '',
        criteria,
        ...(hasOptionMedia ? { options: question.options } : {}),
        ...(typeof question.temperature === 'number' ? { temperature: question.temperature } : {}),
        ...(typeof question.normalizePrior === 'boolean' ? { normalizePrior: question.normalizePrior } : {}),
      } as any);
    }
    return dm.evaluateChoice(input, {
      instructions: question.instructions,
      context: question.context,
      criteria,
      ...(hasOptionMedia ? { options: question.options } : {}),
      ...(typeof question.temperature === 'number' ? { temperature: question.temperature } : {}),
      ...(typeof question.normalizePrior === 'boolean' ? { normalizePrior: question.normalizePrior } : {}),
    } as any);
  }

  private async evaluateScoreSmart(
    text: string,
    question: {
      rubric: string[];
      instructions?: string;
      context?: string;
    },
    imageBytes?: Uint8Array
  ): Promise<{ selectedIndex: number; expectedScore: number; probabilities: number[]; lowSignal?: boolean }> {
    const dm = this.taskInstance!;
    const n = question.rubric?.length ?? 0;
    if (n === 0) {
      return { selectedIndex: 0, expectedScore: 0, probabilities: [] };
    }

    const hasImage = Boolean(imageBytes && imageBytes.byteLength > 0);
    if (!hasImage && isTooShort(text)) {
      const probs = question.rubric.map((_, i) => (i === 0 ? 1 : 0));
      return {
        selectedIndex: 0,
        expectedScore: 0,
        probabilities: probs,
        lowSignal: true,
      };
    }

    const cleanRubric = question.rubric.map((entry) => entry.replace(/^\d+\s*:\s*/, ''));

    if (this.isEmbeddingGemma || hasImage) {
      const criteria: Record<string, string> = {};
      cleanRubric.forEach((entry, i) => {
        const sep = entry.indexOf(': ');
        criteria[`__lvl_${i}__`] = sep > 0 ? entry.slice(sep + 2) : entry;
      });
      const choiceRes = await this.evaluateChoiceSmart(
        text,
        {
          instructions: '',
          criteria,
        },
        imageBytes
      );
      const rawProbs = choiceRes.probabilities ?? {};
      const probs = cleanRubric.map((_, i) => rawProbs[`__lvl_${i}__`] ?? 0);
      let best = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) best = i;
      }
      const expected0 = probs.reduce((sum, p, i) => sum + i * p, 0);
      return {
        selectedIndex: best,
        expectedScore: expected0,
        probabilities: probs,
      };
    }

    const scoreRes = await dm.evaluateScore(text, {
      ...question,
      rubric: cleanRubric,
    } as any);
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

    const imageBytes: Uint8Array | undefined =
      data.imageBytes instanceof Uint8Array
        ? data.imageBytes
        : data.imageBytes
          ? new Uint8Array(data.imageBytes)
          : undefined;

    const startTimeMs = performance.now();
    const result =
      data.kind === 'schema' || data.kind === 'polymorphic'
        ? await this.evaluateSchema(data.text ?? '', data.question, imageBytes)
        : data.kind === 'choice'
          ? await this.evaluateChoiceSmart(data.text ?? '', data.question, imageBytes)
          : data.kind === 'score'
            ? await this.evaluateScoreSmart(data.text ?? '', data.question, imageBytes)
            : await this.evaluateBooleanSmart(data.text ?? '', data.question, imageBytes);
    const inferenceTime = performance.now() - startTimeMs;
    self.postMessage({ type: 'DECIDE_RESULT', id: data.id, result, inferenceTime });
  }

  private async evaluateSchema(text: string, schema: any, imageBytes?: Uint8Array): Promise<Record<string, any>> {
    const hasImage = Boolean(imageBytes && imageBytes.byteLength > 0);
    if (hasImage) {
      const dm = this.taskInstance!;
      const questions = (schema.questions ?? []).map((q: any) => ({
        ...q,
        prompt: q.prompt ?? '',
        normalizePrior: q.normalizePrior ?? true,
      }));
      return dm.evaluate(
        { text: text ?? '', imageBytes },
        {
          ...schema,
          context: schema.context ?? '',
          questions,
        }
      );
    }

    const context: string | undefined = schema.context || undefined;
    const result: Record<string, any> = {};
    for (const q of schema.questions ?? []) {
      const type = String(q.type).toLowerCase();
      const options: { label: string; description?: string }[] = q.options ?? [];
      if (type === 'binary' || type === 'boolean') {
        const threshold = typeof q.threshold === 'number' ? q.threshold : 0.5;
        const r = await this.evaluateBooleanSmart(
          text,
          {
            condition: q.prompt,
            context,
            threshold,
            options: options.map((o) => ({ label: o.label, description: o.description ?? '' })),
          },
          imageBytes
        );
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
        const r = await this.evaluateChoiceSmart(text, { criteria, instructions: q.prompt, context }, imageBytes);
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
        const r = await this.evaluateScoreSmart(text, { rubric, instructions: q.prompt, context }, imageBytes);
        const probs: number[] = r.probabilities ?? [];
        const best = r.selectedIndex ?? probs.indexOf(Math.max(...probs));
        result[q.id] = {
          id: q.id,
          label: options[best]?.label ?? String(best + 1),
          confidence: probs[best] ?? 0,
          expectedScore: r.expectedScore,
          lowSignal: r.lowSignal,
          probabilities: options.map((o, i) => ({ label: o.label, probability: probs[i] ?? 0 })),
        };
      }
    }
    return result;
  }
}

new DecisionMakerWorker();
