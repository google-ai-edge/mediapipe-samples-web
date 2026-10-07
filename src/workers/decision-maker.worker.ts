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

class DecisionMakerWorker extends BaseWorker<DecisionMaker> {
  protected async initializeTask(): Promise<void> {
    const fileset = await FilesetResolver.forDecisionTasks(this.getWasmPath(), true);
    fileset.wasmLoaderPath = `${fileset.wasmLoaderPath}?cb=${Date.now()}`; // Force reload

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
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type !== 'DECIDE') return;
    if (!this.taskInstance) {
      self.postMessage({ type: 'ERROR', error: 'Decision Maker not initialized' });
      return;
    }

    const startTimeMs = performance.now();
    const dm = this.taskInstance;
    // 'schema': a whole ClassifierSchema, all questions evaluated on the same input.
    const result =
      data.kind === 'schema'
        ? await this.evaluateSchema(dm, data.text, data.question)
        : data.kind === 'choice'
          ? await dm.evaluateChoice(data.text, data.question)
          : data.kind === 'score'
            ? await dm.evaluateScore(data.text, data.question)
            : await dm.evaluateBoolean(data.text, data.question);
    const inferenceTime = performance.now() - startTimeMs;
    self.postMessage({ type: 'DECIDE_RESULT', id: data.id, result, inferenceTime });
  }

  /**
   * Evaluates a ClassifierSchema with DecisionMaker.evaluate(input, schema).
   * If that native path throws, falls back to one single-question call per
   * question and returns the same ClassifierResult shape ({ [id]: { id, label,
   * confidence, probability?, expectedScore?, probabilities } }).
   */
  private async evaluateSchema(dm: DecisionMaker, text: string, schema: any): Promise<Record<string, any>> {
    try {
      return await dm.evaluate(text, schema);
    } catch (e) {
      console.warn('DecisionMaker.evaluate(schema) failed; evaluating questions one by one.', e);
    }
    const context: string | undefined = schema.context || undefined;
    const result: Record<string, any> = {};
    for (const q of schema.questions ?? []) {
      const type = String(q.type).toLowerCase();
      const options: { label: string; description?: string }[] = q.options ?? [];
      if (type === 'binary' || type === 'boolean') {
        const r = await dm.evaluateBoolean(text, {
          condition: q.prompt,
          context,
          threshold: q.threshold,
          ...(options.length >= 2 ? { options } : {}),
        } as any);
        const p = r.probabilityTrue;
        result[q.id] = {
          id: q.id,
          label: r.value ? 'true' : 'false',
          confidence: r.value ? p : 1 - p,
          probability: p,
          probabilities: [
            { label: 'true', probability: p },
            { label: 'false', probability: 1 - p },
          ],
        };
      } else if (type === 'categorical' || type === 'choice') {
        const criteria = Object.fromEntries(options.map((o) => [o.label, o.description ?? '']));
        const r = await dm.evaluateChoice(text, { criteria, instructions: q.prompt, context } as any);
        const probs: Record<string, number> = r.probabilities ?? {};
        result[q.id] = {
          id: q.id,
          label: r.selectedKey,
          confidence: probs[r.selectedKey] ?? 0,
          probabilities: options.map((o) => ({ label: o.label, probability: probs[o.label] ?? 0 })),
        };
      } else {
        const rubric = options.map((o) => (o.description ? `${o.label}: ${o.description}` : o.label));
        const r = await dm.evaluateScore(text, { rubric, instructions: q.prompt, context } as any);
        const probs: number[] = r.probabilities ?? [];
        const best = probs.indexOf(Math.max(...probs));
        result[q.id] = {
          id: q.id,
          label: options[best]?.label ?? String(best + 1),
          confidence: probs[best] ?? 0,
          expectedScore: probs.reduce((sum, p, i) => sum + i * p, 0), // 0-based, like evaluate()
          probabilities: options.map((o, i) => ({ label: o.label, probability: probs[i] ?? 0 })),
        };
      }
    }
    return result;
  }
}

new DecisionMakerWorker();
