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
    const result =
      data.kind === 'choice'
        ? await dm.evaluateChoice(data.text, data.question)
        : data.kind === 'score'
          ? await dm.evaluateScore(data.text, data.question)
          : await dm.evaluateBoolean(data.text, data.question);
    const inferenceTime = performance.now() - startTimeMs;
    self.postMessage({ type: 'DECIDE_RESULT', id: data.id, result, inferenceTime });
  }
}

new DecisionMakerWorker();
