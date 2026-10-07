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
 * Decision Maker: a text playground for the MediaPipe DecisionMaker task
 * (@mediapipe/tasks-decision). Ask Boolean, Choice and Score questions about
 * any text, combine several questions on one input, or paste a request as JSON
 * (see decision-maker-text.ts).
 *
 * The model is shared with the Dino Game page (see decision-runtime.ts), so a
 * model loaded here is still loaded there.
 */

import template from '../templates/decision-maker.html?raw';
import { InferenceTimer } from '../components/inference-timer';
import { decisionRuntime } from '../components/decision-runtime';
import { mountDecisionModelPanel } from '../components/decision-model-panel';
import { DecisionTextPlayground } from './decision-maker-text';

class DecisionMakerTask {
  /** Shared "Inference Time" badge + history graph, same as the other tasks. */
  private inferenceTimer = new InferenceTimer();
  private textPlayground!: DecisionTextPlayground;
  private disposers: (() => void)[] = [];
  private el: Record<string, HTMLElement> = {};

  constructor(private container: HTMLElement) {}

  init() {
    this.container.innerHTML = template;
    this.container.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));

    this.textPlayground = new DecisionTextPlayground(
      this.el['dm-text-view'],
      (kind, text, question) => decisionRuntime.ask(text, kind, question),
      (text, inferenceTime) => {
        if (inferenceTime === undefined) return this.setStatus(text);
        this.inferenceTimer.record(inferenceTime, decisionRuntime.delegate);
        this.setStatus(`Done in ${Math.round(inferenceTime)}ms`);
      }
    );
    this.textPlayground.init();

    // Model controls and state are shared with the Dino Game page.
    this.disposers.push(mountDecisionModelPanel(this.el['dm-model-panel'], this.el['dm-delegate-panel']));
    this.disposers.push(
      decisionRuntime.subscribe((event) => {
        if (event.type === 'loading') this.inferenceTimer.resetRollingWindow();
        else if (event.type === 'status') this.setStatus(event.text);
        else if (event.type === 'error') this.setStatus(`Error: ${event.error}`);
        else if (event.type === 'state') this.textPlayground.setReady(decisionRuntime.ready);
      })
    );

    this.inferenceTimer.mount();
    this.textPlayground.setReady(decisionRuntime.ready);
    if (decisionRuntime.ready) this.setStatus('Model ready');
    else if (decisionRuntime.loading) this.setStatus(`Loading ${decisionRuntime.loadingLabel}...`);
    else this.setStatus('Select a model and press "Load Model" to begin');
  }

  cleanup() {
    this.inferenceTimer.cleanup();
    // The worker and model stay loaded in decisionRuntime for the next page.
    for (const dispose of this.disposers) dispose();
    this.disposers = [];
  }

  private setStatus(text: string) {
    this.el['status-message'].textContent = text;
    this.inferenceTimer.syncStatusVisibility(text);
  }
}

let activeTask: DecisionMakerTask | null = null;

export async function setupDecisionMaker(container: HTMLElement) {
  activeTask = new DecisionMakerTask(container);
  activeTask.init();
}

export function cleanupDecisionMaker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
