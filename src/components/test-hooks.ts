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
 * Read-only result hooks for automated validation.
 *
 * Tasks call `publishTestResult()` where they already handle a result. The
 * summary is stored on `window.__mpTestResults` so a browser-automation runner
 * can compare the underlying output with a ground truth. The hooks never start
 * inference and never touch the DOM; they are a no-op unless the page is under
 * automation (`navigator.webdriver`).
 */

export interface TestResultRecord {
  /** Increases with every published result, across all tasks. */
  seq: number;
  timestampMs: number;
  task: string;
  payload?: Record<string, unknown>;
  /** Set instead of `payload` when building the summary failed. */
  error?: string;
}

interface TestResultStore {
  seq: number;
  /** The latest result of each task. */
  results: Record<string, TestResultRecord>;
  /** The most recent results of each task, oldest first, for tasks that produce several per action. */
  history: Record<string, TestResultRecord[]>;
}

/** How many recent results are kept per task. */
const HISTORY_LIMIT = 100;

declare global {
  interface Window {
    __mpTestResults?: TestResultStore;
  }
}

/** Whether result summaries are collected (only under browser automation). */
export function isTestHookEnabled(): boolean {
  return typeof navigator !== 'undefined' && navigator.webdriver === true;
}

interface PointLike {
  x: number;
  y: number;
}

/**
 * Summarizes one set of normalized landmarks: how many there are, the box and
 * centre they span, and the position of a few named landmarks (`named` maps a
 * name to a landmark index).
 */
export function summarizeLandmarks(
  landmarks: readonly PointLike[] | undefined,
  named: Record<string, number> = {}
): Record<string, unknown> {
  const points = landmarks ?? [];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let sumX = 0;
  let sumY = 0;
  let nonFinite = 0;
  for (const p of points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      nonFinite++;
      continue;
    }
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
    sumX += p.x;
    sumY += p.y;
  }
  const finite = points.length - nonFinite;
  const namedPoints: Record<string, PointLike | null> = {};
  for (const [name, index] of Object.entries(named)) {
    const p = points[index];
    namedPoints[name] = p ? { x: p.x, y: p.y } : null;
  }
  return {
    count: points.length,
    nonFiniteCount: nonFinite,
    box: finite > 0 ? { minX, minY, maxX, maxY } : null,
    centroid: finite > 0 ? { x: sumX / finite, y: sumY / finite } : null,
    points: namedPoints,
  };
}

/**
 * Publishes the latest result summary of `task`. `buildPayload` is only called
 * when hooks are enabled, and a failure in it is recorded rather than thrown so
 * a hook can never break the task.
 */
export function publishTestResult(task: string, buildPayload: () => Record<string, unknown>): void {
  if (!isTestHookEnabled()) return;
  const store = (window.__mpTestResults ??= { seq: 0, results: {}, history: {} });
  store.seq += 1;
  const record: TestResultRecord = { seq: store.seq, timestampMs: Date.now(), task };
  try {
    record.payload = buildPayload();
  } catch (err) {
    record.error = err instanceof Error ? err.message : String(err);
  }
  store.results[task] = record;
  const history = (store.history[task] ??= []);
  history.push(record);
  if (history.length > HISTORY_LIMIT) history.shift();
}
