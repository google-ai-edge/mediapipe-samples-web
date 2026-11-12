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

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { chromium } from '@playwright/test';
import { TASK_DEFINITIONS, buildValidationMatrix } from './cases.mjs';
import {
  DECISION_TICKETS,
  DINO_SCENE_ACTIONS,
  ID_PHOTO_PLANS,
  IMAGE_EMBEDDER_PAIRS,
  IMAGE_SEGMENTER_PLANS,
  SPEECH_TRANSCRIPT,
  INTERACTIVE_SEGMENTER_PLANS,
  LANGUAGE_DETECTOR_INPUTS,
  SEMANTIC_RETRIEVER_QUERIES,
  SPEECH_FIXTURE,
  TASKS_WITH_RESULT_HOOK,
  TEXT_CLASSIFIER_INPUTS,
  TEXT_EMBEDDER_PAIRS,
  TTS_SENTENCE,
  UNIVERSAL_EMBEDDER_PLANS,
  describeCase,
  getExpectation,
  getFakeMedia,
} from './expectations.mjs';
import { fakeMediaLaunchArgs, writeFakeCameraVideo } from './fake_media.mjs';
import { VERIFICATION, verifyCase, wordErrorRate, wordRecall } from './verify.mjs';
import {
  appendStepSummary,
  createLiveCheckRun,
  formatMarkdownTable,
  loadState,
  saveState,
  summarizeCounts,
  updateLiveCheckRun,
  writeReportFiles,
} from './report.mjs';
import { readInstalledMediaPipeVersions } from './tarballs.mjs';

const BASE_PATH = '/mediapipe-samples-web/';

function parseCliArgs(argv) {
  const [command, ...rest] = argv;
  const args = {};
  for (let i = 0; i < rest.length; i++) {
    const t = rest[i];
    if (!t.startsWith('--')) throw new Error(`Unexpected arg: ${t}`);
    const k = t.slice(2);
    const next = rest[i + 1];
    if (!next || next.startsWith('--')) args[k] = true;
    else {
      args[k] = next;
      i++;
    }
  }
  return { command, args };
}

function isBenignMediaPipeCppError(text) {
  const cleaned = text.trim().replace(/^\[MediaPipe Error\]:\s*/, '');
  return /^(INFO|WARNING|VERBOSE):|^[IW]\d{4}\s/.test(cleaned);
}

function httpGetStatus(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      res.resume();
      resolve(res.statusCode ?? 0);
    });
    req.on('error', () => resolve(0));
  });
}

async function startPreviewServer(port) {
  const viteBin = path.resolve('node_modules', '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite');
  const logs = [];
  const proc = spawn(viteBin, ['preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });
  proc.stdout.on('data', (d) => logs.push(d.toString('utf8')));
  proc.stderr.on('data', (d) => logs.push(d.toString('utf8')));

  const baseUrl = `http://127.0.0.1:${port}${BASE_PATH}`;
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const status = await httpGetStatus(baseUrl);
    if (status >= 200 && status < 400) {
      return {
        origin: `http://127.0.0.1:${port}`,
        baseUrl,
        logs,
        async stop() {
          if (!proc.pid || proc.exitCode !== null) return;
          const exited = new Promise((resolve) => proc.once('exit', resolve));
          try {
            if (process.platform !== 'win32') process.kill(-proc.pid, 'SIGTERM');
            else proc.kill('SIGTERM');
          } catch {}
          const timer = setTimeout(() => {
            try {
              if (process.platform !== 'win32') process.kill(-proc.pid, 'SIGKILL');
              else proc.kill('SIGKILL');
            } catch {}
          }, 2000);
          await exited;
          clearTimeout(timer);
        },
      };
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Timed out starting vite preview at ${baseUrl}:\n${logs.join('')}`);
}

/**
 * RELEASE_VALIDATION_GPU=hardware runs the GPU delegate on the machine's real
 * GPU. The default, `software`, forces SwiftShader so results are comparable
 * with CI runners, which have no GPU.
 */
function useHardwareGpu() {
  return (process.env.RELEASE_VALIDATION_GPU || 'software').toLowerCase() === 'hardware';
}

function buildChromeLaunchArgs(extra = []) {
  return [
    ...(useHardwareGpu() ? [] : ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
    '--ignore-gpu-blocklist',
    '--enable-unsafe-webgpu',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    '--no-sandbox',
    ...extra,
  ];
}

async function probeChromeCapabilities(userDataDir) {
  const candidates = [
    { headless: true, args: ['--headless=new', ...buildChromeLaunchArgs()] },
    { headless: true, args: ['--headless=new', ...buildChromeLaunchArgs(['--enable-features=Vulkan'])] },
  ];
  if (process.platform === 'linux' && process.env.DISPLAY) {
    candidates.push({
      headless: false,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-unsafe-swiftshader',
        '--ignore-gpu-blocklist',
        '--enable-features=Vulkan',
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        '--no-sandbox',
      ],
    });
  }

  let bestProbe = null;
  let bestConfig = candidates[0];

  // WebGPU is only exposed to secure contexts, so the probe page is served from
  // localhost; on about:blank `navigator.gpu` is always missing.
  const probeServer = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><title>capability probe</title>');
  });
  await new Promise((resolve) => probeServer.listen(0, '127.0.0.1', resolve));
  const probeUrl = `http://127.0.0.1:${probeServer.address().port}/`;

  for (const candidate of candidates) {
    let context;
    try {
      context = await chromium.launchPersistentContext(userDataDir, {
        channel: 'chrome',
        headless: candidate.headless,
        args: candidate.args,
        permissions: ['camera', 'microphone'],
      });
      const page = context.pages()[0] || (await context.newPage());
      await page.goto(probeUrl);
      const info = await page.evaluate(async () => {
        const canvas = document.createElement('canvas');
        const gl2 = canvas.getContext('webgl2');
        let webglRenderer = 'unavailable';
        if (gl2) {
          const ext = gl2.getExtension('WEBGL_debug_renderer_info');
          webglRenderer = ext ? String(gl2.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl2.getParameter(gl2.RENDERER));
        }
        let webgpuAdapter = 'unavailable';
        try {
          if (navigator.gpu) {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
              const info = adapter.info || {};
              webgpuAdapter = [info.vendor, info.architecture, info.description].filter(Boolean).join(' / ') || 'available';
              if (adapter.isFallbackAdapter ?? info.isFallbackAdapter) webgpuAdapter += ' (software)';
            }
          }
        } catch (err) {
          webgpuAdapter = `error: ${err?.message || err}`;
        }
        return {
          userAgent: navigator.userAgent,
          webglRenderer,
          webgpuAdapter,
          hasWebGL2: Boolean(gl2),
          hasWebGPU: webgpuAdapter !== 'unavailable' && !webgpuAdapter.startsWith('error:'),
        };
      });
      const versionMatch = info.userAgent.match(/Chrome\/([^\s]+)/);
      const probe = {
        browserVersion: versionMatch ? `Google Chrome ${versionMatch[1]}` : info.userAgent,
        webglRenderer: info.webglRenderer,
        webgpuAdapter: info.webgpuAdapter,
        hasWebGL2: info.hasWebGL2,
        hasWebGPU: info.hasWebGPU,
        headless: candidate.headless,
        launchArgs: candidate.args,
      };
      if (!bestProbe || (probe.hasWebGPU && probe.hasWebGL2 && (!bestProbe.hasWebGPU || !bestProbe.hasWebGL2))) {
        bestProbe = probe;
        bestConfig = candidate;
      }
      if (probe.hasWebGPU && probe.hasWebGL2) break;
    } finally {
      await context?.close().catch(() => {});
    }
  }
  probeServer.close();

  return { ...(bestProbe || {}), headless: bestConfig.headless, launchArgs: bestConfig.args };
}

async function waitForCondition(fn, { timeoutMs = 60000, intervalMs = 250, description = 'condition' } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (err) {
      lastError = err;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(
    `Timed out after ${timeoutMs}ms waiting for ${description}${lastError ? `: ${lastError.message}` : ''}`
  );
}

async function readText(page, selector) {
  const loc = page.locator(selector).first();
  if ((await loc.count().catch(() => 0)) === 0) return '';
  return ((await loc.textContent({ timeout: 1000 }).catch(() => '')) || '').replace(/\s+/g, ' ').trim();
}

async function readDelegateValue(page, defaultVal) {
  const loc = page.locator('#delegate-select');
  if ((await loc.count().catch(() => 0)) === 0) return defaultVal;
  return (await loc.inputValue({ timeout: 500 }).catch(() => defaultVal)) || defaultVal;
}

async function readStatusError(page) {
  const status = await readText(page, '#status-message');
  if (/^(Error|Failed|File Error)\b/i.test(status)) {
    return status;
  }
  const timerStatus = await readText(page, '#inference-timer-status, .dt-pending-title');
  if (/^(Error|Failed)\b/i.test(timerStatus)) {
    return timerStatus;
  }
  return '';
}

/**
 * Result hooks: the app publishes a summary of each result on
 * `window.__mpTestResults` (src/components/test-hooks.ts). The runner only
 * reads them; every inference is started through the UI.
 */
async function readHookSeq(page) {
  return page.evaluate(() => window.__mpTestResults?.seq ?? 0);
}

/**
 * Waits for a result of `task` that is newer than `afterSeq` and satisfies
 * `matches`, so a result left over from an earlier action is never accepted.
 */
async function waitForHookResult(page, task, { afterSeq = 0, matches = () => true, timeoutMs = 60000, description } = {}) {
  return waitForCondition(
    async () => {
      const record = await page.evaluate((t) => window.__mpTestResults?.results?.[t] ?? null, task);
      if (!record || record.seq <= afterSeq) return false;
      if (record.error) return record;
      return matches(record.payload || {}) ? record : false;
    },
    { timeoutMs, description: description || `a ${task} result from the app` }
  );
}

/**
 * Types each input into the task's text box, submits it with the task's button
 * and collects the result and the rows the page shows for it. Used by the
 * tasks that label a piece of text.
 */
async function runTextInputs(page, testCase, { inputs, button, results, matchesModel, timeoutMs, captureStep }) {
  const btn = page.locator(button);
  const runs = [];
  for (const input of inputs) {
    await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: `${button} enabled` });
    const seqBefore = await readHookSeq(page);
    await page.fill('#text-input', input.text);
    await btn.click();
    const hook = await waitForHookResult(page, testCase.task, {
      afterSeq: seqBefore,
      matches: (payload) => payload.text === input.text && matchesModel(payload),
      timeoutMs,
      description: `a ${testCase.task} result for "${input.text.slice(0, 30)}..."`,
    });
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const count = await page.locator(`${results} .classification-item`).count();
        const status = await readText(page, '#status-message');
        return count > 0 && /Done/i.test(status);
      },
      { timeoutMs, description: `${testCase.task} result shown` }
    );
    await captureStep(`"${input.text}" (expected: ${input.label})`);
    runs.push({ text: input.text, hook, ui: await readClassificationRows(page, results) });
  }
  return { runs };
}

/** Every recent result of `task`, oldest first, for tasks that produce several per action. */
async function readHookHistory(page, task) {
  return page.evaluate((t) => window.__mpTestResults?.history?.[t] ?? [], task);
}

/** Like `waitForHookResult`, but returns the latest matching result in the task's history. */
async function waitForHookInHistory(page, task, { afterSeq = 0, matches = () => true, timeoutMs = 60000, description } = {}) {
  return waitForCondition(
    async () => {
      const history = await readHookHistory(page, task);
      const found = history.filter((record) => record.seq > afterSeq && (record.error || matches(record.payload || {})));
      return found.length > 0 ? found[found.length - 1] : false;
    },
    { timeoutMs, description: description || `a ${task} result from the app` }
  );
}

/** Label and score text of each visible classification row, top first. */
async function readClassificationRows(page, containerSelector) {
  return page.evaluate((selector) => {
    return Array.from(document.querySelectorAll(`${selector} .classification-item`)).map((row) => ({
      label: row.querySelector('.class-name')?.textContent?.trim() ?? '',
      score: row.querySelector('.class-score')?.textContent?.trim() ?? '',
    }));
  }, containerSelector);
}

const BRUSH_NAMES = { 1: 'positive', 2: 'negative', 3: 'lasso' };

/**
 * Draws a stroke with the mouse inside `box`. `gesture` is either
 * `{ line: [x1, y1, x2, y2] }` or `{ circle: [cx, cy, radius] }`, in fractions
 * of the box.
 */
async function drawGesture(page, box, gesture) {
  const at = (fx, fy) => [box.x + box.width * fx, box.y + box.height * fy];
  if (gesture.line) {
    const [x1, y1, x2, y2] = gesture.line;
    await page.mouse.move(...at(x1, y1));
    await page.mouse.down();
    await page.mouse.move(...at(x2, y2), { steps: 8 });
    await page.mouse.up();
  } else if (gesture.circle) {
    const [cx, cy, radius] = gesture.circle;
    const segments = 32;
    await page.mouse.move(...at(cx + radius, cy));
    await page.mouse.down();
    for (let i = 1; i <= segments; i++) {
      const angle = (2 * Math.PI * i) / segments;
      await page.mouse.move(...at(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)));
    }
    await page.mouse.up();
  } else {
    throw new Error(`Unknown gesture: ${JSON.stringify(gesture)}`);
  }
}

async function waitForViewportIdle(page, timeoutMs) {
  await waitForCondition(
    async () => {
      const errMsg = await readStatusError(page);
      if (errMsg) throw new Error(errMsg);
      const isLoading = await page
        .evaluate(() => document.querySelector('.viewport')?.classList.contains('loading-model') ?? false)
        .catch(() => false);
      const statusTxt = await readText(page, '#status-message');
      if (!isLoading && !/^(Downloading|Initializing|Loading|Setting image)\b/i.test(statusTxt)) {
        return true;
      }
      return false;
    },
    { timeoutMs, description: 'viewport idle and model settled' }
  );
}

async function selectDelegateAndLoadModel(page, targetDelegate, selectValue, timeoutMs) {
  const select = page.locator('#model-selector-container-standard-select');
  await select.waitFor({ state: 'visible', timeout: 20000 });

  const loadBtn = page.locator('#model-selector-container-standard-load-btn');
  const waitingForManualLoad = await loadBtn.isVisible().catch(() => false);

  // If the page auto-loads its default model on open, wait for that initial load
  // to finish first so BaseTask.initQueue never races two INITs for the same model.
  if (!waitingForManualLoad) {
    await waitForViewportIdle(page, timeoutMs);
  }

  const delegateSelect = page.locator('#delegate-select');
  if ((await delegateSelect.count()) > 0 && (await delegateSelect.isEnabled().catch(() => false))) {
    const currentDelegate = await delegateSelect.inputValue();
    if (currentDelegate !== targetDelegate) {
      await delegateSelect.selectOption(targetDelegate);
      if (!(await loadBtn.isVisible().catch(() => false))) {
        await waitForViewportIdle(page, timeoutMs);
      }
    }
  }

  const currentVal = await select.inputValue();
  if (currentVal !== selectValue) {
    await select.selectOption(selectValue);
  }

  if (await loadBtn.isVisible().catch(() => false)) {
    const enabled = await loadBtn.isEnabled().catch(() => false);
    if (enabled) {
      await loadBtn.click();
    }
  }

  // Wait until the chosen option reports loaded OR an error status appears.
  await waitForCondition(
    async () => {
      const errMsg = await readStatusError(page);
      if (errMsg) throw new Error(errMsg);

      const state = await select
        .evaluate((el, val) => {
          const opt = Array.from(el.options).find((o) => o.value === val);
          return opt ? opt.dataset.state || '' : '';
        }, selectValue)
        .catch(() => '');
      const statusTxt = await readText(page, '#status-message');
      const isLoading = await page
        .evaluate(() => document.querySelector('.viewport')?.classList.contains('loading-model') ?? false)
        .catch(() => false);
      if (state === 'loaded' && !isLoading && !/^(Downloading|Initializing|Loading|Setting image)\b/i.test(statusTxt)) {
        return true;
      }
      return false;
    },
    { timeoutMs, description: `model ${selectValue} (${targetDelegate}) to finish loading` }
  );
}

async function ensureVisionReadyForImage(page, timeoutMs) {
  await waitForCondition(
    async () => {
      const errMsg = await readStatusError(page);
      if (errMsg) throw new Error(errMsg);
      const status = await readText(page, '#status-message');
      const inf = await readText(page, '#inference-time');
      if (/^Done in /i.test(status) && inf && !inf.includes('- ms')) {
        return status;
      }
      return false;
    },
    { timeoutMs, description: 'vision image inference Done' }
  );
}

async function testVisionVideoMode(page) {
  await page.click('#view-mode-toggle button[data-value="video"]');
  const webcamBtn = page.locator('#webcamButton');
  await webcamBtn.waitFor({ state: 'visible', timeout: 10000 });
  await waitForCondition(async () => await webcamBtn.isEnabled(), { timeoutMs: 15000, description: 'webcam button enabled' });
  await webcamBtn.click();

  await waitForCondition(
    async () => {
      const errMsg = await readStatusError(page);
      if (errMsg) throw new Error(errMsg);
      const label = await readText(page, '#webcamButton');
      const status = await readText(page, '#status-message');
      return /Disable Webcam/i.test(label) && (/Done in|Webcam running/i.test(status));
    },
    { timeoutMs: 30000, description: 'webcam stream active & predicting' }
  );
  await page.waitForTimeout(700);
}

async function cleanupWebcamIfActive(page) {
  const label = await readText(page, '#webcamButton');
  if (/Disable Webcam/i.test(label)) {
    await page.click('#webcamButton').catch(() => {});
  }
}

async function runSingleCheck(page, baseUrl, testCase, { screenshotDir, timeoutScale = 1 }) {
  const loadTimeoutMs = (testCase.largeModel ? 600000 : 120000) * timeoutScale;
  const infTimeoutMs = (testCase.largeModel ? 120000 : 60000) * timeoutScale;

  // Reset previous workers / WebGL contexts cleanly while keeping CacheStorage.
  await page.goto('about:blank');
  await page.goto(`${baseUrl}#/home`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    window.localStorage.removeItem('mediapipe-running-mode');
    window.localStorage.removeItem('mediapipe-webcam-active');
  });

  // Open the task the way a user does: from its card on the home page. The
  // model and delegate are then chosen with the page controls.
  await page.locator(`a.cover-card[href="${testCase.route}"]`).first().click();
  await page.locator(`h2:has-text("${testCase.heading}")`).first().waitFor({ state: 'visible', timeout: 20000 });
  // The Dino Game lists text-only models only while its input is set to text.
  if (testCase.kind === 'dino_game') {
    await page.click(testCase.mode.startsWith('image') ? '#dm-mode-image' : '#dm-mode-text');
  }
  await selectDelegateAndLoadModel(page, testCase.delegate, testCase.modelSelectValue, loadTimeoutMs);

  let details = '';
  // What the task produced, for output verification. Stays null for tasks
  // that do not publish a result hook yet.
  let observation = null;
  const hasHook = TASKS_WITH_RESULT_HOOK.has(testCase.task);
  // The delegate the UI settled on (CPU after a GPU fallback).
  const loadedDelegate = await readDelegateValue(page, testCase.delegate);
  const fromLoadedModel = (payload) =>
    payload.model === testCase.modelSelectValue && payload.delegate === loadedDelegate;

  // Cases with several steps (several sentences, pairs or strokes) keep a
  // screenshot of the page after each step, for the review mode of the report.
  const stepShots = [];
  fs.mkdirSync(screenshotDir, { recursive: true });
  const captureStep = async (label) => {
    const rel = `screenshots/${testCase.id}__step${stepShots.length + 1}.png`;
    await page.screenshot({ path: path.join(path.dirname(screenshotDir), rel), fullPage: true }).catch(() => {});
    stepShots.push({ label, src: rel });
  };

  if (testCase.kind === 'vision') {
    await ensureVisionReadyForImage(page, infTimeoutMs);
    if (testCase.mode === 'video') {
      await testVisionVideoMode(page);
      details = 'webcam video inference active';
    } else {
      details = await readText(page, '#status-message');
    }
    if (hasHook) {
      const mode = testCase.mode === 'video' ? 'VIDEO' : 'IMAGE';
      observation = {
        hook: await waitForHookResult(page, testCase.task, {
          // For the webcam, only a frame processed after the camera has settled counts.
          afterSeq: mode === 'VIDEO' ? await readHookSeq(page) : 0,
          matches: (payload) => payload.mode === mode && fromLoadedModel(payload),
          timeoutMs: infTimeoutMs,
          description: `a ${mode} ${testCase.task} result from ${testCase.modelSelectValue} on ${loadedDelegate}`,
        }),
        // Rows of the results panel, for the tasks that have one.
        ui: await readClassificationRows(page, '#classification-results'),
      };
    }
  } else if (testCase.kind === 'image_segmenter') {
    await ensureVisionReadyForImage(page, infTimeoutMs);
    const plan = IMAGE_SEGMENTER_PLANS[testCase.modelId];
    const segmentation = (extra) =>
      waitForHookResult(page, testCase.task, {
        timeoutMs: infTimeoutMs,
        ...extra,
        matches: (payload) => fromLoadedModel(payload) && extra.matches(payload),
      });
    if (testCase.mode === 'video') {
      await testVisionVideoMode(page);
      details = 'webcam video segmentation';
      observation = {
        hook: await segmentation({
          afterSeq: await readHookSeq(page),
          matches: (payload) => payload.mode === 'VIDEO',
          description: 'a mask of a live webcam frame',
        }),
      };
    } else {
      // The page segments its built-in sample photo as soon as the model is ready.
      let hook = await segmentation({
        matches: (payload) => payload.mode === 'IMAGE' && payload.outputType === 'CATEGORY_MASK',
        description: 'a category mask of the built-in sample photo',
      });
      let imageId = hook.payload?.image;
      if (plan?.upload) {
        // Upload a photo through the page's image upload control.
        const seqBefore = await readHookSeq(page);
        await page.setInputFiles('#image-upload', path.resolve('public', plan.upload));
        hook = await segmentation({
          afterSeq: seqBefore,
          matches: (payload) => payload.mode === 'IMAGE' && payload.outputType === 'CATEGORY_MASK' && payload.image !== imageId,
          description: `a category mask of the uploaded ${plan.upload}`,
        });
        imageId = hook.payload?.image;
      }
      details = 'CATEGORY_MASK';
      if (testCase.mode === 'image-confidence') {
        const labels = hook.payload?.labels || [];
        const classIndex = labels.indexOf(plan?.confidence?.label);
        if (classIndex < 0) throw new Error(`Model has no "${plan?.confidence?.label}" class; labels: ${labels.join(', ')}`);
        const seqBefore = await readHookSeq(page);
        await page.selectOption('#output-type', 'CONFIDENCE_MASKS');
        if ((await page.locator('#class-select').inputValue()) !== String(classIndex)) {
          await page.selectOption('#class-select', String(classIndex));
        }
        hook = await segmentation({
          afterSeq: seqBefore,
          matches: (payload) =>
            payload.mode === 'IMAGE' &&
            payload.outputType === 'CONFIDENCE_MASKS' &&
            payload.confidenceClass === classIndex &&
            payload.image === imageId,
          description: `a confidence mask for "${plan.confidence.label}"`,
        });
        details = `CONFIDENCE_MASKS (${plan.confidence.label})`;
      }
      await ensureVisionReadyForImage(page, infTimeoutMs);
      observation = { hook };
    }
  } else if (testCase.kind === 'interactive_segmenter') {
    let target = page.locator('#output_canvas');
    if (testCase.mode === 'video-freeze') {
      await page.click('#view-mode-toggle button[data-value="video"]');
      await page.locator('#webcamButton').click();
      await waitForCondition(async () => await page.locator('#freezeButton').isEnabled(), {
        timeoutMs: 30000,
        description: 'freezeButton enabled',
      });
      // Let the camera deliver real frames before freezing one.
      await page.waitForTimeout(700);
      await page.locator('#freezeButton').click();
      await waitForCondition(
        async () => /Frozen!/i.test(await readText(page, '#status-message')),
        { timeoutMs: 15000, description: 'interactive segmenter frozen frame ready' }
      );
      target = page.locator('#webcam-overlay');
    }
    await target.waitFor({ state: 'visible', timeout: 10000 });
    await page.waitForTimeout(300);
    const box = await target.boundingBox();
    if (!box) throw new Error('Missing bounding box of the drawing surface');
    // Draw each planned stroke with the mouse and capture the mask it produces.
    const steps = [];
    for (const step of INTERACTIVE_SEGMENTER_PLANS[testCase.mode]) {
      await page.selectOption('#stroke-mode-select', String(step.brush));
      const seqBefore = await readHookSeq(page);
      await drawGesture(page, box, step.gesture);
      const hook = await waitForHookResult(page, testCase.task, {
        afterSeq: seqBefore,
        matches: (payload) => payload.strokeCount === steps.length + 1 && fromLoadedModel(payload),
        timeoutMs: infTimeoutMs,
        description: `a mask for stroke ${steps.length + 1} (brush ${step.brush})`,
      });
      await ensureVisionReadyForImage(page, infTimeoutMs);
      await captureStep(`After stroke ${steps.length + 1} (${BRUSH_NAMES[step.brush]} brush)`);
      steps.push({ brush: step.brush, hook });
    }
    observation = { steps };
    details = `${testCase.mode === 'video-freeze' ? 'frozen webcam frame, ' : ''}brush modes ${steps.map((st) => st.brush).join(' then ')}`;
  } else if (testCase.kind === 'image_embedder') {
    const runs = [];
    // Each pair is chosen with the page's sample buttons.
    for (const pair of IMAGE_EMBEDDER_PAIRS) {
      const [first, second] = pair.key;
      const seqBefore = await readHookSeq(page);
      await page.click(`.sample-btn[data-target="1"][data-src="${first}"]`);
      await page.click(`.sample-btn[data-target="2"][data-src="${second}"]`);
      const hook = await waitForHookResult(page, testCase.task, {
        afterSeq: seqBefore,
        matches: (payload) => fromLoadedModel(payload) && payload.images?.[0] === first && payload.images?.[1] === second,
        timeoutMs: infTimeoutMs,
        description: `the similarity of ${first} and ${second}`,
      });
      // The page shows the value of the latest comparison; wait for it to show this one.
      const shown = typeof hook.payload?.similarity === 'number' ? hook.payload.similarity.toFixed(4) : null;
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const status = await readText(page, '#status-message');
          return status === 'Done' && (shown === null || (await readText(page, '#similarity-value')) === shown);
        },
        { timeoutMs: 10000, description: `the page to show similarity ${shown}` }
      ).catch(() => {});
      await captureStep(`${pair.name}: ${first} and ${second}`);
      runs.push({ key: pair.key, hook, ui: await readText(page, '#similarity-value') });
    }
    observation = { runs };
    details = runs.map((run) => `${run.key.join(' vs ')}: ${run.ui}`).join(' | ');
  } else if (testCase.kind === 'audio_classifier') {
    if (testCase.mode === 'file') {
      await page.click('#view-mode-toggle button[data-value="file"]');
      await page.setInputFiles('#audio-upload', SPEECH_FIXTURE);
      const runBtn = page.locator('#run-file-classification');
      await runBtn.waitFor({ state: 'visible', timeout: 10000 });
      const seqBefore = await readHookSeq(page);
      await runBtn.click();
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const status = await readText(page, '#status-message');
          return /^Done in /i.test(status);
        },
        { timeoutMs: infTimeoutMs, description: 'audio file classification Done' }
      );
      details = await readText(page, '#status-message');
      observation = {
        hook: await waitForHookResult(page, testCase.task, {
          afterSeq: seqBefore,
          matches: (payload) => payload.mode === 'AUDIO_CLIPS' && fromLoadedModel(payload),
          timeoutMs: infTimeoutMs,
        }),
        ui: await readClassificationRows(page, '#classification-results'),
      };
    } else {
      const recBtn = page.locator('#recordButton');
      await waitForCondition(async () => await recBtn.isEnabled(), { timeoutMs: 20000, description: 'recordButton enabled' });
      await recBtn.click();
      // Collect every live result, and what the page shows, while the clip plays.
      const hooks = [];
      const uiLabels = [];
      const recordUntil = Date.now() + 5000;
      while (Date.now() < recordUntil) {
        const record = await page.evaluate((t) => window.__mpTestResults?.results?.[t] ?? null, testCase.task);
        if (record && record.seq !== hooks[hooks.length - 1]?.seq && fromLoadedModel(record.payload || {})) hooks.push(record);
        const rows = await readClassificationRows(page, '#classification-results');
        if (rows[0]?.label) uiLabels.push(rows[0].label);
        await page.waitForTimeout(60);
      }
      const recordingStatus = await readText(page, '#status-message');
      details = `mic stream (${recordingStatus}), ${hooks.length} live results`;
      observation = { hooks, uiLabels: [...new Set(uiLabels)] };
      // The screenshot is taken at the end of the case; keep one of the live view too.
      await captureStep('While recording');
      await recBtn.click();
    }
  } else if (testCase.kind === 'audio_transcriber') {
    // The page shows the transcript as one row per segment.
    const readTranscript = () =>
      page.evaluate(() =>
        Array.from(document.querySelectorAll('#asr-transcript .asr-segment:not(.interim) .asr-segment-text'))
          .map((el) => (el.textContent || '').trim())
          .join(' ')
          .trim()
      );
    const seqBefore = await readHookSeq(page);
    if (testCase.mode === 'file') {
      await page.click('#view-mode-toggle button[data-value="file"]');
      await page.setInputFiles('#audio-upload', SPEECH_FIXTURE);
      const runBtn = page.locator('#run-file-transcription');
      await waitForCondition(async () => await runBtn.isEnabled(), {
        timeoutMs: 15000,
        description: 'run-file-transcription enabled',
      });
      await runBtn.click();
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const disabled = await runBtn.isDisabled();
          const status = await readText(page, '#status-message');
          return !disabled && !/Decoding|Transcribing/i.test(status);
        },
        { timeoutMs: infTimeoutMs, description: 'audio file transcription complete' }
      );
      details = await readText(page, '#status-message');
    } else {
      const recBtn = page.locator('#recordButton');
      await waitForCondition(async () => await recBtn.isEnabled(), { timeoutMs: 20000, description: 'recordButton enabled' });
      await recBtn.click();
      // Long enough for the fake microphone to play the whole 4.27 s clip.
      await page.waitForTimeout(6000);
      details = `mic active (${await readText(page, '#status-message')})`;
      await captureStep('While recording');
      await recBtn.click();
    }
    const streamMode = testCase.mode === 'file' ? 'AUDIO_CLIPS' : 'AUDIO_STREAM';
    const hook = await waitForHookResult(page, testCase.task, {
      afterSeq: seqBefore,
      matches: (payload) => payload.final === true && payload.mode === streamMode && fromLoadedModel(payload),
      timeoutMs: infTimeoutMs,
      description: 'the final transcript',
    });
    observation = { hook, ui: await readTranscript() };
    details = `${details} | "${observation.ui}"`;
  } else if (testCase.kind === 'text_to_speech') {
    const btnSelector = testCase.mode === 'stream' ? '#tts-stream-btn' : '#tts-speak-btn';
    const actionBtn = page.locator(btnSelector);
    await waitForCondition(async () => await actionBtn.isEnabled(), {
      timeoutMs: 20000,
      description: `${btnSelector} enabled`,
    });
    // The sentence is typed into the text box, replacing the page's sample text.
    await page.fill('#tts-text-input', TTS_SENTENCE);
    const seqBefore = await readHookSeq(page);
    await actionBtn.click();
    const hook = await waitForHookResult(page, testCase.task, {
      afterSeq: seqBefore,
      matches: (payload) => payload.text === TTS_SENTENCE && fromLoadedModel(payload),
      timeoutMs: infTimeoutMs * 3,
      description: `TTS ${testCase.mode} synthesis`,
    });
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const total = await readText(page, '#tts-metric-total');
        return total && total !== '-';
      },
      { timeoutMs: infTimeoutMs, description: `TTS ${testCase.mode} metrics shown` }
    );
    observation = {
      hook,
      ui: { total: await readText(page, '#tts-metric-total'), audio: await readText(page, '#tts-metric-audio') },
    };
    details = `total=${observation.ui.total}, audio=${observation.ui.audio}`;
  } else if (testCase.kind === 'text_classifier') {
    observation = await runTextInputs(page, testCase, {
      inputs: TEXT_CLASSIFIER_INPUTS,
      button: '#classify-btn',
      results: '#classification-results',
      matchesModel: fromLoadedModel,
      timeoutMs: infTimeoutMs,
      captureStep,
    });
    details = observation.runs.map((run) => `${run.ui[0]?.label} ${run.ui[0]?.score}`).join(' | ');
  } else if (testCase.kind === 'language_detector') {
    observation = await runTextInputs(page, testCase, {
      inputs: LANGUAGE_DETECTOR_INPUTS,
      button: '#detect-btn',
      results: '#detection-results',
      matchesModel: fromLoadedModel,
      timeoutMs: infTimeoutMs,
      captureStep,
    });
    details = observation.runs.map((run) => `${run.ui[0]?.label} ${run.ui[0]?.score}`).join(' | ');
  } else if (testCase.kind === 'text_embedder') {
    const btn = page.locator('#embed-btn');
    const runs = [];
    // Each pair of sentences is typed into the two text boxes and compared.
    for (const pair of TEXT_EMBEDDER_PAIRS) {
      await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: 'embed-btn enabled' });
      const seqBefore = await readHookSeq(page);
      await page.fill('#text-input-1', pair.key[0]);
      await page.fill('#text-input-2', pair.key[1]);
      await btn.click();
      const hook = await waitForHookResult(page, testCase.task, {
        afterSeq: seqBefore,
        matches: (payload) =>
          fromLoadedModel(payload) && payload.texts?.[0] === pair.key[0] && payload.texts?.[1] === pair.key[1],
        timeoutMs: infTimeoutMs,
        description: `the similarity of the ${pair.name}`,
      });
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const val = await readText(page, '#similarity-value');
          const status = await readText(page, '#status-message');
          return /Done/i.test(status) && val && val !== '--';
        },
        { timeoutMs: infTimeoutMs, description: 'text embedder similarity' }
      );
      await captureStep(`${pair.name}`);
      runs.push({ key: pair.key, hook, ui: await readText(page, '#similarity-value') });
    }
    observation = { runs };
    details = runs.map((run) => run.ui).join(' | ');
  } else if (testCase.kind === 'universal_embedder') {
    const plan = UNIVERSAL_EMBEDDER_PLANS[testCase.mode];
    const btn = page.locator('#embed-btn');
    const waitUntilReady = () =>
      waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          return await btn.isEnabled();
        },
        { timeoutMs: loadTimeoutMs, description: 'embed-btn enabled' }
      );
    await waitUntilReady();
    // Changing L2 normalization re-creates the embedder.
    if ((await page.locator('#l2-normalize-select').inputValue()) !== String(plan.l2Normalize)) {
      await page.selectOption('#l2-normalize-select', String(plan.l2Normalize));
      await page.waitForTimeout(500);
      await waitUntilReady();
    }
    const runs = [];
    for (const run of plan.runs) {
      // Set each slot to text or to a sample photo from its gallery.
      for (const [slot, input] of [
        ['a', run.key[0]],
        ['b', run.key[1]],
      ]) {
        await page.click(`#mode-${input.mode}-${slot}`);
        if (input.mode === 'text') await page.fill(`#input-text-${slot}`, input.value);
        else await page.click(`#gallery-strip-${slot} .sample-thumb-btn[title="${input.value.replace(/_/g, ' ')}"]`);
      }
      await waitUntilReady();
      const seqBefore = await readHookSeq(page);
      await btn.click();
      const hook = await waitForHookResult(page, testCase.task, {
        afterSeq: seqBefore,
        matches: (payload) =>
          fromLoadedModel(payload) &&
          payload.l2Normalize === plan.l2Normalize &&
          JSON.stringify(payload.inputs) === JSON.stringify(run.key),
        timeoutMs: infTimeoutMs,
        description: `the similarity of ${run.name}`,
      });
      const shown = typeof hook.payload?.similarity === 'number' ? hook.payload.similarity.toFixed(4) : null;
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const visible = await page.locator('#embedding-results').isVisible();
          return visible && (shown === null || (await readText(page, '#similarity-value')) === shown);
        },
        { timeoutMs: 10000, description: `the page to show similarity ${shown}` }
      ).catch(() => {});
      await captureStep(run.name);
      runs.push({ key: run.key, hook, ui: await readText(page, '#similarity-value') });
    }
    observation = { runs };
    details = runs.map((run) => run.ui).join(' | ');
  } else if (testCase.kind === 'semantic_retriever') {
    const btnIndex = page.locator('#btn-index');
    await waitForCondition(async () => await btnIndex.isEnabled(), { timeoutMs: 20000, description: 'btn-index enabled' });
    await btnIndex.click();
    const queryInput = page.locator('#query-input');
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        return await queryInput.isEnabled();
      },
      { timeoutMs: infTimeoutMs * 4, description: 'semantic indexing complete' }
    );
    const runs = [];
    for (const { query } of SEMANTIC_RETRIEVER_QUERIES) {
      await waitForCondition(async () => await page.locator('#btn-search').isEnabled(), {
        timeoutMs: infTimeoutMs,
        description: 'btn-search enabled',
      });
      const seqBefore = await readHookSeq(page);
      await queryInput.fill(query);
      await page.click('#btn-search');
      const hook = await waitForHookResult(page, testCase.task, {
        afterSeq: seqBefore,
        matches: (payload) => payload.query === query && fromLoadedModel(payload),
        timeoutMs: infTimeoutMs,
        description: `search results for "${query}"`,
      });
      await waitForCondition(async () => (await page.locator('#results-list .search-result-card').count()) > 0, {
        timeoutMs: 10000,
        description: 'search results shown',
      });
      const first = page.locator('#results-list .search-result-card').first();
      const ui = {
        title: ((await first.locator('.result-title').textContent()) || '').trim(),
        score: ((await first.locator('.result-score-label').textContent()) || '').trim(),
      };
      await captureStep(`Query "${query}"`);
      runs.push({ query, hook, ui });
    }
    observation = { runs };
    details = runs.map((run) => `${run.query} -> ${run.ui.title}`).join(' | ');
  } else if (testCase.kind === 'decision_maker') {
    const kind = getExpectation(testCase).kind;
    await page.click(`#dt-kind-toggle button[data-value="${testCase.mode}"]`);
    const evalBtn = page.locator('#dt-evaluate');
    const runs = [];
    // Each ticket is typed into the input box and evaluated with the Evaluate button.
    for (const ticket of DECISION_TICKETS) {
      await waitForCondition(async () => await evalBtn.isEnabled(), { timeoutMs: 20000, description: 'dt-evaluate enabled' });
      const seqBefore = await readHookSeq(page);
      await page.fill('#dt-input', ticket.text);
      await evalBtn.click();
      const hook = await waitForHookInHistory(page, 'decision_runtime', {
        afterSeq: seqBefore,
        matches: (payload) =>
          payload.kind === kind && payload.text === ticket.text && payload.imageByteLength === 0 && fromLoadedModel(payload),
        timeoutMs: infTimeoutMs,
        description: `a ${kind} decision for the ${ticket.name}`,
      });
      await waitForCondition(
        async () => {
          const errMsg = await readStatusError(page);
          if (errMsg) throw new Error(errMsg);
          const visible = await page.locator('#dt-result-body').isVisible();
          const status = await readText(page, '#status-message');
          return visible && /Done in /i.test(status);
        },
        { timeoutMs: infTimeoutMs, description: `decision_maker mode ${testCase.mode}` }
      );
      await captureStep(ticket.name);
      runs.push({
        ticket: ticket.name,
        hook,
        ui: { headline: await readText(page, '#dt-headline'), subtitle: await readText(page, '#dt-subtitle') },
      });
    }
    observation = { runs };
    details = runs.map((run) => `${run.ticket}: ${run.ui.headline} ${run.ui.subtitle}`.slice(0, 110)).join(' | ');
  } else if (testCase.kind === 'id_photo') {
    const plan = ID_PHOTO_PLANS[testCase.mode];
    const verdictSettled = async () => {
      const errMsg = await readStatusError(page);
      if (errMsg) throw new Error(errMsg);
      const verdict = await readText(page, '#dv-verdict-title');
      return Boolean(verdict) && !/EVALUATING|LOADING/i.test(verdict);
    };
    // The page checks its first sample photo as soon as the model is ready.
    await waitForCondition(verdictSettled, { timeoutMs: infTimeoutMs, description: 'the first photo to be checked' });
    await page.click(`#dv-pack-chips .dv-pack-chip[data-pack-id="${plan.pack}"]`);
    const photos = [];
    for (const photo of plan.photos) {
      const presetBtn = page.locator(`#dv-preset-strip .dv-preset-btn[data-preset-id="${photo.preset}"]`);
      await presetBtn.click();
      // A photo is evaluated once; selecting it again shows the stored evaluation.
      const hook = await waitForHookInHistory(page, testCase.task, {
        matches: (payload) => payload.photo === photo.preset && fromLoadedModel(payload),
        timeoutMs: infTimeoutMs,
        description: `the evaluation of ${photo.preset}`,
      });
      await waitForCondition(
        async () => (await verdictSettled()) && (await presetBtn.evaluate((el) => el.classList.contains('active'))),
        { timeoutMs: infTimeoutMs, description: `${photo.preset} to be shown with its verdict` }
      );
      const ui = await page.evaluate(() => ({
        verdict: (document.getElementById('dv-verdict-title')?.textContent || '').trim(),
        pack: document.querySelector('#dv-pack-chips .dv-pack-chip.active')?.getAttribute('data-pack-id') || '',
        rules: Array.from(document.querySelectorAll('#dv-rules-list .dv-rule-card')).map((card) => ({
          id: card.getAttribute('data-rule-id'),
          status: ['pass', 'fail', 'borderline', 'skipped'].find((c) => card.classList.contains(c)) || '',
        })),
      }));
      await captureStep(photo.name);
      photos.push({ preset: photo.preset, hook, ui });
    }
    observation = { photos };
    details = photos.map((p) => `${p.preset}: ${p.ui.verdict}`).join(' | ');
  } else if (testCase.kind === 'dino_game') {
    const versus = testCase.mode.endsWith('versus');
    await page.click(versus ? '#dm-playmode-versus' : '#dm-playmode-tap');
    const stepBtn = page.locator('#dm-step');
    await waitForCondition(async () => await stepBtn.isEnabled(), { timeoutMs: 20000, description: 'dm-step enabled' });
    const seqStart = await readHookSeq(page);
    // Decisions the loaded model made since the case started.
    const decisions = async () =>
      (await readHookHistory(page, testCase.task)).filter(
        (record) => record.seq > seqStart && record.payload?.probabilities && fromLoadedModel(record.payload)
      );
    const isRunning = async () => /Pause/i.test(await readText(page, '#dm-play'));
    // Advances the game one tick at a time with the Next Step button (or lets it
    // run, if the page has started it) until the dino has decided about a scene.
    const advanceUntilDecided = async (action) => {
      const { pattern, scene } = DINO_SCENE_ACTIONS.find((entry) => entry.action === action);
      const deadline = Date.now() + infTimeoutMs;
      while (Date.now() < deadline) {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        if ((await decisions()).some((record) => pattern.test(record.payload.scene))) return;
        if (await isRunning()) await page.waitForTimeout(100);
        else await stepBtn.click({ timeout: 5000 }).catch(() => {});
      }
      throw new Error(`The dino never decided about ${scene} within ${infTimeoutMs}ms`);
    };
    // Every game starts with cacti on the track.
    await advanceUntilDecided('JUMP');
    await captureStep('Cactus right ahead');
    // Tap mode offers a button that drops a bird (and starts the game); the race already has birds.
    if (!versus) await page.click('#dm-spawn-bird');
    await advanceUntilDecided('DUCK');
    await captureStep('Bird right ahead');
    if (await isRunning()) await page.click('#dm-play');
    // Once paused, the panel settles on the latest decision.
    const actionNames = DINO_SCENE_ACTIONS.map((entry) => entry.action);
    await waitForCondition(async () => actionNames.includes(await readText(page, '#dm-action')), {
      timeoutMs: 15000,
      description: 'the action panel to show a decision',
    }).catch(() => {});
    await page.waitForTimeout(300);
    const hooks = await decisions();
    observation = { hooks, ui: { action: await readText(page, '#dm-action') } };
    details = `${hooks.length} decisions: ${[...new Set(hooks.map((h) => h.payload.action))].join(', ')}`;
  }

  const fallbackWarning = testCase.delegate === 'GPU' ? await readText(page, '#fallback-warning') : '';
  const actualDelegateValue = await readDelegateValue(page, testCase.delegate);
  const isFallback = testCase.delegate === 'GPU' && (Boolean(fallbackWarning) || actualDelegateValue === 'CPU');
  const actualDelegate = isFallback ? 'CPU (fallback)' : actualDelegateValue || testCase.delegate;
  const inferenceTimeText = await readText(page, '#inference-time');

  fs.mkdirSync(screenshotDir, { recursive: true });
  const screenshotRel = `screenshots/${testCase.id}.png`;
  await page.screenshot({ path: path.join(path.dirname(screenshotDir), screenshotRel), fullPage: true });
  await cleanupWebcamIfActive(page);

  // Compare what the task produced with the ground truth for this case.
  const verification = verifyCase(getExpectation(testCase), observation, {
    model: testCase.modelSelectValue,
    delegate: loadedDelegate,
  });

  return {
    actualDelegate,
    status: isFallback ? 'FALLBACK_CPU' : 'PASS',
    inferenceTimeText,
    details: isFallback ? `${details} | ${fallbackWarning || 'GPU fell back to CPU'}` : details,
    screenshot: screenshotRel,
    verification,
    observation,
    stepShots,
  };
}

/**
 * The description and inputs of a case for the report's review mode. Input
 * files are copied into the report so it can be opened on its own.
 */
function buildReviewData(testCase, reportDir) {
  const { description, inputs } = describeCase(testCase);
  const inputDir = path.join(reportDir, 'inputs');
  return {
    description,
    inputs: inputs.map((input) => {
      if (!input.path && !input.bytes) return input;
      const name = input.name || path.basename(input.path);
      const target = path.join(inputDir, name);
      if (!fs.existsSync(target)) {
        fs.mkdirSync(inputDir, { recursive: true });
        if (input.bytes) fs.writeFileSync(target, input.bytes);
        else fs.copyFileSync(input.path, target);
      }
      return { kind: input.kind, label: input.label, src: `inputs/${name}` };
    }),
  };
}

/**
 * Final status of a case that ran to completion. PASS and FALLBACK_CPU are only
 * reported when the output was verified; a case with nothing asserted is
 * UNVERIFIED, never a pass.
 */
function resolveStatus(runStatus, verification) {
  if (verification.status === VERIFICATION.FAILED) return 'FAIL';
  if (verification.status === VERIFICATION.UNVERIFIED) return 'UNVERIFIED';
  return runStatus;
}

async function cmdInit(args) {
  const reportDir = path.resolve(args['report-dir'] || 'release-validation-report');
  const statusPath = path.join(reportDir, 'status.json');
  const userDataDir = path.resolve(args['user-data-dir'] || path.join(os.tmpdir(), 'mp-release-chrome-profile'));
  const taskFilter = process.env.TASK_FILTER || '';
  const includeLargeModels = (process.env.INCLUDE_LARGE_MODELS ?? 'true') !== 'false';

  const allCases = buildValidationMatrix({ taskFilter, includeLargeModels });
  if (allCases.length === 0) {
    throw new Error(`No validation cases matched filter="${taskFilter}" includeLargeModels=${includeLargeModels}`);
  }

  fs.mkdirSync(userDataDir, { recursive: true });
  const envProbe = await probeChromeCapabilities(userDataDir);

  const serverUrl = process.env.GITHUB_SERVER_URL || 'https://github.com';
  const repo = process.env.GITHUB_REPOSITORY || '';
  const runId = process.env.GITHUB_RUN_ID || '';
  const runUrl = repo && runId ? `${serverUrl}/${repo}/actions/runs/${runId}` : '';

  const state = {
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    runUrl,
    checkRunId: null,
    checkRunUrl: '',
    totalPlanned: allCases.length,
    currentCase: null,
    environment: {
      browserVersion: envProbe.browserVersion,
      webglRenderer: envProbe.webglRenderer,
      webgpuAdapter: envProbe.webgpuAdapter,
      hasWebGL2: envProbe.hasWebGL2,
      hasWebGPU: envProbe.hasWebGPU,
      headless: envProbe.headless ?? true,
      launchArgs: envProbe.launchArgs,
    },
    packageVersions: readInstalledMediaPipeVersions(),
    results: [],
  };

  await createLiveCheckRun(state);
  saveState(statusPath, state);

  console.log(`Initialized release validation for ${allCases.length} checks.`);
  console.log(`Browser: ${state.environment.browserVersion}`);
  console.log(`WebGL2 Renderer: ${state.environment.webglRenderer}`);
  console.log(`WebGPU Adapter: ${state.environment.webgpuAdapter}`);
  console.log(`Installed @mediapipe packages:`, state.packageVersions);
  if (state.checkRunUrl) {
    console.log(`Live progress monitor (Check Run): ${state.checkRunUrl}`);
  }
  if (state.runUrl) {
    console.log(`Workflow Run & Summary: ${state.runUrl}`);
  }

  appendStepSummary(
    [
      '## MediaPipe Web Comprehensive Release Validation',
      '',
      `- **Planned checks:** ${allCases.length} (large models: \`${includeLargeModels}\`, filter: \`${taskFilter || 'ALL'}\`)`,
      `- **Browser:** \`${state.environment.browserVersion}\``,
      `- **WebGL2:** \`${state.environment.webglRenderer}\``,
      `- **WebGPU:** \`${state.environment.webgpuAdapter}\``,
      state.checkRunUrl ? `- **Live Check Run:** ${state.checkRunUrl}` : '',
      `- **Installed Packages:** ${Object.entries(state.packageVersions).map(([k, v]) => `\`${k}@${v}\``).join(', ')}`,
    ]
      .filter(Boolean)
      .join('\n')
  );
}

/**
 * Creates the files Chrome's fake camera and microphone play for `cases`, and
 * returns the `{ video, audio }` file paths of each case by case id.
 */
async function prepareFakeMedia(cases, mediaDir) {
  const byCase = new Map();
  const videoFiles = new Map();
  let browser = null;
  try {
    for (const testCase of cases) {
      const media = getFakeMedia(testCase);
      let video = null;
      if (media.video) {
        const name = media.video.name;
        if (!videoFiles.has(name)) {
          // Photos are converted to a one-frame video in a throwaway browser page.
          browser = browser || (await chromium.launch({ channel: 'chrome', headless: true }));
          const converter = await browser.newPage();
          const bytes = media.video.bytes || fs.readFileSync(media.video.path);
          const outPath = path.join(mediaDir, `${path.parse(name).name}.y4m`);
          await writeFakeCameraVideo(converter, bytes, path.extname(name), outPath);
          await converter.close();
          videoFiles.set(name, outPath);
        }
        video = videoFiles.get(name);
      }
      byCase.set(testCase.id, { video, audio: media.audio });
    }
  } finally {
    await browser?.close().catch(() => {});
  }
  return byCase;
}

async function cmdRunDomain(args) {
  const domain = args.domain;
  if (!domain) throw new Error('--domain is required');

  const reportDir = path.resolve(args['report-dir'] || 'release-validation-report');
  const statusPath = path.join(reportDir, 'status.json');
  const screenshotDir = path.join(reportDir, 'screenshots');
  const userDataDir = path.resolve(args['user-data-dir'] || path.join(os.tmpdir(), 'mp-release-chrome-profile'));
  const port = Number(args.port || 4174);
  const taskFilter = process.env.TASK_FILTER || '';
  const includeLargeModels = (process.env.INCLUDE_LARGE_MODELS ?? 'true') !== 'false';

  const cases = buildValidationMatrix({ domain, taskFilter, includeLargeModels });
  if (cases.length === 0) {
    console.log(`No checks scheduled for domain "${domain}".`);
    return;
  }

  const state = loadState(statusPath);
  const preview = await startPreviewServer(port);
  const launchArgs = state.environment?.launchArgs || ['--headless=new', ...buildChromeLaunchArgs()];
  const headless = state.environment?.headless ?? true;
  const fakeMedia = await prepareFakeMedia(cases, `${userDataDir}-fake-media`);

  // The retrieval and decision tasks run on WebGPU. Without a real GPU Chrome
  // uses a software adapter, on which the embedding models are many times
  // slower, so those domains get longer timeouts. RELEASE_VALIDATION_TIMEOUT_SCALE
  // overrides the factor.
  const softwareWebGpu = /\(software\)/.test(state.environment?.webgpuAdapter || '');
  const timeoutScale =
    Number(process.env.RELEASE_VALIDATION_TIMEOUT_SCALE) ||
    (softwareWebGpu && (domain === 'retrieval' || domain === 'decision') ? 6 : 1);
  if (timeoutScale !== 1) console.log(`Timeouts for the ${domain} domain are scaled by ${timeoutScale}.`);

  // The fake camera and microphone files are Chrome launch flags, so the
  // browser is restarted whenever the next case needs different ones. The
  // profile is kept, so downloaded models stay cached.
  let context = null;
  let page = null;
  let currentMediaKey = null;
  // Chrome's fake camera and microphone do not reliably start a second time in
  // the same browser session ("Could not start video source"), so a case that
  // captures from them gets a fresh browser if the current one has captured before.
  let captureUsed = false;
  const usesCapture = (testCase) => /^(video|mic)/.test(testCase.mode);
  const openBrowserFor = async (media) => {
    captureUsed = false;
    await context?.close().catch(() => {});
    context = await chromium.launchPersistentContext(userDataDir, {
      channel: 'chrome',
      headless,
      args: [...launchArgs, ...fakeMediaLaunchArgs(media)],
      permissions: ['camera', 'microphone'],
      viewport: { width: 1440, height: 960 },
    });
    await context.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }));
    page = context.pages()[0] || (await context.newPage());
  };

  const domainResults = [];
  let lastCheckUpdateMs = 0;

  try {
    for (const testCase of cases) {
      const media = fakeMedia.get(testCase.id);
      const mediaKey = JSON.stringify(media);
      if (mediaKey !== currentMediaKey || (usesCapture(testCase) && captureUsed)) {
        await openBrowserFor(media);
        currentMediaKey = mediaKey;
      }
      if (usesCapture(testCase)) captureUsed = true;
      const startMs = Date.now();
      state.currentCase = testCase;
      saveState(statusPath, state);

      const pageErrors = [];
      const consoleErrors = [];
      const localNetErrors = [];

      const onPageError = (err) => pageErrors.push(err?.message || String(err));
      const onConsole = (msg) => {
        if (msg.type() === 'error' && !isBenignMediaPipeCppError(msg.text())) {
          consoleErrors.push(msg.text());
        }
      };
      const onReqFailed = (req) => {
        if (req.url().startsWith(preview.origin)) {
          localNetErrors.push(`REQUEST_FAILED ${req.url()}`);
        }
      };
      const onResponse = (res) => {
        if (res.url().startsWith(preview.origin) && res.status() >= 400) {
          localNetErrors.push(`HTTP_${res.status()} ${res.url()}`);
        }
      };

      page.on('pageerror', onPageError);
      page.on('console', onConsole);
      page.on('requestfailed', onReqFailed);
      page.on('response', onResponse);

      let resultRow;
      const review = buildReviewData(testCase, reportDir);
      try {
        const res = await runSingleCheck(page, preview.baseUrl, testCase, { screenshotDir, timeoutScale });
        if (pageErrors.length > 0 || consoleErrors.length > 0 || localNetErrors.length > 0) {
          const combined = [...pageErrors, ...consoleErrors, ...localNetErrors].join(' | ');
          throw new Error(`Runtime errors detected: ${combined}`);
        }
        const verification = res.verification;
        const passedChecks = verification.checks.filter((c) => c.pass).length;
        resultRow = {
          id: testCase.id,
          domain: testCase.domain,
          task: testCase.task,
          modelId: testCase.modelId,
          mode: testCase.mode,
          requestedDelegate: testCase.delegate,
          actualDelegate: res.actualDelegate,
          status: resolveStatus(res.status, verification),
          durationMs: Date.now() - startMs,
          inferenceTimeText: res.inferenceTimeText,
          details: res.details,
          error: verification.status === VERIFICATION.FAILED ? `Output verification failed: ${verification.reason}` : '',
          screenshot: res.screenshot,
          verification: {
            status: verification.status,
            summary:
              verification.status === VERIFICATION.UNVERIFIED
                ? verification.reason
                : `${passedChecks}/${verification.checks.length} output checks passed`,
            checks: verification.checks,
          },
          observation: res.observation,
          review: { ...review, steps: res.stepShots },
        };
      } catch (err) {
        fs.mkdirSync(screenshotDir, { recursive: true });
        const screenshotRel = `screenshots/${testCase.id}.png`;
        await page.screenshot({ path: path.join(reportDir, screenshotRel), fullPage: true }).catch(() => {});
        const extraErrs = [...pageErrors, ...consoleErrors, ...localNetErrors].slice(0, 3).join(' | ');
        resultRow = {
          id: testCase.id,
          domain: testCase.domain,
          task: testCase.task,
          modelId: testCase.modelId,
          mode: testCase.mode,
          requestedDelegate: testCase.delegate,
          actualDelegate: await readDelegateValue(page, 'error'),
          status: 'FAIL',
          durationMs: Date.now() - startMs,
          inferenceTimeText: await readText(page, '#inference-time'),
          details: await readText(page, '#status-message'),
          error: [err?.message || String(err), extraErrs].filter(Boolean).join(' | '),
          screenshot: screenshotRel,
          verification: { status: VERIFICATION.FAILED, summary: 'the case did not run to completion', checks: [] },
          review,
        };
      } finally {
        page.off('pageerror', onPageError);
        page.off('console', onConsole);
        page.off('requestfailed', onReqFailed);
        page.off('response', onResponse);
      }

      domainResults.push(resultRow);
      state.results.push(resultRow);
      state.currentCase = null;
      saveState(statusPath, state);

      const idx = state.results.length;
      const total = state.totalPlanned || idx;
      console.log(
        `[${idx}/${total}] ${resultRow.status} | ${resultRow.domain}/${resultRow.task} | model=${resultRow.modelId} | mode=${resultRow.mode} | req=${resultRow.requestedDelegate} actual=${resultRow.actualDelegate} | ${resultRow.durationMs}ms | ${resultRow.error || `${resultRow.verification.summary} | ${resultRow.details}`}`
      );

      if (Date.now() - lastCheckUpdateMs >= 4000 || resultRow.status === 'FAIL') {
        await updateLiveCheckRun(state);
        lastCheckUpdateMs = Date.now();
      }
    }
  } finally {
    await context?.close().catch(() => {});
    await preview.stop();
  }

  writeReportFiles(reportDir, state);
  await updateLiveCheckRun(state);
  appendStepSummary(`### Domain: \`${domain}\`\n\n${formatMarkdownTable(domainResults)}`);
}

async function cmdFinalize(args) {
  const reportDir = path.resolve(args['report-dir'] || 'release-validation-report');
  const statusPath = path.join(reportDir, 'status.json');
  const failOnFallback = (process.env.FAIL_ON_FALLBACK ?? 'false') === 'true';
  const failOnUnverified = (process.env.FAIL_ON_UNVERIFIED ?? 'false') === 'true';

  const state = loadState(statusPath);
  state.currentCase = null;
  saveState(statusPath, state);
  writeReportFiles(reportDir, state);

  const counts = summarizeCounts(state.results);
  const failed =
    counts.fail > 0 || (failOnFallback && counts.fallback > 0) || (failOnUnverified && counts.unverified > 0);

  await updateLiveCheckRun(state, {
    completed: true,
    conclusion: failed ? 'failure' : 'success',
  });

  appendStepSummary(
    [
      '## Final Release Validation Summary',
      '',
      `- **Total:** ${counts.total}`,
      `- **Passed (output verified):** ${counts.pass}`,
      `- **Fallback to CPU (output verified):** ${counts.fallback}`,
      `- **Failed:** ${counts.fail}`,
      `- **Unverified (ran, output not checked):** ${counts.unverified}`,
      `- **Skipped:** ${counts.skip}`,
      '',
      'Download the `release-validation-report` workflow artifact for `index.html`, `report.md`, `status.json`, and all screenshots.',
    ].join('\n')
  );

  console.log(
    `Finalized report: total=${counts.total}, pass=${counts.pass}, fallback=${counts.fallback}, fail=${counts.fail}, unverified=${counts.unverified}`
  );
  if (failed) {
    process.exitCode = 1;
  }
}

/**
 * Proves the output verifiers: each one must accept a correct output and
 * reject a wrong one, so a release that returns incorrect results cannot pass.
 */
function selfTestVerifiers(allCases) {
  const expectationFor = (task, mode) => {
    const testCase = allCases.find((c) => c.task === task && c.mode === mode);
    assert.ok(testCase, `No matrix case for ${task}/${mode}`);
    return getExpectation(testCase);
  };
  const hookOf = (payload) => ({ seq: 1, task: 'self-test', payload: { model: 'm', delegate: 'CPU', ...payload } });
  const context = { model: 'm', delegate: 'CPU' };
  const expectStatus = (label, expectation, observation, status, ctx = context) => {
    const result = verifyCase(expectation, observation, ctx);
    assert.equal(result.status, status, `${label}: expected ${status}, got ${result.status} (${result.reason})`);
    return result;
  };

  // Object detector on dog.jpg.
  const dog = expectationFor('object_detector', 'image');
  const detection = (label, score, box) =>
    hookOf({ mode: 'IMAGE', imageWidth: 640, imageHeight: 640, detections: [{ label, score, box }] });
  const dogBox = { x: 11, y: 88, width: 512, height: 556 };
  expectStatus('dog detected', dog, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.VERIFIED);
  expectStatus('wrong label', dog, { hook: detection('cat', 0.84, dogBox) }, VERIFICATION.FAILED);
  expectStatus('low score', dog, { hook: detection('dog', 0.2, dogBox) }, VERIFICATION.FAILED);
  expectStatus('box in the wrong place', dog, { hook: detection('dog', 0.84, { x: 500, y: 500, width: 100, height: 100 }) }, VERIFICATION.FAILED);
  expectStatus('no detections', dog, { hook: hookOf({ mode: 'IMAGE', imageWidth: 640, imageHeight: 640, detections: [] }) }, VERIFICATION.FAILED);
  expectStatus('no result published', dog, { hook: null }, VERIFICATION.FAILED);
  expectStatus('result from another model', dog, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.FAILED, { model: 'other', delegate: 'CPU' });
  expectStatus('result from another delegate', dog, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.FAILED, { model: 'm', delegate: 'GPU' });
  expectStatus('hook error', dog, { hook: { seq: 1, task: 'object_detector', error: 'boom' } }, VERIFICATION.FAILED);

  // Text classifier: a positive and a negative sentence.
  const sentiment = expectationFor('text_classifier', 'classify');
  const textRun = (input, topLabel, uiLabel = topLabel) => ({
    text: input.text,
    hook: hookOf({
      text: input.text,
      categories: [
        { label: topLabel, score: 0.9 },
        { label: topLabel === 'positive' ? 'negative' : 'positive', score: 0.1 },
      ],
    }),
    ui: [{ label: uiLabel, score: '90%' }],
  });
  const [positive, negative] = TEXT_CLASSIFIER_INPUTS;
  expectStatus('sentiment correct', sentiment, { runs: [textRun(positive, 'positive'), textRun(negative, 'negative')] }, VERIFICATION.VERIFIED);
  expectStatus('always positive', sentiment, { runs: [textRun(positive, 'positive'), textRun(negative, 'positive')] }, VERIFICATION.FAILED);
  expectStatus('UI shows another label', sentiment, { runs: [textRun(positive, 'positive', 'negative'), textRun(negative, 'negative')] }, VERIFICATION.FAILED);
  expectStatus('one input missing', sentiment, { runs: [textRun(positive, 'positive')] }, VERIFICATION.FAILED);
  const stale = textRun(negative, 'negative');
  stale.hook.payload.text = positive.text;
  expectStatus('result for another text', sentiment, { runs: [textRun(positive, 'positive'), stale] }, VERIFICATION.FAILED);

  // Audio classifier on the speech clip.
  const speech = expectationFor('audio_classifier', 'file');
  const audio = (labels, uiLabel = labels[0]) => ({
    hook: hookOf({
      mode: 'AUDIO_CLIPS',
      results: labels.map((label, i) => ({ timestampMs: i * 975, categories: label ? [{ label, score: 0.95 }] : [] })),
    }),
    ui: [{ label: uiLabel, score: '95%' }],
  });
  expectStatus('speech in every full window', speech, audio(['Speech', 'Speech', 'Speech', 'Speech', null]), VERIFICATION.VERIFIED);
  expectStatus('one window is music', speech, audio(['Speech', 'Music', 'Speech', 'Speech', null]), VERIFICATION.FAILED);
  expectStatus('too few windows', speech, audio(['Speech', 'Speech']), VERIFICATION.FAILED);
  expectStatus('UI shows another label', speech, audio(['Speech', 'Speech', 'Speech', 'Speech'], 'Music'), VERIFICATION.FAILED);

  // Interactive segmenter: positive stroke, then a negative stroke.
  const mask = (fraction, strokes, centroid = { x: 0.54, y: 0.6 }) => ({
    hook: hookOf({
      mode: 'IMAGE',
      hasMask: true,
      strokeCount: strokes.length,
      maskStats: {
        width: 640,
        height: 640,
        foregroundFraction: fraction,
        foregroundCentroid: fraction > 0 ? centroid : null,
        nonFiniteCount: 0,
        strokes: strokes.map(([brushMode, foregroundFraction]) => ({ brushMode, pointCount: 8, foregroundFraction })),
      },
    }),
  });
  const positiveStroke = expectationFor('interactive_segmenter', 'image-positive');
  expectStatus('cat selected', positiveStroke, { steps: [mask(0.67, [[1, 1]])] }, VERIFICATION.VERIFIED);
  expectStatus('empty mask', positiveStroke, { steps: [mask(0, [[1, 0]])] }, VERIFICATION.FAILED);
  expectStatus('whole image selected', positiveStroke, { steps: [mask(1, [[1, 1]])] }, VERIFICATION.FAILED);
  expectStatus('mask elsewhere', positiveStroke, { steps: [mask(0.67, [[1, 1]], { x: 0.1, y: 0.1 })] }, VERIFICATION.FAILED);
  expectStatus('stroke not on the mask', positiveStroke, { steps: [mask(0.67, [[1, 0.2]])] }, VERIFICATION.FAILED);
  const negativeStroke = expectationFor('interactive_segmenter', 'image-negative');
  expectStatus('negative stroke removes part', negativeStroke, { steps: [mask(0.67, [[1, 1]]), mask(0.43, [[1, 1], [2, 0]])] }, VERIFICATION.VERIFIED);
  expectStatus('negative stroke ignored', negativeStroke, { steps: [mask(0.67, [[1, 1]]), mask(0.67, [[1, 1], [2, 1]])] }, VERIFICATION.FAILED);
  const lasso = expectationFor('interactive_segmenter', 'image-lasso');
  expectStatus('lasso selects the cat', lasso, { steps: [mask(0.67, [[3, 0.58]])] }, VERIFICATION.VERIFIED);
  expectStatus('lasso selects nothing', lasso, { steps: [mask(0.002, [[3, 0]], { x: 0.35, y: 0.48 })] }, VERIFICATION.FAILED);

  // Face detector on face_model.png.
  const faceCase = expectationFor('face_detector', 'image');
  const face = (keypoints, box = { x: 224, y: 169, width: 215, height: 215 }, count = 1) =>
    hookOf({
      mode: 'IMAGE',
      imageWidth: 640,
      imageHeight: 640,
      detections: Array.from({ length: count }, () => ({ score: 0.95, box, keypoints })),
    });
  const faceKeypoints = [
    { x: 0.436, y: 0.36 },
    { x: 0.576, y: 0.355 },
    { x: 0.502, y: 0.44 },
  ];
  expectStatus('face found', faceCase, { hook: face(faceKeypoints) }, VERIFICATION.VERIFIED);
  expectStatus('no face', faceCase, { hook: face(faceKeypoints, undefined, 0) }, VERIFICATION.FAILED);
  expectStatus('two faces', faceCase, { hook: face(faceKeypoints, undefined, 2) }, VERIFICATION.FAILED);
  expectStatus('face mirrored', faceCase, { hook: face([faceKeypoints[1], faceKeypoints[0], faceKeypoints[2]]) }, VERIFICATION.FAILED);
  expectStatus('face box elsewhere', faceCase, { hook: face(faceKeypoints, { x: 10, y: 10, width: 215, height: 215 }) }, VERIFICATION.FAILED);
  expectStatus('face box too large', faceCase, { hook: face(faceKeypoints, { x: 0, y: 0, width: 640, height: 640 }) }, VERIFICATION.FAILED);

  // Landmark tasks.
  const group = (count, points, extra = {}) => ({
    count,
    nonFiniteCount: 0,
    box: { minX: 0.3, minY: 0.2, maxX: 0.7, maxY: 0.9 },
    points,
    ...extra,
  });
  const landmarkHook = (payload) => hookOf({ mode: 'IMAGE', imageWidth: 640, imageHeight: 640, ...payload });
  const posePoints = {
    nose: { x: 0.504, y: 0.212 },
    leftShoulder: { x: 0.573, y: 0.298 },
    rightShoulder: { x: 0.437, y: 0.296 },
    leftAnkle: { x: 0.527, y: 0.582 },
    rightAnkle: { x: 0.516, y: 0.853 },
  };
  const poseCase = expectationFor('pose_landmarker', 'image');
  expectStatus('pose found', poseCase, { hook: landmarkHook({ poses: [group(33, posePoints)] }) }, VERIFICATION.VERIFIED);
  expectStatus('no pose', poseCase, { hook: landmarkHook({ poses: [] }) }, VERIFICATION.FAILED);
  expectStatus('wrong landmark count', poseCase, { hook: landmarkHook({ poses: [group(17, posePoints)] }) }, VERIFICATION.FAILED);
  expectStatus('legs swapped', poseCase, { hook: landmarkHook({ poses: [group(33, { ...posePoints, leftAnkle: posePoints.rightAnkle, rightAnkle: posePoints.leftAnkle })] }) }, VERIFICATION.FAILED);
  expectStatus('pose shifted', poseCase, { hook: landmarkHook({ poses: [group(33, { ...posePoints, nose: { x: 0.8, y: 0.2 } })] }) }, VERIFICATION.FAILED);
  expectStatus('non-finite landmarks', poseCase, { hook: landmarkHook({ poses: [group(33, posePoints, { nonFiniteCount: 3 })] }) }, VERIFICATION.FAILED);

  const faceMeshCase = expectationFor('face_landmarker', 'image');
  const meshPoints = {
    noseTip: { x: 0.503, y: 0.443 },
    chin: { x: 0.512, y: 0.597 },
    rightEyeOuter: { x: 0.403, y: 0.359 },
    leftEyeOuter: { x: 0.6, y: 0.347 },
  };
  expectStatus('face mesh found', faceMeshCase, { hook: landmarkHook({ faces: [group(478, meshPoints)], blendshapeCounts: [52], matrixCount: 1 }) }, VERIFICATION.VERIFIED);
  expectStatus('blendshapes missing', faceMeshCase, { hook: landmarkHook({ faces: [group(478, meshPoints)], blendshapeCounts: [], matrixCount: 1 }) }, VERIFICATION.FAILED);
  expectStatus('face mesh upside down', faceMeshCase, { hook: landmarkHook({ faces: [group(478, { ...meshPoints, chin: { x: 0.5, y: 0.2 } })], blendshapeCounts: [52], matrixCount: 1 }) }, VERIFICATION.FAILED);

  const holisticCase = expectationFor('holistic_landmarker', 'image');
  const holistic = (overrides = {}) =>
    landmarkHook({
      pose: group(33, posePoints),
      face: group(478, { noseTip: { x: 0.505, y: 0.21 } }),
      leftHand: group(21, { wrist: { x: 0.509, y: 0.35 } }),
      rightHand: group(21, { wrist: { x: 0.491, y: 0.36 } }),
      ...overrides,
    });
  expectStatus('holistic found', holisticCase, { hook: holistic() }, VERIFICATION.VERIFIED);
  expectStatus('holistic hand missing', holisticCase, { hook: holistic({ leftHand: group(0, {}) }) }, VERIFICATION.FAILED);
  expectStatus('holistic face elsewhere', holisticCase, { hook: holistic({ face: group(478, { noseTip: { x: 0.5, y: 0.6 } }) }) }, VERIFICATION.FAILED);

  const gestureCase = expectationFor('gesture_recognizer', 'image');
  const gesture = (name, uiLabel = `Right: ${name}`) => ({
    hook: landmarkHook({
      hands: [group(21, { thumbTip: { x: 0.503, y: 0.182 } }, { gesture: name, score: 0.73, handednessDisplayName: 'Right' })],
    }),
    ui: [{ label: uiLabel, score: '73%' }],
  });
  expectStatus('thumbs up', gestureCase, gesture('Thumb_Up'), VERIFICATION.VERIFIED);
  expectStatus('wrong gesture', gestureCase, gesture('Victory'), VERIFICATION.FAILED);
  expectStatus('UI shows another gesture', gestureCase, gesture('Thumb_Up', 'Right: Victory'), VERIFICATION.FAILED);

  // Image classifier on dog.jpg.
  const breedCase = expectationFor('image_classifier', 'image');
  const classified = (label, score = 0.85, uiLabel = label) => ({
    hook: hookOf({ mode: 'IMAGE', categories: [{ label, score }] }),
    ui: [{ label: uiLabel, score: `${Math.round(score * 100)}%` }],
  });
  expectStatus('golden retriever', breedCase, classified('golden retriever'), VERIFICATION.VERIFIED);
  expectStatus('wrong breed', breedCase, classified('tabby'), VERIFICATION.FAILED);
  expectStatus('unsure', breedCase, classified('golden retriever', 0.1), VERIFICATION.FAILED);
  expectStatus('UI shows another label', breedCase, classified('golden retriever', 0.85, 'tabby'), VERIFICATION.FAILED);

  // Embedders.
  const similarityRuns = (pairs, values, field) =>
    pairs.map((pair, i) => ({
      key: pair.key,
      hook: hookOf({ [field]: pair.key, similarity: values[i], dimensions: [1024, 1024] }),
      ui: values[i].toFixed(4),
    }));
  const imagePairs = expectationFor('image_embedder', 'dual-image');
  expectStatus('image similarities', imagePairs, { runs: similarityRuns(IMAGE_EMBEDDER_PAIRS, [1, -0.05, 0.1], 'images') }, VERIFICATION.VERIFIED);
  expectStatus('everything identical', imagePairs, { runs: similarityRuns(IMAGE_EMBEDDER_PAIRS, [1, 1, 1], 'images') }, VERIFICATION.FAILED);
  expectStatus('same photo differs', imagePairs, { runs: similarityRuns(IMAGE_EMBEDDER_PAIRS, [0.5, -0.05, 0.1], 'images') }, VERIFICATION.FAILED);
  const wrongUi = similarityRuns(IMAGE_EMBEDDER_PAIRS, [1, -0.05, 0.1], 'images');
  wrongUi[1].ui = '0.9000';
  expectStatus('UI shows another similarity', imagePairs, { runs: wrongUi }, VERIFICATION.FAILED);
  const textPairs = expectationFor('text_embedder', 'embed');
  expectStatus('text similarities', textPairs, { runs: similarityRuns(TEXT_EMBEDDER_PAIRS, [1, 0.86, 0.52], 'texts') }, VERIFICATION.VERIFIED);
  expectStatus('unrelated scores higher', textPairs, { runs: similarityRuns(TEXT_EMBEDDER_PAIRS, [1, 0.5, 0.8], 'texts') }, VERIFICATION.FAILED);
  expectStatus('constant similarity', textPairs, { runs: similarityRuns(TEXT_EMBEDDER_PAIRS, [1, 1, 1], 'texts') }, VERIFICATION.FAILED);

  // Language detector.
  const languages = expectationFor('language_detector', 'detect');
  const languageRuns = (codes) =>
    LANGUAGE_DETECTOR_INPUTS.map((input, i) => ({
      text: input.text,
      hook: hookOf({ text: input.text, languages: [{ code: codes[i], probability: 0.99 }] }),
      ui: [{ label: codes[i], score: '99%' }],
    }));
  expectStatus('languages correct', languages, { runs: languageRuns(['fr', 'en', 'es', 'de', 'ja']) }, VERIFICATION.VERIFIED);
  expectStatus('always English', languages, { runs: languageRuns(['en', 'en', 'en', 'en', 'en']) }, VERIFICATION.FAILED);

  // Image segmenter.
  const segmented = (labels, categories, extra = {}) => ({
    hook: hookOf({
      mode: 'IMAGE',
      hasMask: true,
      outputType: 'CATEGORY_MASK',
      confidenceClass: -1,
      labels,
      maskStats: { width: 640, height: 640, categories, confidence: null },
      ...extra,
    }),
  });
  const deeplabLabels = Array.from({ length: 21 }, (_, i) => (i === 0 ? 'background' : i === 12 ? 'dog' : i === 8 ? 'cat' : `class${i}`));
  const dogMask = [
    { index: 0, fraction: 0.69, centroid: { x: 0.5, y: 0.44 } },
    { index: 12, fraction: 0.31, centroid: { x: 0.5, y: 0.62 } },
  ];
  const deeplabCategory = getExpectation(allCases.find((c) => c.task === 'image_segmenter' && c.modelId === 'deeplab_v3' && c.mode === 'image-category'));
  expectStatus('dog segmented', deeplabCategory, segmented(deeplabLabels, dogMask), VERIFICATION.VERIFIED);
  expectStatus('all background', deeplabCategory, segmented(deeplabLabels, [{ index: 0, fraction: 1, centroid: { x: 0.5, y: 0.5 } }]), VERIFICATION.FAILED);
  expectStatus('dog labelled cat', deeplabCategory, segmented(deeplabLabels, [dogMask[0], { ...dogMask[1], index: 8 }]), VERIFICATION.FAILED);
  expectStatus('dog in the wrong place', deeplabCategory, segmented(deeplabLabels, [dogMask[0], { ...dogMask[1], centroid: { x: 0.1, y: 0.1 } }]), VERIFICATION.FAILED);
  expectStatus('hard-coded statistics', deeplabCategory, { hook: hookOf({ mode: 'IMAGE', hasMask: true, outputType: 'CATEGORY_MASK', labels: deeplabLabels, maskStats: null }) }, VERIFICATION.FAILED);
  const deeplabConfidence = getExpectation(allCases.find((c) => c.task === 'image_segmenter' && c.modelId === 'deeplab_v3' && c.mode === 'image-confidence'));
  const confidence = (stats) =>
    segmented(deeplabLabels, dogMask, {
      outputType: 'CONFIDENCE_MASKS',
      confidenceClass: 12,
      maskStats: { width: 640, height: 640, categories: dogMask, confidence: { index: 12, nonFiniteCount: 0, min: 0, max: 1, mean: 0.306, aboveHalfFraction: 0.307, ...stats } },
    });
  expectStatus('dog confidence', deeplabConfidence, confidence({}), VERIFICATION.VERIFIED);
  expectStatus('flat confidence', deeplabConfidence, confidence({ max: 0.3, mean: 0.3, aboveHalfFraction: 0 }), VERIFICATION.FAILED);
  expectStatus('confidence out of range', deeplabConfidence, confidence({ max: 37 }), VERIFICATION.FAILED);
  expectStatus('confidence of another class', deeplabConfidence, segmented(deeplabLabels, dogMask, { outputType: 'CONFIDENCE_MASKS', confidenceClass: 0, maskStats: { categories: dogMask, confidence: { index: 0, nonFiniteCount: 0, min: 0, max: 1, mean: 0.69, aboveHalfFraction: 0.69 } } }), VERIFICATION.FAILED);
  const selfieCategory = getExpectation(allCases.find((c) => c.task === 'image_segmenter' && c.modelId === 'selfie_segmenter' && c.mode === 'image-category'));
  const personMask = [
    { index: 0, fraction: 0.58, centroid: { x: 0.5, y: 0.6 } },
    { index: 255, fraction: 0.42, centroid: { x: 0.49, y: 0.36 } },
  ];
  expectStatus('person segmented', selfieCategory, segmented(['selfie'], personMask), VERIFICATION.VERIFIED);
  expectStatus('no person', selfieCategory, segmented(['selfie'], [{ index: 255, fraction: 1, centroid: { x: 0.5, y: 0.5 } }]), VERIFICATION.FAILED);

  // Audio transcriber: the transcript is compared word by word with what is said.
  assert.equal(wordErrorRate('the quick brown fox', 'The quick, brown fox.'), 0);
  assert.equal(wordErrorRate('the quick brown fox', 'the quick brown box'), 0.25);
  assert.equal(wordRecall('the quick brown fox', 'uh quick brown fox the quick'), 0.75);
  const spoken = 'please call stella and ask her to bring these things with her from the store';
  const fileTranscript = { type: 'transcript', mode: 'AUDIO_CLIPS', reference: spoken, maxWordErrorRate: 0.2 };
  const heard = (transcript, extra = {}) => ({
    hook: hookOf({ mode: 'AUDIO_CLIPS', final: true, transcript, ...extra }),
    ui: transcript,
  });
  expectStatus('transcript correct', fileTranscript, heard('Please call Stella and ask her to bring these things with her from the store.'), VERIFICATION.VERIFIED);
  expectStatus('transcript of something else', fileTranscript, heard('The weather is nice today.'), VERIFICATION.FAILED);
  expectStatus('empty transcript', fileTranscript, heard(''), VERIFICATION.FAILED);
  expectStatus('transcript not final', fileTranscript, heard(spoken, { final: false }), VERIFICATION.FAILED);
  expectStatus('UI shows another transcript', fileTranscript, { ...heard(spoken), ui: 'something else' }, VERIFICATION.FAILED);
  const micTranscript = { type: 'transcript', mode: 'AUDIO_STREAM', reference: spoken, minWordRecall: 0.6 };
  const overheard = (transcript) => ({ hook: hookOf({ mode: 'AUDIO_STREAM', final: true, transcript }), ui: transcript });
  expectStatus('most of the clip heard', micTranscript, overheard('call Stella and ask her to bring these things with her from the store please call'), VERIFICATION.VERIFIED);
  expectStatus('little of the clip heard', micTranscript, overheard('please call'), VERIFICATION.FAILED);
  // The matrix cases use the transcript of the speech clip, or are unverified while none is set.
  const clipCase = expectationFor('audio_transcriber', 'file');
  if (SPEECH_TRANSCRIPT) {
    expectStatus('speech clip transcribed', clipCase, heard(SPEECH_TRANSCRIPT), VERIFICATION.VERIFIED);
    expectStatus('speech clip misheard', clipCase, heard(spoken), VERIFICATION.FAILED);
  } else {
    expectStatus('no reference transcript', clipCase, heard(spoken), VERIFICATION.UNVERIFIED);
  }

  // Text to speech: the audio must be a plausible rendering of the sentence.
  const speakCase = expectationFor('text_to_speech', 'speak');
  const spokenAudio = (overrides = {}) => {
    const payload = {
      mode: 'SYNTHESIZE',
      text: TTS_SENTENCE,
      cancelled: false,
      chunkCount: 1,
      sampleRateHz: 24000,
      durationSec: 4.5,
      sampleCount: 108000,
      nonFiniteCount: 0,
      rms: 0.08,
      peak: 0.7,
      ...overrides,
    };
    return { hook: hookOf(payload), ui: { audio: `${payload.durationSec.toFixed(2)} s`, total: '900 ms' } };
  };
  expectStatus('speech synthesized', speakCase, spokenAudio(), VERIFICATION.VERIFIED);
  expectStatus('silent audio', speakCase, spokenAudio({ rms: 0, peak: 0 }), VERIFICATION.FAILED);
  expectStatus('audio far too short', speakCase, spokenAudio({ durationSec: 0.2, sampleCount: 4800 }), VERIFICATION.FAILED);
  expectStatus('audio far too long', speakCase, spokenAudio({ durationSec: 60, sampleCount: 1440000 }), VERIFICATION.FAILED);
  expectStatus('invalid samples', speakCase, spokenAudio({ nonFiniteCount: 12 }), VERIFICATION.FAILED);
  expectStatus('clipped audio', speakCase, spokenAudio({ peak: 9 }), VERIFICATION.FAILED);
  expectStatus('audio for another sentence', speakCase, spokenAudio({ text: 'Hello.' }), VERIFICATION.FAILED);
  expectStatus('stream result for a speak case', speakCase, spokenAudio({ mode: 'STREAM' }), VERIFICATION.FAILED);
  expectStatus('UI shows another length', speakCase, { ...spokenAudio(), ui: { audio: '9.99 s' } }, VERIFICATION.FAILED);

  // Universal Embedder and Semantic Retriever.
  const crossModal = expectationFor('universal_embedder', 'text-image-l2true');
  const embedderRuns = (plan, values, extra = {}) =>
    plan.runs.map((run, i) => ({
      key: run.key,
      hook: hookOf({ delegate: 'GPU', inputs: run.key, similarity: values[i], dimensions: [768, 768], norms: [1, 1], l2Normalize: plan.l2Normalize, ...extra }),
      ui: values[i].toFixed(4),
    }));
  const gpu = { model: 'm', delegate: 'GPU' };
  const crossPlan = UNIVERSAL_EMBEDDER_PLANS['text-image-l2true'];
  expectStatus('descriptions match their photos', crossModal, { runs: embedderRuns(crossPlan, [0.69, 0.55, 0.73, 0.52]) }, VERIFICATION.VERIFIED, gpu);
  expectStatus('descriptions match the wrong photos', crossModal, { runs: embedderRuns(crossPlan, [0.5, 0.7, 0.5, 0.7]) }, VERIFICATION.FAILED, gpu);
  expectStatus('embeddings not normalized', crossModal, { runs: embedderRuns(crossPlan, [0.69, 0.55, 0.73, 0.52], { norms: [3.2, 1] }) }, VERIFICATION.FAILED, gpu);
  expectStatus('L2 setting ignored', crossModal, { runs: embedderRuns(crossPlan, [0.69, 0.55, 0.73, 0.52], { l2Normalize: false }) }, VERIFICATION.FAILED, gpu);

  const searchCase = expectationFor('semantic_retriever', 'index-and-search');
  const searches = (topIds, indexedCount = 10) =>
    SEMANTIC_RETRIEVER_QUERIES.map((q, i) => ({
      query: q.query,
      hook: hookOf({ delegate: 'GPU', query: q.query, indexedCount, results: [{ id: topIds[i], score: 0.6 }, { id: 'open_book', score: 0.4 }] }),
      ui: { title: SEMANTIC_RETRIEVER_QUERIES.find((x) => x.topId === topIds[i])?.topTitle ?? 'Other', score: 'Similarity 0.6000' },
    }));
  const rightIds = SEMANTIC_RETRIEVER_QUERIES.map((q) => q.topId);
  expectStatus('queries find their photos', searchCase, { runs: searches(rightIds) }, VERIFICATION.VERIFIED, gpu);
  expectStatus('a query finds the wrong photo', searchCase, { runs: searches([rightIds[1], ...rightIds.slice(1)]) }, VERIFICATION.FAILED, gpu);
  expectStatus('index incomplete', searchCase, { runs: searches(rightIds, 3) }, VERIFICATION.FAILED, gpu);

  // Decision Maker.
  const [outage, billing] = DECISION_TICKETS;
  const triage = (ticket, urgent, category, severity) => ({
    hook: hookOf({
      delegate: 'GPU',
      ok: true,
      kind: 'schema',
      text: ticket.text,
      imageByteLength: 0,
      result: { is_urgent: { label: String(urgent) }, category: { label: category }, severity: { expectedScore: severity } },
    }),
    ui: { headline: 'Triage', subtitle: `is_urgent: ${urgent ? 'TRUE' : 'FALSE'} (94%) · category: ${category} (94%) · severity: ${(severity + 1).toFixed(2)}/5` },
  });
  const schemaCase = expectationFor('decision_maker', 'polymorphic');
  expectStatus('tickets triaged', schemaCase, { runs: [triage(outage, true, 'bug', 3.96), triage(billing, false, 'billing', 2)] }, VERIFICATION.VERIFIED, gpu);
  expectStatus('outage not urgent', schemaCase, { runs: [triage(outage, false, 'bug', 3.96), triage(billing, false, 'billing', 2)] }, VERIFICATION.FAILED, gpu);
  expectStatus('wrong department', schemaCase, { runs: [triage(outage, true, 'billing', 3.96), triage(billing, false, 'billing', 2)] }, VERIFICATION.FAILED, gpu);
  expectStatus('same severity for both', schemaCase, { runs: [triage(outage, true, 'bug', 2), triage(billing, false, 'billing', 2)] }, VERIFICATION.FAILED, gpu);
  const yesNoCase = expectationFor('decision_maker', 'boolean');
  const yesNo = (ticket, value, headline = value ? 'Yes' : 'No') => ({
    hook: hookOf({ delegate: 'GPU', ok: true, kind: 'boolean', text: ticket.text, imageByteLength: 0, result: { value, probabilityTrue: value ? 0.94 : 0.02 } }),
    ui: { headline },
  });
  expectStatus('urgency answered', yesNoCase, { runs: [yesNo(outage, true), yesNo(billing, false)] }, VERIFICATION.VERIFIED, gpu);
  expectStatus('always yes', yesNoCase, { runs: [yesNo(outage, true), yesNo(billing, true)] }, VERIFICATION.FAILED, gpu);
  expectStatus('UI shows the opposite answer', yesNoCase, { runs: [yesNo(outage, true, 'No'), yesNo(billing, false)] }, VERIFICATION.FAILED, gpu);

  // ID Photo.
  const passportCase = expectationFor('id_photo', 'preset-passport');
  const passportPlan = ID_PHOTO_PLANS['preset-passport'];
  const checkedPhoto = (preset, failing = [], verdict = failing.length ? `${failing.length} ISSUE` : 'COMPLIANT') => ({
    preset,
    hook: hookOf({ delegate: 'GPU', photo: preset, rules: passportPlan.rules.map((id) => ({ id, passed: !failing.includes(id) })) }),
    ui: { verdict, pack: 'passport_id', rules: passportPlan.rules.map((id) => ({ id, status: failing.includes(id) ? 'fail' : 'pass' })) },
  });
  const [sunglasses, compliant] = passportPlan.photos;
  expectStatus('photos judged', passportCase, { photos: [checkedPhoto(sunglasses.preset, ['no_glasses']), checkedPhoto(compliant.preset)] }, VERIFICATION.VERIFIED, gpu);
  expectStatus('sunglasses accepted', passportCase, { photos: [checkedPhoto(sunglasses.preset), checkedPhoto(compliant.preset)] }, VERIFICATION.FAILED, gpu);
  expectStatus('compliant photo rejected', passportCase, { photos: [checkedPhoto(sunglasses.preset, ['no_glasses']), checkedPhoto(compliant.preset, ['eyes_open'])] }, VERIFICATION.FAILED, gpu);
  expectStatus('UI verdict disagrees', passportCase, { photos: [checkedPhoto(sunglasses.preset, ['no_glasses'], 'COMPLIANT'), checkedPhoto(compliant.preset)] }, VERIFICATION.FAILED, gpu);

  // Dino Game.
  const dinoCase = expectationFor('dino_game', 'text-tap');
  const dino = (pairs, uiAction = pairs[pairs.length - 1][1]) => ({
    hooks: pairs.map(([scene, action], i) => ({
      seq: i + 1,
      payload: { model: 'm', delegate: 'GPU', inputMode: 'text', scene, action, probabilities: {} },
    })),
    ui: { action: uiAction },
  });
  const clear = 'The path ahead is clear. No obstacles in sight.';
  const cactus = 'Ground obstacle right ahead: cactus.';
  const bird = 'Low flying obstacle right ahead: bird.';
  const far = 'Obstacle spotted in the distance: cactus. Hold a steady pace.';
  expectStatus('dino decisions fit', dinoCase, dino([[clear, 'SPRINT'], [far, 'WAIT'], [cactus, 'JUMP'], [bird, 'DUCK']]), VERIFICATION.VERIFIED, gpu);
  expectStatus('dino ducks under a cactus', dinoCase, dino([[clear, 'SPRINT'], [cactus, 'DUCK'], [bird, 'DUCK']]), VERIFICATION.FAILED, gpu);
  expectStatus('dino never met a bird', dinoCase, dino([[clear, 'SPRINT'], [cactus, 'JUMP']]), VERIFICATION.FAILED, gpu);
  expectStatus('UI shows another action', dinoCase, dino([[clear, 'SPRINT'], [cactus, 'JUMP'], [bird, 'DUCK']], 'JUMP'), VERIFICATION.FAILED, gpu);
  // With image input an obstacle is judged while it is still some way off, so holding the pace is
  // also right; the action for the other kind of obstacle is not.
  const dinoImageCase = expectationFor('dino_game', 'image-tap');
  const dinoImage = (pairs) => {
    const observation = dino(pairs);
    observation.hooks.forEach((h) => (h.payload.inputMode = 'image'));
    return observation;
  };
  expectStatus('image: waits, then jumps and ducks', dinoImageCase, dinoImage([[cactus, 'WAIT'], [cactus, 'JUMP'], [bird, 'DUCK']]), VERIFICATION.VERIFIED, gpu);
  expectStatus('image: ducks under a cactus', dinoImageCase, dinoImage([[cactus, 'DUCK'], [cactus, 'JUMP'], [bird, 'DUCK']]), VERIFICATION.FAILED, gpu);
  expectStatus('image: never jumps', dinoImageCase, dinoImage([[cactus, 'WAIT'], [bird, 'DUCK']]), VERIFICATION.FAILED, gpu);
  expectStatus('text: waiting at a cactus is wrong', dinoCase, dino([[cactus, 'WAIT'], [cactus, 'JUMP'], [bird, 'DUCK']]), VERIFICATION.FAILED, gpu);

  // A case without an expectation is reported as unverified, never as verified.
  expectStatus('no expectation', null, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.UNVERIFIED);
  expectStatus('explicitly unverified', { unverified: 'no known input' }, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.UNVERIFIED);

  // Webcam cases see the same photo through the fake camera, so the same truth applies to live frames.
  const dogOnWebcam = expectationFor('object_detector', 'video');
  const liveDetection = hookOf({ mode: 'VIDEO', imageWidth: 640, imageHeight: 640, detections: [{ label: 'dog', score: 0.84, box: dogBox }] });
  expectStatus('dog on the webcam', dogOnWebcam, { hook: liveDetection }, VERIFICATION.VERIFIED);
  expectStatus('image result for a webcam case', dogOnWebcam, { hook: detection('dog', 0.84, dogBox) }, VERIFICATION.FAILED);
  expectStatus('nothing on the webcam', dogOnWebcam, { hook: hookOf({ mode: 'VIDEO', imageWidth: 640, imageHeight: 640, detections: [] }) }, VERIFICATION.FAILED);

  // Microphone: the speech clip is played on a loop.
  const liveSpeech = expectationFor('audio_classifier', 'mic');
  const live = (labels, uiLabels = ['Speech']) => ({
    hooks: labels.map((label, i) => ({
      seq: i + 1,
      payload: { model: 'm', delegate: 'CPU', mode: 'AUDIO_STREAM', results: [{ categories: label ? [{ label, score: 0.9 }] : [] }] },
    })),
    uiLabels,
  });
  expectStatus('live speech', liveSpeech, live(['Speech', null, 'Speech', 'Speech', 'Speech', 'Speech', null, 'Speech']), VERIFICATION.VERIFIED);
  expectStatus('live results are music', liveSpeech, live(['Music', 'Music', 'Music', 'Music', 'Music', 'Music'], ['Music']), VERIFICATION.FAILED);
  expectStatus('too few live results', liveSpeech, live(['Speech', null, null]), VERIFICATION.FAILED);
  expectStatus('no live results', liveSpeech, live([]), VERIFICATION.FAILED);
  expectStatus('half the chunks mislabelled', liveSpeech, live(['Speech', 'Music', 'Speech', 'Music', 'Speech', 'Music', 'Speech', 'Speech']), VERIFICATION.FAILED);
  expectStatus('UI never shows the label', liveSpeech, live(['Speech', 'Speech', 'Speech', 'Speech', 'Speech'], []), VERIFICATION.FAILED);
  assert.equal(resolveStatus('PASS', { status: VERIFICATION.UNVERIFIED }), 'UNVERIFIED');
  assert.equal(resolveStatus('PASS', { status: VERIFICATION.FAILED }), 'FAIL');
  assert.equal(resolveStatus('FALLBACK_CPU', { status: VERIFICATION.VERIFIED }), 'FALLBACK_CPU');
  assert.equal(resolveStatus('PASS', { status: VERIFICATION.VERIFIED }), 'PASS');

  // Every task that publishes a result hook has an expectation entry for each of its cases.
  for (const testCase of allCases) {
    if (TASKS_WITH_RESULT_HOOK.has(testCase.task)) {
      assert.ok(getExpectation(testCase), `Missing expectation for ${testCase.id}`);
    }
  }
  assert.ok(fs.existsSync(SPEECH_FIXTURE), `Missing fixture ${SPEECH_FIXTURE}`);
}

function cmdSelfTest() {
  const mainTs = fs.readFileSync('src/main.ts', 'utf8');
  const mainRoutes = [...mainTs.matchAll(/'(\/[^']+)':\s*\{/g)].map((m) => m[1]).filter((r) => r !== '/home');
  const definedRoutes = new Set(TASK_DEFINITIONS.map((d) => d.route.replace(/^#/, '')));
  for (const route of mainRoutes) {
    assert.ok(definedRoutes.has(route), `Route ${route} in src/main.ts missing from TASK_DEFINITIONS`);
  }

  // Every task is opened from its card on the home page, so each route needs one.
  const coverPage = fs.readFileSync('src/ui/cover-page.ts', 'utf8');
  for (const def of TASK_DEFINITIONS) {
    assert.ok(coverPage.includes(`href="${def.route}"`), `No home-page card links to ${def.route}`);
  }

  const allCases = buildValidationMatrix({ includeLargeModels: true });
  const ids = new Set(allCases.map((c) => c.id));
  assert.equal(ids.size, allCases.length, 'Case IDs must be unique');
  assert.ok(allCases.length >= 150, `Expected comprehensive matrix, got ${allCases.length}`);

  selfTestVerifiers(allCases);

  const tmpReport = fs.mkdtempSync(path.join(os.tmpdir(), 'mp-report-test-'));
  try {
    writeReportFiles(tmpReport, {
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      environment: { browserVersion: 'Chrome Test', webglRenderer: 'SwiftShader', webgpuAdapter: 'Test' },
      packageVersions: { '@mediapipe/tasks-vision': '1.2.2' },
      results: [
        {
          id: 'vision__object_detector__efficientdet_lite0__image__CPU',
          domain: 'vision',
          task: 'object_detector',
          modelId: 'efficientdet_lite0',
          mode: 'image',
          requestedDelegate: 'CPU',
          actualDelegate: 'CPU',
          status: 'PASS',
          durationMs: 1200,
          inferenceTimeText: '12 ms',
          details: 'Done in 12ms',
          error: '',
          screenshot: 'screenshots/test.png',
        },
      ],
    });
    assert.ok(fs.existsSync(path.join(tmpReport, 'report.md')));
    assert.ok(fs.existsSync(path.join(tmpReport, 'index.html')));
  } finally {
    fs.rmSync(tmpReport, { recursive: true, force: true });
  }

  console.log(`release_validate/runner.mjs self-test passed (${allCases.length} matrix cases across ${mainRoutes.length} routes).`);
}

async function main() {
  const { command, args } = parseCliArgs(process.argv.slice(2));
  if (command === 'init') await cmdInit(args);
  else if (command === 'run-domain') await cmdRunDomain(args);
  else if (command === 'finalize') await cmdFinalize(args);
  else if (command === 'self-test') cmdSelfTest();
  else throw new Error(`Unknown command: ${command}`);
}

main().catch((err) => {
  console.error(`ERROR: ${err?.message || err}`);
  process.exit(1);
});
