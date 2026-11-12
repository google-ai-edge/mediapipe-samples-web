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
import { TASK_DEFINITIONS, buildSampleWavBuffer, buildValidationMatrix } from './cases.mjs';
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

function buildChromeLaunchArgs(extra = []) {
  return [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
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
      await page.goto('about:blank');
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

async function runSingleCheck(page, baseUrl, testCase, { screenshotDir, wavPath }) {
  const loadTimeoutMs = testCase.largeModel ? 600000 : 120000;
  const infTimeoutMs = testCase.largeModel ? 120000 : 60000;

  // Reset previous workers / WebGL contexts cleanly while keeping CacheStorage.
  await page.goto('about:blank');
  await page.goto(`${baseUrl}#/home`, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => {
    window.localStorage.removeItem('mediapipe-running-mode');
    window.localStorage.removeItem('mediapipe-webcam-active');
  });

  const targetUrl = `${baseUrl}?delegate=${encodeURIComponent(testCase.delegate)}${testCase.route}`;
  await page.goto(targetUrl, { waitUntil: 'domcontentloaded' });
  await page.locator(`h2:has-text("${testCase.heading}")`).first().waitFor({ state: 'visible', timeout: 20000 });
  await selectDelegateAndLoadModel(page, testCase.delegate, testCase.modelSelectValue, loadTimeoutMs);

  let details = '';

  if (testCase.kind === 'vision') {
    await ensureVisionReadyForImage(page, infTimeoutMs);
    if (testCase.mode === 'video') {
      await testVisionVideoMode(page);
      details = 'webcam video inference active';
    } else {
      details = await readText(page, '#status-message');
    }
  } else if (testCase.kind === 'image_segmenter') {
    await ensureVisionReadyForImage(page, infTimeoutMs);
    if (testCase.mode === 'image-confidence') {
      await page.selectOption('#output-type', 'CONFIDENCE_MASKS');
      await ensureVisionReadyForImage(page, infTimeoutMs);
      details = 'CONFIDENCE_MASKS';
    } else if (testCase.mode === 'video') {
      await testVisionVideoMode(page);
      details = 'webcam video segmentation';
    } else {
      details = 'CATEGORY_MASK';
    }
  } else if (testCase.kind === 'interactive_segmenter') {
    if (testCase.mode === 'video-freeze') {
      await page.click('#view-mode-toggle button[data-value="video"]');
      await page.locator('#webcamButton').click();
      await waitForCondition(async () => await page.locator('#freezeButton').isEnabled(), {
        timeoutMs: 30000,
        description: 'freezeButton enabled',
      });
      await page.locator('#freezeButton').click();
      await waitForCondition(
        async () => /Frozen!/i.test(await readText(page, '#status-message')),
        { timeoutMs: 15000, description: 'interactive segmenter frozen frame ready' }
      );
      const overlay = page.locator('#webcam-overlay');
      await overlay.waitFor({ state: 'visible', timeout: 10000 });
      const box = await overlay.boundingBox();
      if (!box) throw new Error('Missing #webcam-overlay bounding box');
      await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.5);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.5, { steps: 6 });
      await page.mouse.up();
      await ensureVisionReadyForImage(page, infTimeoutMs);
      details = 'webcam freeze + stroke';
    } else {
      const brushMap = { 'image-positive': '1', 'image-negative': '2', 'image-lasso': '3' };
      await page.selectOption('#stroke-mode-select', brushMap[testCase.mode]);
      const canvas = page.locator('#output_canvas');
      await canvas.waitFor({ state: 'visible', timeout: 10000 });
      await page.waitForTimeout(300);
      const box = await canvas.boundingBox();
      if (!box) throw new Error('Missing #output_canvas bounding box');
      await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.5, { steps: 6 });
      await page.mouse.up();
      await ensureVisionReadyForImage(page, infTimeoutMs);
      details = `brush mode ${brushMap[testCase.mode]}`;
    }
  } else if (testCase.kind === 'image_embedder') {
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const sim = await readText(page, '#similarity-value');
        const status = await readText(page, '#status-message');
        return status === 'Done' && sim && sim !== '--';
      },
      { timeoutMs: infTimeoutMs, description: 'image embedder similarity' }
    );
    details = `similarity=${await readText(page, '#similarity-value')}`;
  } else if (testCase.kind === 'audio_classifier') {
    if (testCase.mode === 'file') {
      await page.click('#view-mode-toggle button[data-value="file"]');
      await page.setInputFiles('#audio-upload', wavPath);
      const runBtn = page.locator('#run-file-classification');
      await runBtn.waitFor({ state: 'visible', timeout: 10000 });
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
    } else {
      const recBtn = page.locator('#recordButton');
      await waitForCondition(async () => await recBtn.isEnabled(), { timeoutMs: 20000, description: 'recordButton enabled' });
      await recBtn.click();
      await page.waitForTimeout(1600);
      const recordingStatus = await readText(page, '#status-message');
      details = `mic stream (${recordingStatus})`;
      await recBtn.click();
    }
  } else if (testCase.kind === 'audio_transcriber') {
    if (testCase.mode === 'file') {
      await page.click('#view-mode-toggle button[data-value="file"]');
      await page.setInputFiles('#audio-upload', wavPath);
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
      await page.waitForTimeout(2000);
      details = `mic active (${await readText(page, '#status-message')})`;
      await recBtn.click();
      await page.waitForTimeout(1000);
    }
  } else if (testCase.kind === 'text_to_speech') {
    const btnSelector = testCase.mode === 'stream' ? '#tts-stream-btn' : '#tts-speak-btn';
    const actionBtn = page.locator(btnSelector);
    await waitForCondition(async () => await actionBtn.isEnabled(), {
      timeoutMs: 20000,
      description: `${btnSelector} enabled`,
    });
    await actionBtn.click();
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const total = await readText(page, '#tts-metric-total');
        const enabled = await actionBtn.isEnabled();
        return enabled && total && total !== '-';
      },
      { timeoutMs: infTimeoutMs, description: `TTS ${testCase.mode} synthesis` }
    );
    details = `total=${await readText(page, '#tts-metric-total')}, audio=${await readText(page, '#tts-metric-audio')}`;
  } else if (testCase.kind === 'text_classifier') {
    const btn = page.locator('#classify-btn');
    await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: 'classify-btn enabled' });
    await page.fill('#text-input', 'This MediaPipe release validation test is working wonderfully!');
    await btn.click();
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const count = await page.locator('#classification-results .classification-item').count();
        const status = await readText(page, '#status-message');
        return count > 0 && /Done/i.test(status);
      },
      { timeoutMs: infTimeoutMs, description: 'text classification result' }
    );
    details = await readText(page, '#classification-results .classification-item >> nth=0');
  } else if (testCase.kind === 'language_detector') {
    const btn = page.locator('#detect-btn');
    await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: 'detect-btn enabled' });
    await page.fill('#text-input', 'Bonjour, ceci est un test de validation automatique.');
    await btn.click();
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const count = await page.locator('#detection-results .classification-item').count();
        const status = await readText(page, '#status-message');
        return count > 0 && /Done/i.test(status);
      },
      { timeoutMs: infTimeoutMs, description: 'language detection result' }
    );
    details = await readText(page, '#detection-results .classification-item >> nth=0');
  } else if (testCase.kind === 'text_embedder') {
    const btn = page.locator('#embed-btn');
    await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: 'embed-btn enabled' });
    await page.fill('#text-input-1', 'I love watching sunny sunsets at the beach.');
    await page.fill('#text-input-2', 'Warm ocean evenings are wonderful.');
    await btn.click();
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
    details = `similarity=${await readText(page, '#similarity-value')}`;
  } else if (testCase.kind === 'universal_embedder') {
    if (testCase.mode === 'text-text-l2true') {
      await page.click('#mode-text-a');
      await page.click('#mode-text-b');
      await page.selectOption('#l2-normalize-select', 'true');
    } else if (testCase.mode === 'image-image-l2false') {
      await page.click('#mode-image-a');
      await page.click('#mode-image-b');
      await page.selectOption('#l2-normalize-select', 'false');
    } else {
      await page.click('#mode-text-a');
      await page.click('#mode-image-b');
      await page.selectOption('#l2-normalize-select', 'true');
    }
    const btn = page.locator('#embed-btn');
    await waitForCondition(async () => await btn.isEnabled(), { timeoutMs: 20000, description: 'embed-btn enabled' });
    await btn.click();
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const visible = await page.locator('#embedding-results').isVisible();
        const sim = await readText(page, '#similarity-value');
        return visible && sim && sim !== '-';
      },
      { timeoutMs: infTimeoutMs, description: 'universal embedder similarity' }
    );
    details = `similarity=${await readText(page, '#similarity-value')}`;
  } else if (testCase.kind === 'semantic_retriever') {
    const btnIndex = page.locator('#btn-index');
    await waitForCondition(async () => await btnIndex.isEnabled(), { timeoutMs: 20000, description: 'btn-index enabled' });
    await btnIndex.click();
    const queryInput = page.locator('#query-input');
    await waitForCondition(async () => await queryInput.isEnabled(), {
      timeoutMs: infTimeoutMs * 2,
      description: 'semantic indexing complete',
    });
    await queryInput.fill('cute cat');
    await page.click('#btn-search');
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        return (await page.locator('#results-list .result-item').count()) > 0;
      },
      { timeoutMs: infTimeoutMs, description: 'semantic search results' }
    );
    details = `top=${await readText(page, '#results-list .result-title >> nth=0')}`;
  } else if (testCase.kind === 'decision_maker') {
    await page.click(`#dt-kind-toggle button[data-value="${testCase.mode}"]`);
    const evalBtn = page.locator('#dt-evaluate');
    await waitForCondition(async () => await evalBtn.isEnabled(), { timeoutMs: 20000, description: 'dt-evaluate enabled' });
    await evalBtn.click();
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
    details = await readText(page, '#dt-headline');
  } else if (testCase.kind === 'id_photo') {
    if (testCase.mode === 'rule-pack-badge') {
      await page.click('#dv-pack-chips button[data-pack-id="badge"]');
    }
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const status = await readText(page, '#status-message');
        const readiness = await readText(page, '#dv-readiness-label');
        return /in \d+ms/i.test(status) && readiness && !readiness.includes('--');
      },
      { timeoutMs: infTimeoutMs, description: `id_photo mode ${testCase.mode}` }
    );
    details = await readText(page, '#dv-readiness-label');
  } else if (testCase.kind === 'dino_game') {
    if (testCase.mode === 'image-tap') {
      await page.click('#dm-mode-image');
      await page.click('#dm-playmode-tap');
    } else if (testCase.mode === 'text-versus') {
      await page.click('#dm-mode-text');
      await page.click('#dm-playmode-versus');
    } else {
      await page.click('#dm-mode-text');
      await page.click('#dm-playmode-tap');
    }
    const stepBtn = page.locator('#dm-step');
    await waitForCondition(async () => await stepBtn.isEnabled(), { timeoutMs: 20000, description: 'dm-step enabled' });
    await stepBtn.click();
    await waitForCondition(
      async () => {
        const errMsg = await readStatusError(page);
        if (errMsg) throw new Error(errMsg);
        const inf = await readText(page, '#inference-time');
        return inf && !inf.includes('- ms');
      },
      { timeoutMs: infTimeoutMs, description: `dino_game mode ${testCase.mode}` }
    );
    details = await readText(page, '#inference-time');
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

  return {
    actualDelegate,
    status: isFallback ? 'FALLBACK_CPU' : 'PASS',
    inferenceTimeText,
    details: isFallback ? `${details} | ${fallbackWarning || 'GPU fell back to CPU'}` : details,
    screenshot: screenshotRel,
  };
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
  const wavPath = path.join(reportDir, 'sample-16k.wav');
  if (!fs.existsSync(wavPath)) {
    fs.mkdirSync(reportDir, { recursive: true });
    fs.writeFileSync(wavPath, buildSampleWavBuffer());
  }

  const preview = await startPreviewServer(port);
  const launchArgs = state.environment?.launchArgs || ['--headless=new', ...buildChromeLaunchArgs()];
  const headless = state.environment?.headless ?? true;
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: 'chrome',
    headless,
    args: launchArgs,
    permissions: ['camera', 'microphone'],
    viewport: { width: 1440, height: 960 },
  });

  const domainResults = [];
  let lastCheckUpdateMs = 0;

  try {
    await context.route('**/favicon.ico', (route) => route.fulfill({ status: 204, body: '' }));
    const page = context.pages()[0] || (await context.newPage());

    for (const testCase of cases) {
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
      try {
        const res = await runSingleCheck(page, preview.baseUrl, testCase, { screenshotDir, wavPath });
        if (pageErrors.length > 0 || consoleErrors.length > 0 || localNetErrors.length > 0) {
          const combined = [...pageErrors, ...consoleErrors, ...localNetErrors].join(' | ');
          throw new Error(`Runtime errors detected: ${combined}`);
        }
        resultRow = {
          id: testCase.id,
          domain: testCase.domain,
          task: testCase.task,
          modelId: testCase.modelId,
          mode: testCase.mode,
          requestedDelegate: testCase.delegate,
          actualDelegate: res.actualDelegate,
          status: res.status,
          durationMs: Date.now() - startMs,
          inferenceTimeText: res.inferenceTimeText,
          details: res.details,
          error: '',
          screenshot: res.screenshot,
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
        `[${idx}/${total}] ${resultRow.status} | ${resultRow.domain}/${resultRow.task} | model=${resultRow.modelId} | mode=${resultRow.mode} | req=${resultRow.requestedDelegate} actual=${resultRow.actualDelegate} | ${resultRow.durationMs}ms | ${resultRow.error || resultRow.details}`
      );

      if (Date.now() - lastCheckUpdateMs >= 4000 || resultRow.status === 'FAIL') {
        await updateLiveCheckRun(state);
        lastCheckUpdateMs = Date.now();
      }
    }
  } finally {
    await context.close().catch(() => {});
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

  const state = loadState(statusPath);
  state.currentCase = null;
  saveState(statusPath, state);
  writeReportFiles(reportDir, state);

  const counts = summarizeCounts(state.results);
  const failed = counts.fail > 0 || (failOnFallback && counts.fallback > 0);

  await updateLiveCheckRun(state, {
    completed: true,
    conclusion: failed ? 'failure' : 'success',
  });

  appendStepSummary(
    [
      '## Final Release Validation Summary',
      '',
      `- **Total:** ${counts.total}`,
      `- **Passed:** ${counts.pass}`,
      `- **Fallback to CPU:** ${counts.fallback}`,
      `- **Failed:** ${counts.fail}`,
      `- **Skipped:** ${counts.skip}`,
      '',
      'Download the `release-validation-report` workflow artifact for `index.html`, `report.md`, `status.json`, and all screenshots.',
    ].join('\n')
  );

  console.log(`Finalized report: total=${counts.total}, pass=${counts.pass}, fallback=${counts.fallback}, fail=${counts.fail}`);
  if (failed) {
    process.exitCode = 1;
  }
}

function cmdSelfTest() {
  const mainTs = fs.readFileSync('src/main.ts', 'utf8');
  const mainRoutes = [...mainTs.matchAll(/'(\/[^']+)':\s*\{/g)].map((m) => m[1]).filter((r) => r !== '/home');
  const definedRoutes = new Set(TASK_DEFINITIONS.map((d) => d.route.replace(/^#/, '')));
  for (const route of mainRoutes) {
    assert.ok(definedRoutes.has(route), `Route ${route} in src/main.ts missing from TASK_DEFINITIONS`);
  }

  const allCases = buildValidationMatrix({ includeLargeModels: true });
  const ids = new Set(allCases.map((c) => c.id));
  assert.equal(ids.size, allCases.length, 'Case IDs must be unique');
  assert.ok(allCases.length >= 150, `Expected comprehensive matrix, got ${allCases.length}`);

  const wav = buildSampleWavBuffer();
  assert.equal(wav.slice(0, 4).toString('ascii'), 'RIFF');
  assert.equal(wav.slice(8, 12).toString('ascii'), 'WAVE');

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
