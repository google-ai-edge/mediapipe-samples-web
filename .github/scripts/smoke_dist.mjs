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

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { chromium } from '@playwright/test';

const DEFAULT_PORT = 4173;
const BASE_PATH = '/mediapipe-samples-web/';

function parseCliArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i++;
    }
  }
  return args;
}

function isBenignMediaPipeCppError(text) {
  const cleaned = text.trim().replace(/^\[MediaPipe Error\]:\s*/, '');
  return /^(INFO|WARNING|VERBOSE):|^I0000\s|^W0000\s/.test(cleaned);
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

async function waitForServer(url, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const status = await httpGetStatus(url);
    if (status >= 200 && status < 400) {
      return;
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`Timed out waiting for preview server at ${url}`);
}

async function waitForLocatorText(locator, pattern, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  let lastText = '';
  while (Date.now() < deadline) {
    lastText = ((await locator.textContent().catch(() => '')) || '').trim();
    if (pattern.test(lastText)) {
      return lastText;
    }
    if (/^Error\b/i.test(lastText)) {
      throw new Error(`UI entered error state: ${lastText}`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`Timed out waiting for text matching ${pattern}; last text was ${JSON.stringify(lastText)}`);
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  const port = Number(args.port || DEFAULT_PORT);
  const failureDir = args['failure-dir'] ? path.resolve(args['failure-dir']) : path.resolve('test-results/smoke-dist');
  const distDir = path.resolve('dist');
  if (!fs.existsSync(path.join(distDir, 'index.html'))) {
    throw new Error(`Built dist/index.html not found at ${distDir}. Run pnpm run build first.`);
  }

  const origin = `http://127.0.0.1:${port}`;
  const baseUrl = `${origin}${BASE_PATH}`;

  const viteBin = path.resolve('node_modules', '.bin', process.platform === 'win32' ? 'vite.cmd' : 'vite');
  const previewLogs = [];
  const previewProc = spawn(
    viteBin,
    ['preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
    {
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: process.platform !== 'win32',
    }
  );
  previewProc.stdout.on('data', (d) => previewLogs.push(d.toString('utf8')));
  previewProc.stderr.on('data', (d) => previewLogs.push(d.toString('utf8')));

  async function stopPreviewServer() {
    if (!previewProc.pid || previewProc.exitCode !== null) return;
    const exited = new Promise((resolve) => previewProc.once('exit', resolve));
    try {
      if (process.platform !== 'win32') {
        process.kill(-previewProc.pid, 'SIGTERM');
      } else {
        previewProc.kill('SIGTERM');
      }
    } catch {
      // Process may already have exited.
    }
    const timer = setTimeout(() => {
      try {
        if (process.platform !== 'win32') {
          process.kill(-previewProc.pid, 'SIGKILL');
        } else {
          previewProc.kill('SIGKILL');
        }
      } catch {
        // Ignore if already terminated.
      }
    }, 2000);
    await exited;
    clearTimeout(timer);
  }

  let browser;
  let page;
  const pageErrors = [];
  const consoleErrors = [];
  const localNetworkFailures = [];

  try {
    await waitForServer(baseUrl);

    browser = await chromium.launch({
      headless: true,
      args: [
        '--headless=new',
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--ignore-gpu-blocklist',
      ],
    });
    const context = await browser.newContext();
    page = await context.newPage();

    page.on('pageerror', (err) => {
      pageErrors.push(err?.stack || err?.message || String(err));
    });
    page.on('console', (msg) => {
      if (msg.type() !== 'error') return;
      const text = msg.text();
      if (!isBenignMediaPipeCppError(text)) {
        consoleErrors.push(text);
      }
    });
    page.on('requestfailed', (req) => {
      const url = req.url();
      if (url.startsWith(origin)) {
        localNetworkFailures.push(`REQUEST_FAILED ${url}: ${req.failure()?.errorText || 'unknown'}`);
      }
    });
    page.on('response', (res) => {
      const url = res.url();
      if (url.startsWith(origin) && res.status() >= 400) {
        localNetworkFailures.push(`HTTP_${res.status()} ${url}`);
      }
    });

    const localModels = [
      ['**/efficientdet_lite0.tflite', path.resolve('tests/assets/efficientdet_lite0.tflite')],
      ['**/bert_classifier.tflite', path.resolve('tests/assets/bert_classifier.tflite')],
      ['**/yamnet.tflite', path.resolve('tests/assets/yamnet.tflite')],
    ];
    for (const [pattern, assetPath] of localModels) {
      if (fs.existsSync(assetPath)) {
        await page.route(pattern, (route) => route.fulfill({ path: assetPath }));
      }
    }

    console.log(`Running production dist/ smoke test against ${baseUrl}...`);

    // 1. Cover page
    await page.goto(`${baseUrl}#/home`, { waitUntil: 'domcontentloaded' });
    await page.locator('.cover-page').waitFor({ state: 'visible', timeout: 15000 });
    console.log('  [ok] #/home cover page rendered');

    // 2. Vision task + WASM worker init
    await page.goto(`${baseUrl}?delegate=CPU#/vision/object_detector`, { waitUntil: 'domcontentloaded' });
    await page.locator('h2:has-text("Object Detection")').waitFor({ state: 'visible', timeout: 15000 });
    await waitForLocatorText(page.locator('#status-message'), /(Model loaded\. Ready\.)|(Done)|(Ready)/, 60000);
    console.log('  [ok] #/vision/object_detector initialized with WASM');

    // 3. Text task + WASM worker init
    await page.goto(`${baseUrl}#/text/text_classifier`, { waitUntil: 'domcontentloaded' });
    await page.locator('h2:has-text("Text Classification")').waitFor({ state: 'visible', timeout: 15000 });
    await waitForLocatorText(page.locator('#classify-btn'), /^Classify$/, 60000);
    console.log('  [ok] #/text/text_classifier initialized with WASM');

    // 4. Audio task + WASM worker init
    await page.goto(`${baseUrl}#/audio/audio_classifier`, { waitUntil: 'domcontentloaded' });
    await page.locator('h2:has-text("Audio Classifier")').waitFor({ state: 'visible', timeout: 15000 });
    await waitForLocatorText(page.locator('#status-message'), /^Ready$/, 60000);
    console.log('  [ok] #/audio/audio_classifier initialized with WASM');

    // 5. Retrieval bundle route
    await page.goto(`${baseUrl}#/retrieval/universal_embedder`, { waitUntil: 'domcontentloaded' });
    await page.locator('h2:has-text("Universal Embedder")').waitFor({ state: 'visible', timeout: 15000 });
    console.log('  [ok] #/retrieval/universal_embedder rendered');

    // 6. Decision bundle route
    await page.goto(`${baseUrl}#/decision/decision_maker`, { waitUntil: 'domcontentloaded' });
    await page.locator('.output-header h2:has-text("Decision Maker")').waitFor({ state: 'visible', timeout: 15000 });
    console.log('  [ok] #/decision/decision_maker rendered');

    if (pageErrors.length > 0 || consoleErrors.length > 0 || localNetworkFailures.length > 0) {
      throw new Error(
        [
          pageErrors.length ? `Page errors:\n  - ${pageErrors.join('\n  - ')}` : '',
          consoleErrors.length ? `Console errors:\n  - ${consoleErrors.join('\n  - ')}` : '',
          localNetworkFailures.length ? `Local network failures:\n  - ${localNetworkFailures.join('\n  - ')}` : '',
        ]
          .filter(Boolean)
          .join('\n')
      );
    }

    console.log('Production dist/ headless smoke test passed cleanly.');
  } catch (err) {
    if (page) {
      fs.mkdirSync(failureDir, { recursive: true });
      const screenshotPath = path.join(failureDir, 'smoke-dist-failure.png');
      await page.screenshot({ path: screenshotPath, fullPage: true }).catch(() => {});
      console.error(`Saved failure screenshot to ${screenshotPath}`);
    }
    if (pageErrors.length) console.error('Recorded page errors:', pageErrors);
    if (consoleErrors.length) console.error('Recorded console errors:', consoleErrors);
    if (localNetworkFailures.length) console.error('Recorded local network failures:', localNetworkFailures);
    if (previewLogs.length) console.error('Preview server output:\n', previewLogs.join(''));
    throw err;
  } finally {
    await browser?.close().catch(() => {});
    await stopPreviewServer();
  }
}

main().catch((err) => {
  console.error(`ERROR: ${err?.message || err}`);
  process.exit(1);
});
