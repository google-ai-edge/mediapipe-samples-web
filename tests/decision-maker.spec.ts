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

import { test, expect } from '@playwright/test';

/** Hosts the Decision Maker's built-in models live on. */
const MODEL_HOSTS = /huggingface\.co|storage\.googleapis\.com/;

test.describe('Decision Maker Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));
  });

  test('should not download a model until "Initialize Task" is pressed', async ({ page }) => {
    const modelRequests: string[] = [];
    page.on('request', (req) => {
      if (MODEL_HOSTS.test(req.url())) modelRequests.push(req.url());
    });

    await page.goto('/#/decision/decision_maker');
    await page.waitForLoadState('domcontentloaded');

    await expect(page.locator('.output-header h2')).toHaveText('Decision Maker');

    // Shared model selector in explicit-load mode
    const containerId = 'model-selector-container';
    await expect(page.locator(`#${containerId}-tab-standard`)).toBeVisible();
    const modelSelect = page.locator('.model-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue('embeddinggemma2_270m');
    // The text+vision model is the same file the retrieval demos use, so it can be reused from the cache.
    await expect(modelSelect.locator('option[value="embeddinggemma2_text_vision_440m"]')).toHaveText(
      /EmbeddingGemma-2 Text-Vision 440M/
    );
    const loadBtn = page.locator(`#${containerId}-standard-load-btn`);
    await expect(loadBtn).toBeVisible();
    await expect(loadBtn).toBeEnabled();
    await expect(loadBtn).toHaveText('Initialize Task');
    await expect(page.locator(`#${containerId}-standard-status`)).toHaveText(/Not loaded yet/);
    await expect(page.locator(`#${containerId}-model-badge`)).toBeHidden();

    // Nothing runs before a model is loaded
    await expect(page.locator('#status-message')).toHaveText('Load a model to begin');
    // The Dino Game (and its play/step controls) is its own page now; here the Evaluate button is the gate.
    await expect(page.locator('#dt-evaluate')).toBeDisabled();

    // Changing the delegate or the model selection must not start a download either
    await page.selectOption('#delegate-select', 'CPU');
    await page.selectOption('.model-select', 'laya_s256');
    await page.waitForTimeout(1500);
    expect(modelRequests).toEqual([]);
    await expect(page.locator('#status-message')).toHaveText('Load a model to begin');
  });

  test('should request the selected model only after "Initialize Task" is pressed', async ({ page }) => {
    // Serve a fake, tiny model so the test neither downloads nor needs a real model.
    const modelRequests: string[] = [];
    await page.route(
      (url) => MODEL_HOSTS.test(url.href),
      (route) => {
        modelRequests.push(route.request().url());
        route.fulfill({ status: 200, contentType: 'application/octet-stream', body: Buffer.alloc(16) });
      }
    );

    await page.goto('/#/decision/decision_maker');
    await page.waitForLoadState('domcontentloaded');
    await expect(page.locator('#status-message')).toHaveText('Load a model to begin');
    expect(modelRequests).toEqual([]);

    await page.click('#model-selector-container-standard-load-btn');

    await expect.poll(() => modelRequests.length, { timeout: 15000 }).toBeGreaterThan(0);
    expect(modelRequests[0]).toContain('embeddinggemma-2-text-270m.litertlm');
    await expect(page.locator('#status-message')).not.toHaveText('Load a model to begin');
  });

  test('should switch model selector tabs', async ({ page }) => {
    await page.goto('/#/decision/decision_maker');
    await page.waitForLoadState('domcontentloaded');
    const containerId = 'model-selector-container';

    await page.click(`#${containerId}-toggle button[data-value="upload"]`);
    await expect(page.locator(`#${containerId}-tab-upload`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-tab-standard`)).not.toHaveClass(/active/);
    await expect(page.locator('.model-upload')).toHaveAttribute('accept', '.task,.tflite,.litertlm');

    await page.click(`#${containerId}-toggle button[data-value="standard"]`);
    await expect(page.locator(`#${containerId}-tab-standard`)).toHaveClass(/active/);
  });

  test('should switch between Text, Vision, and Game tabs and evaluate ID photo presets', async ({ page }) => {
    await page.goto('/#/decision/decision_maker');
    await page.waitForLoadState('domcontentloaded');

    const toggle = page.locator('#view-mode-toggle');
    await expect(toggle.locator('button[data-value="text"]')).toHaveClass(/active/);
    await expect(toggle.locator('button[data-value="vision"]')).toBeVisible();
    await expect(toggle.locator('button[data-value="game"]')).toBeVisible();

    // Switch to Vision tab
    await toggle.locator('button[data-value="vision"]').click();
    await expect(toggle.locator('button[data-value="vision"]')).toHaveClass(/active/);
    await expect(page.locator('#dm-vision-view')).toBeVisible();
    await expect(page.locator('#dv-preset-strip .dv-preset-btn')).toHaveCount(8);
    await expect(page.locator('#dv-rules-list .dv-rule-card')).toHaveCount(9);
    await expect(page.locator('#dv-verdict-title')).toHaveText('COMPLIANT');
    await expect(page.locator('#dv-verdict-sub')).toHaveText('9 / 9 Passed');

    // Select a failing preset (Tilted Selfie -> 8 Fails)
    await page.locator('#dv-preset-strip .dv-preset-btn[data-preset-id="id_fail_tilted_glasses_selfie.jpg"]').click();
    await expect(page.locator('#dv-verdict-title')).toHaveText('8 ISSUES DETECTED');
    await expect(page.locator('#dv-verdict-sub')).toHaveText('1 / 9 Passed');
  });
});

