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
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Text Classification Task', () => {
  let modelFetches = 0;

  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[BROWSER ${msg.type()}] ${msg.text()}`));
    page.on('pageerror', (exc) => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    // Route model to local asset if available (downloaded by global-setup)
    modelFetches = 0;
    await page.route('**/bert_classifier.tflite', (route) => {
      const assetPath = path.join(__dirname, 'assets', 'bert_classifier.tflite');
      console.log(`Intercepting BERT model request to: ${assetPath}`);
      modelFetches++;
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/text/text_classifier');
  });

  test('should load model and classify text', async ({ page }) => {
    const classifyBtn = page.locator('#classify-btn');
    await expect(classifyBtn).toHaveText('Classify', { timeout: 30000 });
    await expect(classifyBtn).toBeEnabled();

    // Input text
    await page.fill('#text-input', 'I love this product, it is amazing!');
    await classifyBtn.click();

    // Wait for results
    const resultsContainer = page.locator('#classification-results');
    await expect(resultsContainer).not.toBeEmpty();

    // Check for positive sentiment
    await expect(resultsContainer).toContainText('positive');

    // Verify inference time is displayed
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('ms');
    await expect(inferenceTime).not.toContainText('- ms');
  });

  test('should cache the downloaded model and reuse it after a reload', async ({ page }) => {
    const classifyBtn = page.locator('#classify-btn');
    await expect(classifyBtn).toHaveText('Classify', { timeout: 30000 });
    expect(modelFetches).toBe(1);

    // The download was written to the shared model cache (keyed by URL).
    await expect
      .poll(
        () =>
          page.evaluate(async () => {
            const cache = await caches.open('mediapipe-models-v1');
            return (await cache.keys()).map((req) => req.url);
          }),
        { timeout: 10000 }
      )
      .toContainEqual(expect.stringContaining('bert_classifier.tflite'));

    // A second load is served from the cache: no new network request, and the
    // dropdown shows the model as loaded (badge on the control, tag in the list).
    await page.reload();
    await expect(classifyBtn).toHaveText('Classify', { timeout: 30000 });
    expect(modelFetches).toBe(1);
    const badge = page.locator('#model-selector-container-model-badge');
    await expect(badge).toBeVisible();
    await expect(badge).toHaveText(/Loaded/);
    await expect(badge).toHaveAttribute('data-state', 'loaded');
    await expect(page.locator('.model-select option[value="bert_classifier"]')).toHaveAttribute('data-state', 'loaded');
    await expect(page.locator('.model-select option[value="average_word_classifier"]')).not.toHaveAttribute(
      'data-state',
      /.*/
    );
  });
});
