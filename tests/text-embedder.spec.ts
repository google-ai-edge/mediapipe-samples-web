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

test.describe('Text Embedding Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log(`[BROWSER ${msg.type()}] ${msg.text()}`));
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    await page.route('**/universal_sentence_encoder.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'universal_sentence_encoder.tflite');
      console.log(`Intercepting USE model request to: ${assetPath}`);
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/text/text_embedder');
  });

  test('should load model and compute similarity', async ({ page }) => {
    const embedBtn = page.locator('#embed-btn');
    // Model load might take time
    await expect(embedBtn).toHaveText('Compute Similarity', { timeout: 30000 });
    await expect(embedBtn).toBeEnabled();

    // Set text inputs manually or via chips
    await page.fill('#text-input-1', 'This is a positive test.');
    await page.fill('#text-input-2', 'This is also a positive test.');

    // Compute
    await embedBtn.click();

    // Wait for results
    const resultsContainer = page.locator('#embedding-results');
    await expect(resultsContainer).toBeVisible();
    await expect(page.locator('#similarity-value')).not.toHaveText('--');

    // Check reasonable similarity
    const similarity = await page.locator('#similarity-value').innerText();
    console.log('Similarity:', similarity);
    expect(parseFloat(similarity)).toBeGreaterThan(0.5);

    // Verify inference time is displayed
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('ms');
  });
});
