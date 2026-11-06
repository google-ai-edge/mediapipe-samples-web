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

import { test, expect, type Page } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MODEL_CACHE_NAME = 'mediapipe-models-v1';
const BERT_URL_PART = 'bert_classifier.tflite';

/** Whether the shared model cache holds a model whose URL contains `part`. */
async function isCached(page: Page, part: string): Promise<boolean> {
  return page.evaluate(
    async ([cacheName, urlPart]) => {
      const cache = await caches.open(cacheName);
      return (await cache.keys()).some((r) => r.url.includes(urlPart));
    },
    [MODEL_CACHE_NAME, part]
  );
}

test.describe('Model cache: delete the active model', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/bert_classifier.tflite', (route) =>
      route.fulfill({ path: path.join(__dirname, 'assets', 'bert_classifier.tflite') })
    );
    await page.route('**/average_word_classifier.tflite', (route) =>
      route.fulfill({ path: path.join(__dirname, 'assets', 'average_word_classifier.tflite') })
    );
    await page.goto('#/text/text_classifier');
  });

  test('removes the active model from the cache and keeps it working', async ({ page }) => {
    const classifyBtn = page.locator('#classify-btn');
    await expect(classifyBtn).toHaveText('Classify', { timeout: 30000 });
    await expect(classifyBtn).toBeEnabled();

    // The active model is cached; the badge has no delete button, but the
    // model's entry in the option list does.
    await expect.poll(() => isCached(page, BERT_URL_PART), { timeout: 15000 }).toBe(true);
    const badge = page.locator('.model-badge-container');
    await expect(badge).toBeVisible();
    await expect(badge).toContainText('Active');
    await expect(page.locator('.model-delete-btn')).toBeHidden();

    await page.locator('.model-select').click();
    const deleteIcon = page.locator('option[data-state="loaded"] .model-option-delete');
    await expect(deleteIcon).toBeVisible();
    await expect(deleteIcon).toHaveAttribute('title', /stays active/);
    await deleteIcon.click();
    await page.keyboard.press('Escape');

    // Gone from the cache, still the active model, and no delete icon left.
    await expect.poll(() => isCached(page, BERT_URL_PART)).toBe(false);
    await expect(badge).toBeVisible();
    await expect(badge).toContainText('Active');
    await expect(page.locator('option[data-state="loaded"] .model-option-delete')).toHaveCount(0);

    // The model in memory still classifies.
    await page.locator('#text-input').fill('This movie was absolutely fantastic!');
    await classifyBtn.click();
    await expect(page.locator('#classification-results .classification-item')).toHaveCount(2);
  });
});
