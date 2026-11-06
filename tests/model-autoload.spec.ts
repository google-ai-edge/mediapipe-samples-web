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

test.describe('Per-model auto-download', () => {
  test('small models load on selection, large ones wait for Initialize Task', async ({ page }) => {
    let largeModelFetches = 0;
    await page.route('**/universal_sentence_encoder.tflite', (route) =>
      route.fulfill({ path: path.join(__dirname, 'assets', 'universal_sentence_encoder.tflite') })
    );
    // Embedding Gemma (~184 MB) must not download until the user asks for it.
    await page.route('**/embedding_gemma.task', (route) => {
      largeModelFetches++;
      route.abort();
    });

    await page.goto('#/text/text_embedder');
    const select = page.locator('#model-selector-container-standard-select');
    const loadBtn = page.locator('#model-selector-container-standard-load-btn');
    const status = page.locator('#model-selector-container-standard-status');

    // The small default model loads on page open, with no button or status line.
    await expect(page.locator('#embed-btn')).toBeEnabled({ timeout: 30000 });
    await expect(loadBtn).toBeHidden();
    await expect(status).toBeHidden();

    // Picking the large model shows "Initialize Task" and downloads nothing.
    await select.selectOption('embedding_gemma_v1');
    await expect(loadBtn).toBeVisible();
    await expect(loadBtn).toHaveText('Initialize Task');
    await expect(status).toBeVisible();
    await expect(status).toHaveText('Not loaded yet');
    await page.waitForTimeout(1000);
    expect(largeModelFetches).toBe(0);

    // Picking the small model again loads it right away and hides the button.
    await select.selectOption('universal_sentence_encoder');
    await expect(loadBtn).toBeHidden();
    await expect(page.locator('.model-badge-container')).toContainText('Active', { timeout: 30000 });
    expect(largeModelFetches).toBe(0);

    // Initialize Task starts the large download.
    await select.selectOption('embedding_gemma_v1');
    await loadBtn.click();
    await expect.poll(() => largeModelFetches).toBeGreaterThan(0);
  });
});
