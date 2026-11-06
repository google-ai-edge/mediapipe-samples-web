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
    await expect(loadBtn).toHaveText('Download and Initialize');
    await expect(status).toBeVisible();
    await expect(status).toHaveText('Model not downloaded (175.3 MB)');
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

  test('shows the download, then the initialization, of a large model', async ({ page }) => {
    await page.route('**/universal_sentence_encoder.tflite', (route) =>
      route.fulfill({ path: path.join(__dirname, 'assets', 'universal_sentence_encoder.tflite') })
    );
    // Hold the large download open so the "downloading" state can be checked.
    let releaseDownload: () => void = () => {};
    const downloadHeld = new Promise<void>((resolve) => (releaseDownload = resolve));
    await page.route('**/embedding_gemma.task', async (route) => {
      await downloadHeld;
      await route.abort();
    });

    await page.goto('#/text/text_embedder');
    await expect(page.locator('#embed-btn')).toBeEnabled({ timeout: 30000 });
    const loadBtn = page.locator('#model-selector-container-standard-load-btn');
    const status = page.locator('#model-selector-container-standard-status');
    const viewport = page.locator('.viewport');

    await page.locator('#model-selector-container-standard-select').selectOption('embedding_gemma_v1');
    await loadBtn.click();

    // Downloading: the button shows it, the status line is hidden, and the
    // viewport overlay says so.
    await expect(loadBtn).toHaveText('Downloading…');
    await expect(loadBtn).toBeDisabled();
    await expect(status).toBeHidden();
    await expect(viewport).toHaveClass(/loading-model/);
    await expect(viewport).toHaveAttribute('data-loading-text', 'Downloading model…');
    await expect(page.locator('#status-message')).toHaveText('Downloading model…');

    // A failed download clears the overlay and re-enables the button.
    releaseDownload();
    await expect(page.locator('#status-message')).toHaveText(/^Error/, { timeout: 30000 });
    await expect(viewport).not.toHaveClass(/loading-model/);
    await expect(loadBtn).toBeEnabled();
    await expect(loadBtn).toHaveText('Download and Initialize');
  });
});
