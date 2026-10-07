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

test.describe('Universal Embedder Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));

    await page.goto('/#/retrieval/universal_embedder');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state and settings', async ({ page }) => {
    // Verify task container and heading
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('.output-header h2')).toHaveText('Universal Embedder');

    // Settings Controls
    const l2Select = page.locator('#l2-normalize-select');
    await expect(l2Select).toBeVisible();
    await expect(l2Select).toHaveValue('true');

    // Status messages
    const statusMsg = page.locator('#status-message');
    await expect(statusMsg).toBeVisible();
    await expect(statusMsg).toContainText('model');

    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toBeVisible();
    await expect(inferenceTime).toContainText('Inference Time: - ms');

    // Model selector tabs
    const standardTab = page.locator('#model-selector-container-tab-standard');
    await expect(standardTab).toBeVisible();
    const modelSelect = page.locator('#model-selector-container-standard-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect.locator('option')).toHaveCount(2);
    const loadModelBtn = page.locator('#model-selector-container-standard-load-btn');
    await expect(loadModelBtn).toBeVisible();
    await expect(loadModelBtn).toHaveCSS('justify-content', 'center');

    // Presets bar
    const presetBar = page.locator('.preset-bar');
    await expect(presetBar).toBeVisible();
    const presetButtons = presetBar.locator('.preset-pair-btn');
    await expect(presetButtons).toHaveCount(5);

    // Input Slots initial mode (Slot A: Text, Slot B: Image)
    await expect(page.locator('#mode-text-a')).toHaveClass(/active/);
    await expect(page.locator('#panel-text-a')).toBeVisible();
    await expect(page.locator('#panel-image-a')).toBeHidden();

    await expect(page.locator('#mode-image-b')).toHaveClass(/active/);
    await expect(page.locator('#panel-image-b')).toBeVisible();
    await expect(page.locator('#panel-text-b')).toBeHidden();
    await expect(page.locator('#gallery-strip-b')).toBeVisible();

    // Compute button is disabled initially before model is loaded
    const embedBtn = page.locator('#embed-btn');
    await expect(embedBtn).toBeVisible();
    await expect(embedBtn).toHaveText('Compute Similarity');
    await expect(embedBtn).toBeDisabled();
    await expect(embedBtn).toHaveCSS('justify-content', 'center');
  });

  test('should switch input modes between text and image for both slots', async ({ page }) => {
    // Slot A: switch to Image mode
    await page.click('#mode-image-a');
    await expect(page.locator('#mode-image-a')).toHaveClass(/active/);
    await expect(page.locator('#mode-text-a')).not.toHaveClass(/active/);
    await expect(page.locator('#panel-image-a')).toBeVisible();
    await expect(page.locator('#panel-text-a')).toBeHidden();
    await expect(page.locator('#gallery-strip-a')).toBeVisible();

    // Slot A: switch back to Text mode
    await page.click('#mode-text-a');
    await expect(page.locator('#mode-text-a')).toHaveClass(/active/);
    await expect(page.locator('#panel-text-a')).toBeVisible();
    await expect(page.locator('#panel-image-a')).toBeHidden();

    // Slot B: switch to Text mode
    await page.click('#mode-text-b');
    await expect(page.locator('#mode-text-b')).toHaveClass(/active/);
    await expect(page.locator('#mode-image-b')).not.toHaveClass(/active/);
    await expect(page.locator('#panel-text-b')).toBeVisible();
    await expect(page.locator('#panel-image-b')).toBeHidden();

    // Slot B: switch back to Image mode
    await page.click('#mode-image-b');
    await expect(page.locator('#mode-image-b')).toHaveClass(/active/);
    await expect(page.locator('#panel-image-b')).toBeVisible();
    await expect(page.locator('#panel-text-b')).toBeHidden();
  });

  test('should load cross-modal presets into input slots', async ({ page }) => {
    // Click "Cat (Text ↔ Image)"
    const catPreset = page.locator('.preset-pair-btn', { hasText: 'Cat (Text ↔ Image)' });
    await catPreset.click();

    // Slot A: Text mode with cat description
    await expect(page.locator('#mode-text-a')).toHaveClass(/active/);
    await expect(page.locator('#input-text-a')).toHaveValue('a cute furry pet cat');

    // Slot B: Image mode with cat image
    await expect(page.locator('#mode-image-b')).toHaveClass(/active/);
    const previewB = page.locator('#preview-img-b');
    await expect(previewB).toHaveAttribute('src', /.*cute_cat\.jpg/);

    // Click "Apple ↔ Banana (Image ↔ Image)"
    const appleBananaPreset = page.locator('.preset-pair-btn', { hasText: 'Apple ↔ Banana' });
    await appleBananaPreset.click();

    // Both slots in Image mode
    await expect(page.locator('#mode-image-a')).toHaveClass(/active/);
    await expect(page.locator('#mode-image-b')).toHaveClass(/active/);
    await expect(page.locator('#preview-img-a')).toHaveAttribute('src', /.*red_apple\.jpg/);
    await expect(page.locator('#preview-img-b')).toHaveAttribute('src', /.*yellow_banana\.jpg/);

    // Click "Beach ↔ Mountain (Text ↔ Text)"
    const beachMountainPreset = page.locator('.preset-pair-btn', { hasText: 'Beach ↔ Mountain' });
    await beachMountainPreset.click();

    // Both slots in Text mode
    await expect(page.locator('#mode-text-a')).toHaveClass(/active/);
    await expect(page.locator('#mode-text-b')).toHaveClass(/active/);
    await expect(page.locator('#input-text-a')).toHaveValue('sunny ocean beach with waves');
    await expect(page.locator('#input-text-b')).toHaveValue('snowy winter mountain peak');
  });

  test('should update selected sample image from gallery strip', async ({ page }) => {
    // In Slot B (default Image mode), click a sample thumbnail from the gallery strip
    const openBookThumb = page.locator('#gallery-strip-b .sample-thumb-btn[title="open book"]');
    await expect(openBookThumb).toBeVisible();
    await openBookThumb.click();
    await expect(openBookThumb).toHaveClass(/selected/);

    // Verify preview image src and title are updated
    const previewB = page.locator('#preview-img-b');
    await expect(previewB).toHaveAttribute('src', /.*open_book\.jpg/);
    await expect(previewB).toHaveAttribute('title', 'open book');
  });

  test('should switch model selector tabs', async ({ page }) => {
    const containerId = 'model-selector-container';

    // Click "Upload" tab
    await page.click(`#${containerId}-toggle button[data-value="upload"]`);
    await expect(page.locator(`#${containerId}-tab-upload`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-tab-standard`)).not.toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-file-input`)).toBeAttached();

    // Click back to "Standard" tab
    await page.click(`#${containerId}-toggle button[data-value="standard"]`);
    await expect(page.locator(`#${containerId}-tab-standard`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-standard-select`)).toBeVisible();
  });
});
