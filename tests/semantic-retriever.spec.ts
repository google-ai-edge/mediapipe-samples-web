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

test.describe('Semantic Retriever Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));

    await page.goto('/#/retrieval/semantic_retriever');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state and settings', async ({ page }) => {
    // Task container & heading
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('.output-header h2')).toHaveText('Semantic Retriever');

    // Status messages
    const statusMsg = page.locator('#status-message');
    await expect(statusMsg).toBeVisible();
    await expect(statusMsg).toContainText('model');

    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toBeVisible();
    await expect(inferenceTime).toContainText('Inference Time: - ms');

    // Model selector container
    const standardTab = page.locator('#model-selector-container-tab-standard');
    await expect(standardTab).toBeVisible();
    const modelSelect = page.locator('#model-selector-container-standard-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect.locator('option')).toHaveCount(2);
  });

  test('should render 10 sample images in the index gallery with correct captions', async ({ page }) => {
    const galleryContainer = page.locator('#index-gallery-grid');
    await expect(galleryContainer).toBeVisible();

    const galleryItems = galleryContainer.locator('.index-gallery-item');
    await expect(galleryItems).toHaveCount(10);

    const expectedLabels = [
      'red apple',
      'yellow banana',
      'cute cat',
      'fast car',
      'green tree',
      'blue sky',
      'coffee mug',
      'open book',
      'sunny beach',
      'snowy mountain',
    ];

    for (let i = 0; i < expectedLabels.length; i++) {
      const item = galleryItems.nth(i);
      const img = item.locator('img');
      await expect(img).toBeVisible();
      await expect(img).toHaveAttribute('alt', expectedLabels[i]);
      await expect(img).toHaveAttribute('src', /.*\.jpg/);
    }
  });

  test('should verify action buttons ordering and responsiveness', async ({ page }) => {
    const actionsRow = page.locator('.retriever-step-actions');
    const indexBtn = page.locator('#btn-index');
    const customBtn = actionsRow.locator('label.sample-chip');

    await expect(indexBtn).toBeVisible();
    await expect(indexBtn).toContainText('Index 10 sample images');

    await expect(customBtn).toBeVisible();
    await expect(customBtn).toContainText('Add Custom Image');

    // Verify ordering: "Index 10 sample images" is first, "Add Custom Image" is second
    const children = actionsRow.locator('> *');
    await expect(children.first()).toHaveId('btn-index');
    await expect(children.last()).toContainText('Add Custom Image');
  });

  test('should verify query search controls and initial empty state', async ({ page }) => {
    // Search input
    const searchInput = page.locator('#query-input');
    await expect(searchInput).toBeVisible();
    await expect(searchInput).toBeDisabled();
    await expect(searchInput).toHaveAttribute('placeholder', /Search images by meaning/);

    // Search button is disabled initially before indexing
    const searchBtn = page.locator('#btn-search');
    await expect(searchBtn).toBeVisible();
    await expect(searchBtn).toContainText('Search');
    await expect(searchBtn).toBeDisabled();

    // Suggestion chips are disabled initially before indexing
    const suggestionChips = page.locator('.suggestion-chip');
    await expect(suggestionChips).toHaveCount(5);

    // Results list shows initial instruction prompt before indexing
    const resultsList = page.locator('#results-list');
    await expect(resultsList).toBeVisible();
    await expect(resultsList.locator('.no-results')).toContainText('Index the sample images above');
  });

  test('should switch model selector tabs in semantic retriever', async ({ page }) => {
    const containerId = 'model-selector-container';

    // Click "Upload" tab
    await page.click(`#${containerId}-toggle button[data-value="upload"]`);
    await expect(page.locator(`#${containerId}-tab-upload`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-tab-standard`)).not.toHaveClass(/active/);

    // Click back to "Standard" tab
    await page.click(`#${containerId}-toggle button[data-value="standard"]`);
    await expect(page.locator(`#${containerId}-tab-standard`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-standard-select`)).toBeVisible();
  });
});
