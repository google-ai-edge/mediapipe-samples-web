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

test.describe('Image Classifier Task', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/#/vision/image_classifier');
    // Wait for the page to load the model (Ready OR Done in...)
    await expect(page.locator('#status-message')).toHaveText(/Ready|Done in/, { timeout: 60000 });
  });

  test('should load model and handle image inference', async ({ page }) => {
    const status = page.locator('#status-message');
    await expect(status).toHaveText(/Ready|Done in/, { timeout: 30000 });

    // Check if default results are displayed for the default image
    const resultsContainer = page.locator('#classification-results');
    await expect(resultsContainer).toBeVisible();

    await expect(resultsContainer).toContainText(/dog|golden retriever|labrador/i, { timeout: 10000 });
  });

  test('should handle delegate switching', async ({ page }) => {
    // Wait for initial load
    await expect(page.locator('#status-message')).toHaveText(/Ready|Done in/, { timeout: 30000 });

    // Switch to CPU
    await page.selectOption('#delegate-select', 'CPU');

    // Wait for reload
    await expect(page.locator('#status-message')).toHaveText(/Downloading model…|Initializing model…/, { timeout: 5000 });
    await expect(page.locator('#status-message')).toHaveText(/Ready|Done in/, { timeout: 30000 });
  });

  test('should update results when max results slider changes', async ({ page }) => {
    await expect(page.locator('#status-message')).toHaveText(/Ready|Done in/, { timeout: 30000 });

    // Change max results to 1
    await page.fill('#max-results', '1');
    await page.dispatchEvent('#max-results', 'input');

    // Check if we only have 1 result row
    const rows = page.locator('.classification-item');
    await expect(rows).toHaveCount(1);
  });
});
