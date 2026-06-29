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

test.describe('Interactive Segmenter Task', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.click('a[href="#/vision/interactive_segmenter"]');
    await page.waitForSelector('#status-message', { state: 'visible', timeout: 30000 });
  });

  test('should load model and handle click interaction', async ({ page }) => {
    // Wait for model ready
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 30000 });

    // Switch to Image view mode
    await page.click('button[data-value="image"]');

    // Wait for view change to propagate and image to be visible
    const testImage = page.locator('#test-image');
    await expect(testImage).toBeVisible();

    // Click on the image to trigger segmentation
    // Wait for Ready status before interaction
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 30000 });

    // We click near the center of the image (assuming cat/dog object is central)
    // Adding minor delay to ensure event listeners are fully attached to new view state
    await page.waitForTimeout(500); 
    await testImage.click({ position: { x: 150, y: 150 } }); // Assuming 300x300 image roughly

    // Wait for "Done in ..." status
    await expect(page.locator('#status-message')).toHaveText(/Done in/, { timeout: 15000 });

    // Check inference time
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
  });

  test('should handle delegate selection', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 60000 });
  });
});
