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

test.describe('Holistic Landmarker Task', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the Holistic Landmarker page
    await page.goto('/#/vision/holistic_landmarker');

    // Wait for the page to show the status message
    await page.waitForSelector('#status-message', { state: 'visible', timeout: 30000 });
  });

  test('should load model and handle image inference', async ({ page }) => {
    // 1. Initial State: "Initializing..." then "Loading Model..." then "Ready"
    const status = page.locator('#status-message');

    // Wait for Ready status (model active) or processing
    await expect(status).toHaveText(/Ready|Processing|Done/, { timeout: 30000 });

    // 2. Select Image Tab (should be default)
    const tabImage = page.locator('#view-mode-toggle button[data-value="image"]');
    await expect(tabImage).toHaveClass(/active/);

    // 3. Verify Default Image is present
    const testImage = page.locator('#test-image');
    await expect(testImage).toBeVisible();

    // 4. Trigger detection if not already auto-triggered, or verify results
    // The code auto-triggers on 'Ready' if image is present.
    // Wait for "Done in ..." 
    await expect(status).toHaveText(/Done in/, { timeout: 30000 });

    // 5. Verify Inference Time is updated
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toHaveText(/Inference Time: \d+\.\d+ ms/);
  });

  test('should handle delegate switching', async ({ page }) => {
    const status = page.locator('#status-message');
    await expect(status).toHaveText(/Ready|Processing|Done/, { timeout: 30000 });

    // Switch to CPU
    await page.selectOption('#delegate-select', 'CPU');

    // Should trigger re-initialization
    await expect(status).toHaveText(/Loading Model|Ready/, { timeout: 10000 });
    // Eventually Ready again (or processing/done if auto-triggered)
    await expect(status).toHaveText(/Ready|Processing|Done/, { timeout: 60000 });
  });
});
