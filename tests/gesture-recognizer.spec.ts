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

test.describe('Gesture Recognizer Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log(`[Browser Console] ${msg.text()}`));
    page.on('pageerror', err => console.log(`[Browser Error] ${err.message}`));
    await page.goto('#/vision/gesture_recognizer');
    await page.waitForSelector('.viewport.loading-model', { state: 'detached', timeout: 30000 });
  });

  test('should load model and handle image upload', async ({ page }) => {
    // Check initial state
    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 30000 });

    // Upload image
    const fileInput = page.locator('#image-upload');
    await fileInput.setInputFiles('public/hand_model.png'); // Use hand image for better detection

    // Wait for result
    await expect(page.locator('#status-message')).toHaveText(/Done in/, { timeout: 15000 });
    await expect(page.locator('#classification-results')).not.toBeEmpty();

    // Check inference time display
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
  });

  test('should support webcam toggling', async ({ page }) => {
    await page.click('#view-mode-toggle button[data-value="video"]');
    await page.waitForSelector('#webcamButton:not([disabled])');

    const webcamBtn = page.locator('#webcamButton');

    // Explicitly click to start since we don't auto-start from localStorage by default
    await webcamBtn.click();

    await expect(webcamBtn).not.toHaveText('Initializing...', { timeout: 15000 });

    // Since we clicked it, it should eventually be "Disable Webcam"
    await expect(webcamBtn).toHaveText('Disable Webcam', { timeout: 10000 });
    await expect(page.locator('#status-message')).toHaveText(/(Webcam running...)|(Done)|(Ready)/, { timeout: 15000 });

    // Click to stop
    await webcamBtn.click();
    await expect(webcamBtn).toHaveText('Enable Webcam');
  });

  test('should handle delegate switching', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');
    // Wait for re-initialization
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded. Ready.)|(Ready)|(Done)/, { timeout: 60000 });

    // Switch back to GPU
    await page.selectOption('#delegate-select', 'GPU');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded. Ready.)|(Ready)|(Done)/, { timeout: 60000 });
  });
});
