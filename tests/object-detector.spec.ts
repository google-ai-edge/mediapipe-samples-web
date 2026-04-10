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

test.describe('Object Detection Task', () => {
  let imagePath: string;

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, '..', 'public', 'dog.jpg');
  });

  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      // Filter out overly noisy WebGL verbose logs from Chromium if desired, but keep for now
      console.log(`[BROWSER ${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    // Clear local storage to ensure fresh start
    await page.addInitScript(() => window.localStorage.clear());
    // Force CPU by default to ensure reliability in standard tests
    await page.goto('?delegate=CPU#/vision/object_detector');
    await page.waitForSelector('h2:has-text("Object Detection")');
    // Wait for model to load
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Running detection\.\.\.)|(Done)|(Ready)/, { timeout: 60000 });
  });

  // Intercept network requests for models to serve locally
  test.beforeEach(async ({ page }) => {
    await page.route('**/efficientdet_lite0.tflite', route => {
      route.fulfill({ path: path.join(__dirname, 'assets', 'efficientdet_lite0.tflite') });
    });
    await page.route('**/efficientdet_lite2.tflite', route => {
      route.fulfill({ path: path.join(__dirname, 'assets', 'efficientdet_lite2.tflite') });
    });
  });

  test('should verify full object detection cycle (CPU, GPU, Webcam)', async ({ page }) => {
    // 1. Check default settings
    await expect(page.locator('.model-select')).toHaveValue('efficientdet_lite0');
    // Ensure we are testing CPU
    await page.selectOption('#delegate-select', 'CPU');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Running detection\.\.\.)|(Done)|(Ready)/, { timeout: 60000 });

    // 2. Upload Image & Run CPU Detection
    await page.click('#view-mode-toggle button[data-value="image"]'); // Switch to Image tab
    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 30000 });

    // Check results
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    const resultsText = await page.locator('#test-results').textContent();
    const detections = JSON.parse(resultsText || '[]');
    expect(detections.length).toBeGreaterThan(0);
    const firstCat = detections[0].categories[0];
    expect(firstCat.categoryName.toLowerCase()).toContain('dog');

    // 3. Test Max Results & Threshold changes
    await page.fill('#score-threshold', '0.1');
    await page.fill('#max-results', '5');
    await page.locator('#score-threshold').dispatchEvent('input');
    await page.locator('#max-results').dispatchEvent('input');

    await expect(page.locator('#score-threshold-value')).toHaveText('0.1');
    await expect(page.locator('#max-results-value')).toHaveText('5');

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });

    // 4. Switch to GPU
    await page.selectOption('#delegate-select', 'GPU');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 90000 });

    // 5. Webcam Toggling
    await page.click('#view-mode-toggle button[data-value="video"]');
    await page.waitForSelector('#webcamButton:not([disabled])');

    await page.click('#webcamButton');

    await expect(page.locator('#webcamButton')).not.toHaveText('Initializing...', { timeout: 15000 });
    await expect(page.locator('#status-message')).toHaveText(/(Webcam running\.\.\.)|(Done)|(Ready)/, { timeout: 15000 });

    const camContainer = page.locator('.cam-container');
    await expect(camContainer).toBeVisible();

    const bgColor = await camContainer.evaluate(el => {
      return window.getComputedStyle(el).backgroundColor;
    });
    expect(bgColor).toMatch(/rgba?\(0,\s*0,\s*0,\s*0\)|transparent/);

    await page.click('#webcamButton');
    await expect(page.locator('#webcamButton')).toHaveText('Enable Webcam', { timeout: 10000 });
  });


  test('should handle custom model uploads', async ({ page }) => {
    const modelPath = path.resolve(__dirname, 'assets', 'efficientdet_lite0.tflite');

    await page.click('#model-selector-container-toggle button[data-value="upload"]');
    await page.setInputFiles('.model-upload', modelPath);

    await expect(page.locator('.upload-status')).toHaveText('efficientdet_lite0.tflite');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 90000 });

    // Verify it still detects
    await page.click('#view-mode-toggle button[data-value="image"]');
    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 15000 });
    const resultsText = await page.locator('#test-results').textContent();
    expect(JSON.parse(resultsText || '[]').length).toBeGreaterThan(0);
  });
});
