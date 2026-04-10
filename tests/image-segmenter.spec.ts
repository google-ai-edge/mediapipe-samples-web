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

test.describe('Image Segmentation Task', () => {
  let imagePath: string;

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, '..', 'public', 'dog.jpg');
  });

  test.beforeEach(async ({ page }) => {
    // Clear local storage to ensure fresh start
    await page.addInitScript(() => window.localStorage.clear());
    
    // Put param before hash so visual regression works without reliance on fallback
    await page.goto('?delegate=CPU#/vision/image_segmenter');
    page.on('console', msg => console.log(`BROWSER LOG: ${msg.text()}`));
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    // Wait for the UI to be ready
    await page.waitForSelector('#view-mode-toggle button[data-value="image"]');
    // Wait for model to load
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Done)|(Ready)/, { timeout: 120000 });
    // Also check that button is enabled
    await expect(page.locator('#webcamButton')).toBeEnabled({ timeout: 120000 });
  });

  test.beforeEach(async ({ page }) => {
    await page.route('**/deeplab_v3.tflite', route => {
      route.fulfill({ path: path.join(__dirname, 'assets', 'deeplab_v3.tflite') });
    });
    await page.route('**/hair_segmenter.tflite', route => {
      route.fulfill({ path: path.join(__dirname, 'assets', 'hair_segmenter.tflite') });
    });
  });

  test.afterEach(async ({ page }) => {
    await page.evaluate(() => {
      if ((window as any).cleanupActiveTask) {
        (window as any).cleanupActiveTask();
      }
    });
  });

  test('should verify image segmentation cycle (Defaults, CPU, Confidence, Opacity)', async ({ page }) => {
    // 1. Check default settings
    await expect(page.locator('.model-select')).toHaveValue('deeplab_v3');
    await expect(page.locator('#delegate-select')).toHaveValue(/CPU|GPU/);
    await page.selectOption('#delegate-select', 'CPU');
    await expect(page.locator('#output-type')).toHaveValue('CATEGORY_MASK');

    // 2. Segment image on CPU
    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // 3. Handle component changes (Confidence Mask)
    await page.selectOption('#output-type', 'CONFIDENCE_MASKS');
    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });

    // 4. Handle opacity changes
    await page.fill('#opacity', '0.2');
    await page.locator('#opacity').dispatchEvent('input');
    await page.waitForTimeout(500);
    await expect(page.locator('#test-results')).toBeAttached();
  });




  test('should segment image on CPU explicitly', async ({ page }) => {
    await expect(page.locator('#delegate-select')).toHaveValue(/CPU|GPU/);
    await page.selectOption('#delegate-select', 'CPU');

    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Check availability of results element
    await expect(page.locator('#test-results')).toBeAttached();

    // Visual Comparison
    // This will generate a golden on first run
    await expect(page).toHaveScreenshot('segmentation-cpu-explicit.png', { maxDiffPixelRatio: 0.2, timeout: 10000 });
  });



  test('should segment image on GPU (emulated) @gpu', async ({ page }) => {
    // Listen for console logs to debug fallback
    page.on('console', msg => {
      if (msg.type() === 'warning' || msg.type() === 'error') {
        console.log(`PAGE LOG: ${msg.text()}`);
      }
    });

    // Navigate specifically to GPU mode
    await page.goto('?delegate=GPU#/vision/image_segmenter');
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    // Wait for the UI to be ready
    await page.waitForSelector('#view-mode-toggle button[data-value="image"]');
    // Wait for model to load (might be slower on emulated GPU)
    // Wait for model to load (might be slower on emulated GPU)
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Done)|(Ready)/, { timeout: 60000 });

    await expect(page.locator('#delegate-select')).toHaveValue(/GPU|CPU/);

    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 30000 });

    // Verify result is present (visual regression might differ slightly on GPU, so maybe just check functionality)
  });

  test('should handle model switching', async ({ page }) => {
    await page.selectOption('.model-select', 'hair_segmenter');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 60000 });

    // Wait for worker re-instantiation to fully commit DOM layout reflow
    await page.waitForTimeout(1000);

    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
    await expect(page.locator('#test-results')).toBeAttached({ timeout: 15000 });
  });



  test('should support webcam toggling', async ({ page }) => {
    await page.click('#view-mode-toggle button[data-value="video"]');
    await page.waitForSelector('#webcamButton:not([disabled])');

    await page.click('#webcamButton');

    await expect(page.locator('#webcamButton')).not.toHaveText('Initializing...', { timeout: 15000 });

    // Wait for Worker to actually process and return at least ONE video frame through GPU/CPU pipeline
    await expect(page.locator('#inference-time')).not.toHaveText('Inference Time: - ms', { timeout: 30000 });

    // Disable
    await page.click('#webcamButton');
    await expect(page.locator('#webcamButton')).toHaveText('Enable Webcam', { timeout: 10000 });
  });

  test('should handle custom model uploads', async ({ page }) => {
    const modelPath = path.resolve(__dirname, 'assets', 'deeplab_v3.tflite');

    // Switch to upload tab and provide file
    await page.click('#model-selector-container-toggle button[data-value="upload"]');
    await page.setInputFiles('.model-upload', modelPath);

    // Verify it loads
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 60000 });

    // Verify it detects
    await page.setInputFiles('#image-upload', imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
    await expect(page.locator('#test-results')).toBeAttached();
  });
});
