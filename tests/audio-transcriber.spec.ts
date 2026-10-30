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

test.describe('Audio Transcriber Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));

    await page.goto('/mediapipe-samples-web/#/audio/audio_transcriber');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state, model selection, and view switching', async ({ page }) => {
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('h2')).toHaveText('Audio Transcriber');

    const modelSelect = page.locator('.model-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue('moonshine_tiny');

    const modeSelect = page.locator('#transcription-mode-select');
    await expect(modeSelect).toHaveValue('streaming');

    const sampleBtns = page.locator('#sample-clips-container .sample-btn');
    await expect(sampleBtns).toHaveCount(5);

    const tabMic = page.locator('#view-mode-toggle button[data-value="microphone"]');
    const viewFile = page.locator('#view-file');
    const viewMic = page.locator('#view-microphone');

    await expect(viewFile).toBeVisible();
    await expect(viewMic).not.toBeVisible();

    await tabMic.click();
    await expect(viewMic).toBeVisible();
    await expect(viewFile).not.toBeVisible();
  });
});
