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

test.describe('Text-to-Speech Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));

    await page.goto('/mediapipe-samples-web/#/audio/text_to_speech');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state, presets, and view switching', async ({ page }) => {
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('h2')).toHaveText('Text-to-Speech');

    const modelSelect = page.locator('.model-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue('phonon_31m');

    const voiceSelect = page.locator('#tts-voice-select');
    await expect(voiceSelect).toHaveValue('freya');

    const presetBtns = page.locator('#tts-presets-bar .sample-btn');
    await expect(presetBtns).toHaveCount(10);

    // Click Spanish preset and verify model, voice & reader update
    await page.locator('#tts-presets-bar .sample-btn[data-preset="es-ES"]').click();
    await expect(page.locator('#tts-locale-select')).toHaveValue('es-ES');
    await expect(modelSelect).toHaveValue('pocket_tts_100m_multi');
    await expect(voiceSelect).toHaveValue('estelle');

    // Switch to Custom Text view
    const tabEditor = page.locator('#view-mode-toggle button[data-value="editor"]');
    const viewReader = page.locator('#view-reader');
    const viewEditor = page.locator('#view-editor');

    await expect(viewReader).toBeVisible();
    await expect(viewEditor).not.toBeVisible();

    await tabEditor.click();
    await expect(viewEditor).toBeVisible();
    await expect(viewReader).not.toBeVisible();
  });
});
