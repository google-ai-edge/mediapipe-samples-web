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

test.describe('Text to Speech Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', (err) => console.log(`[Browser Error]: ${err}`));
    page.on('requestfailed', (req) =>
      console.log(`[Request Failed]: ${req.url()} - ${req.failure()?.errorText}`)
    );

    await page.goto('/#/audio/text_to_speech');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state and settings', async ({ page }) => {
    // 1. Page Load & Headers
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('h2')).toHaveText('Text to Speech');

    // 2. Status & Inference Time
    const statusMessage = page.locator('#status-message');
    await expect(statusMessage).toBeVisible();
    await expect(statusMessage).toHaveText('Model not loaded');

    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toBeVisible();
    await expect(inferenceTime).toHaveText('Inference Time: Model not loaded');

    // 3. Model Selection
    const modelSelect = page.locator('#model-selector-container-standard-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue('kokoro_82m');
    // Only the Kokoro model should be exposed
    await expect(modelSelect.locator('option')).toHaveCount(1);
    await expect(modelSelect.locator('option')).toHaveText('Kokoro-82M (9 languages)');

    const initButton = page.locator('#model-selector-container-standard-load-btn');
    await expect(initButton).toBeVisible();
    await expect(initButton).toHaveText('Initialize Task');

    // 4. Settings
    const voiceSelect = page.locator('#voice-select');
    await expect(voiceSelect).toBeVisible();
    await expect(voiceSelect).toBeDisabled();

    const speakingRateSlider = page.locator('#speaking-rate');
    await expect(speakingRateSlider).toBeVisible();
    await expect(speakingRateSlider).toHaveValue('1');
    await expect(page.locator('#speaking-rate-value')).toHaveText('1.0');

    const delegateSelect = page.locator('#delegate-select');
    await expect(delegateSelect).toBeVisible();
    await expect(delegateSelect).toHaveValue('GPU');

    // 5. Text Input & Sample Presets
    const sampleButtons = page.locator('#tts-samples .sample-chip');
    await expect(sampleButtons).toHaveCount(4);

    const textInput = page.locator('#tts-text-input');
    await expect(textInput).toBeVisible();
    await expect(textInput).toHaveValue(/Welcome to MediaPipe Text to Speech/);

    // 6. Action Buttons (disabled until model is loaded)
    const speakBtn = page.locator('#tts-speak-btn');
    await expect(speakBtn).toBeVisible();
    await expect(speakBtn).toBeDisabled();

    const streamBtn = page.locator('#tts-stream-btn');
    await expect(streamBtn).toBeVisible();
    await expect(streamBtn).toBeDisabled();

    const stopBtn = page.locator('#tts-stop-btn');
    await expect(stopBtn).toBeVisible();
    await expect(stopBtn).toBeDisabled();

    const downloadBtn = page.locator('#tts-download-btn');
    await expect(downloadBtn).toBeVisible();
    await expect(downloadBtn).toBeDisabled();

    // 7. Output Panel (Waveform, Metrics, Transcript)
    await expect(page.locator('#tts-waveform')).toBeVisible();
    await expect(page.locator('#tts-metric-first')).toHaveText('-');
    await expect(page.locator('#tts-metric-total')).toHaveText('-');
    await expect(page.locator('#tts-metric-audio')).toHaveText('-');
    await expect(page.locator('#tts-metric-rtf')).toHaveText('-');

    const transcript = page.locator('#tts-transcript');
    await expect(transcript).toBeVisible();
    await expect(transcript).toBeEmpty();
  });

  test('should handle model selector tabs between standard and upload', async ({ page }) => {
    const containerId = 'model-selector-container';

    // Click "Upload" tab
    await page.click(`#${containerId}-toggle button[data-value="upload"]`);
    await expect(page.locator(`#${containerId}-tab-upload`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-tab-standard`)).not.toHaveClass(/active/);
    const fileInput = page.locator(`#${containerId}-file-input`);
    await expect(fileInput).toBeAttached();
    await expect(fileInput).toHaveAttribute('accept', '.litertlm');

    // Click back to "Standard" tab
    await page.click(`#${containerId}-toggle button[data-value="standard"]`);
    await expect(page.locator(`#${containerId}-tab-standard`)).toHaveClass(/active/);
    await expect(page.locator(`#${containerId}-standard-select`)).toBeVisible();
  });

  test('should update text input when sample chips are clicked', async ({ page }) => {
    const textInput = page.locator('#tts-text-input');

    await page.click('#tts-samples button[data-sample="story"]');
    await expect(textInput).toHaveValue(/The lighthouse keeper climbed the spiral stairs/);

    await page.click('#tts-samples button[data-sample="numbers"]');
    await expect(textInput).toHaveValue(/Your flight to Zurich departs on March 3rd/);

    await page.click('#tts-samples button[data-sample="welcome"]');
    await expect(textInput).toHaveValue(/Welcome to MediaPipe Text to Speech/);
  });

  test('should support speaking rate adjustment and delegate switching', async ({ page }) => {
    const speakingRateSlider = page.locator('#speaking-rate');
    const speakingRateValue = page.locator('#speaking-rate-value');

    await speakingRateSlider.fill('1.4');
    await speakingRateSlider.dispatchEvent('input');
    await speakingRateSlider.dispatchEvent('change');
    await expect(speakingRateValue).toHaveText('1.4');

    const delegateSelect = page.locator('#delegate-select');
    await delegateSelect.selectOption('CPU');
    await expect(delegateSelect).toHaveValue('CPU');

    await delegateSelect.selectOption('GPU');
    await expect(delegateSelect).toHaveValue('GPU');

    // Should not trigger model download while model is not yet requested
    await expect(page.locator('#status-message')).toHaveText('Model not loaded');
  });

  test('should render output controls and waveform layout', async ({ page }) => {
    const canvas = page.locator('#tts-waveform');
    await expect(canvas).toBeVisible();

    const speakBtn = page.locator('#tts-speak-btn');
    await expect(speakBtn).toBeVisible();
    await expect(speakBtn).toContainText('Speak');

    const streamBtn = page.locator('#tts-stream-btn');
    await expect(streamBtn).toBeVisible();
    await expect(streamBtn).toContainText('Stream');

    const stopBtn = page.locator('#tts-stop-btn');
    await expect(stopBtn).toBeVisible();
    await expect(stopBtn).toContainText('Stop');

    const downloadBtn = page.locator('#tts-download-btn');
    await expect(downloadBtn).toBeVisible();
    await expect(downloadBtn).toContainText('WAV');
  });

  test('should adapt layout for mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    await page.waitForLoadState('domcontentloaded');

    const controlsPanel = page.locator('.controls-panel');
    await expect(controlsPanel).toBeVisible();

    const box = await controlsPanel.boundingBox();
    if (box) {
      expect(box.width).toBeGreaterThan(340);
      expect(box.width).toBeLessThan(376);
    }

    const mobileHeader = page.locator('.mobile-header');
    await expect(mobileHeader).toBeVisible();

    const select = page.locator('#mobile-task-select');
    await expect(select).toHaveValue('#/audio/text_to_speech');
  });
});
