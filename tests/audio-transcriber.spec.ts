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
    page.on('requestfailed', (req) => console.log(`[Request Failed]: ${req.url()} - ${req.failure()?.errorText}`));

    await page.goto('/#/audio/audio_transcriber');
    await page.waitForLoadState('domcontentloaded');
  });

  test('should verify initial UI state and settings', async ({ page }) => {
    // 1. Page Load & Headers
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('h2')).toHaveText('Audio Transcriber');

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
    await expect(modelSelect).toHaveValue('parakeet_tdt_v3');
    // Only the Parakeet model should be exposed
    await expect(modelSelect.locator('option')).toHaveCount(1);
    await expect(modelSelect.locator('option')).toContainText('Parakeet-TDT-0.6B-v3 (25 languages)');
    await expect(modelSelect.locator('option')).toContainText('586.1 MB');

    const initButton = page.locator('#model-selector-container-standard-load-btn');
    await expect(initButton).toBeVisible();
    await expect(initButton).toHaveText('Initialize Task');

    // 4. Settings
    const interimCheckbox = page.locator('#interim-results');
    await expect(interimCheckbox).toBeVisible();
    await expect(interimCheckbox).toBeChecked();

    const delegateSelect = page.locator('#delegate-select');
    await expect(delegateSelect).toBeVisible();
    await expect(delegateSelect).toHaveValue('GPU');

    // 5. Default View (Microphone)
    const viewMic = page.locator('#view-microphone');
    const viewFile = page.locator('#view-file');
    await expect(viewMic).toBeVisible();
    await expect(viewFile).not.toBeVisible();

    // 6. Transcript Panel
    await expect(page.locator('.asr-transcript-card')).toBeVisible();
    await expect(page.locator('.asr-transcript-title')).toHaveText('Transcript');
    const transcript = page.locator('#asr-transcript');
    await expect(transcript).toBeVisible();
    await expect(transcript).toBeEmpty();

    const copyBtn = page.locator('#asr-copy-btn');
    await expect(copyBtn).toBeVisible();
    await expect(copyBtn).toBeDisabled();

    const clearBtn = page.locator('#asr-clear-btn');
    await expect(clearBtn).toBeVisible();
    await expect(clearBtn).toBeDisabled();
  });

  test('should handle view mode switching between microphone and file upload', async ({ page }) => {
    const tabFile = page.locator('#view-mode-toggle button[data-value="file"]');
    const tabMic = page.locator('#view-mode-toggle button[data-value="mic"]');
    const viewMic = page.locator('#view-microphone');
    const viewFile = page.locator('#view-file');

    // Initial state: Microphone visible
    await expect(viewMic).toBeVisible();
    await expect(viewFile).not.toBeVisible();

    // Switch to File view
    await tabFile.click();
    await expect(viewFile).toBeVisible();
    await expect(viewMic).not.toBeVisible();

    // Switch back to Microphone view
    await tabMic.click();
    await expect(viewMic).toBeVisible();
    await expect(viewFile).not.toBeVisible();
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

  test('should support interim results toggle and delegate switching', async ({ page }) => {
    const interimCheckbox = page.locator('#interim-results');
    await expect(interimCheckbox).toBeChecked();

    await interimCheckbox.uncheck();
    await expect(interimCheckbox).not.toBeChecked();

    await interimCheckbox.check();
    await expect(interimCheckbox).toBeChecked();

    const delegateSelect = page.locator('#delegate-select');
    await delegateSelect.selectOption('CPU');
    await expect(delegateSelect).toHaveValue('CPU');

    await delegateSelect.selectOption('GPU');
    await expect(delegateSelect).toHaveValue('GPU');
  });

  test('should layout microphone view with waveform canvas and record button', async ({ page }) => {
    const canvas = page.locator('#waveform-canvas');
    await expect(canvas).toBeVisible();

    const recordButton = page.locator('#recordButton');
    await expect(recordButton).toBeVisible();
    await expect(recordButton).toBeDisabled();
    await expect(recordButton).toContainText('Start Recording');
  });

  test('should layout file upload view with dropzone and preview controls', async ({ page }) => {
    // Switch to File tab
    await page.click('#view-mode-toggle button[data-value="file"]');

    const dropzone = page.locator('.upload-dropzone');
    await expect(dropzone).toBeVisible();

    const fileInput = page.locator('#audio-upload');
    await expect(fileInput).toBeAttached();
    await expect(fileInput).toHaveAttribute('accept', 'audio/*,video/*');

    // Simulate dragover to check border highlight styling
    await dropzone.dispatchEvent('dragover');
    await expect(dropzone).toHaveCSS('border-color', 'rgb(0, 127, 139)');

    // Simulate uploaded audio preview state
    const previewContainer = page.locator('#audio-preview-container');
    await page.evaluate(() => {
      const pc = document.getElementById('audio-preview-container');
      if (pc) {
        pc.style.display = 'flex';
        pc.style.flexDirection = 'column';
      }
    });

    await expect(previewContainer).toBeVisible();
    await expect(previewContainer).toHaveCSS('flex-direction', 'column');

    const audioPlayer = page.locator('#audio-player');
    await expect(audioPlayer).toBeVisible();

    const transcribeBtn = page.locator('#run-file-transcription');
    await expect(transcribeBtn).toBeVisible();
    await expect(transcribeBtn).toBeDisabled();
    await expect(transcribeBtn).toContainText('Transcribe Audio');
  });

  test('should render initial transcript container with placeholder styling', async ({ page }) => {
    const transcript = page.locator('#asr-transcript');
    await expect(transcript).toBeVisible();
    await expect(transcript).toBeEmpty();

    const clearBtn = page.locator('#asr-clear-btn');
    await expect(clearBtn).toBeDisabled();

    const copyBtn = page.locator('#asr-copy-btn');
    await expect(copyBtn).toBeDisabled();
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
    await expect(select).toHaveValue('#/audio/audio_transcriber');
  });
});
