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
    page.on('requestfailed', (req) => console.log(`[Request Failed]: ${req.url()} - ${req.failure()?.errorText}`));

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
    // Kokoro (GPU and CPU variants), PocketTTS (English and multilingual) and Phonon
    const options = modelSelect.locator('option');
    await expect(options).toHaveCount(5);
    await expect(options.nth(0)).toContainText('Kokoro-82M (9 languages)');
    await expect(options.nth(0)).toContainText('281.1 MB');
    await expect(options.nth(1)).toContainText('Kokoro-82M CPU (9 languages)');
    await expect(options.nth(2)).toContainText('PocketTTS-100M (English)');
    await expect(options.nth(3)).toContainText('PocketTTS-100M (7 languages)');
    await expect(options.nth(4)).toContainText('Phonon-31M (English)');

    const initButton = page.locator('#model-selector-container-standard-load-btn');
    await expect(initButton).toBeVisible();
    await expect(initButton).toHaveText('Download and Initialize');

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

    // 7. Output Panel (Waveform, Metrics, Interactive Sentence Transcript)
    await expect(page.locator('#tts-waveform')).toBeVisible();
    await expect(page.locator('#tts-metric-first')).toHaveText('-');
    await expect(page.locator('#tts-metric-total')).toHaveText('-');
    await expect(page.locator('#tts-metric-audio')).toHaveText('-');
    await expect(page.locator('#tts-metric-rtf')).toHaveText('-');

    const transcript = page.locator('#tts-transcript');
    await expect(transcript).toBeVisible();
    await expect(transcript.locator('.tts-paragraph')).toHaveCount(3);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(4);
    await expect(page.locator('#tts-sentence-badge')).toHaveText('4 sentences');
    await expect(page.locator('#tts-prev-sent-btn')).toBeDisabled();
    await expect(page.locator('#tts-next-sent-btn')).toBeEnabled();
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

  test('should update text input and interactive sentence reader when sample chips are clicked', async ({ page }) => {
    const textInput = page.locator('#tts-text-input');
    const transcript = page.locator('#tts-transcript');

    await page.click('#tts-samples button[data-sample="story"]');
    await expect(textInput).toHaveValue(/The lighthouse keeper climbed the spiral stairs/);
    await expect(page.locator('#tts-samples button[data-sample="story"]')).toHaveClass(/\bactive\b/);
    await expect(page.locator('#tts-samples button[data-sample="welcome"]')).not.toHaveClass(/\bactive\b/);
    await expect(transcript.locator('.tts-paragraph')).toHaveCount(3);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(5);

    await page.click('#tts-samples button[data-sample="numbers"]');
    await expect(textInput).toHaveValue(/Your flight to Zurich departs on March 3rd/);
    await expect(page.locator('#tts-samples button[data-sample="numbers"]')).toHaveClass(/\bactive\b/);
    // "Dr." and "1.5" and "$249.99" must not split sentences prematurely
    await expect(transcript.locator('.tts-paragraph')).toHaveCount(3);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(4);

    await page.click('#tts-samples button[data-sample="article"]');
    await expect(textInput).toHaveValue(/On-device speech synthesis brings natural/);
    await expect(page.locator('#tts-samples button[data-sample="article"]')).toHaveClass(/\bactive\b/);
    await expect(transcript.locator('.tts-paragraph')).toHaveCount(3);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(3);

    await page.click('#tts-samples button[data-sample="welcome"]');
    await expect(textInput).toHaveValue(/Welcome to MediaPipe Text to Speech/);
    await expect(page.locator('#tts-samples button[data-sample="welcome"]')).toHaveClass(/\bactive\b/);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(4);

    // Active chip has distinct background styling and clears on custom edits.
    const activeBg = await page
      .locator('#tts-samples button[data-sample="welcome"]')
      .evaluate((el) => getComputedStyle(el).backgroundColor);
    const inactiveBg = await page
      .locator('#tts-samples button[data-sample="story"]')
      .evaluate((el) => getComputedStyle(el).backgroundColor);
    expect(activeBg).not.toBe(inactiveBg);

    await textInput.fill('Custom user utterance.');
    await expect(page.locator('#tts-samples .sample-chip.active')).toHaveCount(0);
    await expect(transcript.locator('.tts-sentence')).toHaveCount(1);
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

    // Streaming is the primary action and comes first.
    const streamBtn = page.locator('#tts-stream-btn');
    await expect(streamBtn).toBeVisible();
    await expect(streamBtn).toContainText('Stream');
    await expect(streamBtn).not.toHaveClass(/secondary/);
    await expect(page.locator('.tts-actions button').first()).toHaveId('tts-stream-btn');

    // Unary synthesis is secondary and explains that it waits for the whole text.
    const speakBtn = page.locator('#tts-speak-btn');
    await expect(speakBtn).toBeVisible();
    await expect(speakBtn).toContainText('Synthesize all');
    await expect(speakBtn).toHaveClass(/secondary/);
    await expect(speakBtn).toHaveAttribute('title', /entire text into a single clip/);

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

    // Reader hint text truncates with ellipsis rather than clipping raw flex items.
    const hintStyle = await page.locator('.tts-reader-hint-text').evaluate((el) => {
      const s = getComputedStyle(el);
      return { textOverflow: s.textOverflow, overflow: s.overflow, whiteSpace: s.whiteSpace };
    });
    expect(hintStyle).toEqual({ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' });
  });
});

test.describe('Text to Speech Interactive Flows', () => {
  test('should download the selected model and auto-stream the clicked sentence on pre-load click', async ({
    page,
  }) => {
    await page.addInitScript(() => {
      const messages: any[] = [];
      (window as any).__ttsWorkerMessages = messages;
      const OrigWorker = window.Worker;
      class MockTtsWorker {
        onmessage: ((e: MessageEvent) => void) | null = null;
        onerror: ((e: ErrorEvent) => void) | null = null;
        constructor(url: string | URL, opts?: WorkerOptions) {
          if (!String(url).includes('text-to-speech')) {
            return new OrigWorker(url, opts) as any;
          }
          (window as any).__ttsWorkerInstance = this;
        }
        postMessage(data: any) {
          messages.push(data);
        }
        terminate() {}
        addEventListener() {}
        removeEventListener() {}
        dispatchEvent() {
          return true;
        }
      }
      (window as any).Worker = MockTtsWorker;
    });

    await page.goto('/#/audio/text_to_speech');
    await page.waitForLoadState('domcontentloaded');

    // Pick a non-default model before clicking a sentence.
    await page.selectOption('#model-selector-container-standard-select', 'phonon_31m_en');
    await page.locator('#tts-transcript .tts-sentence').nth(2).click();

    // Status should report Downloading model… (not Initializing model…) for the selected model.
    await expect(page.locator('#status-message')).toHaveText('Downloading model…');
    const initMsg = await page.evaluate(() => (window as any).__ttsWorkerMessages.find((m: any) => m.type === 'INIT'));
    expect(initMsg?.modelName).toBe('phonon_31m_en');

    // Finish initialization from the mock worker and verify auto-stream of sentence 3.
    await page.evaluate(() => {
      (window as any).__ttsWorkerInstance.onmessage({
        data: {
          type: 'INIT_DONE',
          modelName: 'phonon_31m_en',
          voices: [
            { voiceId: 'af_heart', displayName: 'Heart', locale: 'en-US', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'bf_emma', displayName: 'Emma', locale: 'en-GB', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'ef_dora', displayName: 'Dora', locale: 'es-ES', sampleRateHz: 24000, isDownloaded: true },
          ],
        },
      });
    });

    await expect(page.locator('#status-message')).toHaveText('Streaming from sentence 3...');
    const streamMsg = await page.evaluate(() =>
      (window as any).__ttsWorkerMessages.find((m: any) => m.type === 'STREAM')
    );
    expect(streamMsg?.text).toMatch(/^By streaming clause chunks/);
  });

  test('should stop playback on voice switch, preserve same-language preset, warm selected voice on re-init, and sync mid-playback edits on stop', async ({
    page,
  }) => {
    await page.addInitScript(() => {
      const messages: any[] = [];
      (window as any).__ttsWorkerMessages = messages;
      const OrigWorker = window.Worker;
      class MockTtsWorker {
        onmessage: ((e: MessageEvent) => void) | null = null;
        onerror: ((e: ErrorEvent) => void) | null = null;
        constructor(url: string | URL, opts?: WorkerOptions) {
          if (!String(url).includes('text-to-speech')) {
            return new OrigWorker(url, opts) as any;
          }
          (window as any).__ttsWorkerInstance = this;
        }
        postMessage(data: any) {
          messages.push(data);
        }
        terminate() {}
        addEventListener() {}
        removeEventListener() {}
        dispatchEvent() {
          return true;
        }
      }
      (window as any).Worker = MockTtsWorker;
    });

    await page.goto('/#/audio/text_to_speech');
    await page.waitForLoadState('domcontentloaded');

    await page.click('#model-selector-container-standard-load-btn');
    await page.evaluate(() => {
      (window as any).__ttsWorkerInstance.onmessage({
        data: {
          type: 'INIT_DONE',
          modelName: 'kokoro_82m',
          voices: [
            { voiceId: 'af_heart', displayName: 'Heart', locale: 'en-US', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'bf_emma', displayName: 'Emma', locale: 'en-GB', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'ef_dora', displayName: 'Dora', locale: 'es-ES', sampleRateHz: 24000, isDownloaded: true },
          ],
        },
      });
    });

    // Switch to Story and start streaming.
    await page.click('#tts-samples button[data-sample="story"]');
    await page.click('#tts-stream-btn');
    await expect(page.locator('#status-message')).toHaveText('Streaming...');

    // Switch between English voices mid-playback: stops stream AND preserves Story.
    await page.selectOption('#voice-select', 'bf_emma');
    await expect(page.locator('#status-message')).toHaveText('Stopped');
    await expect(page.locator('#tts-text-input')).toHaveValue(/The lighthouse keeper climbed the spiral stairs/);
    await expect(page.locator('#tts-samples button[data-sample="story"]')).toHaveClass(/\bactive\b/);
    const cancelCount = await page.evaluate(
      () => (window as any).__ttsWorkerMessages.filter((m: any) => m.type === 'CANCEL').length
    );
    expect(cancelCount).toBeGreaterThanOrEqual(1);

    // Switch to Spanish voice: switches to Spanish sample and clears English chip active state.
    await page.selectOption('#voice-select', 'ef_dora');
    await expect(page.locator('#tts-text-input')).toHaveValue(/Bienvenido a MediaPipe/);
    await expect(page.locator('#tts-samples .sample-chip.active')).toHaveCount(0);

    // Switch back to English voice: restores Story preset.
    await page.selectOption('#voice-select', 'bf_emma');
    await expect(page.locator('#tts-text-input')).toHaveValue(/The lighthouse keeper climbed the spiral stairs/);
    await expect(page.locator('#tts-samples button[data-sample="story"]')).toHaveClass(/\bactive\b/);

    // Re-init on speaking rate change passes the currently selected voiceId.
    const speakingRateSlider = page.locator('#speaking-rate');
    await speakingRateSlider.fill('1.2');
    await speakingRateSlider.dispatchEvent('change');
    const reinitMsg = await page.evaluate(() =>
      (window as any).__ttsWorkerMessages.filter((m: any) => m.type === 'INIT').at(-1)
    );
    expect(reinitMsg?.voiceId).toBe('bf_emma');

    await page.evaluate(() => {
      (window as any).__ttsWorkerInstance.onmessage({
        data: {
          type: 'INIT_DONE',
          modelName: 'kokoro_82m',
          voices: [
            { voiceId: 'af_heart', displayName: 'Heart', locale: 'en-US', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'bf_emma', displayName: 'Emma', locale: 'en-GB', sampleRateHz: 24000, isDownloaded: true },
            { voiceId: 'ef_dora', displayName: 'Dora', locale: 'es-ES', sampleRateHz: 24000, isDownloaded: true },
          ],
        },
      });
    });

    // Mid-playback textarea edit refreshes transcript upon Stop.
    await page.click('#tts-stream-btn');
    await page.locator('#tts-text-input').fill('Edited while streaming.');
    await expect(page.locator('#tts-transcript')).toContainText('The lighthouse keeper');
    await page.click('#tts-stop-btn');
    await expect(page.locator('#tts-transcript')).toHaveText('Edited while streaming.');
  });
});
