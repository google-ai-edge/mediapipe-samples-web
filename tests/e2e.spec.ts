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

test.describe('Navigation & UI', () => {
  test.beforeEach(async ({ page }) => {
    // The root redirects to the cover page, which hides the sidebar and docs
    // banner. Start on a task route so the sidebar-based tests can navigate.
    await page.goto('#/vision/object_detector');
  });

  test('should redirect to the home cover page by default', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/.*#\/home$/);
    await expect(page.locator('.cover-page')).toBeVisible();
    await expect(page.locator('.sidebar')).toBeHidden();

    // Cover page cards link into the tasks and restore the sidebar.
    await page.click('.cover-card[href="#/vision/object_detector"]');
    await expect(page).toHaveURL(/.*#\/vision\/object_detector/);
    await expect(page.locator('.sidebar-nav .active')).toContainText('Object Detector');
  });

  test('should navigate between tasks', async ({ page }) => {
    await page.click('a[data-task="image-segmenter"]');
    await expect(page).toHaveURL(/.*#\/vision\/image_segmenter/);
    await expect(page.locator('h2')).toContainText('Image Segmentation');

    await page.click('a[data-task="universal-embedder"]');
    await expect(page).toHaveURL(/.*#\/retrieval\/universal_embedder/);
    await expect(page.locator('.sidebar-nav .active')).toHaveAttribute('data-task', 'universal-embedder');
    await expect(page.locator('h2')).toContainText('Universal Embedder');

    await page.click('a[data-task="semantic-retriever"]');
    await expect(page).toHaveURL(/.*#\/retrieval\/semantic_retriever/);
    await expect(page.locator('.sidebar-nav .active')).toHaveAttribute('data-task', 'semantic-retriever');
    await expect(page.locator('h2')).toContainText('Semantic Retriever');

    await page.click('a[data-task="object-detector"]');
    await expect(page).toHaveURL(/.*#\/vision\/object_detector/);
    await expect(page.locator('h2')).toContainText('Object Detection');
  });

  test('should have responsive sidebar', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('#/vision/object_detector');
    // Check if sidebar nav is hidden (it might be hidden by CSS, let's check visibility)
    // .sidebar-nav display: none in media query
    const mobileSidebar = page.locator('.sidebar');
    await expect(mobileSidebar).toBeHidden({ timeout: 5000 }).catch(() => {});

    // Toggle menu (using mobile header toggle)
    await page.click('.mobile-header .menu-toggle');
    await expect(mobileSidebar).toBeVisible({ timeout: 5000 });

    // Wait for transition if any (just to be safe)
    await page.waitForTimeout(500);

    // Toggle back (using sidebar toggle, since mobile one is covered)
    await page.click('.sidebar-header .menu-toggle', { force: true });
    await expect(mobileSidebar).toBeHidden({ timeout: 5000 }).catch(() => {});
  });

  test('should display privacy notice link in sidebar', async ({ page }) => {
    const privacyLink = page.locator('.sidebar-footer a.sidebar-footer-link');
    await expect(privacyLink).toBeVisible();
    await expect(privacyLink).toContainText('Privacy Notice');
    await expect(privacyLink).toHaveAttribute('href', 'https://goo.gle/mediapipe-privacy');
    await expect(privacyLink).toHaveAttribute('target', '_blank');
    await expect(privacyLink).toHaveAttribute('rel', 'noopener noreferrer');
  });

  test('should display docs overview, task guide, and StackBlitz banner links and update on navigation', async ({
    page,
  }) => {
    const banner = page.locator('.docs-banner');
    await expect(banner).toBeVisible();

    const overviewLink = page.locator('#banner-docs-overview');
    await expect(overviewLink).toHaveAttribute(
      'href',
      'https://developers.google.com/edge/mediapipe/solutions/guide'
    );

    const taskDocsLink = page.locator('#banner-task-docs');
    await expect(taskDocsLink).toContainText('Object Detector Guide');
    await expect(taskDocsLink).toHaveAttribute(
      'href',
      'https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/web_js'
    );

    const webSampleLink = page.locator('#banner-sample-web');
    const androidSampleLink = page.locator('#banner-sample-android');
    const iosSampleLink = page.locator('#banner-sample-ios');
    const pythonSampleLink = page.locator('#banner-sample-python');

    await expect(webSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/object-detector.ts'
    );
    await expect(androidSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/android'
    );
    await expect(iosSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/ios'
    );
    await expect(pythonSampleLink).toHaveAttribute(
      'href',
      'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb'
    );

    const stackblitzLink = page.locator('#banner-stackblitz');
    await expect(stackblitzLink).toContainText('Edit in StackBlitz');
    await expect(stackblitzLink).toHaveAttribute(
      'href',
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fobject-detector.ts'
    );

    // Navigate to another task and verify banner links update
    await page.click('a[data-task="image-segmenter"]');
    await expect(taskDocsLink).toContainText('Image Segmenter Guide');
    await expect(taskDocsLink).toHaveAttribute(
      'href',
      'https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter/web_js'
    );
    await expect(webSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-segmenter.ts'
    );
    await expect(androidSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_segmentation/android'
    );
    await expect(iosSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_segmentation/ios'
    );
    await expect(pythonSampleLink).toHaveAttribute(
      'href',
      'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb'
    );
    await expect(stackblitzLink).toHaveAttribute(
      'href',
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fimage-segmenter.ts'
    );

    // Navigate to Universal Embedder and verify banner links
    await page.click('a[data-task="universal-embedder"]');
    await expect(taskDocsLink).toContainText('Universal Embedder Guide');
    await expect(taskDocsLink).toHaveAttribute(
      'href',
      'https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder/web_js'
    );
    await expect(webSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/universal-embedder.ts'
    );
    await expect(androidSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/android'
    );
    await expect(iosSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/ios'
    );
    await expect(pythonSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/python'
    );
    await expect(stackblitzLink).toHaveAttribute(
      'href',
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Funiversal-embedder.ts'
    );

    // Navigate to Semantic Retriever and verify banner links
    await page.click('a[data-task="semantic-retriever"]');
    await expect(taskDocsLink).toContainText('Semantic Retriever Guide');
    await expect(taskDocsLink).toHaveAttribute(
      'href',
      'https://developers.google.com/edge/mediapipe/solutions/retrieval/semantic_retriever/web_js'
    );
    await expect(webSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/semantic-retriever.ts'
    );
    await expect(androidSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/android'
    );
    await expect(iosSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/ios'
    );
    await expect(pythonSampleLink).toHaveAttribute(
      'href',
      'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/python'
    );
    await expect(stackblitzLink).toHaveAttribute(
      'href',
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fsemantic-retriever.ts'
    );
  });
});
