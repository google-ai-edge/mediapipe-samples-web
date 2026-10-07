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

import './app_clean.css';
import { setupObjectDetector, cleanupObjectDetector } from './tasks/object-detector';
import { setupImageSegmenter, cleanupImageSegmenter } from './tasks/image-segmenter';
import { setupAudioClassifier, cleanupAudioClassifier } from './tasks/audio-classifier';
import { setupTextClassifier, cleanupTextClassifier } from './tasks/text-classifier';
import { setupTextEmbedder, cleanupTextEmbedder } from './tasks/text-embedder';
import { setupFaceDetector, cleanupFaceDetector } from './tasks/face-detector';
import { setupFaceLandmarker, cleanupFaceLandmarker } from './tasks/face-landmarker';
import { setupHandLandmarker, cleanupHandLandmarker } from './tasks/hand-landmarker';
import { setupPoseLandmarker, cleanupPoseLandmarker } from './tasks/pose-landmarker';
import { setupGestureRecognizer, cleanupGestureRecognizer } from './tasks/gesture-recognizer';
import { setupLanguageDetector, cleanupLanguageDetector } from './tasks/language-detector';
import { setupImageEmbedder, cleanupImageEmbedder } from './tasks/image-embedder';
import { setupInteractiveSegmenter, cleanupInteractiveSegmenter } from './tasks/interactive-segmenter';
import { setupHolisticLandmarker, cleanupHolisticLandmarker } from './tasks/holistic-landmarker';
import { setupImageClassifier, cleanupImageClassifier } from './tasks/image-classifier';
import { setupDecisionMaker, cleanupDecisionMaker } from './tasks/decision-maker';
import { cleanupUniversalEmbedder, setupUniversalEmbedder } from './tasks/universal-embedder.ts';
import { cleanupSemanticRetriever, setupSemanticRetriever } from './tasks/semantic-retriever.ts';

import { renderSidebar } from './ui/sidebar';
import { renderMobileNav } from './ui/mobile-nav';
import { renderBanner, updateBanner } from './ui/banner';
import { setupCoverPage, cleanupCoverPage } from './ui/cover-page';

const app = document.querySelector<HTMLDivElement>('#app')!;

// 1. Setup App Shell
app.innerHTML = `
  <div class="app-container">
    <aside class="sidebar"></aside>
    <div class="sidebar-backdrop" aria-hidden="true"></div>
    <div class="mobile-header">
       <button class="menu-toggle material-icons" aria-label="Open navigation" style="margin-right: 24px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu</button>
       <div id="mobile-nav-container" style="display: flex; align-items: center; flex-grow: 1; min-width: 0;"></div>
    </div>
    <main class="main-content">
      <div id="docs-banner-container"></div>
      <div id="task-stage" class="task-stage"></div>
    </main>
  </div>
`;

// 2. Render Global Components
const sidebar = app.querySelector('.sidebar') as HTMLElement;
renderSidebar(sidebar);

const mobileNavContainer = app.querySelector('#mobile-nav-container') as HTMLElement;
renderMobileNav(mobileNavContainer);

const bannerContainer = app.querySelector('#docs-banner-container') as HTMLElement;
renderBanner(bannerContainer);

// 3. Setup Navigation Logic (off-canvas drawer on small screens)
const sidebarBackdrop = app.querySelector('.sidebar-backdrop') as HTMLElement;

function setSidebarOpen(open: boolean) {
  sidebar.classList.toggle('open', open);
  sidebarBackdrop.classList.toggle('visible', open);
  document.body.classList.toggle('sidebar-open', open);
}

const menuToggles = app.querySelectorAll('.menu-toggle');
menuToggles.forEach((toggle) => {
  toggle.addEventListener('click', () => {
    setSidebarOpen(!sidebar.classList.contains('open'));
  });
});

sidebarBackdrop.addEventListener('click', () => setSidebarOpen(false));

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && sidebar.classList.contains('open')) setSidebarOpen(false);
});

// Close sidebar when a link is clicked
sidebar.addEventListener('click', (e) => {
  if ((e.target as HTMLElement).closest('a')) {
    setSidebarOpen(false);
  }
});

const mainContent = app.querySelector('#task-stage') as HTMLElement;

// 4. Router Setup
const routes = {
  '/home': { setup: setupCoverPage, cleanup: cleanupCoverPage, label: 'Home' },
  '/vision/object_detector': {
    setup: setupObjectDetector,
    cleanup: cleanupObjectDetector,
    label: 'Object Detector',
  },
  '/vision/face_detector': { setup: setupFaceDetector, cleanup: cleanupFaceDetector, label: 'Face Detector' },
  '/vision/face_landmarker': { setup: setupFaceLandmarker, cleanup: cleanupFaceLandmarker, label: 'Face Landmarker' },
  '/vision/hand_landmarker': { setup: setupHandLandmarker, cleanup: cleanupHandLandmarker, label: 'Hand Landmarker' },
  '/vision/pose_landmarker': { setup: setupPoseLandmarker, cleanup: cleanupPoseLandmarker, label: 'Pose Landmarker' },
  '/vision/holistic_landmarker': {
    setup: setupHolisticLandmarker,
    cleanup: cleanupHolisticLandmarker,
    label: 'Holistic Landmarker',
  },
  '/vision/image_classifier': {
    setup: setupImageClassifier,
    cleanup: cleanupImageClassifier,
    label: 'Image Classifier',
  },
  '/vision/gesture_recognizer': {
    setup: setupGestureRecognizer,
    cleanup: cleanupGestureRecognizer,
    label: 'Gesture Recognizer',
  },
  '/vision/interactive_segmenter': {
    setup: setupInteractiveSegmenter,
    cleanup: cleanupInteractiveSegmenter,
    label: 'Interactive Segmenter',
  },
  '/vision/image_segmenter': {
    setup: setupImageSegmenter,
    cleanup: cleanupImageSegmenter,
    label: 'Image Segmenter',
  },
  '/vision/image_embedder': { setup: setupImageEmbedder, cleanup: cleanupImageEmbedder, label: 'Image Embedder' },
  '/audio/audio_classifier': {
    setup: setupAudioClassifier,
    cleanup: cleanupAudioClassifier,
    label: 'Audio Classifier',
  },
  '/text/text_classifier': {
    setup: setupTextClassifier,
    cleanup: cleanupTextClassifier,
    label: 'Text Classifier',
  },
  '/text/language_detector': {
    setup: setupLanguageDetector,
    cleanup: cleanupLanguageDetector,
    label: 'Language Detector',
  },
  '/text/text_embedder': { setup: setupTextEmbedder, cleanup: cleanupTextEmbedder, label: 'Text Embedder' },
  '/decision/decision_maker': { setup: setupDecisionMaker, cleanup: cleanupDecisionMaker, label: 'Decision Maker' },
  '/retrieval/universal_embedder': {
    setup: setupUniversalEmbedder,
    cleanup: cleanupUniversalEmbedder,
    label: 'Universal Embedder',
  },
  '/retrieval/semantic_retriever': {
    setup: setupSemanticRetriever,
    cleanup: cleanupSemanticRetriever,
    label: 'Semantic Retriever',
  },
};

let currentCleanup: (() => void) | undefined;

async function applyRoute() {
  const hash = window.location.hash.slice(1);

  // Handle root or invalid routes by defaulting to home. Changing
  // the hash fires `hashchange`, which routes again with the valid hash.
  if (!hash || !routes[hash as keyof typeof routes]) {
    window.location.hash = '/home';
    return;
  }

  const route = routes[hash as keyof typeof routes];

  // Cleanup previous task
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = undefined;
  }

  // Clear main content area only
  mainContent.innerHTML = '';

  // Setup new task
  const isHome = hash === '/home';
  sidebar.style.display = isHome ? 'none' : '';
  const mobileHeader = app.querySelector('.mobile-header') as HTMLElement;
  if (mobileHeader) mobileHeader.style.display = isHome ? 'none' : '';

  updateBanner(hash);
  await route.setup(mainContent);
  currentCleanup = route.cleanup;
  document.title = `${route.label} - MediaPipe Web Task Demo`;

  // Update active state in sidebar
  const links = sidebar.querySelectorAll('a');
  links.forEach((l) => {
    if (l.getAttribute('href') === `#${hash}`) l.classList.add('active');
    else l.classList.remove('active');
  });
}

// Route changes are applied one at a time: a task's (async) setup must finish
// before it can be cleaned up, otherwise two instances – and two workers
// loading the same model – would be left running.
let routing: Promise<void> = Promise.resolve();
function router() {
  routing = routing.then(applyRoute).catch((err) => console.error('Failed to set up route:', err));
  return routing;
}

window.addEventListener('hashchange', router);

// Initialize router immediately to handle initial load
router();

// Expose cleanup method for testing environments to prevent leaks between tests
(window as any).cleanupActiveTask = () => {
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = undefined;
  }
};
