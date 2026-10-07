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

import './banner.css';

export const MEDIAPIPE_DOCS_OVERVIEW_URL = 'https://developers.google.com/edge/mediapipe/solutions/guide';

export interface TaskPlatformSamples {
  web: string;
  android: string;
  ios: string;
  python: string;
}

export interface TaskBannerLinks {
  label: string;
  docsUrl: string;
  sampleCodeUrl: string;
  samples: TaskPlatformSamples;
  stackblitzUrl: string;
}

export const TASK_BANNER_LINKS: Record<string, TaskBannerLinks> = {
  '/vision/object_detector': {
    label: 'Object Detector',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/object-detector.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/object-detector.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/object_detection/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fobject-detector.ts',
  },
  '/vision/face_detector': {
    label: 'Face Detector',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/face-detector.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/face-detector.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_detector/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_detector/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fface-detector.ts',
  },
  '/vision/face_landmarker': {
    label: 'Face Landmarker',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/face-landmarker.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/face-landmarker.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fface-landmarker.ts',
  },
  '/vision/hand_landmarker': {
    label: 'Hand Landmarker',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/hand-landmarker.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/hand_landmarker/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fhand-landmarker.ts',
  },
  '/vision/pose_landmarker': {
    label: 'Pose Landmarker',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/pose-landmarker.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/pose-landmarker.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/pose_landmarker/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/pose_landmarker/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fpose-landmarker.ts',
  },
  '/vision/holistic_landmarker': {
    label: 'Holistic Landmarker',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/holistic-landmarker.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/holistic-landmarker.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/holistic_landmarker/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/holistic_landmarker/ios',
      python: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/holistic_landmarker/python',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fholistic-landmarker.ts',
  },
  '/vision/image_classifier': {
    label: 'Image Classifier',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/image_classifier/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-classifier.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-classifier.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_classification/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_classification/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_classification/python/image_classifier.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fimage-classifier.ts',
  },
  '/vision/gesture_recognizer': {
    label: 'Gesture Recognizer',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/gesture-recognizer.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/gesture-recognizer.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/gesture_recognizer/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/gesture_recognizer/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fgesture-recognizer.ts',
  },
  '/vision/interactive_segmenter': {
    label: 'Interactive Segmenter',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/interactive_segmenter/web_js',
    sampleCodeUrl:
      'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/interactive-segmenter.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/interactive-segmenter.ts',
      android:
        'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/interactive_segmentation/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/interactive_segmentation/ios',
      python:
        'https://colab.research.google.com/github/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Finteractive-segmenter.ts',
  },
  '/vision/image_segmenter': {
    label: 'Image Segmenter',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-segmenter.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-segmenter.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_segmentation/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_segmentation/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_segmentation/python/image_segmentation.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fimage-segmenter.ts',
  },
  '/vision/image_embedder': {
    label: 'Image Embedder',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/vision/image_embedder/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-embedder.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/image-embedder.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_embedder/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/image_embedder/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fimage-embedder.ts',
  },
  '/audio/audio_classifier': {
    label: 'Audio Classifier',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/audio/audio_classifier/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/audio-classifier.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/audio-classifier.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/audio_classifier/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/audio_classifier/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/audio_classifier/python/audio_classification.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Faudio-classifier.ts',
  },
  '/text/text_classifier': {
    label: 'Text Classifier',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/text/text_classifier/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/text-classifier.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/text-classifier.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_classification/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_classification/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/text_classification/python/text_classifier.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Ftext-classifier.ts',
  },
  '/text/language_detector': {
    label: 'Language Detector',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/text/language_detector/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/language-detector.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/language-detector.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/language_detector/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/language_detector/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/language_detector/python/%5BMediaPipe_Python_Tasks%5D_Language_Detector.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Flanguage-detector.ts',
  },
  '/text/text_embedder': {
    label: 'Text Embedder',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/text/text_embedder/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/text-embedder.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/text-embedder.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_embedder/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/text_embedder/ios',
      python:
        'https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/text_embedder/python/text_embedder.ipynb',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Ftext-embedder.ts',
  },
  '/decision/decision_maker': {
    label: 'Decision Maker',
    docsUrl: MEDIAPIPE_DOCS_OVERVIEW_URL,
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/decision-maker.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/decision-maker.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/decision/android',
      ios: MEDIAPIPE_DOCS_OVERVIEW_URL,
      python: MEDIAPIPE_DOCS_OVERVIEW_URL,
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fdecision-maker.ts',
  },
  '/decision/dino_game': {
    label: 'Decision Maker - Dino Game',
    docsUrl: MEDIAPIPE_DOCS_OVERVIEW_URL,
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/dino-game.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/dino-game.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/decision/android',
      ios: MEDIAPIPE_DOCS_OVERVIEW_URL,
      python: MEDIAPIPE_DOCS_OVERVIEW_URL,
    },
    stackblitzUrl: 'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fdino-game.ts',
  },
  '/retrieval/universal_embedder': {
    label: 'Universal Embedder',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/universal-embedder.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/universal-embedder.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/ios',
      python: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/universal_embedder/python',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Funiversal-embedder.ts',
  },
  '/retrieval/semantic_retriever': {
    label: 'Semantic Retriever',
    docsUrl: 'https://developers.google.com/edge/mediapipe/solutions/retrieval/semantic_retriever/web_js',
    sampleCodeUrl: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/semantic-retriever.ts',
    samples: {
      web: 'https://github.com/google-ai-edge/mediapipe-samples-web/blob/main/src/tasks/semantic-retriever.ts',
      android: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/android',
      ios: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/ios',
      python: 'https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/semantic_retriever/python',
    },
    stackblitzUrl:
      'https://stackblitz.com/github/google-ai-edge/mediapipe-samples-web?file=src%2Ftasks%2Fsemantic-retriever.ts',
  },
};

export function renderBanner(container: HTMLElement) {
  const defaultLinks = TASK_BANNER_LINKS['/vision/object_detector'];
  container.innerHTML = `
    <div class="docs-banner" role="region" aria-label="Documentation and code links">
      <div class="docs-banner-left">
        <a
          id="banner-docs-overview"
          href="${MEDIAPIPE_DOCS_OVERVIEW_URL}"
          target="_blank"
          rel="noopener noreferrer"
          class="docs-banner-link"
        >
          <span class="material-icons docs-banner-icon">menu_book</span>
          <span>MediaPipe Docs</span>
          <span class="material-icons docs-banner-external">open_in_new</span>
        </a>
        <span class="docs-banner-separator" aria-hidden="true">/</span>
        <a
          id="banner-task-docs"
          href="${defaultLinks.docsUrl}"
          target="_blank"
          rel="noopener noreferrer"
          class="docs-banner-link task-guide-link"
        >
          <span id="banner-task-docs-label">${defaultLinks.label} Guide</span>
          <span class="material-icons docs-banner-external">open_in_new</span>
        </a>
      </div>
      <div class="docs-banner-actions">
        <div class="docs-banner-samples-group" role="group" aria-label="Platform code samples">
          <span class="docs-banner-samples-label">
            <span class="material-icons">code</span>
            <span>Samples:</span>
          </span>
          <a
            id="banner-sample-web"
            href="${defaultLinks.samples.web}"
            target="_blank"
            rel="noopener noreferrer"
            class="docs-banner-sample-link"
            title="Web sample code"
          >Web</a>
          <a
            id="banner-sample-android"
            href="${defaultLinks.samples.android}"
            target="_blank"
            rel="noopener noreferrer"
            class="docs-banner-sample-link"
            title="Android sample code"
          >Android</a>
          <a
            id="banner-sample-ios"
            href="${defaultLinks.samples.ios}"
            target="_blank"
            rel="noopener noreferrer"
            class="docs-banner-sample-link"
            title="iOS sample code"
          >iOS</a>
          <a
            id="banner-sample-python"
            href="${defaultLinks.samples.python}"
            target="_blank"
            rel="noopener noreferrer"
            class="docs-banner-sample-link"
            title="Python sample notebook"
          >Python</a>
        </div>
        <a
          id="banner-stackblitz"
          href="${defaultLinks.stackblitzUrl}"
          target="_blank"
          rel="noopener noreferrer"
          class="docs-banner-chip stackblitz-chip"
        >
          <span class="material-icons">bolt</span>
          <span>Edit in StackBlitz</span>
        </a>
      </div>
    </div>
  `;
}

export function updateBanner(routeKey: string) {
  const container = document.getElementById('docs-banner-container');
  if (container) {
    if (routeKey === '/home') {
      container.style.display = 'none';
      return;
    } else {
      container.style.display = 'block';
    }
  }

  const links = TASK_BANNER_LINKS[routeKey] || TASK_BANNER_LINKS['/vision/object_detector'];

  const taskDocsLink = document.getElementById('banner-task-docs') as HTMLAnchorElement | null;
  const taskDocsLabel = document.getElementById('banner-task-docs-label');
  const webSampleLink = document.getElementById('banner-sample-web') as HTMLAnchorElement | null;
  const androidSampleLink = document.getElementById('banner-sample-android') as HTMLAnchorElement | null;
  const iosSampleLink = document.getElementById('banner-sample-ios') as HTMLAnchorElement | null;
  const pythonSampleLink = document.getElementById('banner-sample-python') as HTMLAnchorElement | null;
  const stackblitzLink = document.getElementById('banner-stackblitz') as HTMLAnchorElement | null;

  if (taskDocsLink) {
    taskDocsLink.href = links.docsUrl;
  }
  if (taskDocsLabel) {
    taskDocsLabel.textContent = `${links.label} Guide`;
  }
  if (webSampleLink) {
    webSampleLink.href = links.samples.web;
  }
  if (androidSampleLink) {
    androidSampleLink.href = links.samples.android;
  }
  if (iosSampleLink) {
    iosSampleLink.href = links.samples.ios;
  }
  if (pythonSampleLink) {
    pythonSampleLink.href = links.samples.python;
  }
  if (stackblitzLink) {
    stackblitzLink.href = links.stackblitzUrl;
  }
}
