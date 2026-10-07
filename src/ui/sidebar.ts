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

export function renderSidebar(container: HTMLElement) {
  container.innerHTML = `
    <div class="sidebar-header">
      <button class="menu-toggle material-icons" style="margin-right: 24px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu_open</button>
      <a href="#/home" class="sidebar-logo-text" style="text-decoration: none;">
        <span class="material-icons" style="color: #007f8b; font-size: 32px; margin-right: 4px;">analytics</span>
        <span style="font-weight: 600; color: var(--text-main);">MediaPipe Tasks</span>
      </a>
    </div>
    <nav class="sidebar-nav">
      <div class="category-header">Audio</div>
      <ul>
        <li><a href="#/audio/audio_classifier" class="nav-button nav-audio" data-task="audio-classifier">Audio Classifier</a></li>
      </ul>

      <div class="category-header">Decision</div>
      <ul>
        <li><a href="#/decision/decision_maker" class="nav-button nav-decision" data-task="decision-maker">Decision Maker</a></li>
      </ul>

      <div class="category-header">Retrieval</div>
      <ul>
        <li><a href="#/retrieval/semantic_retriever" class="nav-button nav-retrieval" data-task="semantic-retriever">Semantic Retriever</a></li>
        <li><a href="#/retrieval/universal_embedder" class="nav-button nav-retrieval" data-task="universal-embedder">Universal Embedder</a></li>
      </ul>

      <div class="category-header">Text</div>
      <ul>
        <li><a href="#/text/language_detector" class="nav-button nav-text" data-task="language-detector">Language Detector</a></li>
        <li><a href="#/text/text_classifier" class="nav-button nav-text" data-task="text-classifier">Text Classifier</a></li>
        <li><a href="#/text/text_embedder" class="nav-button nav-text" data-task="text-embedder">Text Embedder</a></li>
      </ul>

      <div class="category-header">Vision</div>
      <ul>
        <li><a href="#/vision/face_detector" class="nav-button nav-vision" data-task="face-detector">Face Detector</a></li>
        <li><a href="#/vision/face_landmarker" class="nav-button nav-vision" data-task="face-landmarker">Face Landmarker</a></li>
        <li><a href="#/vision/gesture_recognizer" class="nav-button nav-vision" data-task="gesture-recognizer">Gesture Recognizer</a></li>
        <li><a href="#/vision/hand_landmarker" class="nav-button nav-vision" data-task="hand-landmarker">Hand Landmarker</a></li>
        <li><a href="#/vision/holistic_landmarker" class="nav-button nav-vision" data-task="holistic-landmarker">Holistic Landmarker</a></li>
        <li><a href="#/vision/image_classifier" class="nav-button nav-vision" data-task="image-classifier">Image Classifier</a></li>
        <li><a href="#/vision/image_embedder" class="nav-button nav-vision" data-task="image-embedder">Image Embedder</a></li>
        <li><a href="#/vision/image_segmenter" class="nav-button nav-vision" data-task="image-segmenter">Image Segmenter</a></li>
        <li><a href="#/vision/interactive_segmenter" class="nav-button nav-vision" data-task="interactive-segmenter">Interactive Segmenter</a></li>
        <li><a href="#/vision/object_detector" class="nav-button nav-vision" data-task="object-detector">Object Detector</a></li>
        <li><a href="#/vision/pose_landmarker" class="nav-button nav-vision" data-task="pose-landmarker">Pose Landmarker</a></li>
      </ul>
    </nav>
    <div class="sidebar-footer">
      <a href="https://goo.gle/mediapipe-privacy" target="_blank" rel="noopener noreferrer" class="sidebar-footer-link">
        <span class="material-icons">privacy_tip</span>
        <span>Privacy Notice</span>
      </a>
    </div>
  `;
}
