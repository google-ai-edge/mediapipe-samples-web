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
      <button class="menu-toggle material-icons" style="margin-right: 12px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu_open</button>
      <div class="sidebar-logo-text">
        <span class="material-icons" style="color: var(--primary); font-size: 32px;">analytics</span>
        <span style="font-weight: 600; color: var(--text-main);">MediaPipe Tasks</span>
      </div>
    </div>
    <nav class="sidebar-nav">
      <div class="category-header">Vision</div>
      <ul>
        <li><a href="#/vision/face_detector" class="nav-button" data-task="face-detector">Face Detection</a></li>
        <li><a href="#/vision/face_landmarker" class="nav-button" data-task="face-landmarker">Face Landmarker</a></li>
        <li><a href="#/vision/gesture_recognizer" class="nav-button" data-task="gesture-recognizer">Gesture Recognizer</a></li>
        <li><a href="#/vision/hand_landmarker" class="nav-button" data-task="hand-landmarker">Hand Landmarker</a></li>
        <li><a href="#/vision/holistic_landmarker" class="nav-button" data-task="holistic-landmarker">Holistic Landmarker</a></li>
        <li><a href="#/vision/image_classifier" class="nav-button" data-task="image-classifier">Image Classifier</a></li>
        <li><a href="#/vision/image_embedder" class="nav-button" data-task="image-embedder">Image Embedding</a></li>
        <li><a href="#/vision/image_segmenter" class="nav-button" data-task="image-segmenter">Image Segmentation</a></li>
        <li><a href="#/vision/interactive_segmenter" class="nav-button" data-task="interactive-segmenter">Interactive Segmenter</a></li>
        <li><a href="#/vision/object_detector" class="nav-button" data-task="object-detector">Object Detection</a></li>
        <li><a href="#/vision/pose_landmarker" class="nav-button" data-task="pose-landmarker">Pose Landmarker</a></li>
      </ul>

      <div class="category-header">Audio</div>
      <ul>
        <li><a href="#/audio/audio_classifier" class="nav-button" data-task="audio-classifier">Audio Classifier</a></li>
      </ul>

      <div class="category-header">Text</div>
      <ul>
        <li><a href="#/text/language_detector" class="nav-button" data-task="language-detector">Language Detection</a></li>
        <li><a href="#/text/text_classifier" class="nav-button" data-task="text-classifier">Text Classification</a></li>
        <li><a href="#/text/text_embedder" class="nav-button" data-task="text-embedder">Text Embedding</a></li>
      </ul>
    </nav>
  `;
}
