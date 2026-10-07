import './cover-page.css';

export function setupCoverPage(container: HTMLElement) {
  container.innerHTML = `
    <div class="cover-page">
      <div class="cover-hero">
        <div class="hero-icon"><span class="material-icons">analytics</span></div>
        <h1 class="cover-title">MediaPipe Tasks</h1>
        <p class="cover-subtitle">Explore on-device machine learning capabilities directly in your browser.</p>
      </div>

      <div class="cover-content">
        <div class="cover-row-3">
          <!-- Audio -->
          <div class="cover-category domain-audio">
            <div class="category-header-wrap">
              <span class="material-icons category-icon">graphic_eq</span>
              <h2>Audio</h2>
            </div>
            <div class="cover-grid">
              <a href="#/audio/audio_classifier" class="cover-card">
                <span class="material-icons card-icon">hearing</span>
                <div class="card-content">
                  <h3>Audio Classifier</h3>
                  <p>Classify audio clips into categories.</p>
                </div>
              </a>
            </div>
          </div>

          <!-- Decision -->
          <div class="cover-category domain-decision">
            <div class="category-header-wrap">
              <span class="material-icons category-icon">psychology</span>
              <h2>Decision</h2>
            </div>
            <div class="cover-grid">
              <a href="#/decision/decision_maker" class="cover-card">
                <span class="material-icons card-icon">rule</span>
                <div class="card-content">
                  <h3>Decision Maker</h3>
                  <p>Make decisions based on logic rules.</p>
                </div>
              </a>
              <a href="#/decision/dino_game" class="cover-card">
                <span class="material-icons card-icon">sports_esports</span>
                <div class="card-content">
                  <h3>Decision Maker - Dino Game</h3>
                  <p>Watch the Decision Maker play a runner game.</p>
                </div>
              </a>
            </div>
          </div>

          <!-- Retrieval -->
          <div class="cover-category domain-retrieval">
            <div class="category-header-wrap">
              <span class="material-icons category-icon">travel_explore</span>
              <h2>Retrieval</h2>
            </div>
            <div class="cover-grid">
              <a href="#/retrieval/semantic_retriever" class="cover-card">
                <span class="material-icons card-icon">manage_search</span>
                <div class="card-content">
                  <h3>Semantic Retriever</h3>
                  <p>Retrieve relevant content semantically.</p>
                </div>
              </a>
              <a href="#/retrieval/universal_embedder" class="cover-card">
                <span class="material-icons card-icon">all_inclusive</span>
                <div class="card-content">
                  <h3>Universal Embedder</h3>
                  <p>Embed across multiple modalities.</p>
                </div>
              </a>
            </div>
          </div>
        </div>


        <!-- Text -->
        <div class="cover-category domain-text">
          <div class="category-header-wrap">
            <span class="material-icons category-icon">notes</span>
            <h2>Text</h2>
          </div>
          <div class="cover-grid">
            <a href="#/text/language_detector" class="cover-card">
              <span class="material-icons card-icon">language</span>
              <div class="card-content">
                <h3>Language Detector</h3>
                <p>Detect the language of a text.</p>
              </div>
            </a>
            <a href="#/text/text_classifier" class="cover-card">
              <span class="material-icons card-icon">text_format</span>
              <div class="card-content">
                <h3>Text Classifier</h3>
                <p>Classify text into categories.</p>
              </div>
            </a>
            <a href="#/text/text_embedder" class="cover-card">
              <span class="material-icons card-icon">abc</span>
              <div class="card-content">
                <h3>Text Embedder</h3>
                <p>Extract feature vectors from text.</p>
              </div>
            </a>
          </div>
        </div>

        <!-- Vision -->
        <div class="cover-category domain-vision">
          <div class="category-header-wrap">
            <span class="material-icons category-icon">visibility</span>
            <h2>Vision</h2>
          </div>
          <div class="cover-grid">
            <a href="#/vision/face_detector" class="cover-card">
              <span class="material-icons card-icon">face</span>
              <div class="card-content">
                <h3>Face Detector</h3>
                <p>Detect faces in images and video.</p>
              </div>
            </a>
            <a href="#/vision/face_landmarker" class="cover-card">
              <span class="material-icons card-icon">face_retouching_natural</span>
              <div class="card-content">
                <h3>Face Landmarker</h3>
                <p>Detect facial landmarks.</p>
              </div>
            </a>
            <a href="#/vision/gesture_recognizer" class="cover-card">
              <span class="material-icons card-icon">sign_language</span>
              <div class="card-content">
                <h3>Gesture Recognizer</h3>
                <p>Recognize hand gestures.</p>
              </div>
            </a>
            <a href="#/vision/hand_landmarker" class="cover-card">
              <span class="material-icons card-icon">back_hand</span>
              <div class="card-content">
                <h3>Hand Landmarker</h3>
                <p>Detect hand landmarks.</p>
              </div>
            </a>
            <a href="#/vision/holistic_landmarker" class="cover-card">
              <span class="material-icons card-icon">accessibility_new</span>
              <div class="card-content">
                <h3>Holistic Landmarker</h3>
                <p>Detect face, hand, and pose landmarks.</p>
              </div>
            </a>
            <a href="#/vision/image_classifier" class="cover-card">
              <span class="material-icons card-icon">image</span>
              <div class="card-content">
                <h3>Image Classifier</h3>
                <p>Classify images into categories.</p>
              </div>
            </a>
            <a href="#/vision/image_embedder" class="cover-card">
              <span class="material-icons card-icon">center_focus_strong</span>
              <div class="card-content">
                <h3>Image Embedder</h3>
                <p>Extract feature vectors from images.</p>
              </div>
            </a>
            <a href="#/vision/image_segmenter" class="cover-card">
              <span class="material-icons card-icon">layers</span>
              <div class="card-content">
                <h3>Image Segmenter</h3>
                <p>Segment objects in images.</p>
              </div>
            </a>
            <a href="#/vision/interactive_segmenter" class="cover-card">
              <span class="material-icons card-icon">touch_app</span>
              <div class="card-content">
                <h3>Interactive Segmenter</h3>
                <p>Select and segment objects interactively.</p>
              </div>
            </a>
            <a href="#/vision/object_detector" class="cover-card">
              <span class="material-icons card-icon">crop_free</span>
              <div class="card-content">
                <h3>Object Detector</h3>
                <p>Detect objects in images.</p>
              </div>
            </a>
            <a href="#/vision/pose_landmarker" class="cover-card">
              <span class="material-icons card-icon">accessibility</span>
              <div class="card-content">
                <h3>Pose Landmarker</h3>
                <p>Detect human poses.</p>
              </div>
            </a>
          </div>
        </div>
      </div>
    </div>
  `;
}

export function cleanupCoverPage() {
  // Nothing to cleanup for cover page
}
