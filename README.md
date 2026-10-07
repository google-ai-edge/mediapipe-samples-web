# MediaPipe Tasks Demo

Experience the power of on-device Machine Learning with MediaPipe Tasks. This
demo showcases real-time audio, vision and text demos using CPU and GPU
acceleration on Web.

Play with the demos here: https://google-ai-edge.github.io/mediapipe-samples-web/

## Supported Tasks

### Vision
- **Face Detection**: Detect faces in images and video.
- **Face Landmarker**: Detect 478 3D face landmarks.
- **Gesture Recognizer**: Recognize hand gestures (e.g., "Thumbs Up", "Peace").
- **Hand Landmarker**: Track 21 3D hand landmarks for both hands.
- **Holistic Landmarker**: Simultaneous tracking of face, hands, and pose.
- **Image Classifier**: Categorize images into defined classes.
- **Image Embedding**: Compare image similarity using vector embeddings.
- **Image Segmentation**: Pixel-perfect masking of objects.
- **Interactive Segmentation**: Click to segment specific objects of interest.
- **Object Detection**: Detect multiple objects with bounding boxes.
- **Pose Landmarker**: Full-body pose tracking with segmentation support.

### Audio
- **Audio Classifier**: Classify ambient sounds in real-time.

### Decision
- **Decision Maker**: Make decisions from text or JSON input using logic rules.
- **Decision Maker - Dino Game**: Watch the Decision Maker play a runner game by choosing to jump, duck or wait.

### Retrieval
- **Semantic Retriever**: Index content into a vector store and retrieve it with natural-language queries.
- **Universal Embedder**: Embed text and images into a shared vector space across modalities.

### Text
- **Language Detection**: Identify the language of input text.
- **Text Classification**: Classify text sentiment (positive/negative).
- **Text Embedding**: Semantic similarity comparison for text.

## Quick Start

1.  **Clone the repository**
2.  **Install dependencies**: `pnpm install`
3.  **Run the app**: `pnpm dev`
4.  **Open browser**: Navigate to the local URL (typically `http://localhost:5173`)

## Testing

We ensure robustness with a comprehensive End-to-End (E2E) test suite using Playwright.

```bash
pnpm test
```
