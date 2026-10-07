import { PoseLandmarker, PoseLandmarkerResult, DrawingUtils } from '@mediapipe/tasks-vision';
import { BaseVisionTask } from '../components/base-vision-task';
// @ts-ignore
import template from '../templates/pose-landmarker.html?raw';
// @ts-ignore

class PoseLandmarkerTask extends BaseVisionTask {
  private drawingUtils: DrawingUtils | undefined;

  private minPoseDetectionConfidence = 0.5;
  private minPosePresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;
  private numPoses = 1;
  private outputSegmentationMasks = false;

  protected override onInitializeUI() {
    // Sliders
    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener('input', () => {
          const val = parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider('num-poses', (val) => {
      this.numPoses = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', numPoses: this.numPoses });
      this.triggerRedetection();
    });

    const outputSegmentationMasksInput = document.getElementById('output-segmentation-masks') as HTMLInputElement;
    if (outputSegmentationMasksInput) {
      outputSegmentationMasksInput.addEventListener('change', () => {
        this.outputSegmentationMasks = outputSegmentationMasksInput.checked;
        this.worker?.postMessage({ type: 'SET_OPTIONS', outputSegmentationMasks: this.outputSegmentationMasks });
        this.triggerRedetection();
      });
    }

    setupSlider('min-pose-detection-confidence', (val) => {
      this.minPoseDetectionConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minPoseDetectionConfidence: this.minPoseDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-pose-presence-confidence', (val) => {
      this.minPosePresenceConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minPosePresenceConfidence: this.minPosePresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-tracking-confidence', (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    // Custom model options for Pose Landmarker
    this.models = {
      pose_landmarker_lite:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      pose_landmarker_full:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
      pose_landmarker_heavy:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    };

    // Override the model selector config for pose specifically
    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'Pose Landmarker (Lite)', value: 'pose_landmarker_lite', isDefault: true },
        { label: 'Pose Landmarker (Full)', value: 'pose_landmarker_full' },
        { label: 'Pose Landmarker (Heavy)', value: 'pose_landmarker_heavy' },
      ]);
    }
  }

  private triggerRedetection() {
    if (this.runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src) {
        this.detectImage(testImage);
      }
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      minPoseDetectionConfidence: this.minPoseDetectionConfidence,
      minPosePresenceConfidence: this.minPosePresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
      numPoses: this.numPoses,
      outputSegmentationMasks: this.outputSegmentationMasks,
    };
  }

  protected override displayImageResult(result: PoseLandmarkerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.beginPath();
    ctx.rect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx); // re-initialize w/ context

      for (const landmark of result.landmarks) {
        this.drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
        });
        this.drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }
    }
  }

  protected override displayVideoResult(result: PoseLandmarkerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.save();
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    this.canvasCtx.beginPath();
    this.canvasCtx.rect(0, 0, this.canvasElement.width, this.canvasElement.height);
    this.canvasCtx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(this.canvasCtx);
      else this.drawingUtils = new DrawingUtils(this.canvasCtx);

      for (const landmark of result.landmarks) {
        this.drawingUtils.drawLandmarks(landmark, {
          radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
        });
        this.drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
      }
    }
    this.canvasCtx.restore();
  }
}

// Keep singleton instance reference to support module-level cleanup from router if needed
let activeTask: PoseLandmarkerTask | null = null;

export async function setupPoseLandmarker(container: HTMLElement) {
  activeTask = new PoseLandmarkerTask({
    container,
    template,
    defaultModelName: 'pose_landmarker_lite',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    models: {
      pose_landmarker_full:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
      pose_landmarker_heavy:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    },
    workerFactory: () =>
      new Worker(new URL('../workers/pose-landmarker.worker.ts', import.meta.url), { type: 'module' }),
  });

  await activeTask.initialize();
}

export function cleanupPoseLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
