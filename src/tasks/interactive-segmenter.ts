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

import { BaseVisionTask, BaseVisionTaskOptions } from '../components/base-vision-task';

// @ts-ignore
import template from '../templates/interactive-segmenter.html?raw';
// @ts-ignore

interface Point {
  x: number;
  y: number;
}
interface Stroke {
  brushMode: number;
  point: Point[];
  isCompleted: boolean;
}

class InteractiveSegmenterTask extends BaseVisionTask {
  private isFrozen = false;
  private webcamCapture!: HTMLCanvasElement;
  private webcamOverlay!: HTMLCanvasElement;
  private freezeButton!: HTMLButtonElement;
  private strokeModeSelect!: HTMLSelectElement;
  private clearStrokesBtn!: HTMLButtonElement;
  private webcamCtx!: CanvasRenderingContext2D;
  private overlayCtx!: CanvasRenderingContext2D;

  private currentStrokeMode: number = 1; // 1: Positive, 2: Negative, 3: Lasso
  private accumulatedStrokes: Stroke[] = [];
  private isPointerDown = false;
  private currentStrokePoints: Point[] = [];
  private currentMaskBitmap: ImageBitmap | null = null;

  constructor(options: BaseVisionTaskOptions) {
    super(options);
  }

  protected override onInitializeUI() {
    this.webcamCapture = document.getElementById('webcam-capture') as HTMLCanvasElement;
    this.webcamOverlay = document.getElementById('webcam-overlay') as HTMLCanvasElement;
    this.freezeButton = document.getElementById('freezeButton') as HTMLButtonElement;
    this.strokeModeSelect = document.getElementById('stroke-mode-select') as HTMLSelectElement;
    this.clearStrokesBtn = document.getElementById('clear-strokes-btn') as HTMLButtonElement;
    this.webcamCtx = this.webcamCapture.getContext('2d', { willReadFrequently: true })!;
    this.overlayCtx = this.webcamOverlay.getContext('2d', { willReadFrequently: true })!;

    this.webcamCapture.style.display = 'none';
    this.webcamOverlay.style.display = 'none';
    this.webcamOverlay.style.position = 'absolute';
    this.webcamOverlay.style.top = '0';
    this.webcamOverlay.style.left = '0';
    this.webcamOverlay.style.pointerEvents = 'none';

    if (this.freezeButton) {
      this.freezeButton.addEventListener('click', this.toggleFreeze.bind(this));
      this.freezeButton.disabled = true; // Disabled initially until webcam starts
    }

    if (this.strokeModeSelect) {
      this.strokeModeSelect.addEventListener('change', () => {
        this.currentStrokeMode = parseInt(this.strokeModeSelect.value, 10) || 1;
      });
    }

    if (this.clearStrokesBtn) {
      this.clearStrokesBtn.addEventListener('click', () => {
        this.clearStrokes();
      });
    }

    const testImage = document.getElementById('test-image') as HTMLImageElement;

    const getNormalizedPoint = (
      e: MouseEvent | PointerEvent,
      targetEl: HTMLElement,
      source: 'image' | 'webcam'
    ): Point => {
      const rect = targetEl.getBoundingClientRect();
      let clickX = e.clientX - rect.left;
      let clickY = e.clientY - rect.top;
      let x = clickX / rect.width;
      const y = clickY / rect.height;
      if (source === 'webcam') {
        x = 1 - x;
      }
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    };

    const triggerSegment = async (source: 'image' | 'webcam') => {
      if (!this.isWorkerReady) return;

      let originalBitmapSource: HTMLImageElement | HTMLCanvasElement;
      if (source === 'image') {
        if (!testImage.src) return;
        originalBitmapSource = testImage;
      } else {
        if (!this.isFrozen) return;
        originalBitmapSource = this.webcamCapture;
      }

      this.updateStatus('Segmenting...');
      try {
        const bitmap = await createImageBitmap(originalBitmapSource);
        this.worker?.postMessage(
          {
            type: 'SEGMENT',
            bitmap,
            strokes: this.accumulatedStrokes,
          },
          [bitmap]
        );
      } catch (err) {
        console.error(err);
      }
    };

    const setupInteractiveEvents = (targetEl: HTMLElement, source: 'image' | 'webcam') => {
      targetEl.style.cursor = 'crosshair';
      if (source === 'image') {
        targetEl.style.pointerEvents = 'auto';
      }

      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return; // Only main left click
        if (source === 'webcam' && !this.isFrozen) return;

        try {
          targetEl.setPointerCapture(e.pointerId);
        } catch (_) {}

        this.isPointerDown = true;
        this.currentStrokePoints = [getNormalizedPoint(e, targetEl, source)];
        this.redrawOverlay(source);
      };

      const onPointerMove = (e: PointerEvent) => {
        if (!this.isPointerDown) return;
        const pt = getNormalizedPoint(e, targetEl, source);
        const lastPt = this.currentStrokePoints[this.currentStrokePoints.length - 1];
        if (!lastPt || Math.hypot(pt.x - lastPt.x, pt.y - lastPt.y) > 0.003) {
          this.currentStrokePoints.push(pt);
          this.redrawOverlay(source);
        }
      };

      const onPointerUp = (e: PointerEvent) => {
        if (!this.isPointerDown) return;
        this.isPointerDown = false;
        try {
          targetEl.releasePointerCapture(e.pointerId);
        } catch (_) {}

        if (this.currentStrokePoints.length > 0) {
          this.accumulatedStrokes.push({
            brushMode: this.currentStrokeMode,
            point: [...this.currentStrokePoints],
            isCompleted: true,
          });
          this.currentStrokePoints = [];
          this.redrawOverlay(source);
          triggerSegment(source);
        }
      };

      targetEl.addEventListener('pointerdown', onPointerDown);
      targetEl.addEventListener('pointermove', onPointerMove);
      targetEl.addEventListener('pointerup', onPointerUp);
      targetEl.addEventListener('pointercancel', onPointerUp);
    };

    setupInteractiveEvents(testImage, 'image');
    setupInteractiveEvents(this.canvasElement, 'image');
    setupInteractiveEvents(this.webcamCapture, 'webcam');
    setupInteractiveEvents(this.webcamOverlay, 'webcam');

    if (this.video) {
      this.video.style.cursor = 'pointer';
      this.video.addEventListener('click', () => {
        if (!this.isFrozen && this.video.srcObject) {
          this.toggleFreeze();
        }
      });
    }
  }

  private redrawOverlay(source: 'image' | 'webcam') {
    const ctx = source === 'webcam' ? this.overlayCtx : this.canvasCtx;
    const canvas = source === 'webcam' ? this.webcamOverlay : this.canvasElement;
    if (!ctx || !canvas) return;

    if (source === 'image') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && (canvas.width === 0 || canvas.height === 0 || canvas.width === 300)) {
        canvas.width = testImage.naturalWidth || testImage.clientWidth || 300;
        canvas.height = testImage.naturalHeight || testImage.clientHeight || 300;
      }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (this.currentMaskBitmap) {
      ctx.drawImage(this.currentMaskBitmap, 0, 0, canvas.width, canvas.height);
    }

    for (const stroke of this.accumulatedStrokes) {
      this.drawSingleStrokeOnCanvas(ctx, stroke);
    }

    if (this.isPointerDown && this.currentStrokePoints.length > 0) {
      const activeStroke: Stroke = {
        brushMode: this.currentStrokeMode,
        point: this.currentStrokePoints,
        isCompleted: false,
      };
      this.drawSingleStrokeOnCanvas(ctx, activeStroke);
    }
  }

  private drawSingleStrokeOnCanvas(ctx: CanvasRenderingContext2D, stroke: Stroke) {
    if (!stroke.point || stroke.point.length === 0) return;

    const width = ctx.canvas.width;
    const height = ctx.canvas.height;

    ctx.save();
    let color = 'rgba(76, 175, 80, 0.85)'; // Positive (Green)
    let lineWidth = 4;
    if (stroke.brushMode === 2) {
      color = 'rgba(229, 57, 53, 0.85)'; // Negative (Red)
      lineWidth = 4;
    } else if (stroke.brushMode === 3) {
      color = 'rgba(33, 150, 243, 0.85)'; // Lasso (Blue)
      lineWidth = 3;
    }

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (stroke.point.length === 1) {
      const p = stroke.point[0];
      ctx.beginPath();
      ctx.arc(p.x * width, p.y * height, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(stroke.point[0].x * width, stroke.point[0].y * height);
      for (let i = 1; i < stroke.point.length; i++) {
        ctx.lineTo(stroke.point[i].x * width, stroke.point[i].y * height);
      }
      if (stroke.brushMode === 3) {
        if (stroke.isCompleted || stroke.point.length > 2) {
          ctx.closePath();
        }
        ctx.fillStyle = 'rgba(33, 150, 243, 0.15)';
        ctx.fill();
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  private clearStrokes() {
    this.accumulatedStrokes = [];
    this.currentStrokePoints = [];
    this.isPointerDown = false;

    if (this.currentMaskBitmap) {
      this.currentMaskBitmap.close();
      this.currentMaskBitmap = null;
    }

    if (this.canvasCtx) {
      this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
    if (this.overlayCtx) {
      this.overlayCtx.clearRect(0, 0, this.webcamOverlay.width, this.webcamOverlay.height);
    }

    this.worker?.postMessage({ type: 'CLEAR' });
    this.updateStatus('Strokes cleared');
  }

  // Interactive Segmenter responds to CLI clicks, not continuous video frames
  protected override async predictWebcam() {}

  protected override async detectImage(_: HTMLImageElement) {
    if (this.runningMode !== 'IMAGE') this.runningMode = 'IMAGE';
    this.isWorkerReady = true;
    this.updateStatus('Ready');
  }

  protected override async enableCam() {
    await super.enableCam();
    if (this.freezeButton) {
      this.freezeButton.disabled = false;
      this.isFrozen = false;
      this.freezeButton.innerText = 'Freeze & Segment';
      this.webcamCapture.style.display = 'none';
      this.webcamOverlay.style.display = 'none';
      this.video.style.display = 'block';
    }

    const infoSpan = document.querySelector('.instructions-banner span:nth-of-type(2)') as HTMLSpanElement;
    if (infoSpan)
      infoSpan.innerText = 'Click anywhere on the webcam feed to freeze it, then click the object to segment.';
  }

  protected override stopCam(persistState = true) {
    super.stopCam(persistState);
    if (this.freezeButton) {
      this.freezeButton.disabled = true;
      this.isFrozen = false;
      this.webcamCapture.style.display = 'none';
      this.webcamOverlay.style.display = 'none';
      this.video.style.display = 'block';
      this.overlayCtx.clearRect(0, 0, this.webcamOverlay.width, this.webcamOverlay.height);
      this.webcamCtx.clearRect(0, 0, this.webcamCapture.width, this.webcamCapture.height);
    }

    const infoSpan = document.querySelector('.instructions-banner span:nth-of-type(2)') as HTMLSpanElement;
    if (infoSpan) infoSpan.innerText = 'Click on an object in the image or video to segment it.';
  }

  private toggleFreeze() {
    if (!this.video || !this.video.srcObject) return;

    if (!this.isFrozen) {
      this.webcamCapture.width = this.video.videoWidth;
      this.webcamCapture.height = this.video.videoHeight;
      this.webcamOverlay.width = this.video.videoWidth;
      this.webcamOverlay.height = this.video.videoHeight;

      this.webcamCtx.drawImage(this.video, 0, 0);

      this.video.style.display = 'none';
      this.webcamCapture.style.display = 'block';
      this.webcamOverlay.style.display = 'block';
      this.webcamOverlay.style.pointerEvents = 'auto';
      this.webcamOverlay.classList.add('clickable');
      this.webcamOverlay.style.width = '100%';

      this.isFrozen = true;
      this.freezeButton.innerText = 'Unfreeze';
      this.updateStatus('Frozen! Click on object to segment');
      const infoSpan = document.querySelector('.instructions-banner span:nth-of-type(2)') as HTMLSpanElement;
      if (infoSpan) infoSpan.innerText = 'Click on an object to segment it, or click Unfreeze to restart.';
    } else {
      this.isFrozen = false;
      this.freezeButton.innerText = 'Freeze & Segment';
      this.video.style.display = 'block';
      this.webcamCapture.style.display = 'none';
      this.webcamOverlay.style.display = 'none';
      this.webcamOverlay.style.pointerEvents = 'none';

      this.overlayCtx.clearRect(0, 0, this.webcamOverlay.width, this.webcamOverlay.height);
      this.updateStatus('Ready to freeze');
    }
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;
    if (type === 'SEGMENT_RESULT') {
      const { maskBitmap, inferenceTime } = event.data;
      if (inferenceTime > 0) {
        this.updateInferenceTime(inferenceTime);
        this.updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      }

      if (this.currentMaskBitmap) {
        this.currentMaskBitmap.close();
        this.currentMaskBitmap = null;
      }
      this.currentMaskBitmap = maskBitmap;
      this.redrawOverlay(this.runningMode === 'VIDEO' ? 'webcam' : 'image');
    } else {
      super.handleWorkerMessage(event);
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {};
  }

  protected override displayImageResult() {}
  protected override displayVideoResult() {}
}

let activeTask: InteractiveSegmenterTask | null = null;

export async function setupInteractiveSegmenter(container: HTMLElement) {
  activeTask = new InteractiveSegmenterTask({
    container,
    template,
    defaultModelName: 'interactive_segmentation',
    defaultModelUrl:
      'https://storage.googleapis.com/mediapipe-models/interactive_segmenter_v2/magic_touch/int8/1/interactive_segmentation.task',
    defaultDelegate: 'CPU',
    workerFactory: () =>
      new Worker(new URL('../workers/interactive-segmenter.worker.ts', import.meta.url), { type: 'module' }),
  });
  await activeTask.initialize();
}

export function cleanupInteractiveSegmenter() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
