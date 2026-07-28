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
import { Point, Stroke, BrushMode } from '@mediapipe/tasks-vision';

class InteractiveSegmenterTask extends BaseVisionTask {
  private isFrozen = false;
  private webcamCapture!: HTMLCanvasElement;
  private webcamOverlay!: HTMLCanvasElement;
  private freezeButton!: HTMLButtonElement;
  private strokeModeSelect!: HTMLSelectElement;
  private clearStrokesBtn!: HTMLButtonElement;
  private webcamCtx!: CanvasRenderingContext2D;
  private overlayCtx!: CanvasRenderingContext2D;

  private currentStrokeMode: BrushMode = 1; // 1: Positive, 2: Negative, 3: Lasso
  private accumulatedStrokes: Stroke[] = [];
  private isPointerDown = false;
  private currentStrokePoints: Point[] = [];
  private currentMaskBitmap: ImageBitmap | null = null;
  private imageSet = false;
  private readonly MIN_STROKE_LENGTH = 0.05;
  private hasDrawnValidStroke = false;

  /** True if the worker is currently computing a mask. */
  private isWorkerProcessing = false;
  /** True if the user drew new strokes while the worker was busy. */
  private hasUnprocessedStrokes = false;
  /** An auto-incrementing ID used to drop stale results. */
  private activeRequestId = 0;

  constructor(options: BaseVisionTaskOptions) {
    super(options);
  }

  private async setImageOnWorker(source: HTMLImageElement | HTMLCanvasElement) {
    if (!this.worker || !this.isWorkerReady) return;
    this.updateStatus('Setting image...');
    try {
      const bitmap = await createImageBitmap(source);
      this.worker.postMessage(
        {
          type: 'SET_IMAGE',
          bitmap: bitmap,
        },
        [bitmap]
      );
      this.imageSet = true;
      this.updateStatus('Ready');
    } catch (err) {
      console.error('Failed to set image on worker:', err);
      this.updateStatus('Error setting image');
    }
  }

  private getStrokeLength(points: Point[]): number {
    let len = 0;
    for (let i = 1; i < points.length; i++) {
      len += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    }
    return len;
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
      // e.offsetX gives the exact pixel coordinate relative to the padding edge, but can be error-prone
      // with certain flex/grid layouts. getBoundingClientRect is safer.
      const rect = targetEl.getBoundingClientRect();
      let clickX = e.clientX - rect.left;
      let clickY = e.clientY - rect.top;
      let x = clickX / rect.width;
      const y = clickY / rect.height;

      // The WebCam feed visually mirrors logic, so X must be flipped.
      // Image mode natively aligns, so no X flip is necessary.
      if (source === 'webcam') {
        x = 1 - x;
      }
      return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
    };

    const setupInteractiveEvents = (targetEl: HTMLElement, source: 'image' | 'webcam') => {
      targetEl.style.cursor = 'crosshair';
      if (source === 'image') {
        targetEl.style.pointerEvents = 'auto';
      }

      const progressContainer = document.getElementById('stroke-progress-container');
      const progressCircle = document.getElementById('stroke-progress-circle');
      const progressTooltip = document.getElementById('stroke-progress-tooltip');

      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return; // Only main left click
        if (source === 'webcam' && !this.isFrozen) return;

        if (this.strokeModeSelect) {
          this.currentStrokeMode = parseInt(this.strokeModeSelect.value, 10) || 1;
        }

        try {
          targetEl.setPointerCapture(e.pointerId);
        } catch (_) {}

        this.isPointerDown = true;
        this.currentStrokePoints = [getNormalizedPoint(e, targetEl, source)];
        this.redrawOverlay(source);

        if (progressContainer && progressCircle && progressTooltip) {
          progressContainer.style.display = 'block';
          progressContainer.style.opacity = '1';
          progressContainer.style.transform = 'translate(-50%, -50%) scale(1)';
          progressContainer.style.left = `${e.clientX}px`;
          progressContainer.style.top = `${e.clientY}px`;
          progressCircle.setAttribute('stroke-dasharray', '0, 100');
          progressCircle.setAttribute('stroke', '#ff5722');
          progressCircle.style.opacity = '1';

          if (!this.hasDrawnValidStroke) {
            progressTooltip.style.opacity = '1';
          } else {
            progressTooltip.style.opacity = '0';
          }
        }
      };

      const onPointerMove = (e: PointerEvent) => {
        if (!this.isPointerDown) return;
        const pt = getNormalizedPoint(e, targetEl, source);
        const lastPt = this.currentStrokePoints[this.currentStrokePoints.length - 1];
        if (!lastPt || Math.hypot(pt.x - lastPt.x, pt.y - lastPt.y) > 0.003) {
          this.currentStrokePoints.push(pt);
          this.redrawOverlay(source);
        }

        if (progressContainer && progressCircle) {
          progressContainer.style.left = `${e.clientX}px`;
          progressContainer.style.top = `${e.clientY}px`;

          const len = this.getStrokeLength(this.currentStrokePoints);
          const percentage = Math.min(100, (len / this.MIN_STROKE_LENGTH) * 100);
          progressCircle.setAttribute('stroke-dasharray', `${percentage}, 100`);

          if (percentage >= 100) {
            progressCircle.setAttribute('stroke', '#4caf50');
            if (progressTooltip) progressTooltip.style.opacity = '0';
          } else {
            progressCircle.setAttribute('stroke', '#ff5722');
            if (progressTooltip && !this.hasDrawnValidStroke) progressTooltip.style.opacity = '1';
          }
        }
      };

      const onPointerUp = (e: PointerEvent) => {
        if (!this.isPointerDown) return;
        this.isPointerDown = false;
        try {
          targetEl.releasePointerCapture(e.pointerId);
        } catch (_) {}

        if (this.currentStrokePoints.length > 0) {
          const len = this.getStrokeLength(this.currentStrokePoints);
          const isSingleClick = this.currentStrokePoints.length === 1;

          if (!isSingleClick && len < this.MIN_STROKE_LENGTH) {
            // Stroke too short
            if (progressContainer && progressTooltip) {
              if (!this.hasDrawnValidStroke) {
                progressTooltip.style.opacity = '1';
              }
              setTimeout(() => {
                progressContainer.style.opacity = '0';
                setTimeout(() => {
                  progressContainer.style.display = 'none';
                }, 200);
              }, 600);
            }
            this.currentStrokePoints = [];
            this.redrawOverlay(source);
            return;
          }

          // Stroke was valid! Mark it.
          this.hasDrawnValidStroke = true;

          if (progressContainer) {
            progressContainer.style.display = 'none';
          }

          this.accumulatedStrokes.push({
            brushMode: this.currentStrokeMode,
            point: [...this.currentStrokePoints],
            isCompleted: true,
          });
          this.currentStrokePoints = [];
          this.redrawOverlay(source);

          // Start the animation instantly on mouse up
          if (source === 'image') {
            const testImg = document.getElementById('test-image') as HTMLImageElement;
            if (testImg) testImg.classList.add('breathing-animation');
          } else {
            this.webcamCapture.classList.add('breathing-animation');
          }

          this.hasUnprocessedStrokes = true;
          if (!this.isWorkerProcessing) {
            this.triggerSegment(source);
          }
        } else {
          if (progressContainer) progressContainer.style.display = 'none';
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

  private async triggerSegment(source: 'image' | 'webcam') {
    if (!this.isWorkerReady) return;

    if (!this.imageSet) {
      console.warn('Image not set on worker yet!');
      this.updateStatus('Waiting for image setup...');
      return;
    }

    this.hasUnprocessedStrokes = false;
    this.isWorkerProcessing = true;
    this.activeRequestId++;
    const reqId = this.activeRequestId;

    this.updateStatus('Segmenting...');
    try {
      this.worker?.postMessage({
        type: 'SEGMENT',
        strokes: this.accumulatedStrokes,
        reqId,
      });
    } catch (err) {
      console.error(err);
    }
  }

  protected override async initializeTask(): Promise<void> {
    this.clearStrokes();
    await super.initializeTask();
  }

  protected override setupImageUpload() {
    super.setupImageUpload();
    const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
    imageUpload?.addEventListener('change', () => {
      this.clearStrokes();
      this.imageSet = false;
    });
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

    // Check if it's a single point or a workaround stroke (2 very close points)
    const isSinglePoint =
      stroke.point.length === 1 ||
      (stroke.point.length === 2 &&
        Math.abs(stroke.point[0].x - stroke.point[1].x) < 0.002 &&
        Math.abs(stroke.point[0].y - stroke.point[1].y) < 0.002);

    if (isSinglePoint) {
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

    this.isWorkerProcessing = false;
    this.hasUnprocessedStrokes = false;
    this.activeRequestId++;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    if (testImage) testImage.classList.remove('breathing-animation');
    if (this.webcamCapture) this.webcamCapture.classList.remove('breathing-animation');

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

    this.worker?.postMessage({ type: 'CLEAR', reqId: this.activeRequestId });
    this.updateStatus('Strokes cleared');
  }

  // Interactive Segmenter responds to CLI clicks, not continuous video frames
  protected override async predictWebcam() {}

  protected override async detectImage(image: HTMLImageElement) {
    this.clearStrokes();
    this.imageSet = false;
    if (this.runningMode !== 'IMAGE') this.runningMode = 'IMAGE';
    this.isWorkerReady = true;
    this.updateStatus('Ready');

    if (image) {
      this.canvasElement.width = image.naturalWidth;
      this.canvasElement.height = image.naturalHeight;
      await this.setImageOnWorker(image);
    }
  }

  protected override async enableCam() {
    this.clearStrokes();
    this.imageSet = false;
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
      infoSpan.innerText =
        'Click anywhere on the webcam feed to freeze it, then draw a stroke on the object to segment.';
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
    if (infoSpan) infoSpan.innerText = 'Draw a stroke on an object in the image to segment it.';
  }

  private async toggleFreeze() {
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
      this.clearStrokes();
      this.imageSet = false;
      this.freezeButton.innerText = 'Unfreeze';
      await this.setImageOnWorker(this.webcamCapture);
      this.updateStatus('Frozen! Draw on object to segment');
      const infoSpan = document.querySelector('.instructions-banner span:nth-of-type(2)') as HTMLSpanElement;
      if (infoSpan) infoSpan.innerText = 'Draw a stroke on an object to segment it, or click Unfreeze to restart.';
    } else {
      this.isFrozen = false;
      this.clearStrokes();
      this.imageSet = false;
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
      const { reqId } = event.data;

      // Drop stale results
      if (reqId !== undefined && reqId !== this.activeRequestId) {
        if (event.data.maskBitmap) {
          event.data.maskBitmap.close();
        }
        return;
      }

      if (this.hasUnprocessedStrokes) {
        // The user drew more strokes while we were segmenting!
        setTimeout(() => this.triggerSegment(this.runningMode === 'VIDEO' ? 'webcam' : 'image'), 0);
      } else {
        this.isWorkerProcessing = false;
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage) testImage.classList.remove('breathing-animation');
        if (this.webcamCapture) this.webcamCapture.classList.remove('breathing-animation');
      }

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
    defaultDelegate: 'GPU',
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
