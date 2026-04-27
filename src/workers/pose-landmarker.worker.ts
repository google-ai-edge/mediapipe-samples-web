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

import { PoseLandmarker, PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class PoseLandmarkerWorker extends BaseWorker<PoseLandmarker> {
  protected async initializeTask(): Promise<void> {
    const vision = await this.getVisionFileset();
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      minPoseDetectionConfidence: this.currentOptions.minPoseDetectionConfidence,
      minPosePresenceConfidence: this.currentOptions.minPosePresenceConfidence,
      minTrackingConfidence: this.currentOptions.minTrackingConfidence,
      numPoses: this.currentOptions.numPoses,
      outputSegmentationMasks: this.currentOptions.outputSegmentationMasks,
      runningMode: this.currentOptions.runningMode,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        minPoseDetectionConfidence: this.currentOptions.minPoseDetectionConfidence,
        minPosePresenceConfidence: this.currentOptions.minPosePresenceConfidence,
        minTrackingConfidence: this.currentOptions.minTrackingConfidence,
        numPoses: this.currentOptions.numPoses,
        outputSegmentationMasks: this.currentOptions.outputSegmentationMasks,
        runningMode: this.currentOptions.runningMode,
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'DETECT_IMAGE' || data.type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = data;
      const requiredMode = data.type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        console.warn('PoseLandmarker not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();
      let result: PoseLandmarkerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = this.taskInstance.detectForVideo(bitmap, timestampMs);
        } else {
          result = this.taskInstance.detect(bitmap);
        }
      } catch (e: any) {
        console.error('Worker detection error:', e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Detection failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      self.postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime,
      });
    }
  }
}

new PoseLandmarkerWorker();
