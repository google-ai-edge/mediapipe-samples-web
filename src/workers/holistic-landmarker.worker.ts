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

import { HolisticLandmarker } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class HolisticLandmarkerWorker extends BaseWorker<HolisticLandmarker> {
  protected async initializeTask(data: any): Promise<void> {
    const vision = await this.getVisionFileset();

    this.taskInstance = await HolisticLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.currentOptions.modelAssetPath,
        delegate: this.currentOptions.delegate || 'GPU',
      },
      runningMode: data?.runningMode || 'IMAGE',
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minFaceSuppressionThreshold: 0.5,
      minHandLandmarksConfidence: 0.5,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minPoseSuppressionThreshold: 0.5,
    });
  }

  protected async updateOptions(data: any): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        runningMode: data.runningMode || this.currentOptions.runningMode || 'IMAGE',
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minFaceSuppressionThreshold: 0.5,
        minHandLandmarksConfidence: 0.5,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minPoseSuppressionThreshold: 0.5,
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    const { type, ...rest } = data;

    if (type === 'DETECT_IMAGE' && this.taskInstance) {
      const startTimeMs = performance.now();
      const result = this.taskInstance.detect(rest.bitmap);
      self.postMessage({
        type: 'DETECT_RESULT',
        result,
        mode: 'IMAGE',
        inferenceTime: performance.now() - startTimeMs,
      });
    } else if (type === 'DETECT_VIDEO' && this.taskInstance) {
      try {
        const startTimeMs = performance.now();
        const result = this.taskInstance.detectForVideo(rest.bitmap, rest.timestampMs);
        self.postMessage({
          type: 'DETECT_RESULT',
          result,
          mode: 'VIDEO',
          inferenceTime: performance.now() - startTimeMs,
        });
      } catch (e) {
        console.warn('Video detection error', e);
      }
    } else if (type === 'CLEANUP') {
      this.taskInstance?.close();
      this.taskInstance = undefined;
    }
  }
}

new HolisticLandmarkerWorker();
