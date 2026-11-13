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

import { AudioClassifierResult } from '@mediapipe/tasks-audio';
import { BaseAudioTask, BaseAudioTaskOptions } from '../components/base-audio-task';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';

// @ts-ignore
import template from '../templates/audio-classifier.html?raw';
// @ts-ignore

class AudioClassifierTask extends BaseAudioTask {
  private classificationResultUI: ClassificationResult | undefined;

  // Options
  private maxResults = 3;
  private scoreThreshold = 0.5;

  // Visualization State
  private WAVEFORM_HISTORY_SIZE = 8000;
  private waveformBuffer = new Float32Array(this.WAVEFORM_HISTORY_SIZE);
  private waveformSnapshots: Float32Array[] = [];
  private MAX_SNAPSHOTS = 5;

  private canvasElement!: HTMLCanvasElement;
  private canvasCtx!: CanvasRenderingContext2D;

  constructor(options: BaseAudioTaskOptions) {
    super(options);
  }

  protected override onInitializeUI() {
    this.canvasElement = document.getElementById('waveform-canvas') as HTMLCanvasElement;
    if (this.canvasElement) {
      this.canvasCtx = this.canvasElement.getContext('2d')!;
    }

    this.classificationResultUI = new ClassificationResult('classification-results');

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

    setupSlider('max-results', (val) => {
      this.maxResults = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', maxResults: this.maxResults });
    });

    setupSlider('score-threshold', (val) => {
      this.scoreThreshold = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold: this.scoreThreshold });
    });

    // Delegate override default to GPU
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
    if (delegateSelect) {
      delegateSelect.value = 'GPU';
    }
    this.currentDelegate = 'GPU';

    this.models = {
      yamnet: 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([{ label: 'Yamnet (AudioSet)', value: 'yamnet', isDefault: true }]);
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      maxResults: this.maxResults,
      scoreThreshold: this.scoreThreshold,
    };
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    if (type === 'CLASSIFY_RESULT') {
      const { results, inferenceTime } = event.data;
      const avgTime = this.updateInferenceTime(inferenceTime);
      this.updateStatus(`Done in ${Math.round(avgTime)}ms`);
      this.displayClassificationResults(results);
      this.publishTestResult('audio_classifier', () => ({
        mode: this.runningMode,
        resultCount: results?.length ?? 0,
        // One entry per classified window, each with its highest-scoring categories.
        results: ((results ?? []) as AudioClassifierResult[]).slice(0, 32).map((r) => ({
          timestampMs: r.timestampMs ?? null,
          categories: [...(r.classifications?.[0]?.categories ?? [])]
            .sort((a, b) => b.score - a.score)
            .slice(0, 5)
            .map((c) => ({ label: c.categoryName, score: c.score })),
        })),
      }));
    } else {
      super.handleWorkerMessage(event);
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();
    const recBtn = document.getElementById('recordButton') as HTMLButtonElement;
    if (recBtn) recBtn.disabled = false;
  }

  protected override clearResults(): void {
    if (this.classificationResultUI) {
      this.classificationResultUI.clear();
    }
    const resultsContainer = document.querySelector('.audio-viewport .results-container') as HTMLElement;
    if (resultsContainer) {
      resultsContainer.classList.remove('active');
      resultsContainer.style.display = 'none';
    }
  }

  protected override onAudioFileLoaded(file: File) {
    const runBtn = document.getElementById('run-file-classification') as HTMLButtonElement;
    if (runBtn) {
      runBtn.onclick = async () => {
        this.updateStatus('Processing file...');
        runBtn.disabled = true;
        try {
          await this.processAudioFile(file);
        } catch (err) {
          console.error(err);
          this.updateStatus('File Error');
        } finally {
          runBtn.disabled = false;
        }
      };
    }
  }

  protected override processAudioData(data: Float32Array, sampleRate: number): void {
    this.visualizeWaveform(data);

    if (this.isWorkerReady && this.worker) {
      this.worker.postMessage({
        type: 'CLASSIFY',
        audioData: data,
        sampleRate: sampleRate,
        timestampMs: performance.now(),
      });
    }
  }

  private async processAudioFile(file: File) {
    const arrayBuffer = await file.arrayBuffer();
    const tempCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);

    const inputData = audioBuffer.getChannelData(0);

    // Visualize first few seconds of file
    this.visualizeWaveform(inputData.slice(0, 4096));

    if (this.isWorkerReady && this.worker) {
      this.worker.postMessage({
        type: 'CLASSIFY',
        audioData: inputData,
        sampleRate: tempCtx.sampleRate,
        timestampMs: 0,
      });
    }

    await tempCtx.close();
  }

  private updateWaveformBuffer(newData: Float32Array) {
    const validData = newData.length > 0 ? newData : new Float32Array(newData.length);
    const newBuffer = new Float32Array(this.WAVEFORM_HISTORY_SIZE);

    newBuffer.set(this.waveformBuffer.subarray(validData.length), 0);
    newBuffer.set(validData, this.WAVEFORM_HISTORY_SIZE - validData.length);

    this.waveformBuffer = newBuffer;
  }

  private visualizeWaveform(newData: Float32Array) {
    if (!this.canvasCtx || !this.canvasElement) return;

    if (this.canvasElement.width !== this.canvasElement.clientWidth) {
      this.canvasElement.width = this.canvasElement.clientWidth;
      this.canvasElement.height = this.canvasElement.clientHeight;
    }

    const width = this.canvasElement.width;
    const height = this.canvasElement.height;

    this.updateWaveformBuffer(newData);

    this.waveformSnapshots.push(new Float32Array(this.waveformBuffer));
    if (this.waveformSnapshots.length > this.MAX_SNAPSHOTS) {
      this.waveformSnapshots.shift();
    }

    this.canvasCtx.fillStyle = '#f8f9fa';
    this.canvasCtx.fillRect(0, 0, width, height);

    this.canvasCtx.lineWidth = 2;

    this.waveformSnapshots.forEach((snapshot, index) => {
      const isLast = index === this.waveformSnapshots.length - 1;
      const alpha = (index + 1) / (this.waveformSnapshots.length + 1);

      if (isLast) {
        this.canvasCtx.strokeStyle = `rgba(0, 96, 100, 1.0)`;
        this.canvasCtx.lineWidth = 2.5;
      } else {
        this.canvasCtx.strokeStyle = `rgba(0, 151, 167, ${alpha * 0.5})`;
        this.canvasCtx.lineWidth = 2;
      }

      this.canvasCtx.beginPath();
      const sliceWidth = width / snapshot.length;
      let x = 0;

      const step = Math.ceil(snapshot.length / width);

      for (let i = 0; i < snapshot.length; i += step) {
        const v = snapshot[i];
        const y = (v * height) / 1.5 + height / 2;

        if (i === 0) {
          this.canvasCtx.moveTo(x, y);
        } else {
          this.canvasCtx.lineTo(x, y);
        }
        x += sliceWidth * step;
      }
      this.canvasCtx.stroke();
    });
  }

  private displayClassificationResults(results: AudioClassifierResult[]) {
    if (!results || results.length === 0 || !this.classificationResultUI) return;

    const classifications = results[0].classifications[0].categories;
    classifications.sort((a, b) => b.score - a.score);

    const resultsContainer = document.querySelector('.audio-viewport .results-container') as HTMLElement;
    if (resultsContainer) {
      resultsContainer.classList.add('active');
      resultsContainer.style.display = 'block';
    }

    const topResults = classifications.slice(0, this.maxResults);
    const items: ClassificationItem[] = topResults.map((c) => ({
      label: c.categoryName,
      score: c.score,
    }));

    this.classificationResultUI.updateResults(items);
  }
}

let activeTask: AudioClassifierTask | null = null;

export async function setupAudioClassifier(container: HTMLElement) {
  activeTask = new AudioClassifierTask({
    container,
    template,
    defaultModelName: 'yamnet',
    defaultModelUrl: 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite',
    workerFactory: () =>
      new Worker(new URL('../workers/audio-classifier.worker.ts', import.meta.url), { type: 'module' }),
    defaultDelegate: 'GPU',
  });

  await activeTask.initialize();
}

export function cleanupAudioClassifier() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
