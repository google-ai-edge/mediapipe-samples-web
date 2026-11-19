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

import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function downloadFile(fileName: string, url: string): Promise<string> {
    const downloadPath = path.resolve(__dirname, 'assets');
    if (!fs.existsSync(downloadPath)) {
        fs.mkdirSync(downloadPath, { recursive: true });
    }
    const filePath = path.join(downloadPath, fileName);
    
    // Check if file already exists to avoid re-downloading
    if (fs.existsSync(filePath)) {
        return filePath;
    }

    console.log(`Downloading ${fileName} from ${url}...`);

    // Hugging Face answers with a redirect to its CDN, so follow redirects.
    const maxRedirects = 5;
    return new Promise((resolve, reject) => {
        const request = (currentUrl: string, redirectsLeft: number) => {
            https.get(currentUrl, (response) => {
                const status = response.statusCode ?? 0;
                if (status >= 300 && status < 400 && response.headers.location) {
                    response.resume(); // Discard the redirect body.
                    if (redirectsLeft === 0) {
                        reject(new Error(`Failed to download ${url}: too many redirects`));
                        return;
                    }
                    request(new URL(response.headers.location, currentUrl).href, redirectsLeft - 1);
                    return;
                }
                if (status !== 200) {
                    response.resume();
                    reject(new Error(`Failed to download ${url}: ${status}`));
                    return;
                }
                const file = fs.createWriteStream(filePath);
                response.pipe(file);
                file.on('finish', () => {
                    file.close();
                    resolve(filePath);
                });
            }).on('error', (err) => {
                fs.unlink(filePath, () => {}); // Delete the file async.
                reject(err);
            });
        };
        request(url, maxRedirects);
    });
}

export default async function globalSetup() {
    const assets = [

        {
            name: 'efficientdet_lite0.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ObjectDetector-EfficientDet-Lite0/resolve/main/efficientdet_lite0_float32.tflite'
        },
        {
            name: 'efficientdet_lite2.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ObjectDetector-EfficientDet-Lite2/resolve/main/efficientdet_lite2_float32.tflite'
        },
        {
            name: 'deeplab_v3.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ImageSegmenter-DeepLab-V3/resolve/main/deeplab_v3.tflite'
        },
        {
            name: 'hair_segmenter.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ImageSegmenter-Hair/resolve/main/hair_segmenter.tflite'
        },
        {
            name: 'yamnet.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-AudioClassifier-YAMNet/resolve/main/yamnet.tflite'
        },
        {
            name: 'bert_classifier.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-TextClassifier-MobileBERT/resolve/main/bert_classifier.tflite'
        },
        {
            name: 'average_word_classifier.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-TextClassifier-AverageWord/resolve/main/average_word_classifier.tflite'
        },
        {
            name: 'universal_sentence_encoder.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-TextEmbedder-Universal-Sentence-Encoder/resolve/main/universal_sentence_encoder.tflite'
        },
        {
            name: 'blaze_face_short_range.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-FaceDetector-BlazeFace-Short-Range/resolve/main/blaze_face_short_range.tflite'
        },
        {
            name: 'language_detector.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-LanguageDetector/resolve/main/language_detector.tflite'
        },
        {
            name: 'mobilenet_v3_small.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ImageEmbedder-MobileNet-V3-Small/resolve/main/mobilenet_v3_small.tflite'
        },
        {
            name: 'mobilenet_v3_large.tflite',
            url: 'https://huggingface.co/litert-community/MediaPipe-ImageEmbedder-MobileNet-V3-Large/resolve/main/mobilenet_v3_large.tflite'
        }
    ];

    console.log('--- Playwright Global Setup: Fetching Required Test Assets ---');
    try {
        await Promise.all(assets.map(asset => downloadFile(asset.name, asset.url)));
        console.log('--- Test Assets Verification Complete ---');
    } catch (error) {
        console.error('Error during global setup test asset fetch:', error);
        throw error;
    }
}
