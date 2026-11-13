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

/**
 * Known input for the webcam and microphone cases.
 *
 * Chrome's fake camera normally shows a moving test pattern with nothing to
 * detect. `--use-file-for-fake-video-capture` makes it play a file instead, so
 * the webcam cases see the same photo as the image cases and can be verified
 * against the same ground truth. The file must be raw video (Y4M); this module
 * converts a sample photo into a one-frame Y4M, which Chrome loops.
 */

import fs from 'node:fs';
import path from 'node:path';

const MIME_BY_EXTENSION = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png' };

/**
 * Decodes an image in the browser and returns it as one I420 frame, the pixel
 * format Y4M uses. Runs inside the page.
 */
async function imageToI420InPage({ dataUrl }) {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  // I420 stores colour at half resolution, so both sides must be even.
  const width = img.naturalWidth - (img.naturalWidth % 2);
  const height = img.naturalHeight - (img.naturalHeight % 2);
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, width, height).data;

  const ySize = width * height;
  const cSize = ySize / 4;
  const out = new Uint8Array(ySize + 2 * cSize);
  const clamp = (v) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  // Full-range BT.601, matching the "C420jpeg" colour space named in the header.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      out[y * width + x] = clamp(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
    }
  }
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (const [dx, dy] of [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const i = ((y + dy) * width + (x + dx)) * 4;
        r += rgba[i];
        g += rgba[i + 1];
        b += rgba[i + 2];
      }
      r /= 4;
      g /= 4;
      b /= 4;
      const c = (y / 2) * (width / 2) + x / 2;
      out[ySize + c] = clamp(-0.168736 * r - 0.331264 * g + 0.5 * b + 128);
      out[ySize + cSize + c] = clamp(0.5 * r - 0.418688 * g - 0.081312 * b + 128);
    }
  }

  let binary = '';
  for (let i = 0; i < out.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, out.subarray(i, i + 0x8000));
  }
  return { width, height, base64: btoa(binary) };
}

/**
 * Writes `outPath`, a one-frame Y4M video of the image in `imageBytes`, using
 * `page` (any open browser page) to decode the image.
 */
export async function writeFakeCameraVideo(page, imageBytes, extension, outPath) {
  const mime = MIME_BY_EXTENSION[extension.toLowerCase()];
  if (!mime) throw new Error(`Unsupported image type for the fake camera: ${extension}`);
  const dataUrl = `data:${mime};base64,${Buffer.from(imageBytes).toString('base64')}`;
  const frame = await page.evaluate(imageToI420InPage, { dataUrl });
  const header = Buffer.from(`YUV4MPEG2 W${frame.width} H${frame.height} F30:1 Ip A1:1 C420jpeg\n`, 'ascii');
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.concat([header, Buffer.from('FRAME\n', 'ascii'), Buffer.from(frame.base64, 'base64')]));
  return { width: frame.width, height: frame.height };
}

/** Chrome flags that feed a video file to the fake camera and a WAV file to the fake microphone. */
export function fakeMediaLaunchArgs({ video, audio }) {
  const args = [];
  if (video) args.push(`--use-file-for-fake-video-capture=${video}`);
  if (audio) args.push(`--use-file-for-fake-audio-capture=${audio}`);
  return args;
}
