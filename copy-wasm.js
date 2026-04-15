import fs from 'fs';
import path from 'path';

const srcDirs = [
  'node_modules/@mediapipe/tasks-audio/wasm',
  'node_modules/@mediapipe/tasks-vision/wasm',
  'node_modules/@mediapipe/tasks-text/wasm',
];

const destDir = 'public/wasm';

if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

srcDirs.forEach(srcDir => {
  if (fs.existsSync(srcDir)) {
    const files = fs.readdirSync(srcDir);
    files.forEach(file => {
      const srcFile = path.join(srcDir, file);
      const destFile = path.join(destDir, file);
      fs.copyFileSync(srcFile, destFile);
      console.log(`Copied ${srcFile} to ${destFile}`);
    });
  } else {
    console.warn(`Source directory does not exist: ${srcDir}`);
  }
});

console.log('Successfully prepared WASM static assets.');
