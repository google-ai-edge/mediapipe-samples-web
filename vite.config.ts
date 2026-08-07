import { defineConfig, type Plugin } from 'vite';
import { copyWasmFiles } from './copy-wasm.js';

function mediapipeWasmPlugin(): Plugin {
  return {
    name: 'mediapipe-wasm-plugin',
    buildStart() {
      copyWasmFiles();
    },
    configureServer() {
      copyWasmFiles();
    },
  };
}

export default defineConfig({
  base: '/mediapipe-samples-web/',

  plugins: [mediapipeWasmPlugin()],
  optimizeDeps: {
    exclude: [
      '@mediapipe/tasks-vision',
      '@mediapipe/tasks-audio',
      '@mediapipe/tasks-text'
    ]
  },
  worker: {
    format: 'es'
  },
  server: {
    port: 5174,
    strictPort: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  preview: {
    port: 5174,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
});
