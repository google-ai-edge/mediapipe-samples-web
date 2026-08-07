import { createLogger, defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { copyWasmFiles } from './copy-wasm.js';

const customLogger = createLogger();
const originalWarn = customLogger.warn;
customLogger.warn = (msg, options) => {
  if (msg.includes('Failed to load source map') || msg.includes('SOURCEMAP_ERROR')) {
    return;
  }
  originalWarn(msg, options);
};

function mediapipeWasmPlugin(): Plugin {
  return {
    name: 'mediapipe-wasm-plugin',
    buildStart() {
      copyWasmFiles();
    },
    configureServer(server) {
      copyWasmFiles();

      // Middleware to reliably serve WASM and JS loader files in dev & StackBlitz WebContainers
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();
        const urlPath = req.url.split('?')[0];
        const match = urlPath.match(/(?:.*\/)?wasm\/(.+)$/);
        if (match) {
          const fileName = match[1];
          const filePath = path.resolve('public/wasm', fileName);
          if (fs.existsSync(filePath)) {
            if (fileName.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
            } else if (fileName.endsWith('.js') || fileName.endsWith('.mjs')) {
              res.setHeader('Content-Type', 'application/javascript');
            }
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
            res.setHeader('Access-Control-Allow-Origin', '*');
            fs.createReadStream(filePath).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  base: '/mediapipe-samples-web/',
  customLogger,

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
  build: {
    sourcemap: false,
    rollupOptions: {
      onwarn(warning, defaultHandler) {
        if (warning.code === 'SOURCEMAP_ERROR') {
          return;
        }
        defaultHandler(warning);
      },
    },
  },
  server: {
    port: 5174,
    strictPort: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin'
    }
  },
  preview: {
    port: 5174,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin'
    }
  }
});
