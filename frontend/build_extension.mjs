import { build } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { copyFileSync, mkdirSync, cpSync, rmSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const outDir = path.resolve(rootDir, 'extension', 'dist');

console.log('🚀 Building Job Application Optimizer Chrome Extension...');

// 1. Clean output directory
if (existsSync(outDir)) {
  rmSync(outDir, { recursive: true, force: true });
}
mkdirSync(outDir, { recursive: true });

// 2. Build the React UI (Side Panel)
console.log('📦 1/3 Compiling Side Panel React Application...');
await build({
  root: __dirname,
  base: './', // Relative paths for Chrome Extension
  plugins: [react(), tailwindcss()],
  build: {
    outDir: outDir,
    emptyOutDir: false,
  },
});

// Rename or ensure sidepanel.html exists
if (existsSync(path.join(outDir, 'index.html'))) {
  copyFileSync(path.join(outDir, 'index.html'), path.join(outDir, 'sidepanel.html'));
}

// 3. Build Background Service Worker
console.log('📦 2/3 Compiling Service Worker (background.ts)...');
await build({
  root: path.resolve(rootDir, 'extension'),
  build: {
    outDir: outDir,
    emptyOutDir: false,
    lib: {
      entry: path.resolve(rootDir, 'extension', 'src', 'background.ts'),
      name: 'background',
      formats: ['es'],
      fileName: () => 'background.js',
    },
    rollupOptions: {
      output: {
        entryFileNames: 'background.js',
      },
    },
  },
});

// 4. Build Content Script (IIFE format so it executes in page context without module loader)
console.log('📦 3/3 Compiling Content Script (content.ts)...');
await build({
  root: path.resolve(rootDir, 'extension'),
  build: {
    outDir: outDir,
    emptyOutDir: false,
    lib: {
      entry: path.resolve(rootDir, 'extension', 'src', 'content.ts'),
      name: 'content',
      formats: ['iife'],
      fileName: () => 'content.js',
    },
    rollupOptions: {
      output: {
        entryFileNames: 'content.js',
      },
    },
  },
});

// 5. Copy manifest.json & icons
console.log('📋 Copying Manifest and Icons...');
copyFileSync(
  path.resolve(rootDir, 'extension', 'manifest.json'),
  path.join(outDir, 'manifest.json')
);

const iconsSrc = path.resolve(rootDir, 'extension', 'icons');
const iconsDest = path.join(outDir, 'icons');
if (existsSync(iconsSrc)) {
  cpSync(iconsSrc, iconsDest, { recursive: true });
}

// 6. Mirror compiled files to extension root so selecting either 'extension' or 'extension/dist' succeeds
const extensionRoot = path.resolve(rootDir, 'extension');
const filesToMirror = ['background.js', 'content.js', 'sidepanel.html', 'index.html'];
for (const f of filesToMirror) {
  const src = path.join(outDir, f);
  if (existsSync(src)) {
    copyFileSync(src, path.join(extensionRoot, f));
  }
}
// Clear previously mirrored hashed assets first, so rebuilt bundles (their
// names are content-hashed) never accumulate stale generations.
rmSync(path.join(extensionRoot, 'assets'), { recursive: true, force: true });
const assetsSrc = path.join(outDir, 'assets');
if (existsSync(assetsSrc)) {
  cpSync(assetsSrc, path.join(extensionRoot, 'assets'), { recursive: true });
}

console.log('✅ Chrome Extension build finished! Output ready at: extension/ and extension/dist/');

