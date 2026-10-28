// The example's build. Two things a page with anira needs from its bundler and its server:
// - the worklet as one self-contained script: `import url from './my-worklet.ts?worker&url'`
//   has Vite bundle my-worklet.ts and everything it imports into a single ES module file (no
//   chunk to import, since addModule's scope cannot load one at run time) and hands main.ts
//   its URL for registerAudioWorkletForContext;
// - cross-origin isolation: the module's memory is a shared WebAssembly.Memory, which needs
//   `crossOriginIsolated`, so every response carries COOP and COEP (here for `vite` and
//   `vite preview`; a production server sends the same two headers).
import path from 'node:path'
import { defineConfig } from 'vite'

const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig({
  root: __dirname,
  base: './',
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  // Vite 8 (Rolldown); with Vite 5-7: rollupOptions: { output: { inlineDynamicImports: true } }.
  worker: {
    format: 'es',
    rolldownOptions: { output: { codeSplitting: false } },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'esnext',
    rollupOptions: {
      input: {
        index: path.resolve(__dirname, 'index.html'),
        check: path.resolve(__dirname, 'check.html'),
      },
    },
  },
})
