// The example's build. Three things a page with anira needs from its bundler and its server:
// - the worker module and the worklet module as self-contained scripts: `import url from
//   './double-engine.ts?worker&url'` has Vite bundle the file and everything it imports into one
//   ES module (no chunk to import: a worklet scope cannot load one) and hands main.ts its URL;
// - cross-origin isolation: the module's memory is a shared WebAssembly.Memory, which needs
//   `crossOriginIsolated`, so every response carries COOP and COEP (here for `vite` and
//   `vite preview`; a production server sends the same two headers);
// - the package's files as they are: `optimizeDeps.exclude`, so the dev server does not
//   pre-bundle the package (its module's glue loads itself into the Workers and the worklet).
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
  optimizeDeps: { exclude: ['@anira-project/anira'] },
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
