import path from 'path'
import { defineConfig, type Plugin } from 'vite'
import { rollup } from 'rollup'
import dts from 'vite-plugin-dts'
import { viteStaticCopy } from 'vite-plugin-static-copy'

function watchWasm() {
  return {
    name: 'watch-wasm',
    buildStart() {
      this.addWatchFile(path.resolve(__dirname, 'wasm'))
    },
  }
}

/**
 * Prevents Vite's library-mode build from inlining `new URL(path, import.meta.url)`
 * references as base64 data URLs. Instead, the built output keeps the `new URL()`
 * pattern with corrected paths pointing to the built entry points, so that a
 * consumer's app-mode Vite build can detect them (e.g. for worker bundling).
 */
function preserveWorkerUrls(): Plugin {
  // Map source-level paths to their built entry point paths in dist/
  const rewrites: Array<{ sourcePath: string; builtEntry: string }> = [
    {
      sourcePath: './workers/inference-worker.ts',
      builtEntry: 'workers/inference-worker.js',
    },
    {
      sourcePath: './workers/audio-worklet.bundled.js',
      builtEntry: 'workers/audio-worklet.bundled.js',
    },
    {
      sourcePath: '../wasm/AniraWeb.js',
      builtEntry: 'wasm/AniraWeb.js',
    },
    {
      sourcePath: '../wasm/AniraWeb.wasm',
      builtEntry: 'wasm/AniraWeb.wasm',
    },
  ]

  return {
    name: 'preserve-worker-urls',
    enforce: 'pre',

    // In transform (pre): wrap the path in String() so Vite's
    // assetImportMetaUrl plugin won't match it (requires a plain literal).
    transform(code) {
      let result = code
      let modified = false
      for (const { sourcePath } of rewrites) {
        const needle = `new URL('${sourcePath}', import.meta.url)`
        if (result.includes(needle)) {
          result = result.replace(
            needle,
            `new URL(String('${sourcePath}'), import.meta.url)`
          )
          modified = true
        }
      }
      return modified ? result : null
    },

    // In renderChunk: replace String('source-path') with the correct
    // literal path to the built entry, restoring the standard new URL() form.
    renderChunk(code, chunk) {
      let result = code
      let modified = false
      for (const { sourcePath, builtEntry } of rewrites) {
        const pattern = new RegExp(
          `String\\(["']${sourcePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}["']\\)`,
          'g'
        )
        if (!pattern.test(result)) continue
        const chunkDir = path.dirname(chunk.fileName)
        let relPath = path.posix.relative(chunkDir, builtEntry)
        if (!relPath.startsWith('.')) relPath = './' + relPath
        result = result.replace(
          new RegExp(
            `String\\(["']${sourcePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}["']\\)`,
            'g'
          ),
          `'${relPath}'`
        )
        modified = true
      }
      return modified ? result : null
    },
  }
}

/**
 * After the main library build, re-bundles the audio worklet entry point into
 * a single self-contained JS file with all dependencies inlined (except WASM
 * externals). AudioWorklet's addModule() can't resolve ES module imports, so
 * the file loaded by addModule must be fully self-contained.
 */
function bundleAudioWorklet(): Plugin {
  return {
    name: 'bundle-audio-worklet',
    async closeBundle() {
      const distDir = path.resolve(__dirname, 'dist')
      const workletEntry = path.resolve(distDir, 'workers/audio-worklet.js')

      // Stub out new URL() asset references that the AudioWorkletGlobalScope
      // can't resolve (no URL constructor). The worklet receives wasmBinary
      // via postMessage, so these URLs are never used at runtime.
      const stubWasmUrls = {
        name: 'stub-wasm-urls',
        renderChunk(code: string) {
          const result = code.replace(
            /new URL\(["'][^"']*["'],\s*import\.meta\.url\)\.href/g,
            '""'
          )
          return result !== code ? result : null
        },
      }

      const bundle = await rollup({
        input: workletEntry,
        plugins: [stubWasmUrls],
      })
      await bundle.write({
        file: path.resolve(distDir, 'workers/audio-worklet.bundled.js'),
        format: 'es',
        inlineDynamicImports: true,
      })
      await bundle.close()
    },
  }
}

export default defineConfig({
  base: './',
  plugins: [
    preserveWorkerUrls(),
    bundleAudioWorklet(),
    watchWasm(),
    viteStaticCopy({
      targets: [
        { src: 'wasm/*.wasm', dest: 'wasm' },
        { src: 'wasm/*.js', dest: 'wasm' },
      ],
    }),
    dts({
      include: ['src/**/*'],
      outDir: 'dist',
    }),
  ],
  build: {
    lib: {
      entry: {
        index: path.resolve(__dirname, 'src/index.ts'),
        'workers/inference-worker': path.resolve(
          __dirname,
          'src/workers/inference-worker.ts'
        ),
        'workers/audio-worklet': path.resolve(__dirname, 'src/workers/audio-worklet.ts'),
        'workers/worklet-base': path.resolve(__dirname, 'src/workers/worklet-base.ts'),
      },
      formats: ['es'],
    },
    minify: false,
    target: 'esnext',
    outDir: 'dist',
    rollupOptions: {
      external: (id) => {
        // Externalize both WASM files and emscripten-generated JS wrappers
        if (id.includes('/wasm/')) return true
        if (id.endsWith('.wasm')) return true
        return false
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'shared/[name]-[hash].js',
        paths: (id) => {
          // Rewrite paths for wasm imports to be relative to the dist folder
          if (id.includes('/wasm/')) {
            return id.replace(/.*\/wasm\//, './wasm/')
          }
          return id
        },
      },
    },
  },
  // This is a known issue when using WebAssembly with Vite 5.x
  // Need to specify `optimizeDeps.exclude` to NPM packages that uses WebAssembly
  // See: https://github.com/vitejs/vite/issues/8427
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
