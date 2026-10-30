import path from 'path'
import { defineConfig, type Plugin } from 'vite'
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
 * Prevents Vite's library-mode build from inlining `new URL(path, import.meta.url)` references as
 * base64 data URLs. The built output keeps the `new URL()` pattern with the path of the built file
 * (relative to the chunk), so that a consumer's bundler sees it and copies the file: the module's
 * glue and binary (locateFile of Anira.ts), the worklet scope's polyfill (AudioNode.ts), and
 * onnxruntime-web's own binary (its 12.8 MB would otherwise be 17 MB of base64 in every chunk of
 * the ONNX Runtime Web engine).
 */
function preserveUrls(): Plugin {
  const rewrites: Array<{ sourcePath: string; builtEntry: string }> = [
    { sourcePath: './worklet-scope.ts', builtEntry: 'worklet-scope.js' },
    { sourcePath: '../wasm/AniraWeb.js', builtEntry: 'wasm/AniraWeb.js' },
    { sourcePath: '../wasm/AniraWeb.wasm', builtEntry: 'wasm/AniraWeb.wasm' },
    {
      sourcePath: 'ort-wasm-simd-threaded.wasm',
      builtEntry: 'wasm/ort-wasm-simd-threaded.wasm',
    },
  ]
  const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  return {
    name: 'preserve-urls',
    enforce: 'pre',

    // In transform (pre): wrap the path in String() so Vite's assetImportMetaUrl plugin won't
    // match it (it requires a plain literal).
    transform(code) {
      let result = code
      let modified = false
      for (const { sourcePath } of rewrites) {
        // Either quote, with or without a space: the package's sources and the minified ORT.
        const needle = new RegExp(
          `new URL\\((['"])${escape(sourcePath)}\\1,\\s*import\\.meta\\.url\\)`,
          'g'
        )
        if (needle.test(result)) {
          result = result.replace(needle, `new URL(String('${sourcePath}'), import.meta.url)`)
          modified = true
        }
      }
      return modified ? result : null
    },

    // In renderChunk: replace String('source-path') with the literal path of the built file,
    // relative to the chunk, restoring the standard new URL() form.
    renderChunk(code, chunk) {
      let result = code
      let modified = false
      for (const { sourcePath, builtEntry } of rewrites) {
        const pattern = new RegExp(`String\\(["']${escape(sourcePath)}["']\\)`, 'g')
        if (!pattern.test(result)) continue
        let relPath = path.posix.relative(path.dirname(chunk.fileName), builtEntry)
        if (!relPath.startsWith('.')) relPath = './' + relPath
        result = result.replace(pattern, `'${relPath}'`)
        modified = true
      }
      return modified ? result : null
    },
  }
}

export default defineConfig({
  base: './',
  resolve: {
    alias: [
      // onnxruntime-web does not expose ort-wasm-simd-threaded.mjs in its package `exports`;
      // resolve it directly from dist/.
      {
        find: 'onnxruntime-web/ort-wasm-simd-threaded.mjs',
        replacement: path.resolve(
          __dirname,
          'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs'
        ),
      },
    ],
  },
  plugins: [
    preserveUrls(),
    watchWasm(),
    viteStaticCopy({
      targets: [
        { src: 'wasm/*.wasm', dest: 'wasm' },
        { src: 'wasm/*.js', dest: 'wasm' },
        // The ONNX Runtime Web engine's binary, loaded by URL (preserveUrls).
        {
          src: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
          dest: 'wasm',
        },
        // Attribution for native deps statically linked into the WASM binary
        // (populated by cmake/build-wasm.cmake from the downloaded prebuilt).
        { src: 'licenses/*', dest: 'licenses' },
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
        scope: path.resolve(__dirname, 'src/scope.ts'),
        'worklet-scope': path.resolve(__dirname, 'src/worklet-scope.ts'),
        'engines/onnxruntime-web': path.resolve(__dirname, 'src/engines/onnxruntime-web.ts'),
      },
      formats: ['es'],
    },
    minify: false,
    target: 'esnext',
    outDir: 'dist',
    rollupOptions: {
      external: (id) => id.includes('/wasm/') || id.endsWith('.wasm'),
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: 'shared/[name]-[hash].js',
        // The module's glue, relative to the chunk that imports it.
        paths: (id) => (id.includes('/wasm/') ? id.replace(/.*\/wasm\//, './wasm/') : id),
      },
    },
  },
  // Packages that load WebAssembly themselves are not pre-bundled by Vite's dev server
  // (https://github.com/vitejs/vite/issues/8427).
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
