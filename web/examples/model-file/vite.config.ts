// The example's build: cross-origin isolation (the module's memory is a shared
// WebAssembly.Memory), the package left as it is by the dev server, and the repository's
// extras/models served at /extras/models/ by `vite` and `vite preview` (a production server serves
// the configs and their model files the same way: beside each other, as the configs' paths say).
// No worker or worklet module: the built-in engine needs no JavaScript.
import fs from 'node:fs'
import path from 'node:path'
import { defineConfig, type Connect, type Plugin } from 'vite'

const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

const MODELS_PREFIX = '/extras/models/'
const models = path.resolve(__dirname, '../../../extras/models')

const serveModels: Connect.NextHandleFunction = (request, response, next) => {
  const pathname = decodeURIComponent((request.url ?? '').split('?')[0])
  if (!pathname.startsWith(MODELS_PREFIX)) return next()
  const file = path.join(models, pathname.slice(MODELS_PREFIX.length))
  if (!file.startsWith(models + path.sep) || !fs.existsSync(file)) return next()
  for (const [name, value] of Object.entries(crossOriginIsolation))
    response.setHeader(name, value)
  fs.createReadStream(file).pipe(response)
}

const extrasModels = (): Plugin => ({
  name: 'anira-extras-models',
  configureServer: (server) => void server.middlewares.use(serveModels),
  configurePreviewServer: (server) => void server.middlewares.use(serveModels),
})

export default defineConfig({
  root: __dirname,
  base: './',
  plugins: [extrasModels()],
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  optimizeDeps: { exclude: ['@anira-project/anira'] },
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
