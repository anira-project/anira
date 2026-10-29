// The browser test of the built package: serves web/ with the COOP/COEP headers that cross-origin
// isolation (SharedArrayBuffer, the shared WebAssembly.Memory) needs, opens each page in a
// headless browser (Playwright) and prints its checks: test/browser/index.html, ort.html,
// config.html, streams.html, boot-stop.html and concurrent.html (the package as built) and the check pages of the two examples, stage-and-engine
// and model-file (as their own Vite builds bundled them). The repository's extras/models is served at /extras/models/, so a page
// fetches the bundled model configs and their model files (cmake -P extras/fetch-models.cmake
// fetches those) the way a site serves its own. Then the stage-and-engine check page once more under
// Vite's dev server (vite's createServer over the example's config), whose transform of the
// module's glue the worklet scope must survive (decision D2; cmake/wasm-glue-rewrite.cmake). Exit
// code 0 when every check passed and the browser refused no load.
// usage: node test/browser/run.mjs [chromium|firefox] [page...]
//   (after npm run build and npm run examples, from web/; CI runs both browsers; the pages
//   default to all of them and the dev-server run, which a page `dev` names alone)
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, firefox } from 'playwright'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const models = path.resolve(root, '../extras/models')
const MODELS_PREFIX = '/extras/models/'
const types = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
}

// A bundler resolves the package's extensionless imports of the Emscripten module
// ('../wasm/AniraWeb'); this server does it for the page by trying `.js`.
const resolveFile = (file) => {
  if (fs.existsSync(file) && fs.statSync(file).isFile()) return file
  if (path.extname(file) === '' && fs.existsSync(`${file}.js`)) return `${file}.js`
  return null
}

// The requests served, by path: a page reads them from /__requests (ort.html: the model is
// fetched once, by the main thread, not once per Worker).
const requestCounts = {}

const server = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
  if (pathname === '/__requests') {
    response.writeHead(200, {
      'Content-Type': 'application/json',
      'Cross-Origin-Resource-Policy': 'same-origin',
    })
    response.end(JSON.stringify(requestCounts))
    return
  }
  requestCounts[pathname] = (requestCounts[pathname] ?? 0) + 1
  const inModels = pathname.startsWith(MODELS_PREFIX)
  const base = inModels ? models : root
  const file = resolveFile(
    path.join(base, inModels ? pathname.slice(MODELS_PREFIX.length) : pathname)
  )
  if (!file || !file.startsWith(base)) {
    response.writeHead(404)
    response.end()
    return
  }
  response.writeHead(200, {
    'Content-Type': types[path.extname(file)] ?? 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
  fs.createReadStream(file).pipe(response)
})
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
const { port } = server.address()

const allPages = [
  'test/browser/index.html',
  'test/browser/ort.html',
  'test/browser/config.html',
  'test/browser/streams.html',
  'test/browser/boot-stop.html',
  'test/browser/concurrent.html',
  'examples/stage-and-engine/dist/check.html',
  'examples/model-file/dist/check.html',
]
const browserName = process.argv[2] ?? 'chromium'
const browserType = { chromium, firefox }[browserName]
if (!browserType) throw new Error(`no browser '${browserName}': chromium or firefox`)
// live.html needs a live AudioContext that runs: headless Firefox keeps one suspended (no audio
// output to run against), so the page runs in Chromium alone.
// Pages on which any console error or uncaught exception fails (a stop must not log one).
const strictPages = new Set(['test/browser/boot-stop.html'])
const livePages = browserName === 'chromium' ? ['test/browser/live.html'] : []
const pages =
  process.argv.length > 3 ? process.argv.slice(3) : [...allPages, ...livePages, 'dev']
// The example's check page under Vite's dev server.
const DEV_PAGE = 'check.html'
const devServer = async () => {
  const { createServer } = await import('vite')
  const dev = await createServer({
    configFile: path.join(root, 'examples/stage-and-engine/vite.config.ts'),
    server: { port: 0, host: '127.0.0.1' },
    logLevel: 'error',
  })
  await dev.listen()
  return dev
}

// A live AudioContext must run without a user gesture (live.html).
const browser = await browserType.launch(
  browserName === 'chromium'
    ? { args: ['--autoplay-policy=no-user-gesture-required'] }
    : {}
)
let failed = 0
try {
  for (const pathname of pages) {
    console.log(
      `-- ${pathname === 'dev' ? `examples/stage-and-engine/${DEV_PAGE} (vite dev)` : pathname}`
    )
    const dev = pathname === 'dev' ? await devServer() : null
    const page = await browser.newPage()
    page.on('console', (message) => {
      console.log(`page: ${message.text()}`)
      // A load the browser refused, such as Firefox's "Security Error: Content at ... may not
      // load or link to file:///...", fails the page even when its checks pass.
      if (/Security Error|may not load or link to/.test(message.text())) {
        console.log('FAILED: the browser refused a load (above)')
        failed += 1
      } else if (strictPages.has(pathname) && message.type() === 'error') {
        console.log('FAILED: a console error on a page that must log none (above)')
        failed += 1
      }
    })
    page.on('pageerror', (error) => {
      console.log(`page error: ${error.message}`)
      if (strictPages.has(pathname)) {
        console.log(
          'FAILED: an uncaught exception on a page that must throw none (above)'
        )
        failed += 1
      }
    })
    await page.goto(
      dev
        ? `${dev.resolvedUrls.local[0]}${DEV_PAGE}`
        : `http://127.0.0.1:${port}/${pathname}`
    )
    await page.waitForFunction(() => window.anira_test_result, null, { timeout: 120000 })
    const result = await page.evaluate(() => window.anira_test_result)
    if (result.error) {
      console.log(`FAILED: ${result.error}`)
      failed += 1
    } else {
      for (const { ok, what } of result.checks) {
        console.log(`${ok ? 'ok' : 'FAILED'}: ${what}`)
        if (!ok) failed += 1
      }
    }
    await page.close()
    await dev?.close()
  }
} finally {
  await browser.close()
  server.close()
}
console.log(`${failed === 0 ? 'PASSED' : 'FAILED'}: the browser test`)
process.exit(failed === 0 ? 0 : 1)
