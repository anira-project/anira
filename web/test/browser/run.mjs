// The browser test of the built package: serves web/ with the COOP/COEP headers that cross-origin
// isolation (SharedArrayBuffer, the shared WebAssembly.Memory) needs, opens test/browser/ in
// headless Chromium (Playwright) and prints the page's checks. Exit code 0 when every check
// passed. usage: node test/browser/run.mjs   (after npm run build, from web/)
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const types = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
}

// A bundler resolves the package's extensionless imports of the Emscripten module
// ('../wasm/AniraWeb'); this server does it for the page by trying `.js`.
const resolveFile = (file) => {
  if (fs.existsSync(file) && fs.statSync(file).isFile()) return file
  if (path.extname(file) === '' && fs.existsSync(`${file}.js`)) return `${file}.js`
  return null
}

const server = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
  const file = resolveFile(path.join(root, pathname))
  if (!file || !file.startsWith(root)) {
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

const browser = await chromium.launch()
let failed = 0
try {
  const page = await browser.newPage()
  page.on('console', (message) => console.log(`page: ${message.text()}`))
  page.on('pageerror', (error) => console.log(`page error: ${error.message}`))
  await page.goto(`http://127.0.0.1:${port}/test/browser/index.html`)
  await page.waitForFunction(() => window.anira_test_result, null, { timeout: 120000 })
  const result = await page.evaluate(() => window.anira_test_result)
  if (result.error) {
    console.log(`FAILED: ${result.error}`)
    failed = 1
  } else {
    for (const { ok, what } of result.checks) {
      console.log(`${ok ? 'ok' : 'FAILED'}: ${what}`)
      if (!ok) failed += 1
    }
  }
} finally {
  await browser.close()
  server.close()
}
console.log(`${failed === 0 ? 'PASSED' : 'FAILED'}: the browser test`)
process.exit(failed === 0 ? 0 : 1)
