// The bundle check of the built package (after npm run build; npm run examples for the second
// half): a page that only names the ONNX Runtime Web engine (ONNXRUNTIME_WEB_ENGINE_ID from the
// root entry) must not bundle onnxruntime-web. Follows every static import of dist/index.js and
// dist/scope.js (the module's own glue, wasm/AniraWeb.js, aside) and requires that none of the
// modules carries onnxruntime-web's code, that the root entry exports the id, and that the
// examples' built chunks carry none either. Exit code 0 when all hold.
// usage: node test/bundle-check.mjs   (from web/)
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const dist = path.join(root, 'dist')
// What onnxruntime-web's glue carries and anira's own modules never do.
const ORT_MARKERS = ['_OrtRun', '_OrtCreateSession', 'ort-wasm-simd-threaded']
const failures = []

const staticImports = (source) =>
  [
    ...source.matchAll(
      /(?:^|[;\s}])(?:import|export)\s*(?:[^'"()]*?from\s*)?["']([^"']+)["']/g
    ),
  ]
    .map((match) => match[1])
    .filter((specifier) => specifier.startsWith('.'))

const graph = (entry) => {
  const seen = new Set()
  const pending = [entry]
  while (pending.length > 0) {
    const file = pending.pop()
    if (seen.has(file)) continue
    seen.add(file)
    for (const specifier of staticImports(fs.readFileSync(file, 'utf8'))) {
      if (/wasm\/AniraWeb(\.js)?$/.test(specifier)) continue
      const target = path.resolve(path.dirname(file), specifier)
      pending.push(fs.existsSync(target) ? target : `${target}.js`)
    }
  }
  return [...seen]
}

const ortIn = (file) =>
  ORT_MARKERS.filter((marker) => fs.readFileSync(file, 'utf8').includes(marker))

for (const entry of ['index.js', 'scope.js']) {
  const modules = graph(path.join(dist, entry))
  const before = failures.length
  for (const file of modules) {
    const found = ortIn(file)
    if (found.length > 0) {
      failures.push(
        `dist/${entry} imports ${path.relative(root, file)}, which carries ${found}`
      )
    }
  }
  if (failures.length === before) {
    console.log(`dist/${entry}: ${modules.length} modules, no onnxruntime-web code`)
  }
}
const index = fs.readFileSync(path.join(dist, 'index.js'), 'utf8')
if (!/export\s*\{[^}]*\bONNXRUNTIME_WEB_ENGINE_ID\b[^}]*\}/.test(index)) {
  failures.push('dist/index.js does not export ONNXRUNTIME_WEB_ENGINE_ID')
}
const engine = path.join(dist, 'engines/onnxruntime-web.js')
if (graph(engine).every((file) => ortIn(file).length === 0)) {
  failures.push(
    'the check finds no onnxruntime-web code in the engine entry either: stale markers'
  )
}

for (const example of ['processor-and-engine', 'model-file']) {
  const assets = path.join(root, 'examples', example, 'dist', 'assets')
  if (!fs.existsSync(assets)) {
    failures.push(`examples/${example} is not built (npm run examples)`)
    continue
  }
  const before = failures.length
  for (const name of fs.readdirSync(assets).filter((n) => n.endsWith('.js'))) {
    const found = ortIn(path.join(assets, name))
    if (found.length > 0)
      failures.push(`examples/${example}/dist/assets/${name} carries ${found}`)
  }
  if (failures.length === before) {
    console.log(`examples/${example}: no onnxruntime-web code in its chunks`)
  }
}

for (const failure of failures) console.log(`FAILED: ${failure}`)
console.log(
  failures.length === 0 ? 'PASSED: the bundle check' : 'FAILED: the bundle check'
)
process.exit(failures.length === 0 ? 0 : 1)
