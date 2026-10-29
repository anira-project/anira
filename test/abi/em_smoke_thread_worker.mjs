// The Worker of em_smoke_thread.mjs: a second instance of the module on the main instance's
// memory. It enters its own stack and thread-local block (anira_em_init_instance, its first C
// call), installs a hook object whose JavaScript engine doubles its input, and runs
// anira_inference_thread_run_loop until main stops the thread.
import { pathToFileURL } from 'node:url'
import { parentPort, workerData } from 'node:worker_threads'

const { modulePath, memory, thread, stackBase, stackBytes } = workerData
const factory = (await import(pathToFileURL(modulePath).href)).default

let processed = 0
let layout = null
const engine = (key, phase, ctx) => {
  // ANIRA_PHASE_INFERENCE (web/src/abi/enums.ts)
  if (phase !== 3) return -1
  processed += 1
  const m = instance
  const input = m.HEAPU32[(ctx + layout.inputs) >>> 2]
  const output = m.HEAPU32[(ctx + layout.outputs) >>> 2]
  const src = (m._anira_tensor_data_f32(input) >>> 0) >>> 2
  const dst = (m._anira_tensor_data_f32(output) >>> 0) >>> 2
  const n = m._anira_tensor_num_elements(input) >>> 0
  for (let i = 0; i < n; i++) m.HEAPF32[dst + i] = 2 * m.HEAPF32[src + i]
  return 0
}
const refuse = () => {
  throw new Error('the Worker runs no stage in this test')
}
const instance = await factory({
  wasmMemory: memory,
  anira: { version: 1, stage: refuse, engine, log() {}, complete: undefined },
})
instance.stackRestore(stackBase + stackBytes)
if (instance._anira_em_init_instance(stackBase, stackBytes) !== 0) {
  parentPort.postMessage({ type: 'error', message: 'anira_em_init_instance failed' })
} else {
  layout = { inputs: instance._smoke_ctx_inputs_offset(), outputs: instance._smoke_ctx_outputs_offset() }
  parentPort.postMessage({ type: 'entering' })
  instance._anira_inference_thread_run_loop(thread)
  parentPort.postMessage({ type: 'exited', processed })
}
