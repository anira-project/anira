// The driver of em_smoke_thread.c (decision 9.12): the main instance on a shared memory, a
// second instance in a worker_threads Worker (em_smoke_thread_worker.mjs) that runs
// anira_inference_thread_run_loop for real, blocks pushed through the handler from here while
// the Worker's loop runs them, then the stop: has_exited within a bound, and the teardown.
// usage: node em_smoke_thread.mjs <anira_em_smoke_thread.mjs>
import { once } from 'node:events'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { Worker } from 'node:worker_threads'

const BLOCKS = 16
const IDLE_MS = 50
const STOP_BOUND_MS = 100

const modulePath = resolve(process.argv[2])
const factory = (await import(pathToFileURL(modulePath).href)).default
// The module's INITIAL_MEMORY (256 MiB), shared.
const memory = new WebAssembly.Memory({ initial: 4096, maximum: 4096, shared: true })
// Main runs no hook: the engine's process crosses on the Worker.
const refuse = () => {
  throw new Error('the main instance runs no hook in this test')
}
const main = await factory({
  wasmMemory: memory,
  anira: { version: 1, stage: refuse, engine: refuse, log() {}, complete: undefined },
})

let failures = 0
const check = (ok, what) => {
  if (!ok) {
    console.log(`FAILED: ${what}`)
    failures += 1
  }
}

failures += main._smoke_thread_setup()
const worker = new Worker(new URL('./em_smoke_thread_worker.mjs', import.meta.url), {
  workerData: {
    modulePath,
    memory,
    thread: main._smoke_thread() >>> 0,
    stackBase: main._smoke_stack_base() >>> 0,
    stackBytes: main._smoke_stack_bytes() >>> 0,
  },
})
const [entering] = await once(worker, 'message')
check(entering.type === 'entering', `the Worker enters its loop (${JSON.stringify(entering)})`)
const exited = once(worker, 'message')

// An idle loop parks on the wake word; the first block's enqueue wakes it.
await new Promise((r) => setTimeout(r, IDLE_MS))
const matched = main._smoke_thread_run_blocks(BLOCKS)
console.log(`blocks doubled by the Worker's JavaScript engine: ${matched} of ${BLOCKS}`)
check(matched >= BLOCKS - 1, "the Worker's loop ran the blocks")

// Parked again; the stop's notify wakes it, and has_exited follows within the bound.
await new Promise((r) => setTimeout(r, IDLE_MS))
const stopMs = main._smoke_thread_stop(1000)
console.log(`the Worker left its loop ${stopMs.toFixed(2)} ms after the stop`)
check(stopMs >= 0 && stopMs < STOP_BOUND_MS, `has_exited within ${STOP_BOUND_MS} ms of the stop`)
const [left] = await exited
check(left.type === 'exited', `the Worker returned from run_loop (${JSON.stringify(left)})`)
check(left.processed >= BLOCKS - 1, `the Worker's engine processed the blocks (${left.processed})`)
await worker.terminate()

failures += main._smoke_thread_teardown()
console.log(`${failures === 0 ? 'PASSED' : 'FAILED'}: ${failures} failed check(s)`)
process.exit(failures === 0 ? 0 : 1)
