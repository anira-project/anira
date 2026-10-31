// Stopping a context's Wasm Workers while they still boot: the Workers load the module's glue,
// import boot-worker.js and await its engine's init (150 ms), and the context is destroyed after
// 0 to 200 ms, so the stop lands in each of those steps. anira never terminates a Worker there
// (a browser reports that as an error: Firefox NS_ERROR_FAILURE, "uncaught exception"): the
// stop cancels the boot, the Worker finishes it without entering the loop, and a main-thread
// reaper frees its 4 MiB region once it is done. run.mjs fails this page on any console error.
// CYCLES cycles of two Workers would leak far more than the module's memory holds if a region
// were not freed (the core state alone takes about 180 MB while a context lives), so the cycles
// running out and a last context whose pool enters its loops show that they are. The verdict
// is window.anira_test_result.
import {
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
} from '../../dist/index.js'

const QUANTUM = 128
const ENGINE = 'org.anira-test.slow-init'
const DELAYS_MS = [0, 1, 5, 10, 20, 50, 100, 200]
const CYCLES = 64
const THREADS = 2
const NODES = 4

const streamed = (name) => ({
  name,
  dtype: 'float32',
  role: 'streamed',
  axes: [
    ['batch', 1],
    ['channel', 1],
    ['time', QUANTUM],
  ],
  window: { min: QUANTUM, max: QUANTUM },
})

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const prepared = (anira, context, model) => {
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .addEngine(ENGINE, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
  const contract = ContractHandle.hard(anira, {
    blockMin: QUANTUM,
    rate: 48000,
    budgetMs: 2,
  })
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  pipeline.destroy()
  contract.destroy()
  return handler
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogSink(() => {})
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ENGINE }],
      inputs: [streamed('in')],
      outputs: [streamed('out')],
    })
  )
  const workerModule = new URL('./boot-worker.js', import.meta.url)

  const started = performance.now()
  for (let cycle = 0; cycle < CYCLES; cycle++) {
    const context = Context.create(anira, { threads: THREADS, workerModule })
    const handler = prepared(anira, context, model)
    await delay(DELAYS_MS[cycle % DELAYS_MS.length])
    handler.destroy()
    context.destroy()
  }
  check(
    true,
    `${CYCLES} contexts of ${THREADS} Workers each stopped 0 to 200 ms into their boot ` +
      `(${((performance.now() - started) / 1000).toFixed(1)} s)`
  )
  check(
    anira.abi.anira_em_inference_threads_in_loop() === 0,
    'no cancelled Worker entered its loop'
  )

  // The cancelled Workers finish their boot and are reaped; then a context works as before.
  await delay(1000)
  const context = Context.create(anira, { threads: THREADS, workerModule })
  const handler = prepared(anira, context, model)
  await context.ready()
  check(
    anira.abi.anira_em_inference_threads_in_loop() === THREADS,
    `after the cycles a context's ${THREADS} Workers boot and enter their loops`
  )

  // The reaper's timers fired many times by now (emscripten_set_timeout, whose callback ends in
  // the runtime's maybeExit): the runtime is still alive, so the asynchronous creations of
  // audio nodes (the worklet thread, the processor, the node, each a callback into C) still
  // complete, and the last node's C render callback streams through the handler.
  const audio = new OfflineAudioContext(1, QUANTUM * 16, 48000)
  for (let n = 0; n < NODES - 1; n++) (await handler.createAudioNode(audio)).destroy()
  const audioNode = await handler.createAudioNode(audio, { waitMs: 500 })
  const tone = new Float32Array(QUANTUM * 16).fill(0.25)
  const buffer = audio.createBuffer(1, tone.length, 48000)
  buffer.copyToChannel(tone, 0)
  const source = new AudioBufferSourceNode(audio, { buffer })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  const latency = handler.getLatency()
  check(
    rendered[rendered.length - 1] === 0.25 && rendered[latency] === 0.25,
    `after the timers ${NODES} audio nodes were created and the last one rendered through ` +
      `the handler (latency ${latency})`
  )
  audioNode.destroy()
  handler.destroy()
  context.destroy()
  check(context.numInferenceThreads() === 0, 'and stop with the context')
  model.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
