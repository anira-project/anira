// The context's pool on the web: a context of two inference threads, which anira starts as Wasm
// Workers (InferenceThread::start), each importing pool-worker.js (the worker module of the
// context config, anira_em_context_config_set_worker_module) before its loop: a JavaScript engine
// defined on the Worker's registry (defineEngine), no inference Worker of the package. The page
// pushes blocks through a Hard handler over the engine and waits for each on its own thread
// (anira_handler_process_wait); the output must be the input times 2 * the Static gain after the
// handler's latency. Then the context goes and with it the pool. The verdict is
// window.anira_test_result.
import {
  ANIRA_DTYPE_F32,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  AniraWeb,
  Contract,
  ModelConfig,
  anira_log_drain,
  anira_wait_strategy,
  keyOf,
  layout,
} from '../../dist/index.js'

const RATE = 48000
const BLOCK = 128
const BLOCKS = 32
const GAIN = 0.75
const THREADS = 2
const ENGINE_ID = 'org.anira-test.pool-double'

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await AniraWeb.create()
  const { abi } = anira
  const m = abi.module
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))

  const contextConfig = abi.anira_context_config_create()
  abi.anira_context_config_set_threads(
    contextConfig,
    THREADS,
    anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF
  )
  abi.anira_context_config_set_log_drain(
    contextConfig,
    anira_log_drain.ANIRA_LOG_DRAIN_MANUAL,
    10
  )
  abi.anira_em_context_config_set_worker_module(
    contextConfig,
    new URL('./pool-worker.js', import.meta.url).href
  )
  const context = abi.anira_context_create(contextConfig)

  const streamed = (name) => ({
    name,
    dtype: 'float32',
    role: 'streamed',
    axes: [
      ['batch', 1],
      ['channel', 1],
      ['time', BLOCK],
    ],
    window: { min: BLOCK, max: BLOCK },
  })
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ENGINE_ID, path: 'never-opened.bin' }],
      inputs: [
        streamed('in'),
        { name: 'gain', dtype: 'float32', role: 'static', axes: [['any', 1]] },
      ],
      outputs: [streamed('out')],
    }),
    { fetchModel: () => false }
  )
  const pipeline = abi.anira_pipeline_create()
  const variants = m._malloc(4)
  m.HEAPU32[variants >>> 2] = model.ptr
  abi.anira_pipeline_add_inference(pipeline, variants, 1, 0, 0)
  abi.anira_em_pipeline_register_js_engine(
    pipeline,
    ENGINE_ID,
    keyOf(ENGINE_ID),
    0,
    ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL
  )
  const handler = abi.anira_handler_create(context, pipeline)
  const contract = Contract.hard(anira, {
    blockMin: BLOCK,
    rate: RATE,
    budgetMs: 2,
    warmup: 0,
  })
  abi.anira_handler_prepare(handler, contract.ptr)
  check(
    abi.anira_num_inference_threads() === THREADS,
    `the context's pool runs ${THREADS} inference threads (${abi.anira_num_inference_threads()})`
  )
  const latency = abi.anira_handler_get_latency(handler, 0)
  // The Workers start once this thread yields, then import their module: wait until both are
  // inside their loops before the blocks below wait for them on this thread.
  const start = performance.now()
  while (abi.anira_em_inference_threads_in_loop() < THREADS) {
    if (performance.now() - start > 10000) throw new Error('the pool did not start')
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  check(
    true,
    `the pool's Wasm Workers entered their loops ${(performance.now() - start).toFixed(0)} ms ` +
      'after the prepare'
  )

  const tensor = (elements, shape) => {
    const data = m._malloc(elements * 4)
    const dims = m._malloc(shape.length * 8)
    shape.forEach((extent, i) => (m.HEAP64[(dims >>> 3) + i] = BigInt(extent)))
    const record = m._malloc(layout.anira_tensor.size)
    abi.anira_tensor_init_host(record, data, ANIRA_DTYPE_F32, shape.length, dims)
    return { record, data, dims }
  }
  const input = tensor(BLOCK, [1, BLOCK])
  const output = tensor(BLOCK, [1, BLOCK])
  const gain = tensor(1, [1])
  m.HEAPF32[gain.data >>> 2] = GAIN
  abi.anira_handler_set_static_input(handler, 1, gain.record)

  const samples = new Float32Array(BLOCK * BLOCKS)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const rendered = new Float32Array(BLOCK * BLOCKS)
  for (let block = 0; block < BLOCKS; block++) {
    m.HEAPF32.set(samples.subarray(block * BLOCK, (block + 1) * BLOCK), input.data >>> 2)
    // The first blocks wait for the Workers to import their module and init the engine.
    const delivered = abi.anira_handler_process_wait(
      handler,
      input.record,
      0,
      output.record,
      0,
      5000
    )
    if (delivered === BLOCK) {
      rendered.set(
        m.HEAPF32.subarray(output.data >>> 2, (output.data >>> 2) + BLOCK),
        block * BLOCK
      )
    }
  }
  let worst = 0
  let compared = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - 2 * GAIN * samples[i - latency]))
    compared++
  }
  check(
    compared > 0 && worst < 1e-6,
    `the pool's Wasm Workers ran the engine: the output is the input times 2 * ${GAIN} after ` +
      `${latency} samples (${compared} samples, worst error ${worst})`
  )

  abi.anira_handler_destroy(handler)
  abi.anira_pipeline_destroy(pipeline)
  contract.destroy()
  model.destroy()
  abi.anira_context_destroy(context)
  abi.anira_context_config_destroy(contextConfig)
  check(
    abi.anira_num_inference_threads() === 0,
    'the pool stopped with the last context (its Workers left their loops and were terminated)'
  )
  for (const t of [input, output, gain]) for (const p of Object.values(t)) m._free(p)
  m._free(variants)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
