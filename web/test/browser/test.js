// The browser test of the built package (web/dist), run by run.mjs in headless Chromium: the
// main instance, a JavaScript engine that doubles on an inference Worker, a Hard handler of
// block 128, and an OfflineAudioContext rendered through the anira worklet node. The output
// must be the input doubled after the handler's latency; then the Worker is stopped (its
// has_exited stop) and the worklet disposed. The verdict is window.anira_test_result.
import {
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  AniraWeb,
  anira_budget_kind,
  anira_log_drain,
  anira_status,
  anira_wait_strategy,
  anira_warmup_mode,
  callWithError,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const ENGINE_ID = 'org.anira-test.double'

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  check(crossOriginIsolated, 'the page is cross-origin isolated (COOP/COEP)')

  const anira = await AniraWeb.create()
  const { module: m, heap } = anira
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const out = heap.malloc(4)
  const created = (call) => {
    callWithError(heap, call)
    return heap.readU32(out)
  }

  // A refused call throws and catches a C++ exception inside the module: it must come back as
  // its status, which needs the main instance's thread-local block (createAniraWasm).
  const refusal = heap.withCString('{ not json', (json) => {
    const err = heap.malloc(520)
    const status = m._anira_model_config_from_json(json, 10, 0, out, err)
    heap.free(err)
    return status
  })
  check(
    refusal === anira_status.ANIRA_ERROR_JSON,
    `a refused call returns its status (${refusal})`
  )

  const contextConfig = created((err) => m._anira_context_config_create(out, err))
  m._anira_context_config_set_threads(
    contextConfig,
    0,
    anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF
  )
  m._anira_context_config_set_log_drain(
    contextConfig,
    anira_log_drain.ANIRA_LOG_DRAIN_MANUAL,
    10
  )
  const context = created((err) => m._anira_context_create(contextConfig, out, err))

  const modelJson = JSON.stringify({
    models: [{ engine: ENGINE_ID, path: 'never-opened.bin' }],
    inputs: [
      {
        name: 'in',
        dtype: 'float32',
        role: 'streamed',
        axes: [
          ['batch', 1],
          ['channel', 1],
          ['time', QUANTUM],
        ],
        window: { min: QUANTUM, max: QUANTUM },
      },
    ],
    outputs: [
      {
        name: 'out',
        dtype: 'float32',
        role: 'streamed',
        axes: [
          ['batch', 1],
          ['channel', 1],
          ['time', QUANTUM],
        ],
        window: { min: QUANTUM, max: QUANTUM },
      },
    ],
  })
  const model = heap.withCString(modelJson, (json) =>
    created((err) => m._anira_model_config_from_json(json, modelJson.length, 0, out, err))
  )
  const pipeline = created((err) => m._anira_pipeline_create(out, err))
  const variants = heap.malloc(4)
  heap.writeU32(variants, model)
  callWithError(heap, (err) =>
    m._anira_pipeline_add_inference(pipeline, variants, 1, 0, 0, err)
  )
  await anira.registerJSEngine(pipeline, {
    engineId: ENGINE_ID,
    className: 'DoubleEngine',
    modelConfigPtr: model,
    engineFlags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  })

  const handler = created((err) => m._anira_handler_create(context, pipeline, out, err))
  const contract = created((err) =>
    m._anira_contract_create_hard(QUANTUM, QUANTUM, RATE, out, err)
  )
  m._anira_contract_hard_set_budget(
    contract,
    anira_budget_kind.ANIRA_BUDGET_EXPLICIT,
    2.0
  )
  m._anira_contract_hard_set_warmup(contract, anira_warmup_mode.ANIRA_WARMUP_NONE, 0)
  callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))
  const latency = m._anira_handler_get_latency(handler, 0) >>> 0

  const worker = await anira.spinUpInferenceWorker(
    context,
    new URL('./double-worker.js', import.meta.url)
  )

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  await anira.registerAudioWorkletForContext(
    audio,
    new URL('../../dist/workers/audio-worklet.bundled.js', import.meta.url)
  )
  // The offline render runs faster than real time: wait for each quantum's inference.
  const node = await anira.configureAudioWorklet(audio, handler, {
    inputChannels: 1,
    outputChannels: 1,
    waitMs: 500,
  })
  const input = audio.createBuffer(1, QUANTUM * QUANTA, RATE)
  const samples = input.getChannelData(0)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const source = audio.createBufferSource()
  source.buffer = input
  source.connect(node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)

  let worst = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - 2 * samples[i - latency]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the worklet rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input doubled after ${latency} samples (worst error ${worst})`
  )

  await worker.stop()
  check(
    anira.getActiveWorkers().length === 0,
    'the Worker stopped (has_exited) and was destroyed'
  )
  await anira.disposeAudioWorklet(node)
  check(true, 'the worklet was disposed')

  m._anira_handler_destroy(handler)
  m._anira_pipeline_destroy(pipeline)
  m._anira_contract_destroy(contract)
  m._anira_model_config_destroy(model)
  m._anira_context_destroy(context)
  m._anira_context_config_destroy(contextConfig)
  heap.free(variants)
  heap.free(out)
  return { latency, worst, checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch((error) => (window.anira_test_result = { error: String(error?.stack ?? error) }))
