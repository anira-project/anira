// The ONNX Runtime Web engine in headless Chromium, on the package's default inference Worker
// (spinUpInferenceWorker(context) without a worker file): the SimpleGainNetwork model of
// extras/models (output = input * gain, its 476 bytes inlined below as a data: URL the engine
// fetches), a Hard handler of block 128 over its window of 512, and an OfflineAudioContext
// rendered through the default worklet, which serves the Static gain. The engine's init loads
// onnxruntime-web's binary by URL (dist/wasm/ort-wasm-simd-threaded.wasm). The verdict is
// window.anira_test_result.
import {
  AniraWeb,
  ONNXRUNTIME_WEB_ENGINE_ID,
  anira_budget_kind,
  anira_log_drain,
  anira_wait_strategy,
  anira_warmup_mode,
  callWithError,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const WINDOW = 512
const GAIN = 0.5
// extras/models/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.onnx
const MODEL_URL =
  'data:application/octet-stream;base64,CAYSB3B5dG9yY2gaBjIuMTIuMTrCAwonCgRkYXRhCgRnYWluEg5wcm9jZXNzZWRfZGF0YRoEL011bCIDTXVsCioKDnByb2Nlc3NlZF9kYXRhEg0vQWJzX291dHB1dF8wGgQvQWJzIgNBYnMKTAoNL0Fic19vdXRwdXRfMBITL1JlZHVjZU1heF9vdXRwdXRfMBoKL1JlZHVjZU1heCIJUmVkdWNlTWF4Kg8KCGtlZXBkaW1zGACgAQIKRRISL0NvbnN0YW50X291dHB1dF8wGgkvQ29uc3RhbnQiCENvbnN0YW50KhoKBXZhbHVlKg4IARAHSggBAAAAAAAAAKABBApCChMvUmVkdWNlTWF4X291dHB1dF8wChIvQ29uc3RhbnRfb3V0cHV0XzASBHBlYWsaCC9SZXNoYXBlIgdSZXNoYXBlEgptYWluX2dyYXBoWiEKBGRhdGESGQoXCAESEwoCCAEKAggBCgkSB2R5bmFtaWNaEgoEZ2FpbhIKCggIARIECgIIAWI7Cg5wcm9jZXNzZWRfZGF0YRIpCicIARIjCgIIAQoCCAEKGRIXTXVscHJvY2Vzc2VkX2RhdGFfZGltXzJiEgoEcGVhaxIKCggIARIECgIIAUICEAs='

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await AniraWeb.create()
  const { module: m, heap } = anira
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const out = heap.malloc(4)
  const created = (call) => {
    callWithError(heap, call)
    return heap.readU32(out)
  }

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

  const streamed = (name) => ({
    name,
    dtype: 'float32',
    role: 'streamed',
    axes: [
      ['batch', 1],
      ['channel', 1],
      ['time', WINDOW],
    ],
    window: { min: WINDOW, max: WINDOW },
  })
  const scalar = (name) => ({
    name,
    dtype: 'float32',
    role: 'static',
    axes: [['any', 1]],
  })
  const modelJson = JSON.stringify({
    models: [{ engine: ONNXRUNTIME_WEB_ENGINE_ID, path: MODEL_URL }],
    inputs: [streamed('audio_in'), scalar('gain')],
    outputs: [streamed('audio_out'), scalar('gain_out')],
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
    engineId: ONNXRUNTIME_WEB_ENGINE_ID,
    className: 'ONNXRuntimeWebBackend',
    modelConfigPtr: model,
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

  // The package's default Worker: the built-in ONNXRuntimeWebBackend, whose init() fetches the
  // model and instantiates onnxruntime-web before the Worker reports the engine registered.
  const worker = await anira.spinUpInferenceWorker(context)
  check(
    true,
    'the default Worker registered the ONNX Runtime Web engine (model and binary loaded)'
  )

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  await anira.registerAudioWorkletForContext(
    audio,
    new URL('../../dist/workers/audio-worklet.bundled.js', import.meta.url)
  )
  const node = await anira.configureAudioWorklet(audio, handler, {
    inputChannels: 1,
    outputChannels: 1,
    waitMs: 500,
    staticInputs: [1],
    modelConfigPtr: model,
  })
  node.setStaticInput(1, [GAIN])
  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)

  let worst = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - GAIN * samples[i - latency]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the worklet rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times the Static gain ${GAIN} after ${latency} samples ` +
      `(worst error ${worst})`
  )

  await anira.disposeAudioWorklet(node)
  await worker.stop()
  check(anira.getActiveWorkers().length === 0, 'the default Worker stopped')
  m._anira_handler_destroy(handler)
  m._anira_pipeline_destroy(pipeline)
  m._anira_contract_destroy(contract)
  m._anira_model_config_destroy(model)
  m._anira_context_destroy(context)
  m._anira_context_config_destroy(contextConfig)
  heap.free(variants)
  heap.free(out)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
