// The audio path in C: an OfflineAudioContext rendered through the Wasm Audio Worklet node of a
// prepared Hard handler (createAudioNode, anira_em_audio_node_create). The node's render callback
// hands its buffers to the handler as planar tensors and waits for each quantum's inference
// (waitMs: an offline render runs faster than real time); the inference runs on the context's
// pool, two Wasm Workers importing pool-worker.js (input times 2 * the Static gain); the gain is
// the node's AudioParam '0', automated mid-render; a JavaScript stage on the worklet scope's
// registry (worklet-stage.js, loaded by audioWorklet.addModule) halves the input in pre_process.
// The output must be the input times the gain of its quantum after the handler's latency (the
// stage's 0.5 and the engine's 2 cancel). Then the node is destroyed and the pool stopped. The
// verdict is window.anira_test_result.
import {
  ANIRA_EM_HOOK_PRE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
  AniraWeb,
  Contract,
  ModelConfig,
  anira_log_drain,
  anira_wait_strategy,
  createAudioNode,
  keyOf,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const THREADS = 2
// The gain until CHANGE_QUANTUM, then GAIN_AFTER (an AudioParam event at the quantum's start).
const GAIN_BEFORE = 1
const GAIN_AFTER = 0.25
const CHANGE_QUANTUM = 32
const ENGINE_ID = 'org.anira-test.pool-double'
const STAGE = 'org.anira-test.half'

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  check(crossOriginIsolated, 'the page is cross-origin isolated (COOP/COEP)')

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
      ['time', QUANTUM],
    ],
    window: { min: QUANTUM, max: QUANTUM },
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
  abi.anira_em_pipeline_add_js_stage(
    pipeline,
    keyOf(STAGE),
    ANIRA_EM_HOOK_PRE,
    ANIRA_STAGE_FLAG_REALTIME_PRE_POST
  )
  const handler = abi.anira_handler_create(context, pipeline)
  const contract = Contract.hard(anira, {
    blockMin: QUANTUM,
    rate: RATE,
    budgetMs: 2,
    warmup: 0,
  })
  abi.anira_handler_prepare(handler, contract.ptr)
  const latency = abi.anira_handler_get_latency(handler, 0)

  // The pool's Workers start once this thread yields: wait until both are in their loops, so
  // the render's first quantum finds them.
  const start = performance.now()
  while (abi.anira_em_inference_threads_in_loop() < THREADS) {
    if (performance.now() - start > 10000) throw new Error('the pool did not start')
    await new Promise((resolve) => setTimeout(resolve, 5))
  }

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const audioNode = await createAudioNode(anira, audio, handler, {
    inputChannels: 1,
    outputChannels: 1,
    params: [1],
    waitMs: 500,
    workletModules: [new URL('./worklet-stage.js', import.meta.url)],
  })
  check(
    audioNode.node instanceof AudioWorkletNode && audioNode.params.has(1),
    "the node is an AudioWorkletNode with the gain's AudioParam"
  )
  const gain = audioNode.params.get(1)
  gain.setValueAtTime(GAIN_BEFORE, 0)
  gain.setValueAtTime(GAIN_AFTER, (CHANGE_QUANTUM * QUANTUM) / RATE)

  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)

  // Each input block carries the gain of the quantum that submitted it.
  let worst = 0
  for (let j = 0; j + latency < rendered.length; j++) {
    const g = Math.floor(j / QUANTUM) < CHANGE_QUANTUM ? GAIN_BEFORE : GAIN_AFTER
    worst = Math.max(worst, Math.abs(rendered[j + latency] - g * samples[j]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the C render callback rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times the AudioParam's gain after ${latency} samples, through ` +
      `the worklet's JavaScript stage and the pool's engine (worst error ${worst})`
  )

  audioNode.destroy()
  abi.anira_handler_destroy(handler)

  // The same gain as a tensor slot the main thread writes whole (setStaticInput: the node's
  // seqlock block), on a second handler and a second context: a constant 0.5.
  const second = abi.anira_handler_create(context, pipeline)
  abi.anira_handler_prepare(second, contract.ptr)
  const audio2 = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const tensorNode = await createAudioNode(anira, audio2, second, {
    staticInputs: [1],
    waitMs: 500,
    workletModules: [new URL('./worklet-stage.js', import.meta.url)],
  })
  let refused = ''
  try {
    tensorNode.setStaticInput(1, [1, 2])
  } catch (error) {
    refused = error.name
  }
  check(refused === 'AniraError', `a value of the wrong size is refused (${refused})`)
  tensorNode.setStaticInput(1, [0.5])
  const input2 = audio2.createBuffer(1, samples.length, RATE)
  input2.copyToChannel(samples, 0)
  const source2 = new AudioBufferSourceNode(audio2, { buffer: input2 })
  source2.connect(tensorNode.node).connect(audio2.destination)
  source2.start(0)
  const rendered2 = (await audio2.startRendering()).getChannelData(0)
  let worst2 = 0
  for (let j = 0; j + latency < rendered2.length; j++) {
    worst2 = Math.max(worst2, Math.abs(rendered2[j + latency] - 0.5 * samples[j]))
  }
  check(
    worst2 < 1e-6,
    `the tensor slot written with setStaticInput reaches the handler (worst error ${worst2})`
  )
  tensorNode.destroy()
  abi.anira_handler_destroy(second)
  abi.anira_pipeline_destroy(pipeline)
  contract.destroy()
  model.destroy()
  abi.anira_context_destroy(context)
  abi.anira_context_config_destroy(contextConfig)
  m._free(variants)
  check(abi.anira_num_inference_threads() === 0, 'the node and the pool are gone')
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
