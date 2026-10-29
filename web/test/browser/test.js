// The browser test of the built package (web/dist), run by run.mjs in headless Chromium and
// Firefox: the typed layer end to end. The module on the page (Anira), a context whose pool is two
// Wasm Workers importing pool-worker.js (a JavaScript engine that writes its input times 2 * the
// Static gain), a pipeline with that engine and a JavaScript stage (worklet-stage.js: halves the
// input in pre_process, on the worklet scope's registry), a Hard handler of block 128, and an
// OfflineAudioContext rendered through the handler's Wasm Audio Worklet node (C render callback),
// the gain being the node's AudioParam, automated mid-render. The output must be the input times
// the gain of its quantum after the handler's latency (the stage's 0.5 and the engine's 2 cancel).
// Then the gain as a tensor slot written with setStaticInput on a second handler, and the teardown:
// the nodes, the handlers and the context, which stops the pool. The verdict is
// window.anira_test_result. First, before any of it: a warning reaches the console exactly once,
// through the log hook while one is set (it replaces the platform sink, which the glue prints as
// console.error), through the platform sink once the hook is removed.
import {
  ANIRA_EM_HOOK_PRE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  AniraError,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
  anira_status,
  anira_wait_strategy,
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

// Counts the console lines that carry `marker`, whichever method printed them; installed before
// the module, whose glue binds console.error for its stderr when it is instantiated.
const consoleCounter = (marker) => {
  const counts = { log: 0, info: 0, warn: 0, error: 0, debug: 0 }
  for (const method of Object.keys(counts)) {
    const original = console[method].bind(console)
    console[method] = (...args) => {
      if (args.some((arg) => String(arg).includes(marker))) counts[method] += 1
      original(...args)
    }
  }
  return {
    total: () => Object.values(counts).reduce((sum, n) => sum + n, 0),
    counts,
  }
}

// A context whose config asks for a blocking wait, which the core coerces with one warning on
// WebAssembly; no threads, destroyed right away (the next context's config takes effect whole).
const BLOCKING_WARNING = 'ANIRA_WAIT_BLOCKING is not supported'
const warnOnce = (abi) => {
  const config = abi.anira_context_config_create()
  abi.anira_context_config_set_threads(config, 0, anira_wait_strategy.ANIRA_WAIT_BLOCKING)
  const context = abi.anira_context_create(config)
  abi.anira_context_config_destroy(config)
  abi.anira_context_destroy(context)
}

const render = async (audio, audioNode, samples) => {
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  return (await audio.startRendering()).getChannelData(0)
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  check(crossOriginIsolated, 'the page is cross-origin isolated (COOP/COEP)')

  const counter = consoleCounter(BLOCKING_WARNING)
  const anira = await Anira.create()
  const hook = (record) => console.log(`[anira ${record.group}] ${record.message}`)
  anira.setLogHook(hook)
  check(anira.version.length > 0, `the module is anira ${anira.version}`)

  warnOnce(anira.abi)
  check(
    counter.total() === 1 && counter.counts.log === 1,
    `with the log hook set a warning reaches the console once, through the hook ` +
      `(${JSON.stringify(counter.counts)})`
  )
  anira.setLogHook(null)
  warnOnce(anira.abi)
  check(
    counter.total() === 2 && counter.counts.log === 1,
    `without the hook it reaches the console once, through the platform sink ` +
      `(${JSON.stringify(counter.counts)})`
  )
  anira.setLogHook(hook)

  // The bindings throw a refused call as an AniraError with the entry's message.
  let thrown = null
  try {
    anira.abi.anira_model_config_from_json('{ not json', null)
  } catch (error) {
    thrown = error
  }
  check(
    thrown instanceof AniraError &&
      thrown.code === anira_status.ANIRA_ERROR_JSON &&
      thrown.message.startsWith('anira_model_config_from_json: '),
    `a refused call throws an AniraError (${thrown?.message})`
  )

  const context = Context.create(anira, {
    threads: THREADS,
    workerModule: new URL('./pool-worker.js', import.meta.url),
  })

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
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .registerEngine(ENGINE_ID, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
    .addStage(STAGE, { hooks: ANIRA_EM_HOOK_PRE })
  const contract = Contract.hard(anira, { blockMin: QUANTUM, rate: RATE, budgetMs: 2 })
  const handler = Handler.create(context, pipeline).prepare(contract)
  const latency = handler.latency()
  // The first session built the pool.
  check(
    context.threads === THREADS,
    `the context's pool has ${context.threads} Wasm Workers`
  )
  // An offline render waits for every inference: the Workers must be in their loops first.
  await context.ready()
  check(true, `the pool's Wasm Workers are in their loops (latency ${latency})`)

  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const stageModule = new URL('./worklet-stage.js', import.meta.url)

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const audioNode = await handler.createAudioNode(audio, {
    params: [1],
    waitMs: 500,
    workletModules: [stageModule],
  })
  check(
    audioNode.node instanceof AudioWorkletNode && audioNode.params.has(1),
    "the node is an AudioWorkletNode with the gain's AudioParam"
  )
  const gain = audioNode.params.get(1)
  gain.setValueAtTime(GAIN_BEFORE, 0)
  gain.setValueAtTime(GAIN_AFTER, (CHANGE_QUANTUM * QUANTUM) / RATE)
  const rendered = await render(audio, audioNode, samples)
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
  handler.destroy()

  // The same gain as a tensor slot the main thread writes whole: a constant 0.5.
  const second = Handler.create(context, pipeline).prepare(contract)
  const audio2 = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const tensorNode = await second.createAudioNode(audio2, {
    staticInputs: [1],
    waitMs: 500,
    workletModules: [stageModule],
  })
  let refused = ''
  try {
    tensorNode.setStaticInput(1, [1, 2])
  } catch (error) {
    refused = error.name
  }
  check(refused === 'AniraError', `a value of the wrong size is refused (${refused})`)
  tensorNode.setStaticInput(1, [0.5])
  const rendered2 = await render(audio2, tensorNode, samples)
  let worst2 = 0
  for (let j = 0; j + latency < rendered2.length; j++) {
    worst2 = Math.max(worst2, Math.abs(rendered2[j + latency] - 0.5 * samples[j]))
  }
  check(
    worst2 < 1e-6,
    `the tensor slot written with setStaticInput reaches the handler (worst error ${worst2})`
  )
  tensorNode.destroy()
  second.destroy()

  pipeline.destroy()
  contract.destroy()
  model.destroy()
  context.destroy()
  check(context.threads === 0, 'the last context stopped the pool')
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
