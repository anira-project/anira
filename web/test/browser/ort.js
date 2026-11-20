// The ONNX Runtime Web engine (@anira-project/anira/engines/onnxruntime-web) on a pool of two
// Wasm Workers, which import ort-worker.js (the engine with no model of its own). The model config
// ort-gain.model.json names the SimpleGainNetwork model of extras/models (output = input * gain)
// for org.anira.onnxruntime-web: ModelConfig.fromUrl fetches it once on the main thread and
// attaches the bytes, and each Worker's engine creates its session from the entry's bytes
// (its load's `info.modelBytes`), so the .onnx is requested exactly once (run.mjs counts the
// requests). Then a Hard handler of block 128 over the window of 512 and an
// OfflineAudioContext rendered through the handler's audio node, the Static gain its
// AudioParam: the output must be the input times the gain after the handler's latency. The
// same again on a context whose worker module, ort-threads-worker.js, defines the engine with
// two intra-op threads (`numThreads: 2`): its Workers boot only once onnxruntime-web's own
// Worker runs, and the output must be the same.
//
// Last, the engine's sessions under the variants of one handler, on a context of one Worker
// whose module, ort-sessions-worker.js, counts them: a session per loaded model, created at
// the Worker's first inference of it and kept through every switch; a variant added is one
// session more, a variant removed one released, without a further inference; and over many
// rounds of add, select and remove, onnxruntime-web's own memory does not grow. The verdict
// is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_warmup_mode,
  heap,
} from '../../dist/index.js'
import { ONNXRUNTIME_WEB_ENGINE_ID } from '../../dist/engines/onnxruntime-web.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const WINDOW = 512
const GAIN = 0.5
const ONNX =
  '/extras/models/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.onnx'

const requests = async (pathname) =>
  (await (await fetch('/__requests')).json())[pathname] ?? 0

// The gain model streamed through a handler on `context` (its pool booted), an offline render
// of a tone: the largest difference from input * gain after the handler's latency, and the
// rendered energy.
const renderGain = async (anira, context, model) => {
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .addEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 2,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  const latency = handler.getLatency()

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const audioNode = await handler.createAudioNode(audio, { params: [1], waitMs: 500 })
  audioNode.params.get(1).value = GAIN
  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)

  let worst = 0
  for (let i = latency; i < rendered.length; i++) {
    worst = Math.max(worst, Math.abs(rendered[i] - GAIN * samples[i - latency]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)

  audioNode.destroy()
  handler.destroy()
  pipeline.destroy()
  contract.destroy()
  return { latency, worst, energy }
}

// ---- the engine's sessions under a handler's variants ----------------------------------------

const MODELS = new URL('../../extras/models/model-pool/', import.meta.url)
const HOP = 512
// y = w * x: the three variant gain models of extras/models, exact in float32.
const FACTORS = { half: 0.5, double: 2, invert: -1 }
// Rounds of add, select and remove. A session the engine kept would hold on to its blocks of
// onnxruntime-web's memory, which starts at 16 MiB: that memory grew before the 200th session
// when a build of the engine released none.
const ROUNDS = 320
const RENDER_EVERY = 64
// How long a parked Worker may take to release the session of an unloaded model.
const RELEASE_MS = 2000

// Every file is fetched once: the rounds make the same model configs many times over.
const fetched = new Map()
const fetchOnce = async (url) => {
  const key = `${url}`
  if (!fetched.has(key)) {
    const response = await fetch(url)
    fetched.set(key, {
      ok: response.ok,
      status: response.status,
      bytes: new Uint8Array(await response.arrayBuffer()),
    })
  }
  const { ok, status, bytes } = fetched.get(key)
  return {
    ok,
    status,
    bytes,
    arrayBuffer: async () => bytes.slice().buffer,
    text: async () => new TextDecoder().decode(bytes),
  }
}

// A variant gain model on the ONNX Runtime Web engine alone: the bundled config with the
// file of its built-in ONNX Runtime entry under the JavaScript engine's id. With a `mark` the
// file's bytes carry it in the model's doc_string (ModelProto's field 6, appended), so the
// model is one anira has no figure of: an add under a contract that warms up measures it,
// which runs it on the Worker.
const variantModel = async (anira, name, mark = null) => {
  const url = new URL(`variant_gain_${name}.model.json`, MODELS)
  const json = JSON.parse(await (await fetchOnce(url)).text())
  const { path } = json.models.find((entry) => entry.engine === 'onnxruntime')
  json.models = [{ engine: ONNXRUNTIME_WEB_ENGINE_ID, path }]
  const config = await ModelConfig.fromJson(anira, JSON.stringify(json), {
    baseUrl: url,
    fetch: fetchOnce,
    fetchModel: () => mark === null,
  })
  if (mark === null) return config
  const onnx = (await fetchOnce(new URL(path, url))).bytes
  const text = new TextEncoder().encode(`${mark}`)
  const bytes = new Uint8Array(onnx.length + 2 + text.length)
  bytes.set(onnx)
  bytes.set([0x32, text.length], onnx.length)
  bytes.set(text, onnx.length + 2)
  return config.setModelBytes(0, bytes)
}

// Exact in float32 under every factor: odd multiples of 1/512 between -1/2 and 1/2.
const ramp = (frames) => {
  const samples = new Float32Array(frames)
  for (let i = 0; i < frames; i++) samples[i] = (2 * ((i * 37) % 255) - 255) / 512
  return samples
}

// Renders `samples` through a node of the handler and says whether the output is `factor`
// times the input, sample for sample, behind the handler's latency. The handler's stream goes
// on from the render before, whose last samples come first; what is selected between two
// renders runs from the first chunk of the next.
const rendersFactor = async (handler, samples, factor) => {
  const latency = handler.getLatency()
  const audio = new OfflineAudioContext(1, samples.length, RATE)
  const audioNode = await handler.createAudioNode(audio, { waitMs: 2000 })
  const buffer = audio.createBuffer(1, samples.length, RATE)
  buffer.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  audioNode.destroy()
  for (let i = 0; i + latency < rendered.length; i++) {
    if (rendered[i + latency] !== Math.fround(factor * samples[i])) return false
  }
  return true
}

const sessionsOfVariants = async (anira, check) => {
  // What the Worker's engine holds (ort-sessions-worker.js): the sessions it created, the
  // sessions it released, and onnxruntime-web's own memory in 64 KiB pages.
  const counts = anira.module._malloc(12)
  heap(anira.module).HEAP32.fill(0, counts >>> 2, (counts >>> 2) + 3)
  const read = (index) => Atomics.load(heap(anira.module).HEAP32, (counts >>> 2) + index)
  const sessions = () => `${read(0)}/${read(1)}`
  // A release comes at the Worker's next look at its queue, a moment behind the call that
  // unloaded: the page only looks, no inference follows.
  const released = async (count) => {
    const from = performance.now()
    while (read(1) < count && performance.now() - from < RELEASE_MS) {
      await new Promise((resolve) => setTimeout(resolve, 0))
    }
    return read(1) === count
  }

  const context = Context.create(anira, {
    threads: 1,
    workerModule: new URL(`./ort-sessions-worker.js?counts=${counts}`, import.meta.url),
  })
  await context.ready(30000)
  // An explicit budget, so the latency is the same everywhere, and a warm-up of two inferences
  // for every plan without a figure: a prepare and an add run the model on the Worker.
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_FIXED,
    warmupIterations: 2,
  })
  const models = [await variantModel(anira, 'half'), await variantModel(anira, 'double')]
  const pipeline = Pipeline.create(anira)
    .addInference(models)
    .addEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  const handler = InferenceHandler.create(context, pipeline)
  pipeline.destroy()
  models.forEach((model) => model.destroy())
  const samples = ramp(4 * HOP)

  // ---- a session per loaded model, kept through every switch --------------------------------
  const steps = []
  const step = async (what, factor) =>
    steps.push({
      what,
      exact: await rendersFactor(handler, samples, factor),
      sessions: sessions(),
    })
  const beforePrepare = sessions()
  handler.prepare(contract)
  await step('prepare', FACTORS.half)
  handler.setModel(1)
  await step('setModel(1)', FACTORS.double)
  handler.setModel(0)
  await step('setModel(0)', FACTORS.half)
  const inverted = await variantModel(anira, 'invert')
  const added = handler.addVariant(inverted)
  inverted.destroy()
  await step('addVariant', FACTORS.half)
  handler.setModel(added)
  await step(`setModel(${added})`, FACTORS.invert)
  handler.removeVariant(0)
  const firstReleased = await released(1)
  await step('removeVariant(0)', FACTORS.invert)
  handler.setModel(1)
  await step('setModel(1)', FACTORS.double)
  check(
    beforePrepare === '0/0' &&
      added === 2 &&
      firstReleased &&
      steps.every((s) => s.exact) &&
      steps.map((s) => s.sessions).join(' ') === '2/0 2/0 2/0 3/0 3/0 3/1 3/1',
    `the ONNX Runtime Web engine holds a session per loaded model on its Worker: sessions ` +
      `created/released behind each step: ` +
      steps
        .map((s) => `${s.what} ${s.sessions}${s.exact ? '' : ' (WRONG STREAM)'}`)
        .join(', ') +
      ` (${beforePrepare} before the prepare); a switch creates none, an added variant one, ` +
      `a removed one is released without a further inference, and every render is the ` +
      `selected variant's model, sample for sample`
  )

  // ---- rounds of add, select, remove ----------------------------------------------------------
  const pagesBefore = read(2)
  const createdBefore = read(0)
  const cycle = ['half', 'invert', 'double']
  const wrong = []
  let previous = 1
  let rendered = 0
  for (let round = 0; round < ROUNDS; round++) {
    const name = cycle[round % cycle.length]
    const model = await variantModel(anira, name, `round ${round}`)
    const index = handler.addVariant(model)
    model.destroy()
    handler.setModel(index)
    handler.removeVariant(previous)
    previous = index
    if (round % RENDER_EVERY === RENDER_EVERY - 1) {
      rendered += 1
      if (!(await rendersFactor(handler, samples, FACTORS[name]))) wrong.push(round)
    }
  }
  const allReleased = await released(1 + ROUNDS)
  const pagesAfter = read(2)
  check(
    read(0) === createdBefore + ROUNDS &&
      allReleased &&
      pagesAfter === pagesBefore &&
      wrong.length === 0,
    `${ROUNDS} rounds of add, select and remove on the engine: ${read(0) - createdBefore} ` +
      `sessions created and ${read(1) - 1} released, one model's left; onnxruntime-web's ` +
      `memory is ${pagesAfter} pages of 64 KiB behind the rounds and was ${pagesBefore} ` +
      `before them; ${rendered} renders in between, each the selected variant's model ` +
      `(wrong in the rounds: ${wrong.join(' ') || 'none'})`
  )

  // ---- the handler's destroy -----------------------------------------------------------------
  const created = read(0)
  handler.destroy()
  const lastReleased = await released(created)
  check(
    lastReleased,
    `the handler's destroy releases the sessions it still held, on the parked Worker ` +
      `(${sessions()} created/released)`
  )
  contract.destroy()
  context.destroy()
  anira.module._free(counts)
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const model = await ModelConfig.fromUrl(
    anira,
    new URL('./ort-gain.model.json', import.meta.url)
  )

  // One thread per inference.
  const context = Context.create(anira, {
    threads: 2,
    workerModule: new URL('./ort-worker.js', import.meta.url),
  })
  // The Workers' boot awaits the engine's init: the ORT module; the session comes with the
  // entry, at the first call.
  await context.ready(30000)
  check(true, 'the Workers booted the ONNX Runtime Web engine (its binary loaded)')
  const single = await renderGain(anira, context, model)
  check(
    (await requests(ONNX)) === 1,
    `the model was requested once for the config and the pool of two Workers ` +
      `(${await requests(ONNX)} requests)`
  )
  check(single.energy > 1, `the node rendered audio (energy ${single.energy.toFixed(2)})`)
  check(
    single.worst < 1e-6,
    `the output is the input times the Static gain ${GAIN} after ${single.latency} samples ` +
      `(worst error ${single.worst})`
  )
  context.destroy()
  check(context.numInferenceThreads() === 0, 'the Workers stopped with the context')

  // Two intra-op threads per inference: each Worker's runtime starts a Worker of its own.
  const threaded = Context.create(anira, {
    threads: 2,
    workerModule: new URL('./ort-threads-worker.js', import.meta.url),
  })
  await threaded.ready(30000)
  check(
    true,
    'with numThreads 2 the Workers booted the engine (onnxruntime-web started its own Workers)'
  )
  const two = await renderGain(anira, threaded, model)
  check(
    two.energy > 1 && two.worst < 1e-6,
    `with numThreads 2 the output is the input times the Static gain ${GAIN} after ` +
      `${two.latency} samples (worst error ${two.worst}, energy ${two.energy.toFixed(2)})`
  )
  threaded.destroy()
  check(
    threaded.numInferenceThreads() === 0,
    'with numThreads 2 the Workers stopped with the context'
  )

  model.destroy()

  await sessionsOfVariants(anira, check)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
