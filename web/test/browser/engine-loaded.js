// A JavaScript engine's loaded models come and go with their handlers: each is loaded into an
// object of its own by the Worker that runs it, also a model whose record lies where an
// unloaded one's did, and the object is destroyed when the model is unloaded, without a further
// inference. Handlers come and go on one context of one Worker, each over a model of its own on
// the JavaScript engine of weight-worker.js (y = weight * x on the object its Worker loaded for
// the model, as an engine that keeps a session per model does), each with another weight: a
// handler is prepared, an OfflineAudioContext is rendered through its node, the handler is
// destroyed, and the next one is made, whose model record the allocator then places where the
// last one's was, every few rounds. Every render must be the input times its own handler's
// weight, the Worker must load each model once, at its handler's first inference, and destroy
// what it loaded once the handler is destroyed: the Worker is parked then, the unload wakes
// it, and nothing else runs on it before the page looks. (The registry tells loaded models
// apart by a serial of their record; when it compared the record's address and an engine held
// the model of its last load, a handler whose record took a freed one's place rendered the
// weight of a handler before it, in seven of this page's 48 rounds.)
//
// Then a Worker that leaves its loop while a model is still loaded: an inference thread the
// page starts and stops itself, on a context without a pool. Stopped, the Worker destroys what
// it holds before it goes, though no model was unloaded; the Worker started next loads the
// model for itself.
//
// Last, a load that throws: the inference that needed the model fails (zeros at its place in
// the stream, ANIRA_ERROR_ENGINE in rtError()), and the Worker's next inference of the model
// loads it. The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_status,
  anira_warmup_mode,
  heap,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 8
const ROUNDS = 48
const ENGINE_ID = 'org.anira-test.weight'
const WORKER_MODULE = new URL('./weight-worker.js', import.meta.url)
// How long a parked Worker may take to destroy what it loaded for an unloaded model.
const RELEASE_MS = 2000

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

// A model the weight engine runs: its entry's bytes are the weight, no sleep, the address of
// the counts every load and every destroy of the model is counted in, and with `failsOnce` the
// word that makes its first load on a Worker throw.
const weightModel = async (anira, weight, counts, failsOnce = false) => {
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ENGINE_ID }],
      inputs: [streamed('in')],
      outputs: [streamed('out')],
    })
  )
  const bytes = new Uint8Array(failsOnce ? 16 : 12)
  bytes.set(new Uint8Array(new Float32Array([weight, 0]).buffer))
  new DataView(bytes.buffer).setUint32(8, counts, true)
  if (failsOnce) new DataView(bytes.buffer).setUint32(12, 1, true)
  return model.setModelBytes(0, bytes)
}

// What the Workers' engines count of a model (weight-worker.js): [0] its loads, [1] the
// destroys of what they returned, [2] the loads by a Worker that held an object of it already.
const modelCounts = (anira) => {
  const address = anira.module._malloc(12)
  heap(anira.module).HEAP32.fill(0, address >>> 2, (address >>> 2) + 3)
  return {
    address,
    read: (index) => Atomics.load(heap(anira.module).HEAP32, (address >>> 2) + index),
  }
}

// Exact in float32 under every weight of the page (small whole numbers), never zero.
const samples = new Float32Array(QUANTA * QUANTUM)
for (let i = 0; i < samples.length; i++) samples[i] = (2 * ((i * 37) % 255) - 255) / 512

// Renders the samples through a node of the handler. The handler's stream goes on from the
// render before, so the output of sample i is at i + latency in every render.
const render = async (handler) => {
  const audio = new OfflineAudioContext(1, samples.length, RATE)
  const audioNode = await handler.createAudioNode(audio, { waitMs: 2000 })
  const buffer = audio.createBuffer(1, samples.length, RATE)
  buffer.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  audioNode.destroy()
  return rendered
}

const rendersWeight = (rendered, latency, weight) => {
  for (let i = 0; i + latency < rendered.length; i++) {
    if (rendered[i + latency] !== Math.fround(weight * samples[i])) return false
  }
  return true
}

const handlersInTurn = async (anira, check) => {
  const context = Context.create(anira, { threads: 1, workerModule: WORKER_MODULE })
  await context.ready()
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })
  // One block of counts for every round, allocated before the first: the rounds allocate
  // what they did without it.
  const counts = modelCounts(anira)

  const wrong = []
  const miscounted = []
  let compared = 0
  let slowest = 0
  for (let round = 0; round < ROUNDS; round++) {
    const weight = round + 1
    const model = await weightModel(anira, weight, counts.address)
    const pipeline = Pipeline.create(anira).addInference(model).addEngine(ENGINE_ID)
    const handler = InferenceHandler.create(context, pipeline)
    pipeline.destroy()
    model.destroy()
    handler.prepare(contract)
    const latency = handler.getLatency()
    // A prepare without a warm-up runs no inference: nothing is loaded before the stream.
    const loadedByPrepare = counts.read(0) - round

    const rendered = await render(handler)
    // The model is loaded while its handler lives: one load, and no destroy yet.
    const whileLoaded = [counts.read(0) - round, counts.read(1) - round]
    handler.destroy()
    // The handler's destroy unloaded the model and woke the Worker, which destroys its object
    // at its next look at its queue: no inference follows, the page only looks.
    const from = performance.now()
    while (counts.read(1) < round + 1 && performance.now() - from < RELEASE_MS) {
      await new Promise((resolve) => setTimeout(resolve, 0))
    }
    slowest = Math.max(slowest, performance.now() - from)
    if (
      loadedByPrepare !== 0 ||
      whileLoaded.join(' ') !== '1 0' ||
      counts.read(0) !== round + 1 ||
      counts.read(1) !== round + 1
    ) {
      miscounted.push(
        `round ${round}: ${loadedByPrepare} loads by the prepare, ${whileLoaded[0]} loads ` +
          `and ${whileLoaded[1]} destroys with the render done, ` +
          `${counts.read(0) - round} loads and ${counts.read(1) - round} destroys behind ` +
          `the handler's destroy`
      )
    }

    compared += rendered.length - latency
    if (!rendersWeight(rendered, latency, weight)) {
      wrong.push(
        `round ${round} rendered x ${rendered[latency] / samples[0]}, not x ${weight}`
      )
    }
  }
  check(
    wrong.length === 0 && compared > 0,
    `${ROUNDS} handlers in turn, each over its own model on one JavaScript engine and one ` +
      `Worker, each render its own handler's weight, sample for sample (${compared} ` +
      `samples): ${wrong.join('; ') || 'none wrong'}`
  )
  check(
    miscounted.length === 0 &&
      counts.read(0) === ROUNDS &&
      counts.read(1) === ROUNDS &&
      counts.read(2) === 0,
    `the Worker loaded each of the ${ROUNDS} models once, at its handler's first inference, ` +
      `and destroyed what it loaded once the handler was destroyed, parked, with no further ` +
      `inference (${counts.read(0)} loads, ${counts.read(1)} destroys, ${counts.read(2)} ` +
      `loads of a model it still held; the slowest release was seen ` +
      `${slowest.toFixed(1)} ms behind the destroy): ` +
      (miscounted.slice(0, 4).join('; ') || 'every round as it should be')
  )

  contract.destroy()
  context.destroy()
  anira.module._free(counts.address)
}

// An inference thread the page starts and stops with the plain calls, on a context without a
// pool: its Worker imports the worker module and runs the loop, as a Worker of the pool does.
// The stop returns when the Worker has left its loop.
const leavingWorker = async (anira, check) => {
  const { abi } = anira
  const context = Context.create(anira, { threads: 0, workerModule: WORKER_MODULE })
  const start = async () => {
    const thread = abi.anira_inference_thread_create(context.native())
    abi.anira_inference_thread_start(thread)
    await context.ready()
    return thread
  }
  const stop = (thread) => {
    abi.anira_inference_thread_stop(thread)
    abi.anira_inference_thread_destroy(thread)
  }
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })
  const counts = modelCounts(anira)
  const WEIGHT = 3
  const model = await weightModel(anira, WEIGHT, counts.address)
  const pipeline = Pipeline.create(anira).addInference(model).addEngine(ENGINE_ID)
  const handler = InferenceHandler.create(context, pipeline)
  pipeline.destroy()
  model.destroy()

  const seen = []
  const note = () => seen.push(`${counts.read(0)}/${counts.read(1)}`)
  const first = await start()
  handler.prepare(contract)
  const latency = handler.getLatency()
  const firstExact = rendersWeight(await render(handler), latency, WEIGHT)
  note() // 1/0: the first Worker loaded the model
  // The handler lives and its model stays loaded: nothing is unloaded, the Worker just goes.
  stop(first)
  note() // 1/1: and destroyed what it held before it left
  const second = await start()
  const secondExact = rendersWeight(await render(handler), latency, WEIGHT)
  note() // 2/1: the next Worker loads the model for itself
  handler.destroy()
  const from = performance.now()
  while (counts.read(1) < 2 && performance.now() - from < RELEASE_MS) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
  note() // 2/2: the unload reaches it as it reaches a Worker of the pool
  stop(second)
  check(
    firstExact &&
      secondExact &&
      seen.join(' ') === '1/0 1/1 2/1 2/2' &&
      counts.read(2) === 0,
    `a Worker that leaves its loop destroys what it loaded before it goes, though the model ` +
      `stays loaded, and the Worker started next loads the model for itself: loads/destroys ` +
      `behind the first render, the first Worker's stop, the second Worker's render and ` +
      `the handler's destroy: ${seen.join(' ')}; both renders the handler's weight: ` +
      `${firstExact && secondExact}`
  )

  contract.destroy()
  context.destroy()
  anira.module._free(counts.address)
}

// A model whose load throws at the Worker's first inference of it and succeeds at the next:
// natively a load that throws fails the prepare; here the prepare is long over when a Worker
// loads, so the inference fails, as one whose engine fails does, and the load is tried again.
const failingLoad = async (anira, check) => {
  const context = Context.create(anira, { threads: 1, workerModule: WORKER_MODULE })
  await context.ready()
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })
  const counts = modelCounts(anira)
  const WEIGHT = 5
  const model = await weightModel(anira, WEIGHT, counts.address, true)
  const pipeline = Pipeline.create(anira).addInference(model).addEngine(ENGINE_ID)
  const handler = InferenceHandler.create(context, pipeline)
  pipeline.destroy()
  model.destroy()
  handler.prepare(contract)
  const latency = handler.getLatency()
  const before = handler.rtError()
  const rendered = await render(handler)
  const after = handler.rtError()
  // One inference per quantum: the first one's output is the first quantum behind the latency.
  let silent = true
  let exact = true
  for (let i = 0; i + latency < rendered.length; i++) {
    if (i < QUANTUM) silent &&= rendered[i + latency] === 0
    else exact &&= rendered[i + latency] === Math.fround(WEIGHT * samples[i])
  }
  check(
    before === anira_status.ANIRA_OK &&
      after === anira_status.ANIRA_ERROR_ENGINE &&
      silent &&
      exact &&
      counts.read(0) === 1,
    `a load that throws fails the inference that needed the model (zeros for its quantum: ` +
      `${silent}; rtError() ${after} behind the render, ${before} before it), and the ` +
      `Worker's next inference loads the model (${counts.read(0)} load counted; the rest of ` +
      `the render is the model's weight: ${exact})`
  )
  handler.destroy()
  contract.destroy()
  context.destroy()
  anira.module._free(counts.address)
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))
  await handlersInTurn(anira, check)
  await leavingWorker(anira, check)
  await failingLoad(anira, check)
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
