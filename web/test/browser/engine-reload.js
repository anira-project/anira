// A JavaScript engine loads every loaded model into an object of its own, also a model whose
// record lies where an unloaded one's did. Handlers come and go on one context of one Worker,
// each over a model of its own on the JavaScript engine of weight-worker.js (y = weight * x on
// the object its Worker loaded for the model, as an engine that keeps a session per model
// does), each with another weight: a handler is prepared, an OfflineAudioContext is rendered
// through its node, the handler is destroyed, and the next one is made, whose model record the
// allocator then places where the last one's was, every few rounds. Every render must be the
// input times its own handler's weight. (The registry tells loaded models apart by a serial of
// their record; when it compared the record's address, a handler whose record took a freed
// one's place rendered the weight of a handler before it, in seven of this page's 48 rounds.)
// The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_warmup_mode,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 8
const ROUNDS = 48
const ENGINE_ID = 'org.anira-test.weight'

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

// A model the weight engine runs: its entry's bytes are the weight and no sleep.
const weightModel = async (anira, weight) => {
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ENGINE_ID }],
      inputs: [streamed('in')],
      outputs: [streamed('out')],
    })
  )
  return model.setModelBytes(0, new Uint8Array(new Float32Array([weight, 0]).buffer))
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const context = Context.create(anira, {
    threads: 1,
    workerModule: new URL('./weight-worker.js', import.meta.url),
  })
  await context.ready()
  const contract = ContractHandle.hard(anira, {
    hostBlockMin: QUANTUM,
    hostRate: RATE,
    budgetMs: 5,
    warmup: anira_warmup_mode.ANIRA_WARMUP_NONE,
  })

  // Exact in float32 under every weight of the page (small whole numbers), never zero.
  const samples = new Float32Array(QUANTA * QUANTUM)
  for (let i = 0; i < samples.length; i++) samples[i] = (2 * ((i * 37) % 255) - 255) / 512

  const wrong = []
  let compared = 0
  for (let round = 0; round < ROUNDS; round++) {
    const weight = round + 1
    const model = await weightModel(anira, weight)
    const pipeline = Pipeline.create(anira).addInference(model).addEngine(ENGINE_ID)
    const handler = InferenceHandler.create(context, pipeline)
    pipeline.destroy()
    model.destroy()
    handler.prepare(contract)
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
    handler.destroy()

    let exact = true
    for (let i = 0; i + latency < rendered.length; i++) {
      exact &&= rendered[i + latency] === Math.fround(weight * samples[i])
    }
    compared += rendered.length - latency
    if (!exact) {
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

  contract.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
