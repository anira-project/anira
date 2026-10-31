// The ONNX Runtime Web engine (@anira-project/anira/engines/onnxruntime-web) on a pool of two
// Wasm Workers, which import ort-worker.js (the engine with no model of its own). The model config
// ort-gain.model.json names the SimpleGainNetwork model of extras/models (output = input * gain)
// for org.anira.onnxruntime-web: ModelConfig.fromUrl fetches it once on the main thread and
// attaches the bytes, and each Worker's engine creates its session from the entry's bytes
// (engineModel), so the .onnx is requested exactly once (run.mjs counts the requests). Then a
// Hard handler of block 128 over the window of 512 and an OfflineAudioContext rendered through
// the handler's audio node, the Static gain its AudioParam: the output must be the input times
// the gain after the handler's latency. The same again on a context whose worker module,
// ort-threads-worker.js, defines the engine with two intra-op threads (`numThreads: 2`): its
// Workers boot only once onnxruntime-web's own Worker runs, and the output must be the same.
// The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
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
    .registerEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  const contract = Contract.hard(anira, { blockMin: QUANTUM, rate: RATE, budgetMs: 2 })
  const handler = Handler.create(context, pipeline).prepare(contract)
  const latency = handler.latency()

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

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))
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
  check(context.threads === 0, 'the Workers stopped with the context')

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
  check(threaded.threads === 0, 'with numThreads 2 the Workers stopped with the context')

  model.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
