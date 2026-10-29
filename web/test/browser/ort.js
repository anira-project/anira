// The ONNX Runtime Web engine (@anira-project/anira/engines/onnxruntime-web) on a pool of one
// Wasm Worker, which imports ort-worker.js: the SimpleGainNetwork model of extras/models (output =
// input * gain), a Hard handler of block 128 over its window of 512, and an OfflineAudioContext
// rendered through the handler's audio node, the Static gain its AudioParam. The output must be
// the input times the gain after the handler's latency. The verdict is window.anira_test_result.
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

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  anira.setLogHook((record) => console.log(`[anira ${record.group}] ${record.message}`))
  const context = Context.create(anira, {
    threads: 1,
    workerModule: new URL('./ort-worker.js', import.meta.url),
  })

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
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ONNXRUNTIME_WEB_ENGINE_ID, path: 'loaded-by-the-engine.onnx' }],
      inputs: [streamed('audio_in'), scalar('gain')],
      outputs: [streamed('audio_out'), scalar('gain_out')],
    }),
    { fetchModel: () => false }
  )
  const pipeline = Pipeline.create(anira)
    .addInference(model)
    .registerEngine(ONNXRUNTIME_WEB_ENGINE_ID)
  const contract = Contract.hard(anira, { blockMin: QUANTUM, rate: RATE, budgetMs: 2 })
  const handler = Handler.create(context, pipeline).prepare(contract)
  const latency = handler.latency()
  // The Worker's boot awaits the engine's init: the ORT module and the session.
  await context.ready(30000)
  check(true, 'the Worker booted the ONNX Runtime Web engine (model and binary loaded)')

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
  check(energy > 1, `the node rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times the Static gain ${GAIN} after ${latency} samples ` +
      `(worst error ${worst})`
  )

  audioNode.destroy()
  handler.destroy()
  pipeline.destroy()
  contract.destroy()
  model.destroy()
  context.destroy()
  check(context.threads === 0, 'the Worker stopped with the context')
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
