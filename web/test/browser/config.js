// Configs once (D6): a bundled model config and its contract, loaded on the web as a native host
// loads them. extras/models/model-pool/gain.model.json and gain.contract.json are fetched as they
// are in the repository (run.mjs serves extras/models next to the package); ModelConfig.fromUrl
// fetches the model file of every entry whose engine this module has compiled in (the ONNX
// Runtime entry: simple_gain_network_mono.onnx, relative to the config) and attaches it as bytes,
// and leaves the others (LibTorch, TFLite, LiteRT, ExecuTorch) unfetched. The handler runs the
// entry on the module's built-in ONNX Runtime engine, on a pool of one Wasm Worker (no worker
// module: the built-in engine needs no JavaScript), streamed through the handler's audio node in
// an OfflineAudioContext with the Static gain as its AudioParam. The output must be the input
// times the gain after the handler's latency. The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_engine,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const GAIN = 0.5
const MODELS = '../../extras/models/model-pool/'

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const anira = await Anira.create()
  const { abi } = anira
  anira.setLogSink((record) => console.log(`[anira ${record.group}] ${record.message}`))

  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const entries = abi.anira_model_config_model_count(model.native())
  const entryOf = (engine) =>
    [...Array(entries).keys()].find(
      (i) => abi.anira_model_config_model_engine(model.native(), i).engine === engine
    )
  const onnx = entryOf(anira_engine.ANIRA_ENGINE_ONNXRUNTIME)
  check(
    onnx !== undefined &&
      abi.anira_model_config_model_path(model.native(), onnx) === null,
    `the ONNX Runtime entry (${onnx} of ${entries}) carries its fetched bytes`
  )
  const libtorch = entryOf(anira_engine.ANIRA_ENGINE_LIBTORCH)
  check(
    libtorch !== undefined &&
      abi.anira_model_config_model_path(model.native(), libtorch) !== null,
    'an entry of an engine this module lacks keeps its path, unfetched'
  )
  // The file leaves the geometry to the host: the render quantum at the context's rate.
  const contract = (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  ).hardGeometry(QUANTUM, QUANTUM, RATE)
  check(
    abi.anira_contract_hard_budget(contract.native()).explicit_ms === 5,
    'the contract file is read as written (budget 5 ms)'
  )

  const context = Context.create(anira, { threads: 1 })
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  const latency = handler.getLatency()
  check(
    true,
    `the handler prepared on the built-in ONNX Runtime engine (latency ${latency})`
  )
  await context.ready()

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
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
