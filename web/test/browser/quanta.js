// Render quanta other than 128 frames (Web Audio API, Render Quantum Sizes): an
// OfflineAudioContext created with renderSizeHint 256 and 480 (where the browser takes the hint)
// renders process() calls of that size, measureRenderGeometry gives { quantum: size, burst:
// size }, and the C node streams the bundled gain config at that geometry (the built-in ONNX
// Runtime engine, a Hard contract of blocks of exactly one quantum): the output is the input
// times the Static gain after the handler's latency, sample for sample. A browser without the
// hint (renderQuantumSize absent) skips with a line. The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
  measureRenderGeometry,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const RATE = 48000
const GAIN = 0.5

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogHook(() => {})
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  await context.ready()

  for (const size of [256, 480]) {
    const frames = size * 40
    const audio = new OfflineAudioContext({
      numberOfChannels: 1,
      length: frames,
      sampleRate: RATE,
      renderSizeHint: size,
    })
    if (audio.renderQuantumSize === undefined) {
      check(true, `renderSizeHint ${size}: not supported by this browser, skipped`)
      continue
    }
    const geometry = await measureRenderGeometry(audio)
    check(
      audio.renderQuantumSize === size &&
        geometry.quantum === size &&
        geometry.burst === size,
      `renderSizeHint ${size}: renderQuantumSize ${audio.renderQuantumSize}, geometry ` +
        `{ quantum: ${geometry.quantum}, burst: ${geometry.burst} }`
    )
    const contract = (
      await Contract.fromUrl(
        anira,
        new URL(`${MODELS}gain.contract.json`, import.meta.url)
      )
    ).setGeometry(geometry.quantum, geometry.burst, RATE)
    const pipeline = Pipeline.create(anira).addInference(model)
    const handler = Handler.create(context, pipeline).prepare(contract)
    pipeline.destroy()
    contract.destroy()
    const latency = handler.latency()
    const audioNode = await handler.createAudioNode(audio, { params: [1], waitMs: 500 })
    audioNode.params.get(1).value = GAIN

    const samples = new Float32Array(frames)
    for (let i = 0; i < frames; i++)
      samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
    const buffer = audio.createBuffer(1, frames, RATE)
    buffer.copyToChannel(samples, 0)
    const source = new AudioBufferSourceNode(audio, { buffer })
    source.connect(audioNode.node).connect(audio.destination)
    source.start(0)
    const rendered = (await audio.startRendering()).getChannelData(0)
    let worst = 0
    for (let i = latency; i < frames; i++) {
      worst = Math.max(worst, Math.abs(rendered[i] - GAIN * samples[i - latency]))
    }
    check(
      latency > 0 && worst < 1e-6,
      `renderSizeHint ${size}: the node renders the input times ${GAIN} after ${latency} ` +
        `samples (worst error ${worst})`
    )
    audioNode.destroy()
    handler.destroy()
  }

  model.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
