// A live AudioContext (Chromium only: headless Firefox keeps a live context suspended, it has no
// audio output to run against). A live context renders each device callback as a burst of
// render quanta back to back (headless Chromium: 2048 frames, 16 quanta), so a handler must be
// prepared for blocks up to that burst for its inferences to keep up. The bundled gain config
// (extras/models/model-pool/gain.model.json, the built-in ONNX Runtime engine, a pool of one
// Wasm Worker) is prepared with the geometry of renderBurstFrames and the miss policy ZEROS, a
// tone streams through the handler's audio node, and a tap after the node counts the
// 128-frame blocks that are all zero (a missed quantum: the tone never is) once the stream has
// started. Expected: 0. The same with block_max 128 is reported for comparison (the misses the
// helper prevents). The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
  anira_miss_policy,
  renderBurstFrames,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const QUANTUM = 128
// The tap counts from this many seconds after the node starts, for this many seconds.
const SETTLE_S = 0.75
const MEASURE_S = 2

// Counts the all-zero quanta of its input between two frame marks; answers a message with them.
const TAP = `registerProcessor('anira-test-tap', class extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.from = options.processorOptions.from
    this.to = options.processorOptions.to
    this.frame = 0
    this.blocks = 0
    this.zeros = 0
    this.port.onmessage = () => this.port.postMessage({ blocks: this.blocks, zeros: this.zeros })
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (this.frame >= this.from && this.frame < this.to) {
      this.blocks++
      if (!channel || channel.every((x) => x === 0)) this.zeros++
    }
    this.frame += 128
    return true
  }
})`

const measure = async (anira, audio, blockMax) => {
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const contract = (
    await Contract.fromUrl(anira, new URL(`${MODELS}gain.contract.json`, import.meta.url))
  ).setGeometry(QUANTUM, blockMax, audio.sampleRate)
  anira.abi.anira_contract_hard_set_on_miss(
    contract.ptr,
    anira_miss_policy.ANIRA_MISS_ZEROS
  )
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = Handler.create(context, pipeline).prepare(contract)
  pipeline.destroy()
  contract.destroy()
  const latency = handler.latency()
  await context.ready()

  const audioNode = await handler.createAudioNode(audio, { params: [1] })
  audioNode.params.get(1).value = 1
  const tap = new AudioWorkletNode(audio, 'anira-test-tap', {
    processorOptions: {
      from: SETTLE_S * audio.sampleRate,
      to: (SETTLE_S + MEASURE_S) * audio.sampleRate,
    },
  })
  const tone = new OscillatorNode(audio, { frequency: 440 })
  const level = new GainNode(audio, { gain: 0.25 })
  tone.connect(level).connect(audioNode.node).connect(tap).connect(audio.destination)
  tone.start()
  await new Promise((resolve) =>
    setTimeout(resolve, (SETTLE_S + MEASURE_S + 0.25) * 1000)
  )
  const counts = await new Promise((resolve) => {
    tap.port.onmessage = (event) => resolve(event.data)
    tap.port.postMessage(null)
  })
  tone.stop()
  tone.disconnect()
  audioNode.node.disconnect()
  tap.disconnect()
  audioNode.destroy()
  handler.destroy()
  model.destroy()
  context.destroy()
  return { ...counts, latency }
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })

  const audio = new AudioContext({ sampleRate: 48000 })
  await audio.resume()
  check(audio.state === 'running', `the live context runs (${audio.state})`)
  await audio.audioWorklet.addModule(
    URL.createObjectURL(new Blob([TAP], { type: 'text/javascript' }))
  )

  const anira = await Anira.create()
  anira.setLogHook(() => {}) // the misses of the comparison run warn; not shown
  const stopLogPump = anira.startLogPump()

  const burst = renderBurstFrames(audio)
  check(
    burst >= QUANTUM && burst % QUANTUM === 0,
    `renderBurstFrames: ${burst} frames (baseLatency ${audio.baseLatency}, outputLatency ` +
      `${audio.outputLatency})`
  )
  const sized = await measure(anira, audio, burst)
  check(
    sized.blocks > 0 && sized.zeros === 0,
    `block_max ${burst}: ${sized.zeros} of ${sized.blocks} quanta missed after the stream ` +
      `started (latency ${sized.latency})`
  )
  const quantum = await measure(anira, audio, QUANTUM)
  check(
    quantum.blocks > 0,
    `for comparison, block_max ${QUANTUM}: ${quantum.zeros} of ${quantum.blocks} quanta ` +
      `missed (latency ${quantum.latency})`
  )

  stopLogPump()
  await audio.close()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
