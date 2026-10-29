// A live AudioContext (Chromium only: headless Firefox keeps a live context suspended, it has no
// audio output to run against). A live context renders each device callback as a burst of render
// quanta back to back (headless Chromium: 2048 frames, 16 quanta), so a handler must be prepared
// for blocks up to that burst for its inferences to keep up. measureRenderGeometry gives the
// geometry: from baseLatency, and from the probe (forced) the same or up to two quanta more (a
// device buffer of 441 frames reads as 512; a loaded runner's probe saw 640). The bundled gain
// config (extras/models/model-pool/gain.model.json, the built-in ONNX Runtime engine) is prepared
// with it and the miss policy ZEROS, a tone streams through the handler's audio node, and a tap
// after the node counts the quanta that are all zero (a missed quantum: the tone never is) once the
// stream has started. Expected: 0. While it streams, a load processor spins in every call and
// stalls in every 16th (as a model's JavaScript stage does once per callback), and the probe, three
// times under each of two loads, must measure the burst read or more. Then a node streaming at
// block_max = one quantum (which misses where the burst exceeds its latency) is re-prepared
// mid-stream through audioNode.prepare to a block_max past that latency: a plain handler.prepare is
// refused meanwhile, the output goes on, the latency read back changes, and no quantum misses after
// the re-prepare. Where the browser takes renderSizeHint "hardware", the geometry is one quantum of
// the device's size. The verdict is window.anira_test_result.
import {
  Anira,
  AniraError,
  Context,
  Contract,
  Handler,
  ModelConfig,
  Pipeline,
  anira_miss_policy,
  anira_status,
  measureRenderGeometry,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
// The tap counts from this many seconds after the node starts, for this many seconds.
const SETTLE_S = 0.75
const MEASURE_S = 2

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// Counts the all-zero quanta of its input between two frame marks (set by a message, in
// frames from the tap's start); answers a message without marks with the counts.
const TAP = `registerProcessor('anira-test-tap', class extends AudioWorkletProcessor {
  constructor() {
    super()
    this.from = Infinity
    this.to = Infinity
    this.frame = 0
    this.blocks = 0
    this.zeros = 0
    this.port.onmessage = (event) => {
      if (event.data) {
        this.from = this.frame + event.data.afterFrames
        this.to = this.from + event.data.forFrames
        this.blocks = 0
        this.zeros = 0
      } else {
        this.port.postMessage({ blocks: this.blocks, zeros: this.zeros })
      }
    }
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    const frames = channel ? channel.length : 128
    if (this.frame >= this.from && this.frame < this.to) {
      this.blocks++
      if (!channel || channel.every((x) => x === 0)) this.zeros++
    }
    this.frame += frames
    return true
  }
})
// Load on the audio thread, as a model's JavaScript stage puts there: every call spins for a
// moment, and every 16th call (once per callback of 16 quanta, where a stage's pre_process
// unfolds an inference's input) for longer. A message stops it.
registerProcessor('anira-test-load', class extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.spinMs = options.processorOptions.spinMs
    this.stallMs = options.processorOptions.stallMs
    this.calls = 0
    this.stopped = false
    this.port.onmessage = () => (this.stopped = true)
  }
  process() {
    if (this.stopped) return false
    this.calls++
    const end = Date.now() + (this.calls % 16 === 0 ? this.stallMs : this.spinMs)
    let x = 0
    while (Date.now() < end) x += Math.sqrt(x + 1)
    return true
  }
})`

// A stream through the gain config at `geometry`: the handler, its node, a tone and the tap.
const stream = async (anira, audio, geometry) => {
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const contract = (
    await Contract.fromUrl(anira, new URL(`${MODELS}gain.contract.json`, import.meta.url))
  ).setGeometry(geometry.quantum, geometry.burst, audio.sampleRate)
  anira.abi.anira_contract_hard_set_on_miss(
    contract.ptr,
    anira_miss_policy.ANIRA_MISS_ZEROS
  )
  const pipeline = Pipeline.create(anira).addInference(model)
  const handler = Handler.create(context, pipeline).prepare(contract)
  pipeline.destroy()
  await context.ready()

  const audioNode = await handler.createAudioNode(audio, { params: [1] })
  audioNode.params.get(1).value = 1
  const tap = new AudioWorkletNode(audio, 'anira-test-tap')
  const tone = new OscillatorNode(audio, { frequency: 440 })
  const level = new GainNode(audio, { gain: 0.25 })
  tone.connect(level).connect(audioNode.node).connect(tap).connect(audio.destination)
  tone.start()
  const count = async () => {
    tap.port.postMessage({
      afterFrames: SETTLE_S * audio.sampleRate,
      forFrames: MEASURE_S * audio.sampleRate,
    })
    await delay((SETTLE_S + MEASURE_S + 0.25) * 1000)
    return new Promise((resolve) => {
      tap.port.onmessage = (event) => resolve(event.data)
      tap.port.postMessage(null)
    })
  }
  const stop = () => {
    tone.stop()
    tone.disconnect()
    audioNode.node.disconnect()
    tap.disconnect()
    audioNode.destroy()
    handler.destroy()
    contract.destroy()
    model.destroy()
    context.destroy()
  }
  return { handler, audioNode, contract, count, stop }
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
  anira.setLogHook(() => {}) // the misses of the one-quantum run warn; not shown
  const stopLogPump = anira.startLogPump()

  // The geometry: renderQuantumSize where the browser has it, baseLatency's burst, the probe's.
  const geometry = await measureRenderGeometry(audio)
  const quantum = audio.renderQuantumSize > 0 ? audio.renderQuantumSize : 128
  const fromBase =
    Math.ceil((audio.baseLatency * audio.sampleRate) / quantum - 1e-9) * quantum
  check(
    geometry.quantum === quantum && geometry.burst === fromBase,
    `measureRenderGeometry: quantum ${geometry.quantum} (renderQuantumSize ` +
      `${audio.renderQuantumSize ?? 'absent'}), burst ${geometry.burst} (baseLatency ` +
      `${audio.baseLatency})`
  )
  const probed = await measureRenderGeometry(audio, { probe: true })
  // The probe never under-reports, and may over-report: on a loaded machine two device
  // callbacks can run almost back to back and merge into one run.
  check(
    probed.burst >= geometry.burst && probed.burst <= geometry.burst + 2 * quantum,
    `the probe measures the burst read, up to two quanta more: ${probed.burst} frames ` +
      `(read ${geometry.burst})`
  )

  // At the measured geometry: no quantum misses once the stream runs.
  const sized = await stream(anira, audio, geometry)
  const counts = await sized.count()
  check(
    counts.blocks > 0 && counts.zeros === 0,
    `at quantum ${geometry.quantum}, burst ${geometry.burst}: ${counts.zeros} of ` +
      `${counts.blocks} quanta missed (latency ${sized.handler.latency()})`
  )

  // The probe on a loaded audio thread: the node streams, and a load processor spins in every
  // call and stalls once per 16. A run of back-to-back calls then shows late calls inside it,
  // which split at a fixed gap gave a burst of 15 quanta of 16 (a model with a JavaScript stage
  // re-measured 6139 -> 5887 samples of latency). The probe must never report less than the
  // burst read.
  for (const [spinMs, stallMs] of [
    [0.3, 4],
    [0.8, 8],
  ]) {
    const load = new AudioWorkletNode(audio, 'anira-test-load', {
      numberOfInputs: 0,
      processorOptions: { spinMs, stallMs },
    })
    const muted = new GainNode(audio, { gain: 0 })
    load.connect(muted).connect(audio.destination)
    await delay(200)
    const bursts = []
    for (let k = 0; k < 3; k++)
      bursts.push((await measureRenderGeometry(audio, { probe: true })).burst)
    load.port.postMessage('stop')
    load.disconnect()
    muted.disconnect()
    check(
      bursts.every((b) => b >= geometry.burst),
      `loaded (${spinMs} ms per call, ${stallMs} ms every 16th, a node streaming): the probe ` +
        `measures ${bursts.join(', ')} frames, never below the burst read (${geometry.burst})`
    )
  }
  sized.stop()

  // A node at block_max = one quantum, re-prepared mid-stream at a block_max past its latency.
  // It misses only where the burst exceeds its latency (a device buffer of 2048 frames against
  // the latency of 128/128; not one of 512).
  const small = await stream(anira, audio, { quantum, burst: quantum })
  const before = await small.count()
  const latencyBefore = small.handler.latency()
  if (geometry.burst > latencyBefore) {
    check(
      before.zeros > 0,
      `at block_max ${quantum}: ${before.zeros} of ${before.blocks} quanta missed (burst ` +
        `${geometry.burst} > latency ${latencyBefore})`
    )
  } else {
    check(
      true,
      `at block_max ${quantum}: the miss demo skipped, the burst ${geometry.burst} is within ` +
        `the latency ${latencyBefore} (${before.zeros} of ${before.blocks} quanta missed)`
    )
  }
  let refused = null
  try {
    small.handler.prepare(small.contract)
  } catch (error) {
    refused = error
  }
  check(
    refused instanceof AniraError &&
      refused.code === anira_status.ANIRA_ERROR_INVALID_STATE,
    `a plain handler.prepare is refused while the node streams (${refused?.message})`
  )
  // Four bursts, doubled until past the latency before: the latency covers block_max, so it
  // changes.
  let blockMax = 4 * geometry.burst
  while (blockMax <= latencyBefore) blockMax *= 2
  small.contract.setGeometry(geometry.quantum, blockMax, audio.sampleRate)
  const held = performance.now()
  small.audioNode.prepare(small.contract)
  const heldMs = performance.now() - held
  const after = await small.count()
  const latencyAfter = small.handler.latency()
  check(
    after.blocks > 0 && after.zeros === 0,
    `re-prepared mid-stream through the node to block_max ${blockMax}: ${after.zeros} of ` +
      `${after.blocks} quanta missed after`
  )
  check(
    latencyAfter > latencyBefore,
    `the latency read back changed: ${latencyBefore} -> ${latencyAfter} samples (the node ` +
      `held off ${heldMs.toFixed(1)} ms, ~${Math.ceil((heldMs / 1000) * audio.sampleRate)} ` +
      `frames of silence)`
  )
  small.stop()

  // renderSizeHint "hardware", where the browser takes it.
  const hardware = new AudioContext({ renderSizeHint: 'hardware' })
  if (!(hardware.renderQuantumSize > 0)) {
    check(true, 'renderSizeHint "hardware": not supported by this browser, skipped')
  } else {
    await hardware.resume()
    await hardware.audioWorklet.addModule(
      URL.createObjectURL(new Blob([TAP], { type: 'text/javascript' }))
    )
    const g = await measureRenderGeometry(hardware)
    check(
      g.quantum === hardware.renderQuantumSize && g.burst === g.quantum,
      `renderSizeHint "hardware": quantum ${g.quantum} = burst ${g.burst}`
    )
    const run = await stream(anira, hardware, g)
    const c = await run.count()
    check(c.blocks > 0 && c.zeros === 0, `and ${c.zeros} of ${c.blocks} quanta missed`)
    run.stop()
  }
  await hardware.close()

  stopLogPump()
  await audio.close()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
