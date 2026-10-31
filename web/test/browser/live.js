// A live AudioContext (Chromium only: headless Firefox keeps a live context suspended, it has no
// audio output to run against). A live context renders each device callback as a burst of render
// quanta back to back (headless Chromium: 2048 frames, 16 quanta), so a handler must be prepared
// for blocks up to that burst for its inferences to keep up. measureRenderGeometry measures it; the
// test's oracle is Chromium's baseLatency (its callback buffer, in whole quanta), read here and
// never by the library. The bundled gain config (extras/models/model-pool/gain.model.json, the
// built-in ONNX Runtime engine) is prepared at the measured geometry with the miss policy ZEROS, a
// tone streams through the handler's audio node, and a tap after the node counts the quanta that
// are all zero (a missed quantum: the tone never is) once the stream has started. Expected: 0.
// While it streams: a load processor that is not anira's spins in every call and stalls in every
// 16th, and the probe must measure the device's burst or more; then a second node whose JavaScript
// processor spins 8 ms per inference streams too, and with both nodes held the probe measures the
// device's burst exactly (unheld, logged). Then a node streaming at host_block_max = one quantum (which
// misses where the burst exceeds its latency) is re-prepared mid-stream through audioNode.prepare
// to a host_block_max past that latency: a plain handler.prepare is refused meanwhile, the output goes
// on, the latency read back changes, and no quantum misses after the re-prepare. Last, contexts
// created with renderSizeHint 256, 480, 2047 and "hardware" where the browser takes it: the quantum
// is the size, the burst the device's, and a node at that geometry misses no quantum. The verdict
// is window.anira_test_result.
import {
  ANIRA_EM_PROCESSOR_HOOK_PRE,
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_miss_policy,
  anira_status,
  measureRenderGeometry,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const LOAD_PROCESSOR = 'org.anira-test.load'
// The tap counts from this many seconds after the node starts, for this many seconds.
const SETTLE_S = 0.75
const MEASURE_S = 2

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

// The test's oracle of a Chromium context's burst: its baseLatency (the callback buffer over the
// rate) in whole quanta.
const burstOf = (context, quantum) =>
  Math.ceil((context.baseLatency * context.sampleRate) / quantum - 1e-9) * quantum

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
// Load on the audio thread, as a model's JavaScript processor puts there: every call spins for a
// moment, and every 16th call (once per callback of 16 quanta, where a processor's pre_process
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

// A stream through the gain config at `geometry`: the handler, its node, a tone and the tap; with
// `load`, the pipeline carries load-processor.js's processor, which spins on the audio thread once
// per inference.
const stream = async (anira, audio, geometry, { load = false } = {}) => {
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const contract = (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  ).hardHostBlocks(geometry.quantum, geometry.burst, audio.sampleRate)
  anira.abi.anira_contract_hard_set_on_miss(
    contract.native(),
    anira_miss_policy.ANIRA_MISS_ZEROS
  )
  const pipeline = Pipeline.create(anira).addInference(model)
  if (load) pipeline.addProcessor(LOAD_PROCESSOR, { phases: ANIRA_EM_PROCESSOR_HOOK_PRE })
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  pipeline.destroy()
  await context.ready()

  const audioNode = await handler.createAudioNode(audio, {
    params: [1],
    workletModules: load ? [new URL('./load-processor.js', import.meta.url)] : [],
  })
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
  anira.setLogSink(() => {}) // the misses of the one-quantum run warn; not shown
  const stopLogPump = anira.startLogPump()

  // The geometry, measured. The oracle is Chromium's baseLatency, its callback buffer, in whole
  // quanta: read here in the test only, the library never reads it.
  const quantum = audio.renderQuantumSize > 0 ? audio.renderQuantumSize : 128
  const device = burstOf(audio, quantum)
  const started = performance.now()
  const geometry = await measureRenderGeometry(audio)
  const tookMs = (performance.now() - started).toFixed(0)
  check(
    geometry.quantum === quantum && geometry.burst === device,
    `measureRenderGeometry, idle: quantum ${geometry.quantum} (renderQuantumSize ` +
      `${audio.renderQuantumSize ?? 'absent'}), burst ${geometry.burst}, the device's ` +
      `${device} (baseLatency ${audio.baseLatency}), in ${tookMs} ms`
  )

  // At the measured geometry: no quantum misses once the stream runs.
  const sized = await stream(anira, audio, geometry)
  const counts = await sized.count()
  check(
    counts.blocks > 0 && counts.zeros === 0,
    `at quantum ${geometry.quantum}, burst ${geometry.burst}: ${counts.zeros} of ` +
      `${counts.blocks} quanta missed (latency ${sized.handler.getLatency()})`
  )

  // The probe beside load that is not anira's (nothing to hold): a load processor spins in
  // every call and stalls once per 16. The runs then show late calls inside them, which split
  // at a fixed gap gave a burst of 15 quanta of 16; the probe must not report less than the
  // device's burst.
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
    for (let k = 0; k < 3; k++) bursts.push((await measureRenderGeometry(audio)).burst)
    load.port.postMessage('stop')
    load.disconnect()
    muted.disconnect()
    check(
      bursts.every((b) => b >= device),
      `loaded (${spinMs} ms per call, ${stallMs} ms every 16th, a node streaming): the probe ` +
        `measures ${bursts.join(', ')} frames, never below the device's ${device}`
    )
  }

  // The load in an anira node, held: a node whose JavaScript processor spins 8 ms per inference
  // (one every 512 frames: four per callback of 2048) streams beside the first. Unheld the
  // probe sees that load (logged); holding both nodes leaves the audio thread quiet, and the
  // probe measures the device's burst exactly.
  const loaded = await stream(anira, audio, geometry, { load: true })
  await delay(300)
  const unheld = (await measureRenderGeometry(audio)).burst
  const heldBursts = []
  for (let k = 0; k < 3; k++) {
    heldBursts.push(
      (await measureRenderGeometry(audio, { hold: [sized.audioNode, loaded.audioNode] }))
        .burst
    )
  }
  check(
    heldBursts.every((b) => b === device),
    `a node with a processor spinning 8 ms per inference: held, the probe measures ` +
      `${heldBursts.join(', ')} frames, the device's ${device} (unheld: ${unheld})`
  )
  const resumed = await loaded.count()
  check(
    resumed.blocks > 0,
    `the held nodes stream again after the probe (${resumed.blocks} quanta counted, ` +
      `${resumed.zeros} silent under the processor's load)`
  )
  loaded.stop()
  sized.stop()

  // A node at host_block_max = one quantum, re-prepared mid-stream at a host_block_max past its latency.
  // It misses only where the burst exceeds its latency (a device buffer of 2048 frames against
  // the latency of 128/128; not one of 512).
  const small = await stream(anira, audio, { quantum, burst: quantum })
  const before = await small.count()
  const latencyBefore = small.handler.getLatency()
  if (geometry.burst > latencyBefore) {
    check(
      before.zeros > 0,
      `at host_block_max ${quantum}: ${before.zeros} of ${before.blocks} quanta missed (burst ` +
        `${geometry.burst} > latency ${latencyBefore})`
    )
  } else {
    check(
      true,
      `at host_block_max ${quantum}: the miss demo skipped, the burst ${geometry.burst} is within ` +
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
      refused.status === anira_status.ANIRA_ERROR_INVALID_STATE,
    `a plain handler.prepare is refused while the node streams (${refused?.message})`
  )
  // Four bursts, doubled until past the latency before: the latency covers host_block_max, so it
  // changes.
  let hostBlockMax = 4 * geometry.burst
  while (hostBlockMax <= latencyBefore) hostBlockMax *= 2
  small.contract.hardHostBlocks(geometry.quantum, hostBlockMax, audio.sampleRate)
  const held = performance.now()
  small.audioNode.prepare(small.contract)
  const heldMs = performance.now() - held
  const after = await small.count()
  const latencyAfter = small.handler.getLatency()
  check(
    after.blocks > 0 && after.zeros === 0,
    `re-prepared mid-stream through the node to host_block_max ${hostBlockMax}: ${after.zeros} of ` +
      `${after.blocks} quanta missed after`
  )
  check(
    latencyAfter > latencyBefore,
    `the latency read back changed: ${latencyBefore} -> ${latencyAfter} samples (the node ` +
      `held off ${heldMs.toFixed(1)} ms, ~${Math.ceil((heldMs / 1000) * audio.sampleRate)} ` +
      `frames of silence)`
  )
  small.stop()

  // Contexts of another render size, where the browser takes renderSizeHint: 256 and 480 frames
  // (480 is no power of two: a callback of 2048 renders 4 or 5 of them; on CI's callback of 481
  // frames one, and two back to back once every 480 callbacks: a burst of 960 no window of the
  // probe shows, which it takes from the frames per callback), 2047 (one frame short of
  // headless Chromium's 2048: the same case here, one quantum per callback and a double once
  // every 2047, a burst of 4094), and "hardware", the size the browser deems best (Chromium:
  // 128; renderQuantumSize is 0 until the context runs).
  for (const hint of [256, 480, 2047, 'hardware']) {
    const sizedContext = new AudioContext({ sampleRate: 48000, renderSizeHint: hint })
    if (sizedContext.renderQuantumSize === undefined) {
      check(true, `renderSizeHint ${hint}: not supported by this browser, skipped`)
      await sizedContext.close()
      continue
    }
    await sizedContext.resume()
    await sizedContext.audioWorklet.addModule(
      URL.createObjectURL(new Blob([TAP], { type: 'text/javascript' }))
    )
    const size = sizedContext.renderQuantumSize
    const g = await measureRenderGeometry(sizedContext)
    const expected = burstOf(sizedContext, size)
    check(
      g.quantum === size &&
        (hint === 'hardware' || size === hint) &&
        g.burst === expected,
      `renderSizeHint ${hint}: quantum ${g.quantum} (renderQuantumSize ${size}), burst ` +
        `${g.burst}, the device's ${expected}`
    )
    const run = await stream(anira, sizedContext, g)
    const c = await run.count()
    check(
      c.blocks > 0 && c.zeros === 0,
      `renderSizeHint ${hint}: a node at quantum ${size}, burst ${g.burst}: ${c.zeros} of ` +
        `${c.blocks} quanta missed`
    )
    run.stop()
    await sizedContext.close()
  }

  stopLogPump()
  await audio.close()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
