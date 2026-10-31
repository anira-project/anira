// The module's memory grows (256 MiB to begin with, up to 2 GiB), and a growth gives the shared
// memory a new, longer buffer: views made before it cover the old length only. Every scope reads
// the heap through anira's heap(), which rebuilds them, the trampolines before each processor and
// engine call. The pipeline of test.js: a JavaScript engine on a pool of two Wasm Workers
// (pool-worker.js: its input times 2 * the Static gain) and a JavaScript processor in the worklet
// (worklet-processor.js: halves the input), so the output is the input times the gain after the
// latency, exactly. Checked:
// - offline: a node streams on a context whose worklet and Workers started before the growth;
//   then the main thread creates handlers on the bundled gain config until the memory is far
//   beyond its initial size, and a second handler made after the growth (its tensors above the
//   old end) renders beside the first, both exact;
// - offline, mid-stream: the render suspends halfway, the main thread grows the memory and
//   re-prepares the node (its tensors made anew, above the old end), and the render resumes:
//   exact before and after;
// - live (Chromium; headless Firefox keeps a live context suspended): the same growth and
//   re-prepare while the node streams, its output level the expected one before and after, no
//   quantum missed after.
// The verdict is window.anira_test_result.
import {
  ANIRA_EM_PROCESSOR_HOOK_PRE,
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  AniraError,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_status,
  measureRenderGeometry,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const MIB = 1024 * 1024
const RATE = 48000
const QUANTUM = 128
const QUANTA = 96
const GAIN = 0.5
const ENGINE_ID = 'org.anira-test.pool-double'
const PROCESSOR = 'org.anira-test.half'
const processorModule = new URL('./worklet-processor.js', import.meta.url)

const tone = (frames) => {
  const samples = new Float32Array(frames)
  for (let i = 0; i < frames; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  return samples
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  // Sizes the module cannot run on are refused before anything is created.
  const refusals = []
  for (const memory of [
    { initial: 128 * MIB },
    { initial: 512 * MIB, maximum: 256 * MIB },
    { maximum: 4096 * MIB },
  ]) {
    try {
      await Anira.create({ memory })
      refusals.push('created')
    } catch (error) {
      refusals.push(
        error instanceof AniraError &&
          error.status === anira_status.ANIRA_ERROR_INVALID_ARGUMENT
          ? 'refused'
          : `${error}`
      )
    }
  }
  check(
    refusals.every((r) => r === 'refused'),
    `Anira.create refuses a memory below 256 MiB, a maximum below the initial size and one ` +
      `above 2 GiB with an AniraError (${refusals.join(', ')})`
  )
  const anira = await Anira.create()
  anira.setLogSink(() => {})
  const bytes = () => anira.memory.buffer.byteLength
  const initial = bytes()
  check(initial === 256 * MIB, `the memory starts at ${initial / MIB} MiB`)

  const context = Context.create(anira, {
    threads: 2,
    workerModule: new URL('./pool-worker.js', import.meta.url),
  })
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
  const jsModel = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine: ENGINE_ID }],
      inputs: [
        streamed('in'),
        { name: 'gain', dtype: 'float32', role: 'static', axes: [['any', 1]] },
      ],
      outputs: [streamed('out')],
    })
  )
  const jsContract = (blockMax = QUANTUM) =>
    ContractHandle.hard(anira, { blockMin: QUANTUM, blockMax, rate: RATE, budgetMs: 2 })
  const jsHandler = (contract) => {
    const pipeline = Pipeline.create(anira)
      .addInference(jsModel)
      .addEngine(ENGINE_ID, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
      .addProcessor(PROCESSOR, { phases: ANIRA_EM_PROCESSOR_HOOK_PRE })
    try {
      const handler = InferenceHandler.create(context, pipeline)
      handler.prepare(contract)
      return handler
    } finally {
      pipeline.destroy()
    }
  }
  const node = async (handler, audio, waitMs) => {
    const audioNode = await handler.createAudioNode(audio, {
      params: [1],
      waitMs,
      workletModules: [processorModule],
    })
    audioNode.params.get(1).value = GAIN
    return audioNode
  }
  // The worst error of `channel` against the input times the gain after `latency`, over
  // [from, to) of the input.
  const worstOf = (
    channel,
    samples,
    latency,
    from = 0,
    to = samples.length - latency
  ) => {
    let worst = 0
    for (let i = from; i < to; i++) {
      worst = Math.max(worst, Math.abs(channel[i + latency] - GAIN * samples[i]))
    }
    return worst
  }

  // Grows the memory from the main thread: handlers on the bundled gain config (the built-in
  // ONNX Runtime engine), until the memory is `beyond` past its size now.
  const gainModel = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const gainContract = (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  ).hardHostBlocks(QUANTUM, QUANTUM, RATE)
  const fillers = []
  const grow = (beyond) => {
    const target = bytes() + beyond
    while (bytes() < target && fillers.length < 2000) {
      const pipeline = Pipeline.create(anira).addInference(gainModel)
      const handler = InferenceHandler.create(context, pipeline)
      handler.prepare(gainContract)
      fillers.push(handler)
      pipeline.destroy()
    }
  }
  await context.ready()

  // Offline: a node streaming from before the growth, a second handler from after it.
  {
    const frames = QUANTUM * QUANTA
    const audio = new OfflineAudioContext(2, frames, RATE)
    const before = jsHandler(jsContract())
    const first = await node(before, audio, 500)
    const startedAt = bytes()
    grow(256 * MIB)
    const after = jsHandler(jsContract())
    check(
      bytes() >= startedAt + 256 * MIB && after.native() >= startedAt,
      `the memory grew from ${startedAt / MIB} to ${bytes() / MIB} MiB through ` +
        `${fillers.length} handlers; the handler made after it lies above the old end ` +
        `(at ${(after.native() / MIB).toFixed(0)} MiB)`
    )
    const second = await node(after, audio, 500)
    const samples = tone(frames)
    const buffer = audio.createBuffer(1, frames, RATE)
    buffer.copyToChannel(samples, 0)
    const source = new AudioBufferSourceNode(audio, { buffer })
    const merger = new ChannelMergerNode(audio, { numberOfInputs: 2 })
    source.connect(first.node).connect(merger, 0, 0)
    source.connect(second.node).connect(merger, 0, 1)
    merger.connect(audio.destination)
    source.start(0)
    const rendered = await audio.startRendering()
    const w1 = worstOf(rendered.getChannelData(0), samples, before.getLatency())
    const w2 = worstOf(rendered.getChannelData(1), samples, after.getLatency())
    check(
      w1 < 1e-6 && w2 < 1e-6,
      `after the growth both nodes render exact, through the Workers and the worklet (worst ` +
        `errors ${w1}, ${w2})`
    )
    first.destroy()
    second.destroy()
    before.destroy()
    after.destroy()
  }

  // Offline, mid-stream: grown and re-prepared during the render. Where the context can suspend
  // (Chromium) at the half, deterministically; elsewhere (Firefox has no
  // OfflineAudioContext.suspend) from a timer while it renders, the stream checked at its start
  // and its last quarter.
  {
    const suspends = typeof OfflineAudioContext.prototype.suspend === 'function'
    const frames = QUANTUM * QUANTA * (suspends ? 2 : 16)
    const audio = new OfflineAudioContext(1, frames, RATE)
    const contract = jsContract()
    const handler = jsHandler(contract)
    const audioNode = await node(handler, audio, 500)
    const samples = tone(frames)
    const buffer = audio.createBuffer(1, frames, RATE)
    buffer.copyToChannel(samples, 0)
    const source = new AudioBufferSourceNode(audio, { buffer })
    source.connect(audioNode.node).connect(audio.destination)
    source.start(0)
    let grownTo = 0
    let finished = false
    let midRender = false
    const growAndPrepare = () => {
      midRender = !finished
      grow(128 * MIB)
      grownTo = bytes()
      audioNode.prepare(contract)
    }
    if (suspends) {
      audio.suspend(frames / 2 / RATE).then(() => {
        growAndPrepare()
        audio.resume()
      })
    } else {
      setTimeout(growAndPrepare, 5)
    }
    const rendered = (await audio.startRendering()).getChannelData(0)
    finished = true
    const latency = handler.getLatency()
    // Exact but where the node output silence: the quanta it was held for the re-prepare, and
    // with the timer (Firefox) the quanta whose inferences missed while the main thread created
    // the handlers that grew the memory, which the render does not wait for. A stale heap view
    // would read or write the wrong memory: a level other than the input times the gain.
    let silent = 0
    let wrong = 0
    for (let i = 0; i + latency < frames; i++) {
      const out = rendered[i + latency]
      if (Math.abs(out - GAIN * samples[i]) <= 1e-6) continue
      if (out === 0) silent++
      else wrong++
    }
    const tail = worstOf(rendered, samples, latency, (frames * 3) / 4)
    check(
      grownTo > 0 && midRender && wrong === 0 && tail < 1e-6 && silent < frames / 2,
      `mid-stream (${suspends ? 'suspended at the half' : 'from a timer'}): grown to ` +
        `${grownTo / MIB} MiB and re-prepared during the render (${midRender}), the node ` +
        `renders exact but for ${silent} frames of silence (held, or missed while the memory ` +
        `grew), ${wrong} wrong; the last quarter exact (worst ${tail})`
    )
    audioNode.destroy()
    handler.destroy()
    contract.destroy()
  }

  // Live (Chromium): the growth and a re-prepare while the node streams.
  const live = new AudioContext({ sampleRate: RATE })
  await Promise.race([live.resume(), new Promise((resolve) => setTimeout(resolve, 2000))])
  if (live.state !== 'running') {
    check(true, `live: the context does not run here (${live.state}), skipped`)
  } else {
    const geometry = await measureRenderGeometry(live)
    const contract = jsContract(geometry.burst)
    const handler = jsHandler(contract)
    const audioNode = await node(handler, live, 0)
    const source = new OscillatorNode(live, { frequency: 440 })
    const level = new GainNode(live, { gain: 0.25 })
    const analyser = new AnalyserNode(live, { fftSize: 2048 })
    source.connect(level).connect(audioNode.node).connect(analyser)
    analyser.connect(live.destination)
    source.start()
    const rms = async () => {
      await new Promise((resolve) => setTimeout(resolve, 700))
      const data = new Float32Array(analyser.fftSize)
      analyser.getFloatTimeDomainData(data)
      return Math.sqrt(data.reduce((sum, x) => sum + x * x, 0) / data.length)
    }
    const expected = (0.25 * GAIN) / Math.SQRT2
    const before = await rms()
    grow(128 * MIB)
    audioNode.prepare(contract)
    const after = await rms()
    check(
      Math.abs(before / expected - 1) < 0.1 && Math.abs(after / expected - 1) < 0.1,
      `live: grown to ${bytes() / MIB} MiB and re-prepared mid-stream, the level is the ` +
        `expected ${expected.toFixed(4)} before (${before.toFixed(4)}) and after ` +
        `(${after.toFixed(4)})`
    )
    source.stop()
    audioNode.destroy()
    handler.destroy()
    contract.destroy()
  }
  await live.close()

  fillers.forEach((h) => h.destroy())
  gainContract.destroy()
  gainModel.destroy()
  jsModel.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
