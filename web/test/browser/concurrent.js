// Audio nodes created at once on one audio context: each node's creation is its own call (its
// own completion words), and the context's first-node setup (the worklet scope's clock, its
// registration with the module, the worklet thread) is shared by the creations in flight. The
// bundled gain config on the built-in ONNX Runtime engine, one handler per node, each with its
// own Static gain. Offline (Chromium and Firefox): four nodes created at once on a fresh context,
// each channel the input times its gain after the handler's latency; then on another fresh
// context two at once, a fifth node, and two more created while the fifth is destroyed. Live
// (Chromium; headless Firefox keeps a live context suspended): two nodes at once on a fresh
// context, then four, each playing.
// The verdict is window.anira_test_result.
import {
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  measureRenderGeometry,
} from '../../dist/index.js'

const MODELS = '../../extras/models/model-pool/'
const RATE = 48000
const QUANTUM = 128
const QUANTA = 48
const GAINS = [0.25, 0.5, 0.75, 1]

const tone = () => {
  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  return samples
}

const handlers = async (anira, context, model, count, geometry) => {
  const contract = (
    await ContractHandle.fromUrl(
      anira,
      new URL(`${MODELS}gain.contract.json`, import.meta.url)
    )
  ).hardGeometry(geometry.quantum, geometry.burst, RATE)
  const out = []
  for (let i = 0; i < count; i++) {
    const pipeline = Pipeline.create(anira).addInference(model)
    const handler = InferenceHandler.create(context, pipeline)
    handler.prepare(contract)
    out.push(handler)
    pipeline.destroy()
  }
  contract.destroy()
  return out
}

const node = async (handler, audio, gain, waitMs) => {
  const audioNode = await handler.createAudioNode(audio, { params: [1], waitMs })
  audioNode.params.get(1).value = gain
  return audioNode
}

// Renders `nodes` (each with its gain) on `audio`, one output channel each; the worst error
// against the input times each node's gain after the latency.
const renderOffline = async (audio, nodes, gains, latency) => {
  const samples = tone()
  const buffer = audio.createBuffer(1, samples.length, RATE)
  buffer.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer })
  const merger = new ChannelMergerNode(audio, { numberOfInputs: nodes.length })
  nodes.forEach((n, i) => {
    source.connect(n.node)
    n.node.connect(merger, 0, i)
  })
  merger.connect(audio.destination)
  source.start(0)
  const rendered = await audio.startRendering()
  let worst = 0
  nodes.forEach((_, i) => {
    const channel = rendered.getChannelData(i)
    for (let j = 0; j + latency < channel.length; j++) {
      worst = Math.max(worst, Math.abs(channel[j + latency] - gains[i] * samples[j]))
    }
  })
  return worst
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogSink(() => {})
  const context = Context.create(anira, { threads: 1 })
  const model = await ModelConfig.fromUrl(
    anira,
    new URL(`${MODELS}gain.model.json`, import.meta.url)
  )
  const offlineGeometry = { quantum: QUANTUM, burst: QUANTUM }
  await context.ready()

  // Four at once on a fresh context: its first-node setup shared by all four.
  {
    const audio = new OfflineAudioContext(4, QUANTUM * QUANTA, RATE)
    const hs = await handlers(anira, context, model, 4, offlineGeometry)
    const nodes = await Promise.all(hs.map((h, i) => node(h, audio, GAINS[i], 500)))
    const worst = await renderOffline(audio, nodes, GAINS, hs[0].getLatency())
    check(
      nodes.length === 4 && worst < 1e-6,
      `four nodes created at once on a fresh context render their gains (worst error ${worst})`
    )
    nodes.forEach((n) => n.destroy())
    hs.forEach((h) => h.destroy())
  }

  // Two at once, then two more while a fifth is destroyed.
  {
    const audio = new OfflineAudioContext(4, QUANTUM * QUANTA, RATE)
    const hs = await handlers(anira, context, model, 5, offlineGeometry)
    const first = await Promise.all([
      node(hs[0], audio, GAINS[0], 500),
      node(hs[1], audio, GAINS[1], 500),
    ])
    const fifth = await node(hs[4], audio, 1, 500)
    const more = await Promise.all([
      node(hs[2], audio, GAINS[2], 500),
      (async () => fifth.destroy())(),
      node(hs[3], audio, GAINS[3], 500),
    ])
    const nodes = [...first, more[0], more[2]]
    const worst = await renderOffline(audio, nodes, GAINS, hs[0].getLatency())
    check(
      worst < 1e-6,
      `two nodes at once, then two more while a fifth is destroyed, render their gains ` +
        `(worst error ${worst})`
    )
    nodes.forEach((n) => n.destroy())
    hs.forEach((h) => h.destroy())
  }

  // Live: two at once on a fresh context, then four more, each playing (Chromium; Firefox
  // headless cannot run a live context).
  const live = new AudioContext({ sampleRate: RATE })
  await Promise.race([live.resume(), new Promise((resolve) => setTimeout(resolve, 2000))])
  if (live.state !== 'running') {
    check(true, `live: the context does not run here (${live.state}), skipped`)
  } else {
    const geometry = await measureRenderGeometry(live)
    for (const count of [2, 4]) {
      const gains = GAINS.slice(0, count)
      const hs = await handlers(anira, context, model, count, geometry)
      const nodes = await Promise.all(hs.map((h, i) => node(h, live, gains[i], 0)))
      const source = new OscillatorNode(live, { frequency: 440 })
      const analysers = nodes.map((n) => {
        const analyser = new AnalyserNode(live, { fftSize: 2048 })
        source.connect(n.node).connect(analyser)
        analyser.connect(live.destination)
        return analyser
      })
      source.start()
      await new Promise((resolve) => setTimeout(resolve, 1500))
      const rms = analysers.map((analyser) => {
        const data = new Float32Array(analyser.fftSize)
        analyser.getFloatTimeDomainData(data)
        return Math.sqrt(data.reduce((sum, x) => sum + x * x, 0) / data.length)
      })
      check(
        // A sine's rms is its amplitude over sqrt(2); a missed quantum in the window lowers it.
        rms.every((r, i) => Math.abs((r * Math.SQRT2) / gains[i] - 1) < 0.15),
        `live: ${count} nodes created at once each play their gain ` +
          `(rms ${rms.map((r) => r.toFixed(3)).join(', ')})`
      )
      source.stop()
      source.disconnect()
      nodes.forEach((n) => n.destroy())
      hs.forEach((h) => h.destroy())
    }
  }
  await live.close()

  model.destroy()
  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
