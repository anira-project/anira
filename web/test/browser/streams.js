// A model with two Streamed inputs and two Streamed outputs (a and b) through one audio node: the node has an input per Streamed input slot and an output per Streamed output slot,
// each with the slot's channels, and the C render callback calls anira_handler_process_multi
// once per quantum over all of them. streams-worker.js defines the engines: a_out = 2 * a_in,
// b_out = 3 * b_in. First an OfflineAudioContext render that waits for every inference, a mono
// and b stereo: every output exact after the handler's latency. Then both stereo (BYPASS needs
// the outputs' channels to be the anchored input's), an engine slower than the render and no
// waits, under ANIRA_MISS_BYPASS: a missed block passes the anchored input (a, slot 0) through
// to both outputs, as the multi form promises, never zeros on one of them (which is what
// two nodes on one handler, one slot each, would give). The verdict is window.anira_test_result.
import {
  ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
  Anira,
  Context,
  ContractHandle,
  InferenceHandler,
  ModelConfig,
  Pipeline,
  anira_miss_policy,
} from '../../dist/index.js'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const FRAMES = QUANTUM * QUANTA

const streamed = (name, channels) => ({
  name,
  dtype: 'float32',
  role: 'streamed',
  axes: [
    ['batch', 1],
    ['channel', channels],
    ['time', QUANTUM],
  ],
  window: { min: QUANTUM, max: QUANTUM },
})

const tone = (frequency, amplitude) => {
  const samples = new Float32Array(FRAMES)
  for (let i = 0; i < FRAMES; i++) {
    samples[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / RATE)
  }
  return samples
}

const A = [tone(440, 0.25), tone(220, 0.1)]
const B = [tone(330, 0.2), tone(550, 0.15)]

// Renders the channels of a_out, then of b_out, of one node over `engine`.
const render = async (
  anira,
  context,
  { engine, waitMs, aChannels, bChannels, onMiss }
) => {
  const model = await ModelConfig.fromJson(
    anira,
    JSON.stringify({
      models: [{ engine }],
      inputs: [streamed('a_in', aChannels), streamed('b_in', bChannels)],
      outputs: [streamed('a_out', aChannels), streamed('b_out', bChannels)],
    })
  )
  const pipeline = Pipeline.create(anira)
    .inference(model)
    .registerEngine(engine, { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })
  const contract = ContractHandle.hard(anira, {
    blockMin: QUANTUM,
    rate: RATE,
    budgetMs: 2,
  })
  anira.abi.anira_contract_hard_set_on_miss(contract.native(), onMiss)
  const handler = InferenceHandler.create(context, pipeline)
  handler.prepare(contract)
  pipeline.destroy()
  contract.destroy()
  const latencies = [handler.getLatency(0), handler.getLatency(1)]
  await context.ready()

  const total = aChannels + bChannels
  const audio = new OfflineAudioContext(total, FRAMES, RATE)
  // A node that would leave the Streamed slot b_in unserved is refused, unless it says so.
  let refusal = ''
  try {
    ;(await handler.createAudioNode(audio, { inputSlots: [0] })).destroy()
  } catch (error) {
    refusal = String(error.message)
  }
  const partial = await handler.createAudioNode(audio, {
    inputSlots: [0],
    partialStreams: true,
  })
  partial.destroy()
  const audioNode = await handler.createAudioNode(audio, { waitMs })
  const node = audioNode.node
  const source = (tones, channels, input) => {
    const buffer = audio.createBuffer(channels, FRAMES, RATE)
    for (let c = 0; c < channels; c++) buffer.copyToChannel(tones[c], c)
    const node_ = new AudioBufferSourceNode(audio, { buffer })
    node_.connect(node, 0, input)
    return node_
  }
  const sources = [source(A, aChannels, 0), source(B, bChannels, 1)]
  const merger = new ChannelMergerNode(audio, { numberOfInputs: total })
  const split = (output, channels, first) => {
    const splitter = new ChannelSplitterNode(audio, { numberOfOutputs: channels })
    node.connect(splitter, output, 0)
    for (let c = 0; c < channels; c++) splitter.connect(merger, c, first + c)
  }
  split(0, aChannels, 0)
  split(1, bChannels, aChannels)
  merger.connect(audio.destination)
  for (const s_ of sources) s_.start(0)
  const rendered = await audio.startRendering()
  const all = [...Array(total).keys()].map((c) => rendered.getChannelData(c).slice())
  const channels = { a: all.slice(0, aChannels), b: all.slice(aChannels) }
  const shape = { inputs: node.numberOfInputs, outputs: node.numberOfOutputs, refusal }
  audioNode.destroy()
  handler.destroy()
  model.destroy()
  return { channels, latencies, shape }
}

const blockEquals = (x, y, q) => {
  for (let i = q * QUANTUM; i < (q + 1) * QUANTUM; i++) if (x[i] !== y[i]) return false
  return true
}
const blockZero = (x, q) => {
  for (let i = q * QUANTUM; i < (q + 1) * QUANTUM; i++) if (x[i] !== 0) return false
  return true
}

const run = async () => {
  const checks = []
  const check = (ok, what) => checks.push({ ok: Boolean(ok), what })
  const anira = await Anira.create()
  anira.setLogSink(() => {}) // the missed blocks of the second render warn; not shown
  const context = Context.create(anira, {
    threads: 1,
    workerModule: new URL('./streams-worker.js', import.meta.url),
  })

  // 1. Waiting for every inference, a mono and b stereo: exact.
  const exact = await render(anira, context, {
    engine: 'org.anira-test.two-streams',
    waitMs: 500,
    aChannels: 1,
    bChannels: 2,
    onMiss: anira_miss_policy.ANIRA_MISS_ZEROS,
  })
  check(
    exact.shape.inputs === 2 && exact.shape.outputs === 2,
    `one node, an input and an output per Streamed slot (${exact.shape.inputs} and ` +
      `${exact.shape.outputs})`
  )
  check(
    exact.shape.refusal.includes('no input of the node serves the Streamed input slot 1'),
    `a node leaving a Streamed slot unserved is refused (${exact.shape.refusal}); with ` +
      `partialStreams it is created`
  )
  const [L] = exact.latencies
  check(
    exact.latencies[1] === L,
    `both outputs report one latency (${exact.latencies.join(', ')} samples)`
  )
  let worst = 0
  for (let j = 0; j + L < FRAMES; j++) {
    worst = Math.max(
      worst,
      Math.abs(exact.channels.a[0][j + L] - 2 * A[0][j]),
      Math.abs(exact.channels.b[0][j + L] - 3 * B[0][j]),
      Math.abs(exact.channels.b[1][j + L] - 3 * B[1][j])
    )
  }
  const energy = [...exact.channels.a, ...exact.channels.b].map((c) =>
    c.reduce((sum, x) => sum + x * x, 0)
  )
  check(
    energy.every((e) => e > 1),
    `every channel rendered audio (energy ${energy.map((e) => e.toFixed(1)).join(', ')})`
  )
  check(
    worst < 1e-6,
    `a_out = 2 * a_in (mono) and b_out = 3 * b_in (stereo) after ${L} samples, one ` +
      `process_multi per quantum (worst error ${worst})`
  )

  // 2. Both stereo, a slower engine, no waits, ANIRA_MISS_BYPASS: a missed block passes the
  //    anchored input a through to both outputs.
  const missed = await render(anira, context, {
    engine: 'org.anira-test.two-streams-slow',
    waitMs: 0,
    aChannels: 2,
    bChannels: 2,
    onMiss: anira_miss_policy.ANIRA_MISS_BYPASS,
  })
  let bypassed = 0
  let notOnB = 0
  for (let q = 0; q < QUANTA; q++) {
    const aBypassed = [0, 1].every((c) => blockEquals(missed.channels.a[c], A[c], q))
    if (!aBypassed || blockZero(A[0], q)) continue
    bypassed += 1
    if (![0, 1].every((c) => blockEquals(missed.channels.b[c], A[c], q))) notOnB += 1
  }
  check(bypassed > 0, `the slow engine missed blocks: ${bypassed} of ${QUANTA} bypassed`)
  check(
    notOnB === 0,
    `every bypassed block passes the anchored input a through to both outputs ` +
      `(${notOnB} blocks where b_out did not)`
  )

  context.destroy()
  return { checks }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch(
    (error) => (window.anira_test_result = { error: `${error}\n${error?.stack ?? ''}` })
  )
