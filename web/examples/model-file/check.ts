// The example's own check, run headless by web/test/browser/run.mjs on the built example (which
// serves the repository's extras/models at /extras/models/): the bundled gain config rendered
// offline on the built-in ONNX Runtime engine, with the gain automated mid-render. The model
// applies the gain of the block it runs on, so each output block is its input block times the
// gain of that quantum, the handler's latency later (the dry path muted for the exact check), and
// the dry path's delay is that latency. The verdict is window.anira_test_result.
import { createModelFileExample } from './main'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const CHANGE_QUANTUM = 32
const GAIN_BEFORE = 0.5
const GAIN_AFTER = 0.125

type Check = { ok: boolean; what: string }
type Result = { checks: Check[] } | { error: string }

const run = async (): Promise<Result> => {
  const checks: Check[] = []
  const check = (ok: boolean, what: string) => checks.push({ ok, what })
  check(crossOriginIsolated, 'the page is cross-origin isolated (COOP/COEP)')

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const example = await createModelFileExample(audio, {
    modelsUrl: new URL('/extras/models/model-pool/', location.href),
    gain: GAIN_BEFORE,
    waitMs: 500,
    dryLevel: 0,
  })
  example.gain.setValueAtTime(GAIN_BEFORE, 0)
  example.gain.setValueAtTime(GAIN_AFTER, (CHANGE_QUANTUM * QUANTUM) / RATE)
  const samples = new Float32Array(QUANTUM * QUANTA)
  for (let i = 0; i < samples.length; i++)
    samples[i] = 0.25 * Math.sin((2 * Math.PI * 440 * i) / RATE)
  // Kept in a buffer of its own: Firefox detaches an AudioBuffer's channel arrays once a
  // source node plays it.
  const input = audio.createBuffer(1, samples.length, RATE)
  input.copyToChannel(samples, 0)
  const source = new AudioBufferSourceNode(audio, { buffer: input })
  source.connect(example.input)
  example.output.connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  const latency = example.latency()

  let worst = 0
  for (let j = 0; j + latency < rendered.length; j++) {
    const gain = Math.floor(j / QUANTUM) < CHANGE_QUANTUM ? GAIN_BEFORE : GAIN_AFTER
    worst = Math.max(worst, Math.abs(rendered[j + latency] - gain * samples[j]))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the node rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times the model's Static gain after ${latency} samples, the gain ` +
      `automated mid-render from quantum ${CHANGE_QUANTUM} (worst error ${worst})`
  )

  const delay = example.dry.delayTime.value
  check(
    Math.abs(delay - latency / RATE) < 1e-7,
    `the dry path is delayed by the handler's latency (${delay} s = ${latency} / ${RATE})`
  )

  // A new geometry on the node (as after a device change): the latency changes, and so does the
  // dry path's delay. Four quanta, doubled until past the latency: a Hard handler's latency
  // covers block_max, so it changes (a block_max within it may leave it as it was).
  let blockMax = 4 * QUANTUM
  while (blockMax <= latency) blockMax *= 2
  example.applyGeometry({ quantum: QUANTUM, burst: blockMax })
  const changed = example.latency()
  check(
    changed !== latency,
    `the node re-prepared at block_max ${blockMax}: latency ${latency} -> ${changed}`
  )
  check(
    Math.abs(example.dry.delayTime.value - changed / RATE) < 1e-7,
    `the dry path follows the new latency (${changed} / ${RATE} s)`
  )

  example.teardown()
  check(true, 'the example tore down (the node, the handler, the context)')
  return { checks }
}

declare global {
  interface Window {
    anira_test_result?: Result
  }
}

run()
  .then((result) => (window.anira_test_result = result))
  .catch((error: unknown) => {
    window.anira_test_result = {
      error: error instanceof Error ? `${error}\n${error.stack ?? ''}` : String(error),
    }
  })
