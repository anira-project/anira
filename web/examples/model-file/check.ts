// The example's own check, run headless by web/test/browser/run.mjs on the built example (which
// serves the repository's extras/models at /extras/models/): the bundled gain config rendered
// offline on the built-in ONNX Runtime engine, with the gain automated mid-render. The model
// applies the gain of the block it runs on, so each output block is its input block times the
// gain of that quantum, the handler's latency later. The verdict is window.anira_test_result.
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
  source.connect(example.audioNode.node).connect(audio.destination)
  source.start(0)
  const rendered = (await audio.startRendering()).getChannelData(0)
  const { latency } = example

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
