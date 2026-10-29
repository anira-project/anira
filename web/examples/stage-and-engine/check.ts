// The example's own check, run headless by web/test/browser/run.mjs on the built example (and
// under Vite's dev server): the example rendered offline, with the gain automated mid-render.
// Each input block is scaled by the stage (a ramp from the previous block's gain), doubled by the
// engine, and delivered the handler's latency later. The verdict is window.anira_test_result.
import { createGainExample } from './main'

const RATE = 48000
const QUANTUM = 128
const QUANTA = 64
const CHANGE_QUANTUM = 32
const GAIN_BEFORE = 1
const GAIN_AFTER = 0.25

type Check = { ok: boolean; what: string }
type Result = { checks: Check[] } | { error: string }

const run = async (): Promise<Result> => {
  const checks: Check[] = []
  const check = (ok: boolean, what: string) => checks.push({ ok, what })
  check(crossOriginIsolated, 'the page is cross-origin isolated (COOP/COEP)')

  const audio = new OfflineAudioContext(1, QUANTUM * QUANTA, RATE)
  const example = await createGainExample(audio, { gain: GAIN_BEFORE, waitMs: 500 })
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

  // The gain of sample i of input block `block`: the stage ramps over the block the new gain
  // arrives in.
  const gainAt = (block: number, i: number) => {
    if (block < CHANGE_QUANTUM) return GAIN_BEFORE
    if (block > CHANGE_QUANTUM) return GAIN_AFTER
    return GAIN_BEFORE + ((GAIN_AFTER - GAIN_BEFORE) * (i + 1)) / QUANTUM
  }
  let worst = 0
  for (let j = 0; j + latency < rendered.length; j++) {
    const block = Math.floor(j / QUANTUM)
    const expected = 2 * gainAt(block, j % QUANTUM) * samples[j]
    worst = Math.max(worst, Math.abs(rendered[j + latency] - expected))
  }
  const energy = rendered.reduce((sum, x) => sum + x * x, 0)
  check(energy > 1, `the node rendered audio (energy ${energy.toFixed(2)})`)
  check(
    worst < 1e-6,
    `the output is the input times 2 * the stage's gain after ${latency} samples, the gain ` +
      `automated mid-render ramped in over quantum ${CHANGE_QUANTUM} (worst error ${worst})`
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
