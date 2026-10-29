// The render geometry (src/renderGeometry.ts): the run detector of the probe on synthetic
// timestamps, and what measureRenderGeometry reads of a context without measuring.
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { longestRun, measureRenderGeometry } from '../src/renderGeometry'

const RATE = 48000

// The times (ms, at Date.now's resolution) of the process() calls of a context that renders
// `calls` quanta per device callback, back to back, for `callbacks` callbacks. `jitter` moves
// each callback's start by up to that many ms, as a loaded audio thread does.
const schedule = (quantum: number, calls: number, callbacks: number, jitter = 0) => {
  const quantumMs = (quantum / RATE) * 1000
  const times: number[] = []
  let seed = 7
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * jitter
  for (let k = 0; k < callbacks; k++) {
    const start = 1000 + k * calls * quantumMs + random()
    // The calls of one callback 20 microseconds apart: rendering is fast.
    for (let c = 0; c < calls; c++) times.push(Math.floor(start + c * 0.02))
  }
  return { times, quantumMs }
}

test('the longest run of back-to-back calls is the burst: 16, 4 and 1 quanta', () => {
  for (const calls of [16, 4, 1]) {
    const { times, quantumMs } = schedule(128, calls, 40)
    assert.equal(longestRun(times, quantumMs), calls, `${calls} calls per callback`)
  }
})

test('jitter of the callbacks does not merge or split the runs', () => {
  const { times, quantumMs } = schedule(128, 16, 40, 5)
  assert.equal(longestRun(times, quantumMs), 16)
  const even = schedule(256, 1, 200, 1)
  assert.equal(longestRun(even.times, even.quantumMs), 1)
})

test('a hardware-sized quantum (2048 frames) renders one call per callback', () => {
  const { times, quantumMs } = schedule(2048, 1, 20, 2)
  assert.equal(longestRun(times, quantumMs), 1)
})

test('a run straddling a millisecond tick stays one run', () => {
  // 16 calls whose times cross from 1000 to 1001 ms in the middle, then the next callback.
  const times = [
    ...Array<number>(8).fill(1000),
    ...Array<number>(8).fill(1001),
    1043,
    1043,
  ]
  assert.equal(longestRun(times, (128 / RATE) * 1000), 16)
})

test('measureRenderGeometry reads what it can without measuring', async () => {
  const offline = { sampleRate: RATE } as unknown as BaseAudioContext
  assert.deepEqual(await measureRenderGeometry(offline), { quantum: 128, burst: 128 })
  const chromium = {
    sampleRate: RATE,
    state: 'running',
    baseLatency: 2048 / RATE,
  } as unknown as BaseAudioContext
  assert.deepEqual(await measureRenderGeometry(chromium), { quantum: 128, burst: 2048 })
  // renderQuantumSize honoured; a burst reported in frames rounds up to whole quanta.
  const sized = {
    sampleRate: 44100,
    state: 'running',
    baseLatency: 0.01,
    renderQuantumSize: 256,
  } as unknown as BaseAudioContext
  assert.deepEqual(await measureRenderGeometry(sized), { quantum: 256, burst: 512 })
  // renderSizeHint "hardware": the quantum is the device buffer, the burst one quantum.
  const hardware = {
    sampleRate: RATE,
    state: 'running',
    baseLatency: 2048 / RATE,
    renderQuantumSize: 2048,
  } as unknown as BaseAudioContext
  assert.deepEqual(await measureRenderGeometry(hardware), { quantum: 2048, burst: 2048 })
})

test('measuring needs a running context', async () => {
  const gecko = { sampleRate: RATE, state: 'suspended', baseLatency: 0 }
  await assert.rejects(
    measureRenderGeometry(gecko as unknown as BaseAudioContext),
    /needs a running context/
  )
})
