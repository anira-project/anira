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

// The times (ms, floored to Date.now's resolution) of the calls of a context whose device calls
// back every `periodMs` and renders `counts[k % counts.length]` quanta in callback k, each quantum
// taking `renderMs`; every `lateEvery`-th call (counted over the window) starts `lateMs` late,
// and the calls after it follow (a callback starts once the previous one's calls are done).
const loaded = (
  counts: number[],
  periodMs: number,
  renderMs: number,
  callbacks: number,
  lateEvery = 0,
  lateMs = 0
) => {
  const times: number[] = []
  let call = 0
  let t = 0
  for (let k = 0; k < callbacks; k++) {
    t = Math.max(t, 1000 + k * periodMs)
    for (let c = 0; c < counts[k % counts.length]; c++) {
      call++
      if (lateEvery > 0 && call % lateEvery === 0) t += lateMs
      times.push(Math.floor(t))
      t += renderMs
    }
  }
  return times
}

test('a loaded audio thread: the runs stay whole, never shorter than the callback', () => {
  const quantumMs = (128 / RATE) * 1000
  const periodMs = 16 * quantumMs
  // The graph's rendering per quantum from light to most of a quantum (split at a fixed 1.5 ms,
  // Chromium's runs under such loads came out 4 to 15 quanta of 16).
  for (const renderMs of [0.05, 0.5, 1.2, 2.0]) {
    assert.equal(
      longestRun(loaded([16], periodMs, renderMs, 40), quantumMs),
      16,
      `${renderMs}`
    )
  }
  // Late once per callback, at a fixed place in it (a stage's pre_process): the run stays whole.
  for (const late of [3, 6, 12]) {
    const times = loaded([16], periodMs, 0.5, 40, 16, late)
    assert.equal(longestRun(times, quantumMs), 16, `late ${late} ms every 16th call`)
  }
  // Late at places that move from callback to callback, so the callbacks render for different
  // times and the gaps between them differ: never short, and where a gap is short two callbacks
  // may merge.
  for (const [every, late] of [
    [7, 3],
    [5, 6],
    [11, 12],
  ]) {
    const times = loaded([16], periodMs, 0.5, 40, every, late)
    assert.ok(longestRun(times, quantumMs) >= 16, `late ${late} ms every ${every}th call`)
  }
  // Most of a callback's period rendering (1.5 ms per quantum and a 4 ms late call every 5th):
  // the gaps between callbacks shrink; never short.
  const heavy = loaded([16], periodMs, 1.5, 40, 5, 4)
  assert.ok(longestRun(heavy, quantumMs) >= 16)
  // A thread that cannot keep up (2.3 ms per quantum and the late calls) leaves no gap between
  // callbacks: its calls look evenly paced, the limit the rule cannot see past.
  assert.ok(longestRun(loaded([16], periodMs, 2.3, 40, 5, 6), quantumMs) < 16)
})

test('callbacks of a size that is no multiple of the quantum: the largest run', () => {
  const quantumMs = (128 / RATE) * 1000
  // 480 frames (10 ms) per callback: 4, 4, 4, 3 quanta; 441 frames: 3 or 4 quanta (CI).
  assert.equal(longestRun(loaded([4, 4, 4, 3], 10, 0.05, 40), quantumMs), 4)
  assert.equal(longestRun(loaded([4, 3, 4, 3, 3], 9.1875, 0.3, 40), quantumMs), 4)
})

test('an evenly paced context late once: one quantum, or the quanta it caught up', () => {
  const quantumMs = (128 / RATE) * 1000
  assert.equal(longestRun(loaded([1], quantumMs, 0.05, 200), quantumMs), 1)
  // 8 ms late: the next three quanta are due already and follow back to back.
  assert.equal(longestRun(loaded([1], quantumMs, 0.05, 200, 97, 8), quantumMs), 4)
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
