// The render geometry of an audio context: the two numbers a Hard contract's block range needs
// for a handler that streams through an audio node. `quantum`, the frames of one process() call
// (host_block_min), and `burst`, the frames the context renders back to back with no time between
// them (host_block_max): a live AudioContext renders each device callback as a run of quanta, so an
// inference that starts in one run cannot finish before the next quantum of that run, and the
// latency the handler reports must cover the run.
//
// The quantum is the context's, fixed for its life (Web Audio API, Rendering an Audio Graph). The
// burst is measured: the callback's size is never exposed (the rendering loop is called back
// for "a varying number of sample-frames"), and no attribute gives it across browsers
// (Chromium's baseLatency is its callback buffer and stays the first device's after setSinkId,
// Safari's is one quantum, Firefox's is 0). The probe reads the timing of process() calls instead:
// "A running AudioContext can be used to infer the user's audio hardware buffer size via
// AudioWorklet timing" (Privacy Considerations, on the "hardware" render size).

import type { AniraAudioNode } from './AudioNode'

/** The render geometry of a context (see `measureRenderGeometry`). */
export type RenderGeometry = {
  /** Frames per process() call: the contract's smallest block. */
  quantum: number
  /** Frames rendered back to back, a whole number of quanta: the contract's largest block. */
  burst: number
}

/** How `measureRenderGeometry` measures. */
export type RenderGeometryOptions = {
  /**
   * Running audio nodes of the context to hold while the probe measures (`AniraAudioNode.hold`:
   * silence, their handlers left alone), released after, whatever happens. A node's rendering
   * on the audio thread (a JavaScript processor that unfolds a model's input) makes calls late and
   * can hide a callback's run. Pass the nodes that stream when measuring again (`sinkchange`).
   */
  hold?: readonly AniraAudioNode[]
}

/** What `measureRenderGeometry` reads of a context. */
type GeometrySource = {
  readonly sampleRate: number
  readonly state?: string
  readonly renderQuantumSize?: number
}

const DEFAULT_QUANTUM = 128
// The first calls after the probe connects are left out (a context that just started may catch
// up in one longer run); then the window, which grows while the frames per callback cannot yet
// tell which whole number of quanta covers a callback, up to its longest.
const PROBE_SKIP_MS = 100
const PROBE_MS = 300
const PROBE_LONGEST_MS = 2000
const PROBE_STEP_MS = 100

/**
 * The frames of one process() call: `renderQuantumSize` where the browser has the attribute,
 * else 128. A context created with `renderSizeHint: "hardware"` reports 0 until it runs.
 */
const quantumOf = (context: GeometrySource): number => {
  if (context.renderQuantumSize === undefined) return DEFAULT_QUANTUM
  if (!(context.renderQuantumSize > 0)) {
    throw new Error(
      'anira: measureRenderGeometry: the context has no render quantum size yet ' +
        '(renderQuantumSize 0: a "hardware" size is chosen once it runs); ' +
        'await context.resume() first'
    )
  }
  return context.renderQuantumSize
}

// The share of the window the gaps between runs must add beyond a quantum each for the window to
// show runs: a context that renders ahead of the wall clock in runs and then waits.
const BURST_EXCESS = 0.1

/**
 * The gap at which a run of process() calls ends, from the calls' times (ms) in a window.
 *
 * A context that renders a device callback as a run of quanta shows its runs apart by the gaps
 * between callbacks, the longest gaps of the window. Inside a run the calls follow each other at
 * the graph's rendering time per quantum, and on a loaded audio thread that time grows and some
 * calls are late by milliseconds (a JavaScript processor that unfolds an inference's input in one
 * call per callback, a preempted thread): split at a fixed gap, such a run came out a quantum or
 * more short. So a run ends where a gap reaches halfway from one quantum's duration to the
 * longest gap: a call inside a run splits it only when it is late by about half the time between
 * callbacks. That holds where the window shows runs: its longest gap is two quanta or more, at
 * least two gaps reach the halfway mark, and those gaps add a tenth of the window beyond a quantum
 * each (the context renders ahead of the wall clock and then waits). Otherwise the calls are
 * evenly paced, one quantum per callback with a late call now and then, and two calls are one
 * run when they are closer than half a quantum's duration and never closer than 1.5 ms (a run's
 * calls may straddle a millisecond tick of `Date.now`); there the quanta caught up after a late
 * call are one run. The errors left are long ones: two callbacks merge where the gap between
 * them is short (the first rendered late), and evenly paced calls of a quantum shorter than 3 ms
 * merge. Short only where a late call splits every run of the window, or where the rendering
 * takes nearly a quantum's duration per quantum: then no gap is left to see.
 */
export const runGap = (timesMs: ArrayLike<number>, quantumMs: number): number => {
  let longestGap = 0
  for (let i = 1; i < timesMs.length; i++) {
    longestGap = Math.max(longestGap, timesMs[i] - timesMs[i - 1])
  }
  const halfway = (quantumMs + longestGap) / 2
  let reaching = 0
  let excess = 0
  for (let i = 1; i < timesMs.length; i++) {
    const gap = timesMs[i] - timesMs[i - 1]
    if (gap >= halfway) {
      reaching++
      excess += gap - quantumMs
    }
  }
  const windowMs = timesMs.length > 1 ? timesMs[timesMs.length - 1] - timesMs[0] : 0
  const bursts =
    longestGap >= 2 * quantumMs && reaching >= 2 && excess >= BURST_EXCESS * windowMs
  return bursts ? halfway : Math.max(quantumMs / 2, 1.5)
}

/**
 * The longest run of back-to-back process() calls in a window, in calls, from their times (ms,
 * the scope's clock: `Date.now` in an AudioWorkletGlobalScope, which has no `performance`); runs
 * end at `runGap`.
 */
export const longestRun = (timesMs: ArrayLike<number>, quantumMs: number): number => {
  const gap = runGap(timesMs, quantumMs)
  let longest = timesMs.length > 0 ? 1 : 0
  let run = 1
  for (let i = 1; i < timesMs.length; i++) {
    run = timesMs[i] - timesMs[i - 1] < gap ? run + 1 : 1
    longest = Math.max(longest, run)
  }
  return longest
}

/** The times (ms) at which a run begins: the first call, and every call after a `runGap`. */
export const runStarts = (timesMs: ArrayLike<number>, quantumMs: number): number[] => {
  const gap = runGap(timesMs, quantumMs)
  const starts: number[] = []
  for (let i = 0; i < timesMs.length; i++) {
    if (i === 0 || timesMs[i] - timesMs[i - 1] >= gap) starts.push(timesMs[i])
  }
  return starts
}

/**
 * The frames of a device callback, the standard error of that estimate and the run starts it
 * was fitted to (`framesPerCallback`).
 */
export type CallbackFrames = { frames: number; error: number; starts: number }

// The run starts an estimate of the frames per callback needs before it may settle the burst: a
// standard error from fewer is itself too uncertain (seven starts gave 2053 +- 1.8 frames for a
// callback of 2048 in headless Chromium).
const SETTLE_STARTS = 16

/**
 * The frames of one device callback from the times (ms, a microsecond clock) at which runs of
 * calls begin, and the standard error of the estimate; null with fewer than four runs.
 *
 * The callbacks come at a fixed period, the callback's frames over the rate, give or take the
 * system's jitter. Each run begins with a callback, so its start is on that grid; a callback the
 * runs merged over leaves a hole. The first start is left out (the window may begin inside a
 * run). The period is the slope of a least-squares fit of the starts against their places on
 * the grid (each start's place: its distance from the first in periods of the median gap,
 * rounded, so a hole skips a place), fitted again without the starts more than four median
 * residuals off (at least 50 µs: a start a stall delayed), and its standard error falls with the
 * window's length (headless Chromium: about 0.2 frames after 1 s, 0.07 after 2 s). Where every
 * callback renders no quanta or one (a quantum larger than the callback renders every few
 * callbacks), the runs begin every few callbacks and the estimate is the frames between them:
 * a whole quantum, a burst of one.
 */
export const framesPerCallback = (
  startsMs: ArrayLike<number>,
  rate: number
): CallbackFrames | null => {
  // The first start may be inside a run (the window begins anywhere): left out.
  const starts = Array.from(startsMs).slice(1)
  if (starts.length < 4) return null
  const gaps: number[] = []
  for (let i = 1; i < starts.length; i++) gaps.push(starts[i] - starts[i - 1])
  gaps.sort((a, b) => a - b)
  const median = gaps[Math.floor(gaps.length / 2)]
  if (!(median > 0)) return null
  const places = starts.map((t) => Math.round((t - starts[0]) / median))
  const fit = (keep: (i: number) => boolean) => {
    let n = 0
    let mx = 0
    let my = 0
    starts.forEach((t, i) => {
      if (!keep(i)) return
      n++
      mx += places[i]
      my += t
    })
    mx /= n
    my /= n
    let sxx = 0
    let sxy = 0
    starts.forEach((t, i) => {
      if (!keep(i)) return
      sxx += (places[i] - mx) ** 2
      sxy += (places[i] - mx) * (t - my)
    })
    const slope = sxy / sxx
    const residuals = starts.map((t, i) => t - my - slope * (places[i] - mx))
    let squares = 0
    starts.forEach((_, i) => {
      if (keep(i)) squares += residuals[i] ** 2
    })
    return { n, sxx, slope, residuals, se: Math.sqrt(squares / (n - 2) / sxx) }
  }
  const all = fit(() => true)
  if (!(all.sxx > 0)) return null
  // A start a stall delayed is no start on the grid: left out past four times the median
  // residual (at least 50 µs), then the fit again.
  const spread = [...all.residuals].map(Math.abs).sort((a, b) => a - b)
  const bound = Math.max(4 * spread[Math.floor(spread.length / 2)], 0.05)
  const kept = fit((i) => Math.abs(all.residuals[i]) <= bound)
  const result = kept.n >= 4 && kept.sxx > 0 ? kept : all
  return {
    frames: (result.slope * rate) / 1000,
    error: (result.se * rate) / 1000,
    starts: result.n,
  }
}

/**
 * The burst in quanta: the longest run seen, or the quanta a callback of `callback` frames needs
 * (its frames over the quantum, rounded up), whichever is more, and whether that is settled.
 *
 * A callback whose frames are no multiple of the quantum renders a varying number of quanta,
 * and the largest is the burst. It can be rarer than any window: a callback of 481 frames at a
 * quantum of 480 renders one quantum per callback, and two back to back once every 480 callbacks
 * (when the frame left over each time adds up to a quantum), so the runs alone say 480 and the
 * burst is 960. From the frames per callback it is ceil(481 / 480) = 2 quanta. An exact fit (480
 * frames at a quantum of 480, a "hardware" quantum) is one quantum: half a frame is taken off the
 * estimate before rounding up. Settled once the estimate within two standard errors rounds up to
 * one number of quanta; until then the probe watches on.
 */
export const burstQuanta = (
  longest: number,
  callback: CallbackFrames | null,
  quantum: number
): { quanta: number; settled: boolean } => {
  if (callback === null) return { quanta: longest, settled: false }
  const quanta = (frames: number) => Math.max(1, Math.ceil((frames - 0.5) / quantum))
  const margin = 3 * callback.error + 0.1
  const settled =
    callback.starts >= SETTLE_STARTS &&
    quanta(callback.frames - margin) === quanta(callback.frames + margin)
  return { quanta: Math.max(longest, quanta(callback.frames)), settled }
}

// The probe's processor: it records the time of every process() call (Date.now: for the runs)
// in a buffer it allocates once, counts the calls in the shared word 0 and notifies it (for the
// clock Worker's microsecond times), and posts its times once word 1 is set.
const PROBE_PROCESSOR = `registerProcessor('anira-render-probe', class extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const o = options.processorOptions
    this.times = new Float64Array(o.capacity)
    this.count = 0
    this.first = 0
    this.skipUntil = Date.now() + o.skipMs
    this.words = new Int32Array(o.words)
    this.done = false
  }
  process() {
    if (this.done) return false
    const now = Date.now()
    if (this.count < this.times.length) {
      if (now < this.skipUntil) this.first = this.count + 1
      this.times[this.count++] = now
    }
    Atomics.add(this.words, 0, 1)
    Atomics.notify(this.words, 0)
    if (Atomics.load(this.words, 1) !== 0) {
      this.done = true
      this.port.postMessage(this.times.slice(this.first, this.count))
      return false
    }
    return true
  }
})`

// The clock Worker: it waits on the call counter (Atomics.wait: no spinning) and records
// performance.now() at every wake with a new count, after the lead-in, posting what it has every
// step, and last with \`done\` once word 1 is set.
const CLOCK_WORKER = `onmessage = (event) => {
  const { words, skipMs, stepMs } = event.data
  const w = new Int32Array(words)
  const skipUntil = performance.now() + skipMs
  let last = Atomics.load(w, 0)
  let times = []
  let posted = performance.now()
  while (Atomics.load(w, 1) === 0) {
    Atomics.wait(w, 0, last, 20)
    const now = performance.now()
    const count = Atomics.load(w, 0)
    if (count !== last) {
      if (now >= skipUntil) times.push(now)
      last = count
    }
    if (now - posted >= stepMs) {
      postMessage({ times, done: false })
      times = []
      posted = now
    }
  }
  postMessage({ times, done: true })
  close()
}`

const probeModules = new WeakMap<BaseAudioContext, Promise<void>>()

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * The probe on a running context: the burst in frames. The worklet's calls give the runs
 * (`longestRun`); the clock Worker's microsecond times give the frames per callback
 * (`runStarts`, `framesPerCallback`), and the burst is `burstQuanta` of both. It watches 300 ms
 * after a 100 ms lead-in, and on until the burst is settled, at most 2 s. Without a Worker (a
 * page whose policy refuses one) the runs alone.
 */
const probeBurst = async (context: AudioContext, quantum: number) => {
  let loaded = probeModules.get(context)
  if (!loaded) {
    const url = URL.createObjectURL(
      new Blob([PROBE_PROCESSOR], { type: 'text/javascript' })
    )
    loaded = context.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url))
    probeModules.set(context, loaded)
  }
  await loaded
  const quantumMs = (quantum / context.sampleRate) * 1000
  const words = new SharedArrayBuffer(8)
  const stop = () => Atomics.store(new Int32Array(words), 1, 1)
  const clockTimes: number[] = []
  // The clock Worker, until it fails (a page whose policy refuses a blob Worker): then the runs
  // alone.
  let clocked = true
  let clockDone: Promise<void> = Promise.resolve()
  try {
    const url = URL.createObjectURL(new Blob([CLOCK_WORKER], { type: 'text/javascript' }))
    const worker = new Worker(url)
    clockDone = new Promise<void>((resolve) => {
      worker.onmessage = (event: MessageEvent<{ times: number[]; done: boolean }>) => {
        URL.revokeObjectURL(url)
        clockTimes.push(...event.data.times)
        if (event.data.done) resolve()
      }
      worker.onerror = () => {
        URL.revokeObjectURL(url)
        clocked = false
        resolve()
      }
    })
    worker.postMessage({ words, skipMs: PROBE_SKIP_MS, stepMs: PROBE_STEP_MS })
  } catch {
    clocked = false
  }
  const node = new AudioWorkletNode(context, 'anira-render-probe', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    processorOptions: {
      capacity: Math.ceil((PROBE_SKIP_MS + PROBE_LONGEST_MS) / quantumMs) * 4 + 64,
      skipMs: PROBE_SKIP_MS,
      words,
    },
  })
  // Pulled by the graph, heard by nobody.
  const silent = new GainNode(context, { gain: 0 })
  node.connect(silent).connect(context.destination)
  try {
    const times = new Promise<Float64Array>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('anira: the render probe saw no process() calls')),
        PROBE_SKIP_MS + PROBE_LONGEST_MS + 5000
      )
      node.port.onmessage = (event: MessageEvent<Float64Array>) => {
        clearTimeout(timer)
        resolve(event.data)
      }
    })
    await delay(PROBE_SKIP_MS + PROBE_MS)
    // The estimate that settled the burst, kept: more times may move a settled estimate back
    // across a boundary by chance.
    let settled: CallbackFrames | null = null
    for (let watched = PROBE_MS; clocked && watched < PROBE_LONGEST_MS; ) {
      const callback = framesPerCallback(
        runStarts(clockTimes, quantumMs),
        context.sampleRate
      )
      if (burstQuanta(1, callback, quantum).settled) {
        settled = callback
        break
      }
      await delay(PROBE_STEP_MS)
      watched += PROBE_STEP_MS
    }
    stop()
    const calls = await times
    if (calls.length < 2) throw new Error('anira: the render probe saw too few calls')
    // The clock's last times, if it ever started (bounded: a Worker that never loaded posts none).
    await Promise.race([clockDone, delay(1000)])
    const callback =
      settled ??
      (clocked
        ? framesPerCallback(runStarts(clockTimes, quantumMs), context.sampleRate)
        : null)
    return burstQuanta(longestRun(calls, quantumMs), callback, quantum).quanta * quantum
  } finally {
    stop()
    node.disconnect()
    silent.disconnect()
  }
}

/**
 * The render geometry of `context`, what a Hard contract for a handler streaming through an
 * audio node on it declares: `contract.hardHostBlocks(g.quantum, g.burst, context.sampleRate)`.
 *
 * `quantum` is the context's `renderQuantumSize` (Web Audio API, Render Quantum Sizes: the size
 * of every process() call, fixed for the context's life, `setSinkId` included), else 128 where
 * the browser has no such attribute. A context created with `renderSizeHint: "hardware"`
 * reports 0 until it runs: refused then.
 *
 * `burst` is measured, the worst case. An OfflineAudioContext renders quantum by quantum: one
 * quantum. On a running AudioContext a probe AudioWorklet watches process() calls: the longest
 * run of back-to-back calls (`longestRun`), and the frames per device callback, timed by a
 * Worker with a microsecond clock (`framesPerCallback`), whose quanta rounded up count where the
 * largest run is rarer than the window (481 frames at a quantum of 480: 960; `burstQuanta`). A
 * callback whose size is no multiple of the quantum renders a varying number of quanta, and the
 * largest counts (441 frames: 3 or 4 quanta of 128, a burst of 4); a quantum at least the
 * callback's size is one call per callback or fewer, a burst of one quantum. It watches 300 ms
 * after a 100 ms lead-in, and on until the frames per callback settle the burst, at most 2 s
 * (about 1 s in headless Chromium). The nodes in `hold` are held meanwhile, so the audio thread
 * is quiet; the probe's own process() notifies the clock Worker at every call (a futex wake on
 * the audio thread, while it measures). Refused on a context that is not running:
 * `await context.resume()` first. Measure again after the output device changed (`sinkchange`),
 * holding the nodes that stream.
 */
export const measureRenderGeometry = async (
  context: BaseAudioContext,
  options: RenderGeometryOptions = {}
): Promise<RenderGeometry> => {
  const source = context as unknown as GeometrySource
  const quantum = quantumOf(source)
  // An OfflineAudioContext renders quantum by quantum, as fast as it can: no callbacks to watch.
  if ('startRendering' in context) return { quantum, burst: quantum }
  if (source.state !== 'running') {
    throw new Error(
      `anira: measureRenderGeometry needs a running context (it is ${source.state}): ` +
        'await context.resume() first'
    )
  }
  const held: AniraAudioNode[] = []
  try {
    for (const node of options.hold ?? []) {
      node.hold(true)
      held.push(node)
    }
    const burst = await probeBurst(context as AudioContext, quantum)
    return { quantum, burst: Math.max(quantum, burst) }
  } finally {
    // Every node released, a failed release (a node destroyed meanwhile) reported after.
    let failure: unknown
    for (const node of held) {
      try {
        node.hold(false)
      } catch (error) {
        failure ??= error
      }
    }
    if (failure !== undefined) throw failure
  }
}
