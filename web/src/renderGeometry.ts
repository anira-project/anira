// The render geometry of an audio context: the two numbers a Hard contract's block range needs
// for a handler that streams through an audio node. `quantum`, the frames of one process() call
// (block_min), and `burst`, the frames the context renders back to back with no time between
// them (block_max): a live AudioContext renders each device callback as a run of quanta, so an
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
   * on the audio thread (a JavaScript stage that unfolds a model's input) makes calls late and
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
// The first calls after the probe connects are left out of the runs (a context that just
// started may catch up in one longer run), then the runs are taken from this window.
const PROBE_SKIP_MS = 100
const PROBE_MS = 300

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

/**
 * Whether the calls at the end of `timesMs` (ms) are evenly paced, one quantum per callback:
 * their last 4 gaps or more, spanning 50 ms or more, each about a quantum's duration (at least
 * half of it and 1.5 ms, less than one and a half). Back-to-back calls, and the gap after a run,
 * break the pacing. The probe stops on it with a burst of one quantum: a context that renders
 * one quantum per callback shows it within a few callbacks, a quantum larger than the callback
 * included (one call every few callbacks, evenly). Only the first `count` times are read.
 * Wrong only where the calls inside one run are paced like quanta for 50 ms (a run of 20 quanta
 * or more on an audio thread rendering each in about a quantum's duration). Self-contained: the
 * probe's processor runs its source.
 */
export const evenlyPaced = (
  timesMs: ArrayLike<number>,
  quantumMs: number,
  count = timesMs.length
): boolean => {
  const low = Math.max(quantumMs / 2, 1.5)
  const high = 1.5 * quantumMs
  const last = count - 1
  for (let i = last; i > 0; i--) {
    const gap = timesMs[i] - timesMs[i - 1]
    if (gap < low || gap >= high) return false
    if (last - i + 1 >= 4 && timesMs[last] - timesMs[i - 1] >= 50) return true
  }
  return false
}

// The share of the window the gaps between runs must add beyond a quantum each for the window to
// show runs: a context that renders ahead of the wall clock in runs and then waits.
const BURST_EXCESS = 0.1

/**
 * The longest run of process() calls with no wall time between them, in calls, from their
 * timestamps (ms, the scope's clock: `Date.now` resolution in an AudioWorkletGlobalScope, which
 * has no `performance`).
 *
 * A context that renders a device callback as a run of quanta shows its runs apart by the gaps
 * between callbacks, the longest gaps of the window. Inside a run the calls follow each other at
 * the graph's rendering time per quantum, and on a loaded audio thread that time grows and some
 * calls are late by milliseconds (a JavaScript stage that unfolds an inference's input in one
 * call per callback, a preempted thread): split at a fixed gap, such a run came out a quantum or
 * more short. So a run ends where a gap reaches halfway from one quantum's duration to the
 * longest gap: a call inside a run splits it only when it is late by about half the time between
 * callbacks. That holds where the window shows runs: its longest gap is two quanta or more, at
 * least two gaps reach the halfway mark, and those gaps add a tenth of the window beyond a quantum
 * each (the context renders ahead of the wall clock and then waits). Otherwise the calls are
 * evenly paced, one quantum per callback with a late call now and then, and two calls are one
 * run when they are closer than half a quantum's duration and never closer than 1.5 ms (a run's
 * calls may straddle a millisecond tick); there the quanta caught up after a late call are one
 * run. The errors left are long ones: two callbacks merge where the gap between them is short
 * (the first rendered late), and evenly paced calls of a quantum shorter than 3 ms merge. Short
 * only where a late call splits every run of the window, or where the rendering takes nearly a
 * quantum's duration per quantum: then no gap is left to see and the calls look evenly paced.
 */
export const longestRun = (timesMs: ArrayLike<number>, quantumMs: number): number => {
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
  const gap = bursts ? halfway : Math.max(quantumMs / 2, 1.5)
  let longest = timesMs.length > 0 ? 1 : 0
  let run = 1
  for (let i = 1; i < timesMs.length; i++) {
    run = timesMs[i] - timesMs[i - 1] < gap ? run + 1 : 1
    longest = Math.max(longest, run)
  }
  return longest
}

// The probe's processor: it records the time of every process() call in a buffer it allocates
// once, and posts the times after the lead-in and the window, or `{ paced: true }` as soon as the
// calls are evenly paced (`evenlyPaced`, whose source it runs).
const probeProcessor = () => `const evenlyPaced = ${evenlyPaced.toString()}
registerProcessor('anira-render-probe', class extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const o = options.processorOptions
    this.times = new Float64Array(o.capacity)
    this.count = 0
    this.first = 0
    this.quantumMs = o.quantumMs
    this.skipUntil = Date.now() + o.skipMs
    this.until = this.skipUntil + o.windowMs
    this.done = false
  }
  process() {
    if (this.done) return false
    const now = Date.now()
    if (this.count < this.times.length) {
      if (now < this.skipUntil) this.first = this.count + 1
      this.times[this.count++] = now
    }
    if (evenlyPaced(this.times, this.quantumMs, this.count)) {
      this.done = true
      this.port.postMessage({ paced: true })
      return false
    }
    if (now >= this.until || this.count === this.times.length) {
      this.done = true
      this.port.postMessage({ times: this.times.slice(this.first, this.count) })
      return false
    }
    return true
  }
})`

const probeModules = new WeakMap<BaseAudioContext, Promise<void>>()

/** The probe on a running context: the longest run of back-to-back calls it saw, in frames. */
const probeBurst = async (context: AudioContext, quantum: number) => {
  let loaded = probeModules.get(context)
  if (!loaded) {
    const url = URL.createObjectURL(
      new Blob([probeProcessor()], { type: 'text/javascript' })
    )
    loaded = context.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url))
    probeModules.set(context, loaded)
  }
  await loaded
  const quantumMs = (quantum / context.sampleRate) * 1000
  const node = new AudioWorkletNode(context, 'anira-render-probe', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    processorOptions: {
      capacity: Math.ceil((PROBE_SKIP_MS + PROBE_MS) / quantumMs) * 4 + 64,
      quantumMs,
      skipMs: PROBE_SKIP_MS,
      windowMs: PROBE_MS,
    },
  })
  // Pulled by the graph, heard by nobody.
  const silent = new GainNode(context, { gain: 0 })
  node.connect(silent).connect(context.destination)
  try {
    const result = await new Promise<{ paced?: boolean; times?: Float64Array }>(
      (resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error('anira: the render probe saw no process() calls')),
          PROBE_SKIP_MS + PROBE_MS + 5000
        )
        node.port.onmessage = (event) => {
          clearTimeout(timer)
          resolve(event.data)
        }
      }
    )
    if (result.paced) return quantum
    const times = result.times ?? new Float64Array(0)
    if (times.length < 2) throw new Error('anira: the render probe saw too few calls')
    return longestRun(times, quantumMs) * quantum
  } finally {
    node.disconnect()
    silent.disconnect()
  }
}

/**
 * The render geometry of `context`, what a Hard contract for a handler streaming through an
 * audio node on it declares: `contract.setGeometry(g.quantum, g.burst, context.sampleRate)`.
 *
 * `quantum` is the context's `renderQuantumSize` (Web Audio API, Render Quantum Sizes: the size
 * of every process() call, fixed for the context's life, `setSinkId` included), else 128 where
 * the browser has no such attribute. A context created with `renderSizeHint: "hardware"`
 * reports 0 until it runs: refused then.
 *
 * `burst` is measured. An OfflineAudioContext renders quantum by quantum: one quantum. On a
 * running AudioContext a probe AudioWorklet watches process() calls and takes the longest run
 * of back-to-back calls (`longestRun`), in whole quanta: a callback whose size is no multiple
 * of the quantum renders a varying number of quanta, and the largest counts (441 frames: 3 or 4
 * quanta of 128, a burst of 4). It stops after about 50 ms with one quantum where the calls are
 * evenly paced (`evenlyPaced`: one quantum per callback, a quantum at least the callback's
 * size), else it watches 300 ms after a 100 ms lead-in. The nodes in `hold` are held
 * meanwhile, so the audio thread is quiet. Refused on a context that is not running:
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
