// The render geometry of an audio context: the two numbers a Hard contract's block range needs
// for a handler that streams through an audio node. `quantum`, the frames of one process() call
// (block_min), and `burst`, the frames the context renders back to back with no time between
// them (block_max): a live AudioContext renders each device callback as a run of quanta, so an
// inference that starts in one run cannot finish before the next quantum of that run, and the
// latency the handler reports must cover the run.

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
   * Measure with the probe even where `baseLatency` gives the burst (Chromium); for tests, and
   * after an output-device change, since Chromium keeps the `baseLatency` of the device the
   * context was created on. Default false.
   */
  probe?: boolean
  /** How long the probe watches process() calls. Default 300 ms. */
  probeMs?: number
}

/** What `measureRenderGeometry` reads of a context. */
type GeometrySource = {
  readonly sampleRate: number
  readonly state?: string
  readonly baseLatency?: number
  readonly renderQuantumSize?: number
}

const DEFAULT_QUANTUM = 128
const PROBE_MS = 300
// The first calls after the probe connects are left out: a context that just started may catch
// up in one longer run.
const PROBE_SKIP_MS = 100

/** The frames of one process() call: `renderQuantumSize` where the browser has it, else 128. */
const quantumOf = (context: GeometrySource): number =>
  context.renderQuantumSize !== undefined && context.renderQuantumSize > 0
    ? context.renderQuantumSize
    : DEFAULT_QUANTUM

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
// once, and posts the times after the window.
const PROBE_PROCESSOR = `registerProcessor('anira-render-probe', class extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.times = new Float64Array(options.processorOptions.capacity)
    this.count = 0
    this.skipUntil = Date.now() + options.processorOptions.skipMs
    this.until = this.skipUntil + options.processorOptions.windowMs
    this.done = false
  }
  process() {
    if (this.done) return false
    const now = Date.now()
    if (now >= this.skipUntil && this.count < this.times.length) this.times[this.count++] = now
    if (now >= this.until || this.count === this.times.length) {
      this.done = true
      this.port.postMessage(this.times.slice(0, this.count))
      return false
    }
    return true
  }
})`

const probeModules = new WeakMap<BaseAudioContext, Promise<void>>()

/** The probe on a running context: the longest run of back-to-back calls it saw, in frames. */
const probeBurst = async (context: AudioContext, quantum: number, windowMs: number) => {
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
  const node = new AudioWorkletNode(context, 'anira-render-probe', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    processorOptions: {
      capacity: Math.ceil((PROBE_SKIP_MS + windowMs) / quantumMs) * 4 + 64,
      skipMs: PROBE_SKIP_MS,
      windowMs,
    },
  })
  // Pulled by the graph, heard by nobody.
  const silent = new GainNode(context, { gain: 0 })
  node.connect(silent).connect(context.destination)
  try {
    const times = await new Promise<Float64Array>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('anira: the render probe saw no process() calls')),
        windowMs + PROBE_SKIP_MS + 5000
      )
      node.port.onmessage = (event: MessageEvent<Float64Array>) => {
        clearTimeout(timer)
        resolve(event.data)
      }
    })
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
 * `quantum` is the context's `renderQuantumSize` where the browser has it (the Web Audio 1.1
 * render size: `renderSizeHint` asks for one, between 1 frame and 6 times the sample rate, and
 * "hardware" lets the browser pick the device buffer), else 128.
 *
 * `burst`: an OfflineAudioContext renders quantum by quantum, so one quantum. A live context
 * whose `baseLatency` reports its callback buffer (Chromium: the larger of the destination's
 * frames per buffer and the render quantum, over the rate, fixed at construction) gives it at
 * once, rounded up to whole quanta; a context created with renderSizeHint "hardware" has a burst
 * of one quantum there. Where `baseLatency` is 0 (Gecko: no buffering between the graph and the
 * audio subsystem by its definition, though its graph also renders a callback as several quanta)
 * a probe AudioWorklet on the context watches process() calls for about 300 ms (after a 100 ms
 * lead-in it leaves out) and takes the longest run of back-to-back calls (`longestRun`). The
 * probe needs a running context: call `await context.resume()` first. Measure again after the
 * output device changed.
 */
export const measureRenderGeometry = async (
  context: BaseAudioContext,
  options: RenderGeometryOptions = {}
): Promise<RenderGeometry> => {
  const source = context as unknown as GeometrySource
  const quantum = quantumOf(source)
  // An OfflineAudioContext has no baseLatency: quantum by quantum.
  if (source.baseLatency === undefined) return { quantum, burst: quantum }
  const reported = options.probe ? null : reportedBurst(context)
  if (reported !== null) return { quantum, burst: reported }
  if (source.state !== 'running') {
    throw new Error(
      `anira: measureRenderGeometry needs a running context (it is ${source.state}): ` +
        'await context.resume() first'
    )
  }
  const burst = await probeBurst(
    context as AudioContext,
    quantum,
    options.probeMs ?? PROBE_MS
  )
  return { quantum, burst: Math.max(quantum, burst) }
}

/**
 * The burst a context reports without measuring (Chromium's `baseLatency`), or null: what
 * `Handler.createAudioNode` checks the prepared contract against.
 */
export const reportedBurst = (context: BaseAudioContext): number | null => {
  const source = context as unknown as GeometrySource
  const quantum = quantumOf(source)
  if (source.baseLatency === undefined) return quantum
  if (source.baseLatency <= 0) return null
  return (
    Math.max(1, Math.ceil((source.baseLatency * source.sampleRate) / quantum - 1e-9)) *
    quantum
  )
}
