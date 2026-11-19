// What variants.html and variant-set.html share: the page's half of the gate that steps an
// offline render chunk by chunk (variants-worklet.js has the audio thread's), the model configs
// of the variant demo models with a JavaScript engine's entry beside the built-in one, and the
// check of a rendered stream chunk by chunk against the plan each chunk was formed on.
import {
  ANIRA_EM_PROCESSOR_HOOK_PRE,
  ANIRA_EM_PROCESSOR_HOOK_PREPARE,
  ANIRA_EM_PROCESSOR_HOOK_UNPREPARE,
  AniraError,
  ModelConfig,
  anira_status,
  defineProcessor,
  heap,
} from '../../dist/index.js'

export const RATE = 48000
export const QUANTUM = 128
export const MODELS = new URL('../../extras/models/model-pool/', import.meta.url)
export const WORKER_MODULE = new URL('./variants-worker.js', import.meta.url)
export const WEIGHT_ENGINE = 'org.anira-test.weight'
export const ACCUMULATOR_ENGINE = 'org.anira-test.accumulator'

// ---- the gate ---------------------------------------------------------------------------------

export const GATE = 'org.anira-test.gate'
export const GATE_PHASES =
  ANIRA_EM_PROCESSOR_HOOK_PRE |
  ANIRA_EM_PROCESSOR_HOOK_PREPARE |
  ANIRA_EM_PROCESSOR_HOOK_UNPREPARE
export const GATE_WORKLET = new URL('./variants-worklet.js', import.meta.url)

const OPEN = 0x7fffffff
// The gate's block of every prepare of a handler with the gate, newest last, and how often the
// processor's prepare ran: a variant added to a playing handler prepares no processor.
const blocks = []
let prepares = 0
export const gatePrepares = () => prepares

// The main thread's half: prepare makes the block the audio thread's pre_process reads (four
// int32: the chunks that may pass, the chunk held, the chunks formed, the pacing) and
// hands it out as this handler's `prepared` pointer; unprepare frees it.
defineProcessor(GATE, ({ module }) => ({
  prepare(_prepareInfo, outPrepared) {
    const block = module._malloc(16)
    if (block === 0) return anira_status.ANIRA_ERROR_OUT_OF_MEMORY
    heap(module).HEAP32.set([OPEN, -1, 0, 0], block >>> 2)
    heap(module).HEAPU32[outPrepared >>> 2] = block
    blocks.push(block)
    prepares += 1
    return anira_status.ANIRA_OK
  },
  unprepare(prepared) {
    module._free(prepared)
  },
}))

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * The gate of the handler prepared last (take it right behind the prepare). A chunk is counted
 * from the handler's prepare on, across the renders through it. `allow(n)` lets the chunks
 * below n pass and holds chunk n inside its pre_process; `heldAt(n)` waits until the audio
 * thread is there: the render callback is then inside the handler's entry and chunk n has its
 * plan, so what the page selects now runs from chunk n + 1 on. `open` lets every chunk pass,
 * paced at one per so many milliseconds when asked; `holdNext` holds a render that runs open
 * at the next chunk it can and says which.
 */
export const gateOfLastPrepare = (anira) => {
  const base = blocks[blocks.length - 1] >>> 2
  const words = () => heap(anira.module).HEAP32
  const until = async (done, what, timeoutMs) => {
    const deadline = performance.now() + timeoutMs
    for (;;) {
      const result = done()
      if (result !== undefined) return result
      if (performance.now() > deadline) {
        throw new Error(
          `${what} within ${timeoutMs} ms (held at ${Atomics.load(words(), base + 1)}, ` +
            `${Atomics.load(words(), base + 2)} chunks formed)`
        )
      }
      await delay(0)
    }
  }
  return {
    allow: (chunks) => Atomics.store(words(), base, chunks),
    open: (millisecondsPerChunk = 0) => {
      Atomics.store(words(), base + 3, millisecondsPerChunk)
      Atomics.store(words(), base, OPEN)
    },
    formed: () => Atomics.load(words(), base + 2),
    heldAt: (chunk, timeoutMs = 20000) =>
      until(
        () => (Atomics.load(words(), base + 1) === chunk ? chunk : undefined),
        `the render did not reach chunk ${chunk}`,
        timeoutMs
      ),
    holdNext: (timeoutMs = 20000) => {
      // The chunk in its pre_process now may be past the gate already: the first one that
      // can still be held is the one behind it, or a later one when the render ran on.
      const from = Atomics.load(words(), base + 2) + 1
      Atomics.store(words(), base, from)
      return until(
        () => {
          const held = Atomics.load(words(), base + 1)
          return held >= from ? held : undefined
        },
        'the render was not held',
        timeoutMs
      )
    },
  }
}

// ---- models -----------------------------------------------------------------------------------

/** The status a call throws with, or 'returned' when it does not throw. */
export const refusal = (call) => {
  try {
    call()
    return 'returned'
  } catch (error) {
    return error instanceof AniraError ? error.status : `${error}`
  }
}

// Every file is fetched once: a page makes the same model config many times over.
const fetched = new Map()
const fetchOnce = async (url) => {
  const key = `${url}`
  if (!fetched.has(key)) {
    const response = await fetch(url)
    fetched.set(key, {
      ok: response.ok,
      status: response.status,
      bytes: new Uint8Array(await response.arrayBuffer()),
    })
  }
  const { ok, status, bytes } = fetched.get(key)
  return {
    ok,
    status,
    arrayBuffer: async () => bytes.slice().buffer,
    text: async () => new TextDecoder().decode(bytes),
  }
}

/**
 * A bundled model config with one entry more, for a JavaScript engine of variants-worker.js,
 * whose bytes begin with the float32 values `floats` (what that engine computes with) and go on
 * with `ballast` zero bytes (what a model of that size weighs in the heap while it is loaded).
 * `edit` may change the parsed file first.
 */
export const modelWithEngine = async (
  anira,
  file,
  engine,
  floats,
  { edit = (json) => json, ballast = 0 } = {}
) => {
  const json = edit(JSON.parse(await (await fetchOnce(new URL(file, MODELS))).text()))
  json.models.push({ engine })
  const model = await ModelConfig.fromJson(anira, JSON.stringify(json), {
    baseUrl: MODELS,
    fetch: fetchOnce,
  })
  const bytes = new Uint8Array(4 * floats.length + ballast)
  bytes.set(new Uint8Array(new Float32Array(floats).buffer))
  return model.setModelBytes(json.models.length - 1, bytes)
}

// ---- streams ----------------------------------------------------------------------------------

/**
 * Samples that are exact in float32, and stay exact under every factor and every running sum
 * the pages use: odd multiples of 1/512, positive ones up to 1/2 with `positive`, else between
 * -1/2 and 1/2. `step` tells two channels apart.
 */
export const ramp = (frames, { positive = false, step = 37 } = {}) => {
  const samples = new Float32Array(frames)
  for (let i = 0; i < frames; i++) {
    const k = (i * step) % 255
    samples[i] = positive ? (2 * k + 1) / 512 : (2 * k - 255) / 512
  }
  return samples
}

/**
 * Renders `samples` (one Float32Array per channel) through `handler`'s audio node on an
 * OfflineAudioContext, every quantum waiting for its inference; `during()` runs once the
 * render has started and steers it through `gate`, which is opened behind it whatever it did,
 * so the render runs to its end. Returns the rendered channels.
 */
export const renderThrough = async (handler, gate, samples, during = async () => {}) => {
  const frames = samples[0].length
  const audio = new OfflineAudioContext(samples.length, frames, RATE)
  const audioNode = await handler.createAudioNode(audio, {
    waitMs: 2000,
    workletModules: [GATE_WORKLET],
  })
  const buffer = audio.createBuffer(samples.length, frames, RATE)
  samples.forEach((channel, c) => buffer.copyToChannel(channel, c))
  const source = new AudioBufferSourceNode(audio, { buffer })
  source.connect(audioNode.node).connect(audio.destination)
  source.start(0)
  const rendering = audio.startRendering()
  let rendered
  try {
    await during()
  } finally {
    gate.open()
    rendered = await rendering
    audioNode.destroy()
  }
  return samples.map((_, c) => rendered.getChannelData(c))
}

/**
 * The chunks of a rendered channel that are not what `expected(chunk, i)` says for every sample
 * i of the chunk (the input's index; the output is `latency` samples behind), and how many
 * chunks lie inside the render.
 */
export const wrongChunks = (rendered, latency, hop, expected) => {
  const wrong = []
  let chunks = 0
  for (; (chunks + 1) * hop + latency <= rendered.length; chunks++) {
    for (let i = chunks * hop; i < (chunks + 1) * hop; i++) {
      if (rendered[i + latency] !== Math.fround(expected(chunks, i))) {
        wrong.push(chunks)
        break
      }
    }
  }
  return { wrong, chunks }
}
