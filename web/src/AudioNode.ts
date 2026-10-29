import type { AniraBindings } from './abi/bindings'
import { anira_status } from './abi/enums'
import type { WithAbi } from './ModelConfig'

/** How `createAudioNode` builds the node (`anira_em_audio_node_create`). */
export type AudioNodeOptions = {
  /** The node's input channels, the input tensor's; 0 for a node without an input. Default 1. */
  inputChannels?: number
  /** The node's output channels, the output tensor's. Default 1. */
  outputChannels?: number
  /** The Streamed input slot the node's input feeds. Default 0. */
  inputSlot?: number
  /** The Streamed output slot the node plays. Default 0. */
  outputSlot?: number
  /**
   * Static float32 scalar inputs driven by AudioParams, by slot: `audioNode.params.get(slot)` is
   * the k-rate AudioParam (automatable, default 0) whose value the handler gets each quantum it
   * changed.
   */
  params?: number[]
  /** Static inputs of any shape the main thread writes whole with `setStaticInput`, by slot. */
  staticInputs?: number[]
  /**
   * 0 (the default): every quantum processes without waiting (a live context, misses take the
   * miss policy); above 0, each quantum waits up to this many milliseconds for its inference
   * (an offline render).
   */
  waitMs?: number
  /**
   * Modules the context's `audioWorklet` loads first (`addModule`): the worklet-side code of the
   * handler's JavaScript stage, which calls `defineStage` from `@anira-project/anira/scope`.
   */
  workletModules?: (string | URL)[]
  /** How long to wait for the node (the worklet thread starts with the first one). */
  timeoutMs?: number
}

/** A node of `createAudioNode`: the `AudioWorkletNode` and what drives its Static inputs. */
export type AniraAudioNode = {
  /** The Web Audio node: connect it into the graph. */
  readonly node: AudioWorkletNode
  /** The node's handle in the module (`anira_em_audio_node_*`). */
  readonly handle: number
  /** The AudioParam of each `params` slot. */
  readonly params: ReadonlyMap<number, AudioParam>
  /**
   * Write one of the `staticInputs` whole: a typed array or `DataView` of exactly the spec's
   * bytes, or plain numbers as float32. The render callback pushes it before its next process
   * call, never torn (`anira_em_audio_node_set_static_input`).
   */
  setStaticInput(slot: number, values: ArrayBufferView | ArrayLike<number>): void
  /** Stop the node calling the handler and release it (`anira_em_audio_node_destroy`). */
  destroy(): void
}

/** What `renderBurstFrames` reads of a context: a live `AudioContext` has the two latencies. */
export type BurstSource = {
  readonly sampleRate: number
  readonly baseLatency?: number
  readonly outputLatency?: number
  readonly renderQuantumSize?: number
}

/** The burst assumed of a live context that reports no buffer at all (2048 frames at 48 kHz). */
const UNKNOWN_BURST_SECONDS = 2048 / 48000

/**
 * The frames a context renders back to back, as one burst of render quanta with no time between
 * them: what a handler's Hard `block_max` must cover so that an inference started in one burst
 * may complete by the next (`Contract.hard({ blockMax })`, `setGeometry(128, blockMax, rate)`;
 * the latency the handler then reports covers the burst). A live `AudioContext` renders one
 * device callback per burst; an `OfflineAudioContext` renders quantum by quantum (the render
 * quantum, 128).
 *
 * The Web Audio spec defines `baseLatency` as the latency of "passing the audio from the
 * AudioDestinationNode to the audio subsystem". Chromium reports the callback buffer there
 * (third_party/blink/renderer/modules/webaudio/audio_context.cc: `base_latency_` is the larger
 * of the destination's frames per buffer and the render quantum, over the rate), which is the
 * burst. Gecko returns 0 (dom/media/webaudio/AudioContext.h: "Gecko does not do any buffering
 * between rendering the audio and sending it to the audio subsystem") although its graph also
 * fills each audio callback with several quanta; there `outputLatency` stands in for it (the
 * audio stream's output latency, `Graph()->AudioOutputLatency()`, taken to cover the callback
 * buffer: an assumption, conservative since it also counts the device's own latency). A context
 * that reports neither (Gecko's before it runs) is assumed to burst 2048 frames at 48 kHz.
 * The result is a whole number of render quanta, at least one.
 */
export const renderBurstFrames = (context: BurstSource): number => {
  const quantum = context.renderQuantumSize ?? 128
  // An OfflineAudioContext has no baseLatency.
  if (context.baseLatency === undefined) return quantum
  const base = context.baseLatency
  const output = context.outputLatency ?? 0
  const seconds = base > 0 ? base : output > 0 ? output : UNKNOWN_BURST_SECONDS
  return Math.max(1, Math.ceil((seconds * context.sampleRate) / quantum - 1e-9)) * quantum
}

// The audio contexts registered with the module (emscriptenRegisterAudioObject), once each.
const contexts = new WeakMap<BaseAudioContext, number>()

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const writeSlots = (abi: AniraBindings, slots: readonly number[]): number => {
  if (slots.length === 0) return 0
  const m = abi.module
  const ptr = m._malloc(slots.length * 4)
  if (ptr === 0) throw new RangeError('anira: out of memory for the node options')
  slots.forEach((slot, i) => (m.HEAPU32[(ptr >>> 2) + i] = slot))
  return ptr
}

/**
 * A Wasm Audio Worklet node that streams through `handler` (a prepared Hard handler whose block
 * range holds the context's render quantum): the render callback is C
 * (`anira_em_audio_node_create`), the node's buffers are the handler's tensors, the miss rule is
 * applied there. Works on an `AudioContext` and an `OfflineAudioContext`. Destroy the node
 * before the handler.
 */
export const createAudioNode = async (
  anira: WithAbi,
  audioContext: BaseAudioContext,
  handler: number,
  options: AudioNodeOptions = {}
): Promise<AniraAudioNode> => {
  const { abi } = anira
  const m = abi.module
  let context = contexts.get(audioContext)
  if (context === undefined) {
    // The worklet scope's clock, before the module's glue reaches the scope.
    await audioContext.audioWorklet.addModule(
      new URL('./worklet-scope.ts', import.meta.url)
    )
    context = m.emscriptenRegisterAudioObject(audioContext)
    contexts.set(audioContext, context)
  }
  for (const url of options.workletModules ?? []) {
    await audioContext.audioWorklet.addModule(url)
  }
  const params = options.params ?? []
  const tensors = options.staticInputs ?? []
  const words = m._malloc(8)
  if (words === 0) throw new RangeError('anira: out of memory for the node')
  const paramSlots = writeSlots(abi, params)
  const tensorSlots = writeSlots(abi, tensors)
  // The words stay allocated while the creation may still write them (a timeout).
  let settled = true
  try {
    abi.anira_em_audio_node_create(
      handler,
      context,
      options.inputChannels ?? 1,
      options.outputChannels ?? 1,
      options.inputSlot ?? 0,
      options.outputSlot ?? 0,
      paramSlots,
      params.length,
      tensorSlots,
      tensors.length,
      options.waitMs ?? 0,
      words,
      words + 4
    )
    settled = false
    // The creation completes on this thread's event loop: the worklet thread, the processor.
    const deadline = performance.now() + (options.timeoutMs ?? 10000)
    let status = Atomics.load(m.HEAP32, words >>> 2)
    while (status === anira_status.ANIRA_INCOMPLETE) {
      if (performance.now() > deadline) {
        throw new Error('anira: the audio node was not created in time')
      }
      await delay(1)
      status = Atomics.load(m.HEAP32, words >>> 2)
    }
    settled = true
    if (status < 0) {
      throw new Error(`anira: the audio node could not be created (status ${status})`)
    }
    const handle = m.HEAPU32[(words + 4) >>> 2]
    const node = m.emscriptenGetAudioObject(handle) as AudioWorkletNode
    const paramMap = new Map<number, AudioParam>()
    params.forEach((slot, i) => {
      const param = node.parameters.get(String(i))
      if (param) paramMap.set(slot, param)
    })
    return {
      node,
      handle,
      params: paramMap,
      setStaticInput(slot, values) {
        const bytes = ArrayBuffer.isView(values)
          ? new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
          : new Uint8Array(Float32Array.from(values).buffer)
        const ptr = m._malloc(bytes.length)
        if (ptr === 0) throw new RangeError('anira: out of memory for a Static input')
        try {
          m.HEAPU8.set(bytes, ptr)
          abi.anira_em_audio_node_set_static_input(handle, slot, ptr, bytes.length)
        } finally {
          m._free(ptr)
        }
      },
      destroy() {
        abi.anira_em_audio_node_destroy(handle)
      },
    }
  } finally {
    if (settled) m._free(words)
    if (paramSlots) m._free(paramSlots)
    if (tensorSlots) m._free(tensorSlots)
  }
}
