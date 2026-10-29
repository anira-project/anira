import type { AniraBindings } from './abi/bindings'
import { ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS, anira_status } from './abi/enums'
import type { Contract } from './Contract'
import type { WithAbi } from './ModelConfig'

/** How `createAudioNode` builds the node (`anira_em_audio_node_create`). */
export type AudioNodeOptions = {
  /**
   * The Streamed input slots the node's inputs feed, input k the k-th, each with the slot's
   * channels. Default: every Streamed input slot of the model, in slot order (input 0 the first).
   * `[]`: a node without an input.
   */
  inputSlots?: number[]
  /**
   * The Streamed output slots the node's outputs play, output k the k-th, each with the slot's
   * channels. Default: every Streamed output slot of the model, in slot order.
   */
  outputSlots?: number[]
  /**
   * Leave Streamed slots of the model unserved (`ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS`);
   * without it the creation refuses a model slot that no input or output of the node serves.
   */
  partialStreams?: boolean
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
  /**
   * Re-prepare the node's handler under `contract` while the node streams: the way to change the
   * geometry of a running node (`measureRenderGeometry` again after an output-device change).
   * The render callback is held off (silence) while the handler re-prepares, then resumes; read
   * the handler's latency again afterwards. `handler.prepare` refuses while a node streams
   * through the handler (`anira_em_audio_node_prepare`).
   */
  prepare(contract: Contract): void
  /** Stop the node calling the handler and release it (`anira_em_audio_node_destroy`). */
  destroy(): void
}

// The audio contexts registered with the module, once each: the worklet scope's clock loaded,
// then emscriptenRegisterAudioObject. Memoised as the promise, set before its first await, so
// the nodes of one context created at once share one registration: two registrations of one
// context were two handles, and the second started a second worklet thread on the context's one
// AudioWorkletGlobalScope, whose creation never completed (every node after it timed out).
const contexts = new WeakMap<BaseAudioContext, Promise<number>>()

/** @internal The handle of `audioContext` in the module, registered once (exported for the tests). */
export const registeredContext = (
  abi: Pick<AniraBindings, 'module'>,
  audioContext: BaseAudioContext
): Promise<number> => {
  let context = contexts.get(audioContext)
  if (context === undefined) {
    context = (async () => {
      // The worklet scope's clock, before the module's glue reaches the scope.
      await audioContext.audioWorklet.addModule(
        new URL('./worklet-scope.ts', import.meta.url)
      )
      return abi.module.emscriptenRegisterAudioObject(audioContext)
    })()
    contexts.set(audioContext, context)
    // A failed registration is not kept: the next node tries again.
    context.catch(() => contexts.delete(audioContext))
  }
  return context
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

// The slots as a uint32_t array in the heap, 0 for an empty list unless `always` (an empty list
// that must still be a pointer).
const writeSlots = (
  abi: AniraBindings,
  slots: readonly number[],
  always = false
): number => {
  if (slots.length === 0 && !always) return 0
  const m = abi.module
  const ptr = m._malloc(Math.max(1, slots.length) * 4)
  if (ptr === 0) throw new RangeError('anira: out of memory for the node options')
  slots.forEach((slot, i) => (m.HEAPU32[(ptr >>> 2) + i] = slot))
  return ptr
}

/**
 * A Wasm Audio Worklet node that streams through `handler` (a prepared Hard handler whose block
 * range holds the context's render quantum): the render callback is C
 * (`anira_em_audio_node_create`), one input per Streamed input slot and one output per Streamed
 * output slot it serves (all of them by default), their buffers the handler's tensors, one
 * `anira_handler_process_multi` per quantum over all of them, the miss rule applied there. Works on an `AudioContext` and an `OfflineAudioContext`. Destroy the node
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
  const context = await registeredContext(abi, audioContext)
  for (const url of options.workletModules ?? []) {
    await audioContext.audioWorklet.addModule(url)
  }
  const params = options.params ?? []
  const tensors = options.staticInputs ?? []
  const words = m._malloc(8)
  if (words === 0) throw new RangeError('anira: out of memory for the node')
  // A list, even an empty one, names the slots; none (0, the NULL pointer) asks for every
  // Streamed slot of the side.
  const inputSlots = options.inputSlots ? writeSlots(abi, options.inputSlots, true) : 0
  const outputSlots = options.outputSlots ? writeSlots(abi, options.outputSlots, true) : 0
  const paramSlots = writeSlots(abi, params)
  const tensorSlots = writeSlots(abi, tensors)
  // The words stay allocated while the creation may still write them (a timeout).
  let settled = true
  try {
    abi.anira_em_audio_node_create(
      handler,
      context,
      inputSlots,
      options.inputSlots?.length ?? 0,
      outputSlots,
      options.outputSlots?.length ?? 0,
      paramSlots,
      params.length,
      tensorSlots,
      tensors.length,
      options.partialStreams ? ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS : 0,
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
      prepare(contract) {
        abi.anira_em_audio_node_prepare(handle, contract.ptr)
      },
      destroy() {
        abi.anira_em_audio_node_destroy(handle)
      },
    }
  } finally {
    if (settled) m._free(words)
    if (inputSlots) m._free(inputSlots)
    if (outputSlots) m._free(outputSlots)
    if (paramSlots) m._free(paramSlots)
    if (tensorSlots) m._free(tensorSlots)
  }
}
