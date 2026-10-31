import { type AniraBindings, heap } from './abi/bindings'
import { anira_status } from './abi/enums'
import { anira_tensor } from './abi/layout'
import { AniraError } from './AniraError'
import { createAudioNode, type AniraAudioNode, type AudioNodeOptions } from './AudioNode'
import type { Context } from './Context'
import type { ContractHandle } from './ContractHandle'
import type { Pipeline } from './Pipeline'
import { PlanReport } from './PlanReport'
import { staticInputBytes, type InputSpec } from './staticInput'

/**
 * A handler (`anira_handler`): one stream through a pipeline on a context's core, the web twin
 * of `anira::InferenceHandler` of `anira.hpp`, member for member (camelCase for snake_case).
 * Prepare it with a contract (a `PlanReport` back), then stream through it with an audio node
 * (`createAudioNode`). Where the web differs: the Hard process, push and pop forms are the audio
 * node's (`createAudioNode`: the C render callback calls them on the audio thread), a failed call
 * throws an `AniraError` instead of returning its status, and the handler is destroyed with
 * `destroy()` (no RAII): its audio nodes first, then the handler.
 */
export class InferenceHandler {
  readonly abi: AniraBindings
  private readonly handler: number
  /** The model's input specs (the pipeline's variant), which `setStaticInput` writes in. */
  private readonly inputs: readonly InputSpec[]

  private constructor(abi: AniraBindings, handler: number, inputs: readonly InputSpec[]) {
    this.abi = abi
    this.handler = handler
    this.inputs = inputs
  }

  /** `anira_handler_create`: copies the pipeline, which can be destroyed right after. */
  static create(context: Context, pipeline: Pipeline): InferenceHandler {
    return new InferenceHandler(
      context.abi,
      context.abi.anira_handler_create(context.native(), pipeline.native()),
      pipeline.inputSpecs()
    )
  }

  /**
   * `anira_handler_prepare`: copies the contract, which can be destroyed right after, and
   * returns the plan report of this prepare (`planReport()`). Refused while an audio node streams
   * through the handler: re-prepare through the node (`audioNode.prepare(contract)`).
   */
  prepare(contract: ContractHandle): PlanReport {
    this.abi.anira_handler_prepare(this.handler, contract.native())
    return this.planReport()
  }

  /** The report of the last prepare (`anira_handler_plan_report`): a view until the next one. */
  planReport(): PlanReport {
    return new PlanReport(this.abi, this.abi.anira_handler_plan_report(this.handler))
  }

  /** The entries a prepared handler holds in flight at once (`anira_handler_num_entries`). */
  numEntries(): number {
    return this.abi.anira_handler_num_entries(this.handler)
  }

  /** The latency of output `slot` in samples (`anira_handler_get_latency`). */
  getLatency(slot = 0): number {
    return this.abi.anira_handler_get_latency(this.handler, slot)
  }

  /**
   * The latency of every output slot in samples, in slot order, 0 for one that is not Streamed
   * (`anira_handler_get_latencies`).
   */
  getLatencies(): number[] {
    const m = this.abi.module
    const count = m._malloc(4)
    if (count === 0) throw new RangeError('anira: out of memory for the latencies')
    try {
      heap(m).HEAPU32[count >>> 2] = 0
      this.abi.anira_handler_get_latencies(this.handler, count, 0)
      const n = heap(m).HEAPU32[count >>> 2]
      if (n === 0) return []
      const out = m._malloc(n * 4)
      if (out === 0) throw new RangeError('anira: out of memory for the latencies')
      try {
        this.abi.anira_handler_get_latencies(this.handler, count, out)
        return Array.from(heap(m).HEAPU32.subarray(out >>> 2, (out >>> 2) + n))
      } finally {
        m._free(out)
      }
    } finally {
      m._free(count)
    }
  }

  /**
   * Write a Static input whole from the main thread (`anira_handler_set_static_input`), in the
   * spec's dtype and extents: a typed array of the spec's element type, any `ArrayBufferView` of
   * exactly the spec's bytes, or plain numbers in the spec's order, converted to its element
   * type. For a handler no audio node drives (an audio node serves its Static inputs itself: its
   * AudioParams and its own `setStaticInput`).
   */
  setStaticInput(slot: number, values: ArrayBufferView | ArrayLike<number>): void {
    const spec = this.inputs[slot]
    if (spec === undefined) {
      throw new AniraError(
        anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
        `anira: setStaticInput: no input slot ${slot}`
      )
    }
    const bytes = staticInputBytes(spec, values)
    const m = this.abi.module
    const shapeBytes = spec.shape.length * 8
    const block = m._malloc(bytes.length + 8 + shapeBytes + anira_tensor.size)
    if (block === 0) throw new RangeError('anira: out of memory for a Static input')
    try {
      // The data at the block's start (malloc's alignment covers every element size), then
      // the shape (int64) and the record.
      const data = block
      heap(m).HEAPU8.set(bytes, data)
      const shape = (data + bytes.length + 7) & ~7
      spec.shape.forEach(
        (extent, i) => (heap(m).HEAP64[(shape >>> 3) + i] = BigInt(extent))
      )
      const tensor = shape + shapeBytes
      this.abi.anira_tensor_init_host(tensor, data, spec.dtype, spec.shape.length, shape)
      this.abi.anira_handler_set_static_input(this.handler, slot, tensor)
    } finally {
      m._free(block)
    }
  }

  /**
   * The last real-time failure recorded on the handler, `ANIRA_OK` for none since the last
   * prepare or reset (`anira_handler_rt_error`): returned, as the C++ member returns it, never
   * thrown.
   */
  rtError(): number {
    return this.abi.module._anira_handler_rt_error(this.handler)
  }

  /** Select the plan the next chunk runs on, a dense index of the report (`anira_handler_set_plan`). */
  setPlan(plan: number): void {
    this.abi.anira_handler_set_plan(this.handler, plan)
  }

  /** The selected plan (`anira_handler_get_plan`). */
  getPlan(): number {
    return this.abi.anira_handler_get_plan(this.handler)
  }

  /**
   * A Wasm Audio Worklet node that streams through this handler (`createAudioNode`). Prepare the
   * handler with the geometry `measureRenderGeometry` measures on the context first: a handler
   * whose `host_block_max` is below the context's burst misses quanta live.
   */
  createAudioNode(
    audioContext: BaseAudioContext,
    options: AudioNodeOptions = {}
  ): Promise<AniraAudioNode> {
    return createAudioNode(this, audioContext, this.handler, this.inputs, options)
  }

  /** The `anira_handler*` the raw entries take. */
  native(): number {
    return this.handler
  }

  destroy(): void {
    this.abi.anira_handler_destroy(this.handler)
  }
}
