import type { AniraBindings } from './abi/bindings'
import { ANIRA_DTYPE_F32 } from './abi/enums'
import { anira_tensor } from './abi/layout'
import { createAudioNode, type AniraAudioNode, type AudioNodeOptions } from './AudioNode'
import { reportedBurst } from './renderGeometry'
import type { Context } from './Context'
import type { Contract } from './Contract'
import type { Pipeline } from './Pipeline'

/**
 * A handler (`anira_handler`): one stream through a pipeline on a context's core. Prepare it with
 * a contract, then stream through it with an audio node (`createAudioNode`). Destroy its audio
 * nodes first, then the handler.
 */
export class Handler {
  readonly abi: AniraBindings
  /** The `anira_handler*` the raw entries take. */
  readonly ptr: number
  /** The `block_max` of the Hard contract of the last `prepare`; undefined before one. */
  blockMax: number | undefined

  private constructor(abi: AniraBindings, ptr: number) {
    this.abi = abi
    this.ptr = ptr
  }

  /** `anira_handler_create`: copies the pipeline, which can be destroyed right after. */
  static create(context: Context, pipeline: Pipeline): Handler {
    return new Handler(
      context.abi,
      context.abi.anira_handler_create(context.ptr, pipeline.ptr)
    )
  }

  /**
   * `anira_handler_prepare`: copies the contract, which can be destroyed right after. Refused
   * while an audio node streams through the handler: re-prepare through the node
   * (`audioNode.prepare(contract)`).
   */
  prepare(contract: Contract): this {
    this.abi.anira_handler_prepare(this.ptr, contract.ptr)
    try {
      this.blockMax = this.abi.anira_contract_hard_geometry(contract.ptr).block_max
    } catch {
      this.blockMax = undefined // not a Hard contract: no audio node streams through it
    }
    return this
  }

  /** The latency of output `slot` in samples (`anira_handler_get_latency`). */
  latency(slot = 0): number {
    return this.abi.anira_handler_get_latency(this.ptr, slot)
  }

  /**
   * Set a float32 Static input from the main thread (`anira_handler_set_static_input`), its
   * elements in the spec's order; for a handler no audio node drives (an audio node serves its
   * Static inputs itself: its AudioParams and `setStaticInput`).
   */
  setStaticInput(slot: number, values: ArrayLike<number>): void {
    const m = this.abi.module
    const data = m._malloc(values.length * 4 + 8 + anira_tensor.size)
    if (data === 0) throw new RangeError('anira: out of memory for a Static input')
    try {
      m.HEAPF32.set(Float32Array.from(values), data >>> 2)
      const shape = data + values.length * 4
      const aligned = (shape + 7) & ~7
      m.HEAP64[aligned >>> 3] = BigInt(values.length)
      const tensor = aligned + 8
      this.abi.anira_tensor_init_host(tensor, data, ANIRA_DTYPE_F32, 1, aligned)
      this.abi.anira_handler_set_static_input(this.ptr, slot, tensor)
    } finally {
      m._free(data)
    }
  }

  /**
   * A Wasm Audio Worklet node that streams through this handler (`createAudioNode`). Warns when
   * the prepared `block_max` is below the burst the context reports (Chromium's `baseLatency`;
   * `measureRenderGeometry` measures it everywhere): such a handler misses quanta live.
   */
  async createAudioNode(
    audioContext: BaseAudioContext,
    options: AudioNodeOptions = {}
  ): Promise<AniraAudioNode> {
    const burst = reportedBurst(audioContext)
    if (burst !== null && this.blockMax !== undefined && this.blockMax < burst) {
      console.warn(
        `anira: the handler is prepared for blocks up to ${this.blockMax} frames, but this ` +
          `context renders bursts of ${burst} frames: inferences cannot complete within a ` +
          `burst and quanta will miss. Prepare with the geometry of measureRenderGeometry().`
      )
    }
    const node = await createAudioNode(this, audioContext, this.ptr, options)
    const prepare = node.prepare.bind(node)
    // The node's re-prepare is this handler's prepare: keep the block range it was given.
    node.prepare = (contract: Contract) => {
      prepare(contract)
      this.blockMax = this.abi.anira_contract_hard_geometry(contract.ptr).block_max
    }
    return node
  }

  destroy(): void {
    this.abi.anira_handler_destroy(this.ptr)
  }
}
