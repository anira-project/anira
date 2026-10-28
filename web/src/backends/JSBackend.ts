import { anira_status } from '../abi/enums'
import { anira_engine_ctx, anira_tensor } from '../abi/layout'
import type { AniraWasmInstance } from '../runtime/createAniraWasm'
import { Heap } from '../runtime/Heap'
import type { JSEngine } from '../runtime/hooks'

/** What a JavaScript engine is constructed with, in the Worker that runs it. */
export type JSBackendEnv = {
  module: AniraWasmInstance
  heap: Heap
  /** Its reverse-URI id (`anira_em_pipeline_register_js_engine`). */
  engineId: string
  /** The model config and entry its `init()` reads (main-owned, read-only here). */
  modelConfigPtr: number
  modelIndex: number
}

/** A model tensor of one engine call: its descriptor and its float32 data. */
export type EngineTensor = {
  ptr: number
  dtype: number
  /** The element offset of the data in `heap.f32`; 0 for a tensor that is not host float32. */
  f32Index: number
  numElements: number
}

/**
 * The minimal base of a JavaScript engine over `Module.anira.engine`. A subclass loads what it
 * needs in `init()` (the Worker awaits it before it reports the engine registered, which the
 * host waits for before `anira_handler_prepare`) and fills the outputs in `process()`; the
 * tensors of a call come from the `anira_engine_ctx` through `layout.ts`.
 */
export class JSBackend implements JSEngine {
  protected readonly env: JSBackendEnv

  constructor(env: JSBackendEnv) {
    this.env = env
  }

  async init(): Promise<void> {
    // Override in a subclass to load a model.
  }

  process(_ctx: number, _prepared: number): number {
    return anira_status.ANIRA_ERROR_NOT_SUPPORTED
  }

  destroy(): void {
    // Override in a subclass to release what init() built.
  }

  protected inputs(ctx: number): EngineTensor[] {
    const f = anira_engine_ctx.fields
    return this.tensors(
      this.env.heap.readPtr(ctx + f.inputs.offset),
      this.env.heap.readU32(ctx + f.num_inputs.offset)
    )
  }

  protected outputs(ctx: number): EngineTensor[] {
    const f = anira_engine_ctx.fields
    return this.tensors(
      this.env.heap.readPtr(ctx + f.outputs.offset),
      this.env.heap.readU32(ctx + f.num_outputs.offset)
    )
  }

  private tensors(array: number, count: number): EngineTensor[] {
    const { heap, module } = this.env
    const out: EngineTensor[] = []
    for (let i = 0; i < count; i++) {
      const ptr = array + i * anira_tensor.size
      const data = Heap.unsigned(module._anira_tensor_data_f32(ptr))
      out.push({
        ptr,
        dtype: heap.readU32(ptr + anira_tensor.fields.dtype.offset),
        f32Index: data >>> 2,
        numElements:
          data === 0 ? 0 : Heap.unsigned(module._anira_tensor_num_elements(ptr)),
      })
    }
    return out
  }
}
