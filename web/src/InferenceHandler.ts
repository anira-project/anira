import { type AniraBindings, heap } from './abi/bindings'
import { anira_status } from './abi/enums'
import { anira_tensor } from './abi/layout'
import { AniraError } from './AniraError'
import { createAudioNode, type AniraAudioNode, type AudioNodeOptions } from './AudioNode'
import type { Context } from './Context'
import type { ContractHandle } from './ContractHandle'
import type { ModelConfig } from './ModelConfig'
import type { Pipeline } from './Pipeline'
import {
  BACKEND_ID_SIZE,
  PlanReport,
  writeBackendId,
  type PlanQueryBackend,
} from './PlanReport'
import { staticInputBytes, type InputSpec } from './staticInput'

/**
 * A handler (`anira_handler`): one stream through a pipeline on a context's core, the web twin
 * of `anira::InferenceHandler` of `anira.hpp`, member for member (camelCase for snake_case).
 * Prepare it with a contract (a `PlanReport` back), then stream through it with an audio node
 * (`createAudioNode`). Where the web differs: the Hard process, push and pop forms are the audio
 * node's (`createAudioNode`: the C render callback calls them on the audio thread), a failed call
 * throws an `AniraError` instead of returning its status, and the handler is destroyed with
 * `destroy()` (no RAII): its audio nodes first, then the handler. `prepare`, `addVariant`,
 * `removeVariant` and `destroy` are the handler's control entries: each runs on the main thread
 * and blocks it until it is done. `addVariant` and `removeVariant` are made to run while an
 * audio node streams through the handler (the worklet's render callback is the driver thread);
 * `prepare` is refused then.
 */
export class InferenceHandler {
  readonly abi: AniraBindings
  private readonly handler: number
  /** The model's input specs (the same in every variant), which `setStaticInput` writes in. */
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

  /**
   * The newest report (`anira_handler_plan_report`): the one of the last prepare, or the one a
   * later `addVariant` or `removeVariant` published; a view until the next prepare. A
   * `PlanReport` taken earlier keeps the rows it had: read this again behind either call.
   */
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

  /**
   * Select the plan the next chunk runs on, a dense index of the report
   * (`anira_handler_set_plan`). The row of a removed variant (`ANIRA_PLAN_REMOVED`) throws an
   * `AniraError` with `ANIRA_ERROR_NOT_FOUND`, the selection as it was.
   */
  setPlan(plan: number): void {
    this.abi.anira_handler_set_plan(this.handler, plan)
  }

  /** The selected plan (`anira_handler_get_plan`). */
  getPlan(): number {
    return this.abi.anira_handler_get_plan(this.handler)
  }

  /**
   * Select the model the next chunk runs, on the backend that runs now
   * (`anira_handler_set_model`, as `anira::InferenceHandler::set_model`): the plan of `variant`
   * (its position in the list `Pipeline.addInference` took) on the selected plan's engine and
   * provider. One axis of the selection moves, the other stays. When the variant has no plan on
   * that backend, or was removed, the selection stays as it was and the call throws an
   * `AniraError` with `ANIRA_ERROR_NOT_FOUND` (where C++ returns the status). What the switch
   * does to the model's state is the contract's (`HardContractOptions.variantState`).
   */
  setModel(variant: number): void {
    this.abi.anira_handler_set_model(this.handler, variant)
  }

  /**
   * Select the backend the next chunk runs on, for the model that runs now
   * (`anira_handler_set_backend`, as `anira::InferenceHandler::set_backend`): the first plan of
   * the selected plan's variant with every property `backend` names, an engine and a provider,
   * each exact or left out, as `PlanReport.find` reads them
   * (`handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME })`). When the variant
   * has no plan there the selection stays as it was and the call throws an `AniraError` with
   * `ANIRA_ERROR_NOT_FOUND`; a backend against the pair rule (`ANIRA_ENGINE_CUSTOM` without
   * `engineId`, an id beside a built-in value) throws one with `ANIRA_ERROR_INVALID_ARGUMENT`.
   * A backend switch inside one variant keeps the model's state.
   */
  setBackend(backend: PlanQueryBackend): void {
    const m = this.abi.module
    const sp = m.stackSave()
    try {
      const engineId =
        backend.engineId === undefined ? 0 : m.stringToUTF8OnStack(backend.engineId)
      const providerId =
        backend.providerId === undefined ? 0 : m.stringToUTF8OnStack(backend.providerId)
      const record = m.stackAlloc(BACKEND_ID_SIZE)
      writeBackendId(heap(m).HEAPU32, record, backend, engineId, providerId)
      this.abi.anira_handler_set_backend(this.handler, record)
    } finally {
      m.stackRestore(sp)
    }
  }

  /**
   * Add a model of the stage's form to the handler (`anira_handler_add_variant`, as
   * `anira::InferenceHandler::add_variant`) and return its variant index (`PlanInfo.variant`,
   * what `setModel` takes): the next one of the inference stage, which no variant of the
   * handler had before. A control entry like `prepare`: it runs on the main thread and blocks
   * it. Unlike `prepare` it is legal while an audio node streams through the handler, and the
   * stream keeps playing.
   *
   * On an unprepared handler the variant joins the handler's copy of the pipeline, and the next
   * prepare builds it. On a prepared handler the model is validated under the contract of the
   * last prepare, and its plans are built and measured beside the stream, in a session of their
   * own: the models load, and the contract's warm-up pass runs for every new plan without a
   * stored figure, on the pool's Workers, which it shares with the stream meanwhile (a
   * JavaScript engine's `load` and `process` run there for the new model between the calls for
   * the playing one). The call blocks the main thread for all of it, as long as a prepare does
   * for those plans; a pass needs the Workers in their loops (`await context.ready()` first, as
   * before a prepare). It then appends the plans behind the existing ones and publishes a new
   * report: read `planReport()` again (a `PlanReport` taken earlier keeps the rows it had). A
   * plan keeps its index between two prepares; a prepare numbers the plans anew and keeps the
   * variants' indices. The selection, the latency (`getLatency`) and `numEntries()` do not
   * change: `setModel` or `setPlan` selects the new variant, and what the switch does to the
   * model's state is the contract's (`HardContractOptions.variantState`).
   *
   * The processor is not called: it is not prepared again (every variant has the same shapes)
   * and no phase of it runs for the build, so the figure of an added plan is the engine's
   * alone, without the processor's `beforeInference` and `afterInference`, until the next
   * prepare measures the whole. An internal latency other than the reported one and an
   * inference time above the budget the stream is sized for are one warning each, never a
   * refusal. The model is copied: `model` can be destroyed right after.
   *
   * A refused call throws an `AniraError` with the status and the reason and leaves the handler
   * as it was (nothing is published, everything the call built is released):
   * `ANIRA_ERROR_CONFIG` for a rule the model breaks, a form other than the handler's (the
   * message names the variant, the slot and the property; a model of another form is another
   * handler) or a variant without a plan; on a prepared handler also what a load failed with
   * (`ANIRA_ERROR_NO_SUCH_FILE`, `ANIRA_ERROR_MODEL_LOAD`, `ANIRA_ERROR_ENGINE`),
   * `ANIRA_ERROR_EDGE_UNREACHABLE` when no plan of the variant is within the contract's edge
   * cost, and `ANIRA_ERROR_INVALID_STATE` when the warm-up finds no Worker inside its loop.
   * `ANIRA_ERROR_INVALID_STATE` too when called from inside a callback of a control entry of
   * this handler (a JavaScript processor's `prepare`).
   */
  addVariant(model: ModelConfig): number {
    return this.abi.anira_handler_add_variant(this.handler, model.native())
  }

  /**
   * Remove a variant from the handler and free what only it held
   * (`anira_handler_remove_variant`, as `anira::InferenceHandler::remove_variant`). A control
   * entry like `addVariant`: it runs on the main thread and blocks it, and it is legal while an
   * audio node streams through the handler. On an unprepared handler the variant leaves the
   * handler's copy of the pipeline. On a prepared handler its plans leave the table: their rows
   * stay at their indices, marked `ANIRA_PLAN_REMOVED` (`PlanInfo.state`) in a new report (read
   * `planReport()` again; a `PlanReport` taken earlier keeps the rows it had);
   * `PlanReport.find`, `setModel` and `setBackend` skip them, and `setPlan` on one throws with
   * `ANIRA_ERROR_NOT_FOUND`. The call then waits until the render quantum the worklet is inside
   * has returned and until the variant's inferences in flight have ended, without a timeout
   * (their chunks are delivered; a node that is not rendering holds nothing back), and releases
   * the models: a model no other plan of any handler runs is unloaded. It returns when the
   * module's memory is released; what a JavaScript engine loaded for the model in the Workers
   * (its `Engine.Loaded` objects) is destroyed by each Worker at its next look at the queue, at
   * once for a parked one, which this call does not wait for.
   *
   * The variant's index is never given out again (`addVariant` continues behind the highest
   * index the stage has had); a later prepare keeps the indices of the variants that remain and
   * numbers the plans anew, without the removed rows. `numEntries()` and the latency do not
   * change before the next prepare, also when the removed variant had the largest internal
   * latency. Read `getPlan()` again: a selection of one of the variant's plans that raced the
   * call was moved to the first ready plan.
   *
   * A refused call throws an `AniraError` and changes nothing: `ANIRA_ERROR_NOT_FOUND` for an
   * index that names no variant of the handler, or one that was removed;
   * `ANIRA_ERROR_INVALID_STATE` for the last variant the handler has, for the variant of the
   * selected plan (select a plan of another variant first: `setModel`), and when called from
   * inside a callback of a control entry of this handler.
   */
  removeVariant(variant: number): void {
    this.abi.anira_handler_remove_variant(this.handler, variant)
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
