import { type AniraBindings, heap } from './abi/bindings'
import { ANIRA_STAGE_FLAG_REALTIME_PRE_POST } from './abi/enums'
import type { ModelConfig, WithAbi } from './ModelConfig'
import { keyOf } from './registry'

/** A JavaScript engine of the pipeline (`anira_em_pipeline_register_js_engine`). */
export type EngineOptions = {
  /**
   * `ANIRA_ENGINE_FLAG_*`: the engine's promises (`ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL` for an
   * engine that reads no model file of the config). Default none.
   */
  flags?: number
  /** `ANIRA_EM_ENGINE_HOOK_*` beyond process (`ANIRA_EM_ENGINE_HOOK_RESET`). Default none. */
  hooks?: number
}

/** The JavaScript stage of the pipeline (`anira_em_pipeline_add_js_stage`). */
export type StageOptions = {
  /** `ANIRA_EM_HOOK_*`: the phases that cross into JavaScript. */
  hooks: number
  /**
   * `ANIRA_STAGE_FLAG_*`: the stage's real-time promise, stated. Default
   * `ANIRA_STAGE_FLAG_REALTIME_PRE_POST` (a Hard contract requires it of a stage with a
   * pre_process, post_process or reset hook, which run on the audio thread).
   */
  flags?: number
}

/**
 * A pipeline (`anira_pipeline`): the inference stage over model configs and, optionally, a
 * JavaScript stage and JavaScript engines, which the scopes that run them define under the same
 * name (`defineStage`, `defineEngine`). A handler copies it; destroy it once the handlers exist.
 */
export class Pipeline {
  readonly abi: AniraBindings
  /** The `anira_pipeline*` the raw entries take. */
  readonly ptr: number

  private constructor(abi: AniraBindings, ptr: number) {
    this.abi = abi
    this.ptr = ptr
  }

  static create(anira: WithAbi): Pipeline {
    return new Pipeline(anira.abi, anira.abi.anira_pipeline_create())
  }

  /**
   * The inference stage over `models`, the variants (one in this pre-release), on the default
   * candidates: every engine of the build and every registered engine, on the CPU.
   */
  addInference(models: ModelConfig | ModelConfig[]): this {
    const list = Array.isArray(models) ? models : [models]
    const m = this.abi.module
    const variants = m._malloc(list.length * 4)
    if (variants === 0) throw new RangeError('anira: out of memory for the variants')
    try {
      list.forEach((model, i) => (heap(m).HEAPU32[(variants >>> 2) + i] = model.ptr))
      this.abi.anira_pipeline_add_inference(this.ptr, variants, list.length, 0, 0)
    } finally {
      m._free(variants)
    }
    return this
  }

  /** A JavaScript engine under its reverse-URI `id`, which the model entries name. */
  registerEngine(id: string, options: EngineOptions = {}): this {
    this.abi.anira_em_pipeline_register_js_engine(
      this.ptr,
      id,
      keyOf(id),
      options.hooks ?? 0,
      options.flags ?? 0
    )
    return this
  }

  /** The pipeline's one JavaScript stage, under `name`. */
  addStage(name: string, options: StageOptions): this {
    this.abi.anira_em_pipeline_add_js_stage(
      this.ptr,
      keyOf(name),
      options.hooks,
      options.flags ?? ANIRA_STAGE_FLAG_REALTIME_PRE_POST
    )
    return this
  }

  destroy(): void {
    this.abi.anira_pipeline_destroy(this.ptr)
  }
}
