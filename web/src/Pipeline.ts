import { type AniraBindings, heap } from './abi/bindings'
import { ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST } from './abi/enums'
import type { InputSpec } from './staticInput'
import type { ModelConfig, WithAbi } from './ModelConfig'
import { keyOf } from './registry'

/** A JavaScript engine of the pipeline (`addEngine`, `anira_em_pipeline_register_js_engine`). */
export type EngineOptions = {
  /**
   * `ANIRA_ENGINE_FLAG_*`: the engine's promises (`ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL` for an
   * engine that reads no model file of the config). Default none.
   */
  flags?: number
  /** `ANIRA_EM_ENGINE_HOOK_*` beyond process (`ANIRA_EM_ENGINE_HOOK_RESET`). Default none. */
  hooks?: number
}

/** The JavaScript processor of the pipeline (`anira_em_pipeline_add_js_processor`). */
export type ProcessorOptions = {
  /**
   * The phases that cross into JavaScript, an OR of `ANIRA_EM_PROCESSOR_HOOK_*` bits, each
   * `1 << ANIRA_PHASE_*` (the index space of `anira::Processor::k_*`, plus the lifecycle phases a
   * C++ processor fills as members: prepare, release, reset, unprepare, init).
   */
  phases: number
  /**
   * `ANIRA_PROCESSOR_FLAG_*`: the processor's real-time promise, stated. Default
   * `ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST` (a Hard contract requires it of a processor with a
   * pre_process, post_process or reset hook, which run on the audio thread).
   */
  flags?: number
}

/**
 * A pipeline (`anira_pipeline`), the web twin of `anira::Pipeline`, one member per C entry:
 * `addInference` (the inference stage over model configs), `addProcessor` (the custom stage, a
 * JavaScript processor) and `addEngine` (a JavaScript engine the model entries name by id),
 * which the scopes that run them define under the same name (`defineProcessor`,
 * `defineEngine`). Where the web differs: a processor and an engine are named, their code living
 * in each realm that runs them (`add_processor` and `add_engine` take the objects in C++), and
 * the pipeline is destroyed with `destroy()` once the handlers exist (a handler copies it).
 */
export class Pipeline {
  readonly abi: AniraBindings
  private readonly pipeline: number
  private specs: InputSpec[] = []

  private constructor(abi: AniraBindings, pipeline: number) {
    this.abi = abi
    this.pipeline = pipeline
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
      list.forEach((model, i) => (heap(m).HEAPU32[(variants >>> 2) + i] = model.native()))
      this.abi.anira_pipeline_add_inference(this.pipeline, variants, list.length, 0, 0)
    } finally {
      m._free(variants)
    }
    this.specs = list.length > 0 ? list[0].inputSpecs() : []
    return this
  }

  /** @internal The input specs of the inference stage's variant: what a handler's Static inputs take. */
  inputSpecs(): InputSpec[] {
    return this.specs
  }

  /** A JavaScript engine under its reverse-URI `id`, which the model entries name. */
  addEngine(id: string, options: EngineOptions = {}): this {
    this.abi.anira_em_pipeline_register_js_engine(
      this.pipeline,
      id,
      keyOf(id),
      options.hooks ?? 0,
      options.flags ?? 0
    )
    return this
  }

  /** The pipeline's one JavaScript processor, under `name`. */
  addProcessor(name: string, options: ProcessorOptions): this {
    this.abi.anira_em_pipeline_add_js_processor(
      this.pipeline,
      keyOf(name),
      options.phases,
      options.flags ?? ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST
    )
    return this
  }

  /** The `anira_pipeline*` the raw entries take. */
  native(): number {
    return this.pipeline
  }

  destroy(): void {
    this.abi.anira_pipeline_destroy(this.pipeline)
  }
}
