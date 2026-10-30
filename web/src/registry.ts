import { AniraBindings, type AbiModule, heap } from './abi/bindings'
import { anira_phase, anira_status } from './abi/enums'
import { anira_engine_ctx, anira_log_record, anira_tensor } from './abi/layout'

/**
 * What a stage or an engine is constructed with, in the scope that runs it (the main thread, an
 * inference Worker, the audio worklet): the typed C ABI of that scope's module instance and the
 * instance itself (its heap views: `module.HEAPF32`, ...). The memory grows, and the views of
 * a scope are rebuilt by `heap(module)`: the trampolines call it before each call into a stage
 * or an engine (a rebuild at most once after a growth, seven typed arrays: on the audio thread a
 * small JavaScript allocation, never one by the C audio path), so `module.HEAPF32` is current
 * there for memory that existed when the call began.
 */
export type ScopeEnv = {
  abi: AniraBindings
  module: AbiModule
}

/**
 * A JavaScript stage: one method per phase it takes part in. Each runs in the scope that runs
 * that phase of the C call: the audio worklet for `preProcess`, `postProcess` and `reset` under
 * a Hard contract, an inference Worker for `beforeInference` and `afterInference`, the main
 * thread for `init`, `prepare`, `unprepare` and `release`. The pointers are into the shared
 * memory: `ctx` an `anira_stage_ctx` (`layout`), `prepared` what this handler's `prepare` stored
 * in its `outPrepared` slot. A method returns an `anira_status` (nothing is `ANIRA_OK`); one that
 * throws fails its call (the C trampoline catches it and reports it on the console). A stage
 * without `preProcess` / `postProcess` runs anira's default body there.
 */
export interface Stage {
  preProcess?(ctx: number, prepared: number): number | void
  postProcess?(ctx: number, prepared: number): number | void
  beforeInference?(ctx: number, prepared: number): number | void
  afterInference?(ctx: number, prepared: number): number | void
  reset?(ctx: number, prepared: number): void
  prepare?(prepareInfo: number, outPrepared: number): number | void
  unprepare?(prepared: number): void
  init?(initInfo: number): number | void
  release?(): void
}

/**
 * The model entry a JavaScript engine runs, what a C engine's load reads of its row
 * (`anira_engine_load_info.row`, `anira_model_config_model_bytes` / `_model_path`): the
 * entry's bytes as a view of the shared memory (no copy; valid while the loaded model lives,
 * so copy what must outlive the call), or its path when it carries no bytes, or neither for an
 * engine that declares `ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL`. `ModelConfig.fromUrl` attaches the
 * bytes of a JavaScript engine's entry, so the model is fetched once, on the main thread.
 */
export type EngineModel = {
  /** The entry's position in the model config's models list. */
  row: number
  /** The entry's bytes, a view of the shared memory, or null. */
  bytes: Uint8Array | null
  /** The entry's path when it has no bytes, else null. */
  path: string | null
}

/**
 * A JavaScript engine, constructed once per inference Worker: `init` (optional, may be async)
 * runs before the Worker enters its loop; `load` (optional, synchronous) receives the model
 * entry the calls run, on the inference Worker before the first call on a loaded model (and
 * again when a later call runs another one); `process` once per inference (`ctx` an
 * `anira_engine_ctx`, read with `engineTensors`; `prepared` the loaded model's record, which
 * `engineModel` reads), `reset` at a new stream (with `ANIRA_EM_ENGINE_HOOK_RESET`), `query` on
 * the main thread (with `ANIRA_EM_ENGINE_HOOK_QUERY`).
 */
export interface Engine {
  init?(): void | Promise<void>
  load?(model: EngineModel): number | void
  process(ctx: number, prepared: number): number | void
  reset?(ctx: number, prepared: number): void
  query?(initInfo: number, outAvailable: number): number | void
}

export type StageFactory = (env: ScopeEnv) => Stage
export type EngineFactory = (env: ScopeEnv) => Engine

/** A log record as the JavaScript log hook delivers it (`anira_log_record`). */
export type LogRecord = {
  level: number
  flags: number
  sequence: bigint
  group: string
  message: string
}

/**
 * The key a stage name or an engine id crosses the C ABI as (the `js_key` of
 * `anira_em_pipeline_add_js_stage` and `anira_em_pipeline_register_js_engine`): the 32-bit
 * FNV-1a hash of its UTF-16 code units, the same in every scope, so a stage defined under one
 * name on the main thread, the Workers and the worklet is one stage.
 */
export const keyOf = (name: string): number => {
  let hash = 0x811c9dc5
  for (let i = 0; i < name.length; i++) {
    hash ^= name.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

type Entry<F, T> = { name: string; factory: F; instance: T | null }

const status = (result: number | void): number => (result ?? anira_status.ANIRA_OK) | 0

/**
 * The registry of one scope, `globalThis[Symbol.for('anira.registry')]`: the stages and engines
 * the scope's code defined, keyed by `keyOf(name)`, and the instances built over the scope's
 * module. The C trampolines of `anira/anira_em.h` call `stage`, `engine` and `log` with the
 * module of the instance that runs the call; an inference Worker awaits `boot` before its loop.
 */
export class Registry {
  readonly version = 2
  /** @hidden */
  private readonly stages = new Map<number, Entry<StageFactory, Stage>>()
  /** @hidden */
  private readonly engines = new Map<number, Entry<EngineFactory, Engine>>()
  private env: ScopeEnv | null = null
  /** The scope's log sink (`anira_em_set_log_hook`); none: the records are not shown here. */
  logSink: ((record: LogRecord) => void) | null = null

  defineStage(name: string, factory: StageFactory): number {
    return this.define(this.stages, name, factory)
  }

  defineEngine(id: string, factory: EngineFactory): number {
    return this.define(this.engines, id, factory)
  }

  /** @hidden */
  private define<F, T>(map: Map<number, Entry<F, T>>, name: string, factory: F): number {
    const key = keyOf(name)
    const existing = map.get(key)
    if (existing && existing.name !== name) {
      throw new Error(
        `anira: '${name}' and '${existing.name}' share the key ${key}; rename one`
      )
    }
    map.set(key, { name, factory, instance: null })
    return key
  }

  /** The env of this scope's module instance; the first call binds the scope to `module`. */
  envOf(module: AbiModule): ScopeEnv {
    if (this.env === null || this.env.module !== module) {
      this.env = { abi: new AniraBindings(module), module }
    }
    return this.env
  }

  /**
   * Bind the scope to its module instance and build and initialise every engine defined so
   * far: what an inference Worker awaits before it enters its loop.
   */
  async boot(module: AbiModule): Promise<void> {
    const env = this.envOf(module)
    await Promise.all(
      [...this.engines.values()].map(async (entry) => {
        entry.instance ??= entry.factory(env)
        await entry.instance.init?.()
      })
    )
  }

  private stageOf(module: AbiModule, key: number): Stage {
    const entry = this.stages.get(key)
    if (!entry) throw new Error(`no JavaScript stage under key ${key} in this scope`)
    entry.instance ??= entry.factory(this.envOf(module))
    return entry.instance
  }

  private engineOf(module: AbiModule, key: number, idPtr: number): Engine {
    const entry = this.engines.get(key)
    if (!entry) {
      const id = idPtr === 0 ? `key ${key}` : module.UTF8ToString(idPtr)
      throw new Error(
        `no JavaScript engine '${id}' in this scope; define it with defineEngine`
      )
    }
    entry.instance ??= entry.factory(this.envOf(module))
    return entry.instance
  }

  /** The stage trampoline's target. */
  stage(module: AbiModule, key: number, phase: number, a: number, b: number): number {
    const p = anira_phase
    if (phase === p.ANIRA_PHASE_RELEASE) {
      const entry = this.stages.get(key)
      try {
        entry?.instance?.release?.()
      } finally {
        if (entry) entry.instance = null
      }
      return anira_status.ANIRA_OK
    }
    const stage = this.stageOf(heap(module), key)
    switch (phase) {
      case p.ANIRA_PHASE_PRE_PROCESS:
        return stage.preProcess
          ? status(stage.preProcess(a, b))
          : module._anira_stage_default_pre_process(a)
      case p.ANIRA_PHASE_POST_PROCESS:
        return stage.postProcess
          ? status(stage.postProcess(a, b))
          : module._anira_stage_default_post_process(a)
      case p.ANIRA_PHASE_BEFORE_INFERENCE:
        return status(stage.beforeInference?.(a, b))
      case p.ANIRA_PHASE_AFTER_INFERENCE:
        return status(stage.afterInference?.(a, b))
      case p.ANIRA_PHASE_RESET:
        stage.reset?.(a, b)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_PREPARE:
        return status(stage.prepare?.(a, b))
      case p.ANIRA_PHASE_UNPREPARE:
        stage.unprepare?.(a)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_INIT:
        return status(stage.init?.(a))
    }
    throw new Error(`JavaScript stage ${key}: phase ${phase} is not a stage phase`)
  }

  /** The engine trampoline's target. */
  engine(
    module: AbiModule,
    key: number,
    phase: number,
    a: number,
    b: number,
    idPtr: number
  ): number {
    const engine = this.engineOf(heap(module), key, idPtr)
    const p = anira_phase
    switch (phase) {
      case p.ANIRA_PHASE_INFERENCE: {
        const loaded = this.loadOnce(module, engine, b)
        return loaded !== anira_status.ANIRA_OK ? loaded : status(engine.process(a, b))
      }
      case p.ANIRA_PHASE_RESET:
        if (this.loadOnce(module, engine, b) === anira_status.ANIRA_OK)
          engine.reset?.(a, b)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_QUERY:
        return status(engine.query?.(a, b))
    }
    throw new Error(`JavaScript engine ${key}: phase ${phase} is not an engine phase`)
  }

  /** @hidden */
  private readonly loadedModels = new WeakMap<Engine, number>()

  /** @hidden */
  private loadOnce(module: AbiModule, engine: Engine, model: number): number {
    if (!engine.load || model === 0 || this.loadedModels.get(engine) === model) {
      return anira_status.ANIRA_OK
    }
    const loaded = status(engine.load(engineModel(this.envOf(module), model)))
    if (loaded === anira_status.ANIRA_OK) this.loadedModels.set(engine, model)
    return loaded
  }

  /** The log hook's target. */
  log(module: AbiModule, record: number): void {
    if (this.logSink) this.logSink(logRecordAt(module, record))
  }
}

const REGISTRY = Symbol.for('anira.registry')

/** The registry of this scope, installed on its `globalThis` at the first use. */
export const scopeRegistry = (): Registry => {
  const scope = globalThis as unknown as Record<symbol, Registry | undefined>
  return (scope[REGISTRY] ??= new Registry())
}

/**
 * Define a JavaScript stage under `name` in this scope: call it in every scope that runs one of
 * its phases (the main thread's code, the inference Workers' module, the worklet's module), with
 * a factory that builds the scope's instance over its env. Returns the stage's key (`keyOf`).
 */
export const defineStage = (name: string, factory: StageFactory): number =>
  scopeRegistry().defineStage(name, factory)

/**
 * Define a JavaScript engine under its reverse-URI `id` in this scope: in the inference Workers'
 * module (`process`, `reset`; built and `init`ed before each Worker's loop) and, for `query`,
 * on the main thread. Returns the engine's key (`keyOf`).
 */
export const defineEngine = (id: string, factory: EngineFactory): number =>
  scopeRegistry().defineEngine(id, factory)

/** A model tensor of one engine call: its record, its element count and its float32 data. */
export type EngineTensorView = {
  ptr: number
  elements: number
  /** A view of the data when the tensor is host float32, else null. */
  f32: Float32Array | null
}

/** The input and output tensors of an engine call (`anira_engine_ctx`). */
export const engineTensors = (
  env: ScopeEnv,
  ctx: number
): { inputs: EngineTensorView[]; outputs: EngineTensorView[] } => {
  const { module: m, abi } = env
  const f = anira_engine_ctx.fields
  const read = (array: number, count: number): EngineTensorView[] => {
    const out: EngineTensorView[] = []
    for (let i = 0; i < count; i++) {
      const ptr = array + i * anira_tensor.size
      const data = abi.anira_tensor_data_f32(ptr)
      const elements = abi.anira_tensor_num_elements(ptr)
      out.push({
        ptr,
        elements,
        f32: data === 0 ? null : new Float32Array(heap(m).HEAPF32.buffer, data, elements),
      })
    }
    return out
  }
  return {
    inputs: read(
      heap(m).HEAPU32[(ctx + f.inputs.offset) >>> 2],
      heap(m).HEAPU32[(ctx + f.num_inputs.offset) >>> 2]
    ),
    outputs: read(
      heap(m).HEAPU32[(ctx + f.outputs.offset) >>> 2],
      heap(m).HEAPU32[(ctx + f.num_outputs.offset) >>> 2]
    ),
  }
}

// The record a JavaScript engine's calls receive as `prepared` (src/platform/emscripten/
// em_hooks.cpp, EngineModel): four 32-bit words, the row, the bytes, their size, the path.
const ENGINE_MODEL_ROW = 0
const ENGINE_MODEL_BYTES = 4
const ENGINE_MODEL_SIZE = 8
const ENGINE_MODEL_PATH = 12

/** The model entry of an engine call's `prepared` record (see `EngineModel`). */
export const engineModel = (env: ScopeEnv, prepared: number): EngineModel => {
  const m = env.module
  const word = (offset: number) => heap(m).HEAPU32[(prepared + offset) >>> 2]
  const bytes = word(ENGINE_MODEL_BYTES)
  const size = word(ENGINE_MODEL_SIZE)
  const path = word(ENGINE_MODEL_PATH)
  return {
    row: word(ENGINE_MODEL_ROW),
    bytes: bytes === 0 ? null : new Uint8Array(heap(m).HEAPU8.buffer, bytes, size),
    path: path === 0 ? null : m.UTF8ToString(path),
  }
}

/** The fields of an `anira_log_record`. */
export const logRecordAt = (module: AbiModule, record: number): LogRecord => {
  const f = anira_log_record.fields
  const text = (ptr: number) => (ptr === 0 ? '' : module.UTF8ToString(ptr))
  return {
    level: heap(module).HEAPU32[(record + f.level.offset) >>> 2],
    flags: heap(module).HEAPU32[(record + f.flags.offset) >>> 2],
    sequence: heap(module).HEAPU64[(record + f.sequence.offset) >>> 3],
    group: text(heap(module).HEAPU32[(record + f.group.offset) >>> 2]),
    message: text(heap(module).HEAPU32[(record + f.message.offset) >>> 2]),
  }
}
