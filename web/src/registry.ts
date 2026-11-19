import { AniraBindings, type AbiModule, heap } from './abi/bindings'
import { anira_phase, anira_status } from './abi/enums'
import { anira_engine_ctx, anira_log_record, anira_tensor } from './abi/layout'
import { type TypedArray, typedView } from './dtype'

/**
 * What a processor or an engine is constructed with, in the scope that runs it (the main thread, an
 * inference Worker, the audio worklet): the typed C ABI of that scope's module instance and the
 * instance itself (its heap views: `module.HEAPF32`, ...). The memory grows, and the views of
 * a scope are rebuilt by `heap(module)`: the trampolines call it before each call into a processor
 * or an engine (a rebuild at most once after a growth, seven typed arrays: on the audio thread a
 * small JavaScript allocation, never one by the C audio path), so `module.HEAPF32` is current
 * there for memory that existed when the call began.
 */
export type ScopeEnv = {
  abi: AniraBindings
  module: AbiModule
}

/**
 * A JavaScript processor: one method per phase it takes part in. Each runs in the scope that runs
 * that phase of the C call: the audio worklet for `preProcess`, `postProcess` and `reset` under a
 * Hard contract, an inference Worker for `beforeInference` and `afterInference`, the main thread
 * for `init`, `prepare`, `unprepare` and `release`. The pointers are into the shared memory: `ctx`
 * an `anira_processor_ctx` (`layout`), `prepared` what this handler's `prepare` stored in its
 * `outPrepared` slot. A method returns an `anira_status` (nothing is `ANIRA_OK`); one that throws
 * fails its call (the C trampoline catches it and reports it on the console). A processor without
 * `preProcess` / `postProcess` runs anira's default body there.
 */
export interface Processor {
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
 * What a JavaScript engine's `load` receives, the web twin of `anira::EngineLoadInfo`: the
 * model entry the engine loads, as a C engine's load reads it of its row
 * (`anira_engine_load_info.row`, `anira_model_config_model_bytes` / `_model_path`).
 * `ModelConfig.fromUrl` attaches the bytes of a JavaScript engine's entry, so the model is
 * fetched once, on the main thread. Where the web differs: the C load record lives for the
 * load call on the main thread, and a Worker loads later, at its first call for the model, from
 * what anira keeps of the loaded model: its row, that row's path and that row's bytes. The
 * templates, the names, the instances, the provider and the variant's other entries of
 * `anira::EngineLoadInfo` are not carried.
 */
export type EngineLoadInfo = {
  /** This engine's entry in the variant's model list (`row()`). */
  row: number
  /**
   * The entry's bytes (`model_bytes(row())`), or null for a path entry: a view of the shared
   * memory, no copy, valid until the model is unloaded. `destroy` runs behind the unload and
   * must not read them, so copy what the object needs then.
   */
  modelBytes: Uint8Array | null
  /** The entry's path (`model_path(row())`) when it has no bytes, else null. */
  modelPath: string | null
}

/**
 * A JavaScript engine, the web twin of `anira::Engine`: `init`, then `load` once per loaded
 * model, which returns the object the model's inferences run on (`Engine.Loaded`); `query` on
 * the main thread (with `ANIRA_EM_ENGINE_HOOK_QUERY`). Where the web differs, and why:
 *
 * - **One engine per realm.** A Worker is a realm of its own, so `defineEngine`'s factory
 *   builds one engine in each inference Worker, and `load` runs in each: a loaded model is one
 *   `Engine.Loaded` per Worker, built at that Worker's first call for the model, inside the
 *   call (the main thread, where anira loads, cannot reach a Worker's realm). A `load` that
 *   throws fails that call (`ANIRA_ERROR_ENGINE`: zeros, the real-time error latched) and is
 *   tried again at the Worker's next call for the model; natively it fails the prepare.
 * - **`init` has no record and runs at the Worker's boot**, before its loop (it may be
 *   async), instead of at the first prepare that reaches the engine.
 * - **No `release`, `flags`, `consumed_kinds`, `providers` and no id:** the engine lives as
 *   long as its Worker; the id is `defineEngine`'s, the flags and the edge statements are
 *   `Pipeline.addEngine`'s, on the main thread, which adds the engine before a Worker exists.
 */
export interface Engine {
  /** Before the Worker's loop; may be async. What the engine keeps for its Worker's life. */
  init?(): void | Promise<void>
  /**
   * Loads the entry's model and returns the object its inferences run on, which anira owns
   * from then on and destroys when the model is unloaded. Synchronous (the Worker's event
   * loop does not run). It may throw; returning nothing is an error.
   */
  load(info: EngineLoadInfo): Engine.Loaded
  /**
   * Which of the engine's providers are usable here, as a bitmask over the list
   * (`anira::Engine::query`); `initInfo` is the `anira_init_info`. On the main thread, with
   * `ANIRA_EM_ENGINE_HOOK_QUERY`.
   */
  query?(initInfo: number): bigint
}

// The namespace beside the interface gives the object's type its native name, `Engine.Loaded`
// for `anira::Engine::Loaded`; it holds a type alone and compiles to nothing.
export declare namespace Engine {
  /**
   * One loaded model of a JavaScript engine in one Worker, the web twin of
   * `anira::Engine::Loaded`: what `Engine.load` returns. The model's inferences on that Worker
   * run on it, each over the call's `anira_engine_ctx` (`ctx`, read with `engineTensors`), one
   * at a time. Where the web differs, and why:
   *
   * - **`process` and `reset` are members of the `Loaded`: there is no `Prepared`.** Natively
   *   `prepare` returns one `Prepared` per handler, which holds what that handler keeps per
   *   stream. A handler's inferences run on whichever Worker takes them, so an object per
   *   handler in one Worker's realm would see only some of them and could keep nothing of a
   *   stream: what a model keeps between calls is its declared State pair, in the shared
   *   memory. `reset` (with `ANIRA_EM_ENGINE_HOOK_RESET`) runs where the native one does,
   *   before the first inference of a new stream of an exclusive handler and of a fresh
   *   variant state, on the Worker that takes that inference.
   * - **`destroy` stands for the destructor**, as on every class of the package. anira calls
   *   it once the model is unloaded (the last handler that held it was re-prepared or
   *   destroyed, or removed its variant), on the object's own Worker: at once when the Worker
   *   is parked, else behind the inference it is running (of another model: the unload comes
   *   behind the model's last inference). The control call that unloads returns without
   *   waiting for it, where the native one returns with the memory released. A Worker that
   *   leaves its loop destroys what it holds; one that is terminated where it stands (an
   *   engine call that does not return within 5 s of the stop) runs no `destroy`.
   * - **No `edge`:** the main thread states the edges per direction (`Pipeline.addEngine`).
   */
  interface Loaded {
    /** One inference; returns an `anira_status` (nothing is `ANIRA_OK`). */
    process(ctx: number): number | void
    /** A new stream begins with the inference that follows on this Worker (see above). */
    reset?(ctx: number): void
    /** Releases what `load` built for the model. Must not read `EngineLoadInfo.modelBytes`. */
    destroy?(): void
  }
}

export type ProcessorFactory = (env: ScopeEnv) => Processor
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
 * The key a processor name or an engine id crosses the C ABI as (the `js_key` of
 * `anira_em_pipeline_add_js_processor` and `anira_em_pipeline_add_js_engine`): the 32-bit
 * FNV-1a hash of its UTF-16 code units, the same in every scope, so a processor defined under one
 * name on the main thread, the Workers and the worklet is one processor.
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
 * The registry of one scope, `globalThis[Symbol.for('anira.registry')]`: the processors and engines
 * the scope's code defined, keyed by `keyOf(name)`, and the instances built over the scope's
 * module. The C trampolines of `anira/anira_em.h` call `processor`, `engine` and `log` with the
 * module of the instance that runs the call; an inference Worker awaits `boot` before its loop.
 */
export class Registry {
  readonly version = 2
  /** @hidden */
  private readonly processors = new Map<number, Entry<ProcessorFactory, Processor>>()
  /** @hidden */
  private readonly engines = new Map<number, Entry<EngineFactory, Engine>>()
  private env: ScopeEnv | null = null
  /** The scope's log sink (`anira_em_set_log_hook`); none: the records are not shown here. */
  logSink: ((record: LogRecord) => void) | null = null

  defineProcessor(name: string, factory: ProcessorFactory): number {
    return this.define(this.processors, name, factory)
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

  private processorOf(module: AbiModule, key: number): Processor {
    const entry = this.processors.get(key)
    if (!entry) throw new Error(`no JavaScript processor under key ${key} in this scope`)
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

  /** The processor trampoline's target. */
  processor(module: AbiModule, key: number, phase: number, a: number, b: number): number {
    const p = anira_phase
    if (phase === p.ANIRA_PHASE_RELEASE) {
      const entry = this.processors.get(key)
      try {
        entry?.instance?.release?.()
      } finally {
        if (entry) entry.instance = null
      }
      return anira_status.ANIRA_OK
    }
    const processor = this.processorOf(heap(module), key)
    switch (phase) {
      case p.ANIRA_PHASE_PRE_PROCESS:
        return processor.preProcess
          ? status(processor.preProcess(a, b))
          : module._anira_processor_default_pre_process(a)
      case p.ANIRA_PHASE_POST_PROCESS:
        return processor.postProcess
          ? status(processor.postProcess(a, b))
          : module._anira_processor_default_post_process(a)
      case p.ANIRA_PHASE_BEFORE_INFERENCE:
        return status(processor.beforeInference?.(a, b))
      case p.ANIRA_PHASE_AFTER_INFERENCE:
        return status(processor.afterInference?.(a, b))
      case p.ANIRA_PHASE_RESET:
        processor.reset?.(a, b)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_PREPARE:
        return status(processor.prepare?.(a, b))
      case p.ANIRA_PHASE_UNPREPARE:
        processor.unprepare?.(a)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_INIT:
        return status(processor.init?.(a))
    }
    throw new Error(
      `JavaScript processor ${key}: phase ${phase} is not a processor phase`
    )
  }

  /**
   * The engine trampoline's target. `ANIRA_PHASE_INFERENCE` and `ANIRA_PHASE_RESET` run on the
   * scope's object of the call's loaded model (`a` the `anira_engine_ctx`, `b` the model's
   * record), built by the engine's `load` at the scope's first call for the model.
   * `ANIRA_PHASE_QUERY` writes the engine's answer to the mask at `b`. `ANIRA_PHASE_UNLOAD`
   * (`key` 0) releases the scope's objects of unloaded models: `a` is the address of the
   * serials of the models loaded now, `b` their count, and every object of another model is
   * destroyed.
   */
  engine(
    module: AbiModule,
    key: number,
    phase: number,
    a: number,
    b: number,
    idPtr: number
  ): number {
    const p = anira_phase
    switch (phase) {
      case p.ANIRA_PHASE_INFERENCE:
        return status(this.loadedOf(heap(module), key, idPtr, b).process(a))
      case p.ANIRA_PHASE_RESET:
        this.loadedOf(heap(module), key, idPtr, b).reset?.(a)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_QUERY: {
        const available = this.engineOf(heap(module), key, idPtr).query?.(a)
        if (available !== undefined) heap(module).HEAPU64[b >>> 3] = available
        return anira_status.ANIRA_OK
      }
      case p.ANIRA_PHASE_UNLOAD:
        this.unload(new Set(heap(module).HEAPU32.subarray(a >>> 2, (a >>> 2) + b)))
        return anira_status.ANIRA_OK
    }
    throw new Error(`JavaScript engine ${key}: phase ${phase} is not an engine phase`)
  }

  /**
   * @hidden What this scope's engines loaded: the object of each loaded model one of its
   * calls ran here, by the model's serial.
   */
  private readonly loaded = new Map<number, Engine.Loaded>()

  /**
   * @hidden The scope's object of the loaded model whose record is `record`: the engine's
   * `load` at the first call for the model, the same object at every later one.
   */
  private loadedOf(
    module: AbiModule,
    key: number,
    idPtr: number,
    record: number
  ): Engine.Loaded {
    // The record's serial names the loaded model. Its address does not: the record of a model
    // loaded later may lie where the record of an unloaded one lay (a handler destroyed, a
    // variant removed), and the new model must not run on the object of the old one.
    const serial = record === 0 ? 0 : module.HEAPU32[(record + ENGINE_MODEL_SERIAL) >>> 2]
    const known = this.loaded.get(serial)
    if (known !== undefined) return known
    const engine = this.engineOf(module, key, idPtr)
    if (record === 0) {
      throw new Error(
        `JavaScript engine ${key}: a call without its loaded model's record`
      )
    }
    const loaded = engine.load(engineLoadInfo(module, record))
    if (!loaded) {
      throw new Error(
        `JavaScript engine ${key}: load returned no object: nothing can run the inference`
      )
    }
    this.loaded.set(serial, loaded)
    return loaded
  }

  /**
   * @hidden Destroys the scope's object of every model whose serial is not in `stillLoaded`.
   * Every object goes, also behind a `destroy` that throws; the first throw is passed on.
   */
  private unload(stillLoaded: ReadonlySet<number>): void {
    const thrown: unknown[] = []
    for (const [serial, loaded] of this.loaded) {
      if (stillLoaded.has(serial)) continue
      this.loaded.delete(serial)
      try {
        loaded.destroy?.()
      } catch (error) {
        thrown.push(error)
      }
    }
    if (thrown.length > 0) throw thrown[0]
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
 * Define a JavaScript processor under `name` in this scope: call it in every scope that runs one of
 * its phases (the main thread's code, the inference Workers' module, the worklet's module), with
 * a factory that builds the scope's instance over its env. Returns the processor's key (`keyOf`).
 */
export const defineProcessor = (name: string, factory: ProcessorFactory): number =>
  scopeRegistry().defineProcessor(name, factory)

/**
 * Define a JavaScript engine under its reverse-URI `id` in this scope: in the inference Workers'
 * module (built and `init`ed before each Worker's loop; `load` and the loaded models' `process`,
 * `reset` and `destroy` run there) and, for `query`, on the main thread. Returns the engine's
 * key (`keyOf`).
 */
export const defineEngine = (id: string, factory: EngineFactory): number =>
  scopeRegistry().defineEngine(id, factory)

/**
 * A model tensor of one engine call: its record, its element count, its dtype (the spec's) and
 * its data, a typed array of the dtype's element type.
 */
export type EngineTensorView = {
  ptr: number
  elements: number
  /** The tensor's `anira_dtype` (`ANIRA_DTYPE_*`): its spec's, whatever the host streams. */
  dtype: number
  /**
   * A view of the data when the tensor is a packed host block: `Float32Array` for float32,
   * `Float64Array`, `Int8Array`, `Uint8Array`, `Int16Array`, `Int32Array`, `BigInt64Array` for
   * the others with a JavaScript element type, `Uint16Array` (the 16-bit words) for float16 and
   * bfloat16, `Uint8Array` (0 or 1) for bool8; else null.
   */
  data: TypedArray | null
  /** `data` when the tensor is host float32, else null. */
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
      const dtype = heap(m).HEAPU32[(ptr + anira_tensor.fields.dtype.offset) >>> 2]
      const address = abi.anira_tensor_data(ptr, dtype)
      const elements = abi.anira_tensor_num_elements(ptr)
      const data =
        address === 0 ? null : typedView(heap(m).HEAPU8.buffer, address, dtype, elements)
      out.push({
        ptr,
        elements,
        dtype,
        data,
        f32: data instanceof Float32Array ? data : null,
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

// anira's record of a loaded model of a JavaScript engine, which the engine trampoline hands
// to every call (src/platform/emscripten/em_hooks.cpp, EngineModel): five 32-bit words, the
// row, the bytes, their size, the path, and the loaded model's serial, which no other loaded
// model of the module's life has.
const ENGINE_MODEL_ROW = 0
const ENGINE_MODEL_BYTES = 4
const ENGINE_MODEL_SIZE = 8
const ENGINE_MODEL_PATH = 12
const ENGINE_MODEL_SERIAL = 16

// What an engine's load receives of the loaded model's record (see `EngineLoadInfo`).
const engineLoadInfo = (m: AbiModule, record: number): EngineLoadInfo => {
  const word = (offset: number) => heap(m).HEAPU32[(record + offset) >>> 2]
  const bytes = word(ENGINE_MODEL_BYTES)
  const size = word(ENGINE_MODEL_SIZE)
  const path = word(ENGINE_MODEL_PATH)
  return {
    row: word(ENGINE_MODEL_ROW),
    modelBytes: bytes === 0 ? null : new Uint8Array(heap(m).HEAPU8.buffer, bytes, size),
    modelPath: path === 0 ? null : m.UTF8ToString(path),
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
