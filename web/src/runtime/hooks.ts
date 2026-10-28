import { ANIRA_EM_HOOKS_VERSION, anira_phase, anira_status } from '../abi/enums'
import { anira_log_record, anira_stage_ctx } from '../abi/layout'
import type { AniraWasmInstance } from './createAniraWasm'
import type { Heap } from './Heap'

/**
 * A JavaScript stage (`anira_em_pipeline_add_js_stage`): one method per hook bit it was
 * registered with. Each runs on the instance that runs that phase of the C call: the worklet
 * for `preProcess`, `postProcess` and `reset` under a Hard contract, an inference Worker for
 * `beforeInference` and `afterInference`, the main thread for `init`, `prepare`, `unprepare`
 * and `release`. The pointers are into the shared memory: `ctx` an `anira_stage_ctx`
 * (`layout.ts`), `prepared` what this handler's `prepare` stored in its `outPrepared` slot. A
 * method returns an `anira_status` (nothing is `ANIRA_OK`); a method that throws fails its
 * call, caught in the C trampoline. A pre/post bit without its method runs anira's default
 * body (`anira_stage_default_pre_process` / `_post_process`).
 */
export interface JSStage {
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

/** What a JavaScript stage class is constructed with, on each instance that runs it. */
export type JSStageEnv = {
  module: AniraWasmInstance
  heap: Heap
}

/**
 * A JavaScript engine (`anira_em_pipeline_register_js_engine`), run on an inference Worker:
 * `process` for every inference (`ctx` an `anira_engine_ctx`, `layout.ts`), `reset` with
 * `ANIRA_EM_ENGINE_HOOK_RESET`, `query` with `ANIRA_EM_ENGINE_HOOK_QUERY` (on the main thread).
 */
export interface JSEngine {
  process(ctx: number, prepared: number): number | void
  reset?(ctx: number, prepared: number): void
  query?(initInfo: number, outAvailable: number): number | void
}

/** A log record as the JavaScript log hook delivers it (`anira_log_record`). */
export type AniraLogRecord = {
  level: number
  flags: number
  sequence: bigint
  group: string
  message: string
}

/** The stage context fields a JavaScript stage reads most, through `layout.ts`. */
export type StageContext = {
  phase: number
  engine: number
  provider: number
  numInputs: number
  numOutputs: number
  entry: number
}

export const readStageContext = (heap: Heap, ctx: number): StageContext => {
  const f = anira_stage_ctx.fields
  return {
    phase: heap.readU32(ctx + f.phase.offset),
    engine: heap.readU32(ctx + f.engine.offset),
    provider: heap.readU32(ctx + f.provider.offset),
    numInputs: heap.readU32(ctx + f.num_inputs.offset),
    numOutputs: heap.readU32(ctx + f.num_outputs.offset),
    entry: heap.readU32(ctx + f.entry.offset),
  }
}

export const readLogRecord = (heap: Heap, record: number): AniraLogRecord => {
  const f = anira_log_record.fields
  return {
    level: heap.readU32(record + f.level.offset),
    flags: heap.readU32(record + f.flags.offset),
    sequence: heap.readU64(record + f.sequence.offset),
    group: heap.cString(heap.readPtr(record + f.group.offset)),
    message: heap.cString(heap.readPtr(record + f.message.offset)),
  }
}

const status = (result: number | void): number => (result ?? anira_status.ANIRA_OK) | 0

/**
 * The per-instance registries behind `Module.anira`, keyed by `js_key`: what the C
 * trampolines of this instance reach. A key with no entry here is a stage or engine
 * registered on another instance only; the hook throws, which the trampoline reports and
 * turns into a failed call.
 */
export class HookRegistry {
  readonly stages = new Map<number, JSStage>()
  readonly engines = new Map<number, JSEngine>()
  /** The JavaScript log sink of this instance (`anira_em_set_log_hook`). */
  log: ((record: AniraLogRecord) => void) | null = null
  private module: AniraWasmInstance | null = null
  private heap: Heap | null = null

  attach(module: AniraWasmInstance, heap: Heap): void {
    this.module = module
    this.heap = heap
  }

  /** The hook object installed as `Module.anira` before instantiation. */
  hookObject() {
    return {
      version: ANIRA_EM_HOOKS_VERSION,
      stage: (key: number, phase: number, a: number, b: number) =>
        this.dispatchStage(key, phase, a, b),
      engine: (key: number, phase: number, a: number, b: number) =>
        this.dispatchEngine(key, phase, a, b),
      log: (record: number) => {
        if (this.log && this.heap) this.log(readLogRecord(this.heap, record))
      },
      complete: undefined,
    }
  }

  private dispatchStage(key: number, phase: number, a: number, b: number): number {
    const stage = this.stages.get(key)
    if (!stage)
      throw new Error(`no JavaScript stage under js_key ${key} on this instance`)
    const p = anira_phase
    switch (phase) {
      case p.ANIRA_PHASE_PRE_PROCESS:
        return stage.preProcess
          ? status(stage.preProcess(a, b))
          : this.module!._anira_stage_default_pre_process(a)
      case p.ANIRA_PHASE_POST_PROCESS:
        return stage.postProcess
          ? status(stage.postProcess(a, b))
          : this.module!._anira_stage_default_post_process(a)
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
      case p.ANIRA_PHASE_RELEASE:
        stage.release?.()
        this.stages.delete(key)
        return anira_status.ANIRA_OK
    }
    throw new Error(`JavaScript stage ${key}: phase ${phase} is not a stage phase`)
  }

  private dispatchEngine(key: number, phase: number, a: number, b: number): number {
    const engine = this.engines.get(key)
    if (!engine)
      throw new Error(`no JavaScript engine under js_key ${key} on this instance`)
    const p = anira_phase
    switch (phase) {
      case p.ANIRA_PHASE_INFERENCE:
        return status(engine.process(a, b))
      case p.ANIRA_PHASE_RESET:
        engine.reset?.(a, b)
        return anira_status.ANIRA_OK
      case p.ANIRA_PHASE_QUERY:
        return status(engine.query?.(a, b))
    }
    throw new Error(`JavaScript engine ${key}: phase ${phase} is not an engine phase`)
  }
}
