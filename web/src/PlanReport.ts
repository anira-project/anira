import { type AniraBindings, heap } from './abi/bindings'
import { AniraError } from './AniraError'
import {
  ANIRA_VARIANT_ANY,
  anira_engine,
  anira_provider,
  anira_status,
} from './abi/enums'

/**
 * What `PlanReport.find` looks for (`anira_plan_query`, as `anira::PlanQuery`): every field
 * names one property the plan must have; a field left out matches every plan.
 */
export type PlanQuery = {
  /** The variant index of the inference stage; any variant when left out. */
  variant?: number
  /** `anira_engine`; any engine when left out; `ANIRA_ENGINE_CUSTOM` beside `engineId`. */
  engine?: number
  /** The registered id of a custom engine (`engine` `ANIRA_ENGINE_CUSTOM`). */
  engineId?: string
  /** `anira_provider`; any provider when left out; `ANIRA_PROVIDER_CUSTOM` beside `providerId`. */
  provider?: number
  /** The name of a custom provider (`provider` `ANIRA_PROVIDER_CUSTOM`). */
  providerId?: string
}

/** One plan of a report (`anira_plan_info`): the engine and provider it runs on, its budget. */
export type PlanInfo = {
  /** The variant of the inference stage; 0 in this pre-release. */
  variant: number
  /** `anira_engine`; `ANIRA_ENGINE_CUSTOM` beside `engineId`. */
  engine: number
  /** `anira_provider`; `ANIRA_PROVIDER_CUSTOM` beside `providerId`. */
  provider: number
  /** The registered name of a custom engine, null for a built-in one. */
  engineId: string | null
  /** The per-inference budget of this plan, in milliseconds. */
  budgetMs: number
  /** A custom engine's `ANIRA_ENGINE_FLAG_*` promises; 0 for a built-in engine. */
  engineFlags: number
  /** The name of the plan's custom provider, null for one the enum names. */
  providerId: string | null
}

/** One slot of a plan (`anira_plan_slot`): the edge its tensor takes into the engine. */
export type PlanSlot = {
  slot: number
  /** True for an input slot. */
  isInput: boolean
  /** `anira_domain` of the host end and of the model end. */
  domainIn: number
  domainOut: number
  /** `anira_edge_class`: the cost class of the edge taken. */
  edgeClass: number
  allocateClass: number
  /** `anira_wait_strategy`. */
  waitStrategy: number
  recipe: string | null
  reason: string | null
  /** `anira_role`. */
  role: number
  /** `anira_binding`. */
  binding: number
}

/** One extension a plan consumes (`anira_plan_ext`). */
export type PlanExtension = {
  /** Its index in the plan's extension list. */
  index: number
  host: string | null
  /** The extension kind, its registered reverse-URI name. */
  kind: string | null
  consumer: string | null
}

// The wasm32 layouts of the three Tier-2 rows (anira/abi/handler.h): 4-byte pointers, the
// double of anira_plan_info at 24. The rows are enumerated at these strides (the entries take
// the caller's element size and write the fields the stride covers).
const PLAN_INFO = {
  size: 48,
  variant: 4,
  engine: 8,
  provider: 12,
  engineId: 16,
  budgetMs: 24,
}
const PLAN_INFO_TAIL = { engineFlags: 32, providerId: 40 }
const PLAN_SLOT = {
  size: 48,
  slot: 4,
  isInput: 8,
  domainIn: 12,
  domainOut: 16,
  edgeClass: 20,
  allocateClass: 24,
  waitStrategy: 28,
  recipe: 32,
  reason: 36,
  role: 40,
  binding: 44,
}
const PLAN_EXT = { size: 20, index: 4, host: 8, kind: 12, consumer: 16 }
// anira_plan_query and the anira_backend_id it holds at 8.
const PLAN_QUERY = { size: 28, variant: 4, backend: 8 }
const BACKEND_ID = { size: 20, engine: 4, provider: 8, engineId: 12, providerId: 16 }

/**
 * The plan report of a handler's last prepare (`anira_plan_report`): one plan per model entry
 * and candidate that matched, in plan order, with its slots and the extensions it consumes;
 * what `InferenceHandler.prepare` returns, as `anira::PlanReport` of `anira.hpp`. A view: it
 * reads the handler's report at every call, valid until the handler's next prepare or its
 * destroy. A plan is a dense index below `numPlans()`, what `InferenceHandler.setPlan` takes.
 */
export class PlanReport {
  readonly abi: AniraBindings
  private readonly report: number

  constructor(abi: AniraBindings, report: number) {
    this.abi = abi
    this.report = report
  }

  /** The number of plans (`anira_plan_report_num_plans`). */
  numPlans(): number {
    return this.abi.anira_plan_report_num_plans(this.report)
  }

  /** One row per plan, in plan order (`anira_plan_report_plans`). */
  plans(): PlanInfo[] {
    return this.rows(
      PLAN_INFO.size,
      (count, out) =>
        this.abi.anira_plan_report_plans(this.report, PLAN_INFO.size, count, out),
      (m, row) => ({
        variant: heap(m).HEAPU32[(row + PLAN_INFO.variant) >>> 2],
        engine: heap(m).HEAPU32[(row + PLAN_INFO.engine) >>> 2],
        provider: heap(m).HEAPU32[(row + PLAN_INFO.provider) >>> 2],
        engineId: this.text(heap(m).HEAPU32[(row + PLAN_INFO.engineId) >>> 2]),
        budgetMs: heap(m).HEAPF64[(row + PLAN_INFO.budgetMs) >>> 3],
        engineFlags: heap(m).HEAPU32[(row + PLAN_INFO_TAIL.engineFlags) >>> 2],
        providerId: this.text(heap(m).HEAPU32[(row + PLAN_INFO_TAIL.providerId) >>> 2]),
      })
    )
  }

  /** The input (`inputs` true) or output slots of one plan, in tensor order. */
  slots(plan: number, inputs: boolean): PlanSlot[] {
    const u32 = (m: AniraBindings['module'], row: number, offset: number) =>
      heap(m).HEAPU32[(row + offset) >>> 2]
    return this.rows(
      PLAN_SLOT.size,
      (count, out) =>
        this.abi.anira_plan_report_slots(
          this.report,
          plan,
          inputs,
          PLAN_SLOT.size,
          count,
          out
        ),
      (m, row) => ({
        slot: u32(m, row, PLAN_SLOT.slot),
        isInput: u32(m, row, PLAN_SLOT.isInput) !== 0,
        domainIn: u32(m, row, PLAN_SLOT.domainIn),
        domainOut: u32(m, row, PLAN_SLOT.domainOut),
        edgeClass: u32(m, row, PLAN_SLOT.edgeClass),
        allocateClass: u32(m, row, PLAN_SLOT.allocateClass),
        waitStrategy: u32(m, row, PLAN_SLOT.waitStrategy),
        recipe: this.text(u32(m, row, PLAN_SLOT.recipe)),
        reason: this.text(u32(m, row, PLAN_SLOT.reason)),
        role: u32(m, row, PLAN_SLOT.role),
        binding: u32(m, row, PLAN_SLOT.binding),
      })
    )
  }

  /** The extensions one plan consumes (`anira_plan_report_exts`). */
  extensions(plan: number): PlanExtension[] {
    return this.rows(
      PLAN_EXT.size,
      (count, out) =>
        this.abi.anira_plan_report_exts(this.report, plan, PLAN_EXT.size, count, out),
      (m, row) => ({
        index: heap(m).HEAPU32[(row + PLAN_EXT.index) >>> 2],
        host: this.text(heap(m).HEAPU32[(row + PLAN_EXT.host) >>> 2]),
        kind: this.text(heap(m).HEAPU32[(row + PLAN_EXT.kind) >>> 2]),
        consumer: this.text(heap(m).HEAPU32[(row + PLAN_EXT.consumer) >>> 2]),
      })
    )
  }

  /**
   * The first plan, in plan order, that has every property the query names
   * (`anira_plan_report_find`, as `anira::PlanReport::find`): the index `InferenceHandler.setPlan`
   * takes, or null when no plan has them (where C++ returns an empty optional). Allocates
   * nothing on the heap. A query against the pair rule (`ANIRA_ENGINE_CUSTOM` without
   * `engineId`, an id beside a built-in value) throws an AniraError.
   */
  find(query: PlanQuery = {}): number | null {
    if (this.report === 0) {
      throw new AniraError(
        anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
        'anira: the plan report of a handler that is not prepared'
      )
    }
    const m = this.abi.module
    const sp = m.stackSave()
    try {
      const engineId =
        query.engineId === undefined ? 0 : m.stringToUTF8OnStack(query.engineId)
      const providerId =
        query.providerId === undefined ? 0 : m.stringToUTF8OnStack(query.providerId)
      const q = m.stackAlloc(PLAN_QUERY.size)
      const b = q + PLAN_QUERY.backend
      const u32 = heap(m).HEAPU32
      u32[q >>> 2] = PLAN_QUERY.size
      u32[(q + PLAN_QUERY.variant) >>> 2] = query.variant ?? ANIRA_VARIANT_ANY
      u32[b >>> 2] = BACKEND_ID.size
      u32[(b + BACKEND_ID.engine) >>> 2] = query.engine ?? anira_engine.ANIRA_ENGINE_NONE
      u32[(b + BACKEND_ID.provider) >>> 2] =
        query.provider ?? anira_provider.ANIRA_PROVIDER_NONE
      u32[(b + BACKEND_ID.engineId) >>> 2] = engineId
      u32[(b + BACKEND_ID.providerId) >>> 2] = providerId
      try {
        return this.abi.anira_plan_report_find(this.report, q)
      } catch (error) {
        if (
          error instanceof AniraError &&
          error.status === anira_status.ANIRA_ERROR_NOT_SUPPORTED
        ) {
          return null
        }
        throw error
      }
    } finally {
      m.stackRestore(sp)
    }
  }

  /** The `anira_plan_report*` the raw entries take (0 for a handler that is not prepared). */
  native(): number {
    return this.report
  }

  private text(ptr: number): string | null {
    return ptr === 0 ? null : this.abi.module.UTF8ToString(ptr)
  }

  // An enumeration: the count first (out NULL), then the rows at `stride`, read before the
  // buffer is freed.
  private rows<T>(
    stride: number,
    call: (count: number, out: number) => number,
    read: (m: AniraBindings['module'], row: number) => T
  ): T[] {
    if (this.report === 0) {
      throw new AniraError(
        anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
        'anira: the plan report of a handler that is not prepared'
      )
    }
    const m = this.abi.module
    const count = m._malloc(4)
    if (count === 0) throw new RangeError('anira: out of memory for a plan report')
    try {
      heap(m).HEAPU32[count >>> 2] = 0
      call(count, 0)
      const n = heap(m).HEAPU32[count >>> 2]
      if (n === 0) return []
      const out = m._malloc(n * stride)
      if (out === 0) throw new RangeError('anira: out of memory for a plan report')
      try {
        heap(m).HEAPU8.fill(0, out, out + n * stride)
        for (let i = 0; i < n; i++) heap(m).HEAPU32[(out + i * stride) >>> 2] = stride
        call(count, out)
        const written = Math.min(heap(m).HEAPU32[count >>> 2], n)
        const result: T[] = []
        for (let i = 0; i < written; i++) result.push(read(m, out + i * stride))
        return result
      } finally {
        m._free(out)
      }
    } finally {
      m._free(count)
    }
  }
}
