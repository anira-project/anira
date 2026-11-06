import type { AniraBindings } from './abi/bindings'
import { anira_budget_kind, anira_edge_class, anira_warmup_mode } from './abi/enums'
import type { Fetcher, WithAbi } from './ModelConfig'

/**
 * The fields of a Hard contract `ContractHandle.hard` sets, the web's `anira::Hard`: the same
 * fields (camelCase) and the same defaults (a measured budget, the `UNTIL_STABLE` warm-up), with
 * the budget in milliseconds (`std::chrono` on the C++ side).
 */
export type HardContractOptions = {
  /**
   * The smallest block a process call passes, in elements of the model's anchor tensor (samples
   * for an audio anchor).
   */
  hostBlockMin: number
  /** The largest; default `hostBlockMin` (a fixed block, the render quantum). */
  hostBlockMax?: number
  /**
   * The anchor's elements per second the host runs at (the context's sample rate for an audio
   * anchor).
   */
  hostRate: number
  /**
   * The inference budget in milliseconds (`ANIRA_BUDGET_EXPLICIT`, `anira::Hard::budget` from a
   * duration): how long an inference may take before its block is late, a reproducible latency.
   * Left out (the default), the budget the warm-up pass measures (`ANIRA_BUDGET_MEASURED`): the
   * slowest plan's figure, a latency that depends on the machine. The warm-up pass also measures
   * the inference time; a measured budget is the maximum of its last four inferences; under an
   * explicit budget the timings are only reported.
   */
  budgetMs?: number
  /**
   * The warm-up pass at prepare (`anira_warmup_mode`, `anira::Hard::warmup`): `ANIRA_WARMUP_UNTIL_STABLE`
   * (the default) until the last four inference times agree, `ANIRA_WARMUP_FIXED` with
   * `warmupIterations`, `ANIRA_WARMUP_NONE` (an explicit budget only). A prepare that warms up needs
   * the pool's Workers running: `await context.ready()` first.
   */
  warmup?: number
  /** `ANIRA_WARMUP_FIXED`'s count (`anira::Hard::warmup_iterations`); default 0. */
  warmupIterations?: number
  /**
   * The ring dtype per Streamed tensor by canonical name (`anira::Hard::ring_dtypes`,
   * `anira_contract_hard_set_ring_dtype`; the spec's dtype for every tensor not named): the
   * element type of the host's samples, set only where it differs from the model's, and then a
   * processor that converts goes with it (nothing in anira converts). An audio node streams
   * float32: a model of another dtype behind one names `ANIRA_DTYPE_F32` here.
   */
  ringDtypes?: Record<string, number>
  /**
   * The worst edge class a plan may take for any slot (`anira::Hard::edge_cost`,
   * `anira_contract_set_edge_cost`; an `anira_edge_class` value): prepare drops a plan above
   * it. A class of the cost scale is a limit: `ANIRA_EDGE_DEDICATED_TRANSFER` admits every known
   * edge, `ANIRA_EDGE_UNIFIED_TRANSFER` crossings between the CPU and a device within unified
   * memory only (no bus), `ANIRA_EDGE_COPY` no crossing between domains, `ANIRA_EDGE_ZERO_COPY`
   * no copy at all, and every limit refuses an `ANIRA_EDGE_UNKNOWN` edge (a device provider's
   * on the web, which exposes no memory kind). Left out (the default): no limit, every edge
   * admitted, an unknown one included.
   */
  edgeCost?: number
}

/**
 * A contract (`anira_contract`), the web twin of `anira::ContractHandle`: the host's terms for a
 * handler's prepare, from the same `*.contract.json` a native host loads or built field by
 * field. A Hard contract file may leave the host blocks to the host (the bundled `extras/models`
 * ones do): `hardHostBlocks` gives the host blocks and the host rate the audio context runs at.
 * Where the web differs: `fromUrl` fetches the file (`from_file` reads a path), and the handle is
 * destroyed with `destroy()` after the prepare that takes it (no RAII).
 */
export class ContractHandle {
  readonly abi: AniraBindings
  private readonly contract: number

  private constructor(abi: AniraBindings, contract: number) {
    this.abi = abi
    this.contract = contract
  }

  /** Fetch and parse a `*.contract.json` (`anira_contract_from_json`). */
  static async fromUrl(
    anira: WithAbi,
    url: string | URL,
    fetcher: Fetcher = globalThis.fetch
  ): Promise<ContractHandle> {
    const contractUrl = new URL(
      url,
      (globalThis as { location?: { href: string } }).location?.href
    )
    const response = await fetcher(contractUrl)
    if (!response.ok) {
      throw new Error(`anira: fetching ${contractUrl.href} failed: ${response.status}`)
    }
    return ContractHandle.fromJson(anira, await response.text())
  }

  /** Parse a contract's JSON text (`anira_contract_from_json`). */
  static fromJson(anira: WithAbi, json: string): ContractHandle {
    return new ContractHandle(anira.abi, anira.abi.anira_contract_from_json(json))
  }

  /** A Hard contract (`anira_contract_create_hard`) with the given fields. */
  static hard(anira: WithAbi, options: HardContractOptions): ContractHandle {
    const { abi } = anira
    const contract = new ContractHandle(
      abi,
      abi.anira_contract_create_hard(
        options.hostBlockMin,
        options.hostBlockMax ?? options.hostBlockMin,
        options.hostRate
      )
    )
    try {
      // The library's defaults where the options leave a field out, as anira::Hard's.
      if (options.budgetMs === undefined) {
        contract.hardBudget(anira_budget_kind.ANIRA_BUDGET_MEASURED)
      } else {
        contract.hardBudget(anira_budget_kind.ANIRA_BUDGET_EXPLICIT, options.budgetMs)
      }
      contract.hardWarmup(
        options.warmup ?? anira_warmup_mode.ANIRA_WARMUP_UNTIL_STABLE,
        options.warmupIterations ?? 0
      )
      for (const [canonical, dtype] of Object.entries(options.ringDtypes ?? {})) {
        contract.hardRingDtype(canonical, dtype)
      }
      if (options.edgeCost !== undefined) contract.edgeCost(options.edgeCost)
    } catch (error) {
      contract.destroy()
      throw error
    }
    return contract
  }

  /**
   * Patch a Hard contract's host blocks (`anira_contract_hard_set_host_blocks`): the smallest and
   * largest block in elements of the model's anchor tensor and the host rate, the anchor's
   * elements per second (a block may be fractional). What the host provides carries the host
   * prefix (`hostBlockMin`, `hostBlockMax`, `hostRate`).
   */
  hardHostBlocks(hostBlockMin: number, hostBlockMax: number, hostRate: number): this {
    this.abi.anira_contract_hard_set_host_blocks(
      this.contract,
      hostBlockMin,
      hostBlockMax,
      hostRate
    )
    return this
  }

  /**
   * The per-inference budget of this Hard contract (`anira_contract_hard_set_budget`,
   * `anira::ContractHandle::hard_budget`): `ANIRA_BUDGET_MEASURED`, what the warm-up pass
   * measures, or `ANIRA_BUDGET_EXPLICIT` with `explicitMs`.
   */
  hardBudget(kind: number, explicitMs = 0): this {
    this.abi.anira_contract_hard_set_budget(this.contract, kind, explicitMs)
    return this
  }

  /**
   * The warm-up pass of this Hard contract (`anira_contract_hard_set_warmup`,
   * `anira::ContractHandle::hard_warmup`): an `anira_warmup_mode`, with `ANIRA_WARMUP_FIXED`'s
   * count.
   */
  hardWarmup(mode: number, iterations = 0): this {
    this.abi.anira_contract_hard_set_warmup(this.contract, mode, iterations)
    return this
  }

  /**
   * The ring dtype of one tensor under this Hard contract, by canonical name
   * (`anira_contract_hard_set_ring_dtype`, `anira::ContractHandle::hard_ring_dtype`): the
   * element type the host's samples of that tensor carry and its ring holds as is; the spec's
   * dtype for every tensor never set. Nothing converts: a name that is no Streamed tensor is
   * `ANIRA_ERROR_CONFIG` at prepare, and so is a ring dtype that differs from the spec's dtype
   * unless a processor of the pipeline fills the phase that moves that ring and converts.
   */
  hardRingDtype(canonical: string, dtype: number): this {
    this.abi.anira_contract_hard_set_ring_dtype(this.contract, canonical, dtype)
    return this
  }

  /**
   * The worst edge class a plan may take for any slot, on either contract kind
   * (`anira_contract_set_edge_cost`, `anira::ContractHandle::edge_cost`; an `anira_edge_class`
   * value): prepare drops a plan with a slot above it, any limit a plan with an
   * `ANIRA_EDGE_UNKNOWN` edge, and refuses with `ANIRA_ERROR_EDGE_UNREACHABLE` when no plan
   * remains. `undefined` restores no limit (`ANIRA_EDGE_UNKNOWN` to the C entry).
   */
  edgeCost(worstAllowed: number | undefined): this {
    this.abi.anira_contract_set_edge_cost(
      this.contract,
      worstAllowed ?? anira_edge_class.ANIRA_EDGE_UNKNOWN
    )
    return this
  }

  /** The `anira_contract*` the raw entries take. */
  native(): number {
    return this.contract
  }

  destroy(): void {
    this.abi.anira_contract_destroy(this.contract)
  }
}
