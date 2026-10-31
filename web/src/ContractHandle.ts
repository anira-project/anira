import type { AniraBindings } from './abi/bindings'
import { anira_budget_kind, anira_warmup_mode } from './abi/enums'
import type { Fetcher, WithAbi } from './ModelConfig'

/**
 * The fields of a Hard contract `ContractHandle.hard` sets, the web's `anira::Hard`: the same
 * fields (camelCase), with the budget in milliseconds (`std::chrono` on the C++ side) and
 * required, the warm-up fixed or none; the rest keep the library's defaults.
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
   * The inference budget in milliseconds (`ANIRA_BUDGET_EXPLICIT`): how long an inference may
   * take before its block is late. Required in this pre-release, whose prepare refuses the
   * measured default.
   */
  budgetMs: number
  /**
   * Fixed warm-up iterations at prepare (`ANIRA_WARMUP_FIXED`, `anira::Hard::warmup_iterations`);
   * default 0, none (this pre-release refuses the library's `ANIRA_WARMUP_UNTIL_STABLE`
   * default).
   */
  warmupIterations?: number
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
      abi.anira_contract_hard_set_budget(
        contract.contract,
        anira_budget_kind.ANIRA_BUDGET_EXPLICIT,
        options.budgetMs
      )
      const warmup = options.warmupIterations ?? 0
      abi.anira_contract_hard_set_warmup(
        contract.contract,
        warmup > 0
          ? anira_warmup_mode.ANIRA_WARMUP_FIXED
          : anira_warmup_mode.ANIRA_WARMUP_NONE,
        warmup
      )
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

  /** The `anira_contract*` the raw entries take. */
  native(): number {
    return this.contract
  }

  destroy(): void {
    this.abi.anira_contract_destroy(this.contract)
  }
}
