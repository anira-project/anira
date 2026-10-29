import type { AniraBindings } from './abi/bindings'
import { anira_budget_kind, anira_warmup_mode } from './abi/enums'
import type { Fetcher, WithAbi } from './ModelConfig'

/** The fields of a Hard contract `Contract.hard` sets; the rest keep their defaults. */
export type HardContractOptions = {
  /** The smallest block a process call passes, in samples. */
  blockMin: number
  /** The largest; default `blockMin` (a fixed block, the render quantum). */
  blockMax?: number
  /** The host's sample rate. */
  rate: number
  /**
   * The inference budget in milliseconds (`ANIRA_BUDGET_EXPLICIT`): how long an inference may
   * take before its block is late. Required in this pre-release, whose prepare refuses the
   * measured default.
   */
  budgetMs: number
  /**
   * Fixed warmup iterations at prepare (`ANIRA_WARMUP_FIXED`); default 0, none (this pre-release
   * refuses the library's `ANIRA_WARMUP_UNTIL_STABLE` default).
   */
  warmup?: number
}

/**
 * A contract (`anira_contract`): the host's terms for a handler's prepare, from the same
 * `*.contract.json` a native host loads or built field by field. A Hard contract file may leave
 * the geometry to the host (the bundled `extras/models` ones do): `setGeometry` gives the block
 * and the rate the audio context runs at. Destroy it after the prepare that takes it.
 */
export class Contract {
  readonly abi: AniraBindings
  /** The `anira_contract*` the raw entries take. */
  readonly ptr: number

  private constructor(abi: AniraBindings, ptr: number) {
    this.abi = abi
    this.ptr = ptr
  }

  /** Fetch and parse a `*.contract.json` (`anira_contract_from_json`). */
  static async fromUrl(
    anira: WithAbi,
    url: string | URL,
    fetcher: Fetcher = globalThis.fetch
  ): Promise<Contract> {
    const contractUrl = new URL(
      url,
      (globalThis as { location?: { href: string } }).location?.href
    )
    const response = await fetcher(contractUrl)
    if (!response.ok) {
      throw new Error(`anira: fetching ${contractUrl.href} failed: ${response.status}`)
    }
    return Contract.fromJson(anira, await response.text())
  }

  /** Parse a contract's JSON text (`anira_contract_from_json`). */
  static fromJson(anira: WithAbi, json: string): Contract {
    return new Contract(anira.abi, anira.abi.anira_contract_from_json(json))
  }

  /** A Hard contract (`anira_contract_create_hard`) with the given fields. */
  static hard(anira: WithAbi, options: HardContractOptions): Contract {
    const { abi } = anira
    const contract = new Contract(
      abi,
      abi.anira_contract_create_hard(
        options.blockMin,
        options.blockMax ?? options.blockMin,
        options.rate
      )
    )
    try {
      abi.anira_contract_hard_set_budget(
        contract.ptr,
        anira_budget_kind.ANIRA_BUDGET_EXPLICIT,
        options.budgetMs
      )
      const warmup = options.warmup ?? 0
      abi.anira_contract_hard_set_warmup(
        contract.ptr,
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

  /** The geometry of a Hard contract (`anira_contract_hard_set_geometry`). */
  setGeometry(blockMin: number, blockMax: number, rate: number): this {
    this.abi.anira_contract_hard_set_geometry(this.ptr, blockMin, blockMax, rate)
    return this
  }

  destroy(): void {
    this.abi.anira_contract_destroy(this.ptr)
  }
}
