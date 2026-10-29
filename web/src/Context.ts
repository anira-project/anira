import type { AniraBindings } from './abi/bindings'
import { ANIRA_THREADS_AUTO, anira_log_drain, anira_wait_strategy } from './abi/enums'
import type { WithAbi } from './ModelConfig'

/** The fields of a context `Context.create` sets; the rest keep their defaults. */
export type ContextOptions = {
  /**
   * The pool: this many inference threads, which anira starts as Wasm Workers; `'auto'` for half
   * the cores (`ANIRA_THREADS_AUTO`), 0 for none (a host that drives inference itself). Default 1.
   */
  threads?: number | 'auto'
  /**
   * The ES module every inference Worker imports before its loop: the JavaScript engines (and
   * the Worker-side phases of stages) it defines with `defineEngine` / `defineStage` from
   * `@anira-project/anira/scope`. Omitted: none (the module's built-in engines need none).
   */
  workerModule?: string | URL
  /** `anira_log_level`; default the library's (warnings). */
  logLevel?: number
  /** A context config's JSON text (`anira_context_config_from_json`) the fields above amend. */
  json?: string
}

const pageUrl = (): string | undefined =>
  (globalThis as { location?: { href: string } }).location?.href

/**
 * A context (`anira_context`): the core the handlers run on and its pool of inference threads,
 * Wasm Workers on the web. The worker module is the one web-only field of a context config. The
 * log is drained by the host (`ANIRA_LOG_DRAIN_MANUAL`: `anira.startLogPump()`).
 */
export class Context {
  readonly abi: AniraBindings
  /** The `anira_context*` the raw entries take. */
  readonly ptr: number

  private constructor(abi: AniraBindings, ptr: number) {
    this.abi = abi
    this.ptr = ptr
  }

  static create(anira: WithAbi, options: ContextOptions = {}): Context {
    const { abi } = anira
    const config =
      options.json === undefined
        ? abi.anira_context_config_create()
        : abi.anira_context_config_from_json(options.json)
    try {
      const threads = options.threads ?? (options.json === undefined ? 1 : undefined)
      if (threads !== undefined) {
        abi.anira_context_config_set_threads(
          config,
          threads === 'auto' ? ANIRA_THREADS_AUTO : threads,
          anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF
        )
      }
      abi.anira_context_config_set_log_drain(
        config,
        anira_log_drain.ANIRA_LOG_DRAIN_MANUAL,
        10
      )
      if (options.logLevel !== undefined) {
        abi.anira_context_config_set_log_level(config, options.logLevel)
      }
      if (options.workerModule !== undefined) {
        abi.anira_em_context_config_set_worker_module(
          config,
          new URL(options.workerModule, pageUrl()).href
        )
      }
      return new Context(abi, abi.anira_context_create(config))
    } finally {
      abi.anira_context_config_destroy(config)
    }
  }

  /** The pool's size (`anira_num_inference_threads`). */
  get threads(): number {
    return this.abi.anira_num_inference_threads()
  }

  /**
   * Resolve once every inference thread of the pool is inside its loop: a Wasm Worker starts only
   * after this thread yields and imports its module first. Await it before a render that waits
   * for its inferences (an `OfflineAudioContext`); a live context just misses its first quanta.
   */
  async ready(timeoutMs = 10000): Promise<void> {
    const deadline = performance.now() + timeoutMs
    while (this.abi.anira_em_inference_threads_in_loop() < this.threads) {
      if (performance.now() > deadline) {
        throw new Error('anira: the inference Workers did not start in time')
      }
      await new Promise((resolve) => setTimeout(resolve, 5))
    }
  }

  /** Destroy the context; the last one stops the pool. */
  destroy(): void {
    this.abi.anira_context_destroy(this.ptr)
  }
}
