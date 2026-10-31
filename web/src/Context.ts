import type { AniraBindings } from './abi/bindings'
import { ANIRA_THREADS_AUTO, anira_log_drain, anira_wait_strategy } from './abi/enums'
import type { WithAbi } from './ModelConfig'

/** The fields of a context `Context.create` sets; the rest keep their defaults. */
export type ContextOptions = {
  /**
   * The pool: this many inference threads, which anira starts as Wasm Workers; `'auto'` for half
   * the cores (`ANIRA_THREADS_AUTO`). Default the context config's, as natively:
   * `ANIRA_THREADS_AUTO` (or the `json`'s value).
   */
  threads?: number | 'auto'
  /**
   * The ES module every inference Worker imports before its loop: the JavaScript engines (and
   * the Worker-side phases of processors) it defines with `defineEngine` / `defineProcessor` from
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
 * A context (`anira_context`), the web twin of `anira::Context`: the core the handlers run on
 * and its pool of inference threads, Wasm Workers on the web. Where the web differs: the
 * options are the context config's fields in one object (`anira::ContextConfig` on the C++
 * side), the worker module is a web-only field, the log is drained by the host
 * (`ANIRA_LOG_DRAIN_MANUAL`: `anira.startLogPump()`), `ready()` waits for the Workers, which
 * start only once the main thread yields, and the context is destroyed with `destroy()`.
 */
export class Context {
  readonly abi: AniraBindings
  private readonly context: number

  private constructor(abi: AniraBindings, context: number) {
    this.abi = abi
    this.context = context
  }

  static create(anira: WithAbi, options: ContextOptions = {}): Context {
    const { abi } = anira
    const config =
      options.json === undefined
        ? abi.anira_context_config_create()
        : abi.anira_context_config_from_json(options.json)
    try {
      // Unset, the config's own: ANIRA_THREADS_AUTO, the native default (one config for both).
      const threads = options.threads
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

  /**
   * The size of the core's inference thread pool (`anira_num_inference_threads`; the free
   * function `anira::num_inference_threads()` in C++, a member here, where the module is
   * reached through a context).
   */
  numInferenceThreads(): number {
    return this.abi.anira_num_inference_threads()
  }

  /**
   * Resolve once every inference thread of the pool is inside its loop: a Wasm Worker starts only
   * after this thread yields and imports its module first. Await it before a render that waits
   * for its inferences (an `OfflineAudioContext`); a live context just misses its first quanta.
   */
  async ready(timeoutMs = 10000): Promise<void> {
    const deadline = performance.now() + timeoutMs
    while (this.abi.anira_em_inference_threads_in_loop() < this.numInferenceThreads()) {
      if (performance.now() > deadline) {
        throw new Error('anira: the inference Workers did not start in time')
      }
      await new Promise((resolve) => setTimeout(resolve, 5))
    }
  }

  /** The `anira_context*` the raw entries take. */
  native(): number {
    return this.context
  }

  /** Destroy the context; the last one stops the pool. */
  destroy(): void {
    this.abi.anira_context_destroy(this.context)
  }
}
