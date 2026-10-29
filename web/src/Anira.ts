import AniraWebFactory from '../wasm/AniraWeb'
import { AniraBindings, type AbiModule } from './abi/bindings'
import { anira_log_level } from './abi/enums'
import { scopeRegistry, type LogRecord } from './registry'

export type AniraOptions = {
  /** A pre-allocated shared memory; defaults to a fresh 8192-page (512 MiB) one. */
  memory?: WebAssembly.Memory
  /** The module binary, when it is not fetched from the package's URL. */
  wasmBinary?: ArrayBuffer
  /**
   * The packed ABI version (`ANIRA_MAKE_ABI_VERSION(major, minor)`) the caller's code was written
   * against; refused with an `Error` when `anira_check_abi` says this module does not serve it.
   */
  abiVersion?: number
}

// The module's two files, in a form bundlers see (new URL(literal, import.meta.url)) and hand to
// Emscripten's locateFile, which the Wasm Workers and the audio worklet bootstrap go through.
const jsUrl = () => new URL('../wasm/AniraWeb.js', import.meta.url).href
const wasmUrl = () => new URL('../wasm/AniraWeb.wasm', import.meta.url).href

/** The console sink: one line per record, at the console level of its anira level. */
export const consoleLogSink = (record: LogRecord): void => {
  const line = `[${record.group}] ${record.message}`
  switch (record.level) {
    case anira_log_level.ANIRA_LOG_ERROR:
      console.error(line)
      break
    case anira_log_level.ANIRA_LOG_WARNING:
      console.warn(line)
      break
    case anira_log_level.ANIRA_LOG_INFO:
      console.info(line)
      break
    default:
      console.debug(line)
  }
}

/**
 * The module on the page's main thread: its typed C ABI (`abi`), the instance (`module`) and its
 * shared memory. Every other thread of the page (the context's inference Workers, the audio
 * worklet) is started by anira from C on the same memory. One per page.
 */
export class Anira {
  readonly abi: AniraBindings
  readonly module: AbiModule
  readonly memory: WebAssembly.Memory

  private constructor(module: AbiModule, memory: WebAssembly.Memory) {
    this.module = module
    this.memory = memory
    this.abi = scopeRegistry().envOf(module).abi
  }

  /** Instantiate the module on a shared memory and boot the main thread's registry. */
  static async create(options: AniraOptions = {}): Promise<Anira> {
    const memory =
      options.memory ??
      new WebAssembly.Memory({ initial: 8192, maximum: 8192, shared: true })
    const module = (await AniraWebFactory({
      wasmMemory: memory,
      wasmBinary: options.wasmBinary,
      locateFile: (path: string) =>
        path.endsWith('.wasm') ? wasmUrl() : path.endsWith('.js') ? jsUrl() : path,
    })) as unknown as AbiModule
    const anira = new Anira(module, memory)
    if (options.abiVersion !== undefined) {
      try {
        anira.abi.anira_check_abi(options.abiVersion)
      } catch {
        throw new Error(
          `anira: the module (ABI 0x${anira.abi.anira_abi_version().toString(16)}) does not ` +
            `serve ABI 0x${(options.abiVersion >>> 0).toString(16)}`
        )
      }
    }
    await scopeRegistry().boot(module)
    return anira
  }

  /** The library's release identity (`anira_version_string`), e.g. "3.0.0-alpha.2". */
  get version(): string {
    return this.abi.anira_version_string() ?? ''
  }

  /**
   * Route the log of the main thread to `onRecord` (the console when omitted; `null` removes the
   * hook): the records logged here and those `drainLog` delivers here, the real-time records of
   * every thread among them (`anira_em_set_log_hook`). Records a Worker logs itself reach that
   * scope's registry, which writes them to the console unless the Worker's module sets
   * `scopeRegistry().logHook`. While a hook is set it replaces the module's platform sink
   * (stderr, printed as console.error), so each record reaches the console once, at its level;
   * `null` restores the platform sink.
   */
  setLogHook(onRecord: ((record: LogRecord) => void) | null = consoleLogSink): void {
    scopeRegistry().logHook = onRecord
    this.abi.anira_em_set_log_hook(onRecord !== null)
  }

  /**
   * Deliver the queued real-time records (`anira_drain_log`) to the sinks; the web has no drain
   * thread. Returns the number delivered.
   */
  drainLog(): number {
    return this.abi.anira_drain_log()
  }

  /** Drain the log on a main-thread interval; returns the function that stops it. */
  startLogPump(intervalMs = 20): () => void {
    const id = setInterval(() => this.drainLog(), intervalMs)
    return () => clearInterval(id)
  }
}
