import AniraWebFactory from '../wasm/AniraWeb'
import { AniraBindings, type AbiModule } from './abi/bindings'
import { anira_log_level, anira_status } from './abi/enums'
import { AniraError } from './AniraError'
import { scopeRegistry, type LogRecord } from './registry'

/** The module's shared memory, in bytes (rounded up to 64 KiB WebAssembly pages). */
export type AniraMemory = {
  /** What it starts with; default and least 256 MiB (the module's own data and stack in it). */
  initial?: number
  /**
   * What it may grow to; default 2 GiB, the most the module grows to (2 GiB, the module's
   * MAXIMUM_MEMORY). A shared memory cannot move: the browser reserves the maximum as address
   * space when it is created, and commits the pages as they are touched. `maximum === initial`
   * fixes the size (no growth).
   */
  maximum?: number
}

export type AniraOptions = {
  /** The memory's sizes; default 256 MiB growing to 2 GiB. */
  memory?: AniraMemory
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

const PAGE = 65536
const MIB = 1024 * 1024
// The module's INITIAL_MEMORY and MAXIMUM_MEMORY (cmake/build-wasm.cmake).
const MODULE_INITIAL = 256 * MIB
const MODULE_MAXIMUM = 2048 * MIB

/**
 * The module's shared memory at `sizes`. Refused with an AniraError: an initial size below the
 * module's, a maximum below the initial or above the module's; and a maximum the browser cannot
 * reserve (a 32-bit or a constrained device), naming it and the way to a smaller one.
 */
const sharedMemory = (sizes: AniraMemory = {}): WebAssembly.Memory => {
  const initial = Math.ceil((sizes.initial ?? MODULE_INITIAL) / PAGE)
  const maximum = Math.ceil((sizes.maximum ?? MODULE_MAXIMUM) / PAGE)
  const mib = (pages: number) => `${(pages * PAGE) / MIB} MiB`
  if (
    initial * PAGE < MODULE_INITIAL ||
    maximum < initial ||
    maximum * PAGE > MODULE_MAXIMUM
  ) {
    throw new AniraError(
      anira_status.ANIRA_ERROR_INVALID_ARGUMENT,
      `anira: Anira.create: memory of ${mib(initial)} growing to ${mib(maximum)}: the initial ` +
        `size must be ${mib(MODULE_INITIAL / PAGE)} or more, and the maximum between it and ` +
        `${mib(MODULE_MAXIMUM / PAGE)}`
    )
  }
  try {
    return new WebAssembly.Memory({ initial, maximum, shared: true })
  } catch (error) {
    throw new AniraError(
      anira_status.ANIRA_ERROR_OUT_OF_MEMORY,
      `anira: Anira.create: the browser could not reserve a shared memory of ${mib(initial)} ` +
        `growing to ${mib(maximum)} (${error}); pass a smaller maximum, ` +
        `Anira.create({ memory: { maximum } }), down to a fixed size, maximum = initial`
    )
  }
}

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

  /**
   * Instantiate the module on a shared memory (`options.memory`: 256 MiB growing to 2 GiB by
   * default) and boot the main thread's registry.
   */
  static async create(options: AniraOptions = {}): Promise<Anira> {
    const memory = sharedMemory(options.memory)
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
   * `scopeRegistry().logHook`. While a hook is set it replaces the module's platform sink (the
   * console by level), so each record reaches the console once; `null` restores the platform
   * sink.
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
