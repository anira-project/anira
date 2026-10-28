import AniraWebFactory, { type MainModule } from '../../wasm/AniraWeb'
import { ANIRA_EM_HOOKS_VERSION, anira_status } from '../abi/enums'
import { Heap } from './Heap'
import { HookRegistry } from './hooks'
import { enterInstance, type InstanceStack } from './stacks'

// Lazy-evaluated so the module can be imported in AudioWorkletGlobalScope
// where URL may not be available. The URLs are only needed on the main thread.
let _jsUrl: string | undefined
let _wasmUrl: string | undefined

const getJsUrl = () =>
  (_jsUrl ??= new URL('../../wasm/AniraWeb.js', import.meta.url).href)
const getWasmUrl = () =>
  (_wasmUrl ??= new URL('../../wasm/AniraWeb.wasm', import.meta.url).href)

export { getWasmUrl }

/**
 * The emitted module: the C ABI's `_anira_*` exports (`web/src/abi/exports_wasm.txt`),
 * `_malloc` / `_free` and the runtime methods the package reads the heap with.
 */
export type AniraWasmInstance = Omit<MainModule, 'HEAPF32' | 'HEAPU32'> & {
  HEAPF32: Float32Array
  HEAPU32: Uint32Array
}

/** One instance of the module on the shared memory, with its heap views and hook registry. */
export type AniraModule = {
  module: AniraWasmInstance
  memory: WebAssembly.Memory
  heap: Heap
  hooks: HookRegistry
}

export type CreateAniraWasmOptions = {
  /** The module binary; required where `fetch` is unavailable (the audio worklet). */
  wasmBinary?: ArrayBuffer
  /**
   * The stack and thread-local block of an instance that is not the main one (a Worker's,
   * the worklet's), from `allocInstanceStack` on the main instance. Entered before the first
   * C call of the instance (`anira_em_init_instance`).
   */
  stack?: InstanceStack
  /**
   * The packed ABI version (`ANIRA_MAKE_ABI_VERSION(major, minor)`) the caller's raw calls were
   * written against; refused with an `Error` when `_anira_check_abi` says this module does not
   * serve it. Omitted: no check (the package and its module ship together).
   */
  abiVersion?: number
}

/**
 * Instantiate the module on `wasmMemory` and install its hook object, `Module.anira`
 * (`anira/anira_em.h`): the stage, engine and log hooks dispatch into the returned
 * registry, keyed by `js_key`. Every instance (main, each Worker, the worklet) has its own
 * registry; the C trampolines call the hook object of the instance that runs the C call.
 */
export const createAniraWasm = async (
  wasmMemory: WebAssembly.Memory,
  options: CreateAniraWasmOptions = {}
): Promise<AniraModule> => {
  const hooks = new HookRegistry()
  const out = (await AniraWebFactory({
    wasmBinary: options.wasmBinary,
    wasmMemory,
    anira: hooks.hookObject(),
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        return getWasmUrl()
      }
      if (path.endsWith('.js')) {
        return getJsUrl()
      }
      return path
    },
  })) as unknown as AniraWasmInstance
  if (options.stack) {
    enterInstance(out, options.stack)
  }
  const moduleHooksVersion = out._anira_em_hooks_version() >>> 0
  if (moduleHooksVersion !== ANIRA_EM_HOOKS_VERSION) {
    throw new Error(
      `anira: the module's hook object is version ${moduleHooksVersion}, this package's is ` +
        `${ANIRA_EM_HOOKS_VERSION}`
    )
  }
  if (
    options.abiVersion !== undefined &&
    out._anira_check_abi(options.abiVersion >>> 0) !== anira_status.ANIRA_OK
  ) {
    throw new Error(
      `anira: the module (ABI 0x${(out._anira_abi_version() >>> 0).toString(16)}) does not ` +
        `serve ABI 0x${(options.abiVersion >>> 0).toString(16)}`
    )
  }
  const heap = new Heap(out, wasmMemory)
  hooks.attach(out, heap)
  return { module: out, memory: wasmMemory, heap, hooks }
}
