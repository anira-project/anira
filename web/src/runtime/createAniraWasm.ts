import AniraWebFactory, { type MainModule } from '../../wasm/AniraWeb'
import { AniraBindings, type AbiModule } from '../abi/bindings'
import { ANIRA_EM_HOOKS_VERSION } from '../abi/enums'
import { Heap } from './Heap'
import { HookRegistry } from './hooks'
import {
  allocInstanceStack,
  enterInstance,
  installThreadLocalBlock,
  MIN_INSTANCE_STACK_BYTES,
  type InstanceStack,
} from './stacks'

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

/**
 * One instance of the module on the shared memory, with its typed bindings (`abi`, generated
 * from the registry), heap views and hook registry.
 */
export type AniraModule = {
  module: AniraWasmInstance
  abi: AniraBindings
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
      // The worklet scope has no URL; its instance is handed the binary and fetches nothing,
      // though the glue still asks for the name.
      if (typeof URL === 'undefined') return path
      if (path.endsWith('.wasm')) {
        return getWasmUrl()
      }
      if (path.endsWith('.js')) {
        return getJsUrl()
      }
      return path
    },
  })) as unknown as AniraWasmInstance
  const abi = new AniraBindings(out as unknown as AbiModule)
  if (options.stack) {
    enterInstance(out, abi, options.stack)
  } else {
    // The main instance keeps its stack but needs a thread-local block of its own: the module
    // has no main(), so none was installed, and C++ exceptions trap without one. The region
    // lives as long as the instance.
    installThreadLocalBlock(abi, allocInstanceStack(abi, MIN_INSTANCE_STACK_BYTES))
  }
  const moduleHooksVersion = abi.anira_em_hooks_version()
  if (moduleHooksVersion !== ANIRA_EM_HOOKS_VERSION) {
    throw new Error(
      `anira: the module's hook object is version ${moduleHooksVersion}, this package's is ` +
        `${ANIRA_EM_HOOKS_VERSION}`
    )
  }
  if (
    options.abiVersion !== undefined &&
    !abi.anira_check_abi(options.abiVersion >>> 0)
  ) {
    throw new Error(
      `anira: the module (ABI 0x${abi.anira_abi_version().toString(16)}) does not ` +
        `serve ABI 0x${(options.abiVersion >>> 0).toString(16)}`
    )
  }
  const heap = new Heap(out, wasmMemory)
  hooks.attach(out, heap)
  return { module: out, abi, memory: wasmMemory, heap, hooks }
}
