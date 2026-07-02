import AniraWebFactory, { type MainModule } from '../wasm/AniraWeb'

// Lazy-evaluated so the module can be imported in AudioWorkletGlobalScope
// where URL may not be available. The URLs are only needed on the main thread.
let _jsUrl: string | undefined
let _wasmUrl: string | undefined

const getJsUrl = () => (_jsUrl ??= new URL('../wasm/AniraWeb.js', import.meta.url).href)
const getWasmUrl = () =>
  (_wasmUrl ??= new URL('../wasm/AniraWeb.wasm', import.meta.url).href)

export { getWasmUrl }

export type AniraWasmConfig = {
  processBuffers?: (processorPtr: number, inputPtr: number, outputPtr: number) => void
  processPrePost?: (
    prePostProcessorPtr: number,
    inputPtr: number,
    outputPtr: number,
    backend: number,
    phase: number
  ) => void
  wasmBinary?: ArrayBuffer
}

export type AniraWasmInstance = Omit<MainModule, 'HEAPF32' | 'HEAPU32'> & {
  HEAPF32: Float32Array
  HEAPU32: Uint32Array
}

// Export factory with WASM locateFile override
export const createAniraWasm = async (
  wasmMemory: WebAssembly.Memory,
  config?: AniraWasmConfig & Record<string, unknown>
): Promise<AniraWasmInstance> => {
  const { processBuffers, processPrePost, wasmBinary, ...rest } = config ?? {}
  const out = await AniraWebFactory({
    processBuffers: processBuffers ?? (() => {}),
    processPrePost: processPrePost ?? (() => {}),
    wasmBinary,
    ...rest,
    wasmMemory,
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        return getWasmUrl()
      }
      if (path.endsWith('.js')) {
        return getJsUrl()
      }
      return path
    },
  })

  return {
    ...out,
    HEAPF32: out.HEAPF32 as Float32Array,
    HEAPU32: out.HEAPU32 as Uint32Array,
  }
}
