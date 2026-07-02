/// <reference types="vite/client" />

declare module '*.wasm?url' {
  const url: string
  export default url
}

declare module 'onnxruntime-web/ort-wasm-simd-threaded.mjs' {
  const factory: (config?: Record<string, unknown>) => Promise<unknown>
  export default factory
}
