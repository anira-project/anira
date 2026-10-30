// @anira-project/anira: a thin typed layer over the C ABI of the module (abi/bindings.ts, generated
// from abi/anira.yml). The page's main thread instantiates the module (Anira); anira starts the
// other threads from C on the same memory: the context's inference Workers, which import the
// context's worker module, and the audio context's Wasm Audio Worklet. JavaScript stages and
// engines are defined per scope on the scope's registry (defineStage, defineEngine; also in the
// light entry @anira-project/anira/scope for a Worker's or a worklet's module).
export * from './scope'
export * from './Anira'
export * from './Context'
export * from './ModelConfig'
export * from './Contract'
export * from './Pipeline'
export * from './Handler'
export * from './AudioNode'
export {
  measureRenderGeometry,
  type RenderGeometry,
  type RenderGeometryOptions,
} from './renderGeometry'
