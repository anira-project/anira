import { setupInferenceWorker } from './inferenceWorkerHandler'

// Default inference worker: the built-in ONNXRuntimeWebBackend and no stage classes.
// For your own JavaScript engines or stages, create your own worker file:
//
//   import { setupInferenceWorker } from './inferenceWorkerHandler'
//   import { MyEngine } from './my-engine'
//   import { MyStage } from './my-stage'
//   setupInferenceWorker({ MyEngine }, { MyStage })
//
// Then pass its URL to spinUpInferenceWorker().

setupInferenceWorker()
