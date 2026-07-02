import { setupInferenceWorker } from './inferenceWorkerHandler'

// Default inference worker with no custom processor classes.
// To use custom JSBackendBase subclasses, create your own worker file:
//
//   import { setupInferenceWorker } from './inferenceWorkerHandler'
//   import { CustomJSBackend } from './custom-js-backend'
//   setupInferenceWorker({ CustomJSBackend })
//
// Then pass its URL to spinUpInferenceWorker().

setupInferenceWorker()
