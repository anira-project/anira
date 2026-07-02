import type { AniraWasmInstance } from '../factory'

export interface InferenceBackendValues {
  readonly ONNX: number
  readonly CUSTOM: number
}

export const createInferenceBackend = (
  wasmInstance: AniraWasmInstance
): InferenceBackendValues => {
  return Object.freeze({
    ONNX: wasmInstance._get_inference_backend_onnx(),
    CUSTOM: wasmInstance._get_inference_backend_custom(),
  })
}
