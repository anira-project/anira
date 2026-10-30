// The worker module of ort.html: the ONNX Runtime Web engine, defined on each inference Worker's
// registry with no model of its own. Its init loads onnxruntime-web's binary by URL
// (dist/wasm/ort-wasm-simd-threaded.wasm); its load creates the session from the model entry's
// bytes, which ModelConfig.fromUrl fetched once on the main thread.
import {
  ONNXRUNTIME_WEB_ENGINE_ID,
  onnxRuntimeWeb,
} from '../../dist/engines/onnxruntime-web.js'
import { defineEngine } from '../../dist/scope.js'

defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb())
