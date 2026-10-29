// The worker module of ort.html: the ONNX Runtime Web engine over the SimpleGainNetwork model of
// extras/models (output = input * gain), defined on each inference Worker's registry; its init
// loads onnxruntime-web's binary by URL (dist/wasm/ort-wasm-simd-threaded.wasm) and creates the
// session before the Worker enters its loop.
import {
  ONNXRUNTIME_WEB_ENGINE_ID,
  onnxRuntimeWeb,
} from '../../dist/engines/onnxruntime-web.js'
import { defineEngine } from '../../dist/scope.js'

defineEngine(
  ONNXRUNTIME_WEB_ENGINE_ID,
  onnxRuntimeWeb({
    model: new URL(
      '../../extras/models/model-pool/example-models/SimpleGainNetwork/models/simple_gain_network_mono.onnx',
      import.meta.url
    ),
  })
)
