// The second worker module of ort.html: the ONNX Runtime Web engine with two intra-op threads per
// inference. Its init starts onnxruntime-web's module with one Worker of its own (the glue loads
// itself into it) before the inference Worker's loop, so a context with this module is ready only
// once that Worker runs.
import {
  ONNXRUNTIME_WEB_ENGINE_ID,
  onnxRuntimeWeb,
} from '../../dist/engines/onnxruntime-web.js'
import { defineEngine } from '../../dist/scope.js'

defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb({ numThreads: 2 }))
