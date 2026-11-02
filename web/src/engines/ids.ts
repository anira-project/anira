// The ids of the engines the package ships, apart from the engines themselves: a page names an
// engine (addEngine, a model entry) without bundling it. The root entry and the scope entry
// re-export these; only a worker module that defines the engine imports the engine's own entry.

import { anira_edge_class } from '../abi/enums'
import type { EngineOptions } from '../Pipeline'

/** The engine id the ONNX Runtime Web engine is registered under by convention. */
export const ONNXRUNTIME_WEB_ENGINE_ID = 'org.anira.onnxruntime-web'

/**
 * The options the ONNX Runtime Web engine is added with (`addEngine(ONNXRUNTIME_WEB_ENGINE_ID,
 * ONNXRUNTIME_WEB_ENGINE_OPTIONS)`): its edge statement, a copy of every slot in host memory by
 * the engine's own code, between anira's module and onnxruntime-web's (two wasm memories).
 */
export const ONNXRUNTIME_WEB_ENGINE_OPTIONS: EngineOptions = {
  edge: {
    edgeClass: anira_edge_class.ANIRA_EDGE_HOST_COPY,
    reason: "onnxruntime-web: anira's module and onnxruntime-web's are two wasm memories",
  },
}
