// The ids of the engines the package ships, apart from the engines themselves: a page names an
// engine (addEngine, a model entry) without bundling it. The root entry and the scope entry
// re-export these; only a worker module that defines the engine imports the engine's own entry.

import { anira_edge_class } from '../abi/enums'
import type { EngineEdge } from '../Pipeline'

/** The engine id the ONNX Runtime Web engine is registered under by convention. */
export const ONNXRUNTIME_WEB_ENGINE_ID = 'org.anira.onnxruntime-web'

/**
 * The ONNX Runtime Web engine's edge statement: a copy of every slot within host memory by the
 * engine's own code, between anira's module and onnxruntime-web's (two wasm memories).
 * `Pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID)` states it by default; an engine defined under
 * another id states it with `addEngine(id, { edge: ONNXRUNTIME_WEB_ENGINE_EDGE })`.
 */
export const ONNXRUNTIME_WEB_ENGINE_EDGE: EngineEdge = {
  edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
  reason: "onnxruntime-web: copied between the module's heap and ORT's heap",
}

/** @internal The edge statement of each engine the package ships, by its id. */
export const BUNDLED_ENGINE_EDGES: Readonly<Record<string, EngineEdge>> = {
  [ONNXRUNTIME_WEB_ENGINE_ID]: ONNXRUNTIME_WEB_ENGINE_EDGE,
}
