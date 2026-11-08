// The ids of the engines the package ships, apart from the engines themselves: a page names an
// engine (addEngine, a model entry) without bundling it. The root entry and the scope entry
// re-export these; only a worker module that defines the engine imports the engine's own entry.

import { anira_edge_class } from '../abi/enums'
import type { EngineEdges } from '../Pipeline'

/** The engine id the ONNX Runtime Web engine is registered under by convention. */
export const ONNXRUNTIME_WEB_ENGINE_ID = 'org.anira.onnxruntime-web'

/**
 * The ONNX Runtime Web engine's edge statements, one per direction: a copy of every slot within
 * host memory by the engine's own code, between anira's module and onnxruntime-web's (two wasm
 * memories), each input copied into ORT's heap and each output copied out of it.
 * `Pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID)` states them by default; an engine defined
 * under another id states them with `addEngine(id, { edge: ONNXRUNTIME_WEB_ENGINE_EDGES })`.
 */
export const ONNXRUNTIME_WEB_ENGINE_EDGES: EngineEdges = {
  inputs: {
    edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
    reason: "onnxruntime-web: copied into ORT's heap",
  },
  outputs: {
    edgeClass: anira_edge_class.ANIRA_EDGE_COPY,
    reason: "onnxruntime-web: copied out of ORT's heap",
  },
}

/** @internal The edge statements of each engine the package ships, by its id. */
export const BUNDLED_ENGINE_EDGES: Readonly<Record<string, EngineEdges>> = {
  [ONNXRUNTIME_WEB_ENGINE_ID]: ONNXRUNTIME_WEB_ENGINE_EDGES,
}
