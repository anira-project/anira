// The ids of the engines the package ships, apart from the engines themselves: a page names an
// engine (registerEngine, a model entry) without bundling it. The root entry and the scope entry
// re-export these; only a worker module that defines the engine imports the engine's own entry.

/** The engine id the ONNX Runtime Web engine is registered under by convention. */
export const ONNXRUNTIME_WEB_ENGINE_ID = 'org.anira.onnxruntime-web'
