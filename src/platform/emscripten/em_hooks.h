// What the inference threads call of the JavaScript engines' trampolines (em_hooks.cpp) beside
// an engine call: the release of what a thread's JavaScript engines built for models that were
// unloaded. A JavaScript engine's objects live in the realm of the thread that runs its calls
// (one per loaded model there, the registry of web/src/registry.ts), which the control thread
// of an unload cannot reach: it takes the model off the list of the loaded ones and wakes the
// threads, and each releases its own on its own thread. Compiled on the Wasm binary format
// only.
#ifndef ANIRA_PLATFORM_EMSCRIPTEN_EM_HOOKS_H
#define ANIRA_PLATFORM_EMSCRIPTEN_EM_HOOKS_H

namespace anira::em {

/// Releases what the calling thread's JavaScript engines built for every model that was
/// unloaded since the thread's last call: one atomic load when no unload happened since; else
/// the registry of the thread's scope is handed the loaded models of now and releases its
/// object of every other one (the object's destroy). For a thread that runs engine calls,
/// between two jobs (InferenceThread::execute calls it before every dequeue): never inside an
/// engine call, and an unload comes only once no inference of the model is in flight and none
/// can follow, so no call of this thread meets an object released here. A scope without a
/// registry built nothing and is left alone.
void release_unloaded_js_models() noexcept;

/// Releases what the calling thread's JavaScript engines built for every model, loaded or not:
/// for a thread that leaves its inference loop (InferenceThread::run_loop, before it returns),
/// whose scope runs no further call; a Wasm Worker of anira's is terminated behind it.
void release_all_js_models() noexcept;

}  // namespace anira::em

#endif  // ANIRA_PLATFORM_EMSCRIPTEN_EM_HOOKS_H
