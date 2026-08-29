#include "anira/scheduler/InferenceThread.h"

#include <emscripten/emscripten.h>

#include "anira/scheduler/Context.h"

/**
 * C API wrapper over anira::InferenceThread for the Emscripten build.
 *
 * Under __EMSCRIPTEN__, anira::InferenceThread does not inherit from
 * thl::core::Thread and does not own an OS thread. JS Workers drive the
 * processing loop by calling inference_thread_run_loop(), and start/stop
 * just flip an atomic flag inside the class.
 *
 * inference_thread_create_from_context() MUST be called from the main WASM
 * instance (the one that owns the allocator) because it heap-allocates the
 * InferenceThread object. Once created, the object pointer can be shared
 * with worker instances via postMessage — execute() and run_loop() use a
 * non-tokenized dequeue that never allocates, so they are safe to invoke
 * from any WASM instance.
 */

extern "C" {

EMSCRIPTEN_KEEPALIVE
uintptr_t inference_thread_create_from_context() {
    // Ownership passes to JS; inference_thread_destroy() deletes the object.
    return reinterpret_cast<uintptr_t>(anira::Context::make_inference_thread().release());
}

// Number of inference threads currently spun up (started and not yet stopped)
// across all WASM instances. Callable from any instance: the counter lives in
// shared static memory.
EMSCRIPTEN_KEEPALIVE
unsigned int get_num_inference_threads() {
    return anira::Context::get_num_inference_threads();
}

EMSCRIPTEN_KEEPALIVE
int inference_thread_execute(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceThread*>(ptr)->execute() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
void inference_thread_run_loop(uintptr_t ptr) {
    reinterpret_cast<anira::InferenceThread*>(ptr)->run_loop();
}

EMSCRIPTEN_KEEPALIVE
void inference_thread_stop(uintptr_t ptr) {
    reinterpret_cast<anira::InferenceThread*>(ptr)->stop();
}

EMSCRIPTEN_KEEPALIVE
void inference_thread_start(uintptr_t ptr) {
    reinterpret_cast<anira::InferenceThread*>(ptr)->start();
}

EMSCRIPTEN_KEEPALIVE
int inference_thread_should_exit(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceThread*>(ptr)->should_exit() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int inference_thread_is_running(uintptr_t ptr) {
    return reinterpret_cast<anira::InferenceThread*>(ptr)->is_running() ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
void inference_thread_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::InferenceThread*>(ptr);
}

}  // extern "C"
