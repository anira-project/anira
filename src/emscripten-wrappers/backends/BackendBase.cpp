#include <anira/anira.h>
#include <emscripten/emscripten.h>

// ------ BackendBase C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t backendbase_create(uintptr_t inference_config_ptr) {
    auto* config = reinterpret_cast<anira::InferenceConfig*>(inference_config_ptr);
    return reinterpret_cast<uintptr_t>(new anira::BackendBase(*config));
}

EMSCRIPTEN_KEEPALIVE
void backendbase_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::BackendBase*>(ptr);
}

}  // extern "C"
