#include "Adapters.h"

#include <anira/abi/enums.h>

#include <memory>
#include <vector>

#include "Adapter.h"

namespace anira::engine {

std::shared_ptr<BuiltinEngine> make_builtin_engine(anira_engine engine) {
    switch (engine) {
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH: return make_libtorch_engine();
#endif
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME: return make_onnxruntime_engine();
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return make_tflite_engine();
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return make_litert_engine();
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH: return make_executorch_engine();
#endif
        default: return nullptr;
    }
}

std::shared_ptr<Loaded> make_builtin_loaded(const std::shared_ptr<BuiltinEngine>& engine) {
    if (engine == nullptr) { return nullptr; }
    switch (engine->kind()) {
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH: return make_libtorch_loaded(engine);
#endif
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME: return make_onnxruntime_loaded(engine);
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return make_tflite_loaded(engine);
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return make_litert_loaded(engine);
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH: return make_executorch_loaded(engine);
#endif
        default: return nullptr;  // an object of an engine this build does not carry
    }
}

std::vector<ProviderInfo> builtin_providers(anira_engine engine) {
    const std::shared_ptr<BuiltinEngine> object = make_builtin_engine(engine);
    if (object == nullptr) { return {}; }  // not in this build: no provider at all
    return object->providers();
}

}  // namespace anira::engine
