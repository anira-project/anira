#include <emscripten/emscripten.h>
#include "anira/utils/HostConfig.h"
#include "anira/InferenceConfig.h"

// ------ HostConfig C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t hostconfig_create() {
    return reinterpret_cast<uintptr_t>(new anira::HostConfig());
}

EMSCRIPTEN_KEEPALIVE
uintptr_t hostconfig_create_with_params(float buffer_size, float sample_rate, bool allow_smaller_buffers, size_t tensor_index) {
    return reinterpret_cast<uintptr_t>(new anira::HostConfig(buffer_size, sample_rate, allow_smaller_buffers, tensor_index));
}

EMSCRIPTEN_KEEPALIVE
void hostconfig_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::HostConfig*>(ptr);
}

// Property getters
EMSCRIPTEN_KEEPALIVE
float hostconfig_get_buffer_size(uintptr_t ptr) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->m_buffer_size;
}

EMSCRIPTEN_KEEPALIVE
float hostconfig_get_sample_rate(uintptr_t ptr) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->m_sample_rate;
}

EMSCRIPTEN_KEEPALIVE
bool hostconfig_get_allow_smaller_buffers(uintptr_t ptr) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->m_allow_smaller_buffers;
}

EMSCRIPTEN_KEEPALIVE
size_t hostconfig_get_tensor_index(uintptr_t ptr) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->m_tensor_index;
}

// Property setters
EMSCRIPTEN_KEEPALIVE
void hostconfig_set_buffer_size(uintptr_t ptr, float buffer_size) {
    reinterpret_cast<anira::HostConfig*>(ptr)->m_buffer_size = buffer_size;
}

EMSCRIPTEN_KEEPALIVE
void hostconfig_set_sample_rate(uintptr_t ptr, float sample_rate) {
    reinterpret_cast<anira::HostConfig*>(ptr)->m_sample_rate = sample_rate;
}

EMSCRIPTEN_KEEPALIVE
void hostconfig_set_allow_smaller_buffers(uintptr_t ptr, bool allow_smaller_buffers) {
    reinterpret_cast<anira::HostConfig*>(ptr)->m_allow_smaller_buffers = allow_smaller_buffers;
}

EMSCRIPTEN_KEEPALIVE
void hostconfig_set_tensor_index(uintptr_t ptr, size_t tensor_index) {
    reinterpret_cast<anira::HostConfig*>(ptr)->m_tensor_index = tensor_index;
}

// Comparison
EMSCRIPTEN_KEEPALIVE
bool hostconfig_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::HostConfig*>(ptr) == *reinterpret_cast<anira::HostConfig*>(other_ptr);
}

EMSCRIPTEN_KEEPALIVE
bool hostconfig_not_equals(uintptr_t ptr, uintptr_t other_ptr) {
    return *reinterpret_cast<anira::HostConfig*>(ptr) != *reinterpret_cast<anira::HostConfig*>(other_ptr);
}

// Methods
EMSCRIPTEN_KEEPALIVE
float hostconfig_get_relative_buffer_size(uintptr_t ptr, uintptr_t inference_config_ptr, size_t tensor_index, bool input) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->get_relative_buffer_size(
        *reinterpret_cast<anira::InferenceConfig*>(inference_config_ptr),
        tensor_index,
        input
    );
}

EMSCRIPTEN_KEEPALIVE
float hostconfig_get_relative_sample_rate(uintptr_t ptr, uintptr_t inference_config_ptr, size_t tensor_index, bool input) {
    return reinterpret_cast<anira::HostConfig*>(ptr)->get_relative_sample_rate(
        *reinterpret_cast<anira::InferenceConfig*>(inference_config_ptr),
        tensor_index,
        input
    );
}

} // extern "C"
