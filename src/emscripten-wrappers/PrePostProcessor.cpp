#include "anira/PrePostProcessor.h"

#include <emscripten/emscripten.h>

#include "anira/utils/InferenceBackend.h"

// ------ PrePostProcessor C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t prepostprocessor_create(uintptr_t config_ptr) {
    return reinterpret_cast<uintptr_t>(
        new anira::PrePostProcessor(*reinterpret_cast<anira::InferenceConfig*>(config_ptr)));
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::PrePostProcessor*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
uintptr_t prepostprocessor_from_pointer(uintptr_t ptr) {
    return ptr;
}

// Processing
EMSCRIPTEN_KEEPALIVE
void prepostprocessor_pre_process(uintptr_t ptr,
                                  uintptr_t ring_buffers_ptr,
                                  uintptr_t buffers_ptr,
                                  int backend) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->pre_process(
        *reinterpret_cast<std::vector<anira::RingBuffer>*>(ring_buffers_ptr),
        *reinterpret_cast<std::vector<anira::BufferF>*>(buffers_ptr),
        static_cast<anira::InferenceBackend>(backend));
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_post_process(uintptr_t ptr,
                                   uintptr_t buffers_ptr,
                                   uintptr_t ring_buffers_ptr,
                                   int backend) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->post_process(
        *reinterpret_cast<std::vector<anira::BufferF>*>(buffers_ptr),
        *reinterpret_cast<std::vector<anira::RingBuffer>*>(ring_buffers_ptr),
        static_cast<anira::InferenceBackend>(backend));
}

// Input/Output configuration
EMSCRIPTEN_KEEPALIVE
void prepostprocessor_set_input(uintptr_t ptr, float value, size_t channel, size_t tensor_index) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->set_input(value, tensor_index, channel);
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_set_output(uintptr_t ptr, float value, size_t channel, size_t tensor_index) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->set_output(value, tensor_index, channel);
}

EMSCRIPTEN_KEEPALIVE
float prepostprocessor_get_input(uintptr_t ptr, size_t channel, size_t tensor_index) {
    return reinterpret_cast<anira::PrePostProcessor*>(ptr)->get_input(tensor_index, channel);
}

EMSCRIPTEN_KEEPALIVE
float prepostprocessor_get_output(uintptr_t ptr, size_t channel, size_t tensor_index) {
    return reinterpret_cast<anira::PrePostProcessor*>(ptr)->get_output(tensor_index, channel);
}

// Buffer operations
EMSCRIPTEN_KEEPALIVE
void prepostprocessor_pop_samples_from_buffer(uintptr_t ptr,
                                              uintptr_t ring_buffer_ptr,
                                              uintptr_t buffer_ptr,
                                              size_t num_samples) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->pop_samples_from_buffer(
        *reinterpret_cast<anira::RingBuffer*>(ring_buffer_ptr),
        *reinterpret_cast<anira::BufferF*>(buffer_ptr),
        num_samples);
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_pop_samples_from_buffer_window(uintptr_t ptr,
                                                     uintptr_t ring_buffer_ptr,
                                                     uintptr_t buffer_ptr,
                                                     size_t num_samples,
                                                     size_t window_size) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->pop_samples_from_buffer(
        *reinterpret_cast<anira::RingBuffer*>(ring_buffer_ptr),
        *reinterpret_cast<anira::BufferF*>(buffer_ptr),
        num_samples,
        window_size);
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_pop_samples_from_buffer_window_offset(uintptr_t ptr,
                                                            uintptr_t ring_buffer_ptr,
                                                            uintptr_t buffer_ptr,
                                                            size_t num_samples,
                                                            size_t window_size,
                                                            size_t offset) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->pop_samples_from_buffer(
        *reinterpret_cast<anira::RingBuffer*>(ring_buffer_ptr),
        *reinterpret_cast<anira::BufferF*>(buffer_ptr),
        num_samples,
        window_size,
        offset);
}

EMSCRIPTEN_KEEPALIVE
void prepostprocessor_push_samples_to_buffer(uintptr_t ptr,
                                             uintptr_t buffer_ptr,
                                             uintptr_t ring_buffer_ptr,
                                             size_t num_samples) {
    reinterpret_cast<anira::PrePostProcessor*>(ptr)->push_samples_to_buffer(
        *reinterpret_cast<anira::BufferF*>(buffer_ptr),
        *reinterpret_cast<anira::RingBuffer*>(ring_buffer_ptr),
        num_samples);
}

}  // extern "C"