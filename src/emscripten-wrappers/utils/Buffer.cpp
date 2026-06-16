#include "anira/utils/Buffer.h"

#include <emscripten/emscripten.h>

// ------ BufferF C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_create() {
    return reinterpret_cast<uintptr_t>(new anira::BufferF());
}

EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_create_with_size(size_t num_channels, size_t num_samples) {
    return reinterpret_cast<uintptr_t>(new anira::BufferF(num_channels, num_samples));
}

EMSCRIPTEN_KEEPALIVE
void bufferf_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::BufferF*>(ptr);
}

// Properties
EMSCRIPTEN_KEEPALIVE
size_t bufferf_get_num_channels(uintptr_t ptr) {
    return reinterpret_cast<anira::BufferF*>(ptr)->get_num_channels();
}

EMSCRIPTEN_KEEPALIVE
size_t bufferf_get_num_samples(uintptr_t ptr) {
    return reinterpret_cast<anira::BufferF*>(ptr)->get_num_samples();
}

EMSCRIPTEN_KEEPALIVE
void bufferf_resize(uintptr_t ptr, size_t num_channels, size_t num_samples) {
    reinterpret_cast<anira::BufferF*>(ptr)->resize(num_channels, num_samples);
}

// Pointer access
EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_get_read_pointer(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_read_pointer(channel));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_get_read_pointer_at(uintptr_t ptr, size_t channel, size_t sample_index) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_read_pointer(channel, sample_index));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_get_write_pointer(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_write_pointer(channel));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_get_write_pointer_at(uintptr_t ptr, size_t channel, size_t sample_index) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_write_pointer(channel, sample_index));
}

// Array of pointers - writes to provided output array
EMSCRIPTEN_KEEPALIVE
void bufferf_get_array_of_read_pointers(uintptr_t ptr, uintptr_t out_array) {
    anira::BufferF* buffer = reinterpret_cast<anira::BufferF*>(ptr);
    const float* const* ptrs = buffer->get_array_of_read_pointers();
    uintptr_t* out = reinterpret_cast<uintptr_t*>(out_array);
    size_t num_channels = buffer->get_num_channels();
    for (size_t i = 0; i < num_channels; ++i) { out[i] = reinterpret_cast<uintptr_t>(ptrs[i]); }
}

EMSCRIPTEN_KEEPALIVE
void bufferf_get_array_of_write_pointers(uintptr_t ptr, uintptr_t out_array) {
    anira::BufferF* buffer = reinterpret_cast<anira::BufferF*>(ptr);
    float* const* ptrs = buffer->get_array_of_write_pointers();
    uintptr_t* out = reinterpret_cast<uintptr_t*>(out_array);
    size_t num_channels = buffer->get_num_channels();
    for (size_t i = 0; i < num_channels; ++i) { out[i] = reinterpret_cast<uintptr_t>(ptrs[i]); }
}

EMSCRIPTEN_KEEPALIVE
uintptr_t bufferf_data(uintptr_t ptr) {
    return reinterpret_cast<uintptr_t>(reinterpret_cast<anira::BufferF*>(ptr)->data());
}

// Data manipulation
EMSCRIPTEN_KEEPALIVE
void bufferf_swap_data_with_buffer(uintptr_t ptr, uintptr_t other_ptr) {
    reinterpret_cast<anira::BufferF*>(ptr)->swap_data(
        *reinterpret_cast<anira::BufferF*>(other_ptr));
}

EMSCRIPTEN_KEEPALIVE
void bufferf_swap_data_with_raw_pointer(uintptr_t ptr, uintptr_t raw_pointer, size_t size) {
    float* raw_ptr = reinterpret_cast<float*>(raw_pointer);
    reinterpret_cast<anira::BufferF*>(ptr)->swap_data(raw_ptr, size);
}

EMSCRIPTEN_KEEPALIVE
void bufferf_reset_channel_ptr(uintptr_t ptr) {
    reinterpret_cast<anira::BufferF*>(ptr)->reset_channel_ptr();
}

// Sample access
EMSCRIPTEN_KEEPALIVE
float bufferf_get_sample(uintptr_t ptr, size_t channel, size_t sample_index) {
    return reinterpret_cast<anira::BufferF*>(ptr)->get_sample(channel, sample_index);
}

EMSCRIPTEN_KEEPALIVE
void bufferf_set_sample(uintptr_t ptr, size_t channel, size_t sample_index, float value) {
    reinterpret_cast<anira::BufferF*>(ptr)->set_sample(channel, sample_index, value);
}

EMSCRIPTEN_KEEPALIVE
void bufferf_clear(uintptr_t ptr) {
    reinterpret_cast<anira::BufferF*>(ptr)->clear();
}

}  // extern "C"