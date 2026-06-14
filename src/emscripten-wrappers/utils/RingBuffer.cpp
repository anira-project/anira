#include <emscripten/emscripten.h>
#include "anira/utils/RingBuffer.h"

// ------ RingBuffer C API ----

extern "C" {

// Constructor/Destructor
EMSCRIPTEN_KEEPALIVE
uintptr_t ringbuffer_create() {
    return reinterpret_cast<uintptr_t>(new anira::RingBuffer());
}

EMSCRIPTEN_KEEPALIVE
void ringbuffer_destroy(uintptr_t ptr) {
    delete reinterpret_cast<anira::RingBuffer*>(ptr);
}

// Initialization
EMSCRIPTEN_KEEPALIVE
void ringbuffer_initialize_with_positions(uintptr_t ptr, size_t num_channels, size_t num_samples) {
    reinterpret_cast<anira::RingBuffer*>(ptr)->initialize_with_positions(num_channels, num_samples);
}

EMSCRIPTEN_KEEPALIVE
void ringbuffer_clear_with_positions(uintptr_t ptr) {
    reinterpret_cast<anira::RingBuffer*>(ptr)->clear_with_positions();
}

// Sample operations
EMSCRIPTEN_KEEPALIVE
void ringbuffer_push_sample(uintptr_t ptr, size_t channel, float sample) {
    reinterpret_cast<anira::RingBuffer*>(ptr)->push_sample(channel, sample);
}

EMSCRIPTEN_KEEPALIVE
float ringbuffer_pop_sample(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<anira::RingBuffer*>(ptr)->pop_sample(channel);
}

EMSCRIPTEN_KEEPALIVE
float ringbuffer_get_future_sample(uintptr_t ptr, size_t channel, size_t offset) {
    return reinterpret_cast<anira::RingBuffer*>(ptr)->get_future_sample(channel, offset);
}

EMSCRIPTEN_KEEPALIVE
float ringbuffer_get_past_sample(uintptr_t ptr, size_t channel, size_t offset) {
    return reinterpret_cast<anira::RingBuffer*>(ptr)->get_past_sample(channel, offset);
}

// Status queries
EMSCRIPTEN_KEEPALIVE
size_t ringbuffer_get_available_samples(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<anira::RingBuffer*>(ptr)->get_available_samples(channel);
}

EMSCRIPTEN_KEEPALIVE
size_t ringbuffer_get_available_past_samples(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<anira::RingBuffer*>(ptr)->get_available_past_samples(channel);
}

} // extern "C"

