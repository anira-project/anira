#include <emscripten/emscripten.h>

#include "anira/InferenceConfig.h"
#include "anira/utils/Buffer.h"
#include "anira/utils/HostConfig.h"
#include "anira/utils/RingBuffer.h"

// ------ Vector C API ----

extern "C" {

// ------ VectorSizeT ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_size_t_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<size_t>());
}

EMSCRIPTEN_KEEPALIVE
void vector_size_t_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<size_t>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_size_t_push_back(uintptr_t ptr, size_t value) {
    reinterpret_cast<std::vector<size_t>*>(ptr)->push_back(value);
}

EMSCRIPTEN_KEEPALIVE
size_t vector_size_t_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<size_t>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
size_t vector_size_t_get(uintptr_t ptr, size_t index) {
    return (*reinterpret_cast<std::vector<size_t>*>(ptr))[index];
}

// ------ VectorInt64T ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_int64_t_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<int64_t>());
}

EMSCRIPTEN_KEEPALIVE
void vector_int64_t_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<int64_t>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_int64_t_push_back(uintptr_t ptr, int64_t value) {
    reinterpret_cast<std::vector<int64_t>*>(ptr)->push_back(value);
}

EMSCRIPTEN_KEEPALIVE
size_t vector_int64_t_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<int64_t>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
int64_t vector_int64_t_get(uintptr_t ptr, size_t index) {
    return (*reinterpret_cast<std::vector<int64_t>*>(ptr))[index];
}

// ------ VectorFloat ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_float_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<float>());
}

EMSCRIPTEN_KEEPALIVE
void vector_float_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<float>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_float_push_back(uintptr_t ptr, float value) {
    reinterpret_cast<std::vector<float>*>(ptr)->push_back(value);
}

EMSCRIPTEN_KEEPALIVE
size_t vector_float_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<float>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
float vector_float_get(uintptr_t ptr, size_t index) {
    return (*reinterpret_cast<std::vector<float>*>(ptr))[index];
}

// ------ VectorUnsignedInt ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_unsigned_int_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<unsigned int>());
}

EMSCRIPTEN_KEEPALIVE
void vector_unsigned_int_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<unsigned int>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_unsigned_int_push_back(uintptr_t ptr, unsigned int value) {
    reinterpret_cast<std::vector<unsigned int>*>(ptr)->push_back(value);
}

EMSCRIPTEN_KEEPALIVE
size_t vector_unsigned_int_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<unsigned int>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
unsigned int vector_unsigned_int_get(uintptr_t ptr, size_t index) {
    return (*reinterpret_cast<std::vector<unsigned int>*>(ptr))[index];
}

// ------ VectorVectorInt64 ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_vector_int64_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<std::vector<int64_t>>());
}

EMSCRIPTEN_KEEPALIVE
void vector_vector_int64_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<std::vector<int64_t>>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_vector_int64_push_back(uintptr_t ptr, uintptr_t inner_vector_ptr) {
    reinterpret_cast<std::vector<std::vector<int64_t>>*>(ptr)->push_back(
        *reinterpret_cast<std::vector<int64_t>*>(inner_vector_ptr));
}

EMSCRIPTEN_KEEPALIVE
size_t vector_vector_int64_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<std::vector<int64_t>>*>(ptr)->size();
}

// Returns a non-owning pointer to the inner std::vector<int64_t> at `index`.
// Valid only while the outer vector is alive and not resized.
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_vector_int64_get(uintptr_t ptr, size_t index) {
    return reinterpret_cast<uintptr_t>(
        &(*reinterpret_cast<std::vector<std::vector<int64_t>>*>(ptr))[index]);
}

// ------ VectorModelData ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_model_data_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<anira::ModelData>());
}

EMSCRIPTEN_KEEPALIVE
void vector_model_data_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<anira::ModelData>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_model_data_push_back(uintptr_t ptr, uintptr_t model_data_ptr) {
    reinterpret_cast<std::vector<anira::ModelData>*>(ptr)->push_back(
        *reinterpret_cast<anira::ModelData*>(model_data_ptr));
}

EMSCRIPTEN_KEEPALIVE
size_t vector_model_data_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<anira::ModelData>*>(ptr)->size();
}

// ------ VectorTensorShape ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_tensor_shape_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<anira::TensorShape>());
}

EMSCRIPTEN_KEEPALIVE
void vector_tensor_shape_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<anira::TensorShape>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_tensor_shape_push_back(uintptr_t ptr, uintptr_t tensor_shape_ptr) {
    reinterpret_cast<std::vector<anira::TensorShape>*>(ptr)->push_back(
        *reinterpret_cast<anira::TensorShape*>(tensor_shape_ptr));
}

EMSCRIPTEN_KEEPALIVE
size_t vector_tensor_shape_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<anira::TensorShape>*>(ptr)->size();
}

// ------ VectorRingBuffer ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_ring_buffer_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<anira::RingBuffer>());
}

EMSCRIPTEN_KEEPALIVE
void vector_ring_buffer_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<anira::RingBuffer>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_ring_buffer_push_back(uintptr_t ptr, uintptr_t ring_buffer_ptr) {
    reinterpret_cast<std::vector<anira::RingBuffer>*>(ptr)->push_back(
        *reinterpret_cast<anira::RingBuffer*>(ring_buffer_ptr));
}

EMSCRIPTEN_KEEPALIVE
size_t vector_ring_buffer_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<anira::RingBuffer>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
uintptr_t vector_ring_buffer_get(uintptr_t ptr, size_t index) {
    return reinterpret_cast<uintptr_t>(
        &(*reinterpret_cast<std::vector<anira::RingBuffer>*>(ptr))[index]);
}

// ------ VectorBufferF ----
EMSCRIPTEN_KEEPALIVE
uintptr_t vector_buffer_f_create() {
    return reinterpret_cast<uintptr_t>(new std::vector<anira::BufferF>());
}

EMSCRIPTEN_KEEPALIVE
void vector_buffer_f_destroy(uintptr_t ptr) {
    delete reinterpret_cast<std::vector<anira::BufferF>*>(ptr);
}

EMSCRIPTEN_KEEPALIVE
void vector_buffer_f_push_back(uintptr_t ptr, uintptr_t buffer_ptr) {
    reinterpret_cast<std::vector<anira::BufferF>*>(ptr)->push_back(
        *reinterpret_cast<anira::BufferF*>(buffer_ptr));
}

EMSCRIPTEN_KEEPALIVE
size_t vector_buffer_f_size(uintptr_t ptr) {
    return reinterpret_cast<std::vector<anira::BufferF>*>(ptr)->size();
}

EMSCRIPTEN_KEEPALIVE
uintptr_t vector_buffer_f_get(uintptr_t ptr, size_t index) {
    return reinterpret_cast<uintptr_t>(
        &(*reinterpret_cast<std::vector<anira::BufferF>*>(ptr))[index]);
}

// ------ BufferF ----
EMSCRIPTEN_KEEPALIVE
size_t buffer_f_get_num_channels(uintptr_t ptr) {
    return reinterpret_cast<anira::BufferF*>(ptr)->get_num_channels();
}

EMSCRIPTEN_KEEPALIVE
size_t buffer_f_get_num_samples(uintptr_t ptr) {
    return reinterpret_cast<anira::BufferF*>(ptr)->get_num_samples();
}

EMSCRIPTEN_KEEPALIVE
uintptr_t buffer_f_get_read_pointer(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_read_pointer(channel));
}

EMSCRIPTEN_KEEPALIVE
uintptr_t buffer_f_get_write_pointer(uintptr_t ptr, size_t channel) {
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<anira::BufferF*>(ptr)->get_write_pointer(channel));
}

EMSCRIPTEN_KEEPALIVE
void buffer_f_clear(uintptr_t ptr) {
    reinterpret_cast<anira::BufferF*>(ptr)->clear();
}

}  // extern "C"
