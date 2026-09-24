#ifndef ANIRA_TEST_BACKENDS_BACKEND_TEST_SUPPORT_H
#define ANIRA_TEST_BACKENDS_BACKEND_TEST_SUPPORT_H

// Shared helpers for the tests that drive a built-in adapter directly through the engine
// room's interface (no Context, no threads): building the BufferF vectors a chunk holds,
// the descriptors over them and the context of one call an adapter's run takes, checking
// the buffers, and reading a model file into memory for the "model as bytes" paths.

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>
#include <anira/utils/Buffer.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <string>
#include <vector>

#include "backends/Adapter.h"

namespace anira_test {

/// One mono buffer of @p num_samples, every sample set to @p value.
inline anira::BufferF filled_buffer(size_t num_samples, float value) {
    anira::BufferF buffer(1, num_samples);
    for (size_t i = 0; i < num_samples; ++i) { buffer.set_sample(0, i, value); }
    return buffer;
}

/// One mono buffer per entry of @p sizes, each filled with @p value — the shape
/// BackendBase::process() expects for a model's inputs or outputs.
inline std::vector<anira::BufferF> filled_buffers(std::initializer_list<size_t> sizes,
                                                  float value) {
    std::vector<anira::BufferF> buffers;
    buffers.reserve(sizes.size());
    for (size_t size : sizes) { buffers.push_back(filled_buffer(size, value)); }
    return buffers;
}

inline bool all_samples_equal(const anira::BufferF& buffer, float expected) {
    for (size_t i = 0; i < buffer.get_num_samples(); ++i) {
        if (buffer.get_sample(0, i) != expected) { return false; }
    }
    return true;
}

inline bool any_sample_nonzero(const anira::BufferF& buffer) {
    for (size_t i = 0; i < buffer.get_num_samples(); ++i) {
        if (buffer.get_sample(0, i) != 0.F) { return true; }
    }
    return false;
}

/// One descriptor per buffer of @p buffers over its memory, in the shape of the record's
/// tensor at the same slot (the engine dims; rank 0 for a slot the record does not have):
/// what the inference thread hands an adapter for one side of a chunk.
inline std::vector<anira_tensor> descriptors_of(
    std::vector<anira::BufferF>& buffers,
    const std::vector<anira::backend::TensorInfo>& tensors) {
    std::vector<anira_tensor> descriptors(buffers.size());
    for (size_t slot = 0; slot < buffers.size(); ++slot) {
        const bool has_shape = slot < tensors.size();
        anira_tensor_init_host(&descriptors[slot],
                               buffers[slot].data(),
                               ANIRA_DTYPE_F32,
                               has_shape ? static_cast<uint32_t>(tensors[slot].m_dims.size()) : 0U,
                               has_shape ? tensors[slot].m_dims.data() : nullptr);
    }
    return descriptors;
}

/// The context of one call over @p inputs and @p outputs: instance 0, entry 0, no ticket,
/// no flags. What the adapter's run takes.
inline anira_engine_ctx context_of(const std::vector<anira_tensor>& inputs,
                                   std::vector<anira_tensor>& outputs) {
    anira_engine_ctx ctx{};
    ctx.num_inputs = static_cast<uint32_t>(inputs.size());
    ctx.num_outputs = static_cast<uint32_t>(outputs.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.inputs = inputs.data();
    ctx.outputs = outputs.data();
    return ctx;
}

/// Reads a model file whole, so it can be handed to ModelData as bytes. Empty
/// when the file is missing, which the callers assert on.
inline std::vector<char> read_model_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { return {}; }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    file.read(bytes.data(), size);
    return bytes;
}

}  // namespace anira_test

#endif  // ANIRA_TEST_BACKENDS_BACKEND_TEST_SUPPORT_H
