#ifndef ANIRA_TEST_ENGINES_ENGINE_TEST_SUPPORT_H
#define ANIRA_TEST_ENGINES_ENGINE_TEST_SUPPORT_H

// Shared helpers for the tests that drive a built-in adapter directly through the engine
// room's interface (no Context, no threads): the record of a model as a session's prepare
// hands it over, the buffers of a chunk, the descriptors over them and the context of one
// call an adapter's run takes, checking the buffers, and reading a model file into memory for
// the "model as bytes" paths.

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/tensor.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include "engines/Adapter.h"

namespace anira_test {

/// The buffers of one side of a chunk, one mono run per slot.
using Buffers = std::vector<std::vector<float>>;

/// One buffer per entry of @p sizes, each filled with @p value.
inline Buffers filled_buffers(std::initializer_list<size_t> sizes, float value) {
    Buffers buffers;
    buffers.reserve(sizes.size());
    for (const size_t size : sizes) { buffers.emplace_back(size, value); }
    return buffers;
}

inline bool any_sample_nonzero(const std::vector<float>& buffer) {
    for (const float sample : buffer) {
        if (sample != 0.F) { return true; }
    }
    return false;
}

/// The record of a model file for one engine, as a session's prepare hands it to the engine
/// room: float32 tensors of the given engine extents, unnamed (the slots bind by position),
/// `instances` shared slots (none: an exclusive session's model, whose calls run on an executor
/// of the session's own), `warm_up` inferences at load and the entry point `entry` (empty: the
/// file's default).
inline anira::engine::Model model_record(anira_engine engine,
                                         std::string path,
                                         const std::vector<std::vector<int64_t>>& inputs,
                                         const std::vector<std::vector<int64_t>>& outputs,
                                         uint32_t warm_up = 0,
                                         std::string entry = {},
                                         uint32_t instances = 0) {
    const auto side = [](const std::vector<std::vector<int64_t>>& shapes) {
        std::vector<anira::engine::TensorInfo> tensors;
        tensors.reserve(shapes.size());
        for (const std::vector<int64_t>& dims : shapes) {
            anira::engine::TensorInfo tensor;
            tensor.m_dims = dims;
            tensor.m_num_elements = 1;
            for (const int64_t extent : dims) {
                tensor.m_num_elements *= static_cast<size_t>(extent);
            }
            tensors.push_back(std::move(tensor));
        }
        return tensors;
    };
    anira::engine::Model model;
    model.m_engine = engine;
    model.m_path = std::move(path);
    model.m_entry = std::move(entry);
    model.m_inputs = side(inputs);
    model.m_outputs = side(outputs);
    model.m_instances = instances;
    model.m_warm_up = warm_up;
    return model;
}

/// @p model over @p bytes instead of its path: the caller keeps the bytes alive.
inline anira::engine::Model with_bytes(anira::engine::Model model, const std::vector<char>& bytes) {
    model.m_path.clear();
    model.m_bytes = bytes.data();
    model.m_num_bytes = bytes.size();
    return model;
}

/// One descriptor per buffer of @p buffers over its memory, in the shape of the record's
/// tensor at the same slot (the engine dims; rank 0 for a slot the record does not have):
/// what the inference thread hands an adapter for one side of a chunk.
inline std::vector<anira_tensor> descriptors_of(
    Buffers& buffers,
    const std::vector<anira::engine::TensorInfo>& tensors) {
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

#endif  // ANIRA_TEST_ENGINES_ENGINE_TEST_SUPPORT_H
