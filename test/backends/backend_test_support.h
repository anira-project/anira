#ifndef ANIRA_TEST_BACKENDS_BACKEND_TEST_SUPPORT_H
#define ANIRA_TEST_BACKENDS_BACKEND_TEST_SUPPORT_H

// Shared helpers for the tests that drive a backend processor directly (no
// Context, no threads): building the BufferF vectors process() takes, checking
// them, and reading a model file into memory for the "model as bytes" paths.

#include <anira/utils/Buffer.h>

#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <string>
#include <vector>

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
