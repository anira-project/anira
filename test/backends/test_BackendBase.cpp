// BackendBase::process is the roundtrip (CUSTOM) and no-model fallback
// processor, so it must tolerate every tensor layout a valid InferenceConfig
// can describe — including models whose input and output tensor counts differ
// (e.g. a stateful model taking audio + state + prior and returning only
// audio + state). It used to iterate the input count while indexing output[],
// reading past the end of the output vector and crashing the host process the
// moment such a model ran on the default processor. Regression tests for that
// out-of-bounds access and for the chosen semantics: pairwise roundtrip where
// shapes match, extra outputs cleared.

#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

namespace {

constexpr int64_t k_audio_size = 16;
constexpr int64_t k_state_size = 8;
constexpr int64_t k_prior_size = 2;

anira::InferenceConfig make_config(const std::vector<anira::TensorShape>& tensor_shapes) {
    const std::vector<anira::ModelData> model_data = {
        anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)};
    return {model_data, tensor_shapes, 5.0F};
}

anira::BufferF make_buffer(size_t size, float fill_value) {
    anira::BufferF buffer(1, size);
    for (size_t i = 0; i < size; ++i) { buffer.set_sample(0, i, fill_value); }
    return buffer;
}

bool all_samples_equal(const anira::BufferF& buffer, float expected) {
    for (size_t i = 0; i < buffer.get_num_samples(); ++i) {
        if (buffer.get_sample(0, i) != expected) { return false; }
    }
    return true;
}

}  // namespace

TEST(BackendBase, ProcessWithMoreInputsThanOutputs) {
    // audio + state + prior in, audio + state out.
    const std::vector<anira::TensorShape> tensor_shapes = {
        {{{1, 1, k_audio_size}, {1, k_state_size}, {1, k_prior_size}},
         {{1, 1, k_audio_size}, {1, k_state_size}}}};
    anira::InferenceConfig config = make_config(tensor_shapes);
    anira::BackendBase backend(config);

    std::vector<anira::BufferF> input;
    input.push_back(make_buffer(static_cast<size_t>(k_audio_size), 0.25F));
    input.push_back(make_buffer(static_cast<size_t>(k_state_size), 0.5F));
    input.push_back(make_buffer(static_cast<size_t>(k_prior_size), 0.75F));

    std::vector<anira::BufferF> output;
    output.push_back(make_buffer(static_cast<size_t>(k_audio_size), -1.0F));
    output.push_back(make_buffer(static_cast<size_t>(k_state_size), -1.0F));

    backend.process(input, output, nullptr);

    // The pairwise-matching tensors roundtrip; nothing reads past output[1].
    EXPECT_TRUE(all_samples_equal(output[0], 0.25F));
    EXPECT_TRUE(all_samples_equal(output[1], 0.5F));
}

TEST(BackendBase, ProcessWithMoreOutputsThanInputsClearsExtras) {
    const std::vector<anira::TensorShape> tensor_shapes = {
        {{{1, 1, k_audio_size}}, {{1, 1, k_audio_size}, {1, k_state_size}}}};
    anira::InferenceConfig config = make_config(tensor_shapes);
    anira::BackendBase backend(config);

    std::vector<anira::BufferF> input;
    input.push_back(make_buffer(static_cast<size_t>(k_audio_size), 0.25F));

    std::vector<anira::BufferF> output;
    output.push_back(make_buffer(static_cast<size_t>(k_audio_size), -1.0F));
    output.push_back(make_buffer(static_cast<size_t>(k_state_size), -1.0F));

    backend.process(input, output, nullptr);

    EXPECT_TRUE(all_samples_equal(output[0], 0.25F));
    // An output tensor with no matching input carries no meaningful roundtrip
    // data: it is cleared rather than left with stale samples.
    EXPECT_TRUE(all_samples_equal(output[1], 0.0F));
}
