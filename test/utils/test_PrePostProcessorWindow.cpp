// Default PrePostProcessor sliding-window behavior: when the input tensor
// holds more samples than preprocess_input_size (a receptive-field model),
// pre_process must fill the head of the window with ring-buffer history and
// only the tail with fresh samples — previously this required a custom
// PrePostProcessor subclass.

#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr size_t k_window = 8;  // tensor input size
constexpr size_t k_hop = 4;     // fresh samples per inference

anira::InferenceConfig make_window_config() {
    const std::vector<anira::ModelData> model_data = {
        {"unused-by-this-test.pte", anira::InferenceBackend::CUSTOM},
    };
    const std::vector<anira::TensorShape> tensor_shapes = {
        {{{1, 1, static_cast<int64_t>(k_window)}}, {{1, 1, static_cast<int64_t>(k_hop)}}},
    };
    const anira::ProcessingSpec processing_spec({1}, {1}, {k_hop}, {k_hop});
    return {model_data, tensor_shapes, processing_spec, 5.0F};
}

}  // namespace

TEST(PrePostProcessorWindow, HistoryFillsTheWindowHead) {
    anira::InferenceConfig config = make_window_config();
    anira::PrePostProcessor pp(config);

    std::vector<anira::RingBuffer> input(1);
    // Ring must hold the fresh samples plus the window history, like
    // SessionElement::calculate_send_buffer_sizes reserves.
    input[0].initialize_with_positions(1, k_window + k_hop);
    std::vector<anira::BufferF> model_input(1);
    model_input[0].resize(1, k_window);

    // First inference: push 1..4. History is silence, so the window head must
    // be zeros and the tail the fresh samples.
    for (size_t i = 0; i < k_hop; ++i) { input[0].push_sample(0, static_cast<float>(i + 1)); }
    pp.pre_process(input, model_input, anira::InferenceBackend::CUSTOM);
    for (size_t i = 0; i < k_window - k_hop; ++i) {
        EXPECT_FLOAT_EQ(model_input[0].get_sample(0, i), 0.0F) << "head sample " << i;
    }
    for (size_t i = 0; i < k_hop; ++i) {
        EXPECT_FLOAT_EQ(model_input[0].get_sample(0, (k_window - k_hop) + i),
                        static_cast<float>(i + 1))
            << "fresh sample " << i;
    }

    // Second inference: push 5..8. The window must now be 1..8 — the previous
    // hop as history, the new hop as tail.
    for (size_t i = 0; i < k_hop; ++i) {
        input[0].push_sample(0, static_cast<float>(k_hop + i + 1));
    }
    pp.pre_process(input, model_input, anira::InferenceBackend::CUSTOM);
    for (size_t i = 0; i < k_window; ++i) {
        EXPECT_FLOAT_EQ(model_input[0].get_sample(0, i), static_cast<float>(i + 1))
            << "window sample " << i;
    }
}
