// model_function on the EXECUTORCH backend: a .pte can carry several named
// entry points, and the configured method — not a hardcoded "forward" — must
// run. The multi-function SimpleGainNetwork fixture (example-models) packs
// forward/gain2/gain4 (x*1, x*2, x*4 on [1, 1, 64]) into one program, so the
// output gain proves which graph executed.

#ifdef USE_EXECUTORCH

#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/JsonConfigLoader.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t k_size = 64;
constexpr int k_timeout_secs = 5;

anira::InferenceConfig make_config(const std::string& model_function) {
    const std::vector<anira::ModelData> model_data = {
        {std::string(SIMPLEGAIN_MODEL_PATH) + "/simple_gain_network_multifunction.pte",
         anira::InferenceBackend::EXECUTORCH,
         model_function},
    };
    const std::vector<anira::TensorShape> tensor_shapes = {
        {{{1, 1, static_cast<int64_t>(k_size)}}, {{1, 1, static_cast<int64_t>(k_size)}}},
    };
    return {model_data, tensor_shapes, 5.0F};
}

// Push one block of ones and wait until its inference result has been consumed
// back into the ring (available samples return to the pre-push level).
void process_block(anira::InferenceHandler& handler, std::vector<float>& io) {
    std::array<float*, 1> channels = {io.data()};
    const size_t prev = handler.get_available_samples(0);
    handler.process(channels.data(), k_size);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_secs);
    while (handler.get_available_samples(0) != prev) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out waiting for inference";
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// Stream ones through the configured method until the latency drains, then
// return the steady-state output value.
float steady_state_output(const std::string& model_function) {
    anira::InferenceConfig config = make_config(model_function);
    anira::PrePostProcessor pp(config);
    anira::InferenceHandler handler(pp, config);
    handler.prepare({static_cast<float>(k_size), 48000.0F});
    handler.set_inference_backend(anira::InferenceBackend::EXECUTORCH);

    const int warmup_blocks = static_cast<int>(handler.get_latency() / k_size) + 2;
    std::vector<float> io;
    for (int i = 0; i < warmup_blocks + 1; ++i) {
        io.assign(k_size, 1.0F);
        process_block(handler, io);
    }
    for (size_t i = 1; i < k_size; ++i) {
        EXPECT_FLOAT_EQ(io[i], io[0]) << "output not uniform at sample " << i;
    }
    return io[0];
}

}  // namespace

TEST(ExecuTorchModelFunction, DefaultsToForward) {
    EXPECT_FLOAT_EQ(steady_state_output(""), 1.0F);
}

TEST(ExecuTorchModelFunction, NamedMethodSelectsItsGraph) {
    EXPECT_FLOAT_EQ(steady_state_output("gain2"), 2.0F);
    EXPECT_FLOAT_EQ(steady_state_output("gain4"), 4.0F);
}

TEST(ExecuTorchModelFunction, JsonConfigCarriesModelFunction) {
    std::istringstream json{R"({
        "inference_config": {
            "model_data": [
                { "model_path": ")" +
                            std::string(MULTIFUNCTION_GAIN_MODEL_PATH) +
                            R"(/multi_function_gain.pte",
                  "inference_backend": "EXECUTORCH",
                  "model_function": "gain2" }
            ],
            "tensor_shape": [
                { "input_shape": [[1, 1, 64]], "output_shape": [[1, 1, 64]] }
            ],
            "max_inference_time": 5.0
        }
    })"};
    anira::JsonConfigLoader loader(json);
    auto config = loader.get_inference_config();
    ASSERT_NE(config, nullptr);
    ASSERT_EQ(config->m_model_data.size(), 1U);
    EXPECT_EQ(config->m_model_data.front().m_model_function, "gain2");
}

#endif  // USE_EXECUTORCH
