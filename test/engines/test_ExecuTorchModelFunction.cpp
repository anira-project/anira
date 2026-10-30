// model_function on the EXECUTORCH backend: a .pte can carry several named
// entry points, and the configured method — not a hardcoded "forward" — must
// run. The multi-function SimpleGainNetwork fixture (example-models) packs
// forward/gain2/gain4 (x*1, x*2, x*4 on [1, 1, 64]) into one program, so the
// output gain proves which graph executed.

#ifdef USE_EXECUTORCH

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/utils/JsonConfigLoader.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../abi/handler_support.h"

namespace {

constexpr size_t k_size = 64;
constexpr int k_timeout_secs = 5;

// The multi-function program on [1, 1, 64] in and out, its entry point the method named (none:
// the file's default).
anira::ModelConfig make_model(const std::string& model_function) {
    anira::ModelConfig model;
    const uint32_t row = model.add_model_path(
        ANIRA_ENGINE_EXECUTORCH,
        std::string(ANIRA_EXTRAS_MODELS_DIR "/model-pool/example-models/SimpleGainNetwork/models") +
            "/simple_gain_network_multifunction.pte");
    if (!model_function.empty()) {
        model.model_ext(row, anira::ext::Entry{.name = model_function});
    }
    const auto stream = [](const char* name) {
        anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        spec.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, 1)
            .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_size))
            .window(static_cast<int64_t>(k_size), static_cast<int64_t>(k_size), 0);
        return spec;
    };
    model.input(stream("in"));
    model.output(stream("out"));
    return model;
}

// Push one block of ones and wait until its inference result has been consumed
// back into the ring (available samples return to the pre-push level).
void process_block(anira_handler* handler, std::vector<float>& io) {
    const std::array<float*, 1> channels = {io.data()};
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_size)};
    anira_tensor block{};
    anira_tensor_init_host_planar(&block,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    const size_t prev = anira_test::available(handler);
    size_t delivered = 0;
    ASSERT_EQ(anira_handler_process(handler, &block, 0, &block, 0, &delivered), ANIRA_OK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(k_timeout_secs);
    while (anira_test::available(handler) != prev) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out waiting for inference";
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// Stream ones through the configured method until the latency drains, then
// return the steady-state output value.
float steady_state_output(const std::string& model_function) {
    const anira_test::Context context(2);
    const std::vector<anira_backend_id> executorch{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_EXECUTORCH,
                                                    .provider = ANIRA_PROVIDER_CPU,
                                                    .engine_id = nullptr}};
    anira_test::Handler handler(context, make_model(model_function), executorch);
    EXPECT_EQ(handler.prepare(anira_test::explicit_contract(k_size,
                                                            anira_test::k_rate,
                                                            ANIRA_MISS_BYPASS,
                                                            0.0,
                                                            /*budget_ms=*/5.0)),
              ANIRA_OK)
        << handler.m_err.message;

    const int warmup_blocks =
        static_cast<int>(anira_handler_get_latency(handler.m_handler, 0) / k_size) + 2;
    std::vector<float> io;
    for (int i = 0; i < warmup_blocks + 1; ++i) {
        io.assign(k_size, 1.0F);
        process_block(handler.m_handler, io);
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
    std::istringstream json{
        R"({
        "inference_config": {
            "model_data": [
                { "model_path": ")" +
        std::string(ANIRA_EXTRAS_MODELS_DIR "/model-pool/example-models/SimpleGainNetwork/models") +
        R"(/simple_gain_network_multifunction.pte",
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
