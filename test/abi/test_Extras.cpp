// The 3.x extras on the C handler, through anira::Pipeline: extras/models/hybrid-nn/
// HybridNNStage.h (the windows of the HybridNN model), extras/models/BypassEngine.h (the bypass
// of the bundled audio models) and examples/benchmark/advanced-benchmark/ClearEngine.h (the
// benchmark baseline). The oracle is the stream itself: the bypass engine hands each window's
// newest sample to the output, so the output is the input delayed by the handler's latency; a
// test engine that hands out each window's oldest sample instead sees the history the stage put
// ahead of it, 149 samples further back. Every block waits for its own inference, so no block is
// missed and the comparison is exact.

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "../../examples/benchmark/advanced-benchmark/ClearEngine.h"
#include "../../extras/models/BypassEngine.h"
#include "../../extras/models/hybrid-nn/HybridNNStage.h"
#include "../../extras/models/model_files.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

constexpr size_t k_hybridnn_hop = 256;  // hybridnn.model.json: 256 windows per inference
constexpr size_t k_hybridnn_old = 149;  // each window's history (overlap)
constexpr size_t k_blocks = 12;
constexpr const char* k_oldest_engine_id = "org.anira.test.oldest";

/// The first sample of each row of the input window into the output, one per row: the window's
/// oldest sample, where BypassEngine takes the newest.
class OldestEngine final : public anira::Engine {
public:
    OldestEngine() : anira::Engine(k_oldest_engine_id) {}
    uint32_t flags() const noexcept override {
        return ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    }
    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& /*info*/) override {
        return std::make_unique<Loaded>();
    }

private:
    class Prepared final : public anira::Engine::Prepared {
    public:
        anira_status process(anira::EngineContext& ctx) noexcept override {
            const anira::Tensor& in = ctx.inputs()[0];
            const anira::Tensor& out = ctx.outputs()[0];
            const float* source = in.data_f32();
            float* target = out.data_f32();
            if (source == nullptr || target == nullptr) { return ANIRA_ERROR_CONFIG; }
            const size_t row = in.extent(in.ndim - 1);
            const size_t rows = out.num_elements();
            for (size_t r = 0; r < rows; ++r) { target[r] = source[r * row]; }
            return ANIRA_OK;
        }
    };
    class Loaded final : public anira::Engine::Loaded {
    public:
        std::unique_ptr<anira::Engine::Prepared> prepare(
            const anira::PrepareInfo& /*info*/) override {
            return std::make_unique<Prepared>();
        }
    };
};

/// A C handler over an anira::Pipeline: the inference stage on one engine, by id, and the
/// stage when one is given.
struct PipelineHandler {
    PipelineHandler(const anira_test::Context& context,
                    const anira::ModelConfig& model,
                    const std::shared_ptr<anira::Engine>& engine,
                    const std::shared_ptr<anira::Stage>& stage)
        : m_engine(engine), m_stage(stage) {
        const anira::BackendId candidate{.struct_size = sizeof(anira_backend_id),
                                         .engine = ANIRA_ENGINE_CUSTOM,
                                         .provider = ANIRA_PROVIDER_CPU,
                                         .engine_id = engine->id().c_str()};
        anira::Pipeline pipeline;
        pipeline.add(anira::stage::Inference(model, {candidate}).engine(engine));
        if (stage) { pipeline.add(anira::stage::Custom(stage)); }
        EXPECT_EQ(anira_handler_create(context.m_context, pipeline.native(), &m_handler, &m_err),
                  ANIRA_OK)
            << m_err.message;
    }
    ~PipelineHandler() { anira_handler_destroy(m_handler); }
    PipelineHandler(const PipelineHandler&) = delete;
    PipelineHandler& operator=(const PipelineHandler&) = delete;

    // The registrations outlive the handler (they are the test's objects).
    std::shared_ptr<anira::Engine> m_engine;
    std::shared_ptr<anira::Stage> m_stage;
    anira_handler* m_handler = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// The HybridNN model with one source-less entry of the engine.
anira::ModelConfig hybridnn_with(const char* engine_id) {
    anira::ModelConfig model = anira::ModelConfig::from_file(k_hybridnn_model_json);
    model.add_model(engine_id);
    return model;
}

/// k_blocks ramp blocks of one hop through the handler, each waited for, and the output stream.
std::vector<float> stream_through(anira_handler* handler, size_t hop) {
    anira_test::FloatFace face(handler);
    std::vector<float> output;
    for (size_t k = 0; k < k_blocks; ++k) {
        std::vector<float> block = anira_test::ramp(k + 1, hop);
        const size_t prev = anira_test::available(handler);
        const std::array<float*, 1> channels{block.data()};
        size_t delivered = 0;
        EXPECT_EQ(face.process_inplace(channels.data(), hop, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, hop) << "block " << k;
        anira_test::wait_for_block(handler, prev);
        output.insert(output.end(), block.begin(), block.end());
    }
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    return output;
}

/// The input stream stream_through sends, delayed by `delay` samples (zeros ahead of it).
std::vector<float> delayed_input(size_t hop, size_t delay) {
    std::vector<float> input;
    for (size_t k = 0; k < k_blocks; ++k) {
        const std::vector<float> block = anira_test::ramp(k + 1, hop);
        input.insert(input.end(), block.begin(), block.end());
    }
    std::vector<float> expected(input.size(), 0.0F);
    for (size_t n = delay; n < input.size(); ++n) { expected[n] = input[n - delay]; }
    return expected;
}

}  // namespace

// The stage's windows end in the hop's new samples, in order: the bypass of each window's newest
// sample delivers the input stream, delayed by the latency the handler reports.
TEST(Extras, TheHybridNnStageAndTheBypassEngineDelayTheInput) {
    const anira_test::Context context;
    const anira::ModelConfig model = hybridnn_with(k_bypass_engine_id);
    const PipelineHandler handler(context,
                                  model,
                                  std::make_shared<BypassEngine>(),
                                  std::make_shared<HybridNNStage>());
    ASSERT_NE(handler.m_handler, nullptr);
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(
                  handler.m_handler,
                  anira_test::file_contract(k_hybridnn_contract_json, k_hybridnn_hop).native(),
                  &err),
              ANIRA_OK)
        << err.message;
    const uint32_t latency = anira_handler_get_latency(handler.m_handler, 0);
    const std::vector<float> output = stream_through(handler.m_handler, k_hybridnn_hop);
    ASSERT_FALSE(::testing::Test::HasFailure());
    anira_test::expect_same_block(output, delayed_input(k_hybridnn_hop, latency), 0);
}

// The history ahead of each window: its oldest sample is 149 samples before its newest, the
// ring's history across the inferences included, so the oldest-sample engine delivers the input
// delayed by the latency and the window's history.
TEST(Extras, TheHybridNnStagePutsTheHistoryAheadOfEachWindow) {
    const anira_test::Context context;
    const anira::ModelConfig model = hybridnn_with(k_oldest_engine_id);
    const PipelineHandler handler(context,
                                  model,
                                  std::make_shared<OldestEngine>(),
                                  std::make_shared<HybridNNStage>());
    ASSERT_NE(handler.m_handler, nullptr);
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(
                  handler.m_handler,
                  anira_test::file_contract(k_hybridnn_contract_json, k_hybridnn_hop).native(),
                  &err),
              ANIRA_OK)
        << err.message;
    const uint32_t latency = anira_handler_get_latency(handler.m_handler, 0);
    const std::vector<float> output = stream_through(handler.m_handler, k_hybridnn_hop);
    ASSERT_FALSE(::testing::Test::HasFailure());
    anira_test::expect_same_block(output,
                                  delayed_input(k_hybridnn_hop, latency + k_hybridnn_old),
                                  0);
}

// The CNN needs no stage: anira's default fill is the receptive-field window the 2.x
// CNNPrePostProcessor built (test_Stage's NoStageEqualsTheV2Default, TheCnnProcessorIsTheDefault).
// The bypass of its 2048 newest samples delivers the input delayed by the latency.
TEST(Extras, TheBypassEngineDelaysTheCnnInput) {
    constexpr size_t k_cnn_hop = 2048;
    const anira_test::Context context;
    anira::ModelConfig model = anira::ModelConfig::from_file(k_cnn_model_json);
    model.add_model(k_bypass_engine_id);
    const PipelineHandler handler(context, model, std::make_shared<BypassEngine>(), nullptr);
    ASSERT_NE(handler.m_handler, nullptr);
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(
        anira_handler_prepare(handler.m_handler,
                              anira_test::file_contract(k_cnn_contract_json, k_cnn_hop).native(),
                              &err),
        ANIRA_OK)
        << err.message;
    const uint32_t latency = anira_handler_get_latency(handler.m_handler, 0);
    const std::vector<float> output = stream_through(handler.m_handler, k_cnn_hop);
    ASSERT_FALSE(::testing::Test::HasFailure());
    anira_test::expect_same_block(output, delayed_input(k_cnn_hop, latency), 0);
}

// The benchmark baseline writes zeros and runs no model.
TEST(Extras, TheClearEngineZeroesTheOutput) {
    const anira_test::Context context;
    const anira::ModelConfig model = hybridnn_with(k_clear_engine_id);
    const PipelineHandler handler(context,
                                  model,
                                  std::make_shared<ClearEngine>(),
                                  std::make_shared<HybridNNStage>());
    ASSERT_NE(handler.m_handler, nullptr);
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(
                  handler.m_handler,
                  anira_test::file_contract(k_hybridnn_contract_json, k_hybridnn_hop).native(),
                  &err),
              ANIRA_OK)
        << err.message;
    const std::vector<float> output = stream_through(handler.m_handler, k_hybridnn_hop);
    ASSERT_FALSE(::testing::Test::HasFailure());
    anira_test::expect_all(output, 0.0F, "the clear engine's output");
}
