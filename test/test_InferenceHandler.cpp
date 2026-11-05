// anira::InferenceHandler of anira/anira.hpp end to end on GuitarLSTM (extras/models/hybrid-nn):
// the model file, its contract with the host blocks, a Pipeline of the inference stage (the
// engine of the instantiation, or the bypass engine of extras/models/BypassEngine.h) and the
// model's windows (HybridNNProcessor), the handler's Hard entries over one planar float32 tensor.
// The outputs are compared with the reference renders of the model directory, delayed by the
// handler's latency; the bypass instantiations compare the input with itself.
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <tanh/core/Buffer.h>
#include <tanh/core/WavReader.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <ios>
#include <memory>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../extras/models/BypassEngine.h"
#include "../extras/models/hybrid-nn/HybridNNProcessor.h"
#include "../extras/models/model_files.h"
#include "gtest/gtest.h"

// The GuitarLSTM export directories: the wav files the round trips compare against.
constexpr const char* k_guitarlstm_pytorch =
    ANIRA_EXTRAS_MODELS_DIR "/hybrid-nn/GuitarLSTM/pytorch-version/models";
constexpr const char* k_guitarlstm_tensorflow =
    ANIRA_EXTRAS_MODELS_DIR "/hybrid-nn/GuitarLSTM/tensorflow-version/models";

constexpr int k_inference_timeout_s = 2;

namespace {

// Load a mono WAV file into a flat vector.
std::vector<float> load_mono_wav(const std::string& path) {
    std::string error;
    const thl::core::BufferF buffer = thl::core::read_wav(path, &error);
    if (buffer.get_num_channels() == 0) {
        ADD_FAILURE() << "Cannot read " << path << ": " << error;
        return {};
    }
    if (buffer.get_num_channels() != 1) {
        ADD_FAILURE() << path << " has " << buffer.get_num_channels()
                      << " channels, the test fixtures must be mono";
        return {};
    }
    const float* samples = buffer.get_read_pointer(0);
    return {samples, samples + buffer.get_num_samples()};
}

}  // namespace

struct InferenceTestParams {
    anira::EngineKind m_engine;  ///< ANIRA_ENGINE_CUSTOM: the bypass engine
    double m_block;
    double m_host_rate;
    std::string m_input_data_path;
    std::string m_reference_data_path;
    size_t m_reference_data_offset;
    float m_epsilon_rel = 1e-6F;
    float m_epsilon_abs = 1e-7F;
};

namespace {

std::ostream& operator<<(std::ostream& stream, const InferenceTestParams& params) {
    std::string engine;
    switch (params.m_engine) {
        case ANIRA_ENGINE_LIBTORCH: engine = "libtorch"; break;
        case ANIRA_ENGINE_ONNXRUNTIME: engine = "ONNX"; break;
        case ANIRA_ENGINE_TFLITE: engine = "TFlite"; break;
        case ANIRA_ENGINE_LITERT: engine = "LiteRt"; break;
        case ANIRA_ENGINE_EXECUTORCH: engine = "ExecuTorch"; break;
        case ANIRA_ENGINE_CUSTOM: engine = "bypass"; break;
        default: engine = "unknown"; break;
    }
    stream << "{ engine = " << engine << ", block = " << params.m_block
           << ", host_rate = " << params.m_host_rate << " }";
    return stream;
}

/// One handler on GuitarLSTM: the model file plus the bypass engine's entry, the contract file
/// with the instantiation's host blocks, a Pipeline of the inference stage on the one engine of the
/// instantiation and HybridNNProcessor, prepared.
struct HybridNnHandler {
    explicit HybridNnHandler(const InferenceTestParams& params,
                             anira_model_state state = ANIRA_MODEL_STATELESS,
                             const uint32_t* declared_latency = nullptr) {
        m_model.add_model(k_bypass_engine_id);
        m_model.state(state);
        m_contract.hard_host_blocks(params.m_block, params.m_block, params.m_host_rate);
        if (declared_latency != nullptr) {
            m_contract.hard_latency("audio_out", *declared_latency);
        }
        const bool bypass = params.m_engine == ANIRA_ENGINE_CUSTOM;
        const anira::BackendId candidate{.struct_size = sizeof(anira::BackendId),
                                         .engine = static_cast<uint32_t>(params.m_engine),
                                         .provider = ANIRA_PROVIDER_CPU,
                                         .engine_id = bypass ? k_bypass_engine_id : nullptr};
        anira::Pipeline pipeline;
        pipeline.add_engine(std::make_shared<BypassEngine>());
        pipeline.add_inference(m_model, {candidate});
        pipeline.add_processor(std::make_shared<HybridNNProcessor>());
        m_handler = std::make_unique<anira::InferenceHandler>(m_context, pipeline);
        const anira::PlanReport report = m_handler->prepare(m_contract);
        EXPECT_EQ(report.num_plans(), 1U) << "one candidate, one plan";
        // The planar tensor over the block, built once: one channel of m_block samples.
        const std::array<int64_t, 2> extents{1, static_cast<int64_t>(params.m_block)};
        m_block.assign(static_cast<size_t>(params.m_block), 0.F);
        m_plane = m_block.data();
        m_tensor =
            anira::Tensor::from_host_planar<float>(std::span<float* const>(&m_plane, 1), extents);
    }

    size_t available() {
        size_t count = 0;
        EXPECT_EQ(m_handler->get_available_samples(0, 0, &count), ANIRA_OK);
        return count;
    }

    /// Waits until the output ring holds `expected` samples again; false on a timeout.
    bool wait_for_available(size_t expected) {
        const auto start = std::chrono::steady_clock::now();
        while (available() != expected) {
            if (std::chrono::steady_clock::now() >
                start + std::chrono::seconds(k_inference_timeout_s)) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        }
        return true;
    }

    anira::Context m_context{anira::ContextConfig{}};
    anira::ModelConfig m_model = anira::ModelConfig::from_file(k_hybridnn_model_json);
    anira::ContractHandle m_contract = anira::ContractHandle::from_file(k_hybridnn_contract_json);
    std::unique_ptr<anira::InferenceHandler> m_handler;
    std::vector<float> m_block;
    float* m_plane = nullptr;
    anira::Tensor m_tensor{};
};

/// The reference delayed by the latency: `delay` zeros first, then the reference stream.
class DelayedReference {
public:
    explicit DelayedReference(size_t delay) : m_samples(delay, 0.F) {}
    void push(float sample) { m_samples.push_back(sample); }
    float pop() {
        const float sample = m_samples.front();
        m_samples.pop_front();
        return sample;
    }

private:
    std::deque<float> m_samples;
};

/// Compares one processed block with the delayed reference; the first `delay` samples of the
/// stream are zeros on both sides.
void expect_block(const InferenceTestParams& params,
                  DelayedReference& reference,
                  const std::vector<float>& processed,
                  size_t repeat,
                  size_t delay,
                  const char* phase) {
    for (size_t i = 0; i < processed.size(); i++) {
        const float expected = reference.pop();
        const size_t n = (repeat * processed.size()) + i;
        if (n < delay) {
            ASSERT_FLOAT_EQ(expected, 0);
        } else {
            const float epsilon =
                (std::max(std::abs(expected), std::abs(processed[i])) * params.m_epsilon_rel) +
                params.m_epsilon_abs;
            ASSERT_NEAR(expected, processed[i], epsilon)
                << phase << "repeat=" << repeat << ", i=" << i << ", total sample nr: " << n;
        }
    }
}

}  // namespace

// Test fixture for paramterized inference tests
class InferenceTest : public ::testing::TestWithParam<InferenceTestParams> {};

// The declared-latency and reset-hammer contracts live in the scheduler, not in an engine:
// neither path branches on the engine, and the stale-inference path never reaches one. They run
// over the bypass (fastest) and the LibTorch (slowest, widest mid-flight reset window)
// instantiations instead of all engines.
class InferenceControlTest : public ::testing::TestWithParam<InferenceTestParams> {};

TEST_P(InferenceTest, Simple) {
    const InferenceTestParams& params = GetParam();
    const std::vector<float> data_input = load_mono_wav(params.m_input_data_path);
    const std::vector<float> data_reference = load_mono_wav(params.m_reference_data_path);
    ASSERT_TRUE(!data_input.empty());
    ASSERT_TRUE(!data_reference.empty());

    HybridNnHandler h(params);
    const size_t buffer_size = h.m_block.size();
    const auto latency = static_cast<size_t>(h.m_handler->get_latency());
    const size_t delay = latency + params.m_reference_data_offset;
    DelayedReference reference(delay);

    for (size_t repeat = 0; repeat < 150; repeat++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((repeat * buffer_size) + i);
            reference.push(data_reference.at((repeat * buffer_size) + i));
        }
        const size_t prev_samples = h.available();
        ASSERT_EQ(h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0), ANIRA_OK);
        ASSERT_TRUE(h.wait_for_available(prev_samples))
            << "Timeout while waiting for block to be processed";
        expect_block(params, reference, h.m_block, repeat, delay, "");
    }
}

TEST_P(InferenceControlTest, WithDeclaredLatency) {
    const InferenceTestParams& params = GetParam();
    const std::vector<float> data_input = load_mono_wav(params.m_input_data_path);
    const std::vector<float> data_reference = load_mono_wav(params.m_reference_data_path);
    ASSERT_TRUE(!data_input.empty());
    ASSERT_TRUE(!data_reference.empty());

    // A declared stream latency of 0 (anira_contract_hard_set_latency): the output ring starts
    // empty, and a push is popped once its inference has landed.
    const uint32_t declared = 0;
    HybridNnHandler h(params, ANIRA_MODEL_STATELESS, &declared);
    const size_t buffer_size = h.m_block.size();
    ASSERT_EQ(h.m_handler->get_latency(), 0U) << "the declared latency replaces the computed one";
    DelayedReference reference(params.m_reference_data_offset);

    for (size_t repeat = 0; repeat < 150; repeat++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((repeat * buffer_size) + i);
            reference.push(data_reference.at((repeat * buffer_size) + i));
        }
        const size_t prev_samples = h.available();
        ASSERT_EQ(h.m_handler->push_data(h.m_tensor, 0), ANIRA_OK);
        ASSERT_TRUE(h.wait_for_available(prev_samples + buffer_size))
            << "Timeout while waiting for block to be processed";
        ASSERT_EQ(h.m_handler->pop_data(h.m_tensor, 0), ANIRA_OK);
        expect_block(params, reference, h.m_block, repeat, params.m_reference_data_offset, "");
    }
}

TEST_P(InferenceTest, Reset) {
    const InferenceTestParams& params = GetParam();
    const std::vector<float> data_input = load_mono_wav(params.m_input_data_path);
    const std::vector<float> data_reference = load_mono_wav(params.m_reference_data_path);
    ASSERT_TRUE(!data_input.empty());
    ASSERT_TRUE(!data_reference.empty());

    HybridNnHandler h(params);
    const size_t buffer_size = h.m_block.size();
    const auto latency = static_cast<size_t>(h.m_handler->get_latency());
    const size_t delay = latency + params.m_reference_data_offset;

    // First, process some data to "contaminate" the internal state
    DelayedReference reference(delay);
    for (size_t repeat = 0; repeat < 50; repeat++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((repeat * buffer_size) + i);
            reference.push(data_reference.at((repeat * buffer_size) + i));
        }
        const size_t prev_samples = h.available();
        ASSERT_EQ(h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0), ANIRA_OK);
        ASSERT_TRUE(h.wait_for_available(prev_samples))
            << "Timeout while waiting for block to be processed";
        expect_block(params, reference, h.m_block, repeat, delay, "");
    }

    // Now reset the handler: the stream restarts at the latency.
    h.m_handler->reset();
    EXPECT_EQ(h.available(), latency)
        << "Available samples should be " << latency << " after reset";

    // Process data again and verify that output matches reference from the beginning
    DelayedReference restarted(delay);
    for (size_t repeat = 0; repeat < 150; repeat++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((repeat * buffer_size) + i);
            restarted.push(data_reference.at((repeat * buffer_size) + i));
        }
        const size_t prev_samples = h.available();
        ASSERT_EQ(h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0), ANIRA_OK);
        ASSERT_TRUE(h.wait_for_available(prev_samples))
            << "Timeout while waiting for block to be processed";
        expect_block(params, restarted, h.m_block, repeat, delay, "After reset: ");
    }
}

// Hammers the wait-free reset() on a session-exclusive (stateful) session at varying phases
// relative to in-flight inference. Stresses the stateful dispatch-chain reconciliation: resets
// right after submission hit a busy gate (the worker filters the stale pending prefix at its task
// boundary), resets at idle hit a free gate (the calling thread filters), and back-to-back resets
// exercise repeated generation bumps. Recovery is then verified functionally: a wedged dispatch
// gate would time out the per-block completion wait below, and unreclaimed structs would exhaust
// the free pool and corrupt the output.
TEST_P(InferenceControlTest, ResetStatefulHammer) {
    const InferenceTestParams& params = GetParam();
    const std::vector<float> data_input = load_mono_wav(params.m_input_data_path);
    const std::vector<float> data_reference = load_mono_wav(params.m_reference_data_path);
    ASSERT_TRUE(!data_input.empty());
    ASSERT_TRUE(!data_reference.empty());

    // The point of this test: strict in-order, mutually-exclusive dispatch.
    HybridNnHandler h(params, ANIRA_MODEL_STATEFUL);
    const size_t buffer_size = h.m_block.size();
    const auto latency = static_cast<size_t>(h.m_handler->get_latency());

    // Hammer phase: output is discarded; only the reset/dispatch interplay matters.
    for (size_t round = 0; round < 20; round++) {
        for (size_t block = 0; block < 5; block++) {
            for (size_t i = 0; i < buffer_size; i++) {
                h.m_block[i] = data_input.at((block * buffer_size) + i);
            }
            h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0);
            if (block == 2) {
                // Likely mid-flight: the exclusive task submitted above may hold the dispatch
                // gate, forcing the worker-boundary filtering path.
                h.m_handler->reset();
            }
        }
        h.m_handler->reset();
        h.m_handler->reset();  // back-to-back resets must be harmless
        EXPECT_EQ(h.available(), latency)
            << "Available samples should re-anchor to the latency after reset (round " << round
            << ")";
    }

    // Reconciliation barrier: a wait-free reset leaves the structures of in-flight and pending
    // stale work captive until the worker's next task boundary, so the pool can be transiently
    // exhausted right after the hammer: fresh chunks submitted in that window complete as
    // silence at their stream positions (time-aligned, self-recovering). Let the stale work
    // finish, then verify the chain still processes end-to-end: with a wedged dispatch gate the
    // fresh chunks below would never complete and the wait would time out.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    for (size_t block = 0; block < 3; block++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((block * buffer_size) + i);
        }
        const size_t prev_samples = h.available();
        h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0);
        ASSERT_TRUE(h.wait_for_available(prev_samples))
            << "Timeout while reconciling after the reset hammer: the stateful dispatch chain "
               "may be wedged (block "
            << block << ")";
    }

    // Re-anchor from a quiescent state: nothing is in flight anymore, so this reset leaves a
    // fully free pool and the reference phase below is deterministic.
    h.m_handler->reset();
    EXPECT_EQ(h.available(), latency)
        << "Available samples should re-anchor to the latency after the quiescent reset";

    // Recovery phase: after all that, the session must behave exactly like a freshly prepared
    // one: every block completes (a wedged chain would hit the completion timeout) and the
    // output matches the reference from the start.
    const size_t delay = latency + params.m_reference_data_offset;
    DelayedReference reference(delay);
    for (size_t repeat = 0; repeat < 100; repeat++) {
        for (size_t i = 0; i < buffer_size; i++) {
            h.m_block[i] = data_input.at((repeat * buffer_size) + i);
            reference.push(data_reference.at((repeat * buffer_size) + i));
        }
        const size_t prev_samples = h.available();
        h.m_handler->process(h.m_tensor, 0, h.m_tensor, 0);
        ASSERT_TRUE(h.wait_for_available(prev_samples))
            << "Timeout while waiting for block to be processed: the stateful dispatch chain "
               "may be wedged (repeat "
            << repeat << ")";
        expect_block(params, reference, h.m_block, repeat, delay, "After reset hammer: ");
    }
}

namespace {
std::string build_test_name(const testing::TestParamInfo<InferenceTest::ParamType>& info) {
    std::stringstream ss_sample_rate;
    std::stringstream ss_buffer_size;

    // Set precision to 4 decimal places for cleaner names
    ss_sample_rate << std::fixed << std::setprecision(4) << info.param.m_host_rate;
    ss_buffer_size << std::fixed << std::setprecision(4) << info.param.m_block;

    std::string sample_rate_str = ss_sample_rate.str();
    std::string buffer_size_str = ss_buffer_size.str();

    // Replace decimal points with underscores to make valid test names
    std::ranges::replace(sample_rate_str, '.', '_');
    std::ranges::replace(buffer_size_str, '.', '_');

    return sample_rate_str + "x" + buffer_size_str;
}

/// The two host block ranges of every instantiation, over one engine and one pair of wav files.
auto params_of(anira::EngineKind engine,
               const std::string& input,
               const std::string& reference,
               size_t offset,
               float epsilon_rel,
               float epsilon_abs) {
    const auto at = [&](double block) {
        return InferenceTestParams{.m_engine = engine,
                                   .m_block = block,
                                   .m_host_rate = 44100,
                                   .m_input_data_path = input,
                                   .m_reference_data_path = reference,
                                   .m_reference_data_offset = offset,
                                   .m_epsilon_rel = epsilon_rel,
                                   .m_epsilon_abs = epsilon_abs};
    };
    return ::testing::Values(at(2048), at(256));
}
}  // namespace

// The bypass engine delivers each window's newest sample: the input, delayed by the latency.
INSTANTIATE_TEST_SUITE_P(InferenceBypass,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_CUSTOM,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   0,
                                   FLT_EPSILON,
                                   0),
                         build_test_name);

INSTANTIATE_TEST_SUITE_P(InferenceBypass,
                         InferenceControlTest,
                         params_of(ANIRA_ENGINE_CUSTOM,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   0,
                                   FLT_EPSILON,
                                   0),
                         build_test_name);

#ifdef USE_LIBTORCH
INSTANTIATE_TEST_SUITE_P(InferenceLibtorch,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_LIBTORCH,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/y_pred.wav",
                                   149,
                                   1e-6F,
                                   2e-7F),
                         build_test_name);

INSTANTIATE_TEST_SUITE_P(InferenceLibtorch,
                         InferenceControlTest,
                         params_of(ANIRA_ENGINE_LIBTORCH,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/y_pred.wav",
                                   149,
                                   1e-6F,
                                   2e-7F),
                         build_test_name);
#endif

#ifdef USE_ONNXRUNTIME
INSTANTIATE_TEST_SUITE_P(InferenceOnnx,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_ONNXRUNTIME,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/y_pred.wav",
                                   149,
                                   1e-6F,
                                   2e-7F),
                         build_test_name);
#endif

#ifdef USE_TFLITE
INSTANTIATE_TEST_SUITE_P(InferenceTflite,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_TFLITE,
                                   std::string(k_guitarlstm_tensorflow) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_tensorflow) +
                                       "/model_0/y_pred_tflite.wav",
                                   149,
                                   1e-6F,
                                   2e-7F),
                         build_test_name);
#endif

// LiteRT loads the same GuitarLSTM .tflite flatbuffer as the TFLite engine, so it reuses the
// y_pred_tflite.wav reference. Mirrors InferenceTflite above.
#ifdef USE_LITERT
INSTANTIATE_TEST_SUITE_P(InferenceLiteRt,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_LITERT,
                                   std::string(k_guitarlstm_tensorflow) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_tensorflow) +
                                       "/model_0/y_pred_tflite.wav",
                                   149,
                                   1e-6F,
                                   2e-7F),
                         build_test_name);
#endif

// The ExecuTorch .pte is exported ahead-of-time from the same PyTorch weights as the LibTorch
// model, so it reuses the y_pred.wav reference. The looser epsilons cover XNNPACK's different
// (but still float32) op implementations.
#ifdef USE_EXECUTORCH
INSTANTIATE_TEST_SUITE_P(InferenceExecuTorch,
                         InferenceTest,
                         params_of(ANIRA_ENGINE_EXECUTORCH,
                                   std::string(k_guitarlstm_pytorch) + "/model_0/x_test.wav",
                                   std::string(k_guitarlstm_pytorch) + "/model_0/y_pred.wav",
                                   149,
                                   1e-4F,
                                   1e-5F),
                         build_test_name);
#endif

// Under the default candidate set the handler starts on plan 0, the first entry of the model
// file whose engine this build carries.
TEST(InferenceHandlerDefaultPlan, StartsOnTheFirstEntryOfTheBuild) {
    anira::Context context{anira::ContextConfig{}};
    anira::ModelConfig model = anira::ModelConfig::from_file(k_hybridnn_model_json);
    model.add_model(k_bypass_engine_id);
    anira::ContractHandle contract = anira::ContractHandle::from_file(k_hybridnn_contract_json);
    contract.hard_host_blocks(256, 256, 44100);
    anira::Pipeline pipeline;
    pipeline.add_engine(std::make_shared<BypassEngine>());
    pipeline.add_inference(model);
    pipeline.add_processor(std::make_shared<HybridNNProcessor>());
    anira::InferenceHandler handler(context, pipeline);
    const anira::PlanReport report = handler.prepare(contract);
    ASSERT_GE(report.num_plans(), 1U);
    EXPECT_EQ(handler.get_plan(), 0U);

    const std::vector<anira::BackendId> enabled = anira::enabled_engines();
    uint32_t first = ANIRA_ENGINE_CUSTOM;  // the bypass entry: the only one without an engine
    for (uint32_t i = 0; i < model.model_count(); ++i) {
        const auto kind = static_cast<uint32_t>(model.model_engine(i).kind);
        if (std::ranges::any_of(enabled, [kind](const anira::BackendId& row) {
                return row.engine == kind;
            })) {
            first = kind;
            break;
        }
    }
    EXPECT_EQ(report.plans()[handler.get_plan()].engine, first);
}
