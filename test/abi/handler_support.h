// The shared fixtures of the anira/abi/handler.h tests (test_Handler, test_HandlerWait,
// test_Prepare, test_RtError and the AbiCxx pipeline cases): a context and a handler with
// their lifetimes, the bundled gain model with the engine-free custom row, the hand-built
// generator and channel-mismatch models, the contracts, the block data, the waits, the test
// backends a session runs through its custom-processor pointer, and the guard that destroys
// the handler before such a backend dies.
#ifndef ANIRA_TEST_ABI_HANDLER_SUPPORT_H
#define ANIRA_TEST_ABI_HANDLER_SUPPORT_H

#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/backends/BackendBase.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/RtLatch.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "capi/handler.h"

namespace anira_test {

constexpr size_t k_block = 512;  // the gain model's hop (gain.model.json: window 512/512)
constexpr double k_rate = 48000.0;
constexpr int k_wait_s = 20;  // a passing test never waits this long; an ExecuTorch CNN block
                              // takes about 0.7 s here and several seconds under a loaded runner
constexpr const char* k_custom = "anira.v2.custom";

// ---- context and handler ---------------------------------------------------------------------

/// A context over its own config: MANUAL drain by default, so a test decides when
/// anira_drain_log() delivers the real-time records to its RecordCollector.
struct Context {
    explicit Context(uint32_t threads = 2,
                     anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF,
                     anira_log_level level = ANIRA_LOG_ERROR,
                     anira_log_drain drain = ANIRA_LOG_DRAIN_MANUAL,
                     uint32_t interval_ms = 10) {
        EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK) << m_err.message;
        EXPECT_EQ(anira_context_config_set_threads(m_config, threads, wait), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_level(m_config, level), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_drain(m_config, drain, interval_ms), ANIRA_OK);
        EXPECT_EQ(anira_context_create(m_config, &m_context, &m_err), ANIRA_OK) << m_err.message;
    }
    ~Context() {
        anira_context_destroy(m_context);
        anira_context_config_destroy(m_config);
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    anira_context_config* m_config = nullptr;
    anira_context* m_context = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// The bundled gain model plus the engine-free custom row (BackendBase::process: an exact
/// pass-through on this model, since both slots agree in channels and sample counts). With
/// default_custom the handler starts on the custom plan.
inline anira::ModelConfig gain_with_custom(bool default_custom = true) {
    anira::ModelConfig model = anira::ModelConfig::from_file(k_gain_model_json);
    model.add_model_path(k_custom, "custom-processor");
    if (default_custom) { model.default_engine(k_custom); }
    return model;
}

/// The engines of this build the bundled gain and CNN files run on. LiteRT is left out: its
/// runtime refuses simple_gain_network_mono.tflite at the warm-up inference ("Cannot
/// auto-resize tensor args_0_1: no dims_signature exists" -- the static gain scalar), a pair
/// no other test loads; the oracle compares the plans that load on both sides.
inline std::vector<anira_engine> oracle_engines() {
    std::vector<anira_engine> out;
    for (const anira::BackendId& id : anira::enabled_backends()) {
        if (id.engine == static_cast<uint32_t>(ANIRA_ENGINE_LITERT)) { continue; }
        out.push_back(static_cast<anira_engine>(id.engine));
    }
    return out;
}

/// One candidate per engine of oracle_engines() plus the NONE entry that keeps the custom
/// rows: the shape of the default set anira_pipeline_add_inference builds for a NULL list.
inline std::vector<anira_backend_id> custom_candidates() {
    std::vector<anira_backend_id> out;
    for (anira_engine engine : oracle_engines()) {
        out.push_back({.struct_size = sizeof(anira_backend_id),
                       .engine = engine,
                       .provider = ANIRA_PROVIDER_DEFAULT,
                       .engine_id = nullptr});
    }
    out.push_back({.struct_size = sizeof(anira_backend_id),
                   .engine = ANIRA_ENGINE_NONE,
                   .provider = ANIRA_PROVIDER_DEFAULT,
                   .engine_id = nullptr});
    return out;
}

/// The engines of custom_candidates() without the NONE entry.
inline std::vector<anira_backend_id> engine_candidates() {
    std::vector<anira_backend_id> out;
    for (anira_engine engine : oracle_engines()) {
        out.push_back({.struct_size = sizeof(anira_backend_id),
                       .engine = engine,
                       .provider = ANIRA_PROVIDER_DEFAULT,
                       .engine_id = nullptr});
    }
    return out;
}

/// The 2.x InferenceConfig of a bundled model over the same engines the C side runs
/// (oracle_engines(), plus the custom row when with_custom): what the oracle's 2.x handler
/// takes, so both sides' configs compare equal and the core pools one processor per engine.
inline anira::InferenceConfig bridged_2x(const char* model_json,
                                         const char* contract_json,
                                         bool with_custom) {
    anira::ModelConfig cfg = anira::ModelConfig::from_file(model_json);
    std::vector<anira_engine> candidates = oracle_engines();
    if (with_custom) {
        cfg.add_model_path(k_custom, "custom-processor");
        candidates.push_back(ANIRA_ENGINE_NONE);  // keeps the custom entry
    }
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    return anira::v3compat::to_inference_config(cfg, contract, candidates);
}

/// The bundled CNN without a custom row (BackendBase would zero its 15380 -> 2048 output).
inline anira::ModelConfig cnn_model() {
    return anira::ModelConfig::from_file(k_cnn_model_json);
}

/// The C twin of test_OneSidedStreaming's generator: four static parameters in, a
/// 2048-sample stream out, so the anchor resolves to the output.
inline anira::ModelConfig generator_model() {
    anira::ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    anira::TensorSpec param("param", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    param.axis(0, ANIRA_AXIS_ANY, 1).axis(1, ANIRA_AXIS_ANY, 4);
    model.input(param);
    anira::TensorSpec audio_out("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    audio_out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_TIME, 2048).window(2048, 2048, 0);
    model.output(audio_out);
    model.max_instances(2);
    return model;
}

/// A mono streamed input against a stereo streamed output, both 512 wide: BYPASS has no
/// anchored input with the output's channel count.
inline anira::ModelConfig mismatched_channels_model() {
    anira::ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, 512);
    in.window(512, 512, 0);
    model.input(in);
    anira::TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 2).axis(2, ANIRA_AXIS_TIME, 512);
    out.window(512, 512, 0);
    model.output(out);
    return model;
}

/// A Hard contract with an explicit budget and a fixed warm-up (5 ms is gain.contract.json's
/// figure; the generator tests pass 10 ms, test_OneSidedStreaming's max_inference_time).
inline anira::ContractHandle explicit_contract(uint32_t block = k_block,
                                               double rate = k_rate,
                                               anira_miss_policy on_miss = ANIRA_MISS_BYPASS,
                                               double wait_ratio = 0.0,
                                               double budget_ms = 5.0,
                                               uint32_t warmup_iterations = 0) {
    anira::Hard hard;
    hard.block_min = block;
    hard.block_max = block;
    hard.rate = rate;
    hard.budget = ANIRA_BUDGET_EXPLICIT;
    hard.budget_value = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double, std::milli>(budget_ms));
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = warmup_iterations;
    hard.on_miss = on_miss;
    hard.wait_ratio = wait_ratio;
    return anira::ContractHandle(hard);
}

/// A bundled contract file with the host geometry patched in: what the 2.x side bridges,
/// so both handlers' InferenceConfigs compare equal and the core pools one processor.
inline anira::ContractHandle file_contract(const char* contract_json,
                                           uint32_t block,
                                           double rate = k_rate) {
    anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    contract.hard_geometry(block, block, rate);
    return contract;
}

/// A pipeline with one inference stage and the handler over it. An empty span means NULL
/// candidates: the default set (every engine of this build plus the custom entries).
struct Handler {
    Handler(const Context& context,
            const anira::ModelConfig& model,
            std::span<const anira_backend_id> candidates = {}) {
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message;
        const anira_model_config* variants[] = {model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants,
                                               1,
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &m_err),
                  ANIRA_OK)
            << m_err.message;
        EXPECT_EQ(anira_handler_create(context.m_context, m_pipeline, &m_handler, &m_err), ANIRA_OK)
            << m_err.message;
    }
    ~Handler() { destroy(); }
    Handler(const Handler&) = delete;
    Handler& operator=(const Handler&) = delete;

    anira_status prepare(const anira::ContractHandle& contract, anira_error* err = nullptr) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_prepare(m_handler, contract.native(), err != nullptr ? err : &m_err);
    }

    /// Destroys the handler and the pipeline now and nulls the pointers (idempotent; the
    /// destructor calls it): the destroy drains the in-flight work and joins the pool with
    /// the last session.
    void destroy() {
        anira_handler_destroy(m_handler);
        m_handler = nullptr;
        anira_pipeline_destroy(m_pipeline);
        m_pipeline = nullptr;
    }

    anira_handler* m_handler = nullptr;
    anira_pipeline* m_pipeline = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

// ---- data, waiting, comparing ------------------------------------------------------------------

/// Block block_index of a deterministic ramp: distinct across blocks, exactly representable.
inline std::vector<float> ramp(size_t block_index, size_t n = k_block) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) { out[i] = static_cast<float>(block_index * n + i) / 65536.0F; }
    return out;
}

/// Waits until the output ring of tensor_index holds `expected` samples again (the loop of
/// test_InferenceHandler.cpp); fails after k_wait_s.
inline void wait_for_available(anira_handler* handler, size_t expected, uint32_t tensor_index = 0) {
    const auto start = std::chrono::steady_clock::now();
    while (anira_handler_get_available_samples(handler, tensor_index, 0) != expected) {
        if (std::chrono::steady_clock::now() > start + std::chrono::seconds(k_wait_s)) {
            FAIL() << "timeout while waiting for " << expected << " available samples (have "
                   << anira_handler_get_available_samples(handler, tensor_index, 0) << ")";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

/// After a process form: the call popped one block and its inference pushes one hop back,
/// so "available returns to prev" means the block's inference completed and was collected.
inline void wait_for_block(anira_handler* handler, size_t prev, uint32_t tensor_index = 0) {
    wait_for_available(handler, prev, tensor_index);
}

/// The 2.x twin of wait_for_block.
inline void wait_for_block(anira::InferenceHandler& handler, size_t prev) {
    const auto start = std::chrono::steady_clock::now();
    while (handler.get_available_samples(0) != prev) {
        if (std::chrono::steady_clock::now() > start + std::chrono::seconds(k_wait_s)) {
            FAIL() << "timeout while waiting for the 2.x block (" << prev << " expected, have "
                   << handler.get_available_samples(0) << ")";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

/// After a push: the inference of the pushed block adds one hop to the ring.
inline void wait_for_push(anira_handler* handler, size_t prev, size_t block) {
    wait_for_available(handler, prev + block);
}

inline void wait_for_push(anira::InferenceHandler& handler, size_t prev, size_t block) {
    wait_for_block(handler, prev + block);
}

/// Bit equality per sample.
inline void expect_same_block(std::span<const float> c,
                              std::span<const float> v2,
                              size_t block_index) {
    ASSERT_EQ(c.size(), v2.size()) << "block " << block_index;
    for (size_t i = 0; i < c.size(); ++i) {
        EXPECT_EQ(c[i], v2[i]) << "block " << block_index << ", sample " << i;
    }
}

inline void expect_all(std::span<const float> block, float value, const char* what) {
    for (size_t i = 0; i < block.size(); ++i) {
        ASSERT_EQ(block[i], value) << what << ", sample " << i;
    }
}

/// White-box: the session the handler's manager holds, found through the core's list.
inline std::shared_ptr<anira::SessionElement> session_of(const anira_handler* handler) {
    std::shared_ptr<anira::SessionElement> found;
    size_t matches = 0;
    if (handler != nullptr && handler->m_manager != nullptr) {
        const int session_id = handler->m_manager->get_session_id();
        for (const auto& session : anira::Core::get_sessions()) {
            if (session->m_session_id == session_id) {
                found = session;
                ++matches;
            }
        }
    }
    EXPECT_EQ(matches, 1U) << "the handler's session is not registered exactly once";
    return found;
}

/// Control thread, after prepare and before the first block, while the selected plan is the
/// custom row: the inference thread reads the pointer per inference. The backend must outlive
/// the handler (an inference thread may be inside its process() until the handler's destroy
/// has drained the in-flight work), and a stack backend built from h->m_inference_config is
/// declared after the handler: declare a DestroyFirst right after the attach.
inline void attach_processor(const anira_handler* handler, anira::BackendBase& backend) {
    const std::shared_ptr<anira::SessionElement> session = session_of(handler);
    ASSERT_NE(session, nullptr);
    session->m_custom_processor = &backend;
}

/// Lowers the drain summary's interval for the test's lifetime.
struct SummaryInterval {
    explicit SummaryInterval(uint32_t ms) { anira::detail::set_rt_summary_interval_ms(ms); }
    ~SummaryInterval() { anira::detail::set_rt_summary_interval_ms(10000); }
    SummaryInterval(const SummaryInterval&) = delete;
    SummaryInterval& operator=(const SummaryInterval&) = delete;
};

/// The records of the collector whose message contains the fragment, from that source.
inline size_t count_records(RecordCollector& collector, const char* fragment, const char* source) {
    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    size_t count = 0;
    for (const auto& record : collector.m_records) {
        if (record.m_message.find(fragment) != std::string::npos && record.m_source == source) {
            ++count;
        }
    }
    return count;
}

/// The first record whose message contains the fragment, from that source (an empty record
/// when none does).
inline RecordCollector::Record find_record(RecordCollector& collector,
                                           const char* fragment,
                                           const char* source) {
    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const auto& record : collector.m_records) {
        if (record.m_message.find(fragment) != std::string::npos && record.m_source == source) {
            return record;
        }
    }
    return {};
}

// ---- backends ----------------------------------------------------------------------------------

/// Holds every submitted inference on its inference thread while the gate is closed: the
/// driver's next block finds the output ring starved. Opens itself on destruction so a stuck
/// inference cannot hang the session release anira_handler_destroy runs.
class GateBackend : public anira::BackendBase {
public:
    explicit GateBackend(anira::InferenceConfig& config) : anira::BackendBase(config) {}
    ~GateBackend() override { m_open.store(true); }
    GateBackend(const GateBackend&) = delete;
    GateBackend& operator=(const GateBackend&) = delete;

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement> session) override {
        while (!m_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
        anira::BackendBase::process(input, output, std::move(session));
        ++m_calls;
    }

    std::atomic<bool> m_open{true};
    std::atomic<int> m_calls{0};
};

/// Throws from every inference while m_throw is set, else the pass-through.
class ThrowingBackend : public anira::BackendBase {
public:
    explicit ThrowingBackend(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 std::shared_ptr<anira::SessionElement> session) override {
        if (m_throw.load()) { throw std::runtime_error("test backend: inference failed"); }
        anira::BackendBase::process(input, output, std::move(session));
    }

    std::atomic<bool> m_throw{true};
};

/// The port of test_OneSidedStreaming's ParamFillGeneratorBackend: sleeps m_first_sleep_us on
/// its first call and m_sleep_us afterwards, then fills every output sample with parameter 0.
class SleepingParamFillBackend : public anira::BackendBase {
public:
    explicit SleepingParamFillBackend(anira::InferenceConfig& config)
        : anira::BackendBase(config) {}

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        const int sleep_us =
            m_started.fetch_add(1) == 0 && m_first_sleep_us > 0 ? m_first_sleep_us : m_sleep_us;
        if (sleep_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(sleep_us)); }
        const float value = input[0].get_sample(0, 0);
        for (size_t ch = 0; ch < output[0].get_num_channels(); ++ch) {
            float* write_ptr = output[0].get_write_pointer(ch);
            for (size_t s = 0; s < output[0].get_num_samples(); ++s) { write_ptr[s] = value; }
        }
        m_calls.fetch_add(1);
    }

    std::atomic<int> m_calls{0};
    std::atomic<int> m_started{0};
    int m_sleep_us = 0;
    int m_first_sleep_us = 0;
};

/// Declared right after a stack backend attach_processor() handed to the session, so at scope
/// exit it runs before the backend's destructor: it opens a held gate so the in-flight
/// inference can finish, then destroys the handler, which drains the in-flight work and joins
/// the pool. No inference thread is inside the backend's process() when the backend dies.
struct DestroyFirst {
    explicit DestroyFirst(Handler& handler, GateBackend* gate = nullptr)
        : m_handler(handler), m_gate(gate) {}
    ~DestroyFirst() {
        if (m_gate != nullptr) { m_gate->m_open.store(true); }
        m_handler.destroy();
    }
    DestroyFirst(const DestroyFirst&) = delete;
    DestroyFirst& operator=(const DestroyFirst&) = delete;

    Handler& m_handler;
    GateBackend* m_gate = nullptr;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_HANDLER_SUPPORT_H
