// The shared fixtures of the anira/abi/handler.h tests (test_Handler, test_HandlerWait,
// test_Prepare, test_RtError and the AbiCxx pipeline cases): a context and a handler with
// their lifetimes, the bundled gain model with the custom row, the hand-built generator and
// channel-mismatch models, the contracts, the block data, the waits, the engines a custom row
// runs on (the pass-through every Handler adds under anira.v2.custom, the gate engines of the
// C rigs), and the guard that destroys the handler before a gate engine's rig dies.
#ifndef ANIRA_TEST_ABI_HANDLER_SUPPORT_H
#define ANIRA_TEST_ABI_HANDLER_SUPPORT_H

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/RtLatch.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
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
    /// `queue_capacity` 0 keeps the config's default capacity of the real-time log queue.
    explicit Context(uint32_t threads = 2,
                     anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF,
                     anira_log_level level = ANIRA_LOG_ERROR,
                     anira_log_drain drain = ANIRA_LOG_DRAIN_MANUAL,
                     uint32_t interval_ms = 10,
                     uint32_t queue_capacity = 0) {
        EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK) << m_err.message;
        EXPECT_EQ(anira_context_config_set_threads(m_config, threads, wait), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_level(m_config, level), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_drain(m_config, drain, interval_ms), ANIRA_OK);
        if (queue_capacity > 0) {
            EXPECT_EQ(anira_context_config_set_log_queue_capacity(m_config, queue_capacity),
                      ANIRA_OK);
        }
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

/// The bundled gain model plus the custom row, which a Handler runs on its pass-through (an
/// exact copy on this model, since both slots agree in element count). With default_custom the
/// handler starts on the custom plan.
inline anira::ModelConfig gain_with_custom(bool default_custom = true) {
    anira::ModelConfig model = anira::ModelConfig::from_file(k_gain_model_json);
    model.add_model_path(k_custom, "custom-processor");
    if (default_custom) { model.default_engine(k_custom); }
    return model;
}

/// The stereo twin of gain_with_custom(): the bundled stereo gain model ([batch 1, channel 2,
/// time 512] and the static gain) plus the custom row. Two channels are what tells
/// an interleaved block from a planar one; the mono model cannot.
inline anira::ModelConfig stereo_gain_with_custom(bool default_custom = true) {
    anira::ModelConfig model = anira::ModelConfig::from_file(k_stereo_gain_model_json);
    model.add_model_path(k_custom, "custom-processor");
    if (default_custom) { model.default_engine(k_custom); }
    return model;
}

/// The engines of this build the bundled gain and CNN files run on in the suites that also
/// drive the shim's 2.x configurations. LiteRT is left out: its signature lists the gain
/// export's outputs as [peak, processed], which the file's litert row names (a tensors record),
/// and a 2.x configuration, which has no tensor names, binds them by position and fails the
/// shape check at prepare.
inline std::vector<anira_engine> oracle_engines() {
    std::vector<anira_engine> out;
    for (const anira::BackendId& id : anira::enabled_engines()) {
        if (id.engine == static_cast<uint32_t>(ANIRA_ENGINE_LITERT)) { continue; }
        out.push_back(static_cast<anira_engine>(id.engine));
    }
    return out;
}

/// One candidate per engine of oracle_engines() plus the custom engine anira.v2.custom (k_custom,
/// ANIRA_ENGINE_CUSTOM with its id): the shape of the default set anira_pipeline_add_inference
/// builds for a NULL list.
inline std::vector<anira_backend_id> custom_candidates() {
    std::vector<anira_backend_id> out;
    for (anira_engine engine : oracle_engines()) {
        out.push_back({.struct_size = sizeof(anira_backend_id),
                       .engine = static_cast<uint32_t>(engine),
                       .provider = ANIRA_PROVIDER_CPU,
                       .engine_id = nullptr});
    }
    out.push_back({.struct_size = sizeof(anira_backend_id),
                   .engine = ANIRA_ENGINE_CUSTOM,
                   .provider = ANIRA_PROVIDER_CPU,
                   .engine_id = k_custom});
    return out;
}

/// The engines of custom_candidates() without the custom one.
inline std::vector<anira_backend_id> engine_candidates() {
    std::vector<anira_backend_id> out;
    for (anira_engine engine : oracle_engines()) {
        out.push_back({.struct_size = sizeof(anira_backend_id),
                       .engine = static_cast<uint32_t>(engine),
                       .provider = ANIRA_PROVIDER_CPU,
                       .engine_id = nullptr});
    }
    return out;
}

/// The bundled CNN without a custom row (the pass-through would zero its 15380 -> 2048 output).
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
                                               double host_rate = k_rate,
                                               anira_miss_policy on_miss = ANIRA_MISS_BYPASS,
                                               double wait_ratio = 0.0,
                                               double budget_ms = 5.0,
                                               uint32_t warmup_iterations = 0) {
    anira::Hard hard{.host_block_min = static_cast<double>(block),
                     .host_block_max = static_cast<double>(block),
                     .host_rate = host_rate,
                     .budget = std::chrono::duration<double, std::milli>(budget_ms)};
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = warmup_iterations;
    hard.on_miss = on_miss;
    hard.wait_ratio = wait_ratio;
    return anira::ContractHandle(hard);
}

/// A Streamed tensor of `channels` channels of `time` samples ([Channel, Time]) that moves `hop`
/// samples per inference (a window of `hop`, no overlap): the 2.x tensor of channels * time
/// elements with that preprocess or postprocess size.
inline anira::TensorSpec stream_tensor(std::string_view name,
                                       int64_t channels,
                                       int64_t time,
                                       int64_t hop) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_CHANNEL, channels).axis(1, ANIRA_AXIS_TIME, time).window(hop, hop, 0);
    return spec;
}

/// A Static tensor of the given extents, every axis ANIRA_AXIS_ANY.
inline anira::TensorSpec static_tensor(std::string_view name,
                                       std::initializer_list<int64_t> extents) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    uint32_t axis = 0;
    for (const int64_t extent : extents) { spec.axis(axis++, ANIRA_AXIS_ANY, extent); }
    return spec;
}

/// The custom row alone (a Handler runs it on the pass-through, or on the engine it is handed)
/// with `max_instances` instances; the caller adds the tensors.
inline anira::ModelConfig custom_model(uint32_t max_instances = 2) {
    anira::ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    model.max_instances(max_instances);
    return model;
}

/// custom_model() with one mono Streamed tensor of `window` samples on either side ("in",
/// "out"): the 2.x [1, 1, window] in and out of the engine-free scheduler suites.
inline anira::ModelConfig mono_stream_model(int64_t window = k_block, uint32_t max_instances = 2) {
    anira::ModelConfig model = custom_model(max_instances);
    model.input(stream_tensor("in", 1, window, window));
    model.output(stream_tensor("out", 1, window, window));
    return model;
}

/// A bundled contract file with the host blocks patched in.
inline anira::ContractHandle file_contract(const char* contract_json,
                                           uint32_t block,
                                           double host_rate = k_rate) {
    anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    contract.hard_host_blocks(block, block, host_rate);
    return contract;
}

// ---- engines -----------------------------------------------------------------------------------

/// The elements of a tensor: the product of its extents (rank 0 is one element).
inline size_t elements_of(const anira_tensor& tensor) noexcept {
    size_t count = 1;
    for (uint32_t axis = 0; axis < tensor.ndim; ++axis) {
        count *= static_cast<size_t>(tensor.shape[axis]);
    }
    return count;
}

/// The C twin of the 2.x pass-through (2.x's BackendBase::process):
/// output i is a copy of input i when both carry the same element count and dtype, zeros
/// otherwise; every output beyond the input count is zeros; State and Static slots pair by
/// index like every other slot. ANIRA_ERROR_INVALID_ARGUMENT for a tensor that is no packed
/// host tensor (anira_tensor_data answers NULL), which no C handler hands an engine.
inline anira_status passthrough(const anira_engine_ctx& ctx) noexcept {
    for (uint32_t slot = 0; slot < ctx.num_outputs; ++slot) {
        const anira_tensor& out = ctx.outputs[slot];
        void* target = anira_tensor_data(&out, out.dtype);
        if (target == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const size_t bytes =
            elements_of(out) * ANIRA_DTYPE_BITS(out.dtype) * ANIRA_DTYPE_LANES(out.dtype) / 8;
        const anira_tensor* in = slot < ctx.num_inputs ? &ctx.inputs[slot] : nullptr;
        if (in != nullptr && in->dtype == out.dtype && elements_of(*in) == elements_of(out)) {
            const void* source = anira_tensor_data(in, in->dtype);
            if (source == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
            std::memmove(target, source, bytes);
        } else {
            std::memset(target, 0, bytes);
        }
    }
    return ANIRA_OK;
}

/// The process slot of the pass-through engine: passthrough() over the call's tensors.
inline anira_status ANIRA_CALL passthrough_process(const anira_engine_ctx* ctx,
                                                   void* /*prepared*/,
                                                   void* /*user_data*/) noexcept {
    return passthrough(*ctx);
}

/// The pass-through engine: of the nine slots it fills process alone. load and prepare are
/// NULL, so loaded and prepared are NULL in every call and a shared and an exclusive call run
/// the same body; no init, reset, unprepare, unload or release; no providers list
/// (ANIRA_PROVIDER_CPU alone); flags 0.
inline anira_engine_desc passthrough_desc() noexcept {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.process = passthrough_process;
    return desc;
}

/// A new pass-through engine object under `id` (anira.v2.custom by default); the caller owns
/// the handle (anira_custom_engine_destroy).
inline anira_custom_engine* make_passthrough(const char* id = k_custom) {
    const anira_engine_desc desc = passthrough_desc();
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create(id, &desc, &engine, &err), ANIRA_OK) << err.message;
    return engine;
}

/// Adds an engine to a pipeline under the id it was created with, before the handler exists
/// (a handler copies the pipeline at create).
inline void add_engine(anira_pipeline* pipeline, const anira_custom_engine* engine) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_pipeline_add_engine(pipeline, engine, &err), ANIRA_OK) << err.message;
}

/// Adds a pass-through engine object of its own to a pipeline under anira.v2.custom and drops
/// the handle: the pipeline and the handlers created from it hold the object. One object per
/// pipeline, as the engine-free roundtrip it replaces was one per session and never pooled: two
/// pipelines that shared one object would share one loaded model per equal variant (the object
/// is the pool's identity). Its loaded model is NULL (no load), and each handler gets its own
/// prepared handle (NULL).
inline void add_passthrough(anira_pipeline* pipeline) {
    anira_custom_engine* engine = make_passthrough();
    add_engine(pipeline, engine);
    anira_custom_engine_destroy(engine);
}

/// A C engine object over a virtual body, created under anira.v2.custom on the first engine()
/// (user_data = this; process alone of the nine slots) and owned by the rig: its own pooling
/// identity. process waits while the gate is closed, then fails (m_fail_next once, m_fail while
/// set: the status returned, ANIRA_ERROR_ENGINE for m_fail, plus one "test backend: inference
/// failed" record through anira_log_rt, what the legacy room did for a 2.x throw) or runs
/// run(), then counts. Declared BEFORE the Handler that adds it, so the handler, and every
/// inference thread inside process, is gone before the object dies; a DestroyFirst declared
/// after the Handler opens the gate before the handler's destroy drains the in-flight work.
/// A closed gate holds one inference thread per shared slot of the loaded model inside process;
/// the jobs beyond them wait on the model's wait list (their threads go back to the queue); a
/// test that needs N calls inside process at once sets max_instances(N).
class GateEngine {
public:
    GateEngine() = default;
    virtual ~GateEngine() {
        m_open.store(true);
        anira_custom_engine_destroy(m_engine);
    }
    GateEngine(const GateEngine&) = delete;
    GateEngine& operator=(const GateEngine&) = delete;
    GateEngine(GateEngine&&) = delete;
    GateEngine& operator=(GateEngine&&) = delete;

    anira_engine_desc desc() noexcept {
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.user_data = this;
        desc.flags = m_flags;
        desc.process = process_slot;
        return desc;
    }

    /// The engine object, created from desc() under anira.v2.custom on the first call.
    anira_custom_engine* engine() {
        if (m_engine == nullptr) {
            const anira_engine_desc engine_desc = desc();
            anira_error err = ANIRA_ERROR_INIT;
            EXPECT_EQ(anira_custom_engine_create(k_custom, &engine_desc, &m_engine, &err), ANIRA_OK)
                << err.message;
        }
        return m_engine;
    }

    /// One inference: the gate, the failure or run(), the count.
    virtual anira_status process(const anira_engine_ctx& ctx) noexcept {
        wait_open();
        anira_status status = failure();
        if (status == ANIRA_OK) { status = run(ctx); }
        m_calls.fetch_add(1);
        return status;
    }

    /// The body of an inference that does not fail: the pass-through.
    virtual anira_status run(const anira_engine_ctx& ctx) noexcept { return passthrough(ctx); }

    std::atomic<bool> m_open{true};
    std::atomic<int> m_calls{0};
    std::atomic<bool> m_fail{false};                  ///< every inference fails while set
    std::atomic<anira_status> m_fail_next{ANIRA_OK};  ///< the next inference fails with it, once
    uint32_t m_flags = 0;                             ///< the descriptor's, set before engine()

protected:
    void wait_open() const noexcept {
        while (!m_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    }

    /// The status this inference fails with (ANIRA_OK: it does not), logged once per failure.
    anira_status failure() noexcept {
        anira_status status = m_fail_next.exchange(ANIRA_OK);
        if (status == ANIRA_OK && m_fail.load()) { status = ANIRA_ERROR_ENGINE; }
        if (status != ANIRA_OK) {
            anira_log_rt(ANIRA_LOG_ERROR,
                         "anira.test",
                         "test backend: inference failed",
                         status,
                         0);
        }
        return status;
    }

private:
    static anira_status ANIRA_CALL process_slot(const anira_engine_ctx* ctx,
                                                void* /*prepared*/,
                                                void* user_data) noexcept {
        return static_cast<GateEngine*>(user_data)->process(*ctx);
    }

    anira_custom_engine* m_engine = nullptr;
};

/// The port of test_OneSidedStreaming's ParamFillGeneratorBackend: sleeps m_first_sleep_us on
/// its first call and m_sleep_us afterwards, then fills every element of output 0 with element
/// 0 of input 0.
class SleepingParamFillEngine : public GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const int sleep_us =
            m_started.fetch_add(1) == 0 && m_first_sleep_us > 0 ? m_first_sleep_us : m_sleep_us;
        if (sleep_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(sleep_us)); }
        const float* param = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (param == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const float value = param[0];
        for (size_t i = 0; i < elements_of(ctx.outputs[0]); ++i) { out[i] = value; }
        return ANIRA_OK;
    }

    std::atomic<int> m_started{0};
    int m_sleep_us = 0;
    int m_first_sleep_us = 0;
};

/// A pipeline with one inference stage, the processors of `processors` (at most one in this
/// pre-release) and the handler over it. An empty span means NULL candidates: the default set
/// (every engine of this build plus the custom entries). Before the inference stage the
/// pipeline gets the engine a custom row runs on: `custom` (a GateEngine's engine(), declared
/// before this Handler), else a pass-through object of its own (add_passthrough).
struct Handler {
    Handler(const Context& context,
            const anira::ModelConfig& model,
            std::span<const anira_backend_id> candidates = {},
            std::span<const anira_processor_desc> processors = {},
            const anira_custom_engine* custom = nullptr) {
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message;
        if (custom != nullptr) {
            add_engine(m_pipeline, custom);
        } else {
            add_passthrough(m_pipeline);
        }
        const anira_model_config* variants[] = {model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants,
                                               1,
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &m_err),
                  ANIRA_OK)
            << m_err.message;
        for (const anira_processor_desc& processor : processors) {
            EXPECT_EQ(anira_pipeline_add_processor(m_pipeline, &processor, &m_err), ANIRA_OK)
                << m_err.message;
        }
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

/// One "plugin instance": a context over its own config (Context's arguments) and a handler
/// over the mono pass-through (mono_stream_model), prepared on construction under the explicit
/// contract, so that a session (and with it the pool) exists.
struct Instance {
    explicit Instance(uint32_t threads = 2,
                      anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF,
                      anira_log_level level = ANIRA_LOG_ERROR,
                      anira_log_drain drain = ANIRA_LOG_DRAIN_MANUAL,
                      uint32_t interval_ms = 10,
                      uint32_t queue_capacity = 0)
        : m_context(threads, wait, level, drain, interval_ms, queue_capacity) {
        EXPECT_EQ(m_handler.prepare(explicit_contract()), ANIRA_OK) << m_handler.m_err.message;
    }
    ~Instance() = default;
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    Context m_context;
    Handler m_handler{m_context, mono_stream_model()};
};

// ---- data, waiting, comparing ------------------------------------------------------------------

/// Block block_index of a deterministic ramp: distinct across blocks, exactly representable.
inline std::vector<float> ramp(size_t block_index, size_t n = k_block) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) { out[i] = static_cast<float>(block_index * n + i) / 65536.0F; }
    return out;
}

/// anira_handler_get_available_samples as a value: the samples waiting in channel 0 of the
/// output ring, 0 on a failure (a test that cares about the status calls the entry itself).
inline size_t available(anira_handler* handler, uint32_t slot = 0) {
    size_t count = 0;
    anira_handler_get_available_samples(handler, slot, 0, &count);
    return count;
}

/// Waits until the output ring of slot holds `expected` samples again (the loop of
/// test_InferenceHandler.cpp); fails after k_wait_s.
inline void wait_for_available(anira_handler* handler, size_t expected, uint32_t slot = 0) {
    const auto start = std::chrono::steady_clock::now();
    while (available(handler, slot) != expected) {
        if (std::chrono::steady_clock::now() > start + std::chrono::seconds(k_wait_s)) {
            FAIL() << "timeout while waiting for " << expected << " available samples (have "
                   << available(handler, slot) << ")";
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

/// After a process form: the call popped one block and its inference pushes one hop back,
/// so "available returns to prev" means the block's inference completed and was collected.
inline void wait_for_block(anira_handler* handler, size_t prev, uint32_t slot = 0) {
    wait_for_available(handler, prev, slot);
}

/// After a push: the inference of the pushed block adds one hop to the ring.
inline void wait_for_push(anira_handler* handler, size_t prev, size_t block) {
    wait_for_available(handler, prev + block);
}

/// The closed form of a stream through a model that maps every sample on its own (the gain
/// model, a pass-through) when every block's inference is waited for: the processed input
/// delayed by the reported latency, silence first. Fed the input blocks in order since the
/// prepare (or the last reset), each with the factor the plan that ran it applies (the gain an
/// engine of the gain model multiplies by; 1 for a pass-through, which copies), it answers each
/// output block. A plan switch between blocks keeps the line: what is in flight comes out as
/// it was processed.
class DelayLine {
public:
    explicit DelayLine(size_t latency = 0) : m_latency(latency) { reset(); }

    /// The stream restarts: the latency's silence comes first again.
    void reset() { m_pending.assign(m_latency, 0.0F); }

    std::vector<float> next(std::span<const float> in, float factor = 1.0F) {
        for (const float sample : in) { m_pending.push_back(sample * factor); }
        std::vector<float> out(m_pending.begin(),
                               m_pending.begin() + static_cast<std::ptrdiff_t>(in.size()));
        m_pending.erase(m_pending.begin(),
                        m_pending.begin() + static_cast<std::ptrdiff_t>(in.size()));
        return out;
    }

    size_t latency() const noexcept { return m_latency; }

private:
    size_t m_latency;
    std::deque<float> m_pending;
};

/// The Static output of the gain model's engines for one window: the peak, max |x * gain|.
inline float gain_peak(std::span<const float> in, float gain) {
    float peak = 0.0F;
    for (const float sample : in) { peak = std::max(peak, std::abs(sample * gain)); }
    return peak;
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

/// Declared right after the Handler a gate engine runs in, so at scope exit it runs first: it
/// opens the held gate so the in-flight inference can finish, then destroys the handler, which
/// drains the in-flight work and joins the pool.
struct DestroyFirst {
    explicit DestroyFirst(Handler& handler, GateEngine* gate = nullptr)
        : m_handler(handler), m_gate(gate) {}
    ~DestroyFirst() {
        if (m_gate != nullptr) { m_gate->m_open.store(true); }
        m_handler.destroy();
    }
    DestroyFirst(const DestroyFirst&) = delete;
    DestroyFirst& operator=(const DestroyFirst&) = delete;
    DestroyFirst(DestroyFirst&&) = delete;
    DestroyFirst& operator=(DestroyFirst&&) = delete;

    Handler& m_handler;
    GateEngine* m_gate = nullptr;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_HANDLER_SUPPORT_H
