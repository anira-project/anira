// Several variants of one inference stage (anira_pipeline_add_inference with more than one model
// config). A variant is a model of the stage's one form: the variants differ in what they compute
// and in nothing the stream can see, so one handler loads and measures them together, runs them
// in one stream and switches between them while audio runs. What the suite pins:
//
//   the plan table      one plan per variant, entry and candidate, in variant order; a row carries
//                       its variant, anira_plan_report_find resolves it, a processor context
//                       reports it, and an engine's load record names the plan's own variant;
//   the form rule       every variant derives the form of the first (anira::capi::
//                       check_same_form): one refusal per property, each naming the variant, the
//                       slot and the property, at create and again under the contract at prepare;
//                       what is free per variant may differ;
//   the warnings        differing internal latencies warn once and the largest is reported; the
//                       stream is sized for the fewest instances and each row reports its own;
//   the switch          a stream that switches variants equals, block by block, the model that
//                       was selected when the block was formed; nothing fills, the latency stays,
//                       and a chunk in flight finishes on its variant;
//   the state           what the model's state is at a variant switch
//                       (anira_contract_hard_set_variant_state): fresh by default (the declared
//                       State pairs at zeros and the plan's engine reset ahead of the first
//                       inference formed on the other variant), kept when the contract says so; a
//                       plan switch inside one variant keeps it in both modes, and a chunk in
//                       flight across a fresh switch keeps the state it ran with;
//   the selection       anira_handler_set_model moves the variant and keeps the backend,
//                       anira_handler_set_backend the reverse; a pair without a plan changes
//                       nothing (ANIRA_ERROR_NOT_FOUND, latched); the two on two threads both
//                       land;
//   the measurement     every plan of every variant is warmed up, and the session runs the
//                       slowest figure;
//   real models         a bundled model file twice, as two variants, on every engine of the build.
//
// The engine-free cases run on a registered engine whose loaded model is a gain: the factor is
// the number its entry's path spells, read through the load record's variant, so every plan of a
// table answers a stream of its own and a block says which plan ran it. The state cases run on a
// second one, which adds its entry's number to a declared State pair per inference and counts its
// own inferences since its last reset.

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../support/engine_oracle.h"
#include "../support/log_record_collector.h"
#include "float_face.h"
#include "gtest/gtest.h"
#include "handler_support.h"
#include "scheduler/SessionElement.h"

namespace {

using anira_test::k_block;
using anira_test::k_rate;
using anira_test::RecordCollector;

constexpr auto k_wait = std::chrono::seconds(anira_test::k_wait_s);
/// The engine of most rows: it serves the CPU path and a custom provider, two plans of one row.
constexpr const char* k_engine = "org.example.variants";
/// A second engine, on the CPU path alone: an entry only one variant has.
constexpr const char* k_other_engine = "org.example.variants_b";
constexpr const char* k_npu = "com.example.npu";
constexpr std::array<const char*, 2> k_served{"cpu", k_npu};
/// What a loaded model on k_npu multiplies its path's factor by.
constexpr float k_npu_factor = 16.0F;

// ---- the engine ------------------------------------------------------------------------------

/// A registered C engine whose loaded model is a gain: output 0 is input 0 times a factor. The
/// factor is the number the entry's path spells ("2" is 2), read at load through the load
/// record's variant (info->model, info->row), times k_npu_factor on the provider k_npu, so the
/// plans of a table answer different streams. It keeps nothing between two calls. A closed gate
/// holds a call inside process (m_held counts the calls waiting there).
struct FactorEngine {
    struct Loaded {
        float m_factor = 1.0F;
    };
    static constexpr size_t k_capacity = 512;

    std::atomic<bool> m_open{true};
    std::atomic<int> m_held{0};
    std::atomic<int> m_calls{0};
    std::atomic<int> m_loads{0};
    std::atomic<int> m_unloads{0};
    /// Per call, in call order: the factor of the loaded model that ran.
    std::array<std::atomic<float>, k_capacity> m_factors{};

    /// How many of the calls so far ran a loaded model of `factor`.
    int calls_of(float factor) const {
        int count = 0;
        const int calls = std::min<int>(m_calls.load(), static_cast<int>(k_capacity));
        for (int call = 0; call < calls; ++call) {
            if (m_factors.at(static_cast<size_t>(call)).load() == factor) { ++count; }
        }
        return count;
    }
};

anira_status ANIRA_CALL factor_load(const anira_engine_load_info* info,
                                    void* user_data,
                                    void** out_loaded) {
    auto* engine = static_cast<FactorEngine*>(user_data);
    const char* path = anira_model_config_model_path(info->model, info->row);
    if (path == nullptr) { return ANIRA_ERROR_CONFIG; }
    auto loaded = std::make_unique<FactorEngine::Loaded>();
    loaded->m_factor = static_cast<float>(std::atof(path));
    if (info->provider_id != nullptr && std::string_view(info->provider_id) == k_npu) {
        loaded->m_factor *= k_npu_factor;
    }
    engine->m_loads.fetch_add(1);
    *out_loaded = loaded.release();
    return ANIRA_OK;
}

void ANIRA_CALL factor_unload(void* loaded, void* user_data) {
    static_cast<FactorEngine*>(user_data)->m_unloads.fetch_add(1);
    delete static_cast<FactorEngine::Loaded*>(loaded);
}

anira_status ANIRA_CALL factor_process(const anira_engine_ctx* ctx,
                                       void* /*prepared*/,
                                       void* user_data) {
    auto* engine = static_cast<FactorEngine*>(user_data);
    const auto* loaded = static_cast<const FactorEngine::Loaded*>(ctx->loaded);
    if (loaded == nullptr) { return ANIRA_ERROR_INTERNAL; }
    engine->m_held.fetch_add(1);
    while (!engine->m_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    engine->m_held.fetch_sub(1);
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const size_t count =
        std::min(anira_test::elements_of(ctx->inputs[0]), anira_test::elements_of(ctx->outputs[0]));
    for (size_t i = 0; i < count; ++i) { out[i] = in[i] * loaded->m_factor; }
    const auto call = static_cast<size_t>(engine->m_calls.fetch_add(1));
    if (call < FactorEngine::k_capacity) { engine->m_factors.at(call).store(loaded->m_factor); }
    return ANIRA_OK;
}

/// The engine object under `id` over a FactorEngine, serving `providers` (none: the CPU path
/// alone); the caller adds it to a pipeline and destroys the handle.
anira_custom_engine* make_factor_engine(FactorEngine& engine,
                                        const char* id,
                                        std::span<const char* const> providers = {}) {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.load = factor_load;
    desc.unload = factor_unload;
    desc.process = factor_process;
    desc.providers = providers.empty() ? nullptr : providers.data();
    desc.num_providers = static_cast<uint32_t>(providers.size());
    anira_custom_engine* object = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create(id, &desc, &object, &err), ANIRA_OK) << err.message;
    return object;
}

// ---- the engine of the state cases ------------------------------------------------------------

/// The engine with state: a mono stream and one declared State pair.
constexpr const char* k_step_engine = "org.example.variant_steps";

/// A registered C engine with both kinds of state. The declared state is the model's State pair
/// (input slot 1 fed from output slot 1): every inference adds the loaded model's step (the
/// number its entry's path spells, times k_npu_factor on the provider k_npu) to the state it
/// reads, writes the sum as the new state and fills its output hop with it, so a block of the
/// stream says what the state was. The state the engine keeps inside itself is per prepared
/// handle, one per plan of an exclusive handler: the count of its inferences since its last
/// reset. Every call and every reset is recorded in order. A closed gate holds a call inside
/// process.
struct StepEngine {
    struct Loaded {
        float m_step = 0.0F;
    };
    struct Prepared {
        float m_step = 0.0F;
        int m_since_reset = 0;
    };
    static constexpr size_t k_capacity = 64;

    std::atomic<bool> m_open{true};
    std::atomic<int> m_held{0};
    std::atomic<int> m_calls{0};
    std::atomic<int> m_resets{0};
    /// Per call: the step of the loaded model that ran, the declared state as it found it and
    /// its own count before the call.
    std::array<std::atomic<float>, k_capacity> m_step{};
    std::array<std::atomic<float>, k_capacity> m_state_read{};
    std::array<std::atomic<int>, k_capacity> m_since_reset{};
    /// Per reset: the step of the plan whose engine was reset.
    std::array<std::atomic<float>, k_capacity> m_reset_step{};

    template <class Value>
    static std::vector<Value> first_of(const std::array<std::atomic<Value>, k_capacity>& values,
                                       int count) {
        std::vector<Value> out;
        for (int i = 0; i < count && static_cast<size_t>(i) < k_capacity; ++i) {
            out.push_back(values.at(static_cast<size_t>(i)).load());
        }
        return out;
    }
    std::vector<float> steps() const { return first_of(m_step, m_calls.load()); }
    std::vector<float> states_read() const { return first_of(m_state_read, m_calls.load()); }
    std::vector<int> counts() const { return first_of(m_since_reset, m_calls.load()); }
    std::vector<float> resets() const { return first_of(m_reset_step, m_resets.load()); }
};

anira_status ANIRA_CALL step_load(const anira_engine_load_info* info,
                                  void* /*user_data*/,
                                  void** out_loaded) {
    const char* path = anira_model_config_model_path(info->model, info->row);
    if (path == nullptr) { return ANIRA_ERROR_CONFIG; }
    auto loaded = std::make_unique<StepEngine::Loaded>();
    loaded->m_step = static_cast<float>(std::atof(path));
    if (info->provider_id != nullptr && std::string_view(info->provider_id) == k_npu) {
        loaded->m_step *= k_npu_factor;
    }
    *out_loaded = loaded.release();
    return ANIRA_OK;
}

void ANIRA_CALL step_unload(void* loaded, void* /*user_data*/) {
    delete static_cast<StepEngine::Loaded*>(loaded);
}

anira_status ANIRA_CALL step_prepare(const anira_prepare_info* /*info*/,
                                     void* loaded,
                                     void* /*user_data*/,
                                     void** out_prepared) {
    if (loaded == nullptr) { return ANIRA_ERROR_INTERNAL; }
    auto prepared = std::make_unique<StepEngine::Prepared>();
    prepared->m_step = static_cast<const StepEngine::Loaded*>(loaded)->m_step;
    *out_prepared = prepared.release();
    return ANIRA_OK;
}

void ANIRA_CALL step_unprepare(void* prepared, void* /*user_data*/) {
    delete static_cast<StepEngine::Prepared*>(prepared);
}

void ANIRA_CALL step_reset(const anira_engine_ctx* /*ctx*/, void* prepared, void* user_data) {
    auto* engine = static_cast<StepEngine*>(user_data);
    auto* own = static_cast<StepEngine::Prepared*>(prepared);
    if (own == nullptr) { return; }
    own->m_since_reset = 0;
    const auto reset = static_cast<size_t>(engine->m_resets.fetch_add(1));
    if (reset < StepEngine::k_capacity) { engine->m_reset_step.at(reset).store(own->m_step); }
}

anira_status ANIRA_CALL step_process(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    auto* engine = static_cast<StepEngine*>(user_data);
    auto* own = static_cast<StepEngine::Prepared*>(prepared);
    if (own == nullptr || ctx->num_inputs < 2 || ctx->num_outputs < 2) {
        return ANIRA_ERROR_INTERNAL;
    }
    engine->m_held.fetch_add(1);
    while (!engine->m_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
    engine->m_held.fetch_sub(1);
    const float* state_in = anira_tensor_data_f32(&ctx->inputs[1]);
    float* state_out = anira_tensor_data_f32(&ctx->outputs[1]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (state_in == nullptr || state_out == nullptr || out == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const auto call = static_cast<size_t>(engine->m_calls.load());
    if (call < StepEngine::k_capacity) {
        engine->m_step.at(call).store(own->m_step);
        engine->m_state_read.at(call).store(state_in[0]);
        engine->m_since_reset.at(call).store(own->m_since_reset);
    }
    const float next = state_in[0] + own->m_step;
    state_out[0] = next;
    for (size_t i = 0; i < anira_test::elements_of(ctx->outputs[0]); ++i) { out[i] = next; }
    ++own->m_since_reset;
    engine->m_calls.fetch_add(1);
    return ANIRA_OK;
}

anira_custom_engine* make_step_engine(StepEngine& engine) {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.load = step_load;
    desc.unload = step_unload;
    desc.prepare = step_prepare;
    desc.unprepare = step_unprepare;
    desc.reset = step_reset;
    desc.process = step_process;
    desc.providers = k_served.data();
    desc.num_providers = static_cast<uint32_t>(k_served.size());
    anira_custom_engine* object = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create(k_step_engine, &desc, &object, &err), ANIRA_OK)
        << err.message;
    return object;
}

// ---- the models ------------------------------------------------------------------------------

/// A State tensor of four elements.
anira::TensorSpec state_tensor(std::string_view name) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    spec.axis(0, ANIRA_AXIS_ANY, 4);
    return spec;
}

/// The model of the state cases: a mono stream of k_block samples per inference on either side
/// and one declared State pair ("state_in" fed from "state_out"), over one entry of the step
/// engine whose path spells the model's step. A declared pair makes the handler exclusive.
anira::ModelConfig step_model(const char* step) {
    anira::ModelConfig model;
    model.add_model_path(k_step_engine, step);
    model.input(anira_test::stream_tensor("in", 1, k_block, k_block));
    model.input(state_tensor("state_in").state_source("state_out"));
    model.output(anira_test::stream_tensor("out", 1, k_block, k_block));
    model.output(state_tensor("state_out"));
    return model;
}

/// A mono stream of k_block samples per inference on either side ("in", "out") over one entry
/// of `engine` whose path spells the model's factor.
anira::ModelConfig factor_model(const char* factor,
                                const char* engine = k_engine,
                                uint32_t max_instances = 2) {
    anira::ModelConfig model;
    model.add_model_path(engine, factor);
    model.input(anira_test::stream_tensor("in", 1, k_block, k_block));
    model.output(anira_test::stream_tensor("out", 1, k_block, k_block));
    model.max_instances(max_instances);
    return model;
}

anira_backend_id candidate(const char* engine_id, const char* provider_id = nullptr) {
    return {.struct_size = sizeof(anira_backend_id),
            .engine = ANIRA_ENGINE_CUSTOM,
            .provider = static_cast<uint32_t>(provider_id != nullptr ? ANIRA_PROVIDER_CUSTOM
                                                                     : ANIRA_PROVIDER_CPU),
            .engine_id = engine_id,
            .provider_id = provider_id};
}

/// k_engine on the CPU path and on k_npu: two plans per entry of k_engine.
std::vector<anira_backend_id> cpu_and_npu() {
    return {candidate(k_engine), candidate(k_engine, k_npu)};
}

/// An explicit budget and no warm-up: a prepare runs no inference, so an engine's call count is
/// the stream's alone, whatever figure an earlier test of the process left in the core.
anira::ContractHandle plain_contract(anira_miss_policy on_miss = ANIRA_MISS_ZEROS) {
    return anira_test::explicit_contract(k_block, k_rate, on_miss, 0.0, /*budget_ms=*/10.0);
}

// ---- the rig ---------------------------------------------------------------------------------

/// A context, a pipeline with the three engines and an inference stage over `variants` under
/// `candidates` (none: the default set), an optional processor, and the handler over it. The
/// status of the stage's add and of the create are m_status (the first that failed), with its
/// message; a test prepares the handler itself.
class Rig {
public:
    explicit Rig(const std::vector<const anira::ModelConfig*>& variants,
                 const std::vector<anira_backend_id>& candidates = {},
                 const anira_processor_desc* processor = nullptr,
                 uint32_t threads = 2,
                 anira_log_level level = ANIRA_LOG_ERROR)
        : m_context(threads, ANIRA_WAIT_SPIN_BACKOFF, level) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        anira_custom_engine* const first = make_factor_engine(m_engine, k_engine, k_served);
        anira_custom_engine* const second = make_factor_engine(m_other, k_other_engine);
        anira_custom_engine* const third = make_step_engine(m_steps);
        anira_test::add_engine(m_pipeline, first);
        anira_test::add_engine(m_pipeline, second);
        anira_test::add_engine(m_pipeline, third);
        anira_custom_engine_destroy(first);
        anira_custom_engine_destroy(second);
        anira_custom_engine_destroy(third);
        std::vector<const anira_model_config*> natives;
        natives.reserve(variants.size());
        for (const anira::ModelConfig* variant : variants) { natives.push_back(variant->native()); }
        m_status = anira_pipeline_add_inference(m_pipeline,
                                                natives.data(),
                                                static_cast<uint32_t>(natives.size()),
                                                candidates.empty() ? nullptr : candidates.data(),
                                                static_cast<uint32_t>(candidates.size()),
                                                &err);
        if (m_status == ANIRA_OK && processor != nullptr) {
            m_status = anira_pipeline_add_processor(m_pipeline, processor, &err);
        }
        if (m_status == ANIRA_OK) {
            m_status = anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err);
        }
        m_message = err.message;
    }
    ~Rig() {
        m_engine.m_open.store(true);
        m_other.m_open.store(true);
        m_steps.m_open.store(true);
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;
    Rig(Rig&&) = delete;
    Rig& operator=(Rig&&) = delete;

    anira_handler* h() const { return m_handler; }

    anira_status prepare(const anira::ContractHandle& contract) {
        anira_error err = ANIRA_ERROR_INIT;
        const anira_status status = anira_handler_prepare(m_handler, contract.native(), &err);
        m_message = err.message;
        return status;
    }

    /// The rows of the prepared handler's plan report.
    std::vector<anira_plan_info> plans() const {
        const anira_plan_report* report = anira_handler_plan_report(m_handler);
        EXPECT_NE(report, nullptr);
        uint32_t count = anira_plan_report_num_plans(report);
        std::vector<anira_plan_info> rows(count);
        EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
                  ANIRA_OK);
        return rows;
    }

    /// One block through anira_handler_process, in place, without waiting for its inference:
    /// the status, and the block as delivered.
    anira_status process(std::vector<float>& block) const {
        const std::array<float*, 1> planes{block.data()};
        const anira_tensor io = anira_test::planar_f32(planes.data(), 1, block.size());
        size_t delivered = 0;
        const anira_status status = anira_handler_process(m_handler, &io, 0, &io, 0, &delivered);
        EXPECT_EQ(delivered, block.size());
        return status;
    }

    /// One block through the handler, its inference waited for and collected: what the call
    /// delivered. Every block of a stream run this way is on time, so the stream is the closed
    /// form of the plans that ran it.
    std::vector<float> run_block(std::span<const float> in) const {
        std::vector<float> block(in.begin(), in.end());
        const size_t before = anira_test::available(m_handler);
        EXPECT_EQ(process(block), ANIRA_OK) << "a block that is on time is no miss";
        anira_test::wait_for_block(m_handler, before);
        return block;
    }

    /// One block into the input ring alone (anira_handler_push_data): its chunk is formed and
    /// submitted, nothing is popped.
    anira_status push(std::span<const float> in) const {
        const std::array<const float*, 1> planes{in.data()};
        const anira_tensor block = anira_test::planar_f32(planes.data(), 1, in.size());
        return anira_handler_push_data(m_handler, &block, 0);
    }

    /// One block of `samples` out of the output ring alone (anira_handler_pop_data).
    std::vector<float> pop(size_t samples) const {
        std::vector<float> block(samples, -1.0F);
        const std::array<float*, 1> planes{block.data()};
        const anira_tensor out = anira_test::planar_f32(planes.data(), 1, samples);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_pop_data(m_handler, &out, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, samples);
        return block;
    }

    FactorEngine m_engine;  ///< before the handler, which dies first
    FactorEngine m_other;
    StepEngine m_steps;
    anira_status m_status = ANIRA_OK;
    std::string m_message;

private:
    anira_test::Context m_context;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
};

/// The engine id and the provider id of a plan row as strings (empty for a row without one).
std::string engine_id_of(const anira_plan_info& row) {
    return row.engine_id != nullptr ? row.engine_id : "";
}
std::string provider_id_of(const anira_plan_info& row) {
    return row.provider_id != nullptr ? row.provider_id : "";
}

/// anira_plan_report_find as a value: the plan's index, or -1 when no plan matches.
int find_plan(const anira_handler* handler, uint32_t variant, const anira_backend_id* backend) {
    anira_plan_query query = ANIRA_PLAN_QUERY_INIT;
    query.variant = variant;
    if (backend != nullptr) { query.backend = *backend; }
    uint32_t plan = 0;
    const anira_status status =
        anira_plan_report_find(anira_handler_plan_report(handler), &query, &plan);
    if (status == ANIRA_ERROR_NOT_FOUND) { return -1; }
    EXPECT_EQ(status, ANIRA_OK);
    return static_cast<int>(plan);
}

void expect_contains(const std::string& text, const std::string& fragment) {
    EXPECT_NE(text.find(fragment), std::string::npos)
        << "'" << fragment << "' is not in '" << text << "'";
}

// ---- a processor that records the variant every phase is told ----------------------------------

/// What every phase of a chunk was told as its variant (anira_processor_ctx.variant). The two
/// driving-thread phases record into fixed room (they run inside the real-time process call).
struct VariantRecorder {
    static constexpr size_t k_capacity = 64;
    std::array<uint32_t, k_capacity> m_pre{};
    size_t m_num_pre = 0;
    std::array<uint32_t, k_capacity> m_post{};
    size_t m_num_post = 0;
    std::atomic<uint32_t> m_before{ANIRA_VARIANT_ANY};
    std::atomic<uint32_t> m_after{ANIRA_VARIANT_ANY};
};

anira_status ANIRA_CALL record_pre(const anira_processor_ctx* ctx,
                                   void* /*prepared*/,
                                   void* user_data) noexcept {
    auto& recorder = *static_cast<VariantRecorder*>(user_data);
    if (recorder.m_num_pre < VariantRecorder::k_capacity) {
        recorder.m_pre.at(recorder.m_num_pre++) = ctx->variant;
    }
    return anira_processor_default_pre_process(ctx);
}

anira_status ANIRA_CALL record_post(const anira_processor_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) noexcept {
    auto& recorder = *static_cast<VariantRecorder*>(user_data);
    if (recorder.m_num_post < VariantRecorder::k_capacity) {
        recorder.m_post.at(recorder.m_num_post++) = ctx->variant;
    }
    return anira_processor_default_post_process(ctx);
}

anira_status ANIRA_CALL record_before(const anira_processor_ctx* ctx,
                                      void* /*prepared*/,
                                      void* user_data) noexcept {
    static_cast<VariantRecorder*>(user_data)->m_before.store(ctx->variant);
    return ANIRA_OK;
}

anira_status ANIRA_CALL record_after(const anira_processor_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    static_cast<VariantRecorder*>(user_data)->m_after.store(ctx->variant);
    return ANIRA_OK;
}

anira_processor_desc recording_processor(VariantRecorder& recorder) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.user_data = &recorder;
    desc.realtime_phases =
        ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS) | ANIRA_PHASE_BIT(ANIRA_PHASE_POST_PROCESS);
    desc.pre_process = record_pre;
    desc.post_process = record_post;
    desc.before_inference = record_before;
    desc.after_inference = record_after;
    return desc;
}

}  // namespace

// ============================================================================================
// The plan table
// ============================================================================================

// Two variants under two candidates are four plans, in variant order and, within a variant, in
// candidate order; a row carries its variant, and anira_plan_report_find resolves a variant, a
// backend, or both.
TEST(Variants, ThePlanTableSpansTheVariantsInOrder) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;

    const std::vector<anira_plan_info> rows = rig.plans();
    ASSERT_EQ(rows.size(), 4U);
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        EXPECT_EQ(rows[plan].variant, plan / 2);
        EXPECT_EQ(rows[plan].engine, static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM));
        EXPECT_EQ(engine_id_of(rows[plan]), k_engine);
        EXPECT_EQ(provider_id_of(rows[plan]), plan % 2 == 0 ? "" : k_npu);
        EXPECT_EQ(
            rows[plan].provider,
            static_cast<uint32_t>(plan % 2 == 0 ? ANIRA_PROVIDER_CPU : ANIRA_PROVIDER_CUSTOM));
        // The handler's own table and the session's carry the same variant under the index.
        EXPECT_EQ(rig.h()->m_plans.at(plan).m_variant, plan / 2);
        EXPECT_EQ(rig.h()->m_plans.at(plan).m_row, 0U);
        EXPECT_EQ(anira_test::session_of(rig.h())->m_plans.at(plan).m_variant, plan / 2);
    }
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 0U) << "prepare selects the first variant's plan";

    const anira_backend_id cpu = candidate(k_engine);
    const anira_backend_id npu = candidate(k_engine, k_npu);
    EXPECT_EQ(find_plan(rig.h(), 0, nullptr), 0);
    EXPECT_EQ(find_plan(rig.h(), 1, nullptr), 2) << "a variant alone: its first plan";
    EXPECT_EQ(find_plan(rig.h(), ANIRA_VARIANT_ANY, &npu), 1) << "a backend alone: its first plan";
    EXPECT_EQ(find_plan(rig.h(), 0, &cpu), 0);
    EXPECT_EQ(find_plan(rig.h(), 0, &npu), 1);
    EXPECT_EQ(find_plan(rig.h(), 1, &cpu), 2);
    EXPECT_EQ(find_plan(rig.h(), 1, &npu), 3);
    EXPECT_EQ(find_plan(rig.h(), 2, nullptr), -1) << "the stage has two variants";

    // Every plan is a loaded model of its own, loaded from its own variant's entry: the factor
    // its path spells, times the provider's.
    EXPECT_EQ(rig.m_engine.m_loads.load(), 4);
    const std::array<float, 4> factors{2.0F, 2.0F * k_npu_factor, 4.0F, 4.0F * k_npu_factor};
    const std::vector<float> in = anira_test::ramp(1);
    const size_t latency = anira_handler_get_latency(rig.h(), 0);
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        anira_handler_reset(rig.h());
        ASSERT_EQ(anira_handler_set_plan(rig.h(), plan), ANIRA_OK);
        anira_test::DelayLine line(latency);
        for (size_t block = 0; block < 3; ++block) {
            anira_test::expect_same_block(rig.run_block(in),
                                          line.next(in, factors.at(plan)),
                                          block);
        }
    }
}

// An entry pinned to one provider is a plan of that provider's candidate alone, so its variant
// has fewer plans than the other; the table stays in variant order.
TEST(Variants, ASparseSetGivesAVariantFewerPlans) {
    const anira::ModelConfig first = factor_model("2");
    anira::ModelConfig second = factor_model("4");
    second.model_provider(0, std::string_view(k_npu));
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;

    const std::vector<anira_plan_info> rows = rig.plans();
    ASSERT_EQ(rows.size(), 3U);
    EXPECT_EQ(rows[0].variant, 0U);
    EXPECT_EQ(rows[1].variant, 0U);
    EXPECT_EQ(rows[2].variant, 1U);
    EXPECT_EQ(provider_id_of(rows[2]), k_npu);
    const anira_backend_id cpu = candidate(k_engine);
    const anira_backend_id npu = candidate(k_engine, k_npu);
    EXPECT_EQ(find_plan(rig.h(), 1, &cpu), -1) << "the second variant has no plan on the CPU path";
    EXPECT_EQ(find_plan(rig.h(), 1, &npu), 2);
    EXPECT_EQ(find_plan(rig.h(), 1, nullptr), 2);
}

// The default candidate set (a NULL list) is the union over the entries of every variant: the
// custom engine only the second variant names is a candidate, and so is the provider only its
// entry is pinned to.
TEST(Variants, TheDefaultSetIsTheUnionOverTheVariants) {
    const anira::ModelConfig first = factor_model("2");
    anira::ModelConfig second = factor_model("4", k_other_engine);
    const uint32_t pinned = second.add_model_path(k_engine, "8");
    second.model_provider(pinned, std::string_view(k_npu));
    Rig rig({&first, &second});
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;

    const std::vector<anira_plan_info> rows = rig.plans();
    ASSERT_EQ(rows.size(), 3U);
    EXPECT_EQ(rows[0].variant, 0U);
    EXPECT_EQ(engine_id_of(rows[0]), k_engine);
    EXPECT_EQ(provider_id_of(rows[0]), "");
    EXPECT_EQ(rows[1].variant, 1U);
    EXPECT_EQ(engine_id_of(rows[1]), k_other_engine);
    EXPECT_EQ(provider_id_of(rows[1]), "");
    EXPECT_EQ(rows[2].variant, 1U);
    EXPECT_EQ(engine_id_of(rows[2]), k_engine);
    EXPECT_EQ(provider_id_of(rows[2]), k_npu);
    EXPECT_EQ(rig.h()->m_plans.at(2).m_row, 1U) << "the row in its own variant's models[]";
}

// A variant no candidate runs is ANIRA_ERROR_CONFIG at create, as for a stage of one variant,
// and the message names the variant.
TEST(Variants, AVariantWithoutAPlanIsRefusedByName) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4", k_other_engine);
    const Rig rig({&first, &second}, {candidate(k_engine)});
    EXPECT_EQ(rig.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message,
                    "variants[1]: none of the 1 model entries runs on a candidate backend");
    EXPECT_EQ(rig.h(), nullptr);
}

// ============================================================================================
// The form rule
// ============================================================================================

namespace {

/// The base of the form cases: a mono stream on either side and a Static side input.
struct FormParts {
    anira::TensorSpec m_in = anira_test::stream_tensor("in", 1, 512, 512);
    anira::TensorSpec m_aux = anira_test::static_tensor("aux", {1, 4});
    anira::TensorSpec m_out = anira_test::stream_tensor("out", 1, 512, 512);
    bool m_with_aux = true;

    anira::ModelConfig model() const {
        anira::ModelConfig config;
        config.add_model_path(k_engine, "2");
        config.input(m_in);
        if (m_with_aux) { config.input(m_aux); }
        config.output(m_out);
        return config;
    }
};

/// A Streamed tensor [Channel 1, Time `time`] with a declared window.
anira::TensorSpec windowed(std::string_view name,
                           int64_t time,
                           int64_t window_min,
                           int64_t window_max,
                           int64_t overlap) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_CHANNEL, 1).axis(1, ANIRA_AXIS_TIME, time);
    spec.window(window_min, window_max, overlap);
    return spec;
}

/// A stream with two declared State pairs: the State inputs "s1" and "s2" fed from `source1`
/// and `source2` of the State outputs "o1" and "o2".
anira::ModelConfig two_pairs(const char* source1, const char* source2) {
    anira::ModelConfig config;
    config.add_model_path(k_engine, "2");
    config.input(anira_test::stream_tensor("in", 1, 512, 512));
    config.input(state_tensor("s1").state_source(source1));
    config.input(state_tensor("s2").state_source(source2));
    config.output(anira_test::stream_tensor("out", 1, 512, 512));
    config.output(state_tensor("o1"));
    config.output(state_tensor("o2"));
    return config;
}

/// One property of the form: a first variant, a second one that differs in it alone, and how the
/// refusal starts (the variant, the slot, the property and both values).
struct FormCase {
    const char* m_property;
    std::function<anira::ModelConfig()> m_first;
    std::function<anira::ModelConfig()> m_other;
    const char* m_refusal;
};

std::vector<FormCase> form_cases() {
    const auto base = [] { return FormParts{}.model(); };
    std::vector<FormCase> cases;
    cases.push_back({"the number of input tensors",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_with_aux = false;
                         return parts.model();
                     },
                     "variants[1]: the number of input tensors is 1, but 2 in variants[0]"});
    cases.push_back({"the number of output tensors",
                     base,
                     [] {
                         anira::ModelConfig config = FormParts{}.model();
                         config.output(anira_test::static_tensor("peak", {1}));
                         return config;
                     },
                     "variants[1]: the number of output tensors is 2, but 1 in variants[0]"});
    cases.push_back({"the role",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_aux = anira_test::stream_tensor("aux", 1, 512, 512);
                         return parts.model();
                     },
                     "variants[1]: input slot 1 'aux': the role is Streamed, but Static in "
                     "variants[0]"});
    cases.push_back({"the canonical name",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_aux = anira_test::static_tensor("side", {1, 4});
                         return parts.model();
                     },
                     "variants[1]: input slot 1 'aux': the canonical name is 'side', but 'aux' in "
                     "variants[0]"});
    cases.push_back({"the dtype",
                     base,
                     [] {
                         FormParts parts;
                         anira::TensorSpec aux("aux", ANIRA_DTYPE_F64, ANIRA_ROLE_STATIC);
                         aux.axis(0, ANIRA_AXIS_ANY, 1).axis(1, ANIRA_AXIS_ANY, 4);
                         parts.m_aux = std::move(aux);
                         return parts.model();
                     },
                     "variants[1]: input slot 1 'aux': the dtype is float64, but float32 in "
                     "variants[0]"});
    cases.push_back({"the extents",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_aux = anira_test::static_tensor("aux", {1, 8});
                         return parts.model();
                     },
                     "variants[1]: input slot 1 'aux': the shape is [1, 8], but [1, 4] in "
                     "variants[0]"});
    // The same extents, [2, 512], under another axis tag: two ring channels against one.
    cases.push_back({"the channel count",
                     [] {
                         FormParts parts;
                         parts.m_in = anira_test::stream_tensor("in", 2, 512, 512);
                         parts.m_out = anira_test::stream_tensor("out", 2, 512, 512);
                         return parts.model();
                     },
                     [] {
                         FormParts parts;
                         anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
                         in.axis(0, ANIRA_AXIS_ANY, 2).axis(1, ANIRA_AXIS_TIME, 512);
                         in.window(512, 512, 0);
                         parts.m_in = std::move(in);
                         parts.m_out = anira_test::stream_tensor("out", 2, 512, 512);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the channel count is 1, but 2 in "
                     "variants[0]"});
    cases.push_back({"the window minimum",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_in = windowed("in", 512, 256, 512, 0);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the window minimum is 256, but 512 in "
                     "variants[0]"});
    cases.push_back({"the window maximum",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_in = windowed("in", 512, 512, 1024, 0);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the window maximum is 1024, but 512 in "
                     "variants[0]"});
    cases.push_back({"the overlap",
                     base,
                     [] {
                         FormParts parts;
                         parts.m_in = windowed("in", 512, 512, 512, 128);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the overlap is 128, but 0 in variants[0]"});
    // One flexible window, [256, 512], pinned differently: the second variant's output declares
    // a time ratio of 1/1, which fixes the anchor hop at its own window, 512; without it the
    // window takes the smallest hop, 256.
    cases.push_back({"the hop",
                     [] {
                         FormParts parts;
                         parts.m_in = windowed("in", 512, 256, 512, 0);
                         return parts.model();
                     },
                     [] {
                         FormParts parts;
                         parts.m_in = windowed("in", 512, 256, 512, 0);
                         parts.m_out.time_ratio(1, 1);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the hop is 512, but 256 in variants[0]"});
    // The same extents, [4, 4], with the Time tag on the other axis.
    cases.push_back({"the Time axis",
                     [] {
                         FormParts parts;
                         anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
                         in.axis(0, ANIRA_AXIS_TIME, 4).axis(1, ANIRA_AXIS_ANY, 4).window(4, 4, 0);
                         parts.m_in = std::move(in);
                         parts.m_out = anira_test::stream_tensor("out", 1, 4, 4);
                         return parts.model();
                     },
                     [] {
                         FormParts parts;
                         anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
                         in.axis(0, ANIRA_AXIS_ANY, 4).axis(1, ANIRA_AXIS_TIME, 4).window(4, 4, 0);
                         parts.m_in = std::move(in);
                         parts.m_out = anira_test::stream_tensor("out", 1, 4, 4);
                         return parts.model();
                     },
                     "variants[1]: input slot 0 'in': the Time axis is axis 1, but axis 0 in "
                     "variants[0]"});
    cases.push_back({"the anchor",
                     base,
                     [] {
                         anira::ModelConfig config = FormParts{}.model();
                         config.anchor("out");
                         return config;
                     },
                     "variants[1]: the anchor is output slot 0 'out', but input slot 0 'in' in "
                     "variants[0]"});
    cases.push_back({"the State pairing",
                     [] { return two_pairs("o1", "o2"); },
                     [] { return two_pairs("o2", "o1"); },
                     "variants[1]: input slot 1 's1': the state source is output slot 2 'o2', but "
                     "output slot 1 'o1' in variants[0]"});
    cases.push_back({"the exclusivity",
                     base,
                     [] {
                         anira::ModelConfig config = FormParts{}.model();
                         config.state(ANIRA_MODEL_STATEFUL);
                         return config;
                     },
                     "variants[1]: the model state is stateful (an exclusive handler), but "
                     "stateless (a shared handler) in variants[0]"});
    return cases;
}

}  // namespace

// One refusal per property of the form, at create: ANIRA_ERROR_CONFIG, the message naming the
// variant, the slot, the property and both values, and the rule. The pair in the other order is
// refused too; each variant beside itself is accepted.
TEST(Variants, EveryVariantHasTheFormOfTheFirst) {
    for (const FormCase& form : form_cases()) {
        SCOPED_TRACE(form.m_property);
        const anira::ModelConfig first = form.m_first();
        const anira::ModelConfig other = form.m_other();
        {
            const Rig rig({&first, &other});
            EXPECT_EQ(rig.m_status, ANIRA_ERROR_CONFIG);
            expect_contains(rig.m_message, form.m_refusal);
            expect_contains(rig.m_message,
                            "; every variant of an inference stage has the form of the first one "
                            "(a model of another form is another handler)");
            EXPECT_EQ(rig.h(), nullptr);
        }
        {
            const Rig swapped({&other, &first});
            EXPECT_EQ(swapped.m_status, ANIRA_ERROR_CONFIG);
            expect_contains(swapped.m_message, "variants[1]: ");
        }
        {
            const Rig alike({&first, &first});
            EXPECT_EQ(alike.m_status, ANIRA_OK) << alike.m_message;
        }
        {
            const Rig alike({&other, &other});
            EXPECT_EQ(alike.m_status, ANIRA_OK) << alike.m_message;
        }
    }
}

// A third variant is compared with the first, and named by its own index.
TEST(Variants, TheRefusalNamesTheVariantThatDiffers) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    FormParts parts;
    const anira::ModelConfig third = parts.model();
    const Rig rig({&first, &second, &third});
    EXPECT_EQ(rig.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message,
                    "variants[2]: the number of input tensors is 2, but 1 in variants[0]");
}

// The form is compared again under the contract at prepare, where a flexible window is pinned
// to the host block. Both variants take 256 to 1024 samples per inference and produce 256; the
// second declares its output's time ratio as 1/1, which fixes its anchor hop at 256. Without a
// contract both take the smallest hop, 256, and are of one form; under host blocks of 512 the
// first takes 512 per inference and the second 256.
TEST(Variants, TheFormIsComparedAgainUnderTheContract) {
    const auto flexible = [](bool declared_ratio) {
        anira::ModelConfig config;
        config.add_model_path(k_engine, "2");
        anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        in.axis(0, ANIRA_AXIS_CHANNEL, 1).axis(1, ANIRA_AXIS_TIME, ANIRA_DYNAMIC);
        in.window(256, 1024, 0);
        config.input(in);
        anira::TensorSpec out = anira_test::stream_tensor("out", 1, 256, 256);
        if (declared_ratio) { out.time_ratio(1, 1); }
        config.output(out);
        return config;
    };
    const anira::ModelConfig first = flexible(false);
    const anira::ModelConfig second = flexible(true);
    Rig rig({&first, &second});
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    EXPECT_EQ(rig.prepare(plain_contract()), ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message,
                    "variants[1]: input slot 0 'in': the shape is [1, 256], but [1, 512] in "
                    "variants[0]");
    EXPECT_EQ(anira_handler_plan_report(rig.h()), nullptr) << "a refused prepare prepares nothing";

    // Each of them alone prepares under that contract.
    Rig alone({&second});
    ASSERT_EQ(alone.m_status, ANIRA_OK) << alone.m_message;
    EXPECT_EQ(alone.prepare(plain_contract()), ANIRA_OK) << alone.m_message;
}

// What is free per variant: the entries (their number, engines, providers and files), an
// entry's layout and binding name, the default engine, max_instances, and how a window is
// spelled (declared against derived from the Time extent, which resolve to the same one).
TEST(Variants, WhatIsFreePerVariantMayDiffer) {
    anira::ModelConfig first;
    first.add_model_path(k_engine, "2");
    first.input(anira_test::stream_tensor("in", 1, k_block, k_block));
    first.output(anira_test::stream_tensor("out", 1, k_block, k_block));
    first.max_instances(1);

    anira::ModelConfig second;
    second.add_model_path(k_other_engine, "4");
    const uint32_t laid_out = second.add_model_path(k_engine, "8");
    second.default_engine(std::string_view(k_engine));
    // The window left out: derived from the Time extent, the same window of 512.
    anira::TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_CHANNEL, 1).axis(1, ANIRA_AXIS_TIME, static_cast<int64_t>(k_block));
    second.input(in);
    second.output(anira_test::stream_tensor("out", 1, k_block, k_block));
    second.tensor_name(laid_out, "in", "the_exports_input");
    const std::array<uint32_t, 2> time_first{1, 0};
    second.tensor_layout(laid_out, "in", time_first);
    second.max_instances(3);

    Rig rig({&first, &second}, {}, nullptr, /*threads=*/4);
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    const std::vector<anira_plan_info> rows = rig.plans();
    ASSERT_EQ(rows.size(), 3U);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 0U) << "the start is the first variant's";

    // The instances: each row reports its own variant's, the stream is sized for the fewest.
    EXPECT_EQ(rows[0].rt_parallel, 1U);
    EXPECT_EQ(rows[1].rt_parallel, 3U);
    EXPECT_EQ(rows[2].rt_parallel, 3U);
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(rig.h());
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->m_num_instances, 1U);
    EXPECT_EQ(session->m_plans.at(0).m_loaded->model().m_instances, 1U);
    EXPECT_EQ(session->m_plans.at(2).m_loaded->model().m_instances, 3U);
    Rig fewest({&first}, {}, nullptr, /*threads=*/4);
    ASSERT_EQ(fewest.prepare(plain_contract()), ANIRA_OK) << fewest.m_message;
    EXPECT_EQ(anira_handler_get_latency(rig.h(), 0), anira_handler_get_latency(fewest.h(), 0));
    EXPECT_EQ(anira_handler_num_entries(rig.h()), anira_handler_num_entries(fewest.h()));

    // Whether a window was derived is the plan's own variant's.
    const anira_plan_report* report = anira_handler_plan_report(rig.h());
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        uint32_t count = 1;
        anira_plan_slot slot = ANIRA_PLAN_SLOT_INIT;
        ASSERT_EQ(anira_plan_report_slots(report, plan, 1, sizeof(anira_plan_slot), &count, &slot),
                  ANIRA_OK);
        EXPECT_EQ(slot.window_derived, rows[plan].variant) << "plan " << plan;
        EXPECT_EQ(slot.window, static_cast<int64_t>(k_block)) << "plan " << plan;
        EXPECT_EQ(slot.hop, static_cast<int64_t>(k_block)) << "plan " << plan;
    }

    // Every plan streams its own entry: the second variant's laid-out entry among them.
    const std::array<float, 3> factors{2.0F, 4.0F, 8.0F};
    const std::vector<float> in_block = anira_test::ramp(1);
    const size_t latency = anira_handler_get_latency(rig.h(), 0);
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        anira_handler_reset(rig.h());
        ASSERT_EQ(anira_handler_set_plan(rig.h(), plan), ANIRA_OK);
        anira_test::DelayLine line(latency);
        for (size_t block = 0; block < 2; ++block) {
            anira_test::expect_same_block(rig.run_block(in_block),
                                          line.next(in_block, factors.at(plan)),
                                          block);
        }
    }
}

// ============================================================================================
// The warnings
// ============================================================================================

// Variants that differ in their internal latency are no refusal: the handler reports the largest
// figure per output, with one Warning naming the variants and the figures. The receive ring is
// primed without it, so the stream of either variant is delayed by the reported latency less the
// largest internal one.
TEST(Variants, TheLargestInternalLatencyIsReportedWithOneWarning) {
    constexpr uint32_t k_internal = 64;
    const anira::ModelConfig plain = factor_model("2");
    anira::ModelConfig late;
    late.add_model_path(k_engine, "4");
    late.input(anira_test::stream_tensor("in", 1, k_block, k_block));
    late.output(anira_test::stream_tensor("out", 1, k_block, k_block)
                    .latency(static_cast<int64_t>(k_internal)));
    late.max_instances(2);

    anira_drain_log();
    RecordCollector collector;
    Rig rig({&plain, &late}, {}, nullptr, 2, ANIRA_LOG_WARNING);
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    anira_drain_log();
#ifdef ENABLE_LOGGING
    const char* warning =
        "anira_handler_prepare: the variants differ in their internal latency (output slot 0 "
        "'out': variants[0] 0, variants[1] 64 samples, reported 64 samples)";
    EXPECT_EQ(anira_test::count_records(collector, warning, "native"), 1U);
    EXPECT_EQ(anira_test::find_record(collector, warning, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_WARNING));
#endif

    // The reported latency is the late variant's own, whichever variant comes first. (Every
    // context of the test asks for the same log level: the level is the process's.)
    Rig late_alone({&late}, {}, nullptr, 2, ANIRA_LOG_WARNING);
    ASSERT_EQ(late_alone.prepare(plain_contract()), ANIRA_OK) << late_alone.m_message;
    Rig plain_alone({&plain}, {}, nullptr, 2, ANIRA_LOG_WARNING);
    ASSERT_EQ(plain_alone.prepare(plain_contract()), ANIRA_OK) << plain_alone.m_message;
    const uint32_t reported = anira_handler_get_latency(rig.h(), 0);
    EXPECT_EQ(reported, anira_handler_get_latency(late_alone.h(), 0));
    EXPECT_EQ(reported, anira_handler_get_latency(plain_alone.h(), 0) + k_internal);
    Rig swapped({&late, &plain}, {}, nullptr, 2, ANIRA_LOG_WARNING);
    ASSERT_EQ(swapped.prepare(plain_contract()), ANIRA_OK) << swapped.m_message;
    EXPECT_EQ(anira_handler_get_latency(swapped.h(), 0), reported);

    // The engines here delay nothing, so both variants play the internal latency early.
    const std::vector<float> in = anira_test::ramp(1);
    const std::array<float, 2> factors{2.0F, 4.0F};
    for (uint32_t plan = 0; plan < 2; ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        anira_handler_reset(rig.h());
        ASSERT_EQ(anira_handler_set_plan(rig.h(), plan), ANIRA_OK);
        anira_test::DelayLine line(reported - k_internal);
        for (size_t block = 0; block < 2; ++block) {
            anira_test::expect_same_block(rig.run_block(in),
                                          line.next(in, factors.at(plan)),
                                          block);
        }
    }

    // Variants of one internal latency say nothing.
    RecordCollector quiet;
    Rig alike({&late, &late}, {}, nullptr, 2, ANIRA_LOG_WARNING);
    ASSERT_EQ(alike.prepare(plain_contract()), ANIRA_OK) << alike.m_message;
    anira_drain_log();
    EXPECT_EQ(anira_test::count_records(quiet, "differ in their internal latency", "native"), 0U);
}

// ============================================================================================
// The switch
// ============================================================================================

// A stream that switches plans every few blocks, across variants and across the backends of a
// variant, equals block by block the model of the plan that was selected when the block was
// formed, on one delay line: nothing restarts, nothing is filled, the latency stays.
TEST(Variants, ASwitchBetweenVariantsIsSeamless) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    const std::array<float, 4> factors{2.0F, 2.0F * k_npu_factor, 4.0F, 4.0F * k_npu_factor};
    const uint32_t latency = anira_handler_get_latency(rig.h(), 0);
    const uint32_t entries = anira_handler_num_entries(rig.h());

    // Variant 0 and 1 in turn, then the backends within one, then every plan once more.
    const std::array<uint32_t, 12> order{0, 2, 0, 3, 1, 2, 3, 0, 1, 1, 2, 0};
    anira_test::DelayLine line(latency);
    size_t block = 0;
    for (const uint32_t plan : order) {
        ASSERT_EQ(anira_handler_set_plan(rig.h(), plan), ANIRA_OK);
        ASSERT_EQ(anira_handler_get_plan(rig.h()), plan);
        for (int repeat = 0; repeat < 3; ++repeat, ++block) {
            const std::vector<float> in = anira_test::ramp(block);
            anira_test::expect_same_block(rig.run_block(in),
                                          line.next(in, factors.at(plan)),
                                          block);
        }
    }
    EXPECT_EQ(rig.m_engine.m_calls.load(), static_cast<int>(block)) << "one engine call per block";
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK) << "no miss, no drop, no refusal";
    EXPECT_EQ(anira_handler_get_latency(rig.h(), 0), latency);
    EXPECT_EQ(anira_handler_num_entries(rig.h()), entries);
}

// A chunk in flight when the selection moves to another variant finishes on its own variant:
// its engine, and the variant every one of its phases is told. The next chunk runs the new one.
TEST(Variants, AChunkInFlightFinishesOnItsVariant) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4", k_other_engine);
    VariantRecorder recorder;
    const anira_processor_desc processor = recording_processor(recorder);
    Rig rig({&first, &second}, {}, &processor);
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.plans().size(), 2U);
    const size_t latency = anira_handler_get_latency(rig.h(), 0);
    const size_t primed = anira_test::available(rig.h());

    // Chunk 1 is formed under variant 0 and held inside its engine's call.
    rig.m_engine.m_open.store(false);
    const std::vector<float> in = anira_test::ramp(1);
    std::vector<float> block = in;
    static_cast<void>(rig.process(block));
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (rig.m_engine.m_held.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_EQ(rig.m_engine.m_held.load(), 1) << "chunk 1 never reached its engine";

    // The switch to variant 1 lands while it is in flight.
    ASSERT_EQ(anira_handler_set_plan(rig.h(), 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 1U);
    rig.m_engine.m_open.store(true);
    anira_test::wait_for_available(rig.h(), primed);
    EXPECT_EQ(rig.m_engine.m_calls.load(), 1);
    EXPECT_EQ(rig.m_other.m_calls.load(), 0) << "the switch reached the chunk in flight";
    ASSERT_EQ(recorder.m_num_pre, 1U);
    EXPECT_EQ(recorder.m_pre[0], 0U);
    EXPECT_EQ(recorder.m_before.load(), 0U);
    EXPECT_EQ(recorder.m_after.load(), 0U);
    ASSERT_EQ(recorder.m_num_post, 1U);
    EXPECT_EQ(recorder.m_post[0], 0U) << "post_process was told the newly selected variant";

    // Chunk 2 is formed after the switch: variant 1 end to end. The two blocks that follow
    // deliver chunk 1 as variant 0 computed it and chunk 2 as variant 1 did.
    anira_test::DelayLine line(latency);
    static_cast<void>(line.next(in, 2.0F));  // the held call's block, delivered before its chunk
    anira_test::expect_same_block(rig.run_block(in), line.next(in, 4.0F), 1);
    EXPECT_EQ(rig.m_other.m_calls.load(), 1);
    EXPECT_EQ(rig.m_engine.m_calls.load(), 1);
    ASSERT_EQ(recorder.m_num_pre, 2U);
    EXPECT_EQ(recorder.m_pre[1], 1U);
    EXPECT_EQ(recorder.m_before.load(), 1U);
    EXPECT_EQ(recorder.m_after.load(), 1U);
    ASSERT_EQ(recorder.m_num_post, 2U);
    EXPECT_EQ(recorder.m_post[1], 1U);
    anira_test::expect_same_block(rig.run_block(in), line.next(in, 4.0F), 2);
}

// ============================================================================================
// The measurement
// ============================================================================================

// The warm-up pass runs every plan of every variant, a FIXED pass its count per plan, and under
// a MEASURED budget every row carries its own figure while the session runs the slowest.
TEST(Variants, EveryPlanOfEveryVariantIsWarmedUp) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    anira::ContractHandle measured(anira::Hard{.host_block_min = static_cast<double>(k_block),
                                               .host_block_max = static_cast<double>(k_block),
                                               .host_rate = k_rate,
                                               .budget = anira::Budget::measured()});
    measured.hard_warmup(ANIRA_WARMUP_FIXED, 3);
    measured.hard_on_miss(ANIRA_MISS_ZEROS);
    ASSERT_EQ(rig.prepare(measured), ANIRA_OK) << rig.m_message;

    const std::vector<anira_plan_info> rows = rig.plans();
    ASSERT_EQ(rows.size(), 4U);
    EXPECT_EQ(rig.m_engine.m_calls.load(), 12) << "three inferences of each of the four plans";
    for (const float factor : {2.0F, 2.0F * k_npu_factor, 4.0F, 4.0F * k_npu_factor}) {
        EXPECT_EQ(rig.m_engine.calls_of(factor), 3) << "the plan of factor " << factor;
    }
    double slowest = 0.0;
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        EXPECT_EQ(rows[plan].warmup_iterations, 3U);
        EXPECT_EQ(rows[plan].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_FIXED));
        EXPECT_GT(rows[plan].measured_ms, 0.0);
        EXPECT_DOUBLE_EQ(rows[plan].budget_ms, rows[plan].measured_ms);
        slowest = std::max(slowest, rows[plan].budget_ms);
    }
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(rig.h());
    ASSERT_NE(session, nullptr);
    EXPECT_DOUBLE_EQ(session->m_budget_ms, slowest);
}

// ============================================================================================
// Real models
// ============================================================================================

// The bundled CNN twice, as two variants, on every built-in engine of the build: a stream that
// moves to the next plan after every block (the other variant, the next engine) is, block by
// block, the model of the block's plan run directly on the window anira's default fill builds
// (anira_test::EngineOracle), behind the latency priming. The two variants allow different
// instance counts, so each is a loaded model of its own.
TEST(Variants, ABundledModelTwiceRunsOnEveryEngine) {
    constexpr size_t k_cnn_block = 2048;
    constexpr size_t k_cnn_window = 15380;
    const std::vector<anira_backend_id> candidates = anira_test::engine_candidates();
    if (candidates.empty()) { GTEST_SKIP() << "no built-in engine in this build"; }
    anira::ModelConfig first = anira_test::cnn_model();
    first.max_instances(2);
    anira::ModelConfig second = anira_test::cnn_model();
    second.max_instances(1);
    Rig rig({&first, &second}, candidates);
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(anira_test::file_contract(k_cnn_contract_json, k_cnn_block)), ANIRA_OK)
        << rig.m_message;
    anira_handler* h = rig.h();
    const std::vector<anira_plan_info> rows = rig.plans();
    const auto per_variant = static_cast<uint32_t>(rows.size() / 2);
    ASSERT_EQ(rows.size(), 2U * per_variant);
    ASSERT_GE(per_variant, 1U);
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(h);
    ASSERT_NE(session, nullptr);
    for (uint32_t plan = 0; plan < rows.size(); ++plan) {
        EXPECT_EQ(rows[plan].variant, plan / per_variant) << "plan " << plan;
    }
    EXPECT_NE(session->m_plans.at(0).m_loaded, session->m_plans.at(per_variant).m_loaded)
        << "the two variants are two loaded models";
    const uint32_t latency = anira_handler_get_latency(h, 0);

    // Variant 0 on engine e, then variant 1 on it, for every engine, twice over.
    std::vector<uint32_t> plan_of_block;
    for (int round = 0; round < 2; ++round) {
        for (uint32_t engine = 0; engine < per_variant; ++engine) {
            plan_of_block.push_back(engine);
            plan_of_block.push_back(per_variant + engine);
        }
    }
    const size_t priming = anira_test::available(h);
    std::vector<float> signal;
    std::vector<float> stream;
    for (size_t k = 0; k < plan_of_block.size(); ++k) {
        ASSERT_EQ(anira_handler_set_plan(h, plan_of_block[k]), ANIRA_OK);
        std::vector<float> block(k_cnn_block);
        for (size_t i = 0; i < k_cnn_block; ++i) {
            block[i] = 0.5F * std::sin(static_cast<float>(signal.size() + i) * 0.003F);
        }
        signal.insert(signal.end(), block.begin(), block.end());
        const std::vector<float> delivered = rig.run_block(block);
        stream.insert(stream.end(), delivered.begin(), delivered.end());
    }
    EXPECT_EQ(anira_handler_get_latency(h, 0), latency);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    std::map<uint32_t, std::unique_ptr<anira_test::EngineOracle>> oracles;
    std::vector<float> expected(priming, 0.0F);
    for (size_t j = 0; expected.size() < stream.size(); ++j) {
        std::unique_ptr<anira_test::EngineOracle>& oracle = oracles[plan_of_block.at(j)];
        if (oracle == nullptr) {
            oracle = std::make_unique<anira_test::EngineOracle>(h, plan_of_block.at(j));
            ASSERT_TRUE(oracle->ready()) << "plan " << plan_of_block.at(j);
        }
        const std::vector<std::vector<float>> out =
            oracle->run({anira_test::receptive_window(signal, j, k_cnn_block, k_cnn_window)});
        ASSERT_FALSE(out.empty()) << "the oracle's inference " << j << " failed";
        expected.insert(expected.end(), out[0].begin(), out[0].end());
    }
    size_t wrong = 0;
    for (size_t n = 0; n < stream.size(); ++n) {
        if (std::abs(stream[n] - expected[n]) > 1e-5F) {
            if (wrong == 0) {
                ADD_FAILURE() << "sample " << n << ": " << stream[n]
                              << ", the model of the block's plan on its window " << expected[n];
            }
            ++wrong;
        }
    }
    EXPECT_EQ(wrong, 0U) << "of " << stream.size() << " samples";
    EXPECT_GT(
        *std::max_element(expected.begin() + static_cast<std::ptrdiff_t>(priming), expected.end()),
        0.0F)
        << "the oracle's output is not silence";
}

// ============================================================================================
// The state of a variant switch
// ============================================================================================

namespace {

/// The plans of the state rigs: the step engine on the CPU path and on k_npu.
std::vector<anira_backend_id> steps_on_cpu_and_npu() {
    return {candidate(k_step_engine), candidate(k_step_engine, k_npu)};
}

/// One waited block per entry of `plans`, each after selecting its plan: the delivered stream.
std::vector<float> run_plans(const Rig& rig, const std::vector<uint32_t>& plans) {
    std::vector<float> stream;
    const std::vector<float> in = anira_test::ramp(1);
    for (const uint32_t plan : plans) {
        EXPECT_EQ(anira_handler_set_plan(rig.h(), plan), ANIRA_OK);
        const std::vector<float> delivered = rig.run_block(in);
        stream.insert(stream.end(), delivered.begin(), delivered.end());
    }
    return stream;
}

/// The stream of a step model whose chunks left the state at `states`, one hop each: the
/// latency's silence, then every state for one hop, cut to as many hops as there are states.
std::vector<float> stream_of(const std::vector<float>& states, size_t latency) {
    std::vector<float> stream(latency, 0.0F);
    for (const float state : states) { stream.insert(stream.end(), k_block, state); }
    stream.resize(states.size() * k_block);
    return stream;
}

}  // namespace

// The default, ANIRA_VARIANT_STATE_FRESH: the first chunk formed on another variant starts that
// variant from its initial state. The declared State pair reads zeros and the engine of the
// chunk's plan is reset, ahead of that chunk's inference; a switch back is a switch (A, B, A
// starts A fresh again). The stream goes on: no chunk is discarded, the generation and the
// latency stay.
TEST(Variants, AVariantSwitchStartsTheStateFreshByDefault) {
    const anira::ModelConfig first = step_model("1");
    const anira::ModelConfig second = step_model("100");
    Rig rig({&first, &second});
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.plans().size(), 2U);
    EXPECT_EQ(rig.h()->m_contract.hard()->m_variant_state, ANIRA_VARIANT_STATE_FRESH);
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(rig.h());
    ASSERT_NE(session, nullptr);
    EXPECT_TRUE(session->exclusive()) << "a declared State pair";
    const uint64_t generation = session->m_generation.load();
    const uint32_t latency = anira_handler_get_latency(rig.h(), 0);

    const std::vector<float> stream = run_plans(rig, {0, 0, 1, 1, 0, 0, 1});
    EXPECT_EQ(rig.m_steps.steps(), (std::vector<float>{1, 1, 100, 100, 1, 1, 100}));
    EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 0, 100, 0, 1, 0}))
        << "every first inference on another variant reads the initial state";
    EXPECT_EQ(rig.m_steps.counts(), (std::vector<int>{0, 1, 0, 1, 0, 1, 0}))
        << "and runs on an engine that was reset";
    EXPECT_EQ(rig.m_steps.resets(), (std::vector<float>{1, 100, 1, 100}))
        << "one reset at the stream's start, then one per variant switch, of the new plan's engine";
    EXPECT_EQ(stream, stream_of({1, 2, 100, 200, 1, 2, 100}, latency));
    EXPECT_EQ(session->m_state_epoch, 3U) << "three variant switches";
    EXPECT_EQ(session->m_generation.load(), generation) << "no chunk was made stale";
    EXPECT_EQ(anira_handler_get_latency(rig.h(), 0), latency);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);

    // A reset starts the stream over as it always did, and the next variant switch is one again.
    anira_handler_reset(rig.h());
    static_cast<void>(run_plans(rig, {1, 0}));
    EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 0, 100, 0, 1, 0, 0, 0}));
    EXPECT_EQ(rig.m_steps.resets(), (std::vector<float>{1, 100, 1, 100, 100, 1}));
}

// ANIRA_VARIANT_STATE_KEEP: nothing happens at a variant switch. The declared state carries
// over from one variant to the next, and each plan's engine continues from where it stopped
// (it is reset once, at its own first inference of the stream).
TEST(Variants, TheStateIsKeptWhenTheContractSaysSo) {
    const anira::ModelConfig first = step_model("1");
    const anira::ModelConfig second = step_model("100");
    Rig rig({&first, &second});
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    anira::ContractHandle keep = plain_contract();
    keep.hard_variant_state(ANIRA_VARIANT_STATE_KEEP);
    ASSERT_EQ(rig.prepare(keep), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(rig.h()->m_contract.hard()->m_variant_state, ANIRA_VARIANT_STATE_KEEP);
    const uint32_t latency = anira_handler_get_latency(rig.h(), 0);

    const std::vector<float> stream = run_plans(rig, {0, 0, 1, 1, 0, 0, 1});
    EXPECT_EQ(rig.m_steps.steps(), (std::vector<float>{1, 1, 100, 100, 1, 1, 100}));
    EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 2, 102, 202, 203, 204}))
        << "the state carries over";
    EXPECT_EQ(rig.m_steps.counts(), (std::vector<int>{0, 1, 0, 1, 2, 3, 2}))
        << "each engine continues from where it stopped";
    EXPECT_EQ(rig.m_steps.resets(), (std::vector<float>{1, 100}))
        << "each plan's engine once, at its first inference of the stream";
    EXPECT_EQ(stream, stream_of({1, 2, 102, 202, 203, 204, 304}, latency));
    EXPECT_EQ(anira_test::session_of(rig.h())->m_state_epoch, 0U);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);

    // The same handler prepared again under the default starts a variant fresh.
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    static_cast<void>(run_plans(rig, {0, 1}));
    EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 2, 102, 202, 203, 204, 0, 0}));
}

// A plan switch inside one variant (another backend of the same model) re-initialises nothing,
// in both modes: the declared state carries over. Behind a fresh variant switch each plan of the
// new variant is reset at its own first inference, so a backend switch inside it later resets
// that backend's engine once and keeps the declared state; under keep an engine is reset once,
// at its own first inference of the stream.
TEST(Variants, ABackendSwitchInsideAVariantKeepsTheState) {
    for (const anira_variant_state mode : {ANIRA_VARIANT_STATE_FRESH, ANIRA_VARIANT_STATE_KEEP}) {
        SCOPED_TRACE(mode == ANIRA_VARIANT_STATE_FRESH ? "fresh" : "keep");
        anira::ContractHandle contract = plain_contract();
        contract.hard_variant_state(mode);
        const bool fresh = mode == ANIRA_VARIANT_STATE_FRESH;
        {
            // One variant, two backends: the state epoch never moves.
            const anira::ModelConfig only = step_model("1");
            Rig rig({&only}, steps_on_cpu_and_npu());
            ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
            ASSERT_EQ(rig.prepare(contract), ANIRA_OK) << rig.m_message;
            ASSERT_EQ(rig.plans().size(), 2U);
            static_cast<void>(run_plans(rig, {0, 0, 1, 1, 0}));
            EXPECT_EQ(rig.m_steps.steps(), (std::vector<float>{1, 1, 16, 16, 1}));
            EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 2, 18, 34}));
            EXPECT_EQ(rig.m_steps.counts(), (std::vector<int>{0, 1, 0, 1, 2}));
            EXPECT_EQ(rig.m_steps.resets(), (std::vector<float>{1, 16}));
            EXPECT_EQ(anira_test::session_of(rig.h())->m_state_epoch, 0U);
        }
        {
            // Two variants, two backends each: plans 0 and 1 run variant 0 (steps 1 and 16),
            // plans 2 and 3 variant 1 (steps 100 and 1600).
            const anira::ModelConfig first = step_model("1");
            const anira::ModelConfig second = step_model("100");
            Rig rig({&first, &second}, steps_on_cpu_and_npu());
            ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
            ASSERT_EQ(rig.prepare(contract), ANIRA_OK) << rig.m_message;
            ASSERT_EQ(rig.plans().size(), 4U);
            static_cast<void>(run_plans(rig, {0, 1, 2, 1, 0}));
            EXPECT_EQ(rig.m_steps.steps(), (std::vector<float>{1, 16, 100, 16, 1}));
            // 0 to 1 and 1 to 0 stay inside variant 0; 1 to 2 and 2 to 1 switch the variant.
            // Fresh: plan 1 starts over at the switch back, and plan 0, which last ran before
            // the stream left its variant, is reset at its own first inference behind it while
            // the declared state plan 1 left carries over to it.
            EXPECT_EQ(rig.m_steps.states_read(),
                      fresh ? (std::vector<float>{0, 1, 0, 0, 16})
                            : (std::vector<float>{0, 1, 17, 117, 133}));
            EXPECT_EQ(
                rig.m_steps.counts(),
                fresh ? (std::vector<int>{0, 0, 0, 0, 0}) : (std::vector<int>{0, 0, 0, 1, 1}));
            EXPECT_EQ(
                rig.m_steps.resets(),
                fresh ? (std::vector<float>{1, 16, 100, 16, 1}) : (std::vector<float>{1, 16, 100}));
            EXPECT_EQ(anira_test::session_of(rig.h())->m_state_epoch, fresh ? 2U : 0U);
        }
    }
}

// A chunk in flight across a fresh variant switch is delivered, with the state it ran on: the
// switch re-initialises ahead of the first chunk formed on the other variant, never under a chunk
// formed before it. Chunk 1 is held inside variant 0's engine while the selection moves and chunk
// 2 is formed on variant 1; chunk 1 then completes on the state it read, and chunk 2 starts fresh.
TEST(Variants, AChunkInFlightKeepsItsStateAcrossAFreshSwitch) {
    const anira::ModelConfig first = step_model("1");
    const anira::ModelConfig second = step_model("100");
    Rig rig({&first, &second});
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    const size_t primed = anira_test::available(rig.h());
    const std::vector<float> in = anira_test::ramp(1);

    // Two chunks on variant 0 first, so that the held one reads a state other than zero.
    static_cast<void>(run_plans(rig, {0}));
    rig.m_steps.m_open.store(false);
    ASSERT_EQ(rig.push(in), ANIRA_OK);
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (rig.m_steps.m_held.load() == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    ASSERT_EQ(rig.m_steps.m_held.load(), 1) << "the chunk never reached its engine";
    ASSERT_EQ(anira_handler_set_plan(rig.h(), 1), ANIRA_OK);
    ASSERT_EQ(rig.push(in), ANIRA_OK) << "the chunk of the other variant, behind the held one";
    rig.m_steps.m_open.store(true);
    anira_test::wait_for_available(rig.h(), primed + 2 * k_block);

    EXPECT_EQ(rig.m_steps.steps(), (std::vector<float>{1, 1, 100}));
    EXPECT_EQ(rig.m_steps.states_read(), (std::vector<float>{0, 1, 0}))
        << "the held chunk read variant 0's state, the next one the initial state";
    EXPECT_EQ(rig.m_steps.resets(), (std::vector<float>{1, 100}));
    // Both chunks are in the output ring, in order, behind what the first block left of the
    // priming and its own chunk: nothing was discarded at the switch.
    ASSERT_GE(primed, k_block);
    std::vector<float> ring(primed - k_block, 0.0F);
    for (const float state : {1.0F, 2.0F, 100.0F}) { ring.insert(ring.end(), k_block, state); }
    std::vector<float> delivered;
    while (delivered.size() + k_block <= ring.size()) {
        const std::vector<float> block = rig.pop(k_block);
        delivered.insert(delivered.end(), block.begin(), block.end());
    }
    ASSERT_GE(delivered.size(), 2 * k_block);
    for (size_t n = 0; n < delivered.size(); ++n) {
        ASSERT_EQ(delivered[n], ring[n]) << "sample " << n;
    }
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
}

// ============================================================================================
// The selection by axis
// ============================================================================================

namespace {

/// The factor of the loaded model that ran the engine's last inference.
float last_factor(const FactorEngine& engine) {
    const int calls = engine.m_calls.load();
    return calls > 0 ? engine.m_factors.at(static_cast<size_t>(calls - 1)).load() : 0.0F;
}

}  // namespace

// anira_handler_set_model moves the variant and keeps the backend, anira_handler_set_backend
// moves the backend and keeps the variant; the next chunk runs the selected plan. Two variants
// under two backends: plans 0 to 3 are (variant 0, cpu), (variant 0, npu), (variant 1, cpu),
// (variant 1, npu).
TEST(Variants, SetModelKeepsTheBackendAndSetBackendKeepsTheVariant) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    anira_handler* h = rig.h();
    const anira_backend_id cpu = candidate(k_engine);
    const anira_backend_id npu = candidate(k_engine, k_npu);
    const std::vector<float> in = anira_test::ramp(1);
    const auto selects = [&](uint32_t plan, float factor) {
        EXPECT_EQ(anira_handler_get_plan(h), plan);
        static_cast<void>(rig.run_block(in));
        EXPECT_EQ(last_factor(rig.m_engine), factor) << "the chunk ran plan " << plan;
    };

    selects(0, 2.0F);
    EXPECT_EQ(anira_handler_set_model(h, 1), ANIRA_OK);
    selects(2, 4.0F);
    EXPECT_EQ(anira_handler_set_backend(h, &npu), ANIRA_OK);
    selects(3, 4.0F * k_npu_factor);
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_OK);
    selects(1, 2.0F * k_npu_factor);
    EXPECT_EQ(anira_handler_set_backend(h, &cpu), ANIRA_OK);
    selects(0, 2.0F);
    // The axis as it stands is a selection too.
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_backend(h, &cpu), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);

    // A backend names what it names, as the backend of a plan query does: an engine alone is
    // the variant's first plan on that engine, a provider alone its first on that provider,
    // nothing its first plan.
    ASSERT_EQ(anira_handler_set_plan(h, 3), ANIRA_OK);
    anira_backend_id engine_alone = ANIRA_BACKEND_ID_INIT;
    engine_alone.engine = ANIRA_ENGINE_CUSTOM;
    engine_alone.engine_id = k_engine;
    EXPECT_EQ(anira_handler_set_backend(h, &engine_alone), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 2U) << "variant 1's first plan on the engine";
    anira_backend_id provider_alone = ANIRA_BACKEND_ID_INIT;
    provider_alone.provider = ANIRA_PROVIDER_CUSTOM;
    provider_alone.provider_id = k_npu;
    EXPECT_EQ(anira_handler_set_backend(h, &provider_alone), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 3U);
    const anira_backend_id any = ANIRA_BACKEND_ID_INIT;
    EXPECT_EQ(anira_handler_set_backend(h, &any), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);
    // A record that ends before the ids names the engine and the provider alone.
    anira_backend_id head_only = ANIRA_BACKEND_ID_INIT;
    head_only.struct_size = 3 * sizeof(uint32_t);
    head_only.provider = ANIRA_PROVIDER_CPU;
    head_only.engine_id = "never read";
    ASSERT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_backend(h, &head_only), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 0U) << "variant 0's plan on the CPU path";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// A pair without a plan changes nothing: ANIRA_ERROR_NOT_FOUND, the selection as it was, the
// status in rt_error and one record, the kind latched. The second variant's entry is pinned to
// the npu, so the table is (variant 0, cpu), (variant 0, npu), (variant 1, npu).
TEST(Variants, ASelectionWithoutAPlanChangesNothing) {
    const anira::ModelConfig first = factor_model("2");
    anira::ModelConfig second = factor_model("4");
    second.model_provider(0, std::string_view(k_npu));
    anira_drain_log();
    RecordCollector collector;
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    anira_handler* h = rig.h();
    const anira_backend_id cpu = candidate(k_engine);
    const anira_backend_id npu = candidate(k_engine, k_npu);

    // Unprepared: a no-op recorded as NOT_PREPARED; a NULL handler has no word to record on.
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_set_backend(h, &cpu), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_set_model(nullptr, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_set_backend(nullptr, &cpu), ANIRA_ERROR_INVALID_ARGUMENT);

    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.plans().size(), 3U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << "prepare re-arms";

    // Variant 1 has no plan on the CPU path, where the selection stands.
    EXPECT_EQ(anira_handler_set_model(h, 1), ANIRA_ERROR_NOT_FOUND);
    EXPECT_EQ(anira_handler_get_plan(h), 0U) << "the selection is unchanged";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_FOUND);
    // Over the npu it has one; from there the CPU path has none for it.
    EXPECT_EQ(anira_handler_set_backend(h, &npu), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_model(h, 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);
    EXPECT_EQ(anira_handler_set_backend(h, &cpu), ANIRA_ERROR_NOT_FOUND);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);
    // An index that names no variant, the any-variant constant of a query included.
    EXPECT_EQ(anira_handler_set_model(h, 2), ANIRA_ERROR_NOT_FOUND);
    EXPECT_EQ(anira_handler_set_model(h, ANIRA_VARIANT_ANY), ANIRA_ERROR_NOT_FOUND);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);
    // A backend no plan runs on: another engine, another provider.
    const anira_backend_id other = candidate(k_other_engine);
    EXPECT_EQ(anira_handler_set_backend(h, &other), ANIRA_ERROR_NOT_FOUND);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);

    // The arguments: a NULL backend, a record below its head, the pair rule on either axis.
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_backend(h, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_backend_id short_record = ANIRA_BACKEND_ID_INIT;
    short_record.struct_size = 2 * sizeof(uint32_t);
    EXPECT_EQ(anira_handler_set_backend(h, &short_record), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_backend_id custom_without_id = ANIRA_BACKEND_ID_INIT;
    custom_without_id.engine = ANIRA_ENGINE_CUSTOM;
    EXPECT_EQ(anira_handler_set_backend(h, &custom_without_id), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_backend_id id_beside_a_value = ANIRA_BACKEND_ID_INIT;
    id_beside_a_value.provider = ANIRA_PROVIDER_CPU;
    id_beside_a_value.provider_id = k_npu;
    EXPECT_EQ(anira_handler_set_backend(h, &id_beside_a_value), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_get_plan(h), 2U);

    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "anira_handler_set_model: not found", "rt"), 1U)
        << "the first refusal of the kind is the one record";
    EXPECT_EQ(anira_test::count_records(collector, ": not found", "rt"), 1U)
        << "the later ones, of either entry, are counted";
    EXPECT_EQ(
        anira_test::find_record(collector, "anira_handler_set_model: not found", "rt").m_flags,
        ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
#endif
}

// A set_model on one thread against a set_backend on another: both land, whichever comes first,
// since each resolves its plan from the selected one and stores it only while that plan is still
// selected. From (variant 0, cpu) the two calls end on (variant 1, npu), and back.
TEST(Variants, ConcurrentSetModelAndSetBackendBothLand) {
    const anira::ModelConfig first = factor_model("2");
    const anira::ModelConfig second = factor_model("4");
    Rig rig({&first, &second}, cpu_and_npu());
    ASSERT_EQ(rig.m_status, ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.prepare(plain_contract()), ANIRA_OK) << rig.m_message;
    anira_handler* h = rig.h();
    const anira_backend_id cpu = candidate(k_engine);
    const anira_backend_id npu = candidate(k_engine, k_npu);
    struct Move {
        uint32_t m_from;
        uint32_t m_variant;
        const anira_backend_id* m_backend;
        uint32_t m_to;
    };
    const std::array<Move, 2> moves{{{.m_from = 0, .m_variant = 1, .m_backend = &npu, .m_to = 3},
                                     {.m_from = 3, .m_variant = 0, .m_backend = &cpu, .m_to = 0}}};
    for (int round = 0; round < 300; ++round) {
        const Move& move = moves.at(static_cast<size_t>(round % 2));
        ASSERT_EQ(anira_handler_get_plan(h), move.m_from) << "round " << round;
        std::atomic<bool> go{false};
        std::thread model([&] {
            while (!go.load()) {}
            EXPECT_EQ(anira_handler_set_model(h, move.m_variant), ANIRA_OK);
        });
        std::thread backend([&] {
            while (!go.load()) {}
            EXPECT_EQ(anira_handler_set_backend(h, move.m_backend), ANIRA_OK);
        });
        go.store(true);
        model.join();
        backend.join();
        ASSERT_EQ(anira_handler_get_plan(h), move.m_to)
            << "round " << round << ": one call undid the other's axis";
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}
