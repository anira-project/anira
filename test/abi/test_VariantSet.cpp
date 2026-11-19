// The variants of a handler's inference stage, changed at run time: anira_handler_add_variant
// builds and measures the plans of one more model of the stage's form beside the playing stream
// and appends them to the plan table (src/capi/handler.cpp, "the variants of a handler"). What
// the suite pins:
//
//   the add             the stream stays ordered and complete while a variant is added; the new
//                       plans carry figures; a switch to the new variant runs it from the next
//                       chunk on; a report pointer taken before the add stays readable with the
//                       rows it had, and anira_handler_plan_report answers the new one; the
//                       latency, the entry count and the selection do not move;
//   beside the stream   the warm-up of an add runs in a session of its own: no job of it goes
//                       through the playing session, no phase of the processor runs for it, and
//                       the declared state of a playing exclusive handler is what it is on a twin
//                       that never adds;
//   a failed add        a form difference, a model that does not load, a plan whose alignment the
//                       playing stream's chunks lack: the report, the counts, the latency, the
//                       entry count and rt_error are as before, and every load has its unload;
//   an unprepared one   the variant joins the pipeline copy, and the next prepare builds it;
//   the warnings        an internal latency other than the reported one, and a plan slower than
//                       the budget the stream is sized for, are one Warning each;
//   the control lock    a control entry and a Static entry called from an engine's prepare
//                       during a live add are ANIRA_ERROR_INVALID_STATE; from a plain prepare's
//                       callback a control entry is refused too and a Static entry works.
//
// Engine-free: every plan runs on one registered engine whose loaded model is a gain, the factor
// the number its entry's path spells, so a block of the stream says which variant ran it. The
// driving thread is anira_test::BackgroundDriver, which feeds a ramp and checks every block
// against the delayed input times one factor.

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
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <ratio>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../support/log_record_collector.h"
#include "float_face.h"
#include "handler_support.h"
#include "scheduler/Core.h"
#include "scheduler/SessionElement.h"

namespace {

using anira::SessionElement;
using anira_test::BackgroundDriver;
using anira_test::RecordCollector;

constexpr size_t k_hop = 64;       // the window of the stream, and the contract's block
constexpr double k_rate = 6400.0;  // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr double k_budget_ms = 1.0;
constexpr uint32_t k_warmup = 2;  // the inferences of a plan's pass (ANIRA_WARMUP_FIXED)
constexpr const char* k_engine = "org.example.variant_set";
/// A second engine id, which no pipeline of the suite adds: an entry no build can run.
constexpr const char* k_absent_engine = "org.example.variant_set_absent";
/// A second engine of every pipeline of the suite, which only the entries that name it run on.
constexpr const char* k_second_engine = "org.example.variant_set_second";
constexpr const char* k_fail_path = "fail";
/// The factors of the suite are whole numbers below this.
constexpr size_t k_max_factor = 128;

// ---- the engine ------------------------------------------------------------------------------

/// A registered C engine whose loaded model is a gain: output 0 is input 0 times a factor, the
/// number the entry's path spells ("2" is 2), read at load through the load record's variant.
/// The path k_fail_path does not load (ANIRA_ERROR_MODEL_LOAD). Every slot of the lifecycle is
/// filled and counted, and a call is counted under its model's factor, so a test reads which
/// loaded model ran how often. A closed gate of a factor holds the calls of that model inside
/// process (m_held counts them); a sleep of a factor makes that model slow.
///
/// m_stateful: the model has one declared State pair (input slot 1 fed from output slot 1). An
/// inference adds the model's factor to the state it reads, writes the sum as the new state and
/// multiplies the stream by it, so the factor of a block is the state after its inference.
/// m_reads_static: input slot 1 is a Static tensor, whose first element every call notes.
struct SetEngine {
    struct Loaded {
        float m_factor = 1.0F;
    };
    struct Prepared {
        float m_factor = 1.0F;
    };

    bool m_stateful = false;
    bool m_reads_static = false;
    std::atomic<int> m_loads{0};
    std::atomic<int> m_unloads{0};
    std::atomic<int> m_prepares{0};
    std::atomic<int> m_unprepares{0};
    std::array<std::atomic<int>, k_max_factor> m_calls{};
    /// The resets of an exclusive handler's engine, under the factor of the plan that was reset.
    std::array<std::atomic<int>, k_max_factor> m_resets{};
    std::array<std::atomic<bool>, k_max_factor> m_closed{};
    std::array<std::atomic<int>, k_max_factor> m_held{};
    std::array<std::atomic<int>, k_max_factor> m_sleep_us{};
    /// Per factor: every inference of that model fails (ANIRA_ERROR_ENGINE).
    std::array<std::atomic<bool>, k_max_factor> m_failing{};
    /// The record of the prepare that ran last: its flags, its entry count, the rows of its
    /// report.
    std::atomic<uint32_t> m_prepare_flags{0};
    std::atomic<uint32_t> m_prepare_entries{0};
    std::atomic<uint32_t> m_prepare_rows{0};
    std::atomic<float> m_static_seen{0.0F};
    /// What a prepare runs with its record, on the thread of the control entry, when set.
    std::function<void(const anira_prepare_info&)> m_on_prepare;

    static size_t index_of(float factor) {
        return std::min(static_cast<size_t>(factor), k_max_factor - 1);
    }
    int calls_of(float factor) const { return m_calls.at(index_of(factor)).load(); }
    int resets_of(float factor) const { return m_resets.at(index_of(factor)).load(); }
    int held_of(float factor) const { return m_held.at(index_of(factor)).load(); }
    void close(float factor) { m_closed.at(index_of(factor)).store(true); }
    void open(float factor) { m_closed.at(index_of(factor)).store(false); }
    void open_all() {
        for (std::atomic<bool>& closed : m_closed) { closed.store(false); }
    }
};

anira_status ANIRA_CALL set_load(const anira_engine_load_info* info,
                                 void* user_data,
                                 void** out_loaded) {
    auto* engine = static_cast<SetEngine*>(user_data);
    const char* path = anira_model_config_model_path(info->model, info->row);
    if (path == nullptr) { return ANIRA_ERROR_CONFIG; }
    if (std::string_view(path) == k_fail_path) { return ANIRA_ERROR_MODEL_LOAD; }
    auto loaded = std::make_unique<SetEngine::Loaded>();
    loaded->m_factor = static_cast<float>(std::atof(path));
    engine->m_loads.fetch_add(1);
    *out_loaded = loaded.release();
    return ANIRA_OK;
}

void ANIRA_CALL set_unload(void* loaded, void* user_data) {
    static_cast<SetEngine*>(user_data)->m_unloads.fetch_add(1);
    delete static_cast<SetEngine::Loaded*>(loaded);
}

anira_status ANIRA_CALL set_prepare(const anira_prepare_info* info,
                                    void* loaded,
                                    void* user_data,
                                    void** out_prepared) {
    auto* engine = static_cast<SetEngine*>(user_data);
    if (loaded == nullptr) { return ANIRA_ERROR_INTERNAL; }
    engine->m_prepare_flags.store(info->flags);
    engine->m_prepare_entries.store(info->num_entries);
    engine->m_prepare_rows.store(anira_plan_report_num_plans(info->report));
    if (engine->m_on_prepare) { engine->m_on_prepare(*info); }
    auto prepared = std::make_unique<SetEngine::Prepared>();
    prepared->m_factor = static_cast<const SetEngine::Loaded*>(loaded)->m_factor;
    engine->m_prepares.fetch_add(1);
    *out_prepared = prepared.release();
    return ANIRA_OK;
}

void ANIRA_CALL set_unprepare(void* prepared, void* user_data) {
    static_cast<SetEngine*>(user_data)->m_unprepares.fetch_add(1);
    delete static_cast<SetEngine::Prepared*>(prepared);
}

void ANIRA_CALL set_reset(const anira_engine_ctx* /*ctx*/, void* prepared, void* user_data) {
    auto* engine = static_cast<SetEngine*>(user_data);
    const auto* own = static_cast<const SetEngine::Prepared*>(prepared);
    if (own != nullptr) { engine->m_resets.at(SetEngine::index_of(own->m_factor)).fetch_add(1); }
}

anira_status ANIRA_CALL set_process(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    auto* engine = static_cast<SetEngine*>(user_data);
    const auto* own = static_cast<const SetEngine::Prepared*>(prepared);
    if (own == nullptr) { return ANIRA_ERROR_INTERNAL; }
    const size_t index = SetEngine::index_of(own->m_factor);
    engine->m_held.at(index).fetch_add(1);
    while (engine->m_closed.at(index).load()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    engine->m_held.at(index).fetch_sub(1);
    if (const int sleep_us = engine->m_sleep_us.at(index).load(); sleep_us > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    }
    if (engine->m_failing.at(index).load()) { return ANIRA_ERROR_ENGINE; }
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    float gain = own->m_factor;
    if (engine->m_stateful) {
        if (ctx->num_inputs < 2 || ctx->num_outputs < 2) { return ANIRA_ERROR_INTERNAL; }
        const float* state_in = anira_tensor_data_f32(&ctx->inputs[1]);
        float* state_out = anira_tensor_data_f32(&ctx->outputs[1]);
        if (state_in == nullptr || state_out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        gain = state_in[0] + own->m_factor;
        state_out[0] = gain;
    } else if (engine->m_reads_static && ctx->num_inputs > 1) {
        if (const float* value = anira_tensor_data_f32(&ctx->inputs[1])) {
            engine->m_static_seen.store(value[0]);
        }
    }
    for (size_t i = 0; i < anira_test::elements_of(ctx->outputs[0]); ++i) { out[i] = in[i] * gain; }
    engine->m_calls.at(index).fetch_add(1);
    return ANIRA_OK;
}

// ---- the models ------------------------------------------------------------------------------

/// A mono stream of `hop` samples per inference on either side ("in", "out") over one entry of
/// `engine` whose path spells the model's factor.
anira::ModelConfig gain_model(const char* factor,
                              int64_t hop = static_cast<int64_t>(k_hop),
                              const char* engine = k_engine) {
    anira::ModelConfig model;
    model.add_model_path(engine, factor);
    model.input(anira_test::stream_tensor("in", 1, hop, hop));
    model.output(anira_test::stream_tensor("out", 1, hop, hop));
    model.max_instances(2);
    return model;
}

/// gain_model() with one declared State pair ("state_in" fed from "state_out"), which makes
/// the handler exclusive: the model of an engine with m_stateful.
anira::ModelConfig state_model(const char* factor) {
    const auto state = [](std::string_view name) {
        anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
        spec.axis(0, ANIRA_AXIS_ANY, 4);
        return spec;
    };
    const auto hop = static_cast<int64_t>(k_hop);
    anira::ModelConfig model;
    model.add_model_path(k_engine, factor);
    model.input(anira_test::stream_tensor("in", 1, hop, hop));
    model.input(state("state_in").state_source("state_out"));
    model.output(anira_test::stream_tensor("out", 1, hop, hop));
    model.output(state("state_out"));
    return model;
}

/// gain_model() with a Static tensor of one element on either side ("gain", "peak"): the model
/// the two Static entries have a slot in.
anira::ModelConfig static_model(const char* factor) {
    anira::ModelConfig model = gain_model(factor);
    model.input(anira_test::static_tensor("gain", {1}));
    model.output(anira_test::static_tensor("peak", {1}));
    return model;
}

/// An explicit budget of k_budget_ms and a pass of `warmup` inferences per plan (0: no pass, the
/// rows stay unmeasured).
anira::ContractHandle set_contract(uint32_t warmup = k_warmup) {
    return anira_test::explicit_contract(static_cast<uint32_t>(k_hop),
                                         k_rate,
                                         ANIRA_MISS_ZEROS,
                                         0.0,
                                         k_budget_ms,
                                         warmup);
}

/// Waits until `done` answers true; false after anira_test::k_wait_s.
template <typename Predicate>
bool wait_for(Predicate done) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(anira_test::k_wait_s);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

// ---- the rig ---------------------------------------------------------------------------------

/// A context, a pipeline with the engine and an inference stage over `variants` under the
/// default candidate set (or `candidates`), an optional processor, and the handler over it.
class SetRig {
public:
    explicit SetRig(const std::vector<const anira::ModelConfig*>& variants,
                    const anira_processor_desc* processor = nullptr,
                    anira_log_level level = ANIRA_LOG_ERROR,
                    const std::vector<anira_backend_id>& candidates = {})
        : m_context(2, ANIRA_WAIT_SPIN_BACKOFF, level) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        add_engine(m_engine, k_engine);
        add_engine(m_second, k_second_engine);
        std::vector<const anira_model_config*> natives;
        natives.reserve(variants.size());
        for (const anira::ModelConfig* variant : variants) { natives.push_back(variant->native()); }
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               natives.data(),
                                               static_cast<uint32_t>(natives.size()),
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &err),
                  ANIRA_OK)
            << err.message;
        if (processor != nullptr) {
            EXPECT_EQ(anira_pipeline_add_processor(m_pipeline, processor, &err), ANIRA_OK)
                << err.message;
        }
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err), ANIRA_OK)
            << err.message;
    }
    ~SetRig() {
        m_engine.open_all();
        m_second.open_all();
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    SetRig(const SetRig&) = delete;
    SetRig& operator=(const SetRig&) = delete;
    SetRig(SetRig&&) = delete;
    SetRig& operator=(SetRig&&) = delete;

    anira_handler* h() const { return m_handler; }

    anira_status prepare(const anira::ContractHandle& contract) {
        anira_error err = ANIRA_ERROR_INIT;
        const anira_status status = anira_handler_prepare(m_handler, contract.native(), &err);
        m_message = err.message;
        return status;
    }

    /// anira_handler_add_variant: the status, the index through `variant`, the message kept.
    anira_status add(const anira::ModelConfig& model, uint32_t* variant) {
        anira_error err = ANIRA_ERROR_INIT;
        const anira_status status =
            anira_handler_add_variant(m_handler, model.native(), variant, &err);
        m_message = err.message;
        return status;
    }

    SetEngine m_engine;  ///< before the handler, which dies first
    SetEngine m_second;  ///< the engine under k_second_engine
    std::string m_message;

private:
    /// An engine object over `engine` under `id`, added to the pipeline, which holds it.
    void add_engine(SetEngine& engine, const char* id) const {
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.user_data = &engine;
        desc.load = set_load;
        desc.unload = set_unload;
        desc.prepare = set_prepare;
        desc.unprepare = set_unprepare;
        desc.reset = set_reset;
        desc.process = set_process;
        anira_custom_engine* object = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_custom_engine_create(id, &desc, &object, &err), ANIRA_OK) << err.message;
        anira_test::add_engine(m_pipeline, object);
        anira_custom_engine_destroy(object);
    }

    anira_test::Context m_context;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
};

/// The rows of a plan report.
std::vector<anira_plan_info> rows_of(const anira_plan_report* report) {
    uint32_t count = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> rows(count);
    if (count == 0) { return rows; }
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
              ANIRA_OK);
    return rows;
}

/// What a failed add must leave as it was: what a host reads of the handler, and what the
/// handler holds behind it.
struct HandlerState {
    explicit HandlerState(const SetRig& rig)
        : m_report(anira_handler_plan_report(rig.h()))
        , m_num_plans(anira_plan_report_num_plans(m_report))
        , m_latency(anira_handler_get_latency(rig.h(), 0))
        , m_entries(anira_handler_num_entries(rig.h()))
        , m_rt_error(anira_handler_rt_error(rig.h()))
        , m_selected(anira_handler_get_plan(rig.h()))
        , m_variants(rig.h()->m_pipeline.m_variants.size())
        , m_candidates(rig.h()->m_pipeline.m_candidates.size())
        , m_table(rig.h()->m_plans.size())
        , m_reports(rig.h()->m_reports.size())
        , m_strings(rig.h()->m_plan_strings.size())
        , m_session_plans(
              rig.h()->m_manager != nullptr ? anira_test::session_of(rig.h())->m_plans.size() : 0)
        , m_loaded(rig.m_engine.m_loads.load() - rig.m_engine.m_unloads.load())
        , m_prepared(rig.m_engine.m_prepares.load() - rig.m_engine.m_unprepares.load()) {}

    const anira_plan_report* m_report;
    uint32_t m_num_plans;
    uint32_t m_latency;
    uint32_t m_entries;
    anira_status m_rt_error;
    uint32_t m_selected;
    size_t m_variants;
    size_t m_candidates;
    size_t m_table;
    size_t m_reports;
    size_t m_strings;
    size_t m_session_plans;
    int m_loaded;    ///< the engine's loaded models: its loads less its unloads
    int m_prepared;  ///< and its prepared handles
};

void expect_unchanged(const SetRig& rig, const HandlerState& before) {
    const HandlerState now(rig);
    EXPECT_EQ(now.m_report, before.m_report) << "a failed call published a report";
    EXPECT_EQ(now.m_num_plans, before.m_num_plans);
    EXPECT_EQ(now.m_latency, before.m_latency);
    EXPECT_EQ(now.m_entries, before.m_entries);
    EXPECT_EQ(now.m_rt_error, before.m_rt_error);
    EXPECT_EQ(now.m_selected, before.m_selected);
    EXPECT_EQ(now.m_variants, before.m_variants) << "the pipeline copy kept the variant";
    EXPECT_EQ(now.m_candidates, before.m_candidates) << "the candidate set kept its growth";
    EXPECT_EQ(now.m_table, before.m_table);
    EXPECT_EQ(now.m_reports, before.m_reports);
    EXPECT_EQ(now.m_strings, before.m_strings);
    EXPECT_EQ(now.m_session_plans, before.m_session_plans) << "the session's list kept a slot";
    EXPECT_EQ(now.m_loaded, before.m_loaded) << "a load of the failed call has no unload";
    EXPECT_EQ(now.m_prepared, before.m_prepared) << "a prepare of it has no unprepare";
}

/// The stream of a parked driver against the factor every chunk ran with: block c + latency
/// carries `factor_of_chunk(c)`, and the blocks of the latency are silence.
template <typename FactorOfChunk>
void expect_stream(const BackgroundDriver& driver, FactorOfChunk factor_of_chunk) {
    driver.expect_continuous();
    const std::vector<float>& factors = driver.factors();
    const size_t latency = driver.latency_blocks();
    for (size_t block = 0; block < factors.size() && block < latency; ++block) {
        ASSERT_EQ(factors[block], 0.0F) << "the latency's silence, block " << block;
    }
    for (size_t chunk = 0; chunk + latency < factors.size(); ++chunk) {
        ASSERT_EQ(factors[chunk + latency], factor_of_chunk(chunk)) << "chunk " << chunk;
    }
}

void expect_contains(const std::string& text, const std::string& fragment) {
    EXPECT_NE(text.find(fragment), std::string::npos)
        << "'" << fragment << "' is not in '" << text << "'";
}

// ---- a processor that counts what it is called for ---------------------------------------------

/// The calls of a processor's prepare and of its before_inference hook. The hook has an id, so
/// the model's figure is keyed by it (the pass of a prepare runs the hook; an add's does not).
struct HookCounter {
    std::atomic<int> m_prepares{0};
    std::atomic<int> m_before{0};
};

anira_status ANIRA_CALL count_prepare(const anira_prepare_info* /*info*/,
                                      void* user_data,
                                      void** out_prepared) {
    static_cast<HookCounter*>(user_data)->m_prepares.fetch_add(1);
    *out_prepared = nullptr;
    return ANIRA_OK;
}

anira_status ANIRA_CALL count_before(const anira_processor_ctx* /*ctx*/,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    static_cast<HookCounter*>(user_data)->m_before.fetch_add(1);
    return ANIRA_OK;
}

/// What pre_process was told of every chunk of a stream, in the order the chunks were formed:
/// the variant, and whether the engine the context names is the custom one under k_engine or
/// under k_second_engine (0: neither). The phase runs inside the real-time process call, so the
/// room is fixed; the driving thread's own data, read while it is parked.
struct StampRecorder {
    static constexpr size_t k_capacity = size_t{1} << 16;
    std::vector<uint32_t> m_variant = std::vector<uint32_t>(k_capacity, 0);
    std::vector<uint8_t> m_engine = std::vector<uint8_t>(k_capacity, 0);
    size_t m_count = 0;
};

anira_status ANIRA_CALL record_stamp(const anira_processor_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    auto& recorder = *static_cast<StampRecorder*>(user_data);
    anira_engine engine = ANIRA_ENGINE_NONE;
    const char* id = nullptr;
    uint8_t named = 0;
    if (anira_processor_engine(ctx, &engine, &id) == ANIRA_OK && id != nullptr) {
        if (std::strcmp(id, k_engine) == 0) { named = 1; }
        if (std::strcmp(id, k_second_engine) == 0) { named = 2; }
    }
    if (recorder.m_count < StampRecorder::k_capacity) {
        recorder.m_variant[recorder.m_count] = ctx->variant;
        recorder.m_engine[recorder.m_count] = named;
        ++recorder.m_count;
    }
    return anira_processor_default_pre_process(ctx);
}

anira_processor_desc stamp_processor(StampRecorder& recorder) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.user_data = &recorder;
    desc.realtime_phases = ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS);
    desc.pre_process = record_stamp;
    return desc;
}

anira_processor_desc counting_processor(HookCounter& counter) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.user_data = &counter;
    desc.id = "org.example.variant_set_hook";
    desc.prepare = count_prepare;
    desc.before_inference = count_before;
    return desc;
}

}  // namespace

// ==== the add =================================================================================

// A variant added while a free-running driving thread plays joins the plan table behind the
// plans the handler has: the stream stays ordered and complete through the call, the new row
// carries the figure of its own pass, and a switch to the variant runs its model from the next
// chunk on. A report pointer a host took before the add still reads the rows it had; the
// handler's getter answers the new report. The latency, the entry count and the selection are
// what they were.
TEST(VariantSet, AVariantAddedBesideAPlayingStreamJoinsThePlanTable) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const HandlerState before(rig);
    ASSERT_EQ(before.m_num_plans, 1U);

    BackgroundDriver driver(rig.h(), k_hop);
    driver.run_free(1U << 20);
    ASSERT_TRUE(wait_for([&driver] { return driver.done() >= 8; }));

    uint32_t variant = 99;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U) << "the next index of the stage";
    const size_t done_at_add = driver.done();
    ASSERT_TRUE(wait_for([&] { return driver.done() >= done_at_add + 8; }))
        << "the stream stopped behind the add";

    // The report of the add is a new one; the one of the prepare is as it was.
    const anira_plan_report* const report = anira_handler_plan_report(rig.h());
    EXPECT_NE(report, before.m_report);
    EXPECT_EQ(anira_plan_report_num_plans(before.m_report), 1U);
    const std::vector<anira_plan_info> old_rows = rows_of(before.m_report);
    const std::vector<anira_plan_info> rows = rows_of(report);
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].variant, 0U);
    EXPECT_EQ(rows[0].measured_ms, old_rows.at(0).measured_ms) << "the first plan's row moved";
    EXPECT_EQ(rows[1].variant, 1U);
    EXPECT_EQ(rows[1].engine, static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM));
    ASSERT_NE(rows[1].engine_id, nullptr);
    EXPECT_STREQ(rows[1].engine_id, k_engine);
    EXPECT_EQ(rows[1].provider, static_cast<uint32_t>(ANIRA_PROVIDER_CPU));
    // The figure of the plan's own pass, beside the stream.
    EXPECT_EQ(rows[1].warmup_iterations, k_warmup);
    EXPECT_EQ(rows[1].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_FIXED));
    EXPECT_GT(rows[1].measured_ms, 0.0);
    EXPECT_NE(rows[1].rt_verdict, static_cast<uint32_t>(ANIRA_RT_UNMEASURED));
    EXPECT_EQ(rows[1].budget_ms, k_budget_ms) << "an explicit budget is every plan's";
    uint32_t slots = 0;
    EXPECT_EQ(anira_plan_report_slots(report, 1, 1, sizeof(anira_plan_slot), &slots, nullptr),
              ANIRA_OK);
    EXPECT_EQ(slots, 1U);

    // What does not move.
    EXPECT_EQ(anira_handler_get_latency(rig.h(), 0), before.m_latency);
    EXPECT_EQ(anira_handler_num_entries(rig.h()), before.m_entries);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), before.m_selected);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);

    // The engine's side: one more loaded model and one more handle, prepared for the playing
    // stream's entries under the live flag, with the new report in the record.
    EXPECT_EQ(rig.m_engine.m_loads.load(), 2);
    EXPECT_EQ(rig.m_engine.m_prepares.load(), 2);
    EXPECT_EQ(rig.m_engine.m_unloads.load(), 0);
    EXPECT_EQ(rig.m_engine.m_unprepares.load(), 0);
    EXPECT_NE(rig.m_engine.m_prepare_flags.load() & ANIRA_PREPARE_LIVE, 0U);
    EXPECT_EQ(rig.m_engine.m_prepare_entries.load(), before.m_entries);
    EXPECT_EQ(rig.m_engine.m_prepare_rows.load(), 2U);

    // The dense index is the position in the handler's table, the session's list and the
    // published rows.
    const std::shared_ptr<SessionElement> session = anira_test::session_of(rig.h());
    ASSERT_EQ(session->m_plans.size(), 2U);
    ASSERT_NE(session->m_plans[1], nullptr);
    EXPECT_EQ(session->m_plans[1]->m_variant, 1U);
    EXPECT_EQ(rig.h()->m_plans.size(), 2U);
    EXPECT_EQ(anira::Core::get_sessions().size(), 1U) << "the build's session is gone";

    // The switch: every chunk formed behind it runs the new variant.
    driver.stop();
    ASSERT_TRUE(driver.wait());
    const size_t switched_at = driver.done();
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 1U);
    driver.step(8);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver,
                  [switched_at](size_t chunk) { return chunk < switched_at ? 1.0F : 2.0F; });
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
}

// The warm-up of an add runs beside the stream and never through it: the playing model is
// called once per chunk of the stream and for nothing else, the new model for its own pass
// alone, and the processor, whose hook is part of every inference of the stream, is neither
// prepared again nor called for the build.
TEST(VariantSet, TheWarmUpOfAnAddNeverRunsThroughThePlayingSession) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    HookCounter counter;
    const anira_processor_desc processor = counting_processor(counter);
    SetRig rig({&one}, &processor);
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    // The pass of the prepare: its inferences on the one plan, each through the hook.
    ASSERT_EQ(rig.m_engine.calls_of(1.0F), static_cast<int>(k_warmup));
    ASSERT_EQ(counter.m_before.load(), static_cast<int>(k_warmup));
    ASSERT_EQ(counter.m_prepares.load(), 1);

    BackgroundDriver driver(rig.h(), k_hop);
    driver.run_free(1U << 20);
    ASSERT_TRUE(wait_for([&driver] { return driver.done() >= 4; }));
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    const size_t done_at_add = driver.done();
    ASSERT_TRUE(wait_for([&] { return driver.done() >= done_at_add + 4; }));
    driver.stop();
    ASSERT_TRUE(driver.wait());

    const int chunks = static_cast<int>(driver.done());
    EXPECT_EQ(rig.m_engine.calls_of(1.0F), static_cast<int>(k_warmup) + chunks)
        << "an inference of the playing model that is no chunk of the stream";
    EXPECT_EQ(rig.m_engine.calls_of(2.0F), static_cast<int>(k_warmup))
        << "the new model ran its own pass and nothing else";
    EXPECT_EQ(counter.m_before.load(), static_cast<int>(k_warmup) + chunks)
        << "the processor's hook ran for the build";
    EXPECT_EQ(counter.m_prepares.load(), 1) << "the processor was prepared again";
    expect_stream(driver, [](size_t /*chunk*/) { return 1.0F; });

    // The row says what was measured: the pass of the build, without the hook.
    const std::vector<anira_plan_info> rows = rows_of(anira_handler_plan_report(rig.h()));
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[1].warmup_iterations, k_warmup);
    EXPECT_EQ(rows[1].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_FIXED));

    // A second handler of the same pipeline shape prepares the same two models with the
    // hook: its pass measures the second model itself (the figure of the add lacks the hook
    // and is not its answer), and reuses the first one's.
    HookCounter second_counter;
    const anira_processor_desc second_processor = counting_processor(second_counter);
    SetRig second({&one, &two}, &second_processor);
    ASSERT_EQ(second.prepare(set_contract()), ANIRA_OK) << second.m_message;
    const std::vector<anira_plan_info> second_rows = rows_of(anira_handler_plan_report(second.h()));
    ASSERT_EQ(second_rows.size(), 2U);
    EXPECT_EQ(second_rows[0].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_EQ(second_rows[1].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_FIXED));
    EXPECT_EQ(second_counter.m_before.load(), static_cast<int>(k_warmup));
}

// A contract without a warm-up measures nothing at an add either: the model loads, its handle
// is prepared, no inference runs, and the row is unmeasured like the others.
TEST(VariantSet, AnAddUnderAContractWithoutAWarmUpRunsNoInference) {
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract(0)), ANIRA_OK) << rig.m_message;
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
    EXPECT_EQ(rig.m_engine.calls_of(1.0F), 0);
    EXPECT_EQ(rig.m_engine.calls_of(2.0F), 0);
    const std::vector<anira_plan_info> rows = rows_of(anira_handler_plan_report(rig.h()));
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[1].rt_verdict, static_cast<uint32_t>(ANIRA_RT_UNMEASURED));
    EXPECT_EQ(rows[1].warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_NONE));

    BackgroundDriver driver(rig.h(), k_hop);
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    driver.step(6);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [](size_t /*chunk*/) { return 2.0F; });
}

// Several variants added one behind the other beside a free-running stream, each selected as
// soon as it is there: the indices count up, the table grows by one plan per call, and the
// stream runs through every switch without a broken block, on factors that only grow. (The
// factors are powers of two, so a block times its factor is exact however long the ramp has
// run.)
TEST(VariantSet, VariantsAddedOneBehindTheOtherBesideAFreeRunningStream) {
    constexpr uint32_t k_added = 6;
    const anira::ModelConfig one = gain_model("1");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    BackgroundDriver driver(rig.h(), k_hop);
    driver.run_free(1U << 20);
    for (uint32_t round = 0; round < k_added; ++round) {
        const std::string factor = std::to_string(2U << round);
        const anira::ModelConfig model = gain_model(factor.c_str());
        uint32_t variant = 0;
        ASSERT_EQ(rig.add(model, &variant), ANIRA_OK) << rig.m_message;
        ASSERT_EQ(variant, round + 1);
        ASSERT_EQ(anira_handler_set_model(rig.h(), variant), ANIRA_OK);
        const size_t done = driver.done();
        ASSERT_TRUE(wait_for([&] { return driver.done() >= done + 3; }));
    }
    driver.stop();
    ASSERT_TRUE(driver.wait());
    driver.expect_continuous();
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(rig.h())), k_added + 1);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
    // The factors of the stream never fall: a chunk runs the variant selected when it was
    // formed, and the selection only moved to later variants.
    float last = 0.0F;
    for (const float factor : driver.factors()) {
        if (factor == 0.0F) { continue; }  // the latency's silence
        ASSERT_GE(factor, last);
        last = factor;
    }
    EXPECT_EQ(last, static_cast<float>(1U << k_added)) << "the stream ends on the last variant";
    EXPECT_EQ(rig.m_engine.m_loads.load(), static_cast<int>(k_added) + 1);
    EXPECT_EQ(rig.m_engine.m_unloads.load(), 0);
}

// A chunk of an added plan reports what a chunk of any plan reports: its variant, and its
// engine by the id the handler's session holds for its life (the build's session, where the
// plan was made, is gone by then). Under the default candidate set the set grows by the new
// variant's entries: a variant whose entry names another engine of the pipeline, which no
// variant named before, gets its plan as it would have with the pipeline's own variants.
TEST(VariantSet, AChunkOfAnAddedPlanReportsItsVariantAndItsEngine) {
    ASSERT_TRUE(anira_test::fresh_core());
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig four = gain_model("4", static_cast<int64_t>(k_hop), k_second_engine);
    StampRecorder recorder;
    const anira_processor_desc processor = stamp_processor(recorder);
    SetRig rig({&one}, &processor);
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const size_t candidates = rig.h()->m_pipeline.m_candidates.size();

    BackgroundDriver driver(rig.h(), k_hop);
    driver.step(3);
    ASSERT_TRUE(driver.wait());
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(variant, 1U);
    EXPECT_EQ(rig.h()->m_pipeline.m_candidates.size(), candidates)
        << "an entry of an engine the set names added a candidate";
    ASSERT_EQ(rig.add(four, &variant), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(variant, 2U);
    EXPECT_EQ(rig.h()->m_pipeline.m_candidates.size(), candidates + 1)
        << "the default set did not grow by the new variant's engine";
    EXPECT_EQ(rig.m_second.m_loads.load(), 1);

    const std::vector<anira_plan_info> rows = rows_of(anira_handler_plan_report(rig.h()));
    ASSERT_EQ(rows.size(), 3U);
    ASSERT_NE(rows[2].engine_id, nullptr);
    EXPECT_STREQ(rows[2].engine_id, k_second_engine);

    // The same variant on the same backend, then the plan on the other engine by its index
    // (anira_handler_set_model keeps the backend, and the third variant has none on it).
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    driver.step(3);
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(anira_handler_set_model(rig.h(), 2), ANIRA_ERROR_NOT_FOUND);
    ASSERT_EQ(anira_handler_set_plan(rig.h(), 2), ANIRA_OK);
    driver.step(3);
    ASSERT_TRUE(driver.wait());

    ASSERT_EQ(recorder.m_count, 9U);
    const std::array<uint32_t, 9> variants{0, 0, 0, 1, 1, 1, 2, 2, 2};
    const std::array<uint8_t, 9> engines{1, 1, 1, 1, 1, 1, 2, 2, 2};
    for (size_t chunk = 0; chunk < recorder.m_count; ++chunk) {
        EXPECT_EQ(recorder.m_variant[chunk], variants.at(chunk)) << "chunk " << chunk;
        EXPECT_EQ(recorder.m_engine[chunk], engines.at(chunk)) << "chunk " << chunk;
    }
    EXPECT_EQ(rig.m_engine.calls_of(2.0F), static_cast<int>(k_warmup) + 3);
    EXPECT_EQ(rig.m_second.calls_of(4.0F), static_cast<int>(k_warmup) + 3);
}

// A prepare behind an add builds the handler from its pipeline copy, which has the variant now:
// one session over every variant, the report of the prepare the handler's only one again, at
// the address the prepare's report always had.
TEST(VariantSet, APrepareBehindAnAddBuildsEveryVariant) {
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const anira_plan_report* const first = anira_handler_plan_report(rig.h());
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rig.h()->m_reports.size(), 2U);
    EXPECT_NE(anira_handler_plan_report(rig.h()), first);
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);

    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(rig.h()->m_reports.size(), 1U) << "a prepare keeps a report of an earlier add";
    EXPECT_EQ(anira_handler_plan_report(rig.h()), first);
    const std::vector<anira_plan_info> rows = rows_of(first);
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].variant, 0U);
    EXPECT_EQ(rows[1].variant, 1U);
    EXPECT_EQ(anira_handler_get_plan(rig.h()), 0U) << "a prepare selects the first variant's plan";
    EXPECT_EQ(anira_test::session_of(rig.h())->m_plans.size(), 2U);
    EXPECT_EQ(rig.m_engine.m_loads.load() - rig.m_engine.m_unloads.load(), 2);
    EXPECT_EQ(rig.m_engine.m_prepares.load() - rig.m_engine.m_unprepares.load(), 2);

    BackgroundDriver driver(rig.h(), k_hop);
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    driver.step(4);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [](size_t /*chunk*/) { return 2.0F; });
}

// ==== a failed add ============================================================================

// An add that fails leaves the handler as it was, whatever step refused: a model of another
// form (before anything loads), a model that does not load, a variant no candidate runs, and a
// plan whose engine binds on a coarser boundary than the playing stream's chunks have. Nothing
// is published, the pipeline copy and the candidate set are as before, what the call loaded is
// unloaded and what it prepared unprepared, and the stream plays through all of it.
TEST(VariantSet, AFailedAddLeavesTheHandlerAsItWas) {
    const anira::ModelConfig one = gain_model("1");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    BackgroundDriver driver(rig.h(), k_hop);
    driver.run_free(1U << 20);
    ASSERT_TRUE(wait_for([&driver] { return driver.done() >= 4; }));
    const HandlerState before(rig);
    uint32_t variant = 99;

    // Another form: the hop.
    const anira::ModelConfig other_hop = gain_model("2", static_cast<int64_t>(k_hop) / 2);
    EXPECT_EQ(rig.add(other_hop, &variant), ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message, "variants[1]: input slot 0 'in'");
    expect_contains(rig.m_message, "every variant of an inference stage has the form of the");
    expect_unchanged(rig, before);

    // A model its engine does not load: the engine's own status, and no unload is owed.
    const anira::ModelConfig failing = gain_model(k_fail_path);
    EXPECT_EQ(rig.add(failing, &variant), ANIRA_ERROR_MODEL_LOAD);
    expect_unchanged(rig, before);

    // An entry of an engine the pipeline does not have.
    const anira::ModelConfig absent = gain_model("2", static_cast<int64_t>(k_hop), k_absent_engine);
    EXPECT_EQ(rig.add(absent, &variant), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(rig.m_message, "is not added to this pipeline");
    expect_unchanged(rig, before);

    // A plan whose alignment the chunks of the playing stream do not have. Every engine of
    // this build binds on the default boundary, so the test lowers what the session says its
    // chunks were built with (white-box; the pool itself is untouched): the model loads, the
    // add is refused naming both figures, and the model is unloaded again.
    const std::shared_ptr<SessionElement> session = anira_test::session_of(rig.h());
    const size_t alignment = session->m_chunk_alignment;
    ASSERT_GE(alignment, 64U);
    session->m_chunk_alignment = 32;
    const anira::ModelConfig two = gain_model("2");
    const int loads = rig.m_engine.m_loads.load();
    EXPECT_EQ(rig.add(two, &variant), ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message, "variants[1]: models[0] on");
    expect_contains(rig.m_message, std::to_string(alignment) + "-byte boundary");
    expect_contains(rig.m_message, "32-byte one");
    expect_contains(rig.m_message, "prepare");
    EXPECT_EQ(rig.m_engine.m_loads.load(), loads + 1) << "the model loaded before the refusal";
    session->m_chunk_alignment = alignment;
    expect_unchanged(rig, before);

    // A model whose inference fails: its pass, the last step that can refuse, fails the add
    // behind the load and the engine's prepare, which are taken back; the failure is the
    // build's session's, not the playing stream's (rt_error stays what it was).
    const anira::ModelConfig broken = gain_model("7");
    rig.m_engine.m_failing.at(7).store(true);
    const int prepares = rig.m_engine.m_prepares.load();
    EXPECT_EQ(rig.add(broken, &variant), ANIRA_ERROR_ENGINE);
    expect_contains(rig.m_message, "variants[1]: models[0] on ");
    expect_contains(rig.m_message, "the warm-up inference of plan 0 failed");
    EXPECT_EQ(rig.m_engine.m_prepares.load(), prepares + 1) << "the engine was prepared first";
    expect_unchanged(rig, before);

    EXPECT_EQ(variant, 99U) << "a failed add wrote the index";
    const size_t done = driver.done();
    ASSERT_TRUE(wait_for([&] { return driver.done() >= done + 4; }));
    driver.stop();
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [](size_t /*chunk*/) { return 1.0F; });

    // The same model is added behind the refusals, at the index they did not take.
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
}

// Under named candidates the set is the pipeline's and does not grow: a variant whose entries
// run on none of them has no plan and is refused, and one that runs on a candidate is added.
TEST(VariantSet, AVariantNoNamedCandidateRunsIsRefused) {
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig elsewhere =
        gain_model("2", static_cast<int64_t>(k_hop), k_second_engine);
    const anira::ModelConfig two = gain_model("2");
    const std::vector<anira_backend_id> named{{.struct_size = sizeof(anira_backend_id),
                                               .engine = ANIRA_ENGINE_CUSTOM,
                                               .provider = ANIRA_PROVIDER_CPU,
                                               .engine_id = k_engine,
                                               .provider_id = nullptr}};
    SetRig rig({&one}, nullptr, ANIRA_LOG_ERROR, named);
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const HandlerState before(rig);
    uint32_t variant = 99;
    EXPECT_EQ(rig.add(elsewhere, &variant), ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message, "variants[1]: none of the 1 model entries runs on a candidate");
    expect_unchanged(rig, before);
    EXPECT_EQ(rig.m_second.m_loads.load(), 0);
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
    EXPECT_EQ(rig.h()->m_pipeline.m_candidates.size(), before.m_candidates);
}

// The arguments: a NULL handler, model or index is ANIRA_ERROR_INVALID_ARGUMENT.
TEST(VariantSet, AnAddChecksItsArguments) {
    const anira::ModelConfig one = gain_model("1");
    const SetRig rig({&one});
    uint32_t variant = 0;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_handler_add_variant(nullptr, one.native(), &variant, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_add_variant(rig.h(), nullptr, &variant, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_add_variant(rig.h(), one.native(), nullptr, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_add_variant(rig.h(), one.native(), &variant, nullptr), ANIRA_OK)
        << "a NULL err is legal";
    EXPECT_EQ(variant, 1U);
}

// ==== an unprepared handler ===================================================================

// On an unprepared handler an add validates the model as the create did and appends it to the
// pipeline copy: nothing loads, a model of another form is refused and leaves the copy alone,
// and the next prepare builds the variant like one the pipeline brought.
TEST(VariantSet, AVariantAddedToAnUnpreparedHandlerIsBuiltByTheNextPrepare) {
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    SetRig rig({&one});
    const HandlerState before(rig);

    uint32_t variant = 99;
    const anira::ModelConfig other_hop = gain_model("3", static_cast<int64_t>(k_hop) / 2);
    EXPECT_EQ(rig.add(other_hop, &variant), ANIRA_ERROR_CONFIG);
    expect_contains(rig.m_message, "variants[1]: input slot 0 'in'");
    expect_unchanged(rig, before);
    EXPECT_EQ(variant, 99U);

    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
    EXPECT_EQ(rig.m_engine.m_loads.load(), 0) << "an unprepared handler loads nothing";
    EXPECT_EQ(anira_handler_plan_report(rig.h()), nullptr);
    EXPECT_EQ(rig.h()->m_pipeline.m_variants.size(), 2U);

    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const std::vector<anira_plan_info> rows = rows_of(anira_handler_plan_report(rig.h()));
    ASSERT_EQ(rows.size(), 2U);
    EXPECT_EQ(rows[0].variant, 0U);
    EXPECT_EQ(rows[1].variant, 1U);
    EXPECT_EQ(rig.m_engine.m_loads.load(), 2);
    EXPECT_EQ(rig.m_engine.m_prepare_flags.load() & ANIRA_PREPARE_LIVE, 0U)
        << "a plain prepare is no live build";

    BackgroundDriver driver(rig.h(), k_hop);
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    driver.step(6);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [](size_t /*chunk*/) { return 2.0F; });
}

// ==== the state of a playing exclusive handler ================================================

// An exclusive handler (a declared State pair) keeps the model's state between its inferences
// in buffers of its own, and its inferences run one at a time. An add while it plays builds
// and measures its variant on other buffers: the state stream of the playing variant is, block
// by block, the one of a twin handler that never adds, and the closed form (the state grows by
// the model's factor per inference). The new variant then starts from its initial state, as
// the contract's default says.
TEST(VariantSet, AnAddBesideAPlayingExclusiveHandlerLeavesItsStateAlone) {
    // The state grows by one per block, and a block times its state must stay exact: a
    // bounded run, not a free one.
    constexpr size_t k_blocks = 200;
    const anira::ModelConfig one = state_model("1");
    const anira::ModelConfig three = state_model("3");

    SetRig rig({&one});
    rig.m_engine.m_stateful = true;
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    SetRig twin({&one});
    twin.m_engine.m_stateful = true;
    ASSERT_EQ(twin.prepare(set_contract()), ANIRA_OK) << twin.m_message;

    BackgroundDriver driver(rig.h(), k_hop);
    BackgroundDriver twin_driver(twin.h(), k_hop);
    twin_driver.step(k_blocks);
    driver.step(k_blocks);
    ASSERT_TRUE(wait_for([&driver] { return driver.done() >= 8; }));
    // The add, while the handler's driving thread runs the rest of its blocks.
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(three, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
    ASSERT_TRUE(driver.wait());
    ASSERT_TRUE(twin_driver.wait());

    // Chunk c is inference c + 1 of the playing variant: the state after it is c + 1.
    expect_stream(driver, [](size_t chunk) { return static_cast<float>(chunk + 1); });
    expect_stream(twin_driver, [](size_t chunk) { return static_cast<float>(chunk + 1); });
    EXPECT_EQ(driver.factors(), twin_driver.factors());

    // The new variant starts fresh (ANIRA_VARIANT_STATE_FRESH): its first inference reads
    // zeros, whatever its pass left in the build's own buffers.
    ASSERT_EQ(anira_handler_set_model(rig.h(), 1), ANIRA_OK);
    driver.step(4);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [](size_t chunk) {
        return chunk < k_blocks ? static_cast<float>(chunk + 1)
                                : 3.0F * static_cast<float>(chunk - k_blocks + 1);
    });
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK);
}

// The engine of an added plan starts its life in the stream as the engine of a plan does after
// a prepare: its first inference there is the first of a new stream, so an exclusive handler's
// engine is reset ahead of it, whatever its pass in the build's session left in it, also when
// the contract keeps the declared state across a variant switch (ANIRA_VARIANT_STATE_KEEP: the
// pair continues from what the variant before left, the engine's own state starts over). The
// first handler's prepare leaves the playing model's figure in the core, so the second
// handler's prepare sizes its session once, as the build's session of the add is sized once:
// the two count the same generations, and only the reset rule tells them apart.
TEST(VariantSet, TheEngineOfAnAddedPlanIsResetAtItsFirstInferenceInTheStream) {
    ASSERT_TRUE(anira_test::fresh_core());
    constexpr size_t k_blocks = 6;
    const anira::ModelConfig one = state_model("1");
    const anira::ModelConfig three = state_model("3");
    anira::Hard hard{.host_block_min = static_cast<double>(k_hop),
                     .host_block_max = static_cast<double>(k_hop),
                     .host_rate = k_rate,
                     .budget = std::chrono::duration<double, std::milli>(k_budget_ms)};
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = k_warmup;
    hard.on_miss = ANIRA_MISS_ZEROS;
    hard.variant_state = ANIRA_VARIANT_STATE_KEEP;
    const anira::ContractHandle contract(hard);

    SetRig measured({&one});
    measured.m_engine.m_stateful = true;
    ASSERT_EQ(measured.prepare(contract), ANIRA_OK) << measured.m_message;

    SetRig rig({&one});
    rig.m_engine.m_stateful = true;
    ASSERT_EQ(rig.prepare(contract), ANIRA_OK) << rig.m_message;
    ASSERT_EQ(rows_of(anira_handler_plan_report(rig.h())).at(0).warmup_outcome,
              static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    ASSERT_EQ(rig.m_engine.resets_of(1.0F), 0) << "a prepare without a pass runs no inference";

    BackgroundDriver driver(rig.h(), k_hop);
    driver.step(k_blocks);
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(rig.m_engine.resets_of(1.0F), 1) << "the stream's first inference";

    uint32_t variant = 0;
    ASSERT_EQ(rig.add(three, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(rig.m_engine.calls_of(3.0F), static_cast<int>(k_warmup));
    EXPECT_EQ(rig.m_engine.resets_of(3.0F), 1) << "the first inference of the build's pass";

    ASSERT_EQ(anira_handler_set_model(rig.h(), variant), ANIRA_OK);
    driver.step(k_blocks);
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(rig.m_engine.resets_of(3.0F), 2)
        << "the added plan's engine carried its pass into the stream";
    EXPECT_EQ(rig.m_engine.resets_of(1.0F), 1);
    // The declared state is kept across the switch: the new variant adds its factor to the
    // state the first one left.
    expect_stream(driver, [](size_t chunk) {
        return chunk < k_blocks ? static_cast<float>(chunk + 1)
                                : static_cast<float>(k_blocks) +
                                      (3.0F * static_cast<float>(chunk - k_blocks + 1));
    });
}

// ==== the warnings ============================================================================

// Two differences of an added variant are no refusal: an internal latency other than the one
// the handler reports (the latency does not move; the Warning says which way the variant plays
// against it), and an inference time above the budget the stream is sized for.
TEST(VariantSet, AnAddedVariantThatDiffersInItsLatencyOrItsSpeedOnlyWarns) {
    ASSERT_TRUE(anira_test::fresh_core());
    constexpr int64_t k_internal = 16;
    const anira::ModelConfig one = gain_model("1");
    anira::ModelConfig late;
    late.add_model_path(k_engine, "2");
    late.input(anira_test::stream_tensor("in", 1, k_hop, k_hop));
    late.output(anira_test::stream_tensor("out", 1, k_hop, k_hop).latency(k_internal));
    late.max_instances(2);
    const anira::ModelConfig slow = gain_model("3");

    anira_drain_log();
    RecordCollector collector;
    SetRig rig({&one}, nullptr, ANIRA_LOG_WARNING);
    // The third model takes five times the budget the stream is sized for.
    rig.m_engine.m_sleep_us.at(3).store(static_cast<int>(k_budget_ms * 5000.0));
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    const uint32_t latency = anira_handler_get_latency(rig.h(), 0);

    uint32_t variant = 0;
    ASSERT_EQ(rig.add(late, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 1U);
    ASSERT_EQ(rig.add(slow, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(variant, 2U);
    anira_drain_log();
    EXPECT_EQ(anira_handler_get_latency(rig.h(), 0), latency) << "an add moved the latency";
#ifdef ENABLE_LOGGING
    const char* latency_warning =
        "anira_handler_add_variant: variants[1] differs from the handler in its internal latency "
        "(output slot 0 'out': 16 samples, the handler reports 0, so the variant plays 16 samples "
        "late against the reported latency); the latency does not change in this call";
    EXPECT_EQ(anira_test::count_records(collector, latency_warning, "native"), 1U);
    EXPECT_EQ(anira_test::find_record(collector, latency_warning, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_WARNING));
    const char* speed_warning = "ms per inference, the stream is sized for 1.000 ms; expect misses";
    EXPECT_EQ(anira_test::count_records(collector, speed_warning, "native"), 1U);
    expect_contains(anira_test::find_record(collector, speed_warning, "native").m_message,
                    "anira_handler_add_variant: variants[2]: plan 2 (models[0] on ");
    EXPECT_EQ(anira_test::count_records(collector, "differs from the handler", "native"), 1U)
        << "a variant of the reported latency warned";
#endif
    const std::vector<anira_plan_info> rows = rows_of(anira_handler_plan_report(rig.h()));
    ASSERT_EQ(rows.size(), 3U);
    EXPECT_GE(rows[2].measured_ms, k_budget_ms * 5.0);
}

// ==== the control lock ========================================================================

// What an engine's prepare may call on the handler it is prepared for. Inside a plain prepare no
// stream runs: the getters answer and the two Static entries work, as ever, and a control entry
// is refused (it would run inside the prepare that called it). Inside a live build the stream
// plays, and its driving thread is the one writer of a Static store: the Static entries are
// refused too, and nothing is stored. A destroy from either callback does nothing.
TEST(VariantSet, ALiveBuildRefusesControlAndStaticEntriesFromItsCallbacks) {
    const anira::ModelConfig one = static_model("1");
    const anira::ModelConfig two = static_model("2");
    SetRig rig({&one});
    rig.m_engine.m_reads_static = true;
    const anira::ContractHandle contract = set_contract();

    struct Seen {
        anira_status m_prepare = ANIRA_OK;
        anira_status m_add = ANIRA_OK;
        anira_status m_set_static = ANIRA_OK;
        anira_status m_get_static = ANIRA_OK;
        uint32_t m_entries = 0;
        uint32_t m_report_rows = 0;
        float m_peak = -1.0F;
        int m_calls = 0;
    };
    Seen seen;
    float written = 0.0F;
    const std::array<int64_t, 1> shape{1};
    rig.m_engine.m_on_prepare = [&](const anira_prepare_info& info) {
        ++seen.m_calls;
        anira_handler* const handler = info.handler;
        seen.m_prepare = anira_handler_prepare(handler, contract.native(), nullptr);
        uint32_t index = 0;
        seen.m_add = anira_handler_add_variant(handler, two.native(), &index, nullptr);
        anira_handler_destroy(handler);
        const anira_tensor gain = anira_test::whole_f32(&written, shape);
        seen.m_set_static = anira_handler_set_static_input(handler, 1, &gain);
        const anira_tensor peak = anira_test::whole_f32(&seen.m_peak, shape);
        seen.m_get_static = anira_handler_get_static_output(handler, 1, &peak);
        seen.m_entries = anira_handler_num_entries(handler);
        seen.m_report_rows = anira_plan_report_num_plans(anira_handler_plan_report(handler));
    };

    // A plain prepare: the control entries are refused, the Static entries work.
    written = 5.0F;
    ASSERT_EQ(rig.prepare(contract), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(seen.m_calls, 1);
    EXPECT_EQ(seen.m_prepare, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_add, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_set_static, ANIRA_OK);
    EXPECT_EQ(seen.m_get_static, ANIRA_OK);
    EXPECT_EQ(seen.m_peak, 0.0F) << "the store's value: zeros before the first capture";
    EXPECT_GT(seen.m_entries, 0U);
    EXPECT_EQ(seen.m_report_rows, 1U);
    EXPECT_EQ(rig.h()->m_pipeline.m_variants.size(), 1U) << "the refused add left a variant";

    // The stream sees the value the prepare's callback stored.
    BackgroundDriver driver(rig.h(), k_hop);
    driver.step(4);
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(rig.m_engine.m_static_seen.load(), 5.0F);
    driver.run_free(1U << 20);

    // A live build: every one of them is refused, the getters answer for the playing handler
    // (the report it had: the build's is not published yet), and nothing was stored.
    seen = Seen{};
    written = 9.0F;
    uint32_t variant = 0;
    ASSERT_EQ(rig.add(two, &variant), ANIRA_OK) << rig.m_message;
    EXPECT_EQ(seen.m_calls, 1);
    EXPECT_EQ(seen.m_prepare, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_add, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_set_static, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_get_static, ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(seen.m_peak, -1.0F) << "a refused read wrote the caller's memory";
    EXPECT_EQ(seen.m_entries, anira_handler_num_entries(rig.h()));
    EXPECT_EQ(seen.m_report_rows, 1U);
    EXPECT_EQ(anira_handler_rt_error(rig.h()), ANIRA_OK)
        << "the refusal of a callback is not the stream's failure";
    EXPECT_EQ(rig.h()->m_pipeline.m_variants.size(), 2U);

    const size_t done = driver.done();
    ASSERT_TRUE(wait_for([&] { return driver.done() >= done + 4; }));
    driver.stop();
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(rig.m_engine.m_static_seen.load(), 5.0F) << "a live build's callback stored a value";
    expect_stream(driver, [](size_t /*chunk*/) { return 1.0F; });

    // Behind the build the lock is free: the Static entries work from this thread again.
    rig.m_engine.m_on_prepare = nullptr;
    const anira_tensor gain = anira_test::whole_f32(&written, shape);
    EXPECT_EQ(anira_handler_set_static_input(rig.h(), 1, &gain), ANIRA_OK);
}

// A live build holds the handler's control lock like a prepare: a second control entry on
// another thread waits for it, and both land. Two threads add a variant each to one playing
// handler; the indices are the two next ones, each given out once.
TEST(VariantSet, TwoAddsOfOneHandlerTakeTurns) {
    const anira::ModelConfig one = gain_model("1");
    const anira::ModelConfig two = gain_model("2");
    const anira::ModelConfig three = gain_model("3");
    SetRig rig({&one});
    ASSERT_EQ(rig.prepare(set_contract()), ANIRA_OK) << rig.m_message;
    BackgroundDriver driver(rig.h(), k_hop);
    driver.run_free(1U << 20);

    std::array<uint32_t, 2> variants{0, 0};
    std::array<anira_status, 2> statuses{ANIRA_ERROR_INTERNAL, ANIRA_ERROR_INTERNAL};
    std::thread first([&] {
        statuses[0] = anira_handler_add_variant(rig.h(), two.native(), &variants[0], nullptr);
    });
    std::thread second([&] {
        statuses[1] = anira_handler_add_variant(rig.h(), three.native(), &variants[1], nullptr);
    });
    first.join();
    second.join();
    EXPECT_EQ(statuses[0], ANIRA_OK);
    EXPECT_EQ(statuses[1], ANIRA_OK);
    EXPECT_EQ(std::min(variants[0], variants[1]), 1U);
    EXPECT_EQ(std::max(variants[0], variants[1]), 2U);
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(rig.h())), 3U);

    // Each index names the model its call added.
    driver.stop();
    ASSERT_TRUE(driver.wait());
    const size_t first_switch = driver.done();
    ASSERT_EQ(anira_handler_set_model(rig.h(), variants[0]), ANIRA_OK);
    driver.step(4);
    ASSERT_TRUE(driver.wait());
    const size_t second_switch = driver.done();
    ASSERT_EQ(anira_handler_set_model(rig.h(), variants[1]), ANIRA_OK);
    driver.step(4);
    ASSERT_TRUE(driver.wait());
    expect_stream(driver, [=](size_t chunk) {
        if (chunk < first_switch) { return 1.0F; }
        return chunk < second_switch ? 2.0F : 3.0F;
    });
}
