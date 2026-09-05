// anira/abi/handler.h: the C twins of test_InferenceHandlerApi.cpp, with the still-public 2.x
// anira::InferenceHandler as the in-binary oracle. Both handlers drive the same bundled model
// (gain on the engine-free custom plan on every leg; the CNN where an engine is present) block
// by block, each waiting for its own inference, so no block is ever missed and the outputs are
// compared bit for bit.
#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/inference_config_eq.h"
#include "../support/log_record_collector.h"
#include "handler_support.h"

namespace {

using anira_test::attach_processor;
using anira_test::Context;
using anira_test::custom_candidates;
using anira_test::engine_candidates;
using anira_test::expect_all;
using anira_test::expect_same_block;
using anira_test::explicit_contract;
using anira_test::file_contract;
using anira_test::gain_with_custom;
using anira_test::Handler;
using anira_test::k_block;
using anira_test::k_custom;
using anira_test::k_rate;
using anira_test::ramp;
using anira_test::RecordCollector;
using anira_test::wait_for_block;
using anira_test::wait_for_push;

/// The call form both sides are driven through.
enum class Form { InPlace, Separate, Multi, PushPop, PushPopMulti };

constexpr Form k_forms[] = {Form::InPlace,
                            Form::Separate,
                            Form::Multi,
                            Form::PushPop,
                            Form::PushPopMulti};

/// The 2.x side's core config: the same threads, wait strategy and drain as the Context
/// default, so the two users reconcile without a mismatch warning.
anira::CoreConfig oracle_core_config() {
    anira::CoreConfig config(2, anira::WaitStrategy::SpinBackoff, anira::LogLevel::Error);
    config.m_log.m_drain = anira::LogDrain::Manual;
    return config;
}

/// The plan rows of a prepared handler.
std::vector<anira_plan_info> plans_of(const anira_handler* handler) {
    const anira_plan_report* report = anira_handler_plan_report(handler);
    EXPECT_NE(report, nullptr);
    uint32_t count = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> rows(count);
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
              ANIRA_OK);
    return rows;
}

/// The 2.x backend a plan row runs on.
anira::InferenceBackend backend_of(const anira_plan_info& info) {
    // The 2.x enumerators exist only for the engines this build carries.
    switch (info.engine) {
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME: return anira::InferenceBackend::ONNX;
#endif
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH: return anira::InferenceBackend::LIBTORCH;
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return anira::InferenceBackend::LITERT;
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH: return anira::InferenceBackend::EXECUTORCH;
#endif
        default: break;
    }
    EXPECT_EQ(info.engine, ANIRA_ENGINE_NONE);
    EXPECT_NE(info.engine_id, nullptr);
    if (info.engine_id != nullptr) { EXPECT_STREQ(info.engine_id, k_custom); }
    return anira::InferenceBackend::CUSTOM;
}

/// An engine this build does not carry, if there is one.
std::optional<anira_engine> missing_engine() {
    const std::vector<anira::BackendId> enabled = anira::enabled_backends();
    for (anira_engine engine : {ANIRA_ENGINE_ONNXRUNTIME,
                                ANIRA_ENGINE_LIBTORCH,
                                ANIRA_ENGINE_TFLITE,
                                ANIRA_ENGINE_LITERT,
                                ANIRA_ENGINE_EXECUTORCH}) {
        bool found = false;
        for (const anira::BackendId& id : enabled) {
            if (id.engine == static_cast<uint32_t>(engine)) { found = true; }
        }
        if (!found) { return engine; }
    }
    return std::nullopt;
}

/// One C handler and one 2.x handler over the same model, driven in lockstep.
struct Oracle {
    Oracle(const Context& context,
           const anira::ModelConfig& model,
           std::span<const anira_backend_id> candidates,
           anira::InferenceConfig config_2x)
        : m_c(context, model, candidates)
        , m_config_2x(std::move(config_2x))
        , m_pp(m_config_2x)
        , m_v2(m_pp, m_config_2x, oracle_core_config()) {}

    anira_handler* h() { return m_c.m_handler; }

    /// The C side takes the contract, the 2.x side the host geometry; as the precondition of
    /// the pooling claim the two InferenceConfigs must compare equal.
    void prepare(const anira::ContractHandle& contract, size_t block) {
        ASSERT_EQ(m_c.prepare(contract), ANIRA_OK) << m_c.m_err.message;
        m_v2.prepare(anira::HostConfig(static_cast<float>(block), static_cast<float>(k_rate)));
        anira_test::expect_inference_config_eq(m_c.m_handler->m_inference_config, m_config_2x);
        m_num_inputs = m_c.m_handler->m_num_inputs;
        m_num_outputs = m_c.m_handler->m_num_outputs;
    }

    /// One block on both sides; c and v receive the outputs, n_c and n_v the counts. The
    /// static gain slot travels through the multi forms with one value, 1.0F.
    void run_block(size_t k,
                   size_t block,
                   Form form,
                   std::vector<float>& c,
                   std::vector<float>& v,
                   size_t& n_c,
                   size_t& n_v) {
        const std::vector<float> in = ramp(k, block);
        c = in;
        v = in;
        std::vector<float> c_out(block, -1.0F);
        std::vector<float> v_out(block, -1.0F);
        float gain_in_c = 1.0F;
        float gain_in_v = 1.0F;
        float gain_out_c = -1.0F;
        float gain_out_v = -1.0F;

        const size_t prev_c = anira_handler_get_available_samples(h(), 0, 0);
        const size_t prev_v = m_v2.get_available_samples(0);

        switch (form) {
            case Form::InPlace: {
                float* c_ch[1] = {c.data()};
                float* v_ch[1] = {v.data()};
                n_c = anira_handler_process(h(), c_ch, block, 0);
                wait_for_block(h(), prev_c);
                n_v = m_v2.process(v_ch, block);
                wait_for_block(m_v2, prev_v);
                break;
            }
            case Form::Separate: {
                const float* c_in[1] = {c.data()};
                float* c_ch[1] = {c_out.data()};
                const float* v_in[1] = {v.data()};
                float* v_ch[1] = {v_out.data()};
                n_c = anira_handler_process_separate(h(), c_in, block, c_ch, block, 0);
                wait_for_block(h(), prev_c);
                n_v = m_v2.process(v_in, block, v_ch, block);
                wait_for_block(m_v2, prev_v);
                c = c_out;
                v = v_out;
                break;
            }
            case Form::Multi: {
                const float* c_in_ch[1] = {c.data()};
                const float* c_gain_ch[1] = {&gain_in_c};
                const float* const* c_in[2] = {c_in_ch, c_gain_ch};
                size_t c_num_in[2] = {block, 1};
                float* c_out_ch[1] = {c_out.data()};
                float* c_gout_ch[1] = {&gain_out_c};
                float* const* c_outs[2] = {c_out_ch, c_gout_ch};
                size_t c_num_out[2] = {block, 1};
                EXPECT_EQ(anira_handler_process_multi(h(), c_in, c_num_in, c_outs, c_num_out),
                          ANIRA_OK);
                n_c = c_num_out[0];
                wait_for_block(h(), prev_c);

                const float* v_in_ch[1] = {v.data()};
                const float* v_gain_ch[1] = {&gain_in_v};
                const float* const* v_in[2] = {v_in_ch, v_gain_ch};
                size_t v_num_in[2] = {block, 1};
                float* v_out_ch[1] = {v_out.data()};
                float* v_gout_ch[1] = {&gain_out_v};
                float* const* v_outs[2] = {v_out_ch, v_gout_ch};
                size_t v_num_out[2] = {block, 1};
                n_v = m_v2.process(v_in, v_num_in, v_outs, v_num_out)[0];
                wait_for_block(m_v2, prev_v);
                if (m_num_outputs > 1) {
                    EXPECT_EQ(c_num_out[1], 1U) << "block " << k;
                    EXPECT_EQ(v_num_out[1], 1U) << "block " << k;
                    EXPECT_EQ(gain_out_c, gain_out_v) << "block " << k;
                }
                c = c_out;
                v = v_out;
                break;
            }
            case Form::PushPop: {
                const float* c_in[1] = {c.data()};
                float* c_ch[1] = {c_out.data()};
                EXPECT_EQ(anira_handler_push_data(h(), c_in, block, 0), ANIRA_OK);
                wait_for_push(h(), prev_c, block);
                n_c = anira_handler_pop_data(h(), c_ch, block, 0);
                const float* v_in[1] = {v.data()};
                float* v_ch[1] = {v_out.data()};
                m_v2.push_data(v_in, block);
                wait_for_push(m_v2, prev_v, block);
                n_v = m_v2.pop_data(v_ch, block);
                c = c_out;
                v = v_out;
                break;
            }
            case Form::PushPopMulti: {
                const float* c_in_ch[1] = {c.data()};
                const float* c_gain_ch[1] = {&gain_in_c};
                const float* const* c_in[2] = {c_in_ch, c_gain_ch};
                size_t c_num_in[2] = {block, 1};
                float* c_out_ch[1] = {c_out.data()};
                float* c_gout_ch[1] = {&gain_out_c};
                float* const* c_outs[2] = {c_out_ch, c_gout_ch};
                size_t c_num_out[2] = {block, 1};
                EXPECT_EQ(anira_handler_push_data_multi(h(), c_in, c_num_in), ANIRA_OK);
                wait_for_push(h(), prev_c, block);
                EXPECT_EQ(anira_handler_pop_data_multi(h(), c_outs, c_num_out), ANIRA_OK);
                n_c = c_num_out[0];

                const float* v_in_ch[1] = {v.data()};
                const float* v_gain_ch[1] = {&gain_in_v};
                const float* const* v_in[2] = {v_in_ch, v_gain_ch};
                size_t v_num_in[2] = {block, 1};
                float* v_out_ch[1] = {v_out.data()};
                float* v_gout_ch[1] = {&gain_out_v};
                float* const* v_outs[2] = {v_out_ch, v_gout_ch};
                size_t v_num_out[2] = {block, 1};
                m_v2.push_data(v_in, v_num_in);
                wait_for_push(m_v2, prev_v, block);
                n_v = m_v2.pop_data(v_outs, v_num_out)[0];
                if (m_num_outputs > 1) { EXPECT_EQ(gain_out_c, gain_out_v) << "block " << k; }
                c = c_out;
                v = v_out;
                break;
            }
        }
    }

    /// `blocks` blocks in lockstep; every block is delivered in full on both sides and the
    /// outputs are bit-equal.
    void run_blocks(size_t blocks, size_t block, Form form) {
        for (size_t i = 0; i < blocks; ++i) {
            const size_t k = m_next_block++;
            std::vector<float> c;
            std::vector<float> v;
            size_t n_c = 0;
            size_t n_v = 0;
            run_block(k, block, form, c, v, n_c, n_v);
            if (::testing::Test::HasFatalFailure()) { return; }
            EXPECT_EQ(n_c, n_v) << "block " << k;
            EXPECT_EQ(n_c, block) << "block " << k;
            expect_same_block(c, v, k);
        }
    }

    void reset() {
        anira_handler_reset(h());
        m_v2.reset();
    }

    Handler m_c;
    anira::InferenceConfig m_config_2x;
    anira::PrePostProcessor m_pp;
    anira::InferenceHandler m_v2;
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    size_t m_next_block = 1;
};

/// A prepared gain oracle on the custom plan.
struct GainOracle : Oracle {
    explicit GainOracle(const Context& context, bool default_custom = true)
        : Oracle(context,
                 gain_with_custom(default_custom),
                 custom_candidates(),
                 anira_test::bridged_2x(k_gain_model_json, k_gain_contract_json, true)) {}
};

}  // namespace

// ============================================================================================
// Arguments and states
// ============================================================================================

TEST(AbiHandler, NullArgumentsAreRefused) {
    const Context context;
    anira_drain_log();  // leftovers of earlier tests go nowhere
    RecordCollector collector;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_pipeline_create(nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "out"), nullptr) << err.message;

    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    const anira::ModelConfig model = gain_with_custom();
    const anira_model_config* variants[] = {model.native()};
    EXPECT_EQ(anira_pipeline_add_inference(nullptr, variants, 1, nullptr, 0, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants, 1, nullptr, 0, &err), ANIRA_OK)
        << err.message;

    anira_handler* handler = nullptr;
    EXPECT_EQ(anira_handler_create(nullptr, pipeline, &handler, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_create(context.m_context, nullptr, &handler, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_create(context.m_context, pipeline, nullptr, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(handler, nullptr);

    std::vector<float> block(k_block, 0.0F);
    float* ptrs[1] = {block.data()};
    const float* in_ptrs[1] = {block.data()};
    const float* const* in_multi[1] = {in_ptrs};
    float* const* out_multi[1] = {ptrs};
    size_t num_in[1] = {k_block};
    size_t num_out[1] = {k_block};
    uint32_t count = 0;
    EXPECT_EQ(anira_handler_process(nullptr, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_push_data(nullptr, in_ptrs, k_block, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data(nullptr, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_process_multi(nullptr, in_multi, num_in, out_multi, num_out),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_get_latency(nullptr, 0), 0U);
    EXPECT_EQ(anira_handler_get_latencies(nullptr, &count, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_get_available_samples(nullptr, 0, 0), 0U);
    EXPECT_EQ(anira_handler_get_plan(nullptr), 0U);
    EXPECT_EQ(anira_handler_set_plan(nullptr, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(nullptr), ANIRA_OK);
    EXPECT_EQ(anira_handler_plan_report(nullptr), nullptr);
    EXPECT_EQ(anira_plan_report_num_plans(nullptr), 0U);
    EXPECT_EQ(anira_plan_report_plans(nullptr, sizeof(anira_plan_info), &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    anira_handler_reset(nullptr);
    anira_handler_destroy(nullptr);
    anira_pipeline_destroy(nullptr);
    anira_pipeline_destroy(pipeline);

    // Nothing was recorded anywhere: there was no handler to record on.
    anira_drain_log();
    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const auto& record : collector.m_records) {
        EXPECT_NE(record.m_group, "anira.capi") << record.m_message;
    }
}

TEST(AbiHandler, UnpreparedEntriesRecordNotPrepared) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);

    std::vector<float> block(k_block, 0.25F);
    float* ptrs[1] = {block.data()};
    const float* in_ptrs[1] = {block.data()};
    void* typed_ptrs[1] = {block.data()};
    const float* const* in_multi[1] = {in_ptrs};
    float* const* out_multi[1] = {ptrs};
    size_t num_in[1] = {k_block};
    size_t num_out[1] = {k_block};

    EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data(h, in_ptrs, k_block, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data(h, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_process_multi(h, in_multi, num_in, out_multi, num_out),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data_multi(h, in_multi, num_in), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data_multi(h, out_multi, num_out), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_process_typed(h, typed_ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_process_wait(h, ptrs, k_block, ANIRA_WAIT_FOREVER, 0), 0U);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0), 0U);
    EXPECT_EQ(anira_handler_get_latency(h, 0), 0U);
    EXPECT_EQ(anira_handler_set_plan(h, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    EXPECT_EQ(anira_handler_plan_report(h), nullptr);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);

    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "handler not prepared", "rt"), 1U)
        << "the kind is latched: the later refusals are suppressed";
    const RecordCollector::Record record =
        anira_test::find_record(collector, "handler not prepared", "rt");
    EXPECT_EQ(record.m_message, "anira_handler_process: handler not prepared");
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(record.m_source, "rt");
    EXPECT_EQ(record.m_group, "anira.capi");
    EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
#endif

    // The accessor rule: reset re-arms and clears rt_error; get_latencies records nothing.
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    uint32_t count = 0;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // A handler is a user of the core, prepared or not.
    EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE);
}

// ============================================================================================
// The oracle
// ============================================================================================

TEST(AbiHandler, ProcessMatchesTheTwoPointXHandlerOnTheCustomPlan) {
    const Context context;
    GainOracle oracle(context);
    oracle.m_v2.set_inference_backend(anira::InferenceBackend::CUSTOM);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block), k_block));
    ASSERT_EQ(anira_handler_get_latency(oracle.h(), 0), oracle.m_v2.get_latency(0));
    ASSERT_GT(anira_handler_get_latency(oracle.h(), 0), 0U);
    oracle.run_blocks(32, k_block, Form::InPlace);

    // The custom row is the last entry of gain_with_custom, and the rows keep entry order.
    const uint32_t num_plans = anira_plan_report_num_plans(anira_handler_plan_report(oracle.h()));
    ASSERT_GE(num_plans, 1U);
    EXPECT_EQ(anira_handler_get_plan(oracle.h()), num_plans - 1);
}

TEST(AbiHandler, SetPlanSwitchesLikeSetInferenceBackend) {
    const Context context;
    {
        GainOracle oracle(context, /*default_custom=*/false);
        ASSERT_NO_FATAL_FAILURE(
            oracle.prepare(file_contract(k_gain_contract_json, k_block), k_block));
        const std::vector<anira_plan_info> info = plans_of(oracle.h());
        if (info.size() == 1) { GTEST_SKIP() << "one plan: no engine in this build"; }
        // No default engine anywhere: the first entry of the file that is in this build.
        EXPECT_EQ(anira_handler_get_plan(oracle.h()), 0U);

        for (uint32_t i = 0; i < info.size(); ++i) {
            EXPECT_EQ(anira_handler_set_plan(oracle.h(), i), ANIRA_OK);
            EXPECT_EQ(anira_handler_get_plan(oracle.h()), i);
            oracle.m_v2.set_inference_backend(backend_of(info[i]));
            oracle.run_blocks(16, k_block, Form::InPlace);
            oracle.reset();
        }

        for (size_t i = 0; i < info.size(); ++i) {
            EXPECT_EQ(info[i].variant, 0U) << "plan " << i;
            EXPECT_EQ(info[i].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT))
                << "plan " << i;
            EXPECT_DOUBLE_EQ(info[i].budget_ms, 5.0) << "plan " << i;
            if (i + 1 < info.size()) {
                EXPECT_EQ(info[i].engine_id, nullptr) << "plan " << i;
                EXPECT_NE(info[i].engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE)) << "plan " << i;
            } else {
                ASSERT_NE(info[i].engine_id, nullptr);
                EXPECT_STREQ(info[i].engine_id, k_custom);
            }
        }
    }

    // A default engine this build carries: the handler starts on that engine's plan.
    const std::vector<anira_engine> engines = anira_test::oracle_engines();
    ASSERT_FALSE(engines.empty());
    const anira_engine first_engine = engines.front();
    {
        anira::ModelConfig model = gain_with_custom();
        model.default_engine(first_engine);
        const std::vector<anira_backend_id> candidates = custom_candidates();
        Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(file_contract(k_gain_contract_json, k_block)), ANIRA_OK)
            << handler.m_err.message;
        const std::vector<anira_plan_info> info = plans_of(handler.m_handler);
        uint32_t expected = 0;
        bool found = false;
        for (uint32_t i = 0; i < info.size(); ++i) {
            if (info[i].engine == static_cast<uint32_t>(first_engine)) {
                expected = i;
                found = true;
            }
        }
        ASSERT_TRUE(found);
        EXPECT_EQ(anira_handler_get_plan(handler.m_handler), expected);
    }

    // A default engine the build lacks names an entry but no plan: plan 0.
    const std::optional<anira_engine> missing = missing_engine();
    if (missing.has_value()) {
        anira::ModelConfig model = gain_with_custom();
        model.default_engine(*missing);
        const std::vector<anira_backend_id> candidates = custom_candidates();
        Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(file_contract(k_gain_contract_json, k_block)), ANIRA_OK)
            << handler.m_err.message;
        EXPECT_EQ(anira_handler_get_plan(handler.m_handler), 0U);
    }
}

#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME)
TEST(AbiHandler, CnnMatchesTheTwoPointXHandler) {
    constexpr size_t k_cnn_block = 2048;
    const Context context;
    const std::vector<anira_backend_id> candidates = engine_candidates();
    Oracle oracle(context,
                  anira_test::cnn_model(),
                  candidates,
                  anira_test::bridged_2x(k_cnn_model_json, k_cnn_contract_json, false));
    ASSERT_NO_FATAL_FAILURE(
        oracle.prepare(file_contract(k_cnn_contract_json, k_cnn_block), k_cnn_block));
    EXPECT_EQ(anira_handler_get_latency(oracle.h(), 0), oracle.m_v2.get_latency(0));
    const std::vector<anira_plan_info> info = plans_of(oracle.h());
    ASSERT_FALSE(info.empty());
    for (uint32_t i = 0; i < info.size(); ++i) {
        ASSERT_EQ(anira_handler_set_plan(oracle.h(), i), ANIRA_OK);
        oracle.m_v2.set_inference_backend(backend_of(info[i]));
        oracle.run_blocks(12, k_cnn_block, Form::InPlace);
        oracle.reset();
    }
}
#endif

TEST(AbiHandler, SingleMultiPushAndPopFormsAgree) {
    const Context context;
    std::vector<std::unique_ptr<GainOracle>> oracles;
    for (size_t i = 0; i < std::size(k_forms); ++i) {
        oracles.push_back(std::make_unique<GainOracle>(context));
        oracles.back()->m_v2.set_inference_backend(anira::InferenceBackend::CUSTOM);
        ASSERT_NO_FATAL_FAILURE(
            oracles.back()->prepare(file_contract(k_gain_contract_json, k_block), k_block));
    }
    for (size_t k = 1; k <= 16; ++k) {
        std::vector<float> reference;
        for (size_t i = 0; i < oracles.size(); ++i) {
            std::vector<float> c;
            std::vector<float> v;
            size_t n_c = 0;
            size_t n_v = 0;
            oracles[i]->run_block(k, k_block, k_forms[i], c, v, n_c, n_v);
            ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "block " << k << ", form " << i;
            EXPECT_EQ(n_c, k_block) << "block " << k << ", form " << i;
            EXPECT_EQ(n_v, k_block) << "block " << k << ", form " << i;
            if (reference.empty()) { reference = c; }
            expect_same_block(c, reference, k);
            expect_same_block(v, reference, k);
        }
    }
}

TEST(AbiHandler, TypedTwinsAreBitEqualOnF32Slots) {
    const Context context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler floats(context, model, candidates);
    Handler typed(context, model, candidates);
    ASSERT_EQ(floats.prepare(explicit_contract()), ANIRA_OK) << floats.m_err.message;
    ASSERT_EQ(typed.prepare(explicit_contract()), ANIRA_OK) << typed.m_err.message;
    anira_handler* f = floats.m_handler;
    anira_handler* t = typed.m_handler;

    size_t k = 1;
    for (Form form : k_forms) {
        for (size_t i = 0; i < 4; ++i, ++k) {
            const std::vector<float> in = ramp(k);
            std::vector<float> f_buf = in;
            std::vector<float> t_buf = in;
            std::vector<float> f_out(k_block, -1.0F);
            std::vector<float> t_out(k_block, -1.0F);
            float gain = 1.0F;
            float f_gain_out = -1.0F;
            float t_gain_out = -1.0F;
            size_t n_f = 0;
            size_t n_t = 0;
            const size_t prev_f = anira_handler_get_available_samples(f, 0, 0);
            const size_t prev_t = anira_handler_get_available_samples(t, 0, 0);

            float* f_ch[1] = {f_buf.data()};
            void* t_ch[1] = {t_buf.data()};
            const float* f_in[1] = {f_buf.data()};
            const void* t_in[1] = {t_buf.data()};
            float* f_o[1] = {f_out.data()};
            void* t_o[1] = {t_out.data()};
            const float* f_gain_ch[1] = {&gain};
            const void* t_gain_ch[1] = {&gain};
            const float* const* f_ins[2] = {f_in, f_gain_ch};
            const void* const* t_ins[2] = {t_in, t_gain_ch};
            size_t f_num_in[2] = {k_block, 1};
            size_t t_num_in[2] = {k_block, 1};
            float* f_gout_ch[1] = {&f_gain_out};
            void* t_gout_ch[1] = {&t_gain_out};
            float* const* f_outs[2] = {f_o, f_gout_ch};
            void* const* t_outs[2] = {t_o, t_gout_ch};
            size_t f_num_out[2] = {k_block, 1};
            size_t t_num_out[2] = {k_block, 1};

            switch (form) {
                case Form::InPlace:
                    n_f = anira_handler_process(f, f_ch, k_block, 0);
                    n_t = anira_handler_process_typed(t, t_ch, k_block, 0);
                    f_out = f_buf;
                    t_out = t_buf;
                    break;
                case Form::Separate:
                    n_f = anira_handler_process_separate(f, f_in, k_block, f_o, k_block, 0);
                    n_t = anira_handler_process_separate_typed(t, t_in, k_block, t_o, k_block, 0);
                    break;
                case Form::Multi:
                    EXPECT_EQ(anira_handler_process_multi(f, f_ins, f_num_in, f_outs, f_num_out),
                              ANIRA_OK);
                    EXPECT_EQ(
                        anira_handler_process_multi_typed(t, t_ins, t_num_in, t_outs, t_num_out),
                        ANIRA_OK);
                    n_f = f_num_out[0];
                    n_t = t_num_out[0];
                    EXPECT_EQ(f_num_out[1], t_num_out[1]);
                    EXPECT_EQ(f_gain_out, t_gain_out);
                    break;
                case Form::PushPop:
                    EXPECT_EQ(anira_handler_push_data(f, f_in, k_block, 0), ANIRA_OK);
                    EXPECT_EQ(anira_handler_push_data_typed(t, t_in, k_block, 0), ANIRA_OK);
                    wait_for_push(f, prev_f, k_block);
                    wait_for_push(t, prev_t, k_block);
                    n_f = anira_handler_pop_data(f, f_o, k_block, 0);
                    n_t = anira_handler_pop_data_typed(t, t_o, k_block, 0);
                    break;
                case Form::PushPopMulti:
                    EXPECT_EQ(anira_handler_push_data_multi(f, f_ins, f_num_in), ANIRA_OK);
                    EXPECT_EQ(anira_handler_push_data_multi_typed(t, t_ins, t_num_in), ANIRA_OK);
                    wait_for_push(f, prev_f, k_block);
                    wait_for_push(t, prev_t, k_block);
                    EXPECT_EQ(anira_handler_pop_data_multi(f, f_outs, f_num_out), ANIRA_OK);
                    EXPECT_EQ(anira_handler_pop_data_multi_typed(t, t_outs, t_num_out), ANIRA_OK);
                    n_f = f_num_out[0];
                    n_t = t_num_out[0];
                    EXPECT_EQ(f_num_out[1], t_num_out[1]);
                    EXPECT_EQ(f_gain_out, t_gain_out);
                    break;
            }
            if (form != Form::PushPop && form != Form::PushPopMulti) {
                wait_for_block(f, prev_f);
                wait_for_block(t, prev_t);
            }
            ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "block " << k;
            EXPECT_EQ(n_f, n_t) << "block " << k;
            EXPECT_EQ(n_f, k_block) << "block " << k;
            expect_same_block(f_out, t_out, k);
            EXPECT_EQ(anira_handler_rt_error(f), ANIRA_OK) << "block " << k;
            EXPECT_EQ(anira_handler_rt_error(t), ANIRA_OK) << "block " << k;
        }
    }
}

// ============================================================================================
// Accessors
// ============================================================================================

TEST(AbiHandler, AvailableSamplesTracksTheOutputRing) {
    const Context context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0), anira_handler_get_latency(h, 0));
    for (size_t k = 1; k <= 4; ++k) {
        std::vector<float> block = ramp(k);
        float* ptrs[1] = {block.data()};
        const size_t prev = anira_handler_get_available_samples(h, 0, 0);
        EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), k_block);
        wait_for_block(h, prev);
    }
    EXPECT_LE(anira_handler_get_available_samples(h, 0, 0), 8 * k_block);

    EXPECT_EQ(anira_handler_get_available_samples(h, 1, 0), 0U) << "a Static output";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_available_samples(h, 99, 0), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 5), 0U) << "a channel mono has not";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiHandler, LatencyVectorIsIndexAlignedWithZeroForStaticOutputs) {
    const Context context;
    GainOracle oracle(context);
    oracle.m_v2.set_inference_backend(anira::InferenceBackend::CUSTOM);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block), k_block));
    anira_handler* h = oracle.h();

    uint32_t count = 0;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, 2U);
    uint32_t out[2] = {0, 0};
    count = 2;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, out), ANIRA_OK);
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(out[0], anira_handler_get_latency(h, 0));
    EXPECT_GT(out[0], 0U);
    EXPECT_EQ(out[1], 0U) << "gain_out is Static";
    EXPECT_EQ(anira_handler_get_latency(h, 1), 0U);

    uint32_t one[2] = {0, 77};
    count = 1;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, one), ANIRA_INCOMPLETE);
    EXPECT_EQ(one[0], out[0]);
    EXPECT_EQ(one[1], 77U) << "only what fit is written";
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(anira_handler_get_latencies(h, nullptr, out), ANIRA_ERROR_INVALID_ARGUMENT);

    const std::vector<unsigned int> v2 = oracle.m_v2.get_latency_vector();
    ASSERT_EQ(v2.size(), 2U);
    EXPECT_EQ(v2[0], out[0]);
    EXPECT_EQ(v2[1], out[1]);
    EXPECT_EQ(anira_handler_get_latency(h, 99), 0U);
}

TEST(AbiHandler, ResetReSeedsTheStreamAndClearsRtError) {
    const Context context;
    GainOracle oracle(context);
    oracle.m_v2.set_inference_backend(anira::InferenceBackend::CUSTOM);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block), k_block));
    anira_handler* h = oracle.h();
    oracle.run_blocks(3, k_block, Form::InPlace);

    std::vector<float> block = ramp(99);
    float* ptrs[1] = {block.data()};
    EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 99), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    oracle.reset();
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0), anira_handler_get_latency(h, 0))
        << "the priming re-seeded";
    oracle.run_blocks(3, k_block, Form::InPlace);
}

TEST(AbiHandler, SetPlanOutOfRangeIsAConfigNoOp) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    const uint32_t n = anira_plan_report_num_plans(anira_handler_plan_report(h));
    const uint32_t before = anira_handler_get_plan(h);
    EXPECT_EQ(anira_handler_set_plan(h, n), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_handler_get_plan(h), before);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_CONFIG);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(
        anira_test::count_records(collector, "anira_handler_set_plan: configuration error", "rt"),
        1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "anira_handler_set_plan", "rt");
    EXPECT_NE(record.m_flags & ANIRA_LOG_RECORD_CONTRACT_VIOLATION, 0U);
#endif
    EXPECT_EQ(anira_handler_set_plan(h, n), ANIRA_ERROR_CONFIG);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(
        anira_test::count_records(collector, "anira_handler_set_plan: configuration error", "rt"),
        1U)
        << "a second refusal of the same kind adds no record";
#endif
}

// ============================================================================================
// The plan report
// ============================================================================================

TEST(AbiHandler, PlanReportRows) {
    const Context context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    const anira_plan_report* report = anira_handler_plan_report(h);
    ASSERT_NE(report, nullptr);
    for (size_t k = 1; k <= 2; ++k) {
        std::vector<float> block = ramp(k);
        float* ptrs[1] = {block.data()};
        const size_t prev = anira_handler_get_available_samples(h, 0, 0);
        EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), k_block);
        wait_for_block(h, prev);
    }
    EXPECT_EQ(anira_handler_plan_report(h), report) << "the same report between prepares";

    const uint32_t num_plans = anira_plan_report_num_plans(report);
    EXPECT_EQ(num_plans, candidates.size()) << "one row per candidate the gain file names";
    uint32_t count = 0;
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, num_plans);
    std::vector<anira_plan_info> rows(count);
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
              ANIRA_OK);
    EXPECT_EQ(count, num_plans);
    if (num_plans > 1) {
        count = num_plans - 1;
        EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
                  ANIRA_INCOMPLETE);
        EXPECT_EQ(count, num_plans);
    }
    count = num_plans;
    EXPECT_EQ(anira_plan_report_plans(report, 3 * sizeof(uint32_t), &count, rows.data()),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "an element size below the record's head";
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), nullptr, rows.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // Slots: two inputs, two outputs, every one a host slot.
    anira_plan_slot slots[2] = {ANIRA_PLAN_SLOT_INIT, ANIRA_PLAN_SLOT_INIT};
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, 0, 1, sizeof(anira_plan_slot), &count, slots),
              ANIRA_OK);
    ASSERT_EQ(count, 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(slots[i].slot, i);
        EXPECT_EQ(slots[i].is_input, 1U);
        EXPECT_EQ(slots[i].domain_in, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(slots[i].domain_out, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(slots[i].edge_class, static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY));
        EXPECT_EQ(slots[i].allocate_class, static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY));
        EXPECT_EQ(slots[i].wait_strategy, static_cast<uint32_t>(ANIRA_WAIT_SPIN_BACKOFF))
            << "the strategy the core runs: this context is the first user of its generation";
        ASSERT_NE(slots[i].recipe, nullptr);
        EXPECT_STREQ(slots[i].recipe, "host");
        EXPECT_EQ(slots[i].reason, nullptr);
    }
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, 0, 0, sizeof(anira_plan_slot), &count, slots),
              ANIRA_OK);
    EXPECT_EQ(count, 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(slots[i].slot, i);
        EXPECT_EQ(slots[i].is_input, 0U);
    }
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, num_plans, 1, sizeof(anira_plan_slot), &count, slots),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // Extensions: the gain files carry none.
    count = 0;
    EXPECT_EQ(anira_plan_report_exts(report, 0, sizeof(anira_plan_ext), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, 0U);

    // A caller with a larger stride: the row fills the slot record only.
    struct Padded {
        anira_plan_slot s;
        uint64_t pad[2];
    };
    static_assert(sizeof(Padded) == sizeof(anira_plan_slot) + 16);
    Padded padded[2];
    for (Padded& p : padded) {
        p.s = ANIRA_PLAN_SLOT_INIT;
        p.pad[0] = 0xA5A5A5A5A5A5A5A5ULL;
        p.pad[1] = 0x5A5A5A5A5A5A5A5AULL;
    }
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, 0, 1, sizeof(Padded), &count, &padded[0].s),
              ANIRA_OK);
    EXPECT_EQ(count, 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(padded[i].s.slot, i);
        EXPECT_EQ(padded[i].s.is_input, 1U);
        EXPECT_STREQ(padded[i].s.recipe, "host");
        EXPECT_EQ(padded[i].pad[0], 0xA5A5A5A5A5A5A5A5ULL);
        EXPECT_EQ(padded[i].pad[1], 0x5A5A5A5A5A5A5A5AULL);
    }

    // A second context asking for BLOCKING while the first lives reports the strategy the core
    // runs: first-wins (the "wait strategy mismatch" record is expected).
    {
        const Context second(1, ANIRA_WAIT_BLOCKING);
        Handler other(second, model, candidates);
        ASSERT_EQ(other.prepare(explicit_contract()), ANIRA_OK) << other.m_err.message;
        anira_plan_slot slot = ANIRA_PLAN_SLOT_INIT;
        count = 1;
        EXPECT_EQ(anira_plan_report_slots(anira_handler_plan_report(other.m_handler),
                                          0,
                                          1,
                                          sizeof(anira_plan_slot),
                                          &count,
                                          &slot),
                  ANIRA_INCOMPLETE);
        EXPECT_EQ(slot.wait_strategy, static_cast<uint32_t>(ANIRA_WAIT_SPIN_BACKOFF));
    }
}

// ============================================================================================
// Lifetimes
// ============================================================================================

TEST(AbiHandler, ConfigsAreCopiedAndDestroyableRightAfterCreate) {
    const Context context;
    anira_error err = ANIRA_ERROR_INIT;
    anira_handler* h = nullptr;
    {
        const anira::ModelConfig model = gain_with_custom();
        const std::vector<anira_backend_id> candidates = custom_candidates();
        anira_pipeline* pipeline = nullptr;
        ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
        const anira_model_config* variants[] = {model.native()};
        ASSERT_EQ(anira_pipeline_add_inference(pipeline,
                                               variants,
                                               1,
                                               candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &err),
                  ANIRA_OK)
            << err.message;
        ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &h, &err), ANIRA_OK)
            << err.message;
        anira_pipeline_destroy(pipeline);
    }
    {
        const anira::ContractHandle contract = explicit_contract();
        ASSERT_EQ(anira_handler_prepare(h, contract.native(), &err), ANIRA_OK) << err.message;
    }
    for (size_t k = 1; k <= 4; ++k) {
        std::vector<float> block = ramp(k);
        float* ptrs[1] = {block.data()};
        const size_t prev = anira_handler_get_available_samples(h, 0, 0);
        EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), k_block);
        wait_for_block(h, prev);
        if (k == 1) {
            expect_all(block, 0.0F, "the priming zeros");
        } else {
            expect_same_block(block, ramp(k - 1), k);
        }
    }
    anira_handler_destroy(h);
}

TEST(AbiHandler, HandlerDestroyJoinsThePoolWithTheLastSession) {
    {
        const Context context;
        {
            const anira::ModelConfig model = gain_with_custom();
            const std::vector<anira_backend_id> candidates = custom_candidates();
            Handler handler(context, model, candidates);
            ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
            EXPECT_EQ(anira_num_inference_threads(), 2U);
            EXPECT_EQ(anira::Core::get_num_sessions(), 1);
            EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE)
                << "a session, a handler and a context live";
        }
        EXPECT_EQ(anira_num_inference_threads(), 0U) << "the pool joined with the last session";
        EXPECT_EQ(anira::Core::get_num_sessions(), 0);
        EXPECT_EQ(anira::Core::get_num_handlers(), 0U);
        EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE) << "the context still lives";
    }
    EXPECT_EQ(anira_shutdown(), ANIRA_OK) << "nothing lives";
}

TEST(AbiHandler, GeneratorPushIsANoOpAndPopPulls) {
    constexpr size_t k_hop = 2048;
    const Context context;
    const anira::ModelConfig model = anira_test::generator_model();
    const std::vector<anira_backend_id> none{
        {sizeof(anira_backend_id), ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, nullptr}};
    Handler handler(context, model, none);
    ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 0.0, 10.0)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    anira_test::SleepingParamFillBackend backend(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));
    EXPECT_EQ(anira_handler_get_latency(h, 0), k_hop) << "a generator counts from its first pull";

    // A push on a generator's static input submits nothing.
    const float params[4] = {3.0F, 0.0F, 0.0F, 0.0F};
    const float* param_ch[1] = {params};
    EXPECT_EQ(anira_handler_push_data(h, param_ch, 4, 0), ANIRA_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(backend.m_calls.load(), 0);

    // A pull delivers the priming block first, then the parameter's fill.
    for (int call = 0; call < 2; ++call) {
        std::vector<float> out(k_hop, -1.0F);
        const float* const* in[1] = {param_ch};
        size_t num_in[1] = {4};
        float* out_ch[1] = {out.data()};
        float* const* outs[1] = {out_ch};
        size_t num_out[1] = {k_hop};
        const size_t prev = anira_handler_get_available_samples(h, 0, 0);
        ASSERT_EQ(anira_handler_process_multi(h, in, num_in, outs, num_out), ANIRA_OK);
        EXPECT_EQ(num_out[0], k_hop) << "call " << call;
        wait_for_block(h, prev);
        expect_all(out, call == 0 ? 0.0F : 3.0F, call == 0 ? "the priming zeros" : "the fill");
    }
    EXPECT_EQ(anira_handler_get_latency(h, 0), k_hop);
}
