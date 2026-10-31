// anira/abi/handler.h: the C twins of test_InferenceHandlerApi.cpp. The bundled gain model runs
// block by block (on the custom plan's pass-through engine on every leg, on an engine's plan where
// one is present), each block waiting for its own inference, so no block is ever missed and every
// output is compared bit for bit with the closed form of the stream (anira_test::DelayLine).
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/engine_oracle.h"
#include "../support/log_record_collector.h"
#include "capi/port.h"
#include "capi/words.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

using anira_test::Context;
using anira_test::custom_candidates;
using anira_test::DestroyFirst;
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

constexpr std::array<Form, 5> k_forms{Form::InPlace,
                                      Form::Separate,
                                      Form::Multi,
                                      Form::PushPop,
                                      Form::PushPopMulti};

/// The latency of the bundled gain model under its contract at 512 samples and 48 kHz (a
/// 5 ms budget, no wait ratio): the closed form of LatencyCalculator, one hop.
constexpr uint32_t k_gain_latency = 512U;

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

/// An engine this build does not carry, if there is one.
std::optional<anira_engine> missing_engine() {
    const std::vector<anira::BackendId> enabled = anira::enabled_engines();
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

/// A C handler over the bundled gain model with the custom row, driven block by block, each
/// block waited for and checked bit for bit against the closed form of the stream
/// (anira_test::DelayLine): the custom plan's pass-through copies the input and the gain, an
/// engine plan multiplies by the Static gain and answers the peak of the window.
struct Oracle {
    explicit Oracle(const Context& context, bool default_custom = true)
        : m_c(context, gain_with_custom(default_custom), custom_candidates())
        , m_pass_through(default_custom) {}

    anira_handler* h() { return m_c.m_handler; }

    void prepare(const anira::ContractHandle& contract) {
        ASSERT_EQ(m_c.prepare(contract), ANIRA_OK) << m_c.m_err.message;
        m_face = std::make_unique<anira_test::FloatFace>(m_c.m_handler);
        m_line = anira_test::DelayLine(anira_handler_get_latency(h(), 0));
        m_previous_gain_out = c_static_output(1);
        EXPECT_EQ(m_previous_gain_out, 0.0F) << "zero until an inference is collected";
    }

    /// The C side's stored Static output of `slot`, read whole in the spec's shape: the value
    /// of the latest collected inference. The gain model's gain_out holds one element.
    float c_static_output(uint32_t slot) {
        float value = -1.0F;
        const anira::capi::StaticSlot* store = anira::capi::static_slot(h()->m_output_ports, slot);
        EXPECT_NE(store, nullptr) << "output slot " << slot << " is not Static";
        if (store == nullptr) { return value; }
        EXPECT_EQ(store->num_elements(), 1U) << "output slot " << slot;
        if (store->num_elements() != 1U) { return value; }
        const anira_tensor tensor = anira_test::whole_f32(&value, store->shape());
        EXPECT_EQ(anira_handler_get_static_output(h(), slot, &tensor), ANIRA_OK);
        return value;
    }

    /// One block through `form`; c receives the output, n_c the count, m_expected the closed
    /// form of the block. The static gain slot travels through the multi forms with one value,
    /// 1.0F, as the whole tensor in the spec's shape (FloatFace presents a carried Static slot
    /// that way).
    void run_block(size_t k, size_t block, Form form, std::vector<float>& c, size_t& n_c) {
        ASSERT_NE(m_face, nullptr) << "run_block before a successful prepare";
        const std::vector<float> in = ramp(k, block);
        c = in;
        std::vector<float> c_out(block, -1.0F);
        const float gain_in = m_gain;
        float gain_out = -1.0F;
        // The Static output of a block carrying the gain: the custom plan's pass-through copies
        // the gain, the gain model's engines answer the peak of the window.
        const float expected_static = m_pass_through ? gain_in : anira_test::gain_peak(in, gain_in);

        const size_t prev = anira_test::available(h());

        switch (form) {
            case Form::InPlace: {
                const std::array<float*, 1> c_ch{c.data()};
                EXPECT_EQ(m_face->process_inplace(c_ch.data(), block, 0, &n_c), ANIRA_OK)
                    << "block " << k;
                wait_for_block(h(), prev);
                break;
            }
            case Form::Separate: {
                const std::array<const float*, 1> c_in{c.data()};
                const std::array<float*, 1> c_ch{c_out.data()};
                EXPECT_EQ(m_face->process(c_in.data(), block, c_ch.data(), block, 0, &n_c),
                          ANIRA_OK)
                    << "block " << k;
                wait_for_block(h(), prev);
                c = c_out;
                break;
            }
            case Form::Multi: {
                const std::array<const float*, 1> c_in_ch{c.data()};
                const std::array<const float*, 1> c_gain_ch{&gain_in};
                const std::array<const float* const*, 2> c_in{c_in_ch.data(), c_gain_ch.data()};
                const std::array<size_t, 2> c_num_in{block, 1};
                const std::array<float*, 1> c_out_ch{c_out.data()};
                const std::array<float*, 1> c_gout_ch{&gain_out};
                const std::array<float* const*, 2> c_outs{c_out_ch.data(), c_gout_ch.data()};
                const std::array<size_t, 2> c_num_out{block, 1};
                std::array<size_t, 2> c_delivered{7, 7};
                EXPECT_EQ(m_face->process_multi(c_in.data(),
                                                c_num_in.data(),
                                                c_outs.data(),
                                                c_num_out.data(),
                                                c_delivered.data()),
                          ANIRA_OK);
                n_c = c_delivered[0];
                wait_for_block(h(), prev);
                EXPECT_EQ(c_delivered[1], 1U) << "block " << k;
                // A call's Static output is the latest inference collected when it ran: the
                // previous block's, or this block's when its collect ran within the call. Only
                // the settled value, read once the block has been waited for, is certain.
                const float before = m_previous_gain_out;
                const float settled = c_static_output(1);
                EXPECT_EQ(settled, expected_static)
                    << "block " << k << ": the settled Static output";
                EXPECT_TRUE(gain_out == before || gain_out == settled)
                    << "block " << k << ": the call answered " << gain_out
                    << ", neither the previous " << before << " nor the settled " << settled;
                m_previous_gain_out = settled;
                c = c_out;
                break;
            }
            case Form::PushPop: {
                const std::array<const float*, 1> c_in{c.data()};
                const std::array<float*, 1> c_ch{c_out.data()};
                EXPECT_EQ(m_face->push_data(c_in.data(), block, 0), ANIRA_OK);
                wait_for_push(h(), prev, block);
                EXPECT_EQ(m_face->pop_data(c_ch.data(), block, 0, &n_c), ANIRA_OK) << "block " << k;
                c = c_out;
                break;
            }
            case Form::PushPopMulti: {
                const std::array<const float*, 1> c_in_ch{c.data()};
                const std::array<const float*, 1> c_gain_ch{&gain_in};
                const std::array<const float* const*, 2> c_in{c_in_ch.data(), c_gain_ch.data()};
                const std::array<size_t, 2> c_num_in{block, 1};
                const std::array<float*, 1> c_out_ch{c_out.data()};
                const std::array<float*, 1> c_gout_ch{&gain_out};
                const std::array<float* const*, 2> c_outs{c_out_ch.data(), c_gout_ch.data()};
                const std::array<size_t, 2> c_num_out{block, 1};
                std::array<size_t, 2> c_delivered{7, 7};
                EXPECT_EQ(m_face->push_data_multi(c_in.data(), c_num_in.data()), ANIRA_OK);
                wait_for_push(h(), prev, block);
                EXPECT_EQ(
                    m_face->pop_data_multi(c_outs.data(), c_num_out.data(), c_delivered.data()),
                    ANIRA_OK);
                n_c = c_delivered[0];
                EXPECT_EQ(gain_out, expected_static) << "block " << k;
                c = c_out;
                break;
            }
        }
        // The closed form of the block: the multi forms carry the Static gain, the others leave
        // the store at its value (zero until a call carries one).
        if (form == Form::Multi || form == Form::PushPopMulti) { m_static_gain = gain_in; }
        m_expected = m_line.next(in, m_pass_through ? 1.0F : m_static_gain);
    }

    /// `blocks` blocks; every block is delivered in full and equals the closed form.
    void run_blocks(size_t blocks, size_t block, Form form) {
        for (size_t i = 0; i < blocks; ++i) {
            const size_t k = m_next_block++;
            std::vector<float> c;
            size_t n_c = 0;
            run_block(k, block, form, c, n_c);
            if (::testing::Test::HasFatalFailure()) { return; }
            EXPECT_EQ(n_c, block) << "block " << k;
            expect_same_block(c, m_expected, k);
        }
    }

    void reset() {
        anira_handler_reset(h());
        m_line.reset();
    }

    Handler m_c;
    /// The Static gain the multi forms carry; the gain model's engines scale by it, the custom
    /// row's pass-through copies the stream whatever it is.
    float m_gain = 1.0F;
    std::unique_ptr<anira_test::FloatFace> m_face;  ///< the C side's float face, built by prepare()
    float m_previous_gain_out = 0.0F;  ///< the settled Static output after the last Multi block
    size_t m_next_block = 1;
    anira_test::DelayLine m_line;
    std::vector<float> m_expected;  ///< the closed form of the last block run_block() ran
    bool m_pass_through = true;     ///< the running plan copies (the custom plan)
    float m_static_gain = 0.0F;     ///< the Static gain input the engines multiply by
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
    anira_test::add_passthrough(pipeline);
    const anira::ModelConfig model = gain_with_custom();
    const std::array<const anira_model_config*, 1> variants{model.native()};
    EXPECT_EQ(anira_pipeline_add_inference(nullptr, variants.data(), 1, nullptr, 0, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err),
              ANIRA_OK)
        << err.message;

    anira_handler* handler = nullptr;
    EXPECT_EQ(anira_handler_create(nullptr, pipeline, &handler, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_create(context.m_context, nullptr, &handler, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_create(context.m_context, pipeline, nullptr, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(handler, nullptr);

    // The blocks of a float host: planar tensors over its channel pointers, the same tensor
    // on both sides for in place.
    std::vector<float> block(k_block, 0.0F);
    const std::array<float*, 1> ptrs{block.data()};
    const std::array<const float*, 1> in_ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
    const anira_tensor in = anira_test::planar_f32(in_ptrs.data(), 1, k_block);
    uint32_t count = 0;
    size_t delivered = 7;
    EXPECT_EQ(anira_handler_process(nullptr, &io, 0, &io, 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U) << "a refusal writes 0 to the caller's count";
    EXPECT_EQ(anira_handler_process(nullptr, &io, 0, &io, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "the count is optional";
    EXPECT_EQ(anira_handler_push_data(nullptr, &in, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    delivered = 7;
    EXPECT_EQ(anira_handler_pop_data(nullptr, &io, 0, &delivered), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    std::array<size_t, 1> multi_delivered{7};
    EXPECT_EQ(anira_handler_process_multi(nullptr, &in, 1, &io, 1, multi_delivered.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(multi_delivered[0], 0U) << "the counts of a multi form are zeroed on every return";
    EXPECT_EQ(anira_handler_get_latency(nullptr, 0), 0U);
    EXPECT_EQ(anira_handler_get_latencies(nullptr, &count, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    delivered = 7;
    EXPECT_EQ(anira_handler_get_available_samples(nullptr, 0, 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
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
    const Handler handler(context, model, candidates);
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);

    // An unprepared handler has no slots to size an adapter by: the blocks are planar tensors
    // built by hand, as a float host builds them before its first prepare.
    std::vector<float> block(k_block, 0.25F);
    const std::array<float*, 1> ptrs{block.data()};
    const std::array<const float*, 1> in_ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
    const anira_tensor in = anira_test::planar_f32(in_ptrs.data(), 1, k_block);

    size_t delivered = 7;
    EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data(h, &in, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data(h, &io, 0, &delivered), ANIRA_ERROR_NOT_PREPARED);
    std::array<size_t, 1> multi_delivered{7};
    EXPECT_EQ(anira_handler_process_multi(h, &in, 1, &io, 1, multi_delivered.data()),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(multi_delivered[0], 0U) << "a refused multi form zeroes its counts";
    EXPECT_EQ(anira_handler_push_data_multi(h, &in, 1), ANIRA_ERROR_NOT_PREPARED);
    multi_delivered[0] = 7;
    EXPECT_EQ(anira_handler_pop_data_multi(h, &io, 1, multi_delivered.data()),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(multi_delivered[0], 0U) << "a refused multi form zeroes its counts";
    delivered = 7;
    EXPECT_EQ(anira_handler_process_wait(h, &io, 0, &io, 0, &delivered, ANIRA_WAIT_FOREVER),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(delivered, 0U);
    delivered = 7;
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &delivered), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(delivered, 0U);
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

TEST(AbiHandler, ProcessMatchesTheClosedFormOnTheCustomPlan) {
    const Context context;
    Oracle oracle(context);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block)));
    ASSERT_EQ(anira_handler_get_latency(oracle.h(), 0), k_gain_latency);
    oracle.run_blocks(32, k_block, Form::InPlace);

    // The custom row is the last entry of gain_with_custom, and the rows keep entry order.
    const uint32_t num_plans = anira_plan_report_num_plans(anira_handler_plan_report(oracle.h()));
    ASSERT_GE(num_plans, 1U);
    EXPECT_EQ(anira_handler_get_plan(oracle.h()), num_plans - 1);
}

TEST(AbiHandler, SetPlanSwitchesTheRunningPlan) {
    const Context context;
    {
        Oracle oracle(context, /*default_custom=*/false);
        ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block)));
        const std::vector<anira_plan_info> info = plans_of(oracle.h());
        if (info.size() == 1) { GTEST_SKIP() << "one plan: no engine in this build"; }
        // No default engine anywhere: the first entry of the file that is in this build.
        EXPECT_EQ(anira_handler_get_plan(oracle.h()), 0U);

        // The plans answer distinguishable, non-zero streams: under a Static gain of 0.5 the gain
        // model's engines halve the stream, the custom row's pass-through copies it. Every
        // switch goes from one kind to the other (each engine's plan, then the custom one), so
        // the closed form of every block says which plan ran it, and a switch that did not take
        // effect fails the blocks behind it.
        oracle.m_gain = 0.5F;
        const auto custom = static_cast<uint32_t>(info.size() - 1);
        ASSERT_EQ(info[custom].engine, static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM));
        for (uint32_t i = 0; i < custom; ++i) {
            for (const uint32_t plan : {i, custom}) {
                ASSERT_EQ(anira_handler_set_plan(oracle.h(), plan), ANIRA_OK);
                EXPECT_EQ(anira_handler_get_plan(oracle.h()), plan);
                oracle.m_pass_through = plan == custom;
                oracle.run_blocks(8, k_block, Form::Multi);
                oracle.reset();
            }
        }

        for (size_t i = 0; i < info.size(); ++i) {
            EXPECT_EQ(info[i].variant, 0U) << "plan " << i;
            EXPECT_EQ(info[i].provider, static_cast<uint32_t>(ANIRA_PROVIDER_CPU)) << "plan " << i;
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

    // A default engine the build lacks names an entry but no plan: plan 0, and one Warning
    // naming what was asked and the plan the handler starts on (the default provider's rule).
    const std::optional<anira_engine> missing = missing_engine();
    if (missing.has_value()) {
        anira_drain_log();
        RecordCollector collector;
        const Context warning_context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_WARNING);
        anira::ModelConfig model = gain_with_custom();
        model.default_engine(*missing);
        const std::vector<anira_backend_id> candidates = custom_candidates();
        Handler handler(warning_context, model, candidates);
        ASSERT_EQ(handler.prepare(file_contract(k_gain_contract_json, k_block)), ANIRA_OK)
            << handler.m_err.message;
        EXPECT_EQ(anira_handler_get_plan(handler.m_handler), 0U);
        anira_drain_log();
#ifdef ENABLE_LOGGING
        const std::string asked = std::string("no plan runs default_engine '") +
                                  anira::capi::engine_word(*missing) +
                                  "' here; the handler starts on plan 0 (";
        EXPECT_TRUE(collector.has(asked.c_str(), "native")) << asked;
#endif
    }
}

// The selection is one atomic value (the dense plan index, held by the session), so callers
// on several threads cannot leave get_plan and the running engine in disagreement: with an
// index beside a backend, two interleaved calls could store the index of one plan and the
// backend of the other.
TEST(AbiHandler, ConcurrentSetPlanLeavesOneSelection) {
    const Context context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(file_contract(k_gain_contract_json, k_block)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    const std::vector<anira_plan_info> info = plans_of(h);
    if (info.size() < 2) { GTEST_SKIP() << "one plan: no engine in this build"; }

    // No audio runs: only the selection is exercised.
    const auto hammer = [h](uint32_t plan) {
        for (int i = 0; i < 20000; ++i) { EXPECT_EQ(anira_handler_set_plan(h, plan), ANIRA_OK); }
    };
    for (int round = 0; round < 20; ++round) {
        std::thread first(hammer, 0U);
        std::thread second(hammer, 1U);
        first.join();
        second.join();
        const uint32_t selected = anira_handler_get_plan(h);
        ASSERT_LT(selected, 2U) << "round " << round;
        EXPECT_EQ(h->m_manager->get_plan(), selected) << "round " << round;
        EXPECT_EQ(static_cast<uint32_t>(anira_test::session_of(h)->m_plans[selected].m_engine),
                  h->m_plans[selected].m_info.engine)
            << "round " << round << ": get_plan names a plan whose engine is not the one running";
    }

    // Every plan reads back as itself.
    for (uint32_t i = 0; i < info.size(); ++i) {
        EXPECT_EQ(anira_handler_set_plan(h, i), ANIRA_OK);
        EXPECT_EQ(anira_handler_get_plan(h), i);
        EXPECT_EQ(h->m_manager->get_plan(), i);
    }
}

TEST(AbiHandler, SingleMultiPushAndPopFormsAgree) {
    const Context context;
    std::vector<std::unique_ptr<Oracle>> oracles;
    for (size_t i = 0; i < std::size(k_forms); ++i) {
        oracles.push_back(std::make_unique<Oracle>(context));
        ASSERT_NO_FATAL_FAILURE(
            oracles.back()->prepare(file_contract(k_gain_contract_json, k_block)));
    }
    for (size_t k = 1; k <= 16; ++k) {
        std::vector<float> reference;
        for (size_t i = 0; i < oracles.size(); ++i) {
            std::vector<float> c;
            size_t n_c = 0;
            oracles[i]->run_block(k, k_block, k_forms[i], c, n_c);
            ASSERT_FALSE(::testing::Test::HasFatalFailure()) << "block " << k << ", form " << i;
            EXPECT_EQ(n_c, k_block) << "block " << k << ", form " << i;
            if (reference.empty()) { reference = c; }
            expect_same_block(c, reference, k);
            expect_same_block(c, oracles[i]->m_expected, k);
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

    size_t count = 7;
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &count), ANIRA_OK);
    EXPECT_EQ(count, anira_handler_get_latency(h, 0));
    for (size_t k = 1; k <= 4; ++k) {
        std::vector<float> block = ramp(k);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
        wait_for_block(h, prev);
    }
    EXPECT_LE(anira_test::available(h), 8 * k_block);

    count = 7;
    EXPECT_EQ(anira_handler_get_available_samples(h, 1, 0, &count), ANIRA_OK) << "a Static output";
    EXPECT_EQ(count, 0U) << "a Static output has no ring";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    count = 7;
    EXPECT_EQ(anira_handler_get_available_samples(h, 99, 0, &count), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 5, &count), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a channel mono has not";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "the count is the call's only output";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiHandler, LatencyVectorIsIndexAlignedWithZeroForStaticOutputs) {
    const Context context;
    Oracle oracle(context);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block)));
    const anira_handler* h = oracle.h();

    uint32_t count = 0;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, 2U);
    std::array<uint32_t, 2> out{0, 0};
    count = 2;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, out.data()), ANIRA_OK);
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(out[0], anira_handler_get_latency(h, 0));
    EXPECT_GT(out[0], 0U);
    EXPECT_EQ(out[1], 0U) << "gain_out is Static";
    EXPECT_EQ(anira_handler_get_latency(h, 1), 0U);

    std::array<uint32_t, 2> one{0, 77};
    count = 1;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, one.data()), ANIRA_INCOMPLETE);
    EXPECT_EQ(one[0], out[0]);
    EXPECT_EQ(one[1], 77U) << "only what fit is written";
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(anira_handler_get_latencies(h, nullptr, out.data()), ANIRA_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(out[0], k_gain_latency);
    EXPECT_EQ(anira_handler_get_latency(h, 99), 0U);
}

TEST(AbiHandler, ResetReSeedsTheStreamAndClearsRtError) {
    const Context context;
    Oracle oracle(context);
    ASSERT_NO_FATAL_FAILURE(oracle.prepare(file_contract(k_gain_contract_json, k_block)));
    anira_handler* h = oracle.h();
    oracle.run_blocks(3, k_block, Form::InPlace);

    std::vector<float> block = ramp(99);
    const std::array<float*, 1> ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
    EXPECT_EQ(anira_handler_process(h, &io, 99, &io, 99, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    oracle.reset();
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_test::available(h), anira_handler_get_latency(h, 0)) << "the priming re-seeded";
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
// A neural model's stream
// ============================================================================================

// The steerable-nafx CNN through a C handler with no processor, on every plan of the build's
// engines, a fresh stream per plan: the latency priming, then per 2048-sample block the CNN's own
// output on the window anira's default fill builds (the 15380 samples ending with the block, the
// receptive field ahead of the hop), the model run directly in the engine room on windows built
// by hand (anira_test::EngineOracle, anira_test::receptive_window). The CNN has no closed form;
// the oracle shares nothing with the handler but the plan's record.
TEST(AbiHandler, TheCnnStreamIsTheModelOnItsWindows) {
    constexpr size_t k_cnn_block = 2048;
    constexpr size_t k_cnn_window = 15380;
    constexpr size_t k_blocks = 6;
    const std::vector<anira_backend_id> candidates = anira_test::engine_candidates();
    if (candidates.empty()) { GTEST_SKIP() << "no built-in engine in this build"; }
    const Context context;
    uint32_t num_plans = 1;
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        Handler handler(context, anira_test::cnn_model(), candidates);
        ASSERT_EQ(handler.prepare(file_contract(k_cnn_contract_json, k_cnn_block)), ANIRA_OK)
            << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        num_plans = anira_plan_report_num_plans(anira_handler_plan_report(h));
        ASSERT_EQ(anira_handler_set_plan(h, plan), ANIRA_OK);
        anira_test::EngineOracle oracle(h, plan);
        ASSERT_TRUE(oracle.ready());

        const size_t priming = anira_test::available(h);
        std::vector<float> signal;
        std::vector<float> stream;
        for (size_t k = 0; k < k_blocks; ++k) {
            std::vector<float> block(k_cnn_block);
            for (size_t i = 0; i < k_cnn_block; ++i) {
                block[i] = 0.5F * std::sin(static_cast<float>(signal.size() + i) * 0.003F);
            }
            signal.insert(signal.end(), block.begin(), block.end());
            const std::array<float*, 1> ptrs{block.data()};
            const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_cnn_block);
            const size_t prev = anira_test::available(h);
            size_t delivered = 0;
            ASSERT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_OK);
            ASSERT_EQ(delivered, k_cnn_block) << "block " << k << " missed";
            wait_for_block(h, prev);
            stream.insert(stream.end(), block.begin(), block.end());
        }

        std::vector<float> expected(priming, 0.0F);
        for (size_t j = 0; expected.size() < stream.size(); ++j) {
            const std::vector<std::vector<float>> out =
                oracle.run({anira_test::receptive_window(signal, j, k_cnn_block, k_cnn_window)});
            ASSERT_FALSE(out.empty()) << "the oracle's inference " << j << " failed";
            expected.insert(expected.end(), out[0].begin(), out[0].end());
        }
        size_t wrong = 0;
        for (size_t n = 0; n < stream.size(); ++n) {
            if (std::abs(stream[n] - expected[n]) > 1e-5F) {
                if (wrong == 0) {
                    ADD_FAILURE() << "sample " << n << ": " << stream[n] << ", the CNN on its "
                                  << "window " << expected[n];
                }
                ++wrong;
            }
        }
        EXPECT_EQ(wrong, 0U) << "of " << stream.size() << " samples";
        EXPECT_GT(*std::max_element(expected.begin() + static_cast<std::ptrdiff_t>(priming),
                                    expected.end()),
                  0.0F)
            << "the oracle's output is not silence";
    }
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
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
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

    // Slots: two inputs, two outputs, every one in host memory.
    std::array<anira_plan_slot, 2> slots{ANIRA_PLAN_SLOT_INIT, ANIRA_PLAN_SLOT_INIT};
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, 0, 1, sizeof(anira_plan_slot), &count, slots.data()),
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
    EXPECT_EQ(anira_plan_report_slots(report, 0, 0, sizeof(anira_plan_slot), &count, slots.data()),
              ANIRA_OK);
    EXPECT_EQ(count, 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(slots[i].slot, i);
        EXPECT_EQ(slots[i].is_input, 0U);
    }
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report,
                                      num_plans,
                                      1,
                                      sizeof(anira_plan_slot),
                                      &count,
                                      slots.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // Extensions: the gain files carry none.
    count = 0;
    EXPECT_EQ(anira_plan_report_exts(report, 0, sizeof(anira_plan_ext), &count, nullptr), ANIRA_OK);
    EXPECT_EQ(count, 0U);

    // A caller with a larger stride: the row fills the slot record only.
    struct Padded {
        anira_plan_slot m_slot;
        std::array<uint64_t, 2> m_pad;
    };
    static_assert(sizeof(Padded) == sizeof(anira_plan_slot) + 16);
    std::array<Padded, 2> padded;
    for (Padded& p : padded) {
        p.m_slot = ANIRA_PLAN_SLOT_INIT;
        p.m_pad[0] = 0xA5A5A5A5A5A5A5A5ULL;
        p.m_pad[1] = 0x5A5A5A5A5A5A5A5AULL;
    }
    count = 2;
    EXPECT_EQ(anira_plan_report_slots(report, 0, 1, sizeof(Padded), &count, &padded[0].m_slot),
              ANIRA_OK);
    EXPECT_EQ(count, 2U);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(padded[i].m_slot.slot, i);
        EXPECT_EQ(padded[i].m_slot.is_input, 1U);
        EXPECT_STREQ(padded[i].m_slot.recipe, "host");
        EXPECT_EQ(padded[i].m_pad[0], 0xA5A5A5A5A5A5A5A5ULL);
        EXPECT_EQ(padded[i].m_pad[1], 0x5A5A5A5A5A5A5A5AULL);
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

// Every slot row tells how its plan bound the slot: the custom plan's registered engine (the
// pass-through) binds every slot itself; the ONNX Runtime plan of the gain model binds
// audio_in and both outputs by position
// (the graph names them data, processed_data and peak) and the gain input by its canonical
// name, which the graph has.
TEST(AbiHandler, ASlotReportsHowItsPlanBoundIt) {
    const Context context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    ASSERT_NE(report, nullptr);
    const uint32_t num_plans = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> plans(num_plans);
    uint32_t count = num_plans;
    ASSERT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, plans.data()),
              ANIRA_OK);
    const auto slots_of = [report](uint32_t plan, bool inputs) {
        std::array<anira_plan_slot, 2> rows{ANIRA_PLAN_SLOT_INIT, ANIRA_PLAN_SLOT_INIT};
        uint32_t n = 2;
        EXPECT_EQ(anira_plan_report_slots(report,
                                          plan,
                                          inputs ? 1U : 0U,
                                          sizeof(anira_plan_slot),
                                          &n,
                                          rows.data()),
                  ANIRA_OK);
        EXPECT_EQ(n, 2U);
        return rows;
    };
    bool custom_seen = false;
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        const std::array<anira_plan_slot, 2> inputs = slots_of(plan, true);
        const std::array<anira_plan_slot, 2> outputs = slots_of(plan, false);
        if (plans[plan].engine == static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM)) {
            custom_seen = true;
            for (const anira_plan_slot& row : inputs) {
                EXPECT_EQ(row.binding, static_cast<uint32_t>(ANIRA_BINDING_ENGINE));
            }
            for (const anira_plan_slot& row : outputs) {
                EXPECT_EQ(row.binding, static_cast<uint32_t>(ANIRA_BINDING_ENGINE));
            }
        }
#ifdef USE_ONNXRUNTIME
        if (plans[plan].engine == static_cast<uint32_t>(ANIRA_ENGINE_ONNXRUNTIME)) {
            EXPECT_EQ(inputs[0].binding, static_cast<uint32_t>(ANIRA_BINDING_POSITION));
            EXPECT_EQ(inputs[1].binding, static_cast<uint32_t>(ANIRA_BINDING_NAME));
            EXPECT_EQ(outputs[0].binding, static_cast<uint32_t>(ANIRA_BINDING_POSITION));
            EXPECT_EQ(outputs[1].binding, static_cast<uint32_t>(ANIRA_BINDING_POSITION));
        }
#endif
    }
    EXPECT_TRUE(custom_seen);
    // A caller whose header ends before the tail field gets its rows without it.
    anira_plan_slot head = ANIRA_PLAN_SLOT_INIT;
    head.binding = 77;
    count = 1;
    EXPECT_EQ(
        anira_plan_report_slots(report, 0, 1, offsetof(anira_plan_slot, binding), &count, &head),
        ANIRA_INCOMPLETE);
    EXPECT_EQ(head.binding, 77U) << "the tail field stays the caller's";
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
        anira_test::add_passthrough(pipeline);
        const std::array<const anira_model_config*, 1> variants{model.native()};
        ASSERT_EQ(anira_pipeline_add_inference(pipeline,
                                               variants.data(),
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
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
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

namespace {

/// A gate engine that counts the inferences that reached its process, ahead of the gate.
class CountingGate final : public anira_test::GateEngine {
public:
    anira_status process(const anira_engine_ctx& ctx) noexcept override {
        m_entered.fetch_add(1);
        return GateEngine::process(ctx);
    }

    std::atomic<int> m_entered{0};
};

/// Polls `done` every millisecond until it holds or `timeout` ran out; whether it held.
template <typename Done>
bool wait_until(Done&& done, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

// A pool stop does not wait for the jobs that wait for a busy instance. One instance, two
// threads: the gate holds the first inference inside the engine and the second job waits on
// the model's wait list while its thread goes back to the queue. Core::shutdown (the
// library-unload hook's backstop, the one pool stop a live handler allows) tells every thread
// first: the idle one leaves at once, and the holder, done with its inference, gives the
// waiting job up (its chunk fails: zeros, ANIRA_ERROR_ENGINE on the latch, one log record)
// instead of running it, so the stop returns as soon as the holder's inference does. The
// handler is destroyed cleanly after.
TEST(AbiHandler, APoolStopDoesNotWaitForAThreadWaitingForABusyInstance) {
    const Context context(2);
    anira::ModelConfig model = gain_with_custom();
    model.max_instances(1);
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_CUSTOM,
                                                    .provider = ANIRA_PROVIDER_CPU,
                                                    .engine_id = k_custom}};
    CountingGate gate;
    Handler handler(context, model, candidates, {}, gate.engine());
    const DestroyFirst destroy_first(handler, &gate);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    ASSERT_EQ(anira_num_inference_threads(), 2U);
    anira_drain_log();
    RecordCollector collector;

    // Two blocks, two inferences: the first holds the instance behind the closed gate, the
    // second waits on the wait list.
    gate.m_open.store(false);
    for (size_t k = 1; k <= 2; ++k) {
        const std::vector<float> in = ramp(k);
        const std::array<const float*, 1> in_ch{in.data()};
        const anira_tensor pushed = anira_test::planar_f32(in_ch.data(), 1, k_block);
        ASSERT_EQ(anira_handler_push_data(h, &pushed, 0), ANIRA_OK);
    }
    ASSERT_TRUE(
        wait_until([&gate]() { return gate.m_entered.load() == 1; }, std::chrono::seconds(5)));
    // The idle thread polls the queue every 100 us at most: after 200 ms the job waits.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(gate.m_entered.load(), 1) << "one instance, one inference inside the engine";

    std::atomic<bool> stopped{false};
    std::thread stopper([&stopped]() {
        anira::Core::shutdown();
        stopped.store(true);
    });
    // The loop count, a wait-free read: the pool's size is read under the lifecycle lock,
    // which the stop holds while it joins.
    EXPECT_TRUE(
        wait_until([]() { return anira::scheduler::InferenceThread::get_num_loop_active() == 1U; },
                   std::chrono::seconds(2)))
        << "the idle thread did not leave at the stop";
    EXPECT_EQ(gate.m_calls.load(), 0) << "the holder is still inside the engine";
    EXPECT_FALSE(stopped.load()) << "the stop joins the holder, whose inference still runs";

    gate.m_open.store(true);
    stopper.join();
    EXPECT_EQ(anira_num_inference_threads(), 0U);
    EXPECT_EQ(gate.m_calls.load(), 1) << "the waiter's chunk never reached the engine";
    EXPECT_EQ(gate.m_entered.load(), 1);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_ENGINE) << "the given-up chunk failed";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "stopped before an instance", "rt"), 1U);
#endif
}

TEST(AbiHandler, GeneratorSetsItsStaticInputAndPopPulls) {
    constexpr size_t k_hop = 2048;
    const Context context;
    const anira::ModelConfig model = anira_test::generator_model();
    const std::vector<anira_backend_id> none{{.struct_size = sizeof(anira_backend_id),
                                              .engine = ANIRA_ENGINE_CUSTOM,
                                              .provider = ANIRA_PROVIDER_CPU,
                                              .engine_id = anira_test::k_custom}};
    anira_test::SleepingParamFillEngine generator;
    Handler handler(context, model, none, {}, generator.engine());
    const DestroyFirst destroy_first(handler, &generator);
    ASSERT_EQ(handler.prepare(explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 0.0, 10.0)),
              ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    EXPECT_EQ(anira_handler_get_latency(h, 0), k_hop) << "a generator counts from its first pull";

    // Setting a generator's Static input submits nothing: the pulls drive the inferences.
    const std::array<float, 4> params{3.0F, 0.0F, 0.0F, 0.0F};
    const anira_tensor param = anira_test::whole_f32(params.data(), {1, 4});  // the spec's shape
    EXPECT_EQ(anira_handler_set_static_input(h, 0, &param), ANIRA_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(generator.m_calls.load(), 0);

    // A pull delivers the priming block first, then the parameter's fill.
    for (int call = 0; call < 2; ++call) {
        std::vector<float> out(k_hop, -1.0F);
        const std::array<float*, 1> out_ch{out.data()};
        const anira_tensor pulled = anira_test::planar_f32(out_ch.data(), 1, k_hop);
        std::array<size_t, 1> delivered{7};
        const size_t prev = anira_test::available(h);
        ASSERT_EQ(anira_handler_process_multi(h, &param, 1, &pulled, 1, delivered.data()),
                  ANIRA_OK);
        EXPECT_EQ(delivered[0], k_hop) << "call " << call;
        wait_for_block(h, prev);
        expect_all(out, call == 0 ? 0.0F : 3.0F, call == 0 ? "the priming zeros" : "the fill");
    }
    EXPECT_EQ(anira_handler_get_latency(h, 0), k_hop);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // A generator has no slot the single push form takes: input 0 is Static, and the single
    // form of a Static slot is anira_handler_set_static_input.
    const int calls = generator.m_calls.load();
    EXPECT_EQ(anira_handler_push_data(h, &param, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(generator.m_calls.load(), calls) << "a refused call submits nothing";
}
