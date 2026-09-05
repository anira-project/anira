// anira/abi/handler.h: anira_handler_create and anira_handler_prepare. Every ex-translator
// contract rule through prepare's anira_error, the three miss policies on a starved block, the
// legacy contract's ZEROS, the structural rules at create, a model that does not load, the
// second prepare, the failed prepare, and the context that outlives its destroy while a
// handler lives.
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "../support/v2_documents.h"
#include "fixtures.h"
#include "handler_support.h"

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::attach_processor;
using anira_test::Context;
using anira_test::custom_candidates;
using anira_test::expect_all;
using anira_test::expect_same_block;
using anira_test::explicit_contract;
using anira_test::gain_with_custom;
using anira_test::GateBackend;
using anira_test::generator_model;
using anira_test::Handler;
using anira_test::k_block;
using anira_test::k_custom;
using anira_test::k_rate;
using anira_test::mismatched_channels_model;
using anira_test::ramp;
using anira_test::RecordCollector;
using anira_test::wait_for_available;
using anira_test::wait_for_block;

void expect_contains(const std::string& message, std::string_view needle) {
    EXPECT_NE(message.find(needle), std::string::npos)
        << "expected \"" << needle << "\" in \"" << message << "\"";
}

/// A streamed float32 spec [batch 1, channel <channels>, time <time>] with a fixed window.
TensorSpec streamed(std::string_view name, int64_t time = 512, int64_t channels = 1) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, channels)
        .axis(2, ANIRA_AXIS_TIME, time)
        .window(time, time, 0);
    return spec;
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

/// anira_handler_create through the C entries, for a create that is expected to fail.
struct CreateOutcome {
    anira_status m_status = ANIRA_OK;
    std::string m_message;
};

CreateOutcome try_create(const Context& context,
                         const ModelConfig& model,
                         const std::vector<anira_backend_id>& candidates) {
    CreateOutcome outcome;
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    EXPECT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    const anira_model_config* variants[] = {model.native()};
    outcome.m_status =
        anira_pipeline_add_inference(pipeline,
                                     variants,
                                     1,
                                     candidates.empty() ? nullptr : candidates.data(),
                                     static_cast<uint32_t>(candidates.size()),
                                     &err);
    if (outcome.m_status == ANIRA_OK) {
        anira_handler* handler = nullptr;
        outcome.m_status = anira_handler_create(context.m_context, pipeline, &handler, &err);
        anira_handler_destroy(handler);
    }
    outcome.m_message = err.message;
    anira_pipeline_destroy(pipeline);
    return outcome;
}

/// One in-place gain block through a prepared handler, waited.
void run_waited_block(anira_handler* handler, size_t block_index, size_t n = k_block) {
    std::vector<float> block = ramp(block_index, n);
    float* ptrs[1] = {block.data()};
    const size_t prev = anira_handler_get_available_samples(handler, 0, 0);
    EXPECT_EQ(anira_handler_process(handler, ptrs, n, 0), n) << "block " << block_index;
    wait_for_block(handler, prev);
}

void expect_unprepared(anira_handler* handler) {
    std::vector<float> block(k_block, 0.5F);
    float* ptrs[1] = {block.data()};
    EXPECT_EQ(anira_handler_process(handler, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_ERROR_NOT_PREPARED);
}

}  // namespace

// ============================================================================================
// The contract rules through prepare
// ============================================================================================

TEST(AbiPrepare, AnAsyncContractIsNotSupported) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(ContractHandle(anira::Async{}), &err), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "contract: an Async contract");
    expect_unprepared(handler.m_handler);
}

TEST(AbiPrepare, MeasuredBudgetAndUntilStableWarmupAreNotSupportedUntilTheEstimator) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    Hard defaults;  // the create-time defaults: MEASURED, UNTIL_STABLE
    defaults.block_min = 512;
    defaults.block_max = 512;
    defaults.rate = 48000.0;
    EXPECT_EQ(handler.prepare(ContractHandle(defaults), &err), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "contract: a MEASURED budget");

    Hard until_stable = defaults;
    until_stable.budget = ANIRA_BUDGET_EXPLICIT;
    until_stable.budget_value = std::chrono::milliseconds(5);
    until_stable.warmup = ANIRA_WARMUP_UNTIL_STABLE;
    EXPECT_EQ(handler.prepare(ContractHandle(until_stable), &err), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "contract: UNTIL_STABLE warmup");

    Hard none = until_stable;
    none.warmup = ANIRA_WARMUP_NONE;
    EXPECT_EQ(handler.prepare(ContractHandle(none), &err), ANIRA_OK) << err.message;
}

TEST(AbiPrepare, MissingGeometryIsConfig) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    Hard hard;  // geometry 0/0/0: legal at create
    hard.budget = ANIRA_BUDGET_EXPLICIT;
    hard.budget_value = std::chrono::milliseconds(5);
    hard.warmup = ANIRA_WARMUP_FIXED;
    ContractHandle contract(hard);
    EXPECT_EQ(handler.prepare(contract, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: Hard geometry missing");
    contract.hard_geometry(512, 512, 48000.0);
    EXPECT_EQ(handler.prepare(contract, &err), ANIRA_OK) << err.message;
}

TEST(AbiPrepare, RingDtypeRulesAreConfigAtPrepare) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    ContractHandle ghost = explicit_contract();
    ghost.hard_ring_dtype("ghost", ANIRA_DTYPE_F32);
    EXPECT_EQ(handler.prepare(ghost, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: the ring dtype of 'ghost' names no tensor");

    ContractHandle on_static = explicit_contract();
    on_static.hard_ring_dtype("gain", ANIRA_DTYPE_F32);
    EXPECT_EQ(handler.prepare(on_static, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "the ring dtype of 'gain'");
    expect_contains(err.message, "Static tensor");

    ContractHandle converted = explicit_contract();
    converted.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16);
    EXPECT_EQ(handler.prepare(converted, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "the ring dtype of 'audio_in'");
    expect_contains(err.message, "nothing converts");

    ContractHandle f32 = explicit_contract();
    f32.hard_ring_dtype("audio_in", ANIRA_DTYPE_F32).hard_ring_dtype("audio_out", ANIRA_DTYPE_F32);
    ASSERT_EQ(handler.prepare(f32, &err), ANIRA_OK) << err.message;
    run_waited_block(handler.m_handler, 1);
}

TEST(AbiPrepare, HoldLastAndZerosAreAccepted) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    EXPECT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_HOLD_LAST)), ANIRA_OK)
        << handler.m_err.message;
    {
        // The contract snapshot is the handler's: the handle dies before processing.
        const ContractHandle zeros = explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS);
        EXPECT_EQ(handler.prepare(zeros), ANIRA_OK) << handler.m_err.message;
    }
    run_waited_block(handler.m_handler, 1);
    run_waited_block(handler.m_handler, 2);
}

TEST(AbiPrepare, BypassIsRefusedOnAGenerator) {
    const Context context;
    const ModelConfig model = generator_model();
    const std::vector<anira_backend_id> none{
        {sizeof(anira_backend_id), ANIRA_ENGINE_NONE, ANIRA_PROVIDER_DEFAULT, nullptr}};
    Handler handler(context, model, none);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(explicit_contract(2048, k_rate, ANIRA_MISS_BYPASS, 0.0, 10.0), &err),
              ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "on_miss BYPASS");
    expect_contains(err.message, "anchor is the output");
    expect_contains(err.message, "'audio_out'");
    EXPECT_EQ(
        handler.prepare(explicit_contract(2048, k_rate, ANIRA_MISS_HOLD_LAST, 0.0, 10.0), &err),
        ANIRA_OK)
        << err.message;
    EXPECT_EQ(handler.prepare(explicit_contract(2048, k_rate, ANIRA_MISS_ZEROS, 0.0, 10.0), &err),
              ANIRA_OK)
        << err.message;
}

TEST(AbiPrepare, BypassIsRefusedWhenNoAnchoredInputHasTheOutputsChannelCount) {
    const Context context;
    const ModelConfig model = mismatched_channels_model();
    Handler handler(context, model);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_BYPASS), &err),
              ANIRA_ERROR_CONFIG);
    expect_contains(
        err.message,
        "on_miss BYPASS: output 'out' has 2 channels but the anchored input 'in' has 1");
    EXPECT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS), &err), ANIRA_OK)
        << err.message;
}

// ============================================================================================
// The three policies on a starved block
// ============================================================================================

namespace {

enum class Form { InPlace, Separate, Multi };

/// One block through the form. `static_out` asks for the Static output (gain_out) through the
/// multi form; without it the Static output's pointer is NULL and its request 0.
size_t call(anira_handler* handler,
            Form form,
            const std::vector<float>& in,
            std::vector<float>& out,
            bool static_out,
            float& gain_out) {
    switch (form) {
        case Form::InPlace: {
            out = in;
            float* ch[1] = {out.data()};
            return anira_handler_process(handler, ch, k_block, 0);
        }
        case Form::Separate: {
            out.assign(k_block, -1.0F);
            const float* i[1] = {in.data()};
            float* o[1] = {out.data()};
            return anira_handler_process_separate(handler, i, k_block, o, k_block, 0);
        }
        case Form::Multi: {
            out.assign(k_block, -1.0F);
            float gain = 1.0F;
            const float* in_ch[1] = {in.data()};
            const float* gain_ch[1] = {&gain};
            const float* const* ins[2] = {in_ch, gain_ch};
            size_t num_in[2] = {k_block, 1};
            float* out_ch[1] = {out.data()};
            float* gain_out_ch[1] = {&gain_out};
            float* const* outs[2] = {out_ch, static_out ? gain_out_ch : nullptr};
            size_t num_out[2] = {k_block, static_out ? 1U : 0U};
            EXPECT_EQ(anira_handler_process_multi(handler, ins, num_in, outs, num_out), ANIRA_OK);
            return num_out[0];
        }
    }
    return 0;
}

/// Block b (1-based) carries ramp(b) and, once aligned, delivers ramp(b - 1); block 1 the
/// priming zeros. The gate closes after block 1: block 2 still delivers inference 1's
/// pass-through, block 3 is the miss the policy governs, and once the gate opens the
/// catch-up discards the late block so the stream stays time-aligned.
void run_miss_sequence(anira_miss_policy policy, Form form, bool static_out) {
    // The S7 record is a Warning: the context's level must let it through.
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_DEBUG);
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, policy)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    ASSERT_EQ(anira_handler_get_latency(h, 0), k_block) << "one block of priming";
    anira_drain_log();
    RecordCollector collector;
    GateBackend gate(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, gate));
    std::vector<float> out;
    float gain_out = -1.0F;

    // Block 1: the priming zeros.
    size_t prev = anira_handler_get_available_samples(h, 0, 0);
    EXPECT_EQ(call(h, form, ramp(1), out, static_out, gain_out), k_block);
    expect_all(out, 0.0F, "block 1");
    wait_for_block(h, prev);

    // Block 2: inference 1's pass-through; inference 2 is now stuck on an inference thread.
    gate.m_open.store(false);
    EXPECT_EQ(call(h, form, ramp(2), out, static_out, gain_out), k_block);
    expect_same_block(out, ramp(1), 2);
    if (static_out) { EXPECT_EQ(gain_out, 1.0F) << "block 2: the gain passed through"; }

    // Block 3: a miss under every policy; the buffer content is the policy's.
    gain_out = -1.0F;
    EXPECT_EQ(call(h, form, ramp(3), out, static_out, gain_out), 0U) << "the count says miss";
    switch (policy) {
        case ANIRA_MISS_BYPASS: expect_same_block(out, ramp(3), 3); break;
        case ANIRA_MISS_HOLD_LAST:
            expect_same_block(out, ramp(1), 3);
            if (static_out) { EXPECT_EQ(gain_out, 1.0F) << "the last completed value"; }
            break;
        case ANIRA_MISS_ZEROS: expect_all(out, 0.0F, "block 3"); break;
        default: FAIL() << "unknown policy";
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << "a miss records nothing";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "Missing samples", "rt"), 1U);
#endif

    // The gate opens: inferences 2 and 3 are collected, the catch-up discards ramp(2).
    gate.m_open.store(true);
    wait_for_available(h, 2 * k_block);
    EXPECT_EQ(call(h, form, ramp(4), out, static_out, gain_out), k_block);
    expect_same_block(out, ramp(3), 4);
    wait_for_available(h, k_block);
    prev = anira_handler_get_available_samples(h, 0, 0);
    EXPECT_EQ(call(h, form, ramp(5), out, static_out, gain_out), k_block);
    expect_same_block(out, ramp(4), 5);
    wait_for_block(h, prev);

    if (policy == ANIRA_MISS_BYPASS) {
        // A pop has no input block: the ring's block first, then zeros on the starved pop.
        gate.m_open.store(false);
        const std::vector<float> in = ramp(6);
        const float* in_ch[1] = {in.data()};
        EXPECT_EQ(anira_handler_push_data(h, in_ch, k_block, 0), ANIRA_OK);
        std::vector<float> popped(k_block, -1.0F);
        float* popped_ch[1] = {popped.data()};
        EXPECT_EQ(anira_handler_pop_data(h, popped_ch, k_block, 0), k_block);
        expect_same_block(popped, ramp(5), 6);
        popped.assign(k_block, -1.0F);
        EXPECT_EQ(anira_handler_pop_data(h, popped_ch, k_block, 0), 0U);
        expect_all(popped, 0.0F, "the starved pop");
    }
    gate.m_open.store(true);  // before the handler is destroyed: the release waits for the
                              // in-flight inference
}

}  // namespace

TEST(AbiPrepare, BypassDeliversTheInputOnAStarvedBlock) {
    run_miss_sequence(ANIRA_MISS_BYPASS, Form::InPlace, false);
    run_miss_sequence(ANIRA_MISS_BYPASS, Form::Separate, false);
    run_miss_sequence(ANIRA_MISS_BYPASS, Form::Multi, false);
}

TEST(AbiPrepare, HoldLastRepeatsTheLastDeliveredBlock) {
    run_miss_sequence(ANIRA_MISS_HOLD_LAST, Form::InPlace, false);
    run_miss_sequence(ANIRA_MISS_HOLD_LAST, Form::Multi, true);
}

TEST(AbiPrepare, ZerosZeroFillsAStarvedBlock) {
    run_miss_sequence(ANIRA_MISS_ZEROS, Form::InPlace, false);
}

// ============================================================================================
// The legacy contract, the structural rules at create
// ============================================================================================

TEST(AbiPrepare, TheLegacyContractCarriesZeros) {
    {
        ModelConfig gain = ModelConfig::from_json(anira_test::k_simple_gain_v2);
        std::optional<ContractHandle> legacy = gain.take_legacy_contract();
        ASSERT_TRUE(legacy.has_value());
        ASSERT_NE(legacy->native()->hard(), nullptr);
        EXPECT_EQ(legacy->native()->hard()->m_on_miss, ANIRA_MISS_ZEROS);
    }
    {
        const ContractHandle upgraded = ContractHandle::from_json(anira_test::k_simple_gain_v2);
        ASSERT_NE(upgraded.native()->hard(), nullptr);
        EXPECT_EQ(upgraded.native()->hard()->m_on_miss, ANIRA_MISS_ZEROS);
    }
    {
        const ContractHandle gain = ContractHandle::from_file(k_gain_contract_json);
        ASSERT_NE(gain.native()->hard(), nullptr);
        EXPECT_EQ(gain.native()->hard()->m_on_miss, ANIRA_MISS_BYPASS)
            << "a 3.x file without the key keeps the default";
        const ContractHandle encoder =
            ContractHandle::from_file(k_rave_funk_drum_encoder_contract_json);
        ASSERT_NE(encoder.native()->hard(), nullptr);
        EXPECT_EQ(encoder.native()->hard()->m_on_miss, ANIRA_MISS_ZEROS);
        const ContractHandle decoder =
            ContractHandle::from_file(k_rave_funk_drum_decoder_contract_json);
        ASSERT_NE(decoder.native()->hard(), nullptr);
        EXPECT_EQ(decoder.native()->hard()->m_on_miss, ANIRA_MISS_ZEROS);
        const ContractHandle whole = ContractHandle::from_file(k_rave_funk_drum_contract_json);
        ASSERT_NE(whole.native()->hard(), nullptr);
        EXPECT_EQ(whole.native()->hard()->m_on_miss, ANIRA_MISS_BYPASS);
    }
    {
        // The legacy contract prepares a C handler once patched with a geometry: its on_miss
        // is ZEROS, so no BYPASS rule runs.
        const Context context;
        ModelConfig model = ModelConfig::from_json(anira_test::gain_v2_document());
        model.add_model_path(k_custom, "custom-processor");
        std::optional<ContractHandle> legacy = model.take_legacy_contract();
        ASSERT_TRUE(legacy.has_value());
        legacy->hard_geometry(512, 512, 48000.0);
        const std::vector<anira_backend_id> candidates = custom_candidates();
        Handler handler(context, model, candidates);
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(handler.prepare(*legacy, &err), ANIRA_OK) << err.message;
    }
}

TEST(AbiPrepare, ZeroPlansIsConfigAtCreate) {
    const Context context;
    ModelConfig model;
    model.add_model_path(k_custom, "model.custom");
    model.input(streamed("in"));
    model.output(streamed("out"));
    const std::vector<anira_backend_id> only_onnx{
        {sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DEFAULT, nullptr}};
    const CreateOutcome outcome = try_create(context, model, only_onnx);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "none of the 1 model entries names a candidate engine");
}

TEST(AbiPrepare, StructuralRulesAtCreate) {
    const Context context;
    {
        ModelConfig no_input;
        no_input.add_model_path(k_custom, "model.custom");
        no_input.output(streamed("out"));
        const CreateOutcome outcome = try_create(context, no_input, {});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
        expect_contains(outcome.m_message, "no input tensor");
    }
    {
        ModelConfig other;
        other.add_model_path("de.example.other", "model.other");
        other.input(streamed("in"));
        other.output(streamed("out"));
        const CreateOutcome outcome = try_create(context, other, {});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
        expect_contains(outcome.m_message, "has no 2.x adapter");
    }
    const std::optional<anira_engine> missing = missing_engine();
    if (missing.has_value()) {
        ModelConfig with_custom;
        with_custom.add_model_path(*missing, "model.bin");
        with_custom.add_model_path(k_custom, "model.custom");
        with_custom.input(streamed("in"));
        with_custom.output(streamed("out"));
        {
            // The default set skips the absent engine's entry: the custom row alone.
            Handler handler(context, with_custom);
            ASSERT_NE(handler.m_handler, nullptr);
            ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
            EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(handler.m_handler)),
                      1U);
        }
        const std::vector<anira_backend_id> missing_only{{sizeof(anira_backend_id),
                                                          static_cast<uint32_t>(*missing),
                                                          ANIRA_PROVIDER_DEFAULT,
                                                          nullptr}};
        CreateOutcome outcome = try_create(context, with_custom, missing_only);
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
        expect_contains(outcome.m_message, "is not in this build");

        ModelConfig missing_alone;
        missing_alone.add_model_path(*missing, "model.bin");
        missing_alone.input(streamed("in"));
        missing_alone.output(streamed("out"));
        outcome = try_create(context, missing_alone, {});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
        expect_contains(outcome.m_message, "none of the 1 model entries names a candidate engine");
    }

    // anira_pipeline_add_inference's own refusals.
    const ModelConfig model = gain_with_custom();
    const anira_model_config* variants[] = {model.native()};
    const anira_model_config* two[] = {model.native(), model.native()};
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    {
        anira_handler* handler = nullptr;
        EXPECT_EQ(anira_handler_create(context.m_context, pipeline, &handler, &err),
                  ANIRA_ERROR_CONFIG);
        expect_contains(err.message, "no inference stage");
        EXPECT_EQ(handler, nullptr);
    }
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, two, 2, nullptr, 0, &err),
              ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "one variant per inference stage");
    const anira_backend_id cuda{sizeof(anira_backend_id),
                                ANIRA_ENGINE_ONNXRUNTIME,
                                ANIRA_PROVIDER_CUDA,
                                nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants, 1, &cuda, 1, &err),
              ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "Host-only");
    const anira_backend_id short_row{4, ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_DEFAULT, nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants, 1, &short_row, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "struct_size");
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants, 1, nullptr, 0, &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants, 1, nullptr, 0, &err),
              ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "a second inference stage");
    anira_pipeline_destroy(pipeline);
}

// ============================================================================================
// Loading, the second prepare, the failed prepare
// ============================================================================================

#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME)
TEST(AbiPrepare, AModelThatDoesNotLoadIsReportedAtPrepare) {
    const Context context;
    const std::vector<anira::BackendId> enabled = anira::enabled_backends();
    ASSERT_FALSE(enabled.empty());
    const auto first_engine = static_cast<anira_engine>(enabled.front().engine);
    {
        ModelConfig model;
        model.add_model_path(first_engine, "/nonexistent/anira-test.model");
        model.input(streamed("in"));
        model.output(streamed("out"));
        Handler handler(context, model);
        ASSERT_NE(handler.m_handler, nullptr);
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(handler.prepare(explicit_contract(), &err), ANIRA_ERROR_NO_SUCH_FILE)
            << err.message;
        expect_unprepared(handler.m_handler);
        EXPECT_EQ(anira::Core::get_num_sessions(), 0);
        EXPECT_EQ(anira_num_inference_threads(), 0U) << "the create-session rollback";
    }
#ifdef USE_ONNXRUNTIME
    {
        const std::filesystem::path model_path =
            std::filesystem::temp_directory_path() / "anira_abi_prepare_unloadable.onnx";
        {
            std::ofstream file(model_path, std::ios::binary);
            file << "definitely not a model";
        }
        ModelConfig model;
        model.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, model_path);
        model.input(streamed("in", 2048));
        model.output(streamed("out", 2048));
        {
            Handler handler(context, model);
            ASSERT_NE(handler.m_handler, nullptr);
            anira_error err = ANIRA_ERROR_INIT;
            EXPECT_EQ(
                handler.prepare(explicit_contract(2048, k_rate, ANIRA_MISS_BYPASS, 0.0, 42.66),
                                &err),
                ANIRA_ERROR_MODEL_LOAD)
                << err.message;
            expect_contains(err.message, "onnxruntime");
            expect_unprepared(handler.m_handler);
            EXPECT_EQ(anira::Core::get_num_sessions(), 0);
            EXPECT_EQ(anira_num_inference_threads(), 0U);
        }
        std::filesystem::remove(model_path);
    }
#endif
}
#endif

TEST(AbiPrepare, ASecondPrepareReplacesTheSessionWhole) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract(512)), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    run_waited_block(h, 1);
    run_waited_block(h, 2);

    ASSERT_EQ(handler.prepare(explicit_contract(256)), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(anira::Core::get_num_sessions(), 1);
    const size_t latency = anira_handler_get_latency(h, 0);
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0), latency) << "a fresh stream";
    EXPECT_NE(anira_handler_plan_report(h), nullptr);
    // Two 256-sample blocks: one 512-sample hop, one inference.
    for (size_t k = 1; k <= 2; ++k) {
        std::vector<float> block = ramp(k, 256);
        float* ptrs[1] = {block.data()};
        EXPECT_EQ(anira_handler_process(h, ptrs, 256, 0), 256U);
    }
    wait_for_available(h, latency);
}

TEST(AbiPrepare, AFailedPrepareLeavesTheHandlerUnprepared) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    run_waited_block(h, 1);

    EXPECT_EQ(handler.prepare(ContractHandle(anira::Async{})), ANIRA_ERROR_NOT_SUPPORTED);
    expect_unprepared(h);
    EXPECT_EQ(anira_handler_plan_report(h), nullptr);
    EXPECT_EQ(anira::Core::get_num_sessions(), 0);

    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    run_waited_block(h, 2);
}

TEST(AbiPrepare, TheContextOutlivesItsDestroyWhileAHandlerLives) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_context_config* config = nullptr;
    anira_context* context = nullptr;
    ASSERT_EQ(anira_context_config_create(&config, &err), ANIRA_OK) << err.message;
    ASSERT_EQ(anira_context_config_set_threads(config, 2, ANIRA_WAIT_SPIN_BACKOFF), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_level(config, ANIRA_LOG_ERROR), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10), ANIRA_OK);
    ASSERT_EQ(anira_context_create(config, &context, &err), ANIRA_OK) << err.message;

    const ModelConfig model = gain_with_custom();
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
    anira_handler* h = nullptr;
    ASSERT_EQ(anira_handler_create(context, pipeline, &h, &err), ANIRA_OK) << err.message;
    anira_pipeline_destroy(pipeline);

    // The user's context and config go while the handler is unprepared.
    anira_context_destroy(context);
    anira_context_config_destroy(config);
    EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE) << "a handler is a user of the core";
    EXPECT_EQ(anira_release_core_if_idle(), 0U);
    EXPECT_EQ(anira::Core::get_num_handlers(), 1U);

    {
        const ContractHandle contract = explicit_contract();
        ASSERT_EQ(anira_handler_prepare(h, contract.native(), &err), ANIRA_OK) << err.message;
    }
    run_waited_block(h, 1);
    run_waited_block(h, 2);
    EXPECT_EQ(anira_shutdown(), ANIRA_ERROR_INVALID_STATE) << "a session lives too";
    for (size_t k = 3; k <= 6; ++k) { run_waited_block(h, k); }
    EXPECT_EQ(anira_num_inference_threads(), 2U) << "the handler's copy of the config";

    anira_handler_destroy(h);
    EXPECT_EQ(anira_shutdown(), ANIRA_OK);
    EXPECT_EQ(anira::Core::get_num_contexts(), 0U);
    EXPECT_EQ(anira::Core::get_num_handlers(), 0U);
}
