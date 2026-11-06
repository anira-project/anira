// anira/abi/handler.h: anira_handler_create and anira_handler_prepare. Every contract rule of
// the validator through prepare's anira_error, the three miss policies on a starved block, the
// legacy contract's ZEROS, the structural rules at create, a model that does not load, the
// second prepare, the failed prepare, and the context that outlives its destroy while a
// handler lives.
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <ratio>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "../support/v2_documents.h"
#include "capi/handles.h"
#include "capi/words.h"
#include "fixtures.h"
#include "float_face.h"
#include "handler_support.h"
#include "scheduler/Core.h"
#include "scheduler/LatencyCalculator.h"
#include "scheduler/SessionElement.h"
#include "scheduler/Warmup.h"

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::count_records;
using anira_test::custom_candidates;
using anira_test::DestroyFirst;
using anira_test::expect_all;
using anira_test::expect_same_block;
using anira_test::explicit_contract;
using anira_test::find_record;
using anira_test::gain_with_custom;
using anira_test::GateEngine;
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
    const std::array<const anira_model_config*, 1> variants{model.native()};
    outcome.m_status =
        anira_pipeline_add_inference(pipeline,
                                     variants.data(),
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
    const std::array<float*, 1> ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, n);
    const size_t prev = anira_test::available(handler);
    size_t delivered = 0;
    EXPECT_EQ(anira_handler_process(handler, &io, 0, &io, 0, &delivered), ANIRA_OK)
        << "block " << block_index;
    EXPECT_EQ(delivered, n) << "block " << block_index;
    wait_for_block(handler, prev);
}

void expect_unprepared(anira_handler* handler) {
    std::vector<float> block(k_block, 0.5F);
    const std::array<float*, 1> ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
    EXPECT_EQ(anira_handler_process(handler, &io, 0, &io, 0, nullptr), ANIRA_ERROR_NOT_PREPARED);
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

// The create-time defaults (MEASURED, UNTIL_STABLE) prepare: the warm-up pass measures the
// budget. A MEASURED budget without a warm-up inference (NONE, FIXED 0) is CONFIG naming the
// warm-up; an explicit budget takes every warm-up, UNTIL_STABLE and NONE included.
TEST(AbiPrepare, AMeasuredBudgetNeedsAWarmUpInference) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    const Hard defaults{.host_block_min = 512,
                        .host_block_max = 512,
                        .host_rate = 48000.0,
                        .budget = anira::Budget::measured()};
    EXPECT_EQ(handler.prepare(ContractHandle(defaults), &err), ANIRA_OK) << err.message;

    Hard none = defaults;
    none.warmup = ANIRA_WARMUP_NONE;
    EXPECT_EQ(handler.prepare(ContractHandle(none), &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: a MEASURED budget needs at least one warm-up");
    Hard fixed_zero = defaults;
    fixed_zero.warmup = ANIRA_WARMUP_FIXED;
    fixed_zero.warmup_iterations = 0;
    EXPECT_EQ(handler.prepare(ContractHandle(fixed_zero), &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: a MEASURED budget needs at least one warm-up");

    Hard until_stable = defaults;
    until_stable.budget = std::chrono::milliseconds(5);
    EXPECT_EQ(handler.prepare(ContractHandle(until_stable), &err), ANIRA_OK) << err.message;
    Hard explicit_none = until_stable;
    explicit_none.warmup = ANIRA_WARMUP_NONE;
    EXPECT_EQ(handler.prepare(ContractHandle(explicit_none), &err), ANIRA_OK) << err.message;
}

TEST(AbiPrepare, MissingHostBlocksAreConfig) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    Hard hard{.budget = std::chrono::milliseconds(5)};  // host blocks 0/0/0: legal at create
    hard.warmup = ANIRA_WARMUP_FIXED;
    ContractHandle contract(hard);
    EXPECT_EQ(handler.prepare(contract, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: Hard host blocks missing");
    contract.hard_host_blocks(512, 512, 48000.0);
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

namespace {

/// A mono stream in, two mono streams out (the custom row's pass-through copies the first and
/// zeroes the second), every tensor 512 wide; the second output carries a model latency of 256.
ModelConfig two_stream_outputs_model() {
    ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    model.input(streamed("audio_in"));
    model.output(streamed("audio_out"));
    TensorSpec aux = streamed("aux_out");
    aux.latency(256);
    model.output(aux);
    return model;
}

/// explicit_contract() with the declared stream latency of one output, set through the C entry.
ContractHandle with_latency(const char* canonical, uint32_t samples) {
    ContractHandle contract = explicit_contract();
    EXPECT_EQ(anira_contract_hard_set_latency(contract.native(), canonical, samples), ANIRA_OK);
    return contract;
}

}  // namespace

// The declared stream latency of an output resolves at prepare: a name that is no tensor, an
// input, a Static output and a figure below the model's internal latency are CONFIG. A legal
// declaration replaces the computed figure of its slot alone, and the receive ring is primed
// with it: the first samples popped after prepare are that many zeros.
TEST(AbiPrepare, LatencyRulesAreConfigAtPrepare) {
    const Context context;
    {
        const ModelConfig model = gain_with_custom();
        const std::vector<anira_backend_id> candidates = custom_candidates();
        Handler handler(context, model, candidates);
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(handler.prepare(with_latency("ghost", 1024), &err), ANIRA_ERROR_CONFIG);
        expect_contains(err.message, "contract: the latency of 'ghost' names no tensor");
        EXPECT_EQ(handler.prepare(with_latency("audio_in", 1024), &err), ANIRA_ERROR_CONFIG);
        expect_contains(err.message, "the latency of 'audio_in' is set on an input tensor");
        EXPECT_EQ(handler.prepare(with_latency("gain_out", 1024), &err), ANIRA_ERROR_CONFIG);
        expect_contains(err.message, "the latency of 'gain_out' is set on a Static tensor");
        expect_unprepared(handler.m_handler);
    }

    const ModelConfig model = two_stream_outputs_model();
    const std::vector<anira_backend_id> none{{.struct_size = sizeof(anira_backend_id),
                                              .engine = ANIRA_ENGINE_CUSTOM,
                                              .provider = ANIRA_PROVIDER_CPU,
                                              .engine_id = anira_test::k_custom}};
    Handler handler(context, model, none);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(with_latency("aux_out", 100), &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message,
                    "the latency of 'aux_out' is 100 but the model's internal latency is 256");

    // The computed figures, then the declaration on slot 0 alone.
    ASSERT_EQ(handler.prepare(explicit_contract(), &err), ANIRA_OK) << err.message;
    const uint32_t computed_0 = anira_handler_get_latency(handler.m_handler, 0);
    const uint32_t computed_1 = anira_handler_get_latency(handler.m_handler, 1);
    ASSERT_NE(computed_0, 4096U);
    ASSERT_EQ(handler.prepare(with_latency("audio_out", 4096), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_handler_get_latency(handler.m_handler, 0), 4096U);
    EXPECT_EQ(anira_handler_get_latency(handler.m_handler, 1), computed_1)
        << "an output never named keeps the computed figure";
    uint32_t count = 2;
    std::array<uint32_t, 2> latencies{0, 0};
    ASSERT_EQ(anira_handler_get_latencies(handler.m_handler, &count, latencies.data()), ANIRA_OK);
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(latencies[0], 4096U) << "index-aligned with the output list";
    EXPECT_EQ(latencies[1], computed_1);

    // The priming: the receive ring of slot 0 holds 4096 zeros before any block ran, popped
    // the way a host pops, one declared block (k_block) per call.
    EXPECT_EQ(anira_test::available(handler.m_handler, 0), 4096U);
    std::vector<float> primed(4096, -1.0F);
    for (size_t offset = 0; offset < primed.size(); offset += k_block) {
        const std::array<float*, 1> ch{primed.data() + offset};
        const anira_tensor out = anira_test::planar_f32(ch.data(), 1, k_block);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_pop_data(handler.m_handler, &out, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
    }
    expect_all(primed, 0.0F, "the priming of the declared latency");
}

// The declared host-end domain of a tensor resolves at prepare: a name that is no tensor is
// CONFIG, any domain but host memory is NOT_SUPPORTED naming the tensor (the declaration is
// data in this pre-release), and a Static tensor takes a declaration like a Streamed one. The
// plan report's slot rows carry the declaration on the host's side of every slot.
TEST(AbiPrepare, HostDomainRulesAtPrepare) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_error err = ANIRA_ERROR_INIT;

    ContractHandle ghost = explicit_contract();
    ghost.host_domain("ghost", ANIRA_DOMAIN_HOST);
    EXPECT_EQ(handler.prepare(ghost, &err), ANIRA_ERROR_CONFIG);
    expect_contains(err.message, "contract: the host domain of 'ghost' names no tensor");

    ContractHandle device = explicit_contract();
    device.host_domain("audio_in", ANIRA_DOMAIN_CUDA);
    EXPECT_EQ(handler.prepare(device, &err), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "the host domain of 'audio_in'");
    expect_contains(err.message, "ANIRA_DOMAIN_HOST");

    ContractHandle on_static = explicit_contract();
    on_static.host_domain("gain", ANIRA_DOMAIN_HOST_PINNED);
    EXPECT_EQ(handler.prepare(on_static, &err), ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "the host domain of 'gain'");

    ContractHandle host = explicit_contract();
    host.host_domain("audio_in", ANIRA_DOMAIN_HOST)
        .host_domain("gain", ANIRA_DOMAIN_HOST)
        .host_domain("audio_out", ANIRA_DOMAIN_HOST);
    ASSERT_EQ(handler.prepare(host, &err), ANIRA_OK) << err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    ASSERT_NE(report, nullptr);
    for (const anira_bool inputs : {anira_bool{1}, anira_bool{0}}) {
        uint32_t count = 0;
        ASSERT_EQ(
            anira_plan_report_slots(report, 0, inputs, sizeof(anira_plan_slot), &count, nullptr),
            ANIRA_OK);
        std::vector<anira_plan_slot> rows(count, ANIRA_PLAN_SLOT_INIT);
        ASSERT_EQ(anira_plan_report_slots(report,
                                          0,
                                          inputs,
                                          sizeof(anira_plan_slot),
                                          &count,
                                          rows.data()),
                  ANIRA_OK);
        for (const anira_plan_slot& row : rows) {
            EXPECT_EQ(row.from_domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
            EXPECT_EQ(row.to_domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        }
    }
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
    const std::vector<anira_backend_id> none{{.struct_size = sizeof(anira_backend_id),
                                              .engine = ANIRA_ENGINE_CUSTOM,
                                              .provider = ANIRA_PROVIDER_CPU,
                                              .engine_id = anira_test::k_custom}};
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

/// The calls of a float host over planar tensors, one pointer per channel (InPlace: one tensor
/// on both sides, Separate, Multi: one tensor per slot), and the same calls over packed
/// blocks: one packed mono block per side (Tensor), one tensor per slot (TensorMulti). The
/// Static gain of a multi form is the whole tensor in the spec's shape, [1], under both.
enum class Form { InPlace, Separate, Multi, Tensor, TensorMulti };

constexpr float k_backup = 0.5F;  // what the backup function of ANIRA_MISS_CALLBACK writes

/// The backup function: k_backup into every requested value of every output. The streamed
/// outputs are mono here: one plane under a planar form, one packed block under a packed one; a
/// carried Static output is the whole tensor, rank 1, which holds the stored value when the
/// function is called and is overwritten like the rest. Declared
/// ANIRA_NONBLOCKING (clang refuses to add the attribute through the conversion to
/// anira_miss_fn) and free of assertions: it runs on the driver thread.
anira_status ANIRA_CALL fill_backup(anira_handler* /*handler*/,
                                    const anira_tensor* /*inputs*/,
                                    uint32_t /*num_inputs*/,
                                    const anira_tensor* outputs,
                                    uint32_t num_outputs,
                                    void* /*user_data*/) ANIRA_NONBLOCKING {
    for (uint32_t slot = 0; slot < num_outputs; ++slot) {
        const anira_tensor& output = outputs[slot];
        if (output.ndim == 1) {  // the Static gain: [1], or the empty [0] of a slot left out
            float* values = anira_tensor_data_f32(&output);
            for (int64_t i = 0; values != nullptr && i < output.shape[0]; ++i) {
                values[i] = k_backup;
            }
            continue;
        }
        if (output.shape[1] <= 0) { continue; }
        const bool planar = (output.flags & static_cast<uint32_t>(ANIRA_TENSOR_PLANAR)) != 0U;
        float* samples = planar
                             ? static_cast<float*>(anira_tensor_plane(&output, 0, ANIRA_DTYPE_F32))
                             : anira_tensor_data_f32(&output);
        if (samples == nullptr) { return ANIRA_ERROR_INTERNAL; }
        for (int64_t i = 0; i < output.shape[1]; ++i) { samples[i] = k_backup; }
    }
    return ANIRA_OK;
}

/// What one call delivered: the entry's status and the streamed output's count.
struct Delivered {
    anira_status m_status = ANIRA_OK;
    size_t m_count = 0;
};

/// One block through the form. `static_out` asks for the Static output (gain_out) through the
/// multi form; without it the Static output's pointer is NULL and its request 0.
Delivered call(anira_handler* handler,
               Form form,
               const std::vector<float>& in,
               std::vector<float>& out,
               bool static_out,
               float& gain_out) {
    Delivered delivered;
    switch (form) {
        case Form::InPlace: {
            out = in;
            const std::array<float*, 1> ch{out.data()};
            const anira_tensor io = anira_test::planar_f32(ch.data(), 1, k_block);
            delivered.m_status = anira_handler_process(handler, &io, 0, &io, 0, &delivered.m_count);
            return delivered;
        }
        case Form::Separate: {
            out.assign(k_block, -1.0F);
            const std::array<const float*, 1> i{in.data()};
            const std::array<float*, 1> o{out.data()};
            const anira_tensor in_tensor = anira_test::planar_f32(i.data(), 1, k_block);
            const anira_tensor out_tensor = anira_test::planar_f32(o.data(), 1, k_block);
            delivered.m_status =
                anira_handler_process(handler, &in_tensor, 0, &out_tensor, 0, &delivered.m_count);
            return delivered;
        }
        case Form::Multi: {
            out.assign(k_block, -1.0F);
            const float gain = 1.0F;
            const std::array<const float*, 1> in_ch{in.data()};
            const std::array<float*, 1> out_ch{out.data()};
            const std::array<anira_tensor, 2> ins{anira_test::planar_f32(in_ch.data(), 1, k_block),
                                                  anira_test::whole_f32(&gain, {1})};
            // Without static_out the Static output is an empty tensor with no planes.
            const std::array<anira_tensor, 2> outs{
                anira_test::planar_f32(out_ch.data(), 1, k_block),
                static_out ? anira_test::whole_f32(&gain_out, {1})
                           : anira_test::empty_planar_f32(1)};
            std::array<size_t, 2> counts{7, 7};  // a pure out parameter: written on every return
            delivered.m_status =
                anira_handler_process_multi(handler, ins.data(), 2, outs.data(), 2, counts.data());
            // A missed block delivers 0 on every Streamed slot (the status says what the
            // buffers hold); a carried Static output is its whole tensor on a miss too.
            if (delivered.m_status == ANIRA_MISSED) {
                EXPECT_EQ(counts[0], 0U) << "a missed block delivers 0";
                EXPECT_EQ(counts[1], static_out ? 1U : 0U) << "the stored value, whole";
            }
            delivered.m_count = counts[0];
            return delivered;
        }
        case Form::Tensor: {
            out.assign(k_block, -1.0F);
            const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block)};
            std::vector<float> block = in;
            anira_tensor in_tensor;
            anira_tensor out_tensor;
            anira_tensor_init_host(&in_tensor, block.data(), ANIRA_DTYPE_F32, 2, shape.data());
            anira_tensor_init_host(&out_tensor, out.data(), ANIRA_DTYPE_F32, 2, shape.data());
            delivered.m_count = 7;  // a pure out parameter: written on every return
            delivered.m_status =
                anira_handler_process(handler, &in_tensor, 0, &out_tensor, 0, &delivered.m_count);
            return delivered;
        }
        case Form::TensorMulti: {
            out.assign(k_block, -1.0F);
            const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block)};
            const std::array<int64_t, 1> one{1};   // the gain's spec: one value, rank 1
            const std::array<int64_t, 1> none{0};  // an extent of 0: the slot is left out
            std::vector<float> block = in;
            float gain = 1.0F;
            std::array<anira_tensor, 2> ins{};
            std::array<anira_tensor, 2> outs{};
            anira_tensor_init_host(ins.data(), block.data(), ANIRA_DTYPE_F32, 2, shape.data());
            anira_tensor_init_host(&ins[1], &gain, ANIRA_DTYPE_F32, 1, one.data());
            anira_tensor_init_host(outs.data(), out.data(), ANIRA_DTYPE_F32, 2, shape.data());
            // Without static_out the Static output is an empty tensor with no memory.
            anira_tensor_init_host(&outs[1],
                                   static_out ? &gain_out : nullptr,
                                   ANIRA_DTYPE_F32,
                                   1,
                                   static_out ? one.data() : none.data());
            std::array<size_t, 2> counts{7, 7};
            delivered.m_status =
                anira_handler_process_multi(handler, ins.data(), 2, outs.data(), 2, counts.data());
            // The element count of a carried Static output, on ANIRA_OK and on a miss alike.
            EXPECT_EQ(counts[1], static_out ? 1U : 0U);
            delivered.m_count = counts[0];
            return delivered;
        }
    }
    return delivered;
}

/// The block was delivered in full: ANIRA_OK and the whole request.
void expect_full(const Delivered& delivered, const char* what) {
    EXPECT_EQ(delivered.m_status, ANIRA_OK) << what;
    EXPECT_EQ(delivered.m_count, k_block) << what;
}

/// The block was missed: ANIRA_MISSED, which is a success (ANIRA_FAILED is false), and a
/// delivered count of 0 (the multi form's request array is checked where it is called).
void expect_missed(const Delivered& delivered, const char* what) {
    EXPECT_EQ(delivered.m_status, ANIRA_MISSED) << what;
    EXPECT_TRUE(ANIRA_SUCCEEDED(delivered.m_status)) << what;
    EXPECT_EQ(delivered.m_count, 0U) << what;
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
    GateEngine gate;
    Handler handler(context, model, candidates, {}, gate.engine());
    const DestroyFirst destroy_first(handler, &gate);
    anira::ContractHandle contract = explicit_contract(k_block, k_rate, policy);
    if (policy == ANIRA_MISS_CALLBACK) { contract.hard_miss_fn(&fill_backup, nullptr); }
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    ASSERT_EQ(anira_handler_get_latency(h, 0), k_block) << "one block of priming";
    anira_drain_log();
    RecordCollector collector;
    std::vector<float> out;
    float gain_out = -1.0F;

    // Block 1: the priming zeros.
    size_t prev = anira_test::available(h);
    expect_full(call(h, form, ramp(1), out, static_out, gain_out), "block 1");
    expect_all(out, 0.0F, "block 1");
    wait_for_block(h, prev);

    // Block 2: inference 1's pass-through; inference 2 is now stuck on an inference thread.
    gate.m_open.store(false);
    expect_full(call(h, form, ramp(2), out, static_out, gain_out), "block 2");
    expect_same_block(out, ramp(1), 2);
    if (static_out) { EXPECT_EQ(gain_out, 1.0F) << "block 2: the gain passed through"; }

    // Block 3: a miss under every policy; the status says so, the buffer content is the
    // policy's.
    gain_out = -1.0F;
    expect_missed(call(h, form, ramp(3), out, static_out, gain_out), "block 3");
    switch (policy) {
        case ANIRA_MISS_BYPASS: expect_same_block(out, ramp(3), 3); break;
        case ANIRA_MISS_HOLD_LAST:
            expect_same_block(out, ramp(1), 3);
            if (static_out) { EXPECT_EQ(gain_out, 1.0F) << "the stored value"; }
            break;
        case ANIRA_MISS_ZEROS:
            expect_all(out, 0.0F, "block 3");
            // The miss policies define nothing for a Static output: the stored value.
            if (static_out) { EXPECT_EQ(gain_out, 1.0F) << "the stored value, not zeros"; }
            break;
        case ANIRA_MISS_CALLBACK:
            expect_all(out, k_backup, "block 3");
            if (static_out) { EXPECT_EQ(gain_out, k_backup) << "the whole block is the host's"; }
            break;
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
    expect_full(call(h, form, ramp(4), out, static_out, gain_out), "block 4");
    expect_same_block(out, ramp(3), 4);
    wait_for_available(h, k_block);
    prev = anira_test::available(h);
    expect_full(call(h, form, ramp(5), out, static_out, gain_out), "block 5");
    expect_same_block(out, ramp(4), 5);
    wait_for_block(h, prev);

    if (policy == ANIRA_MISS_BYPASS) {
        // A pop has no input block: the ring's block first, then zeros on the starved pop.
        gate.m_open.store(false);
        const std::vector<float> in = ramp(6);
        const std::array<const float*, 1> in_ch{in.data()};
        const anira_tensor pushed = anira_test::planar_f32(in_ch.data(), 1, k_block);
        EXPECT_EQ(anira_handler_push_data(h, &pushed, 0), ANIRA_OK);
        std::vector<float> popped(k_block, -1.0F);
        const std::array<float*, 1> popped_ch{popped.data()};
        const anira_tensor popped_tensor = anira_test::planar_f32(popped_ch.data(), 1, k_block);
        size_t popped_count = 0;
        EXPECT_EQ(anira_handler_pop_data(h, &popped_tensor, 0, &popped_count), ANIRA_OK);
        EXPECT_EQ(popped_count, k_block);
        expect_same_block(popped, ramp(5), 6);
        popped.assign(k_block, -1.0F);
        EXPECT_EQ(anira_handler_pop_data(h, &popped_tensor, 0, &popped_count), ANIRA_MISSED);
        EXPECT_EQ(popped_count, 0U);
        expect_all(popped, 0.0F, "the starved pop");
    }
    gate.m_open.store(true);  // before the handler is destroyed: the release waits for the
                              // in-flight inference (what destroy_first does either way)
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

// The same sequence through the tensor forms: the policies act on the bytes of the host
// block whatever entry family carried it.
TEST(AbiPrepare, TheTensorFormsFollowTheMissPolicies) {
    run_miss_sequence(ANIRA_MISS_BYPASS, Form::Tensor, false);
    run_miss_sequence(ANIRA_MISS_BYPASS, Form::TensorMulti, false);
    run_miss_sequence(ANIRA_MISS_HOLD_LAST, Form::Tensor, false);
    run_miss_sequence(ANIRA_MISS_HOLD_LAST, Form::TensorMulti, true);
    run_miss_sequence(ANIRA_MISS_ZEROS, Form::TensorMulti, false);
}

// ANIRA_MISS_CALLBACK: the host's function fills the starved block. It is handed the caller's
// own tensors, planar ones over channel pointers and packed ones alike; the block still counts
// as missed and the stream realigns.
TEST(AbiPrepare, CallbackHandsAStarvedBlockToTheHost) {
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::InPlace, false);
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::Separate, false);
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::Multi, true);
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::Tensor, false);
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::TensorMulti, true);
    run_miss_sequence(ANIRA_MISS_CALLBACK, Form::TensorMulti, false);
}

// The request of a tensor form is shape[1] of its output tensors, which no call writes, and
// the counts come back through `delivered`, a pure out parameter. A host that builds its
// tensors once and keeps one delivered array across callbacks must still be asking for a
// block after a miss: the miss leaves zeros in that array, and an entry that read them back as
// requests would make every later call a request for nothing, answered ANIRA_OK.
TEST(AbiPrepare, TensorsBuiltOnceSurviveAMiss) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    GateEngine gate;
    Handler handler(context, model, candidates, {}, gate.engine());
    const DestroyFirst destroy_first(handler, &gate);
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    std::vector<float> in(k_block, 0.0F);
    std::vector<float> out(k_block, -1.0F);
    const float gain = 1.0F;
    const std::array<const float*, 1> in_ch{in.data()};
    const std::array<float*, 1> out_ch{out.data()};
    // Built once, reused by every call below; the Static output is left out (an empty tensor).
    const std::array<anira_tensor, 2> ins{anira_test::planar_f32(in_ch.data(), 1, k_block),
                                          anira_test::whole_f32(&gain, {1})};
    const std::array<anira_tensor, 2> outs{anira_test::planar_f32(out_ch.data(), 1, k_block),
                                           anira_test::empty_planar_f32(1)};
    // The bytes of the output descriptors, as test_HandlerTensor's bytes_of() takes them.
    using DescriptorBytes = std::array<unsigned char, sizeof(outs)>;
    const auto bytes_of_outs = [&outs] {
        DescriptorBytes bytes{};
        std::memcpy(bytes.data(), outs.data(), sizeof(outs));
        return bytes;
    };
    const DescriptorBytes outs_before = bytes_of_outs();
    std::array<size_t, 2> delivered{7, 7};  // never reset by the test
    const auto call_multi = [&](size_t block_index) {
        const std::vector<float> block = ramp(block_index);
        std::ranges::copy(block, in.begin());
        return anira_handler_process_multi(h, ins.data(), 2, outs.data(), 2, delivered.data());
    };

    const size_t prev = anira_test::available(h);
    EXPECT_EQ(call_multi(1), ANIRA_OK) << "the priming zeros";
    EXPECT_EQ(delivered[0], k_block);
    wait_for_block(h, prev);
    gate.m_open.store(false);
    EXPECT_EQ(call_multi(2), ANIRA_OK);
    expect_same_block(out, ramp(1), 2);
    EXPECT_EQ(call_multi(3), ANIRA_MISSED) << "inference 2 is held at the gate";
    EXPECT_EQ(delivered[0], 0U) << "a missed block delivers 0";
    EXPECT_EQ(delivered[1], 0U);
    EXPECT_EQ(bytes_of_outs(), outs_before)
        << "the request is shape[1] of a descriptor no call writes: it survives the miss";
    expect_all(out, 0.0F, "block 3");

    // The same tensors and the same delivered array, untouched by the test: the zeros the miss
    // left in it are not read, and the next call asks for a whole block again.
    gate.m_open.store(true);
    wait_for_available(h, 2 * k_block);
    EXPECT_EQ(call_multi(4), ANIRA_OK);
    EXPECT_EQ(delivered[0], k_block);
    expect_same_block(out, ramp(3), 4);
}

// ============================================================================================
// The legacy contract, the structural rules at create
// ============================================================================================

TEST(AbiPrepare, TheLegacyContractCarriesZeros) {
    {
        ModelConfig gain = ModelConfig::from_json(anira_test::k_simple_gain_v2);
        const std::optional<ContractHandle> legacy = gain.take_legacy_contract();
        if (!legacy.has_value()) { FAIL() << "a version 2 document carries a legacy contract"; }
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
        // The legacy contract prepares a C handler once patched with host blocks: its on_miss
        // is ZEROS, so no BYPASS rule runs.
        const Context context;
        ModelConfig model = ModelConfig::from_json(anira_test::gain_v2_document());
        model.add_model_path(k_custom, "custom-processor");
        std::optional<ContractHandle> legacy = model.take_legacy_contract();
        if (!legacy.has_value()) { FAIL() << "a version 2 document carries a legacy contract"; }
        legacy->hard_host_blocks(512, 512, 48000.0);
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
    const std::vector<anira_backend_id> only_onnx{{.struct_size = sizeof(anira_backend_id),
                                                   .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                                   .provider = ANIRA_PROVIDER_CPU,
                                                   .engine_id = nullptr}};
    const CreateOutcome outcome = try_create(context, model, only_onnx);
    EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
    expect_contains(outcome.m_message, "none of the 1 model entries runs on a candidate backend");
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
        expect_contains(outcome.m_message, "is not added to this pipeline");
    }
    {
        // anira.v2.custom is no exception: without an engine on the pipeline its row is refused
        // like any other custom id's.
        ModelConfig bare;
        bare.add_model_path(k_custom, "model.custom");
        bare.input(streamed("in"));
        bare.output(streamed("out"));
        const CreateOutcome outcome = try_create(context, bare, {});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
        expect_contains(outcome.m_message,
                        "custom engine 'anira.v2.custom' is not added to this pipeline");
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
        const std::vector<anira_backend_id> missing_only{{.struct_size = sizeof(anira_backend_id),
                                                          .engine = static_cast<uint32_t>(*missing),
                                                          .provider = ANIRA_PROVIDER_CPU,
                                                          .engine_id = nullptr}};
        CreateOutcome outcome = try_create(context, with_custom, missing_only);
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
        expect_contains(outcome.m_message, "is not in this build");

        ModelConfig missing_alone;
        missing_alone.add_model_path(*missing, "model.bin");
        missing_alone.input(streamed("in"));
        missing_alone.output(streamed("out"));
        outcome = try_create(context, missing_alone, {});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_CONFIG);
        expect_contains(outcome.m_message,
                        "none of the 1 model entries runs on a candidate backend");
    }

    // anira_pipeline_add_inference's own refusals.
    const ModelConfig model = gain_with_custom();
    const std::array<const anira_model_config*, 1> variants{model.native()};
    const std::array<const anira_model_config*, 2> two{model.native(), model.native()};
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
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, two.data(), 2, nullptr, 0, &err),
              ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "one variant per inference stage");
    // A candidate's provider: its syntax here (a value of the enum, never both a provider of
    // the enum and a provider_id, never an empty provider_id), whether the engine serves it at
    // anira_handler_create, against the context's capabilities.
    const anira_backend_id bad_value{.struct_size = sizeof(anira_backend_id),
                                     .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                     .provider = 0x1000U,
                                     .engine_id = nullptr,
                                     .provider_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &bad_value, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "candidates[0].provider");
    const anira_backend_id both{.struct_size = sizeof(anira_backend_id),
                                .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                .provider = ANIRA_PROVIDER_CUDA,
                                .engine_id = nullptr,
                                .provider_id = "com.example.npu"};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &both, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "the provider_id is set if and only if");
    const anira_backend_id empty_id{.struct_size = sizeof(anira_backend_id),
                                    .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                    .provider = ANIRA_PROVIDER_CUSTOM,
                                    .engine_id = nullptr,
                                    .provider_id = ""};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &empty_id, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "never empty");
    // The pair rule on both axes: a name beside CPU, CUSTOM without an id, NONE (which names
    // no engine, and no provider) and an id beside a built-in engine are refused here.
    const anira_backend_id named_cpu{.struct_size = sizeof(anira_backend_id),
                                     .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                     .provider = ANIRA_PROVIDER_CPU,
                                     .engine_id = nullptr,
                                     .provider_id = "com.example.npu"};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &named_cpu, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "the provider_id is set if and only if");
    const anira_backend_id no_provider{.struct_size = sizeof(anira_backend_id),
                                       .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                       .provider = ANIRA_PROVIDER_NONE,
                                       .engine_id = nullptr,
                                       .provider_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &no_provider, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "names no provider (ANIRA_PROVIDER_NONE)");
    const anira_backend_id nameless{.struct_size = sizeof(anira_backend_id),
                                    .engine = ANIRA_ENGINE_CUSTOM,
                                    .provider = ANIRA_PROVIDER_CPU,
                                    .engine_id = nullptr,
                                    .provider_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &nameless, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "the engine_id is set if and only if");
    const anira_backend_id no_engine{.struct_size = sizeof(anira_backend_id),
                                     .engine = ANIRA_ENGINE_NONE,
                                     .provider = ANIRA_PROVIDER_CPU,
                                     .engine_id = nullptr,
                                     .provider_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &no_engine, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "neither a built-in engine nor ANIRA_ENGINE_CUSTOM");
    const anira_backend_id builtin_id{.struct_size = sizeof(anira_backend_id),
                                      .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                      .provider = ANIRA_PROVIDER_CPU,
                                      .engine_id = "com.example.engine",
                                      .provider_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &builtin_id, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "the engine_id is set if and only if");
    if (const std::vector<anira_engine> engines = anira_test::oracle_engines(); !engines.empty()) {
        // Well formed, and a provider this context's capabilities do not list for the engine
        // (no CUDA on the CI legs): refused at create, naming the entry, the engine and the
        // provider.
        const anira_backend_id cuda{.struct_size = sizeof(anira_backend_id),
                                    .engine = static_cast<uint32_t>(engines[0]),
                                    .provider = ANIRA_PROVIDER_CUDA,
                                    .engine_id = nullptr,
                                    .provider_id = nullptr};
        const CreateOutcome outcome = try_create(context, model, {cuda});
        EXPECT_EQ(outcome.m_status, ANIRA_ERROR_NOT_SUPPORTED);
        expect_contains(outcome.m_message, "does not serve provider 'cuda'");
        expect_contains(outcome.m_message, "capabilities list");
    }
    const anira_backend_id short_row{.struct_size = 4,
                                     .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                     .provider = ANIRA_PROVIDER_CPU,
                                     .engine_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &short_row, 1, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    expect_contains(err.message, "struct_size");
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err),
              ANIRA_OK)
        << err.message;
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err),
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
    const std::vector<anira::BackendId> enabled = anira::enabled_engines();
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
    EXPECT_EQ(anira_test::available(h), latency) << "a fresh stream";
    EXPECT_NE(anira_handler_plan_report(h), nullptr);
    // Two 256-sample blocks: one 512-sample hop, one inference.
    for (size_t k = 1; k <= 2; ++k) {
        std::vector<float> block = ramp(k, 256);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, 256);
        size_t delivered = 0;
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, 256U);
    }
    wait_for_available(h, latency);
}

TEST(AbiPrepare, ThePlanReportIsLoggedAtInfo) {
    // Info passes the process-global level; the records are synchronous ("native"), one per
    // row of the report.
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_INFO);
    RecordCollector collector;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_handler* h = handler.m_handler;
    const uint32_t num_plans = anira_plan_report_num_plans(anira_handler_plan_report(h));
    ASSERT_GE(num_plans, 1U);

    // The records exist only in a build with logging; the prepares above and below run either
    // way (log_report() compiles to nothing under ANIRA_WITH_LOGGING=OFF).
#ifdef ENABLE_LOGGING
    const std::string head = "plan report: plans " + std::to_string(num_plans) +
                             ", input slots 2, output slots 2, selected plan " +
                             std::to_string(anira_handler_get_plan(h));
    EXPECT_EQ(count_records(collector, head.c_str(), "native"), 1U);
    EXPECT_EQ(find_record(collector, head.c_str(), "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_INFO));
    EXPECT_EQ(find_record(collector, head.c_str(), "native").m_group, "anira.capi");
    const anira_plan_report* report = anira_handler_plan_report(h);
    for (uint32_t i = 0; i < num_plans; ++i) {
        const std::string plan = "plan " + std::to_string(i) + ": ";
        EXPECT_EQ(count_records(collector, (plan + "variant 0, engine ").c_str(), "native"), 1U);
        // The edge is the plan's engine's (zero-copy where it binds the slot, a copy where it
        // copies); the dtypes are the spec's and the ring's (none for a Static slot).
        for (const bool is_input : {true, false}) {
            std::array<anira_plan_slot, 2> rows{};
            uint32_t count = 2;  // in: the capacity of rows
            ASSERT_EQ(anira_plan_report_slots(report,
                                              i,
                                              is_input ? 1U : 0U,
                                              sizeof(anira_plan_slot),
                                              &count,
                                              rows.data()),
                      ANIRA_OK);
            ASSERT_EQ(count, 2U);
            const std::array<const char*, 2> names =
                is_input ? std::array<const char*, 2>{"audio_in", "gain"}
                         : std::array<const char*, 2>{"audio_out", "gain_out"};
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const anira_plan_slot& row = rows.at(slot);
                const char* edge = row.edge_class == ANIRA_EDGE_ZERO_COPY ? "zero_copy" : "copy";
                const char* mover = row.moved_by == ANIRA_MOVER_ANIRA    ? "anira"
                                    : row.moved_by == ANIRA_MOVER_ENGINE ? "the engine"
                                                                         : "none";
                const std::string line = plan + (is_input ? "input " : "output ") +
                                         std::to_string(slot) + " '" + names.at(slot) +
                                         "': host -> host, edge " + edge + " (allocate " + edge +
                                         "), moved by " + mover +
                                         ", wait spin_backoff, recipe host, dtype float32, ring " +
                                         (slot == 0 ? "float32" : "none");
                EXPECT_EQ(count_records(collector, line.c_str(), "native"), 1U) << line;
            }
        }
    }
    // The custom row is the selected plan of gain_with_custom(), under the contract's budget.
    const std::string custom = "plan " + std::to_string(anira_handler_get_plan(h)) +
                               ": variant 0, engine " + k_custom +
                               ", provider cpu, budget 5.000 ms";
    EXPECT_EQ(count_records(collector, custom.c_str(), "native"), 1U);
#endif

    // A second prepare logs the new report again.
    ASSERT_EQ(handler.prepare(explicit_contract(256)), ANIRA_OK) << handler.m_err.message;
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_prepare: plan report: ", "native"), 2U);
#endif
}

// The window a slot runs at, in the plan report's rows and the prepare log's slot lines, the same
// in every plan: a window derived from the Time extent ("in"), one derived on a dynamic Time
// axis and pinned to the anchor hop ("out"), a declared flexible one pinned likewise ("env"),
// and no window on a Static slot ("gain").
TEST(AbiPrepare, ThePlanReportShowsTheWindowInUseAndWhetherItWasDerived) {
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_INFO);
    RecordCollector collector;
    const auto stream = [](std::string_view name, int64_t time) {
        TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        spec.axis(0, ANIRA_AXIS_CHANNEL, 1).axis(1, ANIRA_AXIS_TIME, time);
        return spec;
    };
    ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    model.input(stream("in", static_cast<int64_t>(k_block)));
    model.input(anira_test::static_tensor("gain", {1}));
    model.output(stream("out", ANIRA_DYNAMIC));
    model.output(stream("env", ANIRA_DYNAMIC).window(1, ANIRA_UNBOUNDED, 0));
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    const uint32_t num_plans = anira_plan_report_num_plans(report);
    ASSERT_GE(num_plans, 1U);

    struct Expected {
        const char* m_name;
        uint32_t m_derived;
        int64_t m_window;
        const char* m_log;  ///< what the slot line says between the ring and the binding
    };
    constexpr auto k_block_size = static_cast<int64_t>(k_block);
    const std::array<Expected, 2> inputs{
        {{.m_name = "in",
          .m_derived = 1U,
          .m_window = k_block_size,
          .m_log = ", window 512 (derived from the Time extent), hop 512, bound"},
         {.m_name = "gain", .m_derived = 0U, .m_window = 0, .m_log = "ring none, bound"}}};
    const std::array<Expected, 2> outputs{
        {{.m_name = "out",
          .m_derived = 1U,
          .m_window = k_block_size,
          .m_log = ", window 512 (derived, flexible, pinned to the anchor hop), hop 512, bound"},
         {.m_name = "env",
          .m_derived = 0U,
          .m_window = k_block_size,
          .m_log = ", window 512 (flexible, pinned to the anchor hop), hop 512, bound"}}};
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        for (const bool is_input : {true, false}) {
            std::array<anira_plan_slot, 2> rows{};
            uint32_t count = 2;
            ASSERT_EQ(anira_plan_report_slots(report,
                                              plan,
                                              is_input ? 1U : 0U,
                                              sizeof(anira_plan_slot),
                                              &count,
                                              rows.data()),
                      ANIRA_OK);
            ASSERT_EQ(count, 2U);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const Expected& expected = (is_input ? inputs : outputs).at(slot);
                SCOPED_TRACE(expected.m_name);
                const anira_plan_slot& row = rows.at(slot);
                EXPECT_EQ(row.window_derived, expected.m_derived);
                EXPECT_EQ(row.window, expected.m_window);
                EXPECT_EQ(row.hop, expected.m_window) << "no overlap: the hop is the window";
#ifdef ENABLE_LOGGING
                const std::string head = "plan " + std::to_string(plan) + ": " +
                                         (is_input ? "input " : "output ") + std::to_string(slot) +
                                         " '" + expected.m_name + "': ";
                const std::string line = find_record(collector, head.c_str(), "native").m_message;
                EXPECT_NE(line.find(expected.m_log), std::string::npos) << line;
#endif
            }
        }
    }
    // The derivation left the specs alone: the model config still declares no window there.
    const anira::SpecView in = model.input_spec(0);
    EXPECT_EQ(in.window().min, 0);
    EXPECT_EQ(in.window().max, 0);
}

// Every slot row names the spec's dtype and its ring's (0 for a Static slot, which has none),
// and the edge the plan's engine takes with the slot's buffer and who moves its bytes:
// zero-copy, moved by nobody, where it binds the buffer in place (ONNX Runtime, LiteRT and
// TFLite both ways, LibTorch's inputs, a registered engine, which is handed the buffer), a host
// copy with the engine's reason where it copies (LibTorch's outputs and the bundled ExecuTorch
// export's, by the adapter; its inputs, by set_input: gain_edge). The allocate class follows
// the edge on a slot in host memory.
TEST(AbiPrepare, ASlotRowNamesItsDtypesAndTheEnginesEdge) {
    const Context context;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    const uint32_t num_plans = anira_plan_report_num_plans(report);
    std::vector<anira_plan_info> plans(num_plans);
    uint32_t num = num_plans;
    ASSERT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &num, plans.data()),
              ANIRA_OK);
    for (uint32_t i = 0; i < num_plans; ++i) {
        const anira_plan_info& info = plans.at(i);
        for (const bool is_input : {true, false}) {
            std::array<anira_plan_slot, 2> rows{};
            uint32_t count = 2;  // in: the capacity of rows
            ASSERT_EQ(anira_plan_report_slots(report,
                                              i,
                                              is_input ? 1U : 0U,
                                              sizeof(anira_plan_slot),
                                              &count,
                                              rows.data()),
                      ANIRA_OK);
            ASSERT_EQ(count, 2U);
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const anira_plan_slot& row = rows.at(slot);
                SCOPED_TRACE("plan " + std::to_string(i) + (is_input ? " input " : " output ") +
                             std::to_string(slot));
                EXPECT_EQ(row.dtype, ANIRA_DTYPE_F32);
                EXPECT_EQ(row.ring_dtype, slot == 0 ? ANIRA_DTYPE_F32 : 0U)
                    << "a Static slot has no ring";
                const anira_test::SlotEdge edge = anira_test::gain_edge(info.engine, is_input);
                EXPECT_EQ(row.edge_class, edge.m_class);
                EXPECT_EQ(row.allocate_class, row.edge_class);
                EXPECT_EQ(row.moved_by, edge.m_moved_by);
                EXPECT_EQ(row.reason != nullptr, edge.m_class != ANIRA_EDGE_ZERO_COPY);
            }
        }
    }
}

// The contract's edge cost over the built-in engines of the build on the CPU path (the gain
// files): under ANIRA_EDGE_COPY every plan stays (none moves bytes across a bus); under
// ANIRA_EDGE_ZERO_COPY a plan whose engine copies a slot (gain_edge: LibTorch's outputs, the
// memory-planned ExecuTorch export) leaves the table and every other one stays, in candidate
// order, each slot zero-copy and moved by nobody. A build whose engines all copy has no plan
// left: ANIRA_ERROR_EDGE_UNREACHABLE, the message naming the threshold. Skipped in a build
// without a built-in engine (the sanitizer trees).
TEST(AbiPrepare, TheEdgeCostKeepsThePlansWithinIt) {
    const Context context;
    const ModelConfig model = gain_with_custom(false);
    const std::vector<anira_backend_id> candidates = anira_test::engine_candidates();
    if (candidates.empty()) { GTEST_SKIP() << "no built-in engine in this build"; }
    std::vector<uint32_t> bound;  // the engines whose gain plans copy no slot
    for (const anira_backend_id& candidate : candidates) {
        if (anira_test::gain_edge(candidate.engine, true).m_class == ANIRA_EDGE_ZERO_COPY &&
            anira_test::gain_edge(candidate.engine, false).m_class == ANIRA_EDGE_ZERO_COPY) {
            bound.push_back(candidate.engine);
        }
    }
    const auto plan_engines = [](const anira_handler* h) {
        const anira_plan_report* report = anira_handler_plan_report(h);
        std::vector<anira_plan_info> plans(anira_plan_report_num_plans(report));
        auto count = static_cast<uint32_t>(plans.size());
        EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, plans.data()),
                  ANIRA_OK);
        std::vector<uint32_t> engines;
        engines.reserve(plans.size());
        for (const anira_plan_info& plan : plans) { engines.push_back(plan.engine); }
        return engines;
    };
    Handler handler(context, model, candidates);
    ContractHandle contract = explicit_contract();
    contract.edge_cost(ANIRA_EDGE_COPY);
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(plan_engines(handler.m_handler).size(), candidates.size());

    contract.edge_cost(ANIRA_EDGE_ZERO_COPY);
    const anira_status status = handler.prepare(contract);
    if (bound.empty()) {
        EXPECT_EQ(status, ANIRA_ERROR_EDGE_UNREACHABLE);
        EXPECT_NE(std::strstr(handler.m_err.message,
                              "no plan is within the contract's edge cost (zero_copy at most)"),
                  nullptr)
            << handler.m_err.message;
        return;
    }
    ASSERT_EQ(status, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(plan_engines(handler.m_handler), bound);
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    for (uint32_t plan = 0; plan < bound.size(); ++plan) {
        for (const uint32_t inputs : {1U, 0U}) {
            std::array<anira_plan_slot, 2> rows{};
            uint32_t count = 2;
            ASSERT_EQ(anira_plan_report_slots(report,
                                              plan,
                                              inputs,
                                              sizeof(anira_plan_slot),
                                              &count,
                                              rows.data()),
                      ANIRA_OK);
            for (uint32_t slot = 0; slot < count; ++slot) {
                EXPECT_EQ(rows.at(slot).edge_class, ANIRA_EDGE_ZERO_COPY);
                EXPECT_EQ(rows.at(slot).moved_by, ANIRA_MOVER_NONE);
            }
        }
    }
}

// The movers of a slot row are bits, and the log lines spell every set of them: a slot both
// anira and the runtime move (a memory-planned ExecuTorch output on a device delegate) says so.
TEST(AbiPrepare, TheLogSpellsEverySetOfMovers) {
    EXPECT_STREQ(anira::capi::movers_word(ANIRA_MOVER_NONE), "none");
    EXPECT_STREQ(anira::capi::movers_word(ANIRA_MOVER_ANIRA), "anira");
    EXPECT_STREQ(anira::capi::movers_word(ANIRA_MOVER_ENGINE), "the engine");
    EXPECT_STREQ(anira::capi::movers_word(ANIRA_MOVER_ANIRA | ANIRA_MOVER_ENGINE),
                 "anira and the engine");
    EXPECT_STREQ(anira::capi::movers_word(4U), "unknown");
}

TEST(AbiPrepare, ThePlanReportIsNotLoggedUnderAnErrorLevel) {
    const Context context;  // ANIRA_LOG_ERROR
    RecordCollector collector;
    const ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(count_records(collector, "anira_handler_prepare: plan", "native"), 0U);
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

// ============================================================================================
// The warm-up pass
// ============================================================================================

namespace {

/// How a ScriptedEngine takes its time: a busy-wait bounds it from above as well (the case
/// asserts an upper bound), a sleep from below alone.
enum class Wait : uint8_t { Busy, Sleep };

/// A registered engine under anira.v2.custom whose call k takes `figure_ms(k)` and then passes
/// through; it counts its calls, its prepares and unprepares, and keeps the entry count of its
/// last prepare's record. Declared before the Handler that adds it.
class ScriptedEngine {
public:
    ScriptedEngine(std::function<double(int)> figure_ms, Wait wait)
        : m_figure_ms(std::move(figure_ms)), m_wait(wait) {
        anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
        desc.user_data = this;
        desc.process = &process_slot;
        desc.prepare = &prepare_slot;
        desc.unprepare = &unprepare_slot;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_custom_engine_create(k_custom, &desc, &m_engine, &err), ANIRA_OK)
            << err.message;
    }
    ~ScriptedEngine() { anira_custom_engine_destroy(m_engine); }
    ScriptedEngine(const ScriptedEngine&) = delete;
    ScriptedEngine& operator=(const ScriptedEngine&) = delete;
    ScriptedEngine(ScriptedEngine&&) = delete;
    ScriptedEngine& operator=(ScriptedEngine&&) = delete;

    const anira_custom_engine* engine() const noexcept { return m_engine; }

    std::atomic<int> m_calls{0};
    std::atomic<int> m_prepares{0};
    std::atomic<int> m_unprepares{0};
    std::atomic<uint32_t> m_last_entries{0};

private:
    static anira_status ANIRA_CALL process_slot(const anira_engine_ctx* ctx,
                                                void* /*prepared*/,
                                                void* user_data) noexcept {
        auto* self = static_cast<ScriptedEngine*>(user_data);
        const int call = self->m_calls.fetch_add(1);
        const auto take = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(self->m_figure_ms(call)));
        if (self->m_wait == Wait::Sleep) {
            std::this_thread::sleep_for(take);
        } else {
            const auto until = std::chrono::steady_clock::now() + take;
            while (std::chrono::steady_clock::now() < until) {}
        }
        return anira_test::passthrough(*ctx);
    }
    static anira_status ANIRA_CALL prepare_slot(const anira_prepare_info* info,
                                                void* /*loaded*/,
                                                void* user_data,
                                                void** out_prepared) {
        auto* self = static_cast<ScriptedEngine*>(user_data);
        self->m_prepares.fetch_add(1);
        self->m_last_entries.store(info->num_entries);
        *out_prepared = nullptr;
        return ANIRA_OK;
    }
    static void ANIRA_CALL unprepare_slot(void* /*prepared*/, void* user_data) {
        static_cast<ScriptedEngine*>(user_data)->m_unprepares.fetch_add(1);
    }

    std::function<double(int)> m_figure_ms;
    Wait m_wait;
    anira_custom_engine* m_engine = nullptr;
};

/// The create-time defaults, a MEASURED budget and the UNTIL_STABLE warm-up, at host blocks of
/// `block` at `rate`.
ContractHandle measured_contract(uint32_t block = k_block, double rate = k_rate) {
    return ContractHandle(Hard{.host_block_min = static_cast<double>(block),
                               .host_block_max = static_cast<double>(block),
                               .host_rate = rate,
                               .budget = anira::Budget::measured()});
}

/// The plan row `plan` of a prepared handler's report.
const anira_plan_info& plan_row(const Handler& handler, uint32_t plan = 0) {
    return handler.m_handler->m_report.m_plans.at(plan);
}

/// Every row of a prepared handler: a positive real-time factor over at least one parallel
/// inference, the verdict the thresholds give its load (0.75, 1), or UNMEASURED with no pass; a
/// measured row's median at or below its figure.
void expect_rows_follow_the_thresholds(const Handler& handler) {
    for (const anira_plan_info& row : handler.m_handler->m_report.m_plans) {
        EXPECT_GT(row.rt_factor, 0.0);
        EXPECT_GE(row.rt_parallel, 1U);
        if (row.rt_verdict == ANIRA_RT_UNMEASURED) {
            EXPECT_EQ(row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_NONE));
            EXPECT_EQ(row.warmup_iterations, 0U);
            continue;
        }
        const double load = row.rt_factor / static_cast<double>(row.rt_parallel);
        const anira_rt_verdict verdict = load >= 1.0    ? ANIRA_RT_NOT_REALTIME
                                         : load >= 0.75 ? ANIRA_RT_MARGINAL
                                                        : ANIRA_RT_COMFORTABLE;
        EXPECT_EQ(row.rt_verdict, static_cast<uint32_t>(verdict)) << "load " << load;
        EXPECT_GT(row.measured_ms, 0.0);
        EXPECT_LE(row.median_ms, row.measured_ms);
        EXPECT_GT(row.warmup_iterations, 0U);
    }
}

/// `blocks` ramp blocks through a prepared mono handler, each waited for its inference: the
/// stream, a pure function of the calls.
std::vector<float> waited_stream(anira_handler* handler, size_t blocks) {
    std::vector<float> stream;
    for (size_t index = 0; index < blocks; ++index) {
        std::vector<float> block = ramp(index);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
        EXPECT_EQ(anira_handler_process_wait(handler, &io, 0, &io, 0, nullptr, ANIRA_WAIT_FOREVER),
                  ANIRA_OK)
            << "block " << index;
        stream.insert(stream.end(), block.begin(), block.end());
    }
    return stream;
}

}  // namespace

// UNTIL_STABLE leaves the cold start out: a first inference of 10 ms and 1 ms after it stop no
// earlier than the fifth (the first is in no window of four that agrees), and the plan's budget,
// the window's maximum, is not the first time. The engine runs nothing else at prepare.
TEST(AbiWarmup, UntilStableLeavesTheColdStartOut) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int call) { return call == 0 ? 10.0 : 1.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_GE(engine.m_calls.load(), 5);
    EXPECT_LE(engine.m_calls.load(), 32);
    EXPECT_GE(plan_row(handler).budget_ms, 1.0);
    EXPECT_LT(plan_row(handler).budget_ms, 10.0) << "the cold start is in no window";
    EXPECT_EQ(plan_row(handler).warmup_iterations, static_cast<uint32_t>(engine.m_calls.load()));
    EXPECT_DOUBLE_EQ(plan_row(handler).measured_ms, plan_row(handler).budget_ms);
    expect_rows_follow_the_thresholds(handler);
}

// Times that never agree within 20 % (1 and 5 ms alternating) stop the pass at its cap of 32,
// and the budget is the last four's maximum, a 5 ms one.
TEST(AbiWarmup, AnUnsettledPassStopsAtItsCap) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_WARNING);
    RecordCollector collector;
    const ScriptedEngine engine([](int call) { return call % 2 == 0 ? 1.0 : 5.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), 32);
    EXPECT_GE(plan_row(handler).budget_ms, 5.0);
    EXPECT_EQ(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_CAPPED));
    EXPECT_EQ(plan_row(handler).warmup_iterations, 32U);
    expect_rows_follow_the_thresholds(handler);
#ifdef ENABLE_LOGGING
    const char* capped = "the warm-up did not settle in 32 inferences";
    EXPECT_EQ(count_records(collector, capped, "native"), 1U);
    EXPECT_EQ(find_record(collector, capped, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_WARNING));
#endif
}

// FIXED n runs n inferences of every plan, whatever their times, under either budget: the
// custom plan of the gain model among the built-in ones runs its own n, and nothing at load.
TEST(AbiWarmup, FixedRunsItsCountPerPlan) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 0.0; }, Wait::Busy);
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, gain_with_custom(), candidates, {}, engine.engine());
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 5.0, 3)),
              ANIRA_OK)
        << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), 3);
    EXPECT_DOUBLE_EQ(plan_row(handler).budget_ms, 5.0) << "an explicit budget stays";
    for (const anira_plan_info& row : handler.m_handler->m_report.m_plans) {
        EXPECT_EQ(row.warmup_iterations, 3U);
        EXPECT_EQ(row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_FIXED));
    }
    expect_rows_follow_the_thresholds(handler);

    ContractHandle measured = measured_contract();
    measured.hard_warmup(ANIRA_WARMUP_FIXED, 4);
    ASSERT_EQ(handler.prepare(measured), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), 3 + 4);
    for (const anira_plan_info& row : handler.m_handler->m_report.m_plans) {
        EXPECT_GT(row.budget_ms, 0.0) << "every plan was measured";
    }
}

// A MEASURED budget is the slowest plan's: the session is sized for the maximum of the plans'
// figures, and each plan's row carries its own.
TEST(AbiWarmup, TheSessionRunsTheSlowestPlansFigure) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 0.0; }, Wait::Busy);
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, gain_with_custom(), candidates, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    double slowest = 0.0;
    for (const anira_plan_info& row : handler.m_handler->m_report.m_plans) {
        EXPECT_GT(row.budget_ms, 0.0);
        slowest = std::max(slowest, row.budget_ms);
    }
    const std::shared_ptr<anira::SessionElement> session =
        anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    EXPECT_DOUBLE_EQ(session->m_budget_ms, slowest);
    EXPECT_GE(engine.m_calls.load(), 4) << "UNTIL_STABLE runs four at least";
}

// A model too slow for the stream prepares: an inference of 30 ms at a hop of 512 at 48 kHz
// (10.7 ms) on one instance is sized as an infeasible explicit budget is (for one host block's
// burst), its budget what the pass measured; a Hard prepare never answers ANIRA_ERROR_BUDGET.
// The stream runs, its misses filled under on_miss.
TEST(AbiWarmup, AModelTooSlowForTheStreamStillPrepares) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_WARNING);
    RecordCollector collector;
    const ScriptedEngine engine([](int /*call*/) { return 30.0; }, Wait::Sleep);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_GE(plan_row(handler).budget_ms, 30.0);
    // 30 ms over a hop period of 512 / 48 kHz = 10.67 ms: 2.8 hop periods at least, on the one
    // instance the model allows.
    EXPECT_EQ(plan_row(handler).rt_verdict, static_cast<uint32_t>(ANIRA_RT_NOT_REALTIME));
    EXPECT_GE(plan_row(handler).rt_factor, 2.8);
    EXPECT_EQ(plan_row(handler).rt_parallel, 1U);
    expect_rows_follow_the_thresholds(handler);
#ifdef ENABLE_LOGGING
    const char* verdict = "NOT real-time on this machine";
    EXPECT_EQ(count_records(collector, verdict, "native"), 1U);
    EXPECT_EQ(find_record(collector, verdict, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_WARNING));
    EXPECT_EQ(count_records(collector, "the pool cannot keep up", "native"), 1U);
#endif
    const std::shared_ptr<anira::SessionElement> session =
        anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    EXPECT_DOUBLE_EQ(session->m_budget_ms, plan_row(handler).budget_ms);
    std::vector<float> block(k_block, 0.5F);
    const std::array<float*, 1> ptrs{block.data()};
    const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_block);
    const anira_status status = anira_handler_process(handler.m_handler, &io, 0, &io, 0, nullptr);
    EXPECT_TRUE(status == ANIRA_OK || status == ANIRA_MISSED) << status;
}

// Under an explicit budget the pass's timings are only reported: the session is sized for the
// declared figure, which the plan's row keeps, however slow the pass was.
TEST(AbiWarmup, AnExplicitBudgetIsNotReplacedByTheMeasurement) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_WARNING);
    RecordCollector collector;
    const ScriptedEngine engine([](int /*call*/) { return 30.0; }, Wait::Sleep);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 1.0, 4)),
              ANIRA_OK)
        << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), 4);
    EXPECT_DOUBLE_EQ(plan_row(handler).budget_ms, 1.0);
    const std::shared_ptr<anira::SessionElement> session =
        anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    EXPECT_DOUBLE_EQ(session->m_budget_ms, 1.0);
    // The factor and the verdict are the measurement's, not the declared budget's.
    EXPECT_GE(plan_row(handler).measured_ms, 30.0);
    EXPECT_GE(plan_row(handler).rt_factor, 2.8);
    EXPECT_EQ(plan_row(handler).rt_verdict, static_cast<uint32_t>(ANIRA_RT_NOT_REALTIME));
    expect_rows_follow_the_thresholds(handler);
#ifdef ENABLE_LOGGING
    const char* above = "above the explicit budget 1.000 ms; the latency is sized for 1.000 ms";
    EXPECT_EQ(count_records(collector, above, "native"), 1U);
    EXPECT_EQ(find_record(collector, above, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_WARNING));
#endif
}

// An inference of 0.85 hop periods on one instance is above the comfortable load by
// construction: marginal, or not real-time when the machine adds to it, never comfortable.
TEST(AbiWarmup, AMarginalLoadIsNeverComfortable) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    constexpr double k_hop_period_ms = 1000.0 * k_block / k_rate;
    const ScriptedEngine engine([](int /*call*/) { return 0.85 * k_hop_period_ms; }, Wait::Sleep);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_GE(plan_row(handler).rt_factor, 0.85);
    EXPECT_NE(plan_row(handler).rt_verdict, static_cast<uint32_t>(ANIRA_RT_COMFORTABLE));
    EXPECT_NE(plan_row(handler).rt_verdict, static_cast<uint32_t>(ANIRA_RT_UNMEASURED));
    expect_rows_follow_the_thresholds(handler);
}

// An explicit budget without a warm-up measures nothing: the row is UNMEASURED, its pass fields
// zero, its real-time factor the declared budget's (5 ms over a hop period of 512 / 48 kHz on
// the model's two instances), and the prepare logs it at Info.
TEST(AbiWarmup, AnExplicitBudgetWithoutAWarmUpIsUnmeasured) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_INFO);
    RecordCollector collector;
    Handler handler(context, anira_test::mono_stream_model());
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_plan_info& row = plan_row(handler);
    EXPECT_EQ(row.rt_verdict, static_cast<uint32_t>(ANIRA_RT_UNMEASURED));
    EXPECT_EQ(row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_NONE));
    EXPECT_EQ(row.warmup_iterations, 0U);
    EXPECT_DOUBLE_EQ(row.measured_ms, 0.0);
    EXPECT_DOUBLE_EQ(row.median_ms, 0.0);
    EXPECT_DOUBLE_EQ(row.rt_factor, 5.0 * k_rate / (1000.0 * k_block));
    EXPECT_EQ(row.rt_parallel, 2U);
    expect_rows_follow_the_thresholds(handler);
#ifdef ENABLE_LOGGING
    const char* unmeasured = "no warm-up; real-time factor 0.47 of 2 parallel inferences";
    EXPECT_EQ(count_records(collector, unmeasured, "native"), 1U);
    EXPECT_EQ(find_record(collector, unmeasured, "native").m_level,
              static_cast<uint32_t>(ANIRA_LOG_INFO));
    EXPECT_EQ(count_records(collector, "budget 5.000 ms, explicit: unmeasured; latency ", "native"),
              1U);
#endif
}

// The pass's jobs are ordinary jobs: a context without a pool whose host runs no inference thread
// refuses a prepare that warms up (INVALID_STATE naming the running thread it needs) and takes
// one that does not (an explicit budget without a warm-up).
TEST(AbiWarmup, AWarmUpWithoutARunningThreadIsInvalidState) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(0);
    Handler handler(context, anira_test::mono_stream_model());
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(measured_contract(), &err), ANIRA_ERROR_INVALID_STATE);
    expect_contains(err.message, "a warm-up needs a running inference thread");
    expect_unprepared(handler.m_handler);
    EXPECT_EQ(handler.prepare(explicit_contract(), &err), ANIRA_OK) << err.message;
}

// A failing inference of the pass fails the prepare with ENGINE naming the plan, and the handler
// is unprepared.
TEST(AbiWarmup, AFailedWarmUpInferenceFailsThePrepare) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    GateEngine engine;
    engine.m_fail.store(true);
    Handler handler(context, anira_test::mono_stream_model(), {}, {}, engine.engine());
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(measured_contract(), &err), ANIRA_ERROR_ENGINE);
    expect_contains(err.message, "the warm-up inference of plan 0 failed");
    expect_unprepared(handler.m_handler);
    engine.m_fail.store(false);
    EXPECT_EQ(handler.prepare(measured_contract(), &err), ANIRA_OK) << err.message;
}

// The user code a prepare runs sees the record of the pass's sizing. A figure that keeps the
// entry count leaves it prepared once; one that moves it (an inference of 20 ms on four instances
// at a hop of 512 at 48 kHz needs more structs than one clock step) unprepares it and prepares it
// again with the final record.
TEST(AbiWarmup, AnEntryCountThatMovesPreparesTheUserCodeAgain) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(4);
    {
        const ScriptedEngine fast([](int /*call*/) { return 0.0; }, Wait::Busy);
        Handler handler(context, anira_test::mono_stream_model(k_block, 4), {}, {}, fast.engine());
        ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
        EXPECT_EQ(fast.m_prepares.load(), 1);
        EXPECT_EQ(fast.m_unprepares.load(), 0);
        EXPECT_EQ(fast.m_last_entries.load(), anira_handler_num_entries(handler.m_handler));
    }
    const ScriptedEngine slow([](int /*call*/) { return 20.0; }, Wait::Sleep);
    Handler handler(context, anira_test::mono_stream_model(k_block, 4), {}, {}, slow.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    const anira::capi::HardContract& hard = *handler.m_handler->m_contract.hard();
    const std::shared_ptr<anira::SessionElement> session =
        anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    const anira::LatencyCalculator at_pass(handler.m_handler->m_derived,
                                           hard,
                                           session->m_num_instances,
                                           static_cast<double>(anira::clock_step_ns()) / 1e6);
    const anira::LatencyCalculator at_figure(handler.m_handler->m_derived,
                                             hard,
                                             session->m_num_instances,
                                             session->m_budget_ms);
    ASSERT_NE(at_pass.get_num_structs(), at_figure.get_num_structs())
        << "the geometry no longer moves the entry count";
    EXPECT_EQ(slow.m_prepares.load(), 2);
    EXPECT_EQ(slow.m_unprepares.load(), 1);
    EXPECT_EQ(slow.m_last_entries.load(), anira_handler_num_entries(handler.m_handler));
}

// The pass leaves no trace in the stream: after a FIXED 3 prepare the stream equals, sample for
// sample, that of a twin prepared without a warm-up, on the custom pass-through and on the
// built-in gain of the first engine of this build.
TEST(AbiWarmup, ThePassLeavesNoTraceInTheStream) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ContractHandle warmed = explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 5.0, 3);
    const ContractHandle unwarmed =
        explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 5.0, 0);
    {
        Handler twin_a(context, anira_test::mono_stream_model());
        Handler twin_b(context, anira_test::mono_stream_model());
        ASSERT_EQ(twin_a.prepare(warmed), ANIRA_OK) << twin_a.m_err.message;
        ASSERT_EQ(twin_b.prepare(unwarmed), ANIRA_OK) << twin_b.m_err.message;
        EXPECT_EQ(waited_stream(twin_a.m_handler, 8), waited_stream(twin_b.m_handler, 8));
    }
    const std::vector<anira_engine> engines = anira_test::oracle_engines();
    if (engines.empty()) { GTEST_SKIP() << "no built-in engine in this build"; }
    const std::vector<anira_backend_id> candidates{
        {.struct_size = sizeof(anira_backend_id),
         .engine = static_cast<uint32_t>(engines.front()),
         .provider = ANIRA_PROVIDER_CPU,
         .engine_id = nullptr}};
    Handler twin_a(context, gain_with_custom(false), candidates);
    Handler twin_b(context, gain_with_custom(false), candidates);
    ASSERT_EQ(twin_a.prepare(warmed), ANIRA_OK) << twin_a.m_err.message;
    ASSERT_EQ(twin_b.prepare(unwarmed), ANIRA_OK) << twin_b.m_err.message;
    float gain_value = 0.5F;
    const std::array<int64_t, 1> gain_shape{1};
    anira_tensor gain_tensor{};
    anira_tensor_init_host(&gain_tensor, &gain_value, ANIRA_DTYPE_F32, 1, gain_shape.data());
    ASSERT_EQ(anira_handler_set_static_input(twin_a.m_handler, 1, &gain_tensor), ANIRA_OK);
    ASSERT_EQ(anira_handler_set_static_input(twin_b.m_handler, 1, &gain_tensor), ANIRA_OK);
    const std::vector<float> stream = waited_stream(twin_a.m_handler, 8);
    EXPECT_EQ(stream, waited_stream(twin_b.m_handler, 8));
    EXPECT_TRUE(std::ranges::any_of(stream, [](float x) { return x != 0.0F; }));
}

// ============================================================================================
// The model's figure (the core's figure index)
// ============================================================================================

namespace {

/// An inference hook that does nothing: what puts a processor's identity into the figure's key.
anira_status ANIRA_CALL quiet_hook(const anira_processor_ctx* /*ctx*/,
                                   void* /*prepared*/,
                                   void* /*user_data*/) noexcept {
    return ANIRA_OK;
}

/// A processor that fills before_inference, with `id` (NULL: none).
anira_processor_desc hooked_processor(const char* id) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.before_inference = &quiet_hook;
    desc.id = id;
    return desc;
}

}  // namespace

// Two handlers of one model and one contract on one context: the second reuses the figure the
// first measured, runs no pass (the engine's count does not grow during its prepare) and reports
// the same budget and latency, its outcome REUSED with the first's counts.
TEST(AbiFigure, TwoHandlersOfOneModelShareItsFigure) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler first(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(first.prepare(measured_contract()), ANIRA_OK) << first.m_err.message;
    const int calls = engine.m_calls.load();
    ASSERT_GE(calls, 4);
    Handler second(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(second.prepare(measured_contract()), ANIRA_OK) << second.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "no pass";
    const anira_plan_info& measured_row = plan_row(first);
    const anira_plan_info& reused_row = plan_row(second);
    EXPECT_NE(measured_row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_EQ(reused_row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_EQ(reused_row.warmup_iterations, measured_row.warmup_iterations);
    EXPECT_DOUBLE_EQ(reused_row.measured_ms, measured_row.measured_ms);
    EXPECT_DOUBLE_EQ(reused_row.budget_ms, measured_row.budget_ms);
    EXPECT_EQ(anira_handler_get_latency(second.m_handler, 0),
              anira_handler_get_latency(first.m_handler, 0));
    expect_rows_follow_the_thresholds(second);
}

// A re-prepare of one handler at another host block of the same window reuses the model's
// figure: no pass, REUSED, the same inference time.
TEST(AbiFigure, ARePrepareReusesTheFigure) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    const int calls = engine.m_calls.load();
    const double figure = plan_row(handler).measured_ms;
    ASSERT_EQ(handler.prepare(measured_contract(k_block / 2)), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "no pass";
    EXPECT_EQ(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_DOUBLE_EQ(plan_row(handler).measured_ms, figure);
    EXPECT_DOUBLE_EQ(plan_row(handler).budget_ms, figure);
}

// The figure is the model's, the factor the contract's: the same model at 48 and at 96 kHz has
// one inference time and a real-time factor twice as large, and an inference of 6 ms on one
// instance at a hop of 512 is not real-time at 96 kHz (5.3 ms per hop) by construction.
TEST(AbiFigure, AnotherSampleRateKeepsTheTimeAndDoublesTheFactor) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 6.0; }, Wait::Sleep);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract(k_block, 48000.0)), ANIRA_OK)
        << handler.m_err.message;
    const anira_plan_info at_48k = plan_row(handler);
    ASSERT_EQ(handler.prepare(measured_contract(k_block, 96000.0)), ANIRA_OK)
        << handler.m_err.message;
    const anira_plan_info at_96k = plan_row(handler);
    EXPECT_EQ(at_96k.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_DOUBLE_EQ(at_96k.measured_ms, at_48k.measured_ms);
    EXPECT_NEAR(at_96k.rt_factor, 2.0 * at_48k.rt_factor, 1e-9);
    EXPECT_EQ(at_96k.rt_verdict, static_cast<uint32_t>(ANIRA_RT_NOT_REALTIME));
}

// A figure is reused only by a warm-up it answers: a FIXED 2 figure is measured again under
// UNTIL_STABLE, which replaces it; the UNTIL_STABLE figure then answers both UNTIL_STABLE and
// FIXED n.
TEST(AbiFigure, AWeakerFigureIsMeasuredAgain) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ContractHandle fixed_two = measured_contract();
    fixed_two.hard_warmup(ANIRA_WARMUP_FIXED, 2);
    ASSERT_EQ(handler.prepare(fixed_two), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), 2);
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_GE(engine.m_calls.load(), 2 + 4) << "measured again";
    EXPECT_NE(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    const int calls = engine.m_calls.load();
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(fixed_two), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "the UNTIL_STABLE figure answers both";
    EXPECT_EQ(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
}

// Two prepares of one model on two threads at once: one measures, the other waits for its
// figure and reuses it; the engine ran one pass.
TEST(AbiFigure, TwoPreparesAtOnceMeasureTheModelOnce) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context(4);
    const ScriptedEngine engine([](int /*call*/) { return 2.0; }, Wait::Sleep);
    Handler first(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    Handler second(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    anira_status first_status = ANIRA_ERROR_INTERNAL;
    anira_status second_status = ANIRA_ERROR_INTERNAL;
    std::thread one([&] { first_status = first.prepare(measured_contract()); });
    std::thread two([&] { second_status = second.prepare(measured_contract()); });
    one.join();
    two.join();
    ASSERT_EQ(first_status, ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second_status, ANIRA_OK) << second.m_err.message;
    const bool first_reused =
        plan_row(first).warmup_outcome == static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED);
    const bool second_reused =
        plan_row(second).warmup_outcome == static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED);
    EXPECT_NE(first_reused, second_reused) << "exactly one measured";
    const anira_plan_info& measured_row = first_reused ? plan_row(second) : plan_row(first);
    EXPECT_EQ(engine.m_calls.load(), static_cast<int>(measured_row.warmup_iterations))
        << "one pass";
    EXPECT_DOUBLE_EQ(plan_row(first).budget_ms, plan_row(second).budget_ms);
}

// The processor's inference hooks are in the figure's key: two registrations of a processor
// with a before_inference and one id share the figure; without an id each registration measures
// its own; a processor without inference hooks never enters the key.
TEST(AbiFigure, TheProcessorsInferenceHooksAreInTheKey) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    const auto prepared_calls = [&](const anira_processor_desc& processor) {
        const std::array<anira_processor_desc, 1> processors{processor};
        Handler handler(context,
                        anira_test::mono_stream_model(k_block, 1),
                        {},
                        processors,
                        engine.engine());
        EXPECT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
        return engine.m_calls.load();
    };
    const int first = prepared_calls(hooked_processor("org.anira.test.window"));
    ASSERT_GE(first, 4);
    EXPECT_EQ(prepared_calls(hooked_processor("org.anira.test.window")), first)
        << "one id: one figure";
    const int without_id = prepared_calls(hooked_processor(nullptr));
    EXPECT_GT(without_id, first) << "no id: the registration's own";
    EXPECT_GT(prepared_calls(hooked_processor(nullptr)), without_id)
        << "another registration without an id measures again";
    const anira_processor_desc unhooked = ANIRA_PROCESSOR_DESC_INIT;
    const int plain = prepared_calls(unhooked);
    EXPECT_GT(plain, without_id) << "no hooks: the model's key alone, measured once";
    EXPECT_EQ(prepared_calls(unhooked), plain) << "another processor without hooks reuses it";
}

// The figures live as long as the core's state: once the last context and handler are gone and
// the core is shut down and freed (anira_release_core_if_idle), the next prepare of the model
// measures again.
TEST(AbiFigure, TheFiguresGoWithTheCore) {
    ASSERT_TRUE(anira_test::fresh_core());
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    int calls = 0;
    {
        const Context context;
        Handler handler(context,
                        anira_test::mono_stream_model(k_block, 1),
                        {},
                        {},
                        engine.engine());
        ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
        calls = engine.m_calls.load();
    }
    ASSERT_EQ(anira_shutdown(), ANIRA_OK);
    ASSERT_NE(anira_release_core_if_idle(), 0U) << "the core's state is freed";
    const Context context;
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_GT(engine.m_calls.load(), calls) << "measured again";
    EXPECT_NE(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
}

// Every pass feeds the index, under either budget kind (D16): an explicit budget with
// UNTIL_STABLE measures the model, and a measured budget then reuses it, its budget the figure.
TEST(AbiFigure, AnExplicitPassFeedsALaterMeasuredBudget) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ContractHandle declared = measured_contract();
    declared.hard_budget(ANIRA_BUDGET_EXPLICIT, std::chrono::milliseconds(5));
    ASSERT_EQ(handler.prepare(declared), ANIRA_OK) << handler.m_err.message;
    const int calls = engine.m_calls.load();
    ASSERT_GE(calls, 4);
    const double figure = plan_row(handler).measured_ms;
    EXPECT_DOUBLE_EQ(plan_row(handler).budget_ms, 5.0);
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "no pass";
    EXPECT_EQ(plan_row(handler).warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_DOUBLE_EQ(plan_row(handler).measured_ms, figure);
    EXPECT_DOUBLE_EQ(plan_row(handler).budget_ms, figure);
}

// The reverse (D16): a measured budget's figure is reused by an explicit budget, which keeps
// its declared budget and latency and reports the figure's factor and verdict.
TEST(AbiFigure, AnExplicitBudgetReusesTheFigureAndKeepsItsSizing) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler explicit_twin(context,
                          anira_test::mono_stream_model(k_block, 1),
                          {},
                          {},
                          engine.engine());
    ASSERT_EQ(explicit_twin.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 5.0)),
              ANIRA_OK)
        << explicit_twin.m_err.message;
    const uint32_t declared_latency = anira_handler_get_latency(explicit_twin.m_handler, 0);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    const int calls = engine.m_calls.load();
    const anira_plan_info measured_row = plan_row(handler);
    ContractHandle declared = measured_contract();
    declared.hard_budget(ANIRA_BUDGET_EXPLICIT, std::chrono::milliseconds(5));
    ASSERT_EQ(handler.prepare(declared), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "no pass";
    const anira_plan_info& row = plan_row(handler);
    EXPECT_EQ(row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_REUSED));
    EXPECT_DOUBLE_EQ(row.budget_ms, 5.0);
    EXPECT_EQ(anira_handler_get_latency(handler.m_handler, 0), declared_latency);
    EXPECT_DOUBLE_EQ(row.rt_factor, measured_row.rt_factor);
    EXPECT_EQ(row.rt_verdict, measured_row.rt_verdict);
}

// NONE is the opt-out (D17): an explicit budget without a warm-up neither measures nor reads
// the model's figure; its row is UNMEASURED, its factor the declared budget's.
TEST(AbiFigure, AnExplicitBudgetWithoutAWarmUpIgnoresTheFigure) {
    ASSERT_TRUE(anira_test::fresh_core());
    const Context context;
    const ScriptedEngine engine([](int /*call*/) { return 1.0; }, Wait::Busy);
    Handler handler(context, anira_test::mono_stream_model(k_block, 1), {}, {}, engine.engine());
    ASSERT_EQ(handler.prepare(measured_contract()), ANIRA_OK) << handler.m_err.message;
    const int calls = engine.m_calls.load();
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS, 0.0, 5.0, 0)),
              ANIRA_OK)
        << handler.m_err.message;
    EXPECT_EQ(engine.m_calls.load(), calls) << "no pass";
    const anira_plan_info& row = plan_row(handler);
    EXPECT_EQ(row.rt_verdict, static_cast<uint32_t>(ANIRA_RT_UNMEASURED));
    EXPECT_EQ(row.warmup_outcome, static_cast<uint32_t>(ANIRA_WARMUP_OUTCOME_NONE));
    EXPECT_DOUBLE_EQ(row.measured_ms, 0.0);
    EXPECT_DOUBLE_EQ(row.rt_factor, 5.0 * k_rate / (1000.0 * k_block));
}
