// anira/abi/handler.h: anira_handler_create and anira_handler_prepare. Every ex-translator
// contract rule through prepare's anira_error, the three miss policies on a starved block, the
// legacy contract's ZEROS, the structural rules at create, a model that does not load, the
// second prepare, the failed prepare, and the context that outlives its destroy while a
// handler lives.
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
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
#include "float_face.h"
#include "handler_support.h"

namespace {

using anira::ContractHandle;
using anira::Hard;
using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::attach_processor;
using anira_test::Context;
using anira_test::count_records;
using anira_test::custom_candidates;
using anira_test::DestroyFirst;
using anira_test::expect_all;
using anira_test::expect_same_block;
using anira_test::explicit_contract;
using anira_test::find_record;
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
            EXPECT_EQ(row.domain_in, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
            EXPECT_EQ(row.domain_out, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
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
                                              .engine = ANIRA_ENGINE_NONE,
                                              .provider = ANIRA_PROVIDER_DEFAULT,
                                              .engine_id = nullptr}};
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
    Handler handler(context, model, candidates);
    anira::ContractHandle contract = explicit_contract(k_block, k_rate, policy);
    if (policy == ANIRA_MISS_CALLBACK) { contract.hard_miss_fn(&fill_backup, nullptr); }
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    ASSERT_EQ(anira_handler_get_latency(h, 0), k_block) << "one block of priming";
    anira_drain_log();
    RecordCollector collector;
    GateBackend gate(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, gate));
    const DestroyFirst destroy_first(handler, &gate);
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
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract(k_block, k_rate, ANIRA_MISS_ZEROS)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    GateBackend gate(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, gate));
    const DestroyFirst destroy_first(handler, &gate);

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
        // The legacy contract prepares a C handler once patched with a geometry: its on_miss
        // is ZEROS, so no BYPASS rule runs.
        const Context context;
        ModelConfig model = ModelConfig::from_json(anira_test::gain_v2_document());
        model.add_model_path(k_custom, "custom-processor");
        std::optional<ContractHandle> legacy = model.take_legacy_contract();
        if (!legacy.has_value()) { FAIL() << "a version 2 document carries a legacy contract"; }
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
    const std::vector<anira_backend_id> only_onnx{{.struct_size = sizeof(anira_backend_id),
                                                   .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                                   .provider = ANIRA_PROVIDER_DEFAULT,
                                                   .engine_id = nullptr}};
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
        const std::vector<anira_backend_id> missing_only{{.struct_size = sizeof(anira_backend_id),
                                                          .engine = static_cast<uint32_t>(*missing),
                                                          .provider = ANIRA_PROVIDER_DEFAULT,
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
        expect_contains(outcome.m_message, "none of the 1 model entries names a candidate engine");
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
    const anira_backend_id cuda{.struct_size = sizeof(anira_backend_id),
                                .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                .provider = ANIRA_PROVIDER_CUDA,
                                .engine_id = nullptr};
    EXPECT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, &cuda, 1, &err),
              ANIRA_ERROR_NOT_SUPPORTED);
    expect_contains(err.message, "Host-only");
    const anira_backend_id short_row{.struct_size = 4,
                                     .engine = ANIRA_ENGINE_ONNXRUNTIME,
                                     .provider = ANIRA_PROVIDER_DEFAULT,
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
    for (uint32_t i = 0; i < num_plans; ++i) {
        const std::string plan = "plan " + std::to_string(i) + ": ";
        EXPECT_EQ(count_records(collector, (plan + "variant 0, engine ").c_str(), "native"), 1U);
        const std::string row =
            "': host -> host, edge zero_copy (allocate zero_copy), wait "
            "spin_backoff, recipe host";
        for (const char* slot :
             {"input 0 'audio_in", "input 1 'gain", "output 0 'audio_out", "output 1 'gain_out"}) {
            std::string line = plan;
            line += slot;
            line += row;
            EXPECT_EQ(count_records(collector, line.c_str(), "native"), 1U) << plan << slot;
        }
    }
    // The custom row is the selected plan of gain_with_custom(), under the contract's budget.
    const std::string custom = "plan " + std::to_string(anira_handler_get_plan(h)) +
                               ": variant 0, engine " + k_custom +
                               ", provider default, budget 5.000 ms";
    EXPECT_EQ(count_records(collector, custom.c_str(), "native"), 1U);
#endif

    // A second prepare logs the new report again.
    ASSERT_EQ(handler.prepare(explicit_contract(256)), ANIRA_OK) << handler.m_err.message;
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_prepare: plan report: ", "native"), 2U);
#endif
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
