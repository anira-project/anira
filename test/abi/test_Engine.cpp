// anira/abi/engine.h: the registration of a custom engine on a pipeline
// (anira_pipeline_register_engine), the lifetime of its carrier, the stage's twin, and the
// engine at run time: a model entry naming a registered id is a plan the C handler prepares
// through the engine's prepare, runs through its process on every inference and resets at a
// stream boundary, on every leg of the build (a registered engine needs no built-in one).
#include <anira/abi/build_info.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "backends/Adapter.h"
#include "capi/engine.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::explicit_contract;
using anira_test::k_custom;
using anira_test::RecordCollector;

constexpr size_t k_hop = anira_test::k_block;

/// What an engine's callbacks count on its registration.
struct EngineLife {
    int m_released = 0;
    int m_prepared = 0;
    int m_unprepared = 0;
    int m_processed = 0;
};

anira_status ANIRA_CALL engine_process(const anira_engine_ctx* /*ctx*/,
                                       void* /*prepared*/,
                                       void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_processed;
    return ANIRA_OK;
}

void ANIRA_CALL engine_reset(const anira_engine_ctx* /*ctx*/,
                             void* /*prepared*/,
                             void* /*user_data*/) {}

anira_status ANIRA_CALL engine_prepare(const anira_engine_prepare_info* /*info*/,
                                       void* user_data,
                                       void** /*out_prepared*/) {
    ++static_cast<EngineLife*>(user_data)->m_prepared;
    return ANIRA_OK;
}

void ANIRA_CALL engine_unprepare(void* /*prepared*/, void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_unprepared;
}

void ANIRA_CALL engine_release(void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_released;
}

/// A descriptor with every slot filled and no promise, over `life`.
anira_engine_desc full_engine(EngineLife& life) {
    anira_engine_desc engine = ANIRA_ENGINE_DESC_INIT;
    engine.user_data = &life;
    engine.process = engine_process;
    engine.reset = engine_reset;
    engine.prepare = engine_prepare;
    engine.unprepare = engine_unprepare;
    engine.release = engine_release;
    return engine;
}

/// The id the runtime cases register their engine under, and the path of its row, which
/// names no file: anira never opens a registered row's path.
constexpr const char* k_gain_id = "org.example.gain";
constexpr const char* k_never_opened = "never-opened.bin";

/// A mono stream, [1, 1, hop] in and out, on one model entry that names `engine_id`: the
/// engine-free custom row (anira.v2.custom) runs it as an exact pass-through; another id is
/// what a registered engine runs.
ModelConfig stream_model(const char* engine_id = k_custom, const char* path = "custom-processor") {
    ModelConfig model;
    model.add_model_path(engine_id, path);
    TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_hop));
    in.window(static_cast<int64_t>(k_hop), static_cast<int64_t>(k_hop), 0);
    model.input(in);
    TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_hop));
    out.window(static_cast<int64_t>(k_hop), static_cast<int64_t>(k_hop), 0);
    model.output(out);
    return model;
}

/// The stream model on the registered gain engine's row.
ModelConfig gain_model() {
    return stream_model(k_gain_id, k_never_opened);
}

/// A pipeline handle with its lifetime; the inference stage is added on request.
struct Pipe {
    Pipe() { EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message; }
    ~Pipe() { destroy(); }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    Pipe(Pipe&&) = delete;
    Pipe& operator=(Pipe&&) = delete;

    void add_inference(const ModelConfig& model,
                       const std::vector<anira_backend_id>& candidates = {}) {
        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants.data(),
                                               1,
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &m_err),
                  ANIRA_OK)
            << m_err.message;
    }
    anira_status register_engine(const char* id, const anira_engine_desc& engine) {
        m_err = ANIRA_ERROR_INIT;
        return anira_pipeline_register_engine(m_pipeline, id, &engine, &m_err);
    }
    anira_status create_handler(const Context& context, anira_handler** out) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_create(context.m_context, m_pipeline, out, &m_err);
    }
    void destroy() {
        anira_pipeline_destroy(m_pipeline);
        m_pipeline = nullptr;
    }

    anira_pipeline* m_pipeline = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
};

// ---- the gain engine ---------------------------------------------------------------------

/// What one prepare of the gain engine kept: the record's facts and the templates, read
/// through the prepared pointer on every process call (the control thread writes it once, the
/// inference threads read it).
struct GainPrepared {
    uint32_t m_row = 0;
    std::string m_path;
    std::string m_engine_id;
    uint32_t m_instances = 0;
    std::vector<anira_tensor> m_inputs;
    std::vector<anira_tensor> m_outputs;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

/// A C engine that multiplies its first input into its first output by `m_gain`, checking
/// every call against the record its prepare kept. What the callbacks count and keep on the
/// registration: process and reset run on the inference threads, so they touch atomics only;
/// prepare, unprepare and release run on the control thread.
struct GainEngine {
    float m_gain = 1.0F;
    /// The status the next process call returns in place of ANIRA_OK, once.
    std::atomic<anira_status> m_fail_next{ANIRA_OK};
    /// Whether prepare refuses (its status) instead of preparing.
    anira_status m_refuse_prepare = ANIRA_OK;

    std::atomic<int> m_processed{0};
    std::atomic<int> m_resets{0};
    int m_prepared = 0;
    int m_unprepared = 0;
    int m_released = 0;
    std::vector<void*> m_handed_out;         ///< every prepared pointer prepare handed back
    std::vector<void*> m_handed_back;        ///< every prepared pointer unprepare received
    std::vector<uint32_t> m_instances_seen;  ///< info.instances of every prepare

    // What process saw: the descriptors against the templates, the scalars of the context.
    std::atomic<uint32_t> m_bad_tensors{0};   ///< a descriptor unlike its template
    std::atomic<uint32_t> m_bad_counts{0};    ///< num_inputs / num_outputs unlike the record's
    std::atomic<uint32_t> m_bad_scalars{0};   ///< a ticket, a flag or a reserved slot set
    std::atomic<uint32_t> m_bad_prepared{0};  ///< a prepared pointer that is no GainPrepared
    std::atomic<uint32_t> m_max_instance{0};
    std::atomic<uint32_t> m_max_entry{0};
    // The reset boundary: the context reset saw, and whether the process that followed saw
    // the same one.
    std::atomic<const anira_engine_ctx*> m_pending_reset_ctx{nullptr};
    std::atomic<int> m_process_after_reset_same_ctx{0};
    std::atomic<int> m_process_after_reset_other_ctx{0};
    std::atomic<uint32_t> m_reset_num_inputs{0};
    std::atomic<uint32_t> m_reset_entry{0};
};

/// Whether `tensor` is `expected` in dtype, rank and extents.
bool same_shape(const anira_tensor& tensor, const anira_tensor& expected) {
    if (tensor.dtype != expected.dtype || tensor.ndim != expected.ndim) { return false; }
    for (uint32_t axis = 0; axis < expected.ndim && axis < ANIRA_MAX_RANK; ++axis) {
        if (tensor.shape[axis] != expected.shape[axis]) { return false; }
    }
    return true;
}

anira_status ANIRA_CALL gain_prepare(const anira_engine_prepare_info* info,
                                     void* user_data,
                                     void** out_prepared) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_prepared;
    if (engine->m_refuse_prepare != ANIRA_OK) { return engine->m_refuse_prepare; }
    if (info == nullptr || info->struct_size != sizeof(anira_engine_prepare_info) ||
        info->model == nullptr || out_prepared == nullptr || *out_prepared != nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    auto prepared = std::make_unique<GainPrepared>();
    prepared->m_row = info->row;
    // The row's facts through the config getters on the variant: the engine's own way to the
    // model, which anira never opened.
    const char* path = anira_model_config_model_path(info->model, info->row);
    const char* id = anira_model_config_model_engine_id(info->model, info->row);
    prepared->m_path = path != nullptr ? path : "";
    prepared->m_engine_id = id != nullptr ? id : "";
    prepared->m_instances = info->instances;
    for (uint32_t i = 0; i < info->num_inputs; ++i) {
        prepared->m_inputs.push_back(info->inputs[i]);
        prepared->m_input_names.emplace_back(info->input_names[i]);
    }
    for (uint32_t i = 0; i < info->num_outputs; ++i) {
        prepared->m_outputs.push_back(info->outputs[i]);
        prepared->m_output_names.emplace_back(info->output_names[i]);
    }
    engine->m_instances_seen.push_back(info->instances);
    *out_prepared = prepared.release();
    engine->m_handed_out.push_back(*out_prepared);
    return ANIRA_OK;
}

anira_status ANIRA_CALL gain_process(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    engine->m_processed.fetch_add(1);
    const auto* kept = static_cast<const GainPrepared*>(prepared);
    if (kept == nullptr) {
        engine->m_bad_prepared.fetch_add(1);
        return ANIRA_ERROR_INTERNAL;
    }
    // The reset boundary: the process right behind a reset runs on the reset's context.
    if (const anira_engine_ctx* reset_ctx = engine->m_pending_reset_ctx.exchange(nullptr)) {
        if (reset_ctx == ctx) {
            engine->m_process_after_reset_same_ctx.fetch_add(1);
        } else {
            engine->m_process_after_reset_other_ctx.fetch_add(1);
        }
    }
    // Every call against the record: the counts, the descriptors, the scalars.
    if (ctx->num_inputs != kept->m_inputs.size() || ctx->num_outputs != kept->m_outputs.size()) {
        engine->m_bad_counts.fetch_add(1);
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < ctx->num_inputs; ++i) {
        if (!same_shape(ctx->inputs[i], kept->m_inputs[i])) { engine->m_bad_tensors.fetch_add(1); }
    }
    for (uint32_t i = 0; i < ctx->num_outputs; ++i) {
        if (!same_shape(ctx->outputs[i], kept->m_outputs[i])) {
            engine->m_bad_tensors.fetch_add(1);
        }
    }
    if (ctx->ticket != ANIRA_TICKET_INVALID || ctx->flags != 0 || ctx->reserved0 != 0 ||
        ctx->reserved1 != 0 || ctx->reserved_ptr0 != nullptr || ctx->reserved_ptr1 != nullptr ||
        ctx->instance >= kept->m_instances) {
        engine->m_bad_scalars.fetch_add(1);
    }
    uint32_t seen = engine->m_max_instance.load();
    while (ctx->instance > seen &&
           !engine->m_max_instance.compare_exchange_weak(seen, ctx->instance)) {}
    seen = engine->m_max_entry.load();
    while (ctx->entry > seen && !engine->m_max_entry.compare_exchange_weak(seen, ctx->entry)) {}
    // The gain, over the first slot of either side.
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const size_t count = std::min(anira_tensor_num_elements(&ctx->inputs[0]),
                                  anira_tensor_num_elements(&ctx->outputs[0]));
    for (size_t i = 0; i < count; ++i) { out[i] = in[i] * engine->m_gain; }
    // A failure asked for: the block is lost, nothing else is.
    return engine->m_fail_next.exchange(ANIRA_OK);
}

void ANIRA_CALL gain_reset(const anira_engine_ctx* ctx, void* /*prepared*/, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    engine->m_resets.fetch_add(1);
    engine->m_pending_reset_ctx.store(ctx);
    if (ctx != nullptr) {
        engine->m_reset_num_inputs.store(ctx->num_inputs);
        engine->m_reset_entry.store(ctx->entry);
    }
}

void ANIRA_CALL gain_unprepare(void* prepared, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_unprepared;
    engine->m_handed_back.push_back(prepared);
    delete static_cast<GainPrepared*>(prepared);
}

void ANIRA_CALL gain_release(void* user_data) {
    ++static_cast<GainEngine*>(user_data)->m_released;
}

/// The gain engine's descriptor over `engine`, every slot filled, `flags` as given.
anira_engine_desc gain_desc(GainEngine& engine, uint32_t flags = 0) {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.flags = flags;
    desc.process = gain_process;
    desc.reset = gain_reset;
    desc.prepare = gain_prepare;
    desc.unprepare = gain_unprepare;
    desc.release = gain_release;
    return desc;
}

/// The prepared record behind a handed-out pointer.
const GainPrepared& kept_of(const GainEngine& engine, size_t index = 0) {
    return *static_cast<const GainPrepared*>(engine.m_handed_out.at(index));
}

/// The ramp through the handler's one stream, `blocks` calls of one hop each: call k hands
/// block k in and gets block k - 1 back, times `gain` (the priming zeros on the first call).
/// Every call waits for its own inference, so the engine's counts are settled on return.
void run_ramp(anira_handler* handler, size_t blocks, float gain = 1.0F, size_t first = 1) {
    for (size_t k = first; k < first + blocks; ++k) {
        std::vector<float> block = anira_test::ramp(k);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_hop);
        const size_t prev = anira_test::available(handler);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process(handler, &io, 0, &io, 0, &delivered), ANIRA_OK);
        ASSERT_EQ(delivered, k_hop);
        ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_block(handler, prev));
        if (k == 1) {
            anira_test::expect_all(block, 0.0F, "the priming zeros");
            continue;
        }
        std::vector<float> expected = anira_test::ramp(k - 1);
        for (float& sample : expected) { sample *= gain; }
        anira_test::expect_same_block(block, expected, k);
    }
}

/// An engine this build does not carry, if there is one.
std::optional<anira_engine> missing_engine() {
    const std::vector<anira::BackendId> enabled = anira::enabled_backends();
    for (const anira_engine engine : {ANIRA_ENGINE_ONNXRUNTIME,
                                      ANIRA_ENGINE_LIBTORCH,
                                      ANIRA_ENGINE_TFLITE,
                                      ANIRA_ENGINE_LITERT,
                                      ANIRA_ENGINE_EXECUTORCH}) {
        const bool found = std::ranges::any_of(enabled, [engine](const anira::BackendId& id) {
            return id.engine == static_cast<uint32_t>(engine);
        });
        if (!found) { return engine; }
    }
    return std::nullopt;
}

/// The plan rows of a prepared handler's report.
std::vector<anira_plan_info> plan_rows(const anira_handler* handler) {
    const anira_plan_report* report = anira_handler_plan_report(handler);
    EXPECT_NE(report, nullptr);
    if (report == nullptr) { return {}; }
    uint32_t count = 0;
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, nullptr), ANIRA_OK);
    std::vector<anira_plan_info> rows(count, ANIRA_PLAN_INFO_INIT);
    EXPECT_EQ(anira_plan_report_plans(report, sizeof(anira_plan_info), &count, rows.data()),
              ANIRA_OK);
    return rows;
}

/// The slot rows of one side of one plan.
std::vector<anira_plan_slot> slot_rows(const anira_handler* handler, uint32_t plan, bool inputs) {
    const anira_plan_report* report = anira_handler_plan_report(handler);
    EXPECT_NE(report, nullptr);
    if (report == nullptr) { return {}; }
    uint32_t count = 0;
    EXPECT_EQ(anira_plan_report_slots(report,
                                      plan,
                                      inputs ? 1 : 0,
                                      sizeof(anira_plan_slot),
                                      &count,
                                      nullptr),
              ANIRA_OK);
    std::vector<anira_plan_slot> rows(count, ANIRA_PLAN_SLOT_INIT);
    EXPECT_EQ(anira_plan_report_slots(report,
                                      plan,
                                      inputs ? 1 : 0,
                                      sizeof(anira_plan_slot),
                                      &count,
                                      rows.data()),
              ANIRA_OK);
    return rows;
}

/// The extension rows of one plan.
std::vector<anira_plan_ext> ext_rows(const anira_handler* handler, uint32_t plan) {
    const anira_plan_report* report = anira_handler_plan_report(handler);
    EXPECT_NE(report, nullptr);
    if (report == nullptr) { return {}; }
    uint32_t count = 0;
    EXPECT_EQ(anira_plan_report_exts(report, plan, sizeof(anira_plan_ext), &count, nullptr),
              ANIRA_OK);
    std::vector<anira_plan_ext> rows(count, ANIRA_PLAN_EXT_INIT);
    EXPECT_EQ(anira_plan_report_exts(report, plan, sizeof(anira_plan_ext), &count, rows.data()),
              ANIRA_OK);
    return rows;
}

}  // namespace

// Every ANIRA_ERROR_INVALID_ARGUMENT cause of the registration, named in the message: the three
// NULL arguments, an id that is no reverse-URI name or anira's own (anira.v2.custom among them),
// a short struct_size, a flags bit the header does not define, a NULL process and a NULL kinds
// array or entry with a count. A refused call creates no carrier and never calls release.
TEST(AbiEngine, RegisterRefusals) {
    Pipe pipe;
    EngineLife life;
    const anira_engine_desc engine = full_engine(life);
    const auto refused =
        [&pipe, &life](const char* id, const anira_engine_desc* desc, const char* fragment) {
            anira_error err = ANIRA_ERROR_INIT;
            EXPECT_EQ(anira_pipeline_register_engine(pipe.m_pipeline, id, desc, &err),
                      ANIRA_ERROR_INVALID_ARGUMENT)
                << id;
            EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
            EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
            EXPECT_EQ(life.m_released, 0);
        };
    {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_register_engine(nullptr, "org.example.gain", &engine, &err),
                  ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::strstr(err.message, "NULL pipeline"), nullptr) << err.message;
    }
    refused(nullptr, &engine, "NULL engine_id");
    refused("org.example.gain", nullptr, "NULL desc");
    // The id: a reverse-URI name, and not one of anira's own.
    refused("gain", &engine, "'.'");
    refused("anira.gain", &engine, "anira.");
    refused(k_custom, &engine, "anira.");
    // The descriptor.
    anira_engine_desc bad = engine;
    bad.struct_size = static_cast<uint32_t>(offsetof(anira_engine_desc, user_data));
    refused("org.example.gain", &bad, "struct_size");
    bad = engine;
    bad.flags = 8U;  // the reserved alias bit is not defined in this pre-release
    refused("org.example.gain", &bad, "flags");
    bad.flags = 0x80000000U | ANIRA_ENGINE_FLAG_REALTIME_SAFE;
    refused("org.example.gain", &bad, "flags");
    bad = engine;
    bad.process = nullptr;
    refused("org.example.gain", &bad, "process");
    bad = engine;
    bad.num_consumed_kinds = 1;
    refused("org.example.gain", &bad, "consumed_kinds");
    const std::array<const char*, 1> null_kind{nullptr};
    bad.consumed_kinds = null_kind.data();
    refused("org.example.gain", &bad, "consumed_kinds[0]");
    // The same descriptor, well formed, registers: the refusals were the arguments'.
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_OK) << pipe.m_err.message;
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// A caller compiled against a header that ends after process hands over the slots up to it:
// the rest reads as ANIRA_ENGINE_DESC_INIT, so the carrier's engine has no reset, no prepare,
// no unprepare and no release, whatever the memory beyond struct_size holds; the kinds are
// copied, the id owned.
TEST(AbiEngine, AShortDescriptorReadsAsTheDefaults) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);  // release is set: beyond struct_size, not read
    const std::array<const char*, 1> kinds{"model:entry"};
    engine.consumed_kinds = kinds.data();
    engine.num_consumed_kinds = 1;
    engine.flags = ANIRA_ENGINE_FLAG_DYNAMIC_TIME;
    engine.struct_size = static_cast<uint32_t>(offsetof(anira_engine_desc, process) +
                                               sizeof(anira_engine_process_fn));
    {
        const std::string id = "org.example.short";  // dies before the carrier does
        ASSERT_EQ(pipe.register_engine(id.c_str(), engine), ANIRA_OK) << pipe.m_err.message;
    }
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    const anira::capi::EngineCarrier& carrier = *pipe.m_pipeline->m_engines[0];
    EXPECT_EQ(carrier.id(), "org.example.short");
    const anira_engine_desc& kept = carrier.desc();
    EXPECT_EQ(kept.struct_size, sizeof(anira_engine_desc));
    EXPECT_EQ(kept.abi_version, ANIRA_ABI_VERSION);
    EXPECT_EQ(kept.user_data, &life);
    EXPECT_EQ(kept.flags, ANIRA_ENGINE_FLAG_DYNAMIC_TIME);
    EXPECT_EQ(kept.process, engine_process);
    EXPECT_EQ(kept.reset, nullptr);
    EXPECT_EQ(kept.prepare, nullptr);
    EXPECT_EQ(kept.unprepare, nullptr);
    EXPECT_EQ(kept.release, nullptr);
    ASSERT_EQ(carrier.consumed_kinds().size(), 1U);
    EXPECT_EQ(carrier.consumed_kinds()[0], "model:entry");
    ASSERT_NE(kept.consumed_kinds, nullptr);
    EXPECT_NE(kept.consumed_kinds, kinds.data()) << "copied into the carrier";
    EXPECT_STREQ(kept.consumed_kinds[0], "model:entry");
    pipe.destroy();
    EXPECT_EQ(life.m_released, 0) << "the release slot lay beyond struct_size";
}

// The abi_version of the descriptor is checked as anira_check_abi checks it: another major is
// ANIRA_ERROR_ABI_VERSION, no carrier, no release; the header's own registers.
TEST(AbiEngine, AbiVersionIsChecked) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);
    engine.abi_version = ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1U, 0U);
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_ERROR_ABI_VERSION);
    EXPECT_NE(std::strstr(pipe.m_err.message, "ABI"), nullptr) << pipe.m_err.message;
    EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
    engine.abi_version = ANIRA_ABI_VERSION;
    EXPECT_EQ(pipe.register_engine("org.example.gain", engine), ANIRA_OK) << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// One carrier per id: a second registration of an id is ANIRA_ERROR_INVALID_STATE naming it,
// creates no carrier and never calls its release, while a malformed second descriptor reports
// its own fault first (the duplicate is checked last); another id registers beside the first.
TEST(AbiEngine, ADuplicateIdIsInvalidState) {
    Pipe pipe;
    EngineLife first;
    EngineLife second;
    EngineLife third;
    ASSERT_EQ(pipe.register_engine("org.example.twice", full_engine(first)), ANIRA_OK)
        << pipe.m_err.message;
    EXPECT_EQ(pipe.register_engine("org.example.twice", full_engine(second)),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.twice"), nullptr) << pipe.m_err.message;
    EXPECT_NE(std::strstr(pipe.m_err.message, "already"), nullptr) << pipe.m_err.message;
    // The same descriptor a second time is a duplicate as well.
    EXPECT_EQ(pipe.register_engine("org.example.twice", full_engine(first)),
              ANIRA_ERROR_INVALID_STATE);
    // A malformed descriptor under the taken id is reported as such, not as a duplicate.
    anira_engine_desc bad = full_engine(second);
    bad.flags = 8U;
    EXPECT_EQ(pipe.register_engine("org.example.twice", bad), ANIRA_ERROR_INVALID_ARGUMENT);
    bad = full_engine(second);
    bad.process = nullptr;
    EXPECT_EQ(pipe.register_engine("org.example.twice", bad), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(pipe.register_engine("org.example.thrice", full_engine(third)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 2U);
    EXPECT_EQ(pipe.m_pipeline->m_engines[0]->id(), "org.example.twice");
    EXPECT_EQ(pipe.m_pipeline->m_engines[1]->id(), "org.example.thrice");
    pipe.destroy();
    EXPECT_EQ(first.m_released, 1);
    EXPECT_EQ(second.m_released, 0);
    EXPECT_EQ(third.m_released, 1);
}

// A registration is legal before and after anira_pipeline_add_inference, and a handler created
// from the pipeline carries the same carriers; a registered engine no entry names is not a
// plan and not an error, so the handler prepares and runs on the custom row as before. A row
// naming a registered id is a plan: the handler creates, prepares through the engine's
// prepare and runs its stream through the engine's process.
TEST(AbiEngine, LegalBeforeAndAfterAddInference) {
    const Context context;
    Pipe pipe;
    EngineLife before;
    EngineLife after;
    ASSERT_EQ(pipe.register_engine("org.example.before", full_engine(before)), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(stream_model());
    ASSERT_EQ(pipe.register_engine("org.example.after", full_engine(after)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 2U);

    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(handler->m_pipeline.m_engines.size(), 2U);
    EXPECT_EQ(handler->m_pipeline.m_engines[0].get(), pipe.m_pipeline->m_engines[0].get());
    EXPECT_EQ(handler->m_pipeline.m_engines[1].get(), pipe.m_pipeline->m_engines[1].get());
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler);
    ASSERT_NE(report, nullptr);
    EXPECT_EQ(anira_plan_report_num_plans(report), 1U) << "the custom row alone is a plan";
    EXPECT_EQ(before.m_prepared + after.m_prepared, 0) << "no row names either engine";
    anira_handler_destroy(handler);
    EXPECT_EQ(before.m_released + after.m_released, 0) << "the pipeline still holds them";

    // A row naming a registered id, registered after the inference stage: the engine's
    // prepare runs at prepare, its process on every inference, and the stream comes back.
    GainEngine gain;
    Pipe named;
    named.add_inference(stream_model("org.example.after", k_never_opened));
    ASSERT_EQ(named.register_engine("org.example.after", gain_desc(gain)), ANIRA_OK)
        << named.m_err.message;
    anira_handler* runs = nullptr;
    ASSERT_EQ(named.create_handler(context, &runs), ANIRA_OK) << named.m_err.message;
    ASSERT_EQ(anira_handler_prepare(runs, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(runs)), 1U);
    EXPECT_EQ(gain.m_prepared, 1);
    ASSERT_NO_FATAL_FAILURE(run_ramp(runs, 3));
    EXPECT_EQ(gain.m_processed.load(), 3);
    anira_handler_destroy(runs);
    EXPECT_EQ(gain.m_unprepared, 1);
    named.destroy();
    EXPECT_EQ(gain.m_released, 1);
    pipe.destroy();
    EXPECT_EQ(before.m_released, 1);
    EXPECT_EQ(after.m_released, 1);
}

// The carrier is shared by the pipeline and every handler created from it: release fires
// exactly once, with the last of them, whatever the destroy order, after the handlers' sessions
// are gone.
TEST(AbiEngine, ReleaseFiresOnceWithTheLastCarrier) {
    const Context context;
    const ModelConfig model = stream_model();
    const anira::ContractHandle contract = explicit_contract();
    // The three orders: the pipeline first, between the handlers, last.
    const std::array<std::array<int, 3>, 3> orders{{{0, 1, 2}, {1, 0, 2}, {1, 2, 0}}};
    for (const std::array<int, 3>& order : orders) {
        SCOPED_TRACE("order " + std::to_string(order[0]) + std::to_string(order[1]) +
                     std::to_string(order[2]));
        EngineLife life;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine("org.example.shared", full_engine(life)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(model);
        anira_handler* first = nullptr;
        anira_handler* second = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &first), ANIRA_OK) << pipe.m_err.message;
        ASSERT_EQ(pipe.create_handler(context, &second), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
        const std::array<std::function<void()>, 3> destroy{
            [&pipe] { pipe.destroy(); },
            [&first] { anira_handler_destroy(first); },
            [&second] { anira_handler_destroy(second); }};
        for (size_t step = 0; step < 3; ++step) {
            EXPECT_EQ(life.m_released, 0) << "step " << step;
            destroy.at(static_cast<size_t>(order.at(step)))();
        }
        EXPECT_EQ(life.m_released, 1);
    }
}

// A handler copies the pipeline at create: an engine registered afterwards reaches the pipeline
// and not the handler, and its release fires with the pipeline, the handler sharing nothing of it.
TEST(AbiEngine, RegisteredAfterCreateDoesNotReachTheHandler) {
    const Context context;
    Pipe pipe;
    pipe.add_inference(stream_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    EngineLife life;
    ASSERT_EQ(pipe.register_engine("org.example.late", full_engine(life)), ANIRA_OK)
        << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    EXPECT_TRUE(handler->m_pipeline.m_engines.empty());
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1) << "the handler holds no share of the late carrier";
    anira_handler_destroy(handler);
    EXPECT_EQ(life.m_released, 1);
}

// ==== the engine at run time ==================================================================

// A registered gain engine on a row whose path anira never opens runs the stream end to end,
// on every leg of the build: the engine's prepare sees the row, its path through the config
// getters, the canonical names and the templates; every block goes through its process; the
// stream is the ramp times the gain, one block late.
TEST(AbiEngine, PassthroughOnEveryLeg) {
    const Context context;
    GainEngine gain;
    gain.m_gain = 0.5F;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;

    ASSERT_EQ(gain.m_prepared, 1);
    ASSERT_EQ(gain.m_handed_out.size(), 1U);
    const GainPrepared& kept = kept_of(gain);
    EXPECT_EQ(kept.m_row, 0U);
    EXPECT_EQ(kept.m_path, k_never_opened) << "the engine reads the row; anira never opens it";
    EXPECT_EQ(kept.m_engine_id, k_gain_id);
    EXPECT_EQ(kept.m_instances, 1U);
    ASSERT_EQ(kept.m_inputs.size(), 1U);
    ASSERT_EQ(kept.m_outputs.size(), 1U);
    EXPECT_EQ(kept.m_input_names, (std::vector<std::string>{"in"}));
    EXPECT_EQ(kept.m_output_names, (std::vector<std::string>{"out"}));
    for (const anira_tensor* tensor : {&kept.m_inputs[0], &kept.m_outputs[0]}) {
        EXPECT_EQ(tensor->dtype, static_cast<anira_dtype>(ANIRA_DTYPE_F32));
        EXPECT_EQ(tensor->ndim, 3U);
        EXPECT_EQ(tensor->shape[0], 1);
        EXPECT_EQ(tensor->shape[1], 1);
        EXPECT_EQ(tensor->shape[2], static_cast<int64_t>(k_hop));
        EXPECT_EQ(tensor->domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(tensor->handle.host.ptr, nullptr) << "a template carries no memory";
    }
    // The plan: ANIRA_ENGINE_NONE with the id.
    const std::vector<anira_plan_info> plans = plan_rows(handler);
    ASSERT_EQ(plans.size(), 1U);
    EXPECT_EQ(plans[0].engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
    ASSERT_NE(plans[0].engine_id, nullptr);
    EXPECT_STREQ(plans[0].engine_id, k_gain_id);

    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 6, gain.m_gain));
    EXPECT_EQ(gain.m_processed.load(), 6);
    EXPECT_EQ(gain.m_resets.load(), 0) << "a stateless model is never reset";
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_unprepared, 1);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// A custom row whose id no registration serves is ANIRA_ERROR_NOT_SUPPORTED at create, naming
// the id and the entry that registers one; another registration does not serve it.
TEST(AbiEngine, AnUnregisteredCustomIdIsNotSupported) {
    const Context context;
    for (const bool other_registered : {false, true}) {
        SCOPED_TRACE(other_registered ? "another id registered" : "nothing registered");
        GainEngine gain;  // outlives the pipeline, whose carrier's release reads it
        Pipe pipe;
        if (other_registered) {
            ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK)
                << pipe.m_err.message;
        }
        pipe.add_inference(stream_model("org.example.nobody", k_never_opened));
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED)
            << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.nobody"), nullptr)
            << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message, "is not registered on this pipeline"), nullptr)
            << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message, "anira_pipeline_register_engine"), nullptr)
            << pipe.m_err.message;
        EXPECT_EQ(gain.m_prepared, 0);
    }
}

// A built-in engine this build does not carry, named by the candidates, is
// ANIRA_ERROR_NOT_SUPPORTED at create beside the unregistered id: the refusal names the engine
// and the build's engines.
TEST(AbiEngine, AnEngineNotInThisBuildIsNotSupported) {
    const std::optional<anira_engine> missing = missing_engine();
    if (!missing.has_value()) { GTEST_SKIP() << "this build carries every engine"; }
    const Context context;
    // The row on the absent engine, named by an explicit candidate list: not skipped.
    ModelConfig model;
    model.add_model_path(*missing, "model.bin");
    TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, 512);
    in.window(512, 512, 0);
    model.input(in);
    TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, 512);
    out.window(512, 512, 0);
    model.output(out);
    const std::vector<anira_backend_id> only_missing{{.struct_size = sizeof(anira_backend_id),
                                                      .engine = static_cast<uint32_t>(*missing),
                                                      .provider = ANIRA_PROVIDER_DEFAULT,
                                                      .engine_id = nullptr}};
    Pipe absent;
    absent.add_inference(model, only_missing);
    anira_handler* handler = nullptr;
    EXPECT_EQ(absent.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED)
        << absent.m_err.message;
    EXPECT_EQ(handler, nullptr);
    EXPECT_NE(std::strstr(absent.m_err.message, "is not in this build"), nullptr)
        << absent.m_err.message;
}

// A built-in engine's row whose file is missing passes create (nothing loads there) and fails
// prepare with ANIRA_ERROR_NO_SUCH_FILE or ANIRA_ERROR_MODEL_LOAD, the message naming the path.
TEST(AbiEngine, AMissingFileOnARealEngineIsModelLoadOrNoSuchFile) {
    const std::vector<anira_engine> engines = anira_test::oracle_engines();
    if (engines.empty()) { GTEST_SKIP() << "an engine-free build"; }
    const Context context;
    constexpr const char* k_missing_path = "/nonexistent/anira-test-missing-model.bin";
    for (const anira_engine engine : engines) {
        SCOPED_TRACE(static_cast<int>(engine));
        ModelConfig model;
        model.add_model_path(engine, k_missing_path);
        TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        in.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, 1)
            .axis(2, ANIRA_AXIS_TIME, 512);
        in.window(512, 512, 0);
        model.input(in);
        TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
        out.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, 1)
            .axis(2, ANIRA_AXIS_TIME, 512);
        out.window(512, 512, 0);
        model.output(out);
        const std::vector<anira_backend_id> only{{.struct_size = sizeof(anira_backend_id),
                                                  .engine = static_cast<uint32_t>(engine),
                                                  .provider = ANIRA_PROVIDER_DEFAULT,
                                                  .engine_id = nullptr}};
        Pipe pipe;
        pipe.add_inference(model, only);
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        const anira::ContractHandle contract = explicit_contract();
        const anira_status status = anira_handler_prepare(handler, contract.native(), &err);
        EXPECT_TRUE(status == ANIRA_ERROR_NO_SUCH_FILE || status == ANIRA_ERROR_MODEL_LOAD)
            << status << ": " << err.message;
        // The message carries the absolute path in the platform's spelling (a drive letter and
        // backslashes on Windows): the file's name is what every spelling shares.
        EXPECT_NE(std::strstr(err.message, "anira-test-missing-model.bin"), nullptr) << err.message;
        EXPECT_EQ(anira_handler_plan_report(handler), nullptr) << "unprepared";
        anira_handler_destroy(handler);
    }
}

// Two handlers of one pipeline over one model share one prepared model: the engine's prepare
// runs once, with `instances` = the model's max_instances, and both handlers run on it. A
// session-exclusive model (ANIRA_MODEL_STATEFUL) is prepared for each handler alone, with
// `instances` = 1.
TEST(AbiEngine, EqualConfigsShareOnePreparedModelAStatefulOneNever) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        ModelConfig model = gain_model();
        model.max_instances(2);
        pipe.add_inference(model);
        anira_handler* first = nullptr;
        anira_handler* second = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &first), ANIRA_OK) << pipe.m_err.message;
        ASSERT_EQ(pipe.create_handler(context, &second), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
        EXPECT_EQ(gain.m_prepared, 1) << "one prepared model for two equal handlers";
        EXPECT_EQ(gain.m_instances_seen, (std::vector<uint32_t>{2}));
        EXPECT_EQ(anira_test::session_of(first)->m_plans.at(0).m_adapter,
                  anira_test::session_of(second)->m_plans.at(0).m_adapter)
            << "the pooled adapter";
        EXPECT_EQ(anira_test::session_of(first)->m_plans.at(0).m_adapter->model().m_instances, 2U)
            << "the record agrees with the prepare record";
        ASSERT_NO_FATAL_FAILURE(run_ramp(first, 3));
        ASSERT_NO_FATAL_FAILURE(run_ramp(second, 3));
        EXPECT_EQ(gain.m_processed.load(), 6);
        anira_handler_destroy(first);
        EXPECT_EQ(gain.m_unprepared, 0) << "the second handler still shares it";
        anira_handler_destroy(second);
        EXPECT_EQ(gain.m_unprepared, 1);
        pipe.destroy();
        EXPECT_EQ(gain.m_released, 1);
    }
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        ModelConfig model = gain_model();
        model.max_instances(2);
        model.state(ANIRA_MODEL_STATEFUL);
        pipe.add_inference(model);
        anira_handler* first = nullptr;
        anira_handler* second = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &first), ANIRA_OK) << pipe.m_err.message;
        ASSERT_EQ(pipe.create_handler(context, &second), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
        EXPECT_EQ(gain.m_prepared, 2) << "a session-exclusive model is never shared";
        EXPECT_EQ(gain.m_instances_seen, (std::vector<uint32_t>{1, 1}));
        EXPECT_NE(anira_test::session_of(first)->m_plans.at(0).m_adapter,
                  anira_test::session_of(second)->m_plans.at(0).m_adapter);
        for (const anira_handler* handler : {first, second}) {
            const anira::backend::Model& record =
                anira_test::session_of(handler)->m_plans.at(0).m_adapter->model();
            EXPECT_EQ(record.m_instances, 1U) << "the record agrees with the prepare record";
            EXPECT_TRUE(record.m_session_exclusive);
        }
        anira_handler_destroy(first);
        EXPECT_EQ(gain.m_unprepared, 1);
        anira_handler_destroy(second);
        EXPECT_EQ(gain.m_unprepared, 2);
        pipe.destroy();
        EXPECT_EQ(gain.m_released, 1);
    }
}

// The record of a session-exclusive prepared model says one instance whatever the model's
// max_instances, on the registered engine's row and on the row of every built-in engine of the
// build alike (the built-in adapters size their instances from the record, the registered
// engine reads the same count in its prepare record): the two records agree.
TEST(AbiEngine, ASessionExclusiveModelIsPreparedWithOneInstanceOnEveryEngine) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    // The bundled gain model on every engine of the build plus the engine-free custom row.
    ModelConfig model = anira_test::gain_with_custom(false);
    model.state(ANIRA_MODEL_STATEFUL).max_instances(2);
    Pipe pipe;
    pipe.add_inference(model, anira_test::custom_candidates());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(handler);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->m_plans.size(), anira_test::oracle_engines().size() + 1)
        << "one plan per engine of the build, plus the custom row";
    for (const anira::SessionElement::PlanSlot& plan : session->m_plans) {
        ASSERT_NE(plan.m_adapter, nullptr);
        const anira::backend::Model& record = plan.m_adapter->model();
        EXPECT_EQ(record.m_instances, 1U) << "the plan of engine " << plan.m_engine;
        EXPECT_TRUE(record.m_session_exclusive) << "the plan of engine " << plan.m_engine;
    }
    anira_handler_destroy(handler);
}

// Three prepares of one handler are three prepared models: three prepares, three unprepares
// (two at the re-prepares, one at destroy), each unprepare receiving the pointer its prepare
// handed back, in order.
TEST(AbiEngine, UnprepareOncePerPreparedModel) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    const anira::ContractHandle contract = explicit_contract();
    for (int prepare = 1; prepare <= 3; ++prepare) {
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        EXPECT_EQ(gain.m_prepared, prepare);
        EXPECT_EQ(gain.m_unprepared, prepare - 1) << "the previous prepared model went back";
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
    }
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_prepared, 3);
    EXPECT_EQ(gain.m_unprepared, 3);
    EXPECT_EQ(gain.m_handed_back, gain.m_handed_out);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// unprepare is not deferred by an inference thread: while another handler keeps the core's
// pool alive, the thread that ran the last inference of a handler lets go of that handler's
// session with the job, so the re-prepare and the destroy of the handler unprepare its
// prepared model before they return, whatever ran on it. (A session-exclusive model, so that
// each handler has a prepared model of its own to watch.)
TEST(AbiEngine, UnprepareIsNotDeferredByAnInferenceThreadKeepingTheLastJob) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.state(ANIRA_MODEL_STATEFUL);
    pipe.add_inference(model);
    anira_handler* first = nullptr;
    anira_handler* second = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &first), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(pipe.create_handler(context, &second), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_NO_FATAL_FAILURE(run_ramp(first, 3));
    ASSERT_NO_FATAL_FAILURE(run_ramp(second, 3));
    EXPECT_EQ(gain.m_unprepared, 0);
    // The re-prepare of the first handler gives its prepared model back before it returns,
    // the second handler (and with it the pool and its threads) staying alive.
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_unprepared, 1) << "unprepared at the re-prepare, not at a later dequeue";
    EXPECT_EQ(gain.m_prepared, 3);
    ASSERT_NO_FATAL_FAILURE(run_ramp(first, 2));
    anira_handler_destroy(first);
    EXPECT_EQ(gain.m_unprepared, 2) << "unprepared at the destroy";
    ASSERT_NO_FATAL_FAILURE(run_ramp(second, 2, 1.0F, 4));
    anira_handler_destroy(second);
    EXPECT_EQ(gain.m_unprepared, 3);
    // Every pointer handed out came back once (in the order of the re-prepare and the two
    // destroys, not of the prepares).
    std::vector<void*> handed_out = gain.m_handed_out;
    std::vector<void*> handed_back = gain.m_handed_back;
    std::ranges::sort(handed_out);
    std::ranges::sort(handed_back);
    EXPECT_EQ(handed_back, handed_out);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// A prepare the engine refuses fails anira_handler_prepare with the engine's status, naming the
// engine; no unprepare is owed for it, and the handler stays unprepared.
TEST(AbiEngine, ARefusedPrepareFailsThePrepareAndOwesNoUnprepare) {
    const Context context;
    GainEngine gain;
    gain.m_refuse_prepare = ANIRA_ERROR_MODEL_LOAD;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    EXPECT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(err.message, k_gain_id), nullptr) << err.message;
    EXPECT_NE(std::strstr(err.message, "refused prepare"), nullptr) << err.message;
    EXPECT_EQ(anira_handler_plan_report(handler), nullptr);
    EXPECT_EQ(gain.m_prepared, 1);
    // The engine relents: the next prepare succeeds, and destroy unprepares that one alone.
    gain.m_refuse_prepare = ANIRA_OK;
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_prepared, 2);
    EXPECT_EQ(gain.m_unprepared, 1);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// A STATEFUL model without a pair gets its engine reset: once at the first inference after
// prepare and once at the first after anira_handler_reset, each on the context of the process
// that follows, never mid-stream; a stateless model is never reset.
TEST(AbiEngine, ResetIsCalledAtTheFirstInferenceOfANewGeneration) {
    const Context context;
    const anira::ContractHandle contract = explicit_contract();
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        ModelConfig model = gain_model();
        model.state(ANIRA_MODEL_STATEFUL);
        pipe.add_inference(model);
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        EXPECT_TRUE(handler->m_inference_config.m_session_exclusive_processor);
        EXPECT_EQ(gain.m_resets.load(), 0) << "nothing before the first inference";
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        EXPECT_EQ(gain.m_resets.load(), 1) << "once, at the first inference after prepare";
        EXPECT_EQ(gain.m_process_after_reset_same_ctx.load(), 1) << "with that inference's context";
        EXPECT_EQ(gain.m_process_after_reset_other_ctx.load(), 0);
        EXPECT_EQ(gain.m_reset_num_inputs.load(), 1U);
        EXPECT_LT(gain.m_reset_entry.load(), anira_handler_num_entries(handler));
        anira_handler_reset(handler);
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        EXPECT_EQ(gain.m_resets.load(), 2) << "once more, at the first inference after the reset";
        EXPECT_EQ(gain.m_process_after_reset_same_ctx.load(), 2);
        EXPECT_EQ(gain.m_process_after_reset_other_ctx.load(), 0);
        EXPECT_EQ(gain.m_processed.load(), 6);
        // A re-prepare is a new stream too.
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
        EXPECT_EQ(gain.m_resets.load(), 3);
        anira_handler_destroy(handler);
    }
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        pipe.add_inference(gain_model());
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        anira_handler_reset(handler);
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        EXPECT_EQ(gain.m_resets.load(), 0) << "a shared prepared model is never reset";
        EXPECT_EQ(gain.m_processed.load(), 6);
        anira_handler_destroy(handler);
    }
}

// The engine's flags travel into anira_plan_info.engine_flags of every plan of the engine; a
// plan that is no registered engine's reads 0, and ANIRA_PLAN_INFO_INIT's tail is 0.
TEST(AbiEngine, FlagsRoundTripThroughThePlanReport) {
    const anira_plan_info init = ANIRA_PLAN_INFO_INIT;
    EXPECT_EQ(init.engine_flags, 0U);
    EXPECT_EQ(init.reserved, 0U);

    const Context context;
    GainEngine gain;
    constexpr uint32_t k_flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_DYNAMIC_TIME;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain, k_flags)), ANIRA_OK)
        << pipe.m_err.message;
    // Two custom rows, two plans: the registered engine's and the engine-free pass-through's.
    ModelConfig model = gain_model();
    model.add_model_path(k_custom, "custom-processor");
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    const std::vector<anira_plan_info> plans = plan_rows(handler);
    ASSERT_EQ(plans.size(), 2U);
    EXPECT_STREQ(plans[0].engine_id, k_gain_id);
    EXPECT_EQ(plans[0].engine_flags, k_flags);
    EXPECT_EQ(plans[0].reserved, 0U);
    EXPECT_STREQ(plans[1].engine_id, k_custom);
    EXPECT_EQ(plans[1].engine_flags, 0U);
    anira_handler_destroy(handler);

    // A built-in engine's plan reads 0 as well, where this build has one.
    if (!anira_test::oracle_engines().empty()) {
        const anira_test::Handler bundled(context, anira_test::gain_with_custom());
        ASSERT_NE(bundled.m_handler, nullptr);
        anira::ContractHandle gain_contract = anira_test::file_contract(k_gain_contract_json, 512);
        anira_error bundled_err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(bundled.m_handler, gain_contract.native(), &bundled_err),
                  ANIRA_OK)
            << bundled_err.message;
        const std::vector<anira_plan_info> rows = plan_rows(bundled.m_handler);
        ASSERT_GE(rows.size(), 2U);
        for (const anira_plan_info& row : rows) { EXPECT_EQ(row.engine_flags, 0U); }
    }
}

// A status other than ANIRA_OK from process fails the chunk as an engine failure: the block
// delivers zeros at its stream position, ANIRA_ERROR_ENGINE is on anira_handler_rt_error with
// one record naming the status the engine returned, and the next block runs as before.
TEST(AbiEngine, AFailingProcessDeliversZerosAndLatchesItsStatus) {
    anira_drain_log();
    RecordCollector collector;
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;

    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
    // The inference of block 3 fails: block 4's call delivers its zeros.
    gain.m_fail_next.store(ANIRA_ERROR_BUDGET);
    {
        std::vector<float> block = anira_test::ramp(3);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_hop);
        const size_t prev = anira_test::available(handler);
        ASSERT_EQ(anira_handler_process(handler, &io, 0, &io, 0, nullptr), ANIRA_OK);
        ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_block(handler, prev));
        anira_test::expect_same_block(block, anira_test::ramp(2), 3);
    }
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_ERROR_ENGINE);
    {
        std::vector<float> block = anira_test::ramp(4);
        const std::array<float*, 1> ptrs{block.data()};
        const anira_tensor io = anira_test::planar_f32(ptrs.data(), 1, k_hop);
        const size_t prev = anira_test::available(handler);
        ASSERT_EQ(anira_handler_process(handler, &io, 0, &io, 0, nullptr), ANIRA_OK);
        ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_block(handler, prev));
        anira_test::expect_all(block, 0.0F, "the failed inference's block");
    }
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2, 1.0F, 5));
    EXPECT_EQ(gain.m_processed.load(), 6) << "every block reached the engine";
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_ERROR_ENGINE) << "latched until a reset";
    anira_handler_reset(handler);
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "the engine of plan 0 returned", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "the engine of plan 0 returned", "rt");
    EXPECT_NE(record.m_message.find(anira_status_string(ANIRA_ERROR_BUDGET)), std::string::npos)
        << record.m_message;
#endif
    anira_handler_destroy(handler);
}

// Every process call sees the record its prepare kept: the descriptors of the context in the
// spec's dtype and shape (the templates), the counts of the record, an instance below the
// record's instances, an entry below the handler's, ANIRA_TICKET_INVALID, no flags, the
// reserved slots clear; the names are the entry's tensors record where it names a slot and
// the canonical name otherwise.
TEST(AbiEngine, ProcessSeesTheSpecsDtypeAndShapeEveryCall) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.tensor_name(0, "out", "y_out");
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(gain.m_handed_out.size(), 1U);
    EXPECT_EQ(kept_of(gain).m_input_names, (std::vector<std::string>{"in"}));
    EXPECT_EQ(kept_of(gain).m_output_names, (std::vector<std::string>{"y_out"}));
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 8));
    EXPECT_EQ(gain.m_processed.load(), 8);
    EXPECT_EQ(gain.m_bad_prepared.load(), 0U);
    EXPECT_EQ(gain.m_bad_counts.load(), 0U);
    EXPECT_EQ(gain.m_bad_tensors.load(), 0U);
    EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
    EXPECT_LT(gain.m_max_instance.load(), kept_of(gain).m_instances);
    EXPECT_LT(gain.m_max_entry.load(), anira_handler_num_entries(handler));
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_handler_destroy(handler);
}

// The extension rows of a registered engine's plan name the engine by its id: an engine that
// declares "model:entry" consumes the entry extension of its own row, and the walk refuses the
// same row under an engine that declares nothing.
TEST(AbiEngine, TheExtRowsOfARegisteredEngineAreKeyedByItsId) {
    const Context context;
    const std::array<const char*, 1> kinds{"model:entry"};
    {
        GainEngine gain;
        anira_engine_desc desc = gain_desc(gain);
        desc.consumed_kinds = kinds.data();
        desc.num_consumed_kinds = 1;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, desc), ANIRA_OK) << pipe.m_err.message;
        ModelConfig model = gain_model();
        model.model_ext_json(0, "entry", R"({"name": "forward"})");
        pipe.add_inference(model);
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        const anira::ContractHandle contract = explicit_contract();
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        const std::vector<anira_plan_ext> rows = ext_rows(handler, 0);
        ASSERT_EQ(rows.size(), 1U);
        EXPECT_EQ(rows[0].index, 0U);
        EXPECT_STREQ(rows[0].host, "model 0");
        EXPECT_STREQ(rows[0].kind, "entry");
        EXPECT_STREQ(rows[0].consumer, k_gain_id);
        anira_handler_destroy(handler);
    }
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        ModelConfig model = gain_model();
        model.model_ext_json(0, "entry", R"({"name": "forward"})");
        pipe.add_inference(model);
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_EXTENSION_UNCONSUMED)
            << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(pipe.m_err.message, "model 0"), nullptr) << pipe.m_err.message;
    }
}

// A registered engine no row names is not a plan and not an error: the handler prepares the
// rows it has, and the idle engine's prepare never runs.
TEST(AbiEngine, ARegisteredEngineNoRowNamesIsNotAPlan) {
    const Context context;
    GainEngine named;
    GainEngine idle;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine("org.example.idle", gain_desc(idle)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(named)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(handler)), 1U);
    EXPECT_EQ(named.m_prepared, 1);
    EXPECT_EQ(idle.m_prepared, 0);
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
    EXPECT_EQ(idle.m_processed.load(), 0);
    anira_handler_destroy(handler);
    pipe.destroy();
    EXPECT_EQ(idle.m_released, 1);
    EXPECT_EQ(named.m_released, 1);
}

// The carrier is in the pool's key: two pipelines registering one id over one model share
// nothing, each engine's prepare and unprepare run once, and the two handlers run on two
// adapters.
TEST(AbiEngine, TwoPipelinesUnderOneIdShareNothing) {
    const Context context;
    GainEngine first_engine;
    GainEngine second_engine;
    Pipe first;
    Pipe second;
    ASSERT_EQ(first.register_engine(k_gain_id, gain_desc(first_engine)), ANIRA_OK)
        << first.m_err.message;
    ASSERT_EQ(second.register_engine(k_gain_id, gain_desc(second_engine)), ANIRA_OK)
        << second.m_err.message;
    const ModelConfig model = gain_model();
    first.add_inference(model);
    second.add_inference(model);
    anira_handler* first_handler = nullptr;
    anira_handler* second_handler = nullptr;
    ASSERT_EQ(first.create_handler(context, &first_handler), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.create_handler(context, &second_handler), ANIRA_OK) << second.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(first_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_prepare(second_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(first_engine.m_prepared, 1);
    EXPECT_EQ(second_engine.m_prepared, 1);
    EXPECT_NE(anira_test::session_of(first_handler)->m_plans.at(0).m_adapter,
              anira_test::session_of(second_handler)->m_plans.at(0).m_adapter);
    ASSERT_NO_FATAL_FAILURE(run_ramp(first_handler, 2));
    ASSERT_NO_FATAL_FAILURE(run_ramp(second_handler, 2));
    EXPECT_EQ(first_engine.m_processed.load(), 2);
    EXPECT_EQ(second_engine.m_processed.load(), 2);
    anira_handler_destroy(first_handler);
    anira_handler_destroy(second_handler);
    EXPECT_EQ(first_engine.m_unprepared, 1);
    EXPECT_EQ(second_engine.m_unprepared, 1);
    first.destroy();
    second.destroy();
    EXPECT_EQ(first_engine.m_released, 1);
    EXPECT_EQ(second_engine.m_released, 1);
}

// A registered engine received the names and bound its slots itself: every slot row of its
// plan reads ANIRA_BINDING_ENGINE, while the engine-free pass-through's rows read
// ANIRA_BINDING_POSITION.
TEST(AbiEngine, ASlotReportsHowItWasBound) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.register_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.add_model_path(k_custom, "custom-processor");
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    for (const bool inputs : {true, false}) {
        const std::vector<anira_plan_slot> engine_rows = slot_rows(handler, 0, inputs);
        ASSERT_EQ(engine_rows.size(), 1U);
        EXPECT_EQ(engine_rows[0].binding, static_cast<uint32_t>(ANIRA_BINDING_ENGINE));
        const std::vector<anira_plan_slot> custom_rows = slot_rows(handler, 1, inputs);
        ASSERT_EQ(custom_rows.size(), 1U);
        EXPECT_EQ(custom_rows[0].binding, static_cast<uint32_t>(ANIRA_BINDING_POSITION));
    }
    anira_handler_destroy(handler);
}
