// anira/abi/engine.h: the custom engine as an object (anira_custom_engine_create) added to
// pipelines under ids (anira_pipeline_add_engine), the lifetime of its carrier, the stage's
// twin, sharing across pipelines, and the engine at run time through the three levels of its
// lifecycle (anira/abi/lifecycle.h): a model entry naming an added id is a plan the C handler
// runs through the engine's init (once per object, at the first prepare that reaches it), its
// load (once per pooled model, a stateful one too), its prepare (once per handler, with the
// stage's record), its process on every inference (an exclusive handler's calls flagged, on
// what its prepare built) and its reset at a stream boundary, on every leg of the build (a
// registered engine needs no built-in one).
#include <anira/abi/build_info.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
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

/// What an engine's callbacks count on its user_data.
struct EngineLife {
    int m_released = 0;
    int m_inited = 0;
    int m_loaded = 0;
    int m_unloaded = 0;
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

anira_status ANIRA_CALL engine_init(const anira_init_info* /*info*/, void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_inited;
    return ANIRA_OK;
}

anira_status ANIRA_CALL engine_load(const anira_engine_load_info* /*info*/,
                                    void* user_data,
                                    void** /*out_loaded*/) {
    ++static_cast<EngineLife*>(user_data)->m_loaded;
    return ANIRA_OK;
}

void ANIRA_CALL engine_unload(void* /*loaded*/, void* user_data) {
    ++static_cast<EngineLife*>(user_data)->m_unloaded;
}

anira_status ANIRA_CALL engine_prepare(const anira_prepare_info* /*info*/,
                                       void* /*loaded*/,
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
    engine.load = engine_load;
    engine.unload = engine_unload;
    engine.init = engine_init;
    engine.release = engine_release;
    return engine;
}

/// The id the runtime cases register their engine under, and the path of its row, which
/// names no file: anira never opens a registered row's path.
constexpr const char* k_gain_id = "org.example.gain";
constexpr const char* k_never_opened = "never-opened.bin";

/// A mono stream, [1, 1, hop] in and out, on one model entry that names `engine_id`: under
/// anira.v2.custom the pass-through a test adds (anira_test::passthrough_desc) runs it as an
/// exact copy; another id is what a registered engine runs.
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
    /// A new engine over `engine` under `id`, added and its handle destroyed again: the
    /// pipeline holds the only reference then. The first refusal's status (the create's or the
    /// add's); a refused add drops the engine it created, so its release fires.
    anira_status add_new_engine(const char* id, const anira_engine_desc& engine) {
        m_err = ANIRA_ERROR_INIT;
        anira_custom_engine* handle = nullptr;
        anira_status status = anira_custom_engine_create(id, &engine, &handle, &m_err);
        if (status == ANIRA_OK) { status = add_engine(handle); }
        anira_custom_engine_destroy(handle);
        return status;
    }
    /// Adds an existing engine, under the id it was created with.
    anira_status add_engine(const anira_custom_engine* engine) {
        m_err = ANIRA_ERROR_INIT;
        return anira_pipeline_add_engine(m_pipeline, engine, &m_err);
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
/// What the gain engine's load kept of its record: the row's facts through the config getters
/// on the variant, the shared call slots, the templates and the names; and how many prepared
/// handles over it are out (an unload must find none).
struct GainLoaded {
    uint32_t m_row = 0;
    std::string m_path;
    std::string m_engine_id;
    uint32_t m_instances = 0;
    uint32_t m_provider = ANIRA_PROVIDER_DEFAULT;
    std::string m_provider_id;
    std::vector<std::pair<std::string, std::string>> m_options;  ///< the record's provider options
    std::vector<anira_tensor> m_inputs;
    std::vector<anira_tensor> m_outputs;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    int m_live_prepared = 0;
};

/// What the gain engine's prepare kept of one handler: the loaded model it runs on, whether
/// the handler is exclusive (ANIRA_PREPARE_EXCLUSIVE), the handler and its entry count.
struct GainPrepared {
    GainLoaded* m_loaded = nullptr;
    bool m_exclusive = false;
    const anira_handler* m_handler = nullptr;
    uint32_t m_num_entries = 0;
};

/// A C engine that multiplies its first input into its first output by `m_gain`, checking
/// every call against the records its load and its prepare kept. What the callbacks count and
/// keep on the registration: process and reset run on the inference threads, so they touch
/// atomics only; init, load, unload, prepare, unprepare and release run on the control thread.
struct GainEngine {
    float m_gain = 1.0F;
    /// The status the next process call returns in place of ANIRA_OK, once.
    std::atomic<anira_status> m_fail_next{ANIRA_OK};
    /// Whether init, load or prepare refuses (its status) instead of doing its work.
    anira_status m_refuse_init = ANIRA_OK;
    anira_status m_refuse_load = ANIRA_OK;
    anira_status m_refuse_prepare = ANIRA_OK;

    std::atomic<int> m_processed{0};
    std::atomic<int> m_resets{0};
    int m_inited = 0;
    int m_loaded = 0;
    int m_unloaded = 0;
    int m_prepared = 0;
    int m_unprepared = 0;
    int m_released = 0;
    // The overlap watch of prepare: how many are inside at once, the peak, and how long each
    // holds its exit (so that a prepare on another thread has the time to arrive).
    std::atomic<int> m_inside_prepare{0};
    std::atomic<int> m_max_inside_prepare{0};
    int m_prepare_delay_ms = 0;
    std::vector<void*> m_loaded_out;         ///< every loaded pointer load handed back
    std::vector<void*> m_loaded_back;        ///< every loaded pointer unload received
    std::vector<void*> m_handed_out;         ///< every prepared pointer prepare handed back
    std::vector<void*> m_handed_back;        ///< every prepared pointer unprepare received
    std::vector<uint32_t> m_instances_seen;  ///< info.instances of every load
    /// info.provider and info.provider_id of every load, in load order.
    std::vector<std::pair<uint32_t, std::string>> m_providers_seen;
    std::vector<uint32_t> m_flags_seen;  ///< info.flags of every prepare
    int m_bad_prepare_record = 0;        ///< a prepare record or loaded pointer that is off
    int m_unload_before_unprepare = 0;   ///< an unload with a prepared handle still out
    // What init saw.
    uint32_t m_init_struct_size = 0;
    uint32_t m_init_log_level = 0;
    uint32_t m_init_num_threads = 0;
    const anira_context* m_init_context = nullptr;

    // What process saw: the descriptors against the templates, the scalars of the context.
    std::atomic<uint32_t> m_bad_tensors{0};   ///< a descriptor unlike its template
    std::atomic<uint32_t> m_bad_counts{0};    ///< num_inputs / num_outputs unlike the record's
    std::atomic<uint32_t> m_bad_scalars{0};   ///< a ticket, a flag, an instance or a reserved slot
                                              ///< off
    std::atomic<uint32_t> m_bad_prepared{0};  ///< a prepared pointer that is no GainPrepared
    std::atomic<uint32_t> m_bad_loaded{0};    ///< ctx->loaded unlike the handle's loaded model
    std::atomic<uint32_t> m_shared_calls{0};  ///< calls without ANIRA_ENGINE_CALL_EXCLUSIVE
    std::atomic<uint32_t> m_exclusive_calls{0};  ///< calls with it
    std::atomic<uint32_t> m_max_instance{0};
    std::atomic<uint32_t> m_max_entry{0};
    // The reset boundary: the context reset saw, and whether the process that followed saw
    // the same one; a reset of a handle that is not exclusive.
    std::atomic<const anira_engine_ctx*> m_pending_reset_ctx{nullptr};
    std::atomic<int> m_process_after_reset_same_ctx{0};
    std::atomic<int> m_process_after_reset_other_ctx{0};
    std::atomic<uint32_t> m_reset_num_inputs{0};
    std::atomic<uint32_t> m_reset_entry{0};
    std::atomic<uint32_t> m_reset_on_shared{0};
    // The query: what it answers (a bitmask over the descriptor's providers list), how often
    // it ran, what it saw, and whether it refuses.
    uint64_t m_available = ~uint64_t{0};
    int m_queried = 0;
    anira_status m_refuse_query = ANIRA_OK;
    uint32_t m_query_struct_size = 0;
    const anira_context* m_query_context = nullptr;
    int m_queried_after_init = 0;  ///< a query that ran after init (none is expected)
};

/// Whether `loaded` is a pointer load handed back that unload has not received (the counts,
/// since the allocator may hand a freed address out again).
bool loaded_is_live(const GainEngine& engine, void* loaded) {
    return std::ranges::count(engine.m_loaded_out, loaded) >
           std::ranges::count(engine.m_loaded_back, loaded);
}

/// Counts the gain engine's prepares that are inside at once over its scope and keeps the
/// peak; the exit waits m_prepare_delay_ms first, so a prepare on another thread has the time
/// to arrive while this one is inside (the overlap anira's rule forbids).
class PrepareWatch {
public:
    explicit PrepareWatch(GainEngine& engine) : m_engine(engine) {
        const int inside = m_engine.m_inside_prepare.fetch_add(1) + 1;
        int seen = m_engine.m_max_inside_prepare.load();
        while (inside > seen &&
               !m_engine.m_max_inside_prepare.compare_exchange_weak(seen, inside)) {}
    }
    ~PrepareWatch() {
        if (m_engine.m_prepare_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_engine.m_prepare_delay_ms));
        }
        m_engine.m_inside_prepare.fetch_sub(1);
    }
    PrepareWatch(const PrepareWatch&) = delete;
    PrepareWatch& operator=(const PrepareWatch&) = delete;
    PrepareWatch(PrepareWatch&&) = delete;
    PrepareWatch& operator=(PrepareWatch&&) = delete;

private:
    GainEngine& m_engine;
};

/// Whether `tensor` is `expected` in dtype, rank and extents.
bool same_shape(const anira_tensor& tensor, const anira_tensor& expected) {
    if (tensor.dtype != expected.dtype || tensor.ndim != expected.ndim) { return false; }
    for (uint32_t axis = 0; axis < expected.ndim && axis < ANIRA_MAX_RANK; ++axis) {
        if (tensor.shape[axis] != expected.shape[axis]) { return false; }
    }
    return true;
}

anira_status ANIRA_CALL gain_query(const anira_init_info* info,
                                   void* user_data,
                                   uint64_t* out_available) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_queried;
    if (engine->m_inited > 0) { ++engine->m_queried_after_init; }
    if (info == nullptr || out_available == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    engine->m_query_struct_size = info->struct_size;
    engine->m_query_context = info->context;
    if (engine->m_refuse_query != ANIRA_OK) { return engine->m_refuse_query; }
    *out_available = engine->m_available;
    return ANIRA_OK;
}

anira_status ANIRA_CALL gain_init(const anira_init_info* info, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_inited;
    if (engine->m_refuse_init != ANIRA_OK) { return engine->m_refuse_init; }
    if (info == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    engine->m_init_struct_size = info->struct_size;
    engine->m_init_log_level = info->log_level;
    engine->m_init_num_threads = info->num_threads;
    engine->m_init_context = info->context;
    return ANIRA_OK;
}

anira_status ANIRA_CALL gain_load(const anira_engine_load_info* info,
                                  void* user_data,
                                  void** out_loaded) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_loaded;
    if (engine->m_refuse_load != ANIRA_OK) { return engine->m_refuse_load; }
    if (info == nullptr || info->struct_size != sizeof(anira_engine_load_info) ||
        info->model == nullptr || out_loaded == nullptr || *out_loaded != nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    auto loaded = std::make_unique<GainLoaded>();
    loaded->m_row = info->row;
    // The row's facts through the config getters on the variant: the engine's own way to the
    // model, which anira never opened.
    const char* path = anira_model_config_model_path(info->model, info->row);
    const char* id = anira_model_config_model_engine_id(info->model, info->row);
    loaded->m_path = path != nullptr ? path : "";
    loaded->m_engine_id = id != nullptr ? id : "";
    loaded->m_instances = info->instances;
    loaded->m_provider = info->provider;
    loaded->m_provider_id = info->provider_id != nullptr ? info->provider_id : "";
    for (uint32_t k = 0; k < info->num_options; ++k) {
        loaded->m_options.emplace_back(info->option_keys[k], info->option_values[k]);
    }
    for (uint32_t i = 0; i < info->num_inputs; ++i) {
        loaded->m_inputs.push_back(info->inputs[i]);
        loaded->m_input_names.emplace_back(info->input_names[i]);
    }
    for (uint32_t i = 0; i < info->num_outputs; ++i) {
        loaded->m_outputs.push_back(info->outputs[i]);
        loaded->m_output_names.emplace_back(info->output_names[i]);
    }
    engine->m_instances_seen.push_back(info->instances);
    engine->m_providers_seen.emplace_back(loaded->m_provider, loaded->m_provider_id);
    *out_loaded = loaded.release();
    engine->m_loaded_out.push_back(*out_loaded);
    return ANIRA_OK;
}

void ANIRA_CALL gain_unload(void* loaded, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_unloaded;
    engine->m_loaded_back.push_back(loaded);
    auto* kept = static_cast<GainLoaded*>(loaded);
    if (kept != nullptr && kept->m_live_prepared != 0) { ++engine->m_unload_before_unprepare; }
    delete kept;
}

anira_status ANIRA_CALL gain_prepare(const anira_prepare_info* info,
                                     void* loaded,
                                     void* user_data,
                                     void** out_prepared) {
    auto* engine = static_cast<GainEngine*>(user_data);
    const PrepareWatch watch(*engine);
    ++engine->m_prepared;
    if (engine->m_refuse_prepare != ANIRA_OK) { return engine->m_refuse_prepare; }
    // The record is the stage's (the handler, its report), the loaded pointer one load handed
    // back and unload has not received, and the record's lists are the load record's.
    auto* kept = static_cast<GainLoaded*>(loaded);
    if (info == nullptr || info->struct_size != sizeof(anira_prepare_info) ||
        info->handler == nullptr || info->report == nullptr || out_prepared == nullptr ||
        *out_prepared != nullptr || !loaded_is_live(*engine, loaded) ||
        info->num_inputs != kept->m_inputs.size() || info->num_outputs != kept->m_outputs.size()) {
        ++engine->m_bad_prepare_record;
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    auto prepared = std::make_unique<GainPrepared>();
    prepared->m_loaded = kept;
    prepared->m_exclusive = (info->flags & ANIRA_PREPARE_EXCLUSIVE) != 0U;
    prepared->m_handler = info->handler;
    prepared->m_num_entries = info->num_entries;
    ++kept->m_live_prepared;
    engine->m_flags_seen.push_back(info->flags);
    *out_prepared = prepared.release();
    engine->m_handed_out.push_back(*out_prepared);
    return ANIRA_OK;
}

anira_status ANIRA_CALL gain_process(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    engine->m_processed.fetch_add(1);
    const auto* handle = static_cast<const GainPrepared*>(prepared);
    if (handle == nullptr || handle->m_loaded == nullptr) {
        engine->m_bad_prepared.fetch_add(1);
        return ANIRA_ERROR_INTERNAL;
    }
    const GainLoaded* kept = handle->m_loaded;
    if (ctx->loaded != kept) { engine->m_bad_loaded.fetch_add(1); }
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
    // The scalars: no ticket, nothing reserved; an exclusive handle's call carries the flag
    // and instance 0 (it claimed no shared slot), a shared handle's carries no flag and a slot
    // below the loaded model's count.
    const bool exclusive_call = (ctx->flags & ANIRA_ENGINE_CALL_EXCLUSIVE) != 0U;
    if (exclusive_call) {
        engine->m_exclusive_calls.fetch_add(1);
    } else {
        engine->m_shared_calls.fetch_add(1);
    }
    const bool instance_ok = exclusive_call
                                 ? ctx->flags == ANIRA_ENGINE_CALL_EXCLUSIVE && ctx->instance == 0
                                 : ctx->flags == 0 && ctx->instance < kept->m_instances;
    if (ctx->ticket != ANIRA_TICKET_INVALID || ctx->reserved0 != 0 || ctx->reserved1 != 0 ||
        ctx->reserved_ptr1 != nullptr || exclusive_call != handle->m_exclusive || !instance_ok) {
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

void ANIRA_CALL gain_reset(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    engine->m_resets.fetch_add(1);
    engine->m_pending_reset_ctx.store(ctx);
    const auto* handle = static_cast<const GainPrepared*>(prepared);
    if (handle == nullptr || !handle->m_exclusive) { engine->m_reset_on_shared.fetch_add(1); }
    if (ctx != nullptr) {
        engine->m_reset_num_inputs.store(ctx->num_inputs);
        engine->m_reset_entry.store(ctx->entry);
    }
}

void ANIRA_CALL gain_unprepare(void* prepared, void* user_data) {
    auto* engine = static_cast<GainEngine*>(user_data);
    ++engine->m_unprepared;
    engine->m_handed_back.push_back(prepared);
    const std::unique_ptr<GainPrepared> handle(static_cast<GainPrepared*>(prepared));
    // The loaded model outlives its prepared handles: still out, so it may be touched (an
    // unload that came first was counted there).
    if (handle != nullptr && loaded_is_live(*engine, handle->m_loaded)) {
        --handle->m_loaded->m_live_prepared;
    }
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
    desc.load = gain_load;
    desc.unload = gain_unload;
    desc.init = gain_init;
    desc.release = gain_release;
    desc.query = gain_query;
    return desc;
}

/// The load record behind a handed-out loaded pointer.
const GainLoaded& kept_of(const GainEngine& engine, size_t index = 0) {
    return *static_cast<const GainLoaded*>(engine.m_loaded_out.at(index));
}

/// The prepared handle behind a handed-out prepared pointer.
const GainPrepared& handle_of(const GainEngine& engine, size_t index = 0) {
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

// Every ANIRA_ERROR_INVALID_ARGUMENT cause of the engine's create, named in the message: the id
// (NULL, no reverse-URI name, or anira's own), a NULL desc or out, a short struct_size, a flags
// bit the header does not define, a NULL process and a NULL kinds array or entry with a count.
// A refused create hands out nothing and never calls release. The id of a create that succeeds
// is the carrier's own copy; the one id with anira's prefix a host may create is
// anira.v2.custom, the 2.x custom backend's.
TEST(AbiEngine, CreateRefusals) {
    EngineLife life;
    const anira_engine_desc engine = full_engine(life);
    const auto refused = [&life](const char* id,
                                 const anira_engine_desc* desc,
                                 const char* fragment) {
        anira_error err = ANIRA_ERROR_INIT;
        anira_custom_engine* handle = nullptr;
        EXPECT_EQ(anira_custom_engine_create(id, desc, &handle, &err), ANIRA_ERROR_INVALID_ARGUMENT)
            << fragment;
        EXPECT_NE(std::strstr(err.message, fragment), nullptr) << err.message;
        EXPECT_EQ(handle, nullptr);
        EXPECT_EQ(life.m_released, 0);
    };
    // The id: present, a reverse-URI name, and not one of anira's own.
    refused(nullptr, &engine, "NULL engine_id");
    refused("gain", &engine, "'.'");
    refused("anira.gain", &engine, "anira.");
    refused("anira.v2.custom.other", &engine, "anira.");
    refused(k_gain_id, nullptr, "NULL desc");
    {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_custom_engine_create(k_gain_id, &engine, nullptr, &err),
                  ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::strstr(err.message, "NULL out"), nullptr) << err.message;
    }
    anira_engine_desc bad = engine;
    bad.struct_size = static_cast<uint32_t>(offsetof(anira_engine_desc, user_data));
    refused(k_gain_id, &bad, "struct_size");
    bad = engine;
    bad.flags = 16U;  // the bit above the four the header defines
    refused(k_gain_id, &bad, "flags");
    bad.flags = 0x80000000U | ANIRA_ENGINE_FLAG_REALTIME_SAFE;
    refused(k_gain_id, &bad, "flags");
    bad = engine;
    bad.process = nullptr;
    refused(k_gain_id, &bad, "process");
    bad = engine;
    bad.num_consumed_kinds = 1;
    refused(k_gain_id, &bad, "consumed_kinds");
    const std::array<const char*, 1> null_kind{nullptr};
    bad.consumed_kinds = null_kind.data();
    refused(k_gain_id, &bad, "consumed_kinds[0]");
    // The same descriptor, well formed, creates under a well-formed id: the refusals were the
    // arguments'. The id is copied (the caller's string dies first); the handle holds the only
    // reference, so its destroy releases.
    anira_error err = ANIRA_ERROR_INIT;
    anira_custom_engine* handle = nullptr;
    {
        const std::string id = "org.example.created";
        ASSERT_EQ(anira_custom_engine_create(id.c_str(), &engine, &handle, &err), ANIRA_OK)
            << err.message;
    }
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->m_carrier->id(), "org.example.created");
    anira_custom_engine_destroy(handle);
    EXPECT_EQ(life.m_released, 1);
    anira_custom_engine_destroy(nullptr);  // NULL-safe
    // The exception to the prefix: anira.v2.custom is created like any other id.
    handle = nullptr;
    ASSERT_EQ(anira_custom_engine_create(k_custom, &engine, &handle, &err), ANIRA_OK)
        << err.message;
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->m_carrier->id(), k_custom);
    anira_custom_engine_destroy(handle);
    EXPECT_EQ(life.m_released, 2);
}

// Every ANIRA_ERROR_INVALID_ARGUMENT cause of the addition, named in the message: a NULL
// pipeline and a NULL engine (the id was checked at create). A refused addition takes no
// reference: the engine stays the handle's alone.
TEST(AbiEngine, AddRefusals) {
    Pipe pipe;
    EngineLife life;
    const anira_engine_desc desc = full_engine(life);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    {
        anira_error null_err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_add_engine(nullptr, engine, &null_err),
                  ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_NE(std::strstr(null_err.message, "NULL pipeline"), nullptr) << null_err.message;
    }
    EXPECT_EQ(pipe.add_engine(nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(pipe.m_err.message, "NULL engine"), nullptr) << pipe.m_err.message;
    EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
    EXPECT_EQ(pipe.add_engine(engine), ANIRA_OK) << pipe.m_err.message;
    anira_custom_engine_destroy(engine);
    EXPECT_EQ(life.m_released, 0) << "the pipeline holds it";
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// A caller compiled against a header that ends after process hands over the slots up to it:
// the rest reads as ANIRA_ENGINE_DESC_INIT, so the carrier's engine has no reset, no prepare,
// no unprepare, no load, no unload, no init and no release, whatever the memory beyond
// struct_size holds; the kinds and the id are copied into the carrier.
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
        ASSERT_EQ(pipe.add_new_engine(id.c_str(), engine), ANIRA_OK) << pipe.m_err.message;
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
    EXPECT_EQ(kept.load, nullptr);
    EXPECT_EQ(kept.unload, nullptr);
    EXPECT_EQ(kept.init, nullptr);
    EXPECT_EQ(kept.release, nullptr);
    EXPECT_EQ(kept.providers, nullptr);
    EXPECT_EQ(kept.num_providers, 0U);
    EXPECT_TRUE(carrier.serves(ANIRA_PROVIDER_DEFAULT, ""));
    EXPECT_FALSE(carrier.serves(ANIRA_PROVIDER_CUDA, ""));
    ASSERT_EQ(carrier.consumed_kinds().size(), 1U);
    EXPECT_EQ(carrier.consumed_kinds()[0], "model:entry");
    ASSERT_NE(kept.consumed_kinds, nullptr);
    EXPECT_NE(kept.consumed_kinds, kinds.data()) << "copied into the carrier";
    EXPECT_STREQ(kept.consumed_kinds[0], "model:entry");
    pipe.destroy();
    EXPECT_EQ(life.m_released, 0) << "the release slot lay beyond struct_size";
}

// The providers an engine lists are copied into the carrier (the caller's array may die) and
// answer serves(): DEFAULT always, a provider of the enum through its spelling, a custom one by
// its name; a NULL or empty entry, or a NULL list with a count, is refused at create.
TEST(AbiEngine, TheProvidersListIsCopiedAndAnswersServes) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);
    {
        const std::string coreml = "coreml";  // dies before the carrier does
        const std::string npu = "com.example.npu";
        const std::array<const char*, 2> providers{coreml.c_str(), npu.c_str()};
        engine.providers = providers.data();
        engine.num_providers = 2;
        ASSERT_EQ(pipe.add_new_engine("org.example.served", engine), ANIRA_OK)
            << pipe.m_err.message;
    }
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    const anira::capi::EngineCarrier& carrier = *pipe.m_pipeline->m_engines[0];
    EXPECT_EQ(carrier.providers(), (std::vector<std::string>{"coreml", "com.example.npu"}));
    ASSERT_EQ(carrier.desc().num_providers, 2U);
    EXPECT_STREQ(carrier.desc().providers[1], "com.example.npu");
    EXPECT_TRUE(carrier.serves(ANIRA_PROVIDER_DEFAULT, ""));
    EXPECT_TRUE(carrier.serves(ANIRA_PROVIDER_COREML, ""));
    EXPECT_TRUE(carrier.serves(ANIRA_PROVIDER_DEFAULT, "com.example.npu"));
    EXPECT_FALSE(carrier.serves(ANIRA_PROVIDER_CUDA, ""));
    EXPECT_FALSE(carrier.serves(ANIRA_PROVIDER_DEFAULT, "com.example.other"));
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);

    anira_engine_desc bad = full_engine(life);
    bad.num_providers = 1;  // a NULL list with a count
    anira_custom_engine* out = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create("org.example.served", &bad, &out, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "providers"), nullptr) << err.message;
    const std::array<const char*, 1> empty{""};
    bad.providers = empty.data();
    EXPECT_EQ(anira_custom_engine_create("org.example.served", &bad, &out, &err),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(out, nullptr);
    EXPECT_EQ(life.m_released, 1) << "a refused create never calls release";
}

// The abi_version of the descriptor is checked as anira_check_abi checks it: another major is
// ANIRA_ERROR_ABI_VERSION, no carrier, no release; the header's own registers.
TEST(AbiEngine, AbiVersionIsChecked) {
    Pipe pipe;
    EngineLife life;
    anira_engine_desc engine = full_engine(life);
    engine.abi_version = ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1U, 0U);
    EXPECT_EQ(pipe.add_new_engine("org.example.gain", engine), ANIRA_ERROR_ABI_VERSION);
    EXPECT_NE(std::strstr(pipe.m_err.message, "ABI"), nullptr) << pipe.m_err.message;
    EXPECT_TRUE(pipe.m_pipeline->m_engines.empty());
    engine.abi_version = ANIRA_ABI_VERSION;
    EXPECT_EQ(pipe.add_new_engine("org.example.gain", engine), ANIRA_OK) << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 1U);
    pipe.destroy();
    EXPECT_EQ(life.m_released, 1);
}

// One engine per id in a pipeline: an addition whose engine carries an id the pipeline has is
// ANIRA_ERROR_INVALID_STATE naming it and takes no reference, whether it brings another object
// under the id or the engine already there; an engine with another id is added beside the
// first.
TEST(AbiEngine, ADuplicateIdIsInvalidState) {
    Pipe pipe;
    EngineLife first;
    EngineLife second;
    EngineLife third;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_engine_desc first_desc = full_engine(first);
    const anira_engine_desc second_desc = full_engine(second);
    const anira_engine_desc third_desc = full_engine(third);
    anira_custom_engine* first_engine = nullptr;
    anira_custom_engine* second_engine = nullptr;
    anira_custom_engine* third_engine = nullptr;
    ASSERT_EQ(anira_custom_engine_create("org.example.twice", &first_desc, &first_engine, &err),
              ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_custom_engine_create("org.example.twice", &second_desc, &second_engine, &err),
              ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_custom_engine_create("org.example.other", &third_desc, &third_engine, &err),
              ANIRA_OK)
        << err.message;
    ASSERT_EQ(pipe.add_engine(first_engine), ANIRA_OK) << pipe.m_err.message;
    // Another object under the taken id.
    EXPECT_EQ(pipe.add_engine(second_engine), ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.twice"), nullptr) << pipe.m_err.message;
    EXPECT_NE(std::strstr(pipe.m_err.message, "another object"), nullptr) << pipe.m_err.message;
    // The engine already there, a second time.
    EXPECT_EQ(pipe.add_engine(first_engine), ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.twice"), nullptr) << pipe.m_err.message;
    EXPECT_NE(std::strstr(pipe.m_err.message, "already added"), nullptr) << pipe.m_err.message;
    // Another id is added beside the first.
    EXPECT_EQ(pipe.add_engine(third_engine), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 2U);
    EXPECT_EQ(pipe.m_pipeline->m_engines[0]->id(), "org.example.twice");
    EXPECT_EQ(pipe.m_pipeline->m_engines[1]->id(), "org.example.other");
    EXPECT_EQ(pipe.m_pipeline->m_engines[0], first_engine->m_carrier);
    anira_custom_engine_destroy(second_engine);
    EXPECT_EQ(second.m_released, 1) << "the refused addition took no reference";
    anira_custom_engine_destroy(first_engine);
    anira_custom_engine_destroy(third_engine);
    EXPECT_EQ(first.m_released, 0);
    EXPECT_EQ(third.m_released, 0);
    pipe.destroy();
    EXPECT_EQ(first.m_released, 1);
    EXPECT_EQ(third.m_released, 1);
}

// A registration is legal before and after anira_pipeline_add_inference, and a handler created
// from the pipeline carries the same carriers; a registered engine no entry names is not a
// plan and not an error, so the handler prepares and runs on the custom row's pass-through
// alone. A row naming a registered id is a plan: the handler creates, prepares through the
// engine's prepare and runs its stream through the engine's process.
TEST(AbiEngine, LegalBeforeAndAfterAddInference) {
    const Context context;
    Pipe pipe;
    EngineLife before;
    EngineLife after;
    ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.add_new_engine("org.example.before", full_engine(before)), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(stream_model());
    ASSERT_EQ(pipe.add_new_engine("org.example.after", full_engine(after)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.m_pipeline->m_engines.size(), 3U);

    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(handler->m_pipeline.m_engines.size(), 3U);
    EXPECT_EQ(handler->m_pipeline.m_engines[0], pipe.m_pipeline->m_engines[0]);
    EXPECT_EQ(handler->m_pipeline.m_engines[1], pipe.m_pipeline->m_engines[1]);
    EXPECT_EQ(handler->m_pipeline.m_engines[2], pipe.m_pipeline->m_engines[2]);

    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler);
    ASSERT_NE(report, nullptr);
    EXPECT_EQ(anira_plan_report_num_plans(report), 1U) << "the custom row alone is a plan";
    EXPECT_EQ(before.m_inited + after.m_inited, 0) << "no row names either engine";
    EXPECT_EQ(before.m_loaded + after.m_loaded, 0);
    EXPECT_EQ(before.m_prepared + after.m_prepared, 0);
    anira_handler_destroy(handler);
    EXPECT_EQ(before.m_released + after.m_released, 0) << "the pipeline still holds them";

    // A row naming a registered id, registered after the inference stage: the engine's init,
    // load and prepare run at prepare, its process on every inference, and the stream comes
    // back.
    GainEngine gain;
    Pipe named;
    named.add_inference(stream_model("org.example.after", k_never_opened));
    ASSERT_EQ(named.add_new_engine("org.example.after", gain_desc(gain)), ANIRA_OK)
        << named.m_err.message;
    anira_handler* runs = nullptr;
    ASSERT_EQ(named.create_handler(context, &runs), ANIRA_OK) << named.m_err.message;
    ASSERT_EQ(anira_handler_prepare(runs, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(runs)), 1U);
    EXPECT_EQ(gain.m_inited, 1);
    EXPECT_EQ(gain.m_loaded, 1);
    EXPECT_EQ(gain.m_prepared, 1);
    ASSERT_NO_FATAL_FAILURE(run_ramp(runs, 3));
    EXPECT_EQ(gain.m_processed.load(), 3);
    anira_handler_destroy(runs);
    EXPECT_EQ(gain.m_unprepared, 1);
    EXPECT_EQ(gain.m_unloaded, 1);
    named.destroy();
    EXPECT_EQ(gain.m_released, 1);
    pipe.destroy();
    EXPECT_EQ(before.m_released, 1) << "released without ever being initialised";
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
        ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
            << pipe.m_err.message;
        ASSERT_EQ(pipe.add_new_engine("org.example.shared", full_engine(life)), ANIRA_OK)
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
// and not the handler (which holds the custom row's pass-through alone), and its release fires
// with the pipeline, the handler sharing nothing of it.
TEST(AbiEngine, RegisteredAfterCreateDoesNotReachTheHandler) {
    const Context context;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(stream_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    EngineLife life;
    ASSERT_EQ(pipe.add_new_engine("org.example.late", full_engine(life)), ANIRA_OK)
        << pipe.m_err.message;
    EXPECT_EQ(pipe.m_pipeline->m_engines.size(), 2U);
    ASSERT_EQ(handler->m_pipeline.m_engines.size(), 1U);
    EXPECT_EQ(handler->m_pipeline.m_engines[0]->id(), k_custom);
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
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;

    // The three levels, once each: init with the facts of the core (the level, the pool of
    // this context, the handler's context), load with the row's record, prepare with the
    // stage's record and the loaded pointer.
    ASSERT_EQ(gain.m_inited, 1);
    ASSERT_EQ(gain.m_loaded, 1);
    ASSERT_EQ(gain.m_prepared, 1);
    EXPECT_EQ(gain.m_init_struct_size, sizeof(anira_init_info));
    EXPECT_EQ(gain.m_init_log_level, static_cast<uint32_t>(ANIRA_LOG_ERROR))
        << "the context's level, the one in effect";
    EXPECT_EQ(gain.m_init_num_threads, 2U) << "the pool of the context";
    EXPECT_EQ(gain.m_init_context, context.m_context);
    ASSERT_EQ(gain.m_loaded_out.size(), 1U);
    ASSERT_EQ(gain.m_handed_out.size(), 1U);
    const GainLoaded& kept = kept_of(gain);
    const GainPrepared& handle = handle_of(gain);
    EXPECT_EQ(handle.m_loaded, &kept);
    EXPECT_FALSE(handle.m_exclusive) << "a stateless model's handler";
    EXPECT_EQ(handle.m_handler, handler);
    EXPECT_EQ(handle.m_num_entries, anira_handler_num_entries(handler));
    EXPECT_EQ(gain.m_bad_prepare_record, 0);
    EXPECT_EQ(gain.m_flags_seen, (std::vector<uint32_t>{0}));
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
    EXPECT_EQ(gain.m_shared_calls.load(), 6U) << "a shared handle claims a slot per call";
    EXPECT_EQ(gain.m_exclusive_calls.load(), 0U);
    EXPECT_EQ(gain.m_bad_loaded.load(), 0U) << "ctx->loaded is what load handed back";
    EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
    EXPECT_EQ(gain.m_resets.load(), 0) << "a stateless model is never reset";
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_unprepared, 1);
    EXPECT_EQ(gain.m_unloaded, 1) << "the last handler took the loaded model with it";
    EXPECT_EQ(gain.m_unload_before_unprepare, 0) << "unprepare before unload";
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// ==== providers ===============================================================================

namespace {

/// One candidate per pair on the gain engine's id: a provider of the enum, or DEFAULT with a
/// custom name.
using ProviderPair = std::pair<anira_provider, const char*>;
std::vector<anira_backend_id> gain_candidates(std::initializer_list<ProviderPair> providers,
                                              const char* engine_id = k_gain_id) {
    std::vector<anira_backend_id> out;
    for (const auto& [provider, id] : providers) {
        out.push_back({.struct_size = sizeof(anira_backend_id),
                       .engine = ANIRA_ENGINE_NONE,
                       .provider = static_cast<uint32_t>(provider),
                       .engine_id = engine_id,
                       .provider_id = id});
    }
    return out;
}

/// The gain engine's descriptor listing `providers` beyond the default one.
anira_engine_desc gain_desc_serving(GainEngine& engine, std::span<const char* const> providers) {
    anira_engine_desc desc = gain_desc(engine);
    desc.providers = providers.empty() ? nullptr : providers.data();
    desc.num_providers = static_cast<uint32_t>(providers.size());
    return desc;
}

using SeenProvider = std::pair<uint32_t, std::string>;

}  // namespace

// One plan per row and candidate: a neutral row under three candidates of its engine (coreml,
// a custom name, the default provider) is three plans in candidate order, each on its own
// loaded model (the provider is part of the pool key: three loads with the provider in the
// record, three prepares, one init), the report naming the provider and the custom name per
// plan; set_plan switches between them while the stream runs, and every process call runs on
// the selected plan's loaded model and prepared handle.
TEST(AbiEngine, OneRowUnderThreeProvidersIsThreePlansAndThreeLoads) {
    const Context context;
    GainEngine gain;
    gain.m_gain = 0.5F;
    Pipe pipe;
    const std::array<const char*, 2> served{"coreml", "com.example.npu"};
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(gain_model(),
                       gain_candidates({{ANIRA_PROVIDER_COREML, nullptr},
                                        {ANIRA_PROVIDER_DEFAULT, "com.example.npu"},
                                        {ANIRA_PROVIDER_DEFAULT, nullptr}}));
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_inited, 1) << "init is the object's, once";
    EXPECT_EQ(gain.m_loaded, 3) << "a provider is part of the loaded model";
    EXPECT_EQ(gain.m_prepared, 3) << "one prepare per handler and loaded model";
    EXPECT_EQ(gain.m_providers_seen,
              (std::vector<SeenProvider>{{ANIRA_PROVIDER_COREML, ""},
                                         {ANIRA_PROVIDER_DEFAULT, "com.example.npu"},
                                         {ANIRA_PROVIDER_DEFAULT, ""}}))
        << "the loads in plan order, the record naming the provider";
    const std::vector<anira_plan_info> plans = plan_rows(handler);
    ASSERT_EQ(plans.size(), 3U);
    for (const anira_plan_info& plan : plans) {
        EXPECT_EQ(plan.engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
        ASSERT_NE(plan.engine_id, nullptr);
        EXPECT_STREQ(plan.engine_id, k_gain_id);
    }
    EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
    EXPECT_EQ(plans[0].provider_id, nullptr);
    EXPECT_EQ(plans[1].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
    ASSERT_NE(plans[1].provider_id, nullptr);
    EXPECT_STREQ(plans[1].provider_id, "com.example.npu");
    EXPECT_EQ(plans[2].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
    EXPECT_EQ(plans[2].provider_id, nullptr);
    EXPECT_EQ(anira_handler_get_plan(handler), 0U) << "no default engine named: plan 0";
    // Two blocks on each plan in turn: the ramp comes back times the gain on every plan, and
    // the engine sees the loaded pointer of the selected plan behind its prepared handle.
    for (uint32_t plan = 0; plan < 3; ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        ASSERT_EQ(anira_handler_set_plan(handler, plan), ANIRA_OK);
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2, gain.m_gain, 1 + 2 * plan));
        EXPECT_EQ(gain.m_processed.load(), static_cast<int>(2 * (plan + 1)));
        EXPECT_EQ(gain.m_bad_loaded.load(), 0U) << "ctx->loaded is the selected plan's";
        EXPECT_EQ(gain.m_bad_prepared.load(), 0U);
    }
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_unprepared, 3);
    EXPECT_EQ(gain.m_unloaded, 3);
    EXPECT_EQ(gain.m_unload_before_unprepare, 0);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// A candidate naming a provider the engine's descriptor does not list is ANIRA_ERROR_NOT_SUPPORTED
// at create, naming the entry, the engine, the provider and the list (a provider of the enum
// and a custom name alike); an engine whose descriptor lists no provider (the pass-through)
// serves the default provider alone; a candidate that matches no entry is not a plan and not an
// error, and two equal candidates are one plan.
TEST(AbiEngine, AProviderTheEngineDoesNotListIsNotSupportedAtCreate) {
    const Context context;
    GainEngine gain;
    const std::array<const char*, 1> served{"coreml"};
    const auto refused = [&](const std::vector<anira_backend_id>& candidates,
                             const std::vector<const char*>& fragments) {
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(gain_model(), candidates);
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED)
            << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        for (const char* fragment : fragments) {
            EXPECT_NE(std::strstr(pipe.m_err.message, fragment), nullptr)
                << fragment << " in: " << pipe.m_err.message;
        }
        EXPECT_EQ(gain.m_loaded, 0) << "nothing loads at create";
    };
    refused(gain_candidates({{ANIRA_PROVIDER_CUDA, nullptr}}),
            {"models[0]", k_gain_id, "'cuda'", "coreml"});
    refused(gain_candidates({{ANIRA_PROVIDER_DEFAULT, "com.example.other"}}),
            {"models[0]", k_gain_id, "'com.example.other'", "coreml"});
    refused(gain_candidates({{ANIRA_PROVIDER_DEFAULT, nullptr}, {ANIRA_PROVIDER_XNNPACK, nullptr}}),
            {"'xnnpack'"});
    {
        // The pass-through: the NONE entry with a provider keeps every custom row, and the
        // anira.v2.custom row's engine, which lists no provider, serves the default provider
        // alone.
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(stream_model(),

                           gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}, nullptr));
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED)
            << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(pipe.m_err.message, k_custom), nullptr) << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message, "'coreml'"), nullptr) << pipe.m_err.message;
    }
    {
        // A candidate that matches no entry (another id) is no plan and no error, whatever its
        // provider; two equal candidates are one plan.
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        std::vector<anira_backend_id> candidates =
            gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}, {ANIRA_PROVIDER_COREML, nullptr}});
        const std::vector<anira_backend_id> other =
            gain_candidates({{ANIRA_PROVIDER_CUDA, nullptr}}, "org.example.other");
        candidates.push_back(other[0]);
        pipe.add_inference(gain_model(), candidates);
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        const anira::ContractHandle contract = explicit_contract();
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        const std::vector<anira_plan_info> plans = plan_rows(handler);
        ASSERT_EQ(plans.size(), 1U);
        EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
        EXPECT_EQ(gain.m_loaded, 1);
        anira_handler_destroy(handler);
    }
}

// The pool key holds the provider: on one engine object and one variant, two handlers on the
// same provider share one loaded model (whichever pipelines they came from) and a handler on
// another provider loads again; unload with the last holder of each.
TEST(AbiEngine, ThePoolKeyHoldsTheProvider) {
    const Context context;
    GainEngine gain;
    const std::array<const char*, 1> served{"coreml"};
    const anira_engine_desc desc = gain_desc_serving(gain, served);
    anira_error err = ANIRA_ERROR_INIT;
    anira_custom_engine* engine = nullptr;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    Pipe on_coreml;
    Pipe on_default;
    ASSERT_EQ(on_coreml.add_engine(engine), ANIRA_OK) << on_coreml.m_err.message;
    ASSERT_EQ(on_default.add_engine(engine), ANIRA_OK) << on_default.m_err.message;
    anira_custom_engine_destroy(engine);
    on_coreml.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
    on_default.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_DEFAULT, nullptr}}));
    anira_handler* first = nullptr;
    anira_handler* second = nullptr;
    anira_handler* third = nullptr;
    ASSERT_EQ(on_coreml.create_handler(context, &first), ANIRA_OK) << on_coreml.m_err.message;
    ASSERT_EQ(on_coreml.create_handler(context, &second), ANIRA_OK) << on_coreml.m_err.message;
    ASSERT_EQ(on_default.create_handler(context, &third), ANIRA_OK) << on_default.m_err.message;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_loaded, 1) << "the same provider: one loaded model";
    ASSERT_EQ(anira_handler_prepare(third, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_loaded, 2) << "another provider: another loaded model";
    EXPECT_EQ(gain.m_prepared, 3);
    EXPECT_EQ(
        gain.m_providers_seen,
        (std::vector<SeenProvider>{{ANIRA_PROVIDER_COREML, ""}, {ANIRA_PROVIDER_DEFAULT, ""}}));
    EXPECT_EQ(gain.m_inited, 1);
    anira_handler_destroy(first);
    EXPECT_EQ(gain.m_unloaded, 0) << "the second handler still holds the coreml model";
    anira_handler_destroy(second);
    EXPECT_EQ(gain.m_unloaded, 1);
    anira_handler_destroy(third);
    EXPECT_EQ(gain.m_unloaded, 2);
    on_coreml.destroy();
    on_default.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

namespace {

// The query says which of the declared providers are usable here: a candidate on a declared
// provider whose bit the query clears is ANIRA_ERROR_NOT_SUPPORTED at create, the message
// saying the engine declares it but its query reports it unavailable here; a candidate on a
// provider the bit keeps creates. The query runs at create, before init, with the facts of the
// core (the record, the context); a query that refuses fails the create with its status,
// naming the engine and the phase.
TEST(AbiEngine, TheQuerySaysWhichDeclaredProvidersAreUsableHere) {
    const Context context;
    static constexpr std::array<const char*, 2> k_served{"coreml", "com.example.npu"};
    GainEngine gain;
    gain.m_available = 0b01;  // coreml yes, the npu no
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, k_served)), ANIRA_OK)
        << pipe.m_err.message;
    pipe.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    EXPECT_EQ(gain.m_queried, 1) << "one query per create";
    EXPECT_EQ(gain.m_inited, 0) << "before init";
    EXPECT_EQ(gain.m_query_struct_size, sizeof(anira_init_info));
    EXPECT_EQ(gain.m_query_context, context.m_context);
    anira_handler_destroy(handler);
    handler = nullptr;

    Pipe on_npu;
    ASSERT_EQ(on_npu.add_new_engine(k_gain_id, gain_desc_serving(gain, k_served)), ANIRA_OK)
        << on_npu.m_err.message;
    on_npu.add_inference(gain_model(),
                         gain_candidates({{ANIRA_PROVIDER_DEFAULT, "com.example.npu"}}));
    EXPECT_EQ(on_npu.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_EQ(handler, nullptr);
    EXPECT_NE(std::strstr(on_npu.m_err.message, "declares provider 'com.example.npu'"), nullptr)
        << on_npu.m_err.message;
    EXPECT_NE(std::strstr(on_npu.m_err.message, "unavailable here"), nullptr)
        << on_npu.m_err.message;
    EXPECT_NE(std::strstr(on_npu.m_err.message, "usable here: default, coreml"), nullptr)
        << on_npu.m_err.message;
    // A provider the descriptor does not declare at all says so, as before.
    Pipe on_cuda;
    ASSERT_EQ(on_cuda.add_new_engine(k_gain_id, gain_desc_serving(gain, k_served)), ANIRA_OK)
        << on_cuda.m_err.message;
    on_cuda.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_CUDA, nullptr}}));
    EXPECT_EQ(on_cuda.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(std::strstr(on_cuda.m_err.message, "does not serve provider 'cuda'"), nullptr)
        << on_cuda.m_err.message;
    EXPECT_NE(std::strstr(on_cuda.m_err.message,
                          "its descriptor lists: default, coreml, "
                          "com.example.npu"),
              nullptr)
        << on_cuda.m_err.message;
    // A refused query fails the create with its status, naming the engine and the phase.
    gain.m_refuse_query = ANIRA_ERROR_DEVICE;
    Pipe refused;
    ASSERT_EQ(refused.add_new_engine(k_gain_id, gain_desc_serving(gain, k_served)), ANIRA_OK)
        << refused.m_err.message;
    refused.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
    EXPECT_EQ(refused.create_handler(context, &handler), ANIRA_ERROR_DEVICE);
    EXPECT_NE(std::strstr(refused.m_err.message, "'org.example.gain' refused query"), nullptr)
        << refused.m_err.message;
    EXPECT_EQ(gain.m_queried_after_init, 0);
}

// A pipeline's capabilities are the context's rows and then one row per custom engine of the
// pipeline and provider usable here, the default provider first, then the declared providers
// the query keeps, with the engine's id; every call runs the queries. The edges follow the
// CPU rule: zero-copy to the default provider, a host copy the engine makes itself to a
// device one; a provider the query cleared, an engine the pipeline does not add and a domain
// other than the host's have no row; a built-in backend is the context's answer.
TEST(AbiEngine, ThePipelinesCapabilitiesListTheCustomEnginesRows) {
    const Context context;
    static constexpr std::array<const char*, 2> k_served{"coreml", "com.example.npu"};
    GainEngine gain;
    gain.m_available = 0b01;
    GainEngine other;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, k_served)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.add_new_engine("org.example.other", gain_desc(other)), ANIRA_OK)
        << pipe.m_err.message;
    uint32_t context_count = 0;
    ASSERT_EQ(anira_capabilities_backends(anira_context_capabilities(context.m_context),
                                          sizeof(anira_backend_id),
                                          &context_count,
                                          nullptr),
              ANIRA_OK);
    uint32_t count = 0;
    ASSERT_EQ(anira_pipeline_capabilities_backends(pipe.m_pipeline,
                                                   context.m_context,
                                                   sizeof(anira_backend_id),
                                                   &count,
                                                   nullptr),
              ANIRA_OK);
    EXPECT_EQ(count, context_count + 3U) << "gain: default + coreml; other: default";
    EXPECT_EQ(gain.m_queried, 1);
    EXPECT_EQ(other.m_queried, 1);
    std::vector<anira_backend_id> rows(count, ANIRA_BACKEND_ID_INIT);
    ASSERT_EQ(anira_pipeline_capabilities_backends(pipe.m_pipeline,
                                                   context.m_context,
                                                   sizeof(anira_backend_id),
                                                   &count,
                                                   rows.data()),
              ANIRA_OK);
    EXPECT_EQ(gain.m_queried, 2) << "every call runs the query";
    ASSERT_EQ(rows.size(), context_count + 3U);
    for (uint32_t i = 0; i < context_count; ++i) {
        EXPECT_EQ(rows[i].engine_id, nullptr) << "the context's rows lead";
    }
    const anira_backend_id& gain_default = rows[context_count];
    const anira_backend_id& gain_coreml = rows[context_count + 1];
    const anira_backend_id& other_default = rows[context_count + 2];
    EXPECT_EQ(gain_default.engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
    EXPECT_EQ(gain_default.provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
    EXPECT_STREQ(gain_default.engine_id, k_gain_id);
    EXPECT_EQ(gain_default.provider_id, nullptr);
    EXPECT_EQ(gain_coreml.provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
    EXPECT_STREQ(gain_coreml.engine_id, k_gain_id);
    EXPECT_STREQ(other_default.engine_id, "org.example.other");
    EXPECT_EQ(gain.m_inited, 0) << "the queries ran before any init";
    // A short buffer: filled as far as it goes, ANIRA_INCOMPLETE, the count the total.
    uint32_t one = 1;
    anira_backend_id first = ANIRA_BACKEND_ID_INIT;
    EXPECT_EQ(anira_pipeline_capabilities_backends(pipe.m_pipeline,
                                                   context.m_context,
                                                   sizeof(anira_backend_id),
                                                   &one,
                                                   &first),
              ANIRA_INCOMPLETE);
    EXPECT_EQ(one, count);
    EXPECT_EQ(anira_pipeline_capabilities_backends(nullptr, context.m_context, 0, &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_pipeline_capabilities_backends(pipe.m_pipeline, nullptr, 0, &count, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // The edges of the custom rows, and what has none.
    const auto edge = [&](anira_provider provider, const char* provider_id, const char* id) {
        anira_backend_id to = ANIRA_BACKEND_ID_INIT;
        to.engine = ANIRA_ENGINE_NONE;
        to.provider = static_cast<uint32_t>(provider);
        to.engine_id = id;
        to.provider_id = provider_id;
        anira_edge_info row = ANIRA_EDGE_INFO_INIT;
        const anira_status status = anira_pipeline_capabilities_edge(pipe.m_pipeline,
                                                                     context.m_context,
                                                                     ANIRA_DOMAIN_HOST,
                                                                     &to,
                                                                     &row);
        return std::pair<anira_status, anira_edge_info>{status, row};
    };
    const auto [default_status, default_edge] = edge(ANIRA_PROVIDER_DEFAULT, nullptr, k_gain_id);
    ASSERT_EQ(default_status, ANIRA_OK);
    EXPECT_EQ(default_edge.edge_class, static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY));
    EXPECT_EQ(default_edge.available, 1U);
    EXPECT_EQ(default_edge.to_engine, static_cast<uint32_t>(ANIRA_ENGINE_NONE));
    const auto [coreml_status, coreml_edge] = edge(ANIRA_PROVIDER_COREML, nullptr, k_gain_id);
    ASSERT_EQ(coreml_status, ANIRA_OK);
    EXPECT_EQ(coreml_edge.edge_class, static_cast<uint32_t>(ANIRA_EDGE_HOST_COPY));
    EXPECT_EQ(coreml_edge.rung, static_cast<uint32_t>(ANIRA_RUNG_IDENTITY));
    ASSERT_NE(coreml_edge.reason, nullptr);
    EXPECT_NE(std::strstr(coreml_edge.reason, "query"), nullptr) << coreml_edge.reason;
    EXPECT_EQ(edge(ANIRA_PROVIDER_DEFAULT, "com.example.npu", k_gain_id).first,
              ANIRA_ERROR_EDGE_UNREACHABLE)
        << "declared, but the query cleared it";
    EXPECT_EQ(edge(ANIRA_PROVIDER_CUDA, nullptr, k_gain_id).first, ANIRA_ERROR_EDGE_UNREACHABLE)
        << "not declared";
    EXPECT_EQ(edge(ANIRA_PROVIDER_DEFAULT, nullptr, "org.example.absent").first,
              ANIRA_ERROR_EDGE_UNREACHABLE)
        << "not added to the pipeline";
    {
        anira_backend_id to = ANIRA_BACKEND_ID_INIT;
        to.engine = ANIRA_ENGINE_NONE;
        to.engine_id = k_gain_id;
        anira_edge_info row = ANIRA_EDGE_INFO_INIT;
        EXPECT_EQ(anira_pipeline_capabilities_edge(pipe.m_pipeline,
                                                   context.m_context,
                                                   ANIRA_DOMAIN_CUDA,
                                                   &to,
                                                   &row),
                  ANIRA_ERROR_EDGE_UNREACHABLE)
            << "host memory alone reaches a custom engine";
        EXPECT_EQ(anira_pipeline_capabilities_edge(pipe.m_pipeline,
                                                   nullptr,
                                                   ANIRA_DOMAIN_HOST,
                                                   &to,
                                                   &row),
                  ANIRA_ERROR_INVALID_ARGUMENT);
    }
    // A built-in backend: the context's registry answers, through the pipeline entry.
    if (context_count > 0) {
        const anira_backend_id builtin = rows[0];
        anira_edge_info from_pipeline = ANIRA_EDGE_INFO_INIT;
        anira_edge_info from_context = ANIRA_EDGE_INFO_INIT;
        ASSERT_EQ(anira_pipeline_capabilities_edge(pipe.m_pipeline,
                                                   context.m_context,
                                                   ANIRA_DOMAIN_HOST,
                                                   &builtin,
                                                   &from_pipeline),
                  ANIRA_OK);
        ASSERT_EQ(anira_capabilities_edge(anira_context_capabilities(context.m_context),
                                          ANIRA_DOMAIN_HOST,
                                          &builtin,
                                          &from_context),
                  ANIRA_OK);
        EXPECT_EQ(from_pipeline.edge_class, from_context.edge_class);
        EXPECT_EQ(from_pipeline.to_engine, from_context.to_engine);
    }
    // A refused query fails the entry with its status.
    gain.m_refuse_query = ANIRA_ERROR_DEVICE;
    EXPECT_EQ(anira_pipeline_capabilities_backends(pipe.m_pipeline,
                                                   context.m_context,
                                                   sizeof(anira_backend_id),
                                                   &count,
                                                   nullptr),
              ANIRA_ERROR_DEVICE);
    EXPECT_EQ(edge(ANIRA_PROVIDER_DEFAULT, nullptr, k_gain_id).first, ANIRA_ERROR_DEVICE);
}

// A context whose config carries a provider_options extension in JSON, for the tests of the
// sets against a registered engine.
struct OptionsContext {
    explicit OptionsContext(const char* sets_json) {
        EXPECT_EQ(anira_context_config_create(&m_config, &m_err), ANIRA_OK) << m_err.message;
        EXPECT_EQ(anira_context_config_set_threads(m_config, 2, ANIRA_WAIT_SPIN_BACKOFF), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_level(m_config, ANIRA_LOG_ERROR), ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_log_drain(m_config, ANIRA_LOG_DRAIN_MANUAL, 10),
                  ANIRA_OK);
        EXPECT_EQ(anira_context_config_set_ext_json(m_config,
                                                    "provider_options",
                                                    sets_json,
                                                    std::strlen(sets_json),
                                                    &m_err),
                  ANIRA_OK)
            << m_err.message;
        m_status = anira_context_create(m_config, &m_context, &m_err);
    }
    ~OptionsContext() {
        anira_context_destroy(m_context);
        anira_context_config_destroy(m_config);
    }
    OptionsContext(const OptionsContext&) = delete;
    OptionsContext& operator=(const OptionsContext&) = delete;
    OptionsContext(OptionsContext&&) = delete;
    OptionsContext& operator=(OptionsContext&&) = delete;

    anira_context_config* m_config = nullptr;
    anira_context* m_context = nullptr;
    anira_status m_status = ANIRA_OK;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// The gain engine's descriptor serving coreml and, when asked, listing the provider_options
/// kind.
anira_engine_desc gain_desc_options(GainEngine& engine, bool consumes) {
    static constexpr std::array<const char*, 1> k_served{"coreml"};
    static constexpr std::array<const char*, 1> k_kinds{"context:provider_options"};
    anira_engine_desc desc = gain_desc_serving(engine, k_served);
    if (consumes) {
        desc.consumed_kinds = k_kinds.data();
        desc.num_consumed_kinds = 1;
    }
    return desc;
}

}  // namespace

// A registered engine whose descriptor lists "context:provider_options" receives the set of
// its backend in its load record, in key order; the options are part of the loaded
// model, so two contexts with different sets load twice and two with equal sets share.
TEST(AbiEngine, AProviderOptionsSetReachesARegisteredEngineThatConsumesIt) {
    const OptionsContext first(
        R"({"sets": [{"engine": "org.example.gain", "provider": "coreml",
                      "options": {"units": "all", "cache": "off"}}]})");
    ASSERT_EQ(first.m_status, ANIRA_OK) << first.m_err.message;
    const OptionsContext second(
        R"({"sets": [{"engine": "org.example.gain", "provider": "coreml",
                      "options": {"units": "cpu"}}]})");
    ASSERT_EQ(second.m_status, ANIRA_OK) << second.m_err.message;
    const OptionsContext third(
        R"({"sets": [{"engine": "org.example.gain", "provider": "coreml",
                      "options": {"units": "all", "cache": "off"}}]})");
    ASSERT_EQ(third.m_status, ANIRA_OK) << third.m_err.message;
    GainEngine gain;
    const anira_engine_desc desc = gain_desc_options(gain, true);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    Pipe pipe;
    ASSERT_EQ(pipe.add_engine(engine), ANIRA_OK) << pipe.m_err.message;
    anira_custom_engine_destroy(engine);
    pipe.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
    anira_handler* on_first = nullptr;
    anira_handler* on_second = nullptr;
    anira_handler* on_third = nullptr;
    ASSERT_EQ(anira_handler_create(first.m_context, pipe.m_pipeline, &on_first, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_create(second.m_context, pipe.m_pipeline, &on_second, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_create(third.m_context, pipe.m_pipeline, &on_third, &err), ANIRA_OK)
        << err.message;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(on_first, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(gain.m_loaded, 1);
    EXPECT_EQ(kept_of(gain).m_provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
    EXPECT_EQ(
        kept_of(gain).m_options,
        (std::vector<std::pair<std::string, std::string>>{{"cache", "off"}, {"units", "all"}}));
    ASSERT_EQ(anira_handler_prepare(on_second, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_loaded, 2) << "another set: another loaded model";
    EXPECT_EQ(kept_of(gain, 1).m_options,
              (std::vector<std::pair<std::string, std::string>>{{"units", "cpu"}}));
    ASSERT_EQ(anira_handler_prepare(on_third, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_loaded, 2) << "an equal set: the first loaded model, shared";
    EXPECT_EQ(anira_test::session_of(on_first)->m_plans.at(0).m_loaded,
              anira_test::session_of(on_third)->m_plans.at(0).m_loaded);
    anira_handler_destroy(on_first);
    anira_handler_destroy(on_second);
    anira_handler_destroy(on_third);
    EXPECT_EQ(gain.m_unloaded, 2);
}

// A set is checked against its backend at anira_handler_create: a set for a registered engine
// of the pipeline whose descriptor does not list the kind is refused as unconsumed, naming the
// engine, and never rides into the plan's key; a set naming a provider the engine's descriptor
// does not list is ANIRA_ERROR_NOT_SUPPORTED naming the provider, so a misspelled word is never
// silently ignored; a set for a custom engine this pipeline does not add is another pipeline's
// and passes. A set for a built-in engine names a provider the context serves, else the same
// refusal.
TEST(AbiEngine, AProviderOptionsSetIsCheckedAgainstItsBackendAtCreate) {
    const OptionsContext no_reader(
        R"({"sets": [{"engine": "org.example.gain", "provider": "coreml", "options": {"u": "1"}}]})");
    ASSERT_EQ(no_reader.m_status, ANIRA_OK) << no_reader.m_err.message;
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_options(gain, false)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
        anira_handler* handler = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_handler_create(no_reader.m_context, pipe.m_pipeline, &handler, &err),
                  ANIRA_ERROR_EXTENSION_UNCONSUMED);
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(err.message, "org.example.gain"), nullptr) << err.message;
        EXPECT_NE(std::strstr(err.message, "context:provider_options"), nullptr) << err.message;
    }
    const OptionsContext unserved(
        R"({"sets": [{"engine": "org.example.gain", "provider": "cuda", "options": {"u": "1"}}]})");
    ASSERT_EQ(unserved.m_status, ANIRA_OK) << unserved.m_err.message;
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_options(gain, true)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
        anira_handler* handler = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_handler_create(unserved.m_context, pipe.m_pipeline, &handler, &err),
                  ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(err.message, "'cuda'"), nullptr) << err.message;
        EXPECT_NE(std::strstr(err.message, "coreml"), nullptr)
            << "the served list: " << err.message;
    }
    const OptionsContext other_engine(
        R"({"sets": [{"engine": "org.example.other", "provider": "cuda", "options": {"u": "1"}}]})");
    ASSERT_EQ(other_engine.m_status, ANIRA_OK) << other_engine.m_err.message;
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_options(gain, false)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(gain_model(), gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}}));
        anira_handler* handler = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_create(other_engine.m_context, pipe.m_pipeline, &handler, &err),
                  ANIRA_OK)
            << err.message;
        anira_handler_destroy(handler);
    }
#ifdef USE_ONNXRUNTIME
    // A built-in engine's set names a provider the context's capabilities do not list: a
    // custom name no execution provider of this ONNX Runtime has.
    const OptionsContext misspelled(
        R"({"sets": [{"engine": "onnxruntime", "provider": "com.example.cdua", "options": {"device_id": "0"}}]})");
    ASSERT_EQ(misspelled.m_status, ANIRA_OK) << misspelled.m_err.message;
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        pipe.add_inference(gain_model());
        anira_handler* handler = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_handler_create(misspelled.m_context, pipe.m_pipeline, &handler, &err),
                  ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(err.message, "'com.example.cdua'"), nullptr) << err.message;
        EXPECT_NE(std::strstr(err.message, "onnxruntime"), nullptr) << err.message;
    }
#endif
}

// The candidates of anira_pipeline_add_inference are read at the caller's stride, the first
// record's struct_size: a caller whose record is wider than the library's (a later header,
// here a padded layout) hands its array over as it is, and a record whose struct_size differs
// from the first's is refused naming it.
TEST(AbiEngine, CandidatesAreReadAtTheCallersStride) {
    const Context context;
    constexpr size_t k_stride = sizeof(anira_backend_id) + 16;
    std::array<unsigned char, 2 * k_stride> bytes{};
    const std::vector<anira_backend_id> ids =
        gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}, {ANIRA_PROVIDER_DEFAULT, nullptr}});
    for (size_t i = 0; i < ids.size(); ++i) {
        anira_backend_id wide = ids[i];
        wide.struct_size = static_cast<uint32_t>(k_stride);
        std::memcpy(bytes.data() + i * k_stride, &wide, sizeof(wide));
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) the caller's array at its stride
    const auto* candidates = reinterpret_cast<const anira_backend_id*>(bytes.data());
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_options(gain, false)), ANIRA_OK)
        << pipe.m_err.message;
    const ModelConfig model = gain_model();
    const std::array<const anira_model_config*, 1> variants{model.native()};
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(
        anira_pipeline_add_inference(pipe.m_pipeline, variants.data(), 1, candidates, 2, &err),
        ANIRA_OK)
        << err.message;
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(handler)), 2U)
        << "both candidates, the second read at the caller's stride";
    anira_handler_destroy(handler);

    // A second record unlike the first: refused, naming it.
    anira_backend_id narrow = ids[1];
    std::memcpy(bytes.data() + k_stride, &narrow, sizeof(narrow));
    const Pipe other;
    EXPECT_EQ(
        anira_pipeline_add_inference(other.m_pipeline, variants.data(), 1, candidates, 2, &err),
        ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "candidates[1]"), nullptr) << err.message;
    EXPECT_NE(std::strstr(err.message, "one stride"), nullptr) << err.message;
}

// A pinned entry runs on its provider alone: two entries of one engine are legal when their
// pins differ; under explicit candidates the pinned entry matches the candidate naming its pin
// and no other while the neutral entry matches every candidate of the engine; under the
// default set the pinned entry runs on its pin and the neutral one on the default provider;
// two entries with one pin, or two without one, are one entry too many.
TEST(AbiEngine, APinnedEntryRunsOnItsProviderAlone) {
    const Context context;
    GainEngine gain;
    const std::array<const char*, 1> served{"coreml"};
    // models[0] pinned to coreml (an export lowered to it), models[1] neutral.
    ModelConfig model = stream_model(k_gain_id, "lowered.bin");
    model.model_provider(0, ANIRA_PROVIDER_COREML);
    model.add_model_path(k_gain_id, k_never_opened);
    const anira::ContractHandle contract = explicit_contract();
    {
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(
            model,
            gain_candidates({{ANIRA_PROVIDER_COREML, nullptr}, {ANIRA_PROVIDER_DEFAULT, nullptr}}));
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        // The table: (0, coreml), (1, coreml), (1, default), in entry and candidate order.
        ASSERT_EQ(handler->m_plans.size(), 3U);
        EXPECT_EQ(handler->m_plans[0].m_row, 0U);
        EXPECT_EQ(handler->m_plans[1].m_row, 1U);
        EXPECT_EQ(handler->m_plans[2].m_row, 1U);
        const std::vector<anira_plan_info> plans = plan_rows(handler);
        ASSERT_EQ(plans.size(), 3U);
        EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
        EXPECT_EQ(plans[1].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
        EXPECT_EQ(plans[2].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
        EXPECT_EQ(gain.m_loaded, 3) << "three keys: the row or the provider differs";
        EXPECT_EQ(kept_of(gain, 0).m_path, "lowered.bin");
        EXPECT_EQ(kept_of(gain, 0).m_provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
        EXPECT_EQ(kept_of(gain, 2).m_path, k_never_opened);
        EXPECT_EQ(kept_of(gain, 2).m_provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
        anira_handler_destroy(handler);
    }
    {
        // The default set: the pin is a candidate of its own engine.
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(model);
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        const std::vector<anira_plan_info> plans = plan_rows(handler);
        ASSERT_EQ(plans.size(), 2U);
        EXPECT_EQ(handler->m_plans[0].m_row, 0U);
        EXPECT_EQ(plans[0].provider, static_cast<uint32_t>(ANIRA_PROVIDER_COREML));
        EXPECT_EQ(handler->m_plans[1].m_row, 1U);
        EXPECT_EQ(plans[1].provider, static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT));
        anira_handler_destroy(handler);
    }
    {
        // A pinned entry no candidate names is no plan: the neutral entry alone runs.
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(model, gain_candidates({{ANIRA_PROVIDER_DEFAULT, nullptr}}));
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_EQ(handler->m_plans.size(), 1U);
        EXPECT_EQ(handler->m_plans[0].m_row, 1U);
        anira_handler_destroy(handler);
    }
    for (const bool pinned : {true, false}) {
        SCOPED_TRACE(pinned ? "two pinned to coreml" : "two neutral");
        ModelConfig twice = stream_model(k_gain_id, "a.bin");
        twice.add_model_path(k_gain_id, "b.bin");
        if (pinned) {
            twice.model_provider(0, ANIRA_PROVIDER_COREML);
            twice.model_provider(1, ANIRA_PROVIDER_COREML);
        }
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc_serving(gain, served)), ANIRA_OK)
            << pipe.m_err.message;
        pipe.add_inference(twice);
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_CONFIG) << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(pipe.m_err.message, "both name engine"), nullptr)
            << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message,
                              pinned ? "pinned to provider 'coreml'" : "without a provider pin"),
                  nullptr)
            << pipe.m_err.message;
    }
}

// A custom row whose id no engine of the pipeline serves is ANIRA_ERROR_NOT_SUPPORTED at create,
// naming the id and the entry that adds one; an engine under another id does not serve it.
TEST(AbiEngine, AnUnregisteredCustomIdIsNotSupported) {
    const Context context;
    for (const bool other_registered : {false, true}) {
        SCOPED_TRACE(other_registered ? "another id registered" : "nothing registered");
        GainEngine gain;  // outlives the pipeline, whose carrier's release reads it
        Pipe pipe;
        if (other_registered) {
            ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK)
                << pipe.m_err.message;
        }
        pipe.add_inference(stream_model("org.example.nobody", k_never_opened));
        anira_handler* handler = nullptr;
        EXPECT_EQ(pipe.create_handler(context, &handler), ANIRA_ERROR_NOT_SUPPORTED)
            << pipe.m_err.message;
        EXPECT_EQ(handler, nullptr);
        EXPECT_NE(std::strstr(pipe.m_err.message, "org.example.nobody"), nullptr)
            << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message, "is not added to this pipeline"), nullptr)
            << pipe.m_err.message;
        EXPECT_NE(std::strstr(pipe.m_err.message, "anira_pipeline_add_engine"), nullptr)
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

// Two handlers of one pipeline over one model share one loaded model: the engine's load runs
// once, with `instances` = the model's max_instances, its prepare once per handler with the
// stage's record and that loaded pointer, and both handlers run on it (shared calls, a slot
// each). A stateful model is loaded once too: exclusivity is the handler's, not the model's,
// so the load says 0 shared slots, each handler's prepare carries ANIRA_PREPARE_EXCLUSIVE and
// builds what it runs on, and every call of theirs carries ANIRA_ENGINE_CALL_EXCLUSIVE with
// instance 0. Either way unprepare runs per handler, unload once with the last of them, and
// never before the handles are back.
TEST(AbiEngine, EqualConfigsShareOneLoadedModelStatefulOrNot) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
        EXPECT_EQ(gain.m_inited, 1) << "once per engine object";
        EXPECT_EQ(gain.m_loaded, 1) << "one loaded model for two equal handlers";
        EXPECT_EQ(gain.m_prepared, 2) << "a prepare per handler";
        EXPECT_EQ(gain.m_instances_seen, (std::vector<uint32_t>{2}));
        EXPECT_EQ(gain.m_flags_seen, (std::vector<uint32_t>{0, 0}));
        EXPECT_EQ(gain.m_bad_prepare_record, 0);
        EXPECT_EQ(anira_test::session_of(first)->m_plans.at(0).m_loaded,
                  anira_test::session_of(second)->m_plans.at(0).m_loaded)
            << "the pooled loaded model";
        EXPECT_EQ(anira_test::session_of(first)->m_plans.at(0).m_loaded->model().m_instances, 2U)
            << "the record agrees with the load record";
        ASSERT_EQ(gain.m_handed_out.size(), 2U);
        EXPECT_EQ(handle_of(gain, 0).m_loaded, handle_of(gain, 1).m_loaded);
        EXPECT_EQ(handle_of(gain, 0).m_handler, first);
        EXPECT_EQ(handle_of(gain, 1).m_handler, second);
        ASSERT_NO_FATAL_FAILURE(run_ramp(first, 3));
        ASSERT_NO_FATAL_FAILURE(run_ramp(second, 3));
        EXPECT_EQ(gain.m_processed.load(), 6);
        EXPECT_EQ(gain.m_shared_calls.load(), 6U);
        EXPECT_EQ(gain.m_exclusive_calls.load(), 0U);
        EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
        EXPECT_EQ(gain.m_bad_loaded.load(), 0U);
        anira_handler_destroy(first);
        EXPECT_EQ(gain.m_unprepared, 1) << "the first handler's handle went back";
        EXPECT_EQ(gain.m_unloaded, 0) << "the second handler still shares the loaded model";
        anira_handler_destroy(second);
        EXPECT_EQ(gain.m_unprepared, 2);
        EXPECT_EQ(gain.m_unloaded, 1);
        EXPECT_EQ(gain.m_unload_before_unprepare, 0);
        pipe.destroy();
        EXPECT_EQ(gain.m_released, 1);
    }
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
        EXPECT_EQ(gain.m_loaded, 1) << "a stateful model is loaded once too";
        EXPECT_EQ(gain.m_prepared, 2) << "each handler builds its own executor at prepare";
        EXPECT_EQ(gain.m_instances_seen, (std::vector<uint32_t>{0}))
            << "no shared slot: the handlers run on what their prepare built";
        EXPECT_EQ(gain.m_flags_seen,
                  (std::vector<uint32_t>{ANIRA_PREPARE_EXCLUSIVE, ANIRA_PREPARE_EXCLUSIVE}));
        EXPECT_EQ(anira_test::session_of(first)->m_plans.at(0).m_loaded,
                  anira_test::session_of(second)->m_plans.at(0).m_loaded)
            << "pooled: exclusivity is not in the key";
        for (const anira_handler* handler : {first, second}) {
            const anira::SessionElement::PlanSlot& plan =
                anira_test::session_of(handler)->m_plans.at(0);
            EXPECT_EQ(plan.m_loaded->model().m_instances, 0U)
                << "the record agrees with the load record";
            ASSERT_NE(plan.m_prepared, nullptr);
            EXPECT_TRUE(plan.m_prepared->exclusive());
        }
        ASSERT_EQ(gain.m_handed_out.size(), 2U);
        EXPECT_NE(gain.m_handed_out[0], gain.m_handed_out[1]);
        EXPECT_TRUE(handle_of(gain, 0).m_exclusive && handle_of(gain, 1).m_exclusive);
        ASSERT_NO_FATAL_FAILURE(run_ramp(first, 3));
        ASSERT_NO_FATAL_FAILURE(run_ramp(second, 3));
        EXPECT_EQ(gain.m_processed.load(), 6);
        EXPECT_EQ(gain.m_exclusive_calls.load(), 6U)
            << "an exclusive handle's calls carry the flag and instance 0";
        EXPECT_EQ(gain.m_shared_calls.load(), 0U);
        EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
        EXPECT_EQ(gain.m_bad_loaded.load(), 0U);
        EXPECT_EQ(gain.m_resets.load(), 2) << "one stream start per handler";
        EXPECT_EQ(gain.m_reset_on_shared.load(), 0U);
        anira_handler_destroy(first);
        EXPECT_EQ(gain.m_unprepared, 1);
        EXPECT_EQ(gain.m_unloaded, 0);
        anira_handler_destroy(second);
        EXPECT_EQ(gain.m_unprepared, 2);
        EXPECT_EQ(gain.m_unloaded, 1);
        EXPECT_EQ(gain.m_unload_before_unprepare, 0);
        pipe.destroy();
        EXPECT_EQ(gain.m_released, 1);
    }
}

// A pooled loaded model is every equal handler's, and a handler's prepare runs the engine's
// prepare on its caller's thread, outside the core's lifecycle lock: anira serialises the
// prepares (and unprepares) of one loaded model, so handlers prepared at once on several threads
// reach the engine's prepare one after the other, never inside each other. Each prepare holds
// its exit for a while, so the others have the time to arrive while it is inside.
TEST(AbiEngine, PreparesOfOneLoadedModelNeverOverlap) {
    constexpr int k_handlers = 4;
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    GainEngine gain;
    gain.m_prepare_delay_ms = 20;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.max_instances(k_handlers);
    pipe.add_inference(model);
    std::array<anira_handler*, k_handlers> handlers{};
    for (anira_handler*& handler : handlers) {
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    }
    std::array<anira_status, k_handlers> statuses{};
    std::array<anira_error, k_handlers> errs{};
    std::vector<std::thread> threads;
    threads.reserve(k_handlers);
    for (int i = 0; i < k_handlers; ++i) {
        errs.at(i) = ANIRA_ERROR_INIT;
        threads.emplace_back([&, i] {
            statuses.at(i) = anira_handler_prepare(handlers.at(i), contract.native(), &errs.at(i));
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    for (int i = 0; i < k_handlers; ++i) {
        EXPECT_EQ(statuses.at(i), ANIRA_OK) << errs.at(i).message;
    }
    EXPECT_EQ(gain.m_loaded, 1) << "one loaded model for the equal handlers";
    EXPECT_EQ(gain.m_prepared, k_handlers);
    EXPECT_EQ(gain.m_max_inside_prepare.load(), 1)
        << "the prepares of one loaded model never overlap";
    EXPECT_EQ(gain.m_bad_prepare_record, 0);
    for (anira_handler* handler : handlers) { ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2)); }
    for (anira_handler* handler : handlers) { anira_handler_destroy(handler); }
    EXPECT_EQ(gain.m_unprepared, k_handlers);
    EXPECT_EQ(gain.m_unloaded, 1);
    EXPECT_EQ(gain.m_unload_before_unprepare, 0);
}

// A session-exclusive handler's loaded model has no shared call slot whatever the model's
// max_instances, on the registered engine's row and on the row of every built-in engine of the
// build alike (the built-in engines size their shared executors from the record, the
// registered engine reads the same count in its load record), and the handler's prepared
// handle is exclusive on every plan: the records agree.
TEST(AbiEngine, ASessionExclusiveHandlerHasNoSharedSlotOnAnyEngine) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    // The bundled gain model on every engine of the build plus the custom row on its
    // pass-through.
    ModelConfig model = anira_test::gain_with_custom(false);
    model.state(ANIRA_MODEL_STATEFUL).max_instances(2);
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << pipe.m_err.message;
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
        ASSERT_NE(plan.m_loaded, nullptr);
        EXPECT_EQ(plan.m_loaded->model().m_instances, 0U) << "the plan of engine " << plan.m_engine;
        EXPECT_EQ(plan.m_loaded->num_instances(), 0U) << "the plan of engine " << plan.m_engine;
        ASSERT_NE(plan.m_prepared, nullptr) << "the plan of engine " << plan.m_engine;
        EXPECT_TRUE(plan.m_prepared->exclusive()) << "the plan of engine " << plan.m_engine;
    }
    anira_handler_destroy(handler);
}

// Three prepares of one handler are three prepared handles and, the handler being the only
// holder, three loaded models: three prepares and three loads, three unprepares and three
// unloads (two each at the re-prepares, one at destroy), each unprepare and unload receiving
// the pointer its prepare or load handed back, in order, the handle before the model; init
// runs once.
TEST(AbiEngine, UnprepareAndUnloadOncePerPrepareAndLoad) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    const anira::ContractHandle contract = explicit_contract();
    for (int prepare = 1; prepare <= 3; ++prepare) {
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        EXPECT_EQ(gain.m_inited, 1);
        EXPECT_EQ(gain.m_loaded, prepare);
        EXPECT_EQ(gain.m_prepared, prepare);
        EXPECT_EQ(gain.m_unprepared, prepare - 1) << "the previous handle went back";
        EXPECT_EQ(gain.m_unloaded, prepare - 1) << "and the previous loaded model with it";
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
    }
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_prepared, 3);
    EXPECT_EQ(gain.m_unprepared, 3);
    EXPECT_EQ(gain.m_loaded, 3);
    EXPECT_EQ(gain.m_unloaded, 3);
    EXPECT_EQ(gain.m_handed_back, gain.m_handed_out);
    EXPECT_EQ(gain.m_loaded_back, gain.m_loaded_out);
    EXPECT_EQ(gain.m_unload_before_unprepare, 0);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// unprepare is not deferred by an inference thread: while another handler keeps the core's
// pool alive, the thread that ran the last inference of a handler lets go of that handler's
// session with the job, so the re-prepare and the destroy of the handler unprepare its
// prepared handle before they return, whatever ran on it. (A session-exclusive model, so that
// each handler has an executor of its own to watch; the loaded model is one, shared.)
TEST(AbiEngine, UnprepareIsNotDeferredByAnInferenceThreadKeepingTheLastJob) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
    EXPECT_EQ(gain.m_loaded, 1) << "one loaded model, stateful or not";
    // The re-prepare of the first handler gives its prepared handle back before it returns,
    // the second handler (and with it the pool, its threads and the loaded model) staying
    // alive.
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_unprepared, 1) << "unprepared at the re-prepare, not at a later dequeue";
    EXPECT_EQ(gain.m_prepared, 3);
    EXPECT_EQ(gain.m_loaded, 1) << "the second handler held the loaded model throughout";
    EXPECT_EQ(gain.m_unloaded, 0);
    ASSERT_NO_FATAL_FAILURE(run_ramp(first, 2));
    anira_handler_destroy(first);
    EXPECT_EQ(gain.m_unprepared, 2) << "unprepared at the destroy";
    ASSERT_NO_FATAL_FAILURE(run_ramp(second, 2, 1.0F, 4));
    anira_handler_destroy(second);
    EXPECT_EQ(gain.m_unprepared, 3);
    EXPECT_EQ(gain.m_unloaded, 1) << "with the last handler";
    EXPECT_EQ(gain.m_unload_before_unprepare, 0);
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

// A refusal at any level fails anira_handler_prepare with the engine's status, naming the
// engine and the slot, and leaves the handler unprepared. A refused init leaves the engine
// uninitialised (nothing is loaded behind it, and the next prepare calls init again); a
// refused load owes no unload (nothing is prepared behind it); a refused prepare owes no
// unprepare, and the loaded model the failed prepare had acquired goes back with the session
// the failure releases.
TEST(AbiEngine, ARefusalAtAnyLevelFailsThePrepareAndOwesNothing) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();

    gain.m_refuse_init = ANIRA_ERROR_NOT_SUPPORTED;
    EXPECT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(std::strstr(err.message, k_gain_id), nullptr) << err.message;
    EXPECT_NE(std::strstr(err.message, "refused init"), nullptr) << err.message;
    EXPECT_EQ(anira_handler_plan_report(handler), nullptr);
    EXPECT_EQ(gain.m_inited, 1);
    EXPECT_EQ(gain.m_loaded, 0) << "nothing loaded behind a refused init";
    EXPECT_EQ(gain.m_prepared, 0);
    gain.m_refuse_init = ANIRA_OK;

    gain.m_refuse_load = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(err.message, k_gain_id), nullptr) << err.message;
    EXPECT_NE(std::strstr(err.message, "refused load"), nullptr) << err.message;
    EXPECT_EQ(anira_handler_plan_report(handler), nullptr);
    EXPECT_EQ(gain.m_inited, 2) << "init again, since the first was refused";
    EXPECT_EQ(gain.m_loaded, 1);
    EXPECT_EQ(gain.m_unloaded, 0) << "no unload for a refused load";
    EXPECT_EQ(gain.m_prepared, 0) << "nothing prepared behind a refused load";
    gain.m_refuse_load = ANIRA_OK;

    gain.m_refuse_prepare = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(err.message, k_gain_id), nullptr) << err.message;
    EXPECT_NE(std::strstr(err.message, "refused prepare"), nullptr) << err.message;
    EXPECT_EQ(anira_handler_plan_report(handler), nullptr);
    EXPECT_EQ(gain.m_inited, 2) << "initialised: not again";
    EXPECT_EQ(gain.m_loaded, 2);
    EXPECT_EQ(gain.m_unloaded, 1) << "the failed prepare released its session's loaded model";
    EXPECT_EQ(gain.m_prepared, 1);
    EXPECT_EQ(gain.m_unprepared, 0) << "no unprepare for a refused prepare";
    // The engine relents: the next prepare succeeds, and destroy gives that one's handle and
    // loaded model back alone.
    gain.m_refuse_prepare = ANIRA_OK;
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(gain.m_loaded, 3);
    EXPECT_EQ(gain.m_prepared, 2);
    anira_handler_destroy(handler);
    EXPECT_EQ(gain.m_unprepared, 1);
    EXPECT_EQ(gain.m_unloaded, 2);
    EXPECT_EQ(gain.m_unload_before_unprepare, 0);
    pipe.destroy();
    EXPECT_EQ(gain.m_released, 1);
}

// init runs once per engine object, at the first anira_handler_prepare that reaches it (a row
// of the engine survives validation), with the facts of the core in effect: the level, the
// pool of the context, the context of that handler; a second handler, a re-prepare and a
// second pipeline over the same object do not run it again, and the object is released once,
// after everything.
TEST(AbiEngine, InitOncePerEngineObjectAtTheFirstPrepareThatReachesIt) {
    const Context context(2);
    const anira::ContractHandle contract = explicit_contract();
    GainEngine gain;
    const anira_engine_desc desc = gain_desc(gain);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    Pipe first;
    Pipe second;
    ASSERT_EQ(first.add_engine(engine), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.add_engine(engine), ANIRA_OK) << second.m_err.message;
    anira_custom_engine_destroy(engine);
    const ModelConfig model = gain_model();
    first.add_inference(model);
    second.add_inference(model);
    anira_handler* first_handler = nullptr;
    anira_handler* second_handler = nullptr;
    ASSERT_EQ(first.create_handler(context, &first_handler), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.create_handler(context, &second_handler), ANIRA_OK) << second.m_err.message;
    EXPECT_EQ(gain.m_inited, 0) << "init belongs to the first prepare";
    ASSERT_EQ(anira_handler_prepare(first_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(gain.m_inited, 1);
    EXPECT_EQ(gain.m_init_struct_size, sizeof(anira_init_info));
    EXPECT_EQ(gain.m_init_log_level, static_cast<uint32_t>(ANIRA_LOG_ERROR))
        << "the context's level, the one in effect";
    EXPECT_EQ(gain.m_init_num_threads, 2U) << "the pool of the context";
    EXPECT_EQ(gain.m_init_context, context.m_context);
    ASSERT_EQ(anira_handler_prepare(second_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_prepare(first_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(gain.m_inited, 1) << "once, whichever pipeline's handler comes next";
    EXPECT_EQ(gain.m_loaded, 1) << "one engine, one model, two pipelines: one loaded model";
    EXPECT_EQ(gain.m_prepared, 3);
    anira_handler_destroy(first_handler);
    anira_handler_destroy(second_handler);
    first.destroy();
    second.destroy();
    EXPECT_EQ(gain.m_inited, 1);
    EXPECT_EQ(gain.m_unloaded, 1);
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
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
        EXPECT_EQ(gain.m_reset_on_shared.load(), 0U) << "every reset on the exclusive handle";
        EXPECT_EQ(gain.m_exclusive_calls.load(), 8U);
        EXPECT_EQ(gain.m_shared_calls.load(), 0U);
        anira_handler_destroy(handler);
    }
    {
        GainEngine gain;
        Pipe pipe;
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
        pipe.add_inference(gain_model());
        anira_handler* handler = nullptr;
        ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
        anira_error err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        anira_handler_reset(handler);
        ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 3));
        EXPECT_EQ(gain.m_resets.load(), 0) << "a shared handle is never reset";
        EXPECT_EQ(gain.m_processed.load(), 6);
        EXPECT_EQ(gain.m_shared_calls.load(), 6U);
        EXPECT_EQ(gain.m_exclusive_calls.load(), 0U);
        anira_handler_destroy(handler);
    }
}

// The reset boundary is the plan's: after anira_handler_reset every plan of an exclusive
// handler is reset at its own first inference of the new stream, whichever plan ran first,
// and a plan switch alone resets nothing.
TEST(AbiEngine, EveryPlanIsResetAtItsOwnFirstInferenceOfANewStream) {
    const Context context;
    const anira::ContractHandle contract = explicit_contract();
    GainEngine first;
    GainEngine second;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(first)), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(pipe.add_new_engine("org.example.second", gain_desc(second)), ANIRA_OK)
        << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.add_model_path("org.example.second", k_never_opened);
    model.state(ANIRA_MODEL_STATEFUL);
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(plan_rows(handler).size(), 2U);
    ASSERT_EQ(anira_handler_get_plan(handler), 0U);
    // One block per call, the ramp continued across the calls of a stream (both engines pass
    // the block through, one block late).
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 1));
    EXPECT_EQ(first.m_resets.load(), 1) << "the first plan's first inference after prepare";
    EXPECT_EQ(second.m_resets.load(), 0);
    ASSERT_EQ(anira_handler_set_plan(handler, 1), ANIRA_OK);
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 1, 1.0F, 2));
    EXPECT_EQ(second.m_resets.load(), 1) << "the second plan's first inference of the stream";
    ASSERT_EQ(anira_handler_set_plan(handler, 0), ANIRA_OK);
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 1, 1.0F, 3));
    EXPECT_EQ(first.m_resets.load(), 1) << "a plan switch alone is no boundary";
    anira_handler_reset(handler);
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 1));
    EXPECT_EQ(first.m_resets.load(), 2) << "the first plan, at its first inference after the reset";
    EXPECT_EQ(second.m_resets.load(), 1);
    ASSERT_EQ(anira_handler_set_plan(handler, 1), ANIRA_OK);
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 1, 1.0F, 2));
    EXPECT_EQ(second.m_resets.load(), 2)
        << "the second plan too, at its own first inference of the new stream";
    EXPECT_EQ(first.m_processed.load() + second.m_processed.load(), 5);
    anira_handler_destroy(handler);
}

// A detached handle names its engine for as long as a pipeline, a handler or a loaded model
// holds it, and keeps it alive no longer: it adds the engine to a later pipeline while one
// lives, the addition is INVALID_STATE once the engine was released, a handle nothing else
// holds the engine through releases it at the detach, and a second detach is a no-op.
TEST(AbiEngine, ADetachedHandleNamesTheEngineWhileSomethingHoldsIt) {
    EngineLife life;
    const anira_engine_desc desc = full_engine(life);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    {
        Pipe first;
        ASSERT_EQ(first.add_engine(engine), ANIRA_OK) << first.m_err.message;
        anira_custom_engine_detach(engine);
        anira_custom_engine_detach(engine);
        EXPECT_EQ(life.m_released, 0) << "the pipeline holds the engine";
        Pipe second;
        ASSERT_EQ(second.add_engine(engine), ANIRA_OK) << second.m_err.message;
        EXPECT_EQ(first.m_pipeline->m_engines.at(0), second.m_pipeline->m_engines.at(0))
            << "the same engine, through the detached handle";
    }
    EXPECT_EQ(life.m_released, 1) << "the last pipeline released it";
    {
        Pipe third;
        EXPECT_EQ(third.add_engine(engine), ANIRA_ERROR_INVALID_STATE);
        EXPECT_NE(std::strstr(third.m_err.message, "released"), nullptr) << third.m_err.message;
    }
    anira_custom_engine_destroy(engine);
    EXPECT_EQ(life.m_released, 1);

    // Nothing else holds it: the detach releases the engine.
    EngineLife alone;
    const anira_engine_desc alone_desc = full_engine(alone);
    anira_custom_engine* unadded = nullptr;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &alone_desc, &unadded, &err), ANIRA_OK)
        << err.message;
    anira_custom_engine_detach(unadded);
    EXPECT_EQ(alone.m_released, 1);
    anira_custom_engine_destroy(unadded);
    EXPECT_EQ(alone.m_released, 1);
    anira_custom_engine_detach(nullptr);
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
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain, k_flags)), ANIRA_OK)
        << pipe.m_err.message;
    // Two custom rows, two plans: the flagged engine's and the pass-through's, whose descriptor
    // carries flags 0.
    ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << pipe.m_err.message;
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
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    model.tensor_name(0, "out", "y_out");
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(gain.m_loaded_out.size(), 1U);
    EXPECT_EQ(kept_of(gain).m_input_names, (std::vector<std::string>{"in"}));
    EXPECT_EQ(kept_of(gain).m_output_names, (std::vector<std::string>{"y_out"}));
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 8));
    EXPECT_EQ(gain.m_processed.load(), 8);
    EXPECT_EQ(gain.m_bad_prepared.load(), 0U);
    EXPECT_EQ(gain.m_bad_loaded.load(), 0U);
    EXPECT_EQ(gain.m_bad_counts.load(), 0U);
    EXPECT_EQ(gain.m_bad_tensors.load(), 0U);
    EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
    EXPECT_LT(gain.m_max_instance.load(), kept_of(gain).m_instances);
    EXPECT_LT(gain.m_max_entry.load(), anira_handler_num_entries(handler));
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
    anira_handler_destroy(handler);
}

// A row with a layout record: the engine's side of a slot is the spec's extents in the export's
// order, a unit axis inserted where the export has one the spec lacks, and what the load
// record's templates say is what every process call sees, over the chunk's memory as it is (a
// layout moves no data: the spec's [1, 1, hop] and the export's [1, hop, 1] are one packed
// block, so the ramp comes back as it went in).
TEST(AbiEngine, ProcessSeesTheEngineSideExtentsOfALayoutRow) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ModelConfig model = gain_model();
    const std::array<uint32_t, 3> time_before_channel{0, 2, 1};
    const std::array<uint32_t, 4> with_a_unit_axis{0, ANIRA_AXIS_INSERT, 1, 2};
    model.tensor_layout(0, "in", time_before_channel);
    model.tensor_layout(0, "out", with_a_unit_axis);
    pipe.add_inference(model);
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(gain.m_loaded_out.size(), 1U);
    const GainLoaded& kept = kept_of(gain);
    ASSERT_EQ(kept.m_inputs.size(), 1U);
    ASSERT_EQ(kept.m_outputs.size(), 1U);
    EXPECT_EQ(kept.m_inputs[0].ndim, 3U);
    EXPECT_EQ(kept.m_inputs[0].shape[0], 1);
    EXPECT_EQ(kept.m_inputs[0].shape[1], static_cast<int64_t>(k_hop));
    EXPECT_EQ(kept.m_inputs[0].shape[2], 1);
    EXPECT_EQ(kept.m_outputs[0].ndim, 4U);
    EXPECT_EQ(kept.m_outputs[0].shape[0], 1);
    EXPECT_EQ(kept.m_outputs[0].shape[1], 1);
    EXPECT_EQ(kept.m_outputs[0].shape[2], 1);
    EXPECT_EQ(kept.m_outputs[0].shape[3], static_cast<int64_t>(k_hop));
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 4));
    EXPECT_EQ(gain.m_processed.load(), 4);
    EXPECT_EQ(gain.m_bad_counts.load(), 0U);
    EXPECT_EQ(gain.m_bad_tensors.load(), 0U) << "every call carries the templates' extents";
    EXPECT_EQ(gain.m_bad_scalars.load(), 0U);
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
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, desc), ANIRA_OK) << pipe.m_err.message;
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
        ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
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
    ASSERT_EQ(pipe.add_new_engine("org.example.idle", gain_desc(idle)), ANIRA_OK)
        << pipe.m_err.message;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(named)), ANIRA_OK) << pipe.m_err.message;
    pipe.add_inference(gain_model());
    anira_handler* handler = nullptr;
    ASSERT_EQ(pipe.create_handler(context, &handler), ANIRA_OK) << pipe.m_err.message;
    anira_error err = ANIRA_ERROR_INIT;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(handler, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(handler)), 1U);
    EXPECT_EQ(named.m_prepared, 1);
    EXPECT_EQ(idle.m_prepared, 0);
    EXPECT_EQ(idle.m_inited, 0) << "never reached by a prepare: never initialised";
    ASSERT_NO_FATAL_FAILURE(run_ramp(handler, 2));
    EXPECT_EQ(idle.m_processed.load(), 0);
    anira_handler_destroy(handler);
    pipe.destroy();
    EXPECT_EQ(idle.m_released, 1);
    EXPECT_EQ(named.m_released, 1);
}

// The carrier is in the pool's key: two engine objects (two creates, even over equal
// descriptors) under one id in two pipelines over one model share nothing, each engine's init,
// load, prepare, unprepare and unload run once, and the two handlers run on two loaded models.
TEST(AbiEngine, TwoEngineObjectsUnderOneIdShareNothing) {
    const Context context;
    GainEngine first_engine;
    GainEngine second_engine;
    Pipe first;
    Pipe second;
    ASSERT_EQ(first.add_new_engine(k_gain_id, gain_desc(first_engine)), ANIRA_OK)
        << first.m_err.message;
    ASSERT_EQ(second.add_new_engine(k_gain_id, gain_desc(second_engine)), ANIRA_OK)
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
    EXPECT_EQ(first_engine.m_inited, 1);
    EXPECT_EQ(second_engine.m_inited, 1);
    EXPECT_EQ(first_engine.m_loaded, 1);
    EXPECT_EQ(second_engine.m_loaded, 1);
    EXPECT_EQ(first_engine.m_prepared, 1);
    EXPECT_EQ(second_engine.m_prepared, 1);
    EXPECT_NE(anira_test::session_of(first_handler)->m_plans.at(0).m_loaded,
              anira_test::session_of(second_handler)->m_plans.at(0).m_loaded);
    ASSERT_NO_FATAL_FAILURE(run_ramp(first_handler, 2));
    ASSERT_NO_FATAL_FAILURE(run_ramp(second_handler, 2));
    EXPECT_EQ(first_engine.m_processed.load(), 2);
    EXPECT_EQ(second_engine.m_processed.load(), 2);
    anira_handler_destroy(first_handler);
    anira_handler_destroy(second_handler);
    EXPECT_EQ(first_engine.m_unprepared, 1);
    EXPECT_EQ(second_engine.m_unprepared, 1);
    EXPECT_EQ(first_engine.m_unloaded, 1);
    EXPECT_EQ(second_engine.m_unloaded, 1);
    first.destroy();
    second.destroy();
    EXPECT_EQ(first_engine.m_released, 1);
    EXPECT_EQ(second_engine.m_released, 1);
}

// The engine object is the pool's key, not the pipeline: one engine added to two pipelines
// (under its one id, the handle destroyed right after the additions) over equal model
// configurations gives two handlers one loaded model: one init, one load, a prepare per
// handler, one unload with the last handler; release fires once, when the pipelines and the
// handlers are all gone.
TEST(AbiEngine, OneEngineInTwoPipelinesSharesOneLoadedModel) {
    const Context context;
    const anira::ContractHandle contract = explicit_contract();
    GainEngine shared;
    const anira_engine_desc shared_desc = gain_desc(shared);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &shared_desc, &engine, &err), ANIRA_OK)
        << err.message;
    Pipe first;
    Pipe second;
    ASSERT_EQ(first.add_engine(engine), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.add_engine(engine), ANIRA_OK) << second.m_err.message;
    anira_custom_engine_destroy(engine);  // the pipelines hold it now
    EXPECT_EQ(first.m_pipeline->m_engines.at(0), second.m_pipeline->m_engines.at(0));
    const ModelConfig model = gain_model();
    first.add_inference(model);
    second.add_inference(model);
    anira_handler* first_handler = nullptr;
    anira_handler* second_handler = nullptr;
    ASSERT_EQ(first.create_handler(context, &first_handler), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.create_handler(context, &second_handler), ANIRA_OK) << second.m_err.message;
    first.destroy();
    second.destroy();
    EXPECT_EQ(shared.m_released, 0) << "the handlers hold it";
    ASSERT_EQ(anira_handler_prepare(first_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_prepare(second_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(shared.m_inited, 1) << "one engine object: one init";
    EXPECT_EQ(shared.m_loaded, 1) << "one engine, one model: one load";
    EXPECT_EQ(shared.m_prepared, 2) << "a prepare per handler";
    EXPECT_EQ(anira_test::session_of(first_handler)->m_plans.at(0).m_loaded,
              anira_test::session_of(second_handler)->m_plans.at(0).m_loaded);
    ASSERT_NO_FATAL_FAILURE(run_ramp(first_handler, 2));
    ASSERT_NO_FATAL_FAILURE(run_ramp(second_handler, 2));
    EXPECT_EQ(shared.m_processed.load(), 4);
    anira_handler_destroy(first_handler);
    EXPECT_EQ(shared.m_unprepared, 1) << "the first handler's handle";
    EXPECT_EQ(shared.m_unloaded, 0) << "the second handler still runs on the loaded model";
    EXPECT_EQ(shared.m_released, 0);
    anira_handler_destroy(second_handler);
    EXPECT_EQ(shared.m_unprepared, 2);
    EXPECT_EQ(shared.m_unloaded, 1);
    EXPECT_EQ(shared.m_released, 1);
}

// The whole variant is in the pool's key of a custom engine, since its load may read any of
// it: one engine in two pipelines over models that differ in an entry the record does not
// carry (a second row, the pass-through's, which the second pipeline adds) loads twice.
TEST(AbiEngine, OneEngineOverTwoVariantsLoadsTwice) {
    const Context context;
    GainEngine gain;
    const anira_engine_desc desc = gain_desc(gain);
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    ASSERT_EQ(anira_custom_engine_create(k_gain_id, &desc, &engine, &err), ANIRA_OK) << err.message;
    Pipe first;
    Pipe second;
    ASSERT_EQ(first.add_engine(engine), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.add_engine(engine), ANIRA_OK) << second.m_err.message;
    ASSERT_EQ(second.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << second.m_err.message;
    anira_custom_engine_destroy(engine);
    const ModelConfig plain = gain_model();
    ModelConfig extended = gain_model();
    extended.add_model_path(k_custom, "custom-processor");
    first.add_inference(plain);
    second.add_inference(extended);
    anira_handler* first_handler = nullptr;
    anira_handler* second_handler = nullptr;
    ASSERT_EQ(first.create_handler(context, &first_handler), ANIRA_OK) << first.m_err.message;
    ASSERT_EQ(second.create_handler(context, &second_handler), ANIRA_OK) << second.m_err.message;
    const anira::ContractHandle contract = explicit_contract();
    ASSERT_EQ(anira_handler_prepare(first_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_prepare(second_handler, contract.native(), &err), ANIRA_OK)
        << err.message;
    EXPECT_EQ(gain.m_loaded, 2);
    EXPECT_EQ(gain.m_prepared, 2);
    EXPECT_NE(anira_test::session_of(first_handler)->m_plans.at(0).m_loaded,
              anira_test::session_of(second_handler)->m_plans.at(0).m_loaded);
    anira_handler_destroy(first_handler);
    anira_handler_destroy(second_handler);
    first.destroy();
    second.destroy();
    EXPECT_EQ(gain.m_unprepared, 2);
    EXPECT_EQ(gain.m_unloaded, 2);
    EXPECT_EQ(gain.m_released, 1);
}

// A registered engine received the names and bound its slots itself: every slot row of its
// plan reads ANIRA_BINDING_ENGINE, the anira.v2.custom row's pass-through's as well (a built-in
// plan's POSITION and NAME rows are AbiHandler.ASlotReportsHowItsPlanBoundIt's).
TEST(AbiEngine, ASlotReportsHowItWasBound) {
    const Context context;
    GainEngine gain;
    Pipe pipe;
    ASSERT_EQ(pipe.add_new_engine(k_gain_id, gain_desc(gain)), ANIRA_OK) << pipe.m_err.message;
    ASSERT_EQ(pipe.add_new_engine(k_custom, anira_test::passthrough_desc()), ANIRA_OK)
        << pipe.m_err.message;
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
        EXPECT_EQ(custom_rows[0].binding, static_cast<uint32_t>(ANIRA_BINDING_ENGINE));
    }
    anira_handler_destroy(handler);
}
