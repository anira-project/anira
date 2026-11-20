// anira/anira_em.h: the Emscripten companion of the C ABI. The EM_JS thunks into the registry of
// the scope that runs the C call (globalThis[Symbol.for('anira.registry')]), the processor and
// engine trampolines that call them, the heap record behind the descriptors' user_data, the
// JavaScript log sink and the context's worker module. Compiled into anira on the Wasm binary
// format only (CMakeLists.txt); the header is empty everywhere else.
#include "em_hooks.h"

#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/anira_em.h>
#include <emscripten/em_js.h>
#include <emscripten/em_macros.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../capi/capi_internal.h"
#include "../../capi/handles.h"
#include "../../scheduler/InferenceThread.h"
#include "../../utils/Backoff.h"
#include "../../utils/Logger.h"

using anira::capi::translate_exception;

// The thunks. Each catches what the hook throws, reports it on the console (not through the anira
// log, which may be what threw) and returns the failure status the caller passes, so that no
// JavaScript exception unwinds through a C frame. A hook that returns nothing is ANIRA_OK. The
// report is rate-limited per instance, one console line per second at most with the count of the
// throws in between: a hook that throws on every quantum must not flood the audio thread with
// console output. Each thunk installs the shared reporter on its Module at its first throw.
EM_JS_DEPS(anira_em_hooks, "$UTF8ToString");

EM_JS(int,
      anira_em_js_processor,
      (uint32_t key, uint32_t phase, uintptr_t a, uintptr_t b, int failure),
      {
          try {
              var registry = globalThis[Symbol.for ('anira.registry')];
              if (!registry) { throw new Error('no anira registry in this scope'); }
              return registry.processor(Module, key >>> 0, phase, a >>> 0, b >>> 0) | 0;
          } catch (e) {
              var report = Module.aniraReportThrow;
              if (!report) {
                  var state = {last: -1e9, suppressed: 0};
                  report = Module.aniraReportThrow = function(what, e) {
                      var now = Date.now();
                      if (now - state.last < 1000) {
                          state.suppressed++;
                          return;
                      }
                      console.error(
                          "anira: " + what + " threw" +
                              (state.suppressed ? " (and " + state.suppressed +
                                                      " more time(s) since the last report)"
                                                : "") +
                              ":",
                          e);
                      state.last = now;
                      state.suppressed = 0;
                  };
              }
              report("the JavaScript processor hook (key " + (key >>> 0) + ", phase " + phase + ")",
                     e);
              return failure;
          }
      });

EM_JS(int,
      anira_em_js_engine,
      (uint32_t key, uint32_t phase, uintptr_t a, uintptr_t b, uintptr_t engine_id, int failure),
      {
          try {
              var registry = globalThis[Symbol.for ('anira.registry')];
              if (!registry) { throw new Error('no anira registry in this scope'); }
              return registry.engine(Module, key >>> 0, phase, a >>> 0, b >>> 0, engine_id >>> 0) |
                     0;
          } catch (e) {
              var report = Module.aniraReportThrow;
              if (!report) {
                  var state = {last: -1e9, suppressed: 0};
                  report = Module.aniraReportThrow = function(what, e) {
                      var now = Date.now();
                      if (now - state.last < 1000) {
                          state.suppressed++;
                          return;
                      }
                      console.error(
                          "anira: " + what + " threw" +
                              (state.suppressed ? " (and " + state.suppressed +
                                                      " more time(s) since the last report)"
                                                : "") +
                              ":",
                          e);
                      state.last = now;
                      state.suppressed = 0;
                  };
              }
              report("the JavaScript engine hook (" +
                         (engine_id ? UTF8ToString(engine_id) : "key " + (key >>> 0)) + ", phase " +
                         phase + ")",
                     e);
              return failure;
          }
      });

EM_JS(void, anira_em_js_log, (uintptr_t record), {
    try {
        var registry = globalThis[Symbol.for ('anira.registry')];
        if (registry && registry.log) { registry.log(Module, record >>> 0); }
    } catch (e) {
        var report = Module.aniraReportThrow;
        if (!report) {
            var state = {last: -1e9, suppressed: 0};
            report = Module.aniraReportThrow = function(what, e) {
                var now = Date.now();
                if (now - state.last < 1000) {
                    state.suppressed++;
                    return;
                }
                console.error("anira: " + what + " threw" +
                                  (state.suppressed ? " (and " + state.suppressed +
                                                          " more time(s) since the last report)"
                                                    : "") +
                                  ":",
                              e);
                state.last = now;
                state.suppressed = 0;
            };
        }
        report("the JavaScript log hook", e);
    }
});

// The release of what the calling scope's JavaScript engines built per loaded model: the
// registry's engine in `phase` (ANIRA_PHASE_UNLOAD) with key 0, the serials of the loaded
// models of now and their count; the registry releases its object of every other model. A
// scope without a registry built nothing. One call per unload a thread sees, never one per
// chunk, so a throw (an object's destroy) is reported as it comes.
EM_JS(void, anira_em_js_unload_models, (uint32_t phase, uintptr_t loaded, uint32_t count), {
    try {
        var registry = globalThis[Symbol.for ('anira.registry')];
        if (registry && registry.engine) {
            registry.engine(Module, 0, phase, loaded >>> 0, count >>> 0, 0);
        }
    } catch (e) { console.error("anira: releasing what the JavaScript engines loaded threw:", e); }
});

namespace {

constexpr uint32_t k_processor_hooks =
    ANIRA_EM_PROCESSOR_HOOK_PRE | ANIRA_EM_PROCESSOR_HOOK_POST | ANIRA_EM_PROCESSOR_HOOK_BEFORE |
    ANIRA_EM_PROCESSOR_HOOK_AFTER | ANIRA_EM_PROCESSOR_HOOK_PREPARE |
    ANIRA_EM_PROCESSOR_HOOK_RELEASE | ANIRA_EM_PROCESSOR_HOOK_RESET |
    ANIRA_EM_PROCESSOR_HOOK_UNPREPARE | ANIRA_EM_PROCESSOR_HOOK_INIT;
constexpr uint32_t k_engine_hooks = ANIRA_EM_ENGINE_HOOK_RESET | ANIRA_EM_ENGINE_HOOK_QUERY;
// The processor hook bits are the phases' own (ANIRA_EM_HOOKS_VERSION 2): one index space with
// anira.hpp's Processor::k_* bits.
static_assert(ANIRA_EM_PROCESSOR_HOOK_PRE == 1U << ANIRA_PHASE_PRE_PROCESS &&
                  ANIRA_EM_PROCESSOR_HOOK_POST == 1U << ANIRA_PHASE_POST_PROCESS &&
                  ANIRA_EM_PROCESSOR_HOOK_BEFORE == 1U << ANIRA_PHASE_BEFORE_INFERENCE &&
                  ANIRA_EM_PROCESSOR_HOOK_AFTER == 1U << ANIRA_PHASE_AFTER_INFERENCE &&
                  ANIRA_EM_PROCESSOR_HOOK_PREPARE == 1U << ANIRA_PHASE_PREPARE &&
                  ANIRA_EM_PROCESSOR_HOOK_RELEASE == 1U << ANIRA_PHASE_RELEASE &&
                  ANIRA_EM_PROCESSOR_HOOK_RESET == 1U << ANIRA_PHASE_RESET &&
                  ANIRA_EM_PROCESSOR_HOOK_UNPREPARE == 1U << ANIRA_PHASE_UNPREPARE &&
                  ANIRA_EM_PROCESSOR_HOOK_INIT == 1U << ANIRA_PHASE_INIT,
              "a processor hook bit is 1 << its anira_phase");
// What the hook of a processor returns when it throws: the phase fails, its chunk delivers zeros.
constexpr anira_status k_processor_thrown = ANIRA_ERROR_UNKNOWN;
constexpr anira_status k_engine_thrown = ANIRA_ERROR_ENGINE;

// An engine's edge statement for the slots of one direction (anira_em_pipeline_add_js_engine's
// input_edge or output_edge), which the descriptor's edge answers; none: the provider rule.
struct EdgeStatement {
    bool m_stated = false;
    uint32_t m_class = ANIRA_EDGE_ZERO_COPY;
    std::string m_reason;  ///< empty: nothing to say
};

// The user_data of a JavaScript processor or engine: freed by the descriptor's release.
struct HookRecord {
    uint32_t m_js_key = 0;
    uint32_t m_hook_mask = 0;
    std::string m_engine_id;  ///< empty for a processor
    /// A processor's id (anira_processor_desc.id), what its inference hooks are for the model's
    /// figure: the key of the name it is defined under, so that the handlers of every page that
    /// defines the same processor share their figure.
    std::string m_processor_id;
    EdgeStatement m_input_edge;
    EdgeStatement m_output_edge;
};

HookRecord& record_of(void* user_data) noexcept {
    return *static_cast<HookRecord*>(user_data);
}

uintptr_t address(const void* pointer) noexcept {
    return reinterpret_cast<uintptr_t>(pointer);
}

anira_status call_processor(void* user_data, uint32_t phase, uintptr_t a, uintptr_t b) noexcept {
    return static_cast<anira_status>(
        anira_em_js_processor(record_of(user_data).m_js_key, phase, a, b, k_processor_thrown));
}

anira_status call_engine(void* user_data, uint32_t phase, uintptr_t a, uintptr_t b) noexcept {
    const HookRecord& record = record_of(user_data);
    return static_cast<anira_status>(
        anira_em_js_engine(record.m_js_key,
                           phase,
                           a,
                           b,
                           record.m_engine_id.empty() ? 0 : address(record.m_engine_id.c_str()),
                           k_engine_thrown));
}

// ---- the processor trampolines: one per slot shape; the four phases share one, by ctx->phase ----

anira_status ANIRA_CALL processor_phase(const anira_processor_ctx* ctx,
                                        void* prepared,
                                        void* user_data) noexcept {
    return call_processor(user_data, ctx->phase, address(ctx), address(prepared));
}

void ANIRA_CALL processor_reset(const anira_processor_ctx* ctx,
                                void* prepared,
                                void* user_data) noexcept {
    static_cast<void>(
        call_processor(user_data, ANIRA_PHASE_RESET, address(ctx), address(prepared)));
}

anira_status ANIRA_CALL processor_prepare(const anira_prepare_info* info,
                                          void* user_data,
                                          void** out_prepared) noexcept {
    return call_processor(user_data, ANIRA_PHASE_PREPARE, address(info), address(out_prepared));
}

void ANIRA_CALL processor_unprepare(void* prepared, void* user_data) noexcept {
    static_cast<void>(call_processor(user_data, ANIRA_PHASE_UNPREPARE, address(prepared), 0));
}

anira_status ANIRA_CALL processor_init(const anira_init_info* info, void* user_data) noexcept {
    return call_processor(user_data, ANIRA_PHASE_INIT, address(info), 0);
}

void ANIRA_CALL processor_release(void* user_data) noexcept {
    const std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(user_data));
    if ((record->m_hook_mask & ANIRA_EM_PROCESSOR_HOOK_RELEASE) != 0) {
        static_cast<void>(call_processor(user_data, ANIRA_PHASE_RELEASE, 0, 0));
    }
}

// ---- the engine trampolines ----

// One loaded model of a JavaScript engine, the C side of it: the record the descriptor's load
// builds on the control thread from the row of anira_engine_load_info (its path, copied, and
// its bytes, borrowed: the loaded model holds them while it lives, and this record goes with
// its unload), and hands to every process and reset call as their `prepared` pointer (the
// descriptor's prepare returns it). The engine's own object of the model (what its load
// returns, the web twin of anira::Engine::Loaded) cannot be built here: the engine lives in
// the realms of the threads that run its calls. Each of them builds its own from this record,
// at its first call for the model, and keeps it under the record's serial
// (web/src/registry.ts, which reads the layout: five 32-bit words on wasm32).
struct EngineModel {
    uint32_t m_row = 0;
    const void* m_bytes = nullptr;  ///< the entry's bytes, or NULL
    uint32_t m_size = 0;            ///< their size, 0 without
    const char* m_path = nullptr;   ///< into m_path_storage, or NULL for an entry without one
    /// Which loaded model this is: a number no other loaded model of the module's life has. A
    /// scope's registry keeps its engines' objects by it. The record's address does not tell:
    /// the record of a model loaded later may be allocated where an unloaded one's was.
    uint32_t m_serial = 0;
    std::string m_path_storage;
};
static_assert(sizeof(void*) == 4 && offsetof(EngineModel, m_row) == 0 &&
                  offsetof(EngineModel, m_bytes) == 4 && offsetof(EngineModel, m_size) == 8 &&
                  offsetof(EngineModel, m_path) == 12 && offsetof(EngineModel, m_serial) == 16,
              "registry.ts reads these five words");

// The serial of the next loaded model: counted up by every load, on whichever thread it runs.
std::atomic<uint32_t> s_next_model_serial{1};

// The loaded models of now, by serial: what a scope's registry measures its objects against
// when a model was unloaded (anira::em::release_unloaded_js_models). The descriptor's load adds
// a serial and its unload takes it out, on the control threads; a thread that runs engine calls
// copies the list. A spin lock of atomics alone, as the scheduler's wait lists have (a Wasm
// Worker and the main thread wait on it alike), held for a few words and never across a call
// into JavaScript.
std::atomic<bool> s_loaded_locked{false};
std::vector<uint32_t> s_loaded_serials;

struct LoadedLock {
    LoadedLock() noexcept {
        anira::detail::Backoff backoff;
        while (s_loaded_locked.exchange(true, std::memory_order::acquire)) { backoff.pause(); }
    }
    ~LoadedLock() { s_loaded_locked.store(false, std::memory_order::release); }
    LoadedLock(const LoadedLock&) = delete;
    LoadedLock& operator=(const LoadedLock&) = delete;
    LoadedLock(LoadedLock&&) = delete;
    LoadedLock& operator=(LoadedLock&&) = delete;
};

// Counted up by every unload, behind the model's removal from the list: a thread whose last
// look saw another value has objects to release, perhaps. Each thread remembers what it saw.
std::atomic<uint32_t> s_unload_epoch{0};
thread_local uint32_t t_unload_epoch_seen = 0;

anira_status ANIRA_CALL engine_load(const anira_engine_load_info* info,
                                    void* /*user_data*/,
                                    void** out_loaded) noexcept {
    try {
        auto model = std::make_unique<EngineModel>();
        model->m_row = info->row;
        model->m_serial = s_next_model_serial.fetch_add(1, std::memory_order_relaxed);
        if (const char* path = anira_model_config_model_path(info->model, info->row)) {
            model->m_path_storage = path;
            model->m_path = model->m_path_storage.c_str();
        }
        const void* bytes = nullptr;
        size_t size = 0;
        if (anira_model_config_model_bytes(info->model, info->row, &bytes, &size) == ANIRA_OK) {
            model->m_bytes = bytes;
            model->m_size = static_cast<uint32_t>(size);
        }
        {
            const LoadedLock lock;
            s_loaded_serials.push_back(model->m_serial);
        }
        *out_loaded = model.release();
        return ANIRA_OK;
    } catch (...) { return ANIRA_ERROR_OUT_OF_MEMORY; }
}

// The C unload: the control thread of the handler that held the model last, behind the last
// inference of the model on any thread (a session's release drains its jobs, a removal waits
// for the chunks of its plans). It cannot call the engine: the engine's objects of this model
// are in the realms of the threads that ran it. The model leaves the list of the loaded ones,
// and the wake word brings every parked inference thread to its next look at the queue, where
// it releases its own (release_unloaded_js_models); a thread inside an inference of another
// model gets there when that one returns. Nothing here waits for them.
void ANIRA_CALL engine_unload(void* loaded, void* /*user_data*/) noexcept {
    const std::unique_ptr<EngineModel> model(static_cast<EngineModel*>(loaded));
    {
        const LoadedLock lock;
        std::erase(s_loaded_serials, model->m_serial);
    }
    s_unload_epoch.fetch_add(1, std::memory_order::acq_rel);
    anira::scheduler::InferenceThread::notify_work();
}

anira_status ANIRA_CALL engine_prepare(const anira_prepare_info* /*info*/,
                                       void* loaded,
                                       void* /*user_data*/,
                                       void** out_prepared) noexcept {
    *out_prepared = loaded;
    return ANIRA_OK;
}

void ANIRA_CALL engine_unprepare(void* /*prepared*/, void* /*user_data*/) noexcept {}

anira_status ANIRA_CALL engine_process(const anira_engine_ctx* ctx,
                                       void* prepared,
                                       void* user_data) noexcept {
    return call_engine(user_data, ANIRA_PHASE_INFERENCE, address(ctx), address(prepared));
}

void ANIRA_CALL engine_reset(const anira_engine_ctx* ctx,
                             void* prepared,
                             void* user_data) noexcept {
    static_cast<void>(call_engine(user_data, ANIRA_PHASE_RESET, address(ctx), address(prepared)));
}

anira_status ANIRA_CALL engine_query(const anira_init_info* info,
                                     void* user_data,
                                     uint64_t* out_available) noexcept {
    return call_engine(user_data, ANIRA_PHASE_QUERY, address(info), address(out_available));
}

// The statement the main thread gave for the slot's direction (the engine itself runs in the
// Workers); without one, `out` keeps the provider rule of that direction it arrived with.
anira_status ANIRA_CALL engine_edge(void* /*loaded*/,
                                    uint32_t is_input,
                                    uint32_t /*slot*/,
                                    anira_engine_edge* out,
                                    void* user_data) noexcept {
    const HookRecord& record = record_of(user_data);
    const EdgeStatement& stated = is_input != 0 ? record.m_input_edge : record.m_output_edge;
    if (!stated.m_stated) { return ANIRA_OK; }
    out->edge_class = stated.m_class;
    out->reason = stated.m_reason.empty() ? nullptr : stated.m_reason.c_str();
    return ANIRA_OK;
}

// Whether `edge` is a statement the descriptor's edge can answer: NULL (none), or a record whose
// struct_size reaches its class and whose class anira_edge_class names.
bool valid_edge(const anira_engine_edge* edge) noexcept {
    if (edge == nullptr) { return true; }
    return edge->struct_size >= offsetof(anira_engine_edge, edge_class) + sizeof(uint32_t) &&
           (edge->edge_class <= static_cast<uint32_t>(ANIRA_EDGE_DEDICATED_TRANSFER) ||
            edge->edge_class == static_cast<uint32_t>(ANIRA_EDGE_UNKNOWN));
}

// No JavaScript release: a JavaScript engine's lifetime is its Worker's.
void ANIRA_CALL engine_release(void* user_data) noexcept {
    delete static_cast<HookRecord*>(user_data);  // NOLINT(cppcoreguidelines-owning-memory)
}

// ---- the log sink ----

void ANIRA_CALL log_hook(const anira_log_record* record, void* /*user_data*/) {
    anira_em_js_log(address(record));
}

// One registration per module: the instances share one memory and so the sink list.
anira::detail::LogSinkId s_log_hook_sink = 0;

// Fills a processor descriptor of trampolines over a new record; throws std::bad_alloc only.
void fill_processor_desc(anira_processor_desc& desc,
                         uint32_t js_key,
                         uint32_t hook_mask,
                         uint32_t realtime_phases) {
    auto record = std::make_unique<HookRecord>();
    record->m_js_key = js_key;
    record->m_hook_mask = hook_mask;
    desc = ANIRA_PROCESSOR_DESC_INIT;
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_PRE) != 0) { desc.pre_process = &processor_phase; }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_POST) != 0) { desc.post_process = &processor_phase; }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_BEFORE) != 0) {
        desc.before_inference = &processor_phase;
    }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_AFTER) != 0) {
        desc.after_inference = &processor_phase;
    }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_RESET) != 0) { desc.reset = &processor_reset; }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_PREPARE) != 0) { desc.prepare = &processor_prepare; }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_UNPREPARE) != 0) {
        desc.unprepare = &processor_unprepare;
    }
    if ((hook_mask & ANIRA_EM_PROCESSOR_HOOK_INIT) != 0) { desc.init = &processor_init; }
    desc.release = &processor_release;
    // The caller's stated promise, as given: anira_pipeline_add_processor judges the bits.
    desc.realtime_phases = realtime_phases;
    record->m_processor_id = "org.anira.web.processor." + std::to_string(js_key);
    desc.id = record->m_processor_id.c_str();  // copied by the carrier
    desc.user_data = record.release();
}

// The statement of one direction, copied out of the caller's record (valid_edge); none for NULL.
EdgeStatement statement_of(const anira_engine_edge* edge) {
    EdgeStatement stated;
    if (edge == nullptr) { return stated; }
    stated.m_stated = true;
    stated.m_class = edge->edge_class;
    if (edge->struct_size >= offsetof(anira_engine_edge, reason) + sizeof(const char*) &&
        edge->reason != nullptr) {
        stated.m_reason = edge->reason;
    }
    return stated;
}

// Fills an engine descriptor of trampolines over a new record, the two directions' edge
// statements (valid_edge) copied into it; throws std::bad_alloc only.
void fill_engine_desc(anira_engine_desc& desc,
                      uint32_t js_key,
                      uint32_t hook_mask,
                      uint32_t engine_flags,
                      const anira_engine_edge* input_edge,
                      const anira_engine_edge* output_edge) {
    auto record = std::make_unique<HookRecord>();
    record->m_js_key = js_key;
    record->m_hook_mask = hook_mask;
    record->m_input_edge = statement_of(input_edge);
    record->m_output_edge = statement_of(output_edge);
    desc = ANIRA_ENGINE_DESC_INIT;
    desc.process = &engine_process;
    // The model record of every call, and the list of the loaded models (no crossing: the
    // engine runs in the scopes of the inference threads, where each builds its object of a
    // model from the record and releases it once the model left the list).
    desc.load = &engine_load;
    desc.unload = &engine_unload;
    desc.prepare = &engine_prepare;
    desc.unprepare = &engine_unprepare;
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_RESET) != 0) { desc.reset = &engine_reset; }
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_QUERY) != 0) { desc.query = &engine_query; }
    if (input_edge != nullptr || output_edge != nullptr) { desc.edge = &engine_edge; }
    desc.release = &engine_release;
    // As given: anira_custom_engine_create judges the bits.
    desc.flags = engine_flags;
    desc.user_data = record.release();
}

}  // namespace

anira_status ANIRA_CALL anira_em_processor_desc_init(anira_processor_desc* desc,
                                                     uint32_t js_key,
                                                     uint32_t hook_mask,
                                                     uint32_t realtime_phases) ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_processor_hooks) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_processor_desc(*desc, js_key, hook_mask, realtime_phases);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_em_pipeline_add_js_processor(anira_pipeline* pipeline,
                                                           uint32_t js_key,
                                                           uint32_t hook_mask,
                                                           uint32_t realtime_phases,
                                                           anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(
        (hook_mask & ~k_processor_hooks) == 0,
        err,
        ANIRA_ERROR_INVALID_ARGUMENT,
        "JavaScript processor: hook_mask 0x%x carries a bit no ANIRA_EM_PROCESSOR_HOOK_* "
        "defines (they are 0x%x)",
        hook_mask,
        k_processor_hooks);
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    fill_processor_desc(desc, js_key, hook_mask, realtime_phases);
    // A refused addition never calls release: until it succeeds the record is ours.
    std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(desc.user_data));
    const anira_status added = anira_pipeline_add_processor(pipeline, &desc, err);
    if (added == ANIRA_OK) { static_cast<void>(record.release()); }
    return added;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_em_engine_desc_init(anira_engine_desc* desc,
                                                  uint32_t js_key,
                                                  uint32_t hook_mask,
                                                  uint32_t engine_flags,
                                                  const anira_engine_edge* input_edge,
                                                  const anira_engine_edge* output_edge)
    ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_engine_hooks) != 0 || !valid_edge(input_edge) ||
        !valid_edge(output_edge)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_engine_desc(*desc, js_key, hook_mask, engine_flags, input_edge, output_edge);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_em_pipeline_add_js_engine(anira_pipeline* pipeline,
                                                        const char* engine_id,
                                                        uint32_t js_key,
                                                        uint32_t hook_mask,
                                                        uint32_t engine_flags,
                                                        const anira_engine_edge* input_edge,
                                                        const anira_engine_edge* output_edge,
                                                        anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE((hook_mask & ~k_engine_hooks) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "JavaScript engine: hook_mask 0x%x carries a bit no "
                       "ANIRA_EM_ENGINE_HOOK_* defines (they are 0x%x)",
                       hook_mask,
                       k_engine_hooks);
    ANIRA_CAPI_REQUIRE(valid_edge(input_edge) && valid_edge(output_edge),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "JavaScript engine: the %s edge statement is too short to hold its class, "
                       "or names a class anira_edge_class does not (ANIRA_EDGE_ZERO_COPY to "
                       "ANIRA_EDGE_DEDICATED_TRANSFER, or ANIRA_EDGE_UNKNOWN)",
                       valid_edge(input_edge) ? "output" : "input");
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    fill_engine_desc(desc, js_key, hook_mask, engine_flags, input_edge, output_edge);
    // A refused create never calls release: until it succeeds the record is ours.
    std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(desc.user_data));
    if (engine_id != nullptr) { record->m_engine_id = engine_id; }
    anira_custom_engine* engine = nullptr;
    const anira_status created = anira_custom_engine_create(engine_id, &desc, &engine, err);
    if (created != ANIRA_OK) { return created; }
    static_cast<void>(record.release());
    // The pipeline takes its own reference; dropping the handle's releases a refused engine.
    const anira_status added = anira_pipeline_add_engine(pipeline, engine, err);
    anira_custom_engine_destroy(engine);
    return added;
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_em_set_log_hook(anira_bool enable) ANIRA_NOEXCEPT try {
    // The hook takes the platform sink's place (the console by level, em_console.cpp): both would
    // print every record to the console.
    if (enable != 0) {
        if (s_log_hook_sink == 0) {
            s_log_hook_sink = anira::detail::add_log_sink(&log_hook, nullptr, ANIRA_LOG_DEBUG);
            anira::capi::hold_platform_sink_off(true);
        }
    } else if (s_log_hook_sink != 0) {
        anira::detail::remove_log_sink(s_log_hook_sink);
        s_log_hook_sink = 0;
        anira::capi::hold_platform_sink_off(false);
    }
} catch (...) { anira::capi::report_void_failure(__func__); }

void anira::em::release_unloaded_js_models() noexcept {
    const uint32_t epoch = s_unload_epoch.load(std::memory_order::acquire);
    if (epoch == t_unload_epoch_seen) { return; }
    // The loaded models of now, copied out of the lock: read behind the epoch, so every unload
    // the epoch counts has left the list. A model loaded since is in it, and this thread has
    // an object only of a model that was loaded before one of its jobs.
    std::vector<uint32_t> loaded;
    try {
        while (true) {
            size_t needed = 0;
            {
                const LoadedLock lock;
                needed = s_loaded_serials.size();
                if (needed <= loaded.capacity()) {
                    loaded.assign(s_loaded_serials.begin(), s_loaded_serials.end());
                    break;
                }
            }
            loaded.reserve(needed);
        }
    } catch (...) {
        return;  // no memory for the copy: the epoch stays unseen, and the next look tries again
    }
    t_unload_epoch_seen = epoch;
    anira_em_js_unload_models(ANIRA_PHASE_UNLOAD,
                              address(loaded.data()),
                              static_cast<uint32_t>(loaded.size()));
}

void anira::em::release_all_js_models() noexcept {
    // No model is loaded as far as this scope goes: the registry releases every object.
    anira_em_js_unload_models(ANIRA_PHASE_UNLOAD, 0, 0);
}

uint32_t ANIRA_CALL anira_em_inference_threads_in_loop(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return anira::scheduler::InferenceThread::get_num_loop_active();
}

anira_status ANIRA_CALL anira_em_context_config_set_worker_module(anira_context_config* config,
                                                                  const char* url,
                                                                  anira_error* err) ANIRA_NOEXCEPT
    try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "context config: NULL handle");
    config->m_em_worker_module = url != nullptr ? url : "";
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }
