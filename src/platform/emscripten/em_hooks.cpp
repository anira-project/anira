// anira/anira_em.h: the Emscripten companion of the C ABI. The EM_JS thunks into the registry of
// the scope that runs the C call (globalThis[Symbol.for('anira.registry')]), the stage and engine
// trampolines that call them, the heap record behind the descriptors' user_data, the JavaScript
// log sink and the context's worker module. Compiled into anira on the Wasm binary format only
// (CMakeLists.txt); the header is empty everywhere else.
#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/anira_em.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/Logger.h>
#include <emscripten/em_js.h>
#include <emscripten/em_macros.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "../../capi/capi_internal.h"
#include "../../capi/handles.h"

using anira::capi::translate_exception;

// The thunks. Each catches what the hook throws, reports it on the console (not through the anira
// log, which may be what threw) and returns the failure status the caller passes, so that no
// JavaScript exception unwinds through a C frame. A hook that returns nothing is ANIRA_OK. The
// report is rate-limited per instance, one console line per second at most with the count of the
// throws in between: a hook that throws on every quantum must not flood the audio thread with
// console output. Each thunk installs the shared reporter on its Module at its first throw.
EM_JS_DEPS(anira_em_hooks, "$UTF8ToString");

EM_JS(int,
      anira_em_js_stage,
      (uint32_t key, uint32_t phase, uintptr_t a, uintptr_t b, int failure),
      {
          try {
              var registry = globalThis[Symbol.for ('anira.registry')];
              if (!registry) { throw new Error('no anira registry in this scope'); }
              return registry.stage(Module, key >>> 0, phase, a >>> 0, b >>> 0) | 0;
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
              report("the JavaScript stage hook (key " + (key >>> 0) + ", phase " + phase + ")", e);
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

namespace {

constexpr uint32_t k_stage_hooks = ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_POST | ANIRA_EM_HOOK_BEFORE |
                                   ANIRA_EM_HOOK_AFTER | ANIRA_EM_HOOK_PREPARE |
                                   ANIRA_EM_HOOK_RELEASE | ANIRA_EM_HOOK_RESET |
                                   ANIRA_EM_HOOK_UNPREPARE | ANIRA_EM_HOOK_INIT;
constexpr uint32_t k_engine_hooks = ANIRA_EM_ENGINE_HOOK_RESET | ANIRA_EM_ENGINE_HOOK_QUERY;
// The stage hook bits are the phases' own (ANIRA_EM_HOOKS_VERSION 2): one index space with
// anira.hpp's Stage::k_* bits.
static_assert(ANIRA_EM_HOOK_PRE == 1U << ANIRA_PHASE_PRE_PROCESS &&
                  ANIRA_EM_HOOK_POST == 1U << ANIRA_PHASE_POST_PROCESS &&
                  ANIRA_EM_HOOK_BEFORE == 1U << ANIRA_PHASE_BEFORE_INFERENCE &&
                  ANIRA_EM_HOOK_AFTER == 1U << ANIRA_PHASE_AFTER_INFERENCE &&
                  ANIRA_EM_HOOK_PREPARE == 1U << ANIRA_PHASE_PREPARE &&
                  ANIRA_EM_HOOK_RELEASE == 1U << ANIRA_PHASE_RELEASE &&
                  ANIRA_EM_HOOK_RESET == 1U << ANIRA_PHASE_RESET &&
                  ANIRA_EM_HOOK_UNPREPARE == 1U << ANIRA_PHASE_UNPREPARE &&
                  ANIRA_EM_HOOK_INIT == 1U << ANIRA_PHASE_INIT,
              "a stage hook bit is 1 << its anira_phase");
// What the hook of a stage returns when it throws: the phase fails, its chunk delivers zeros.
constexpr anira_status k_stage_thrown = ANIRA_ERROR_UNKNOWN;
constexpr anira_status k_engine_thrown = ANIRA_ERROR_ENGINE;

// The user_data of a JavaScript stage or engine: freed by the descriptor's release.
struct HookRecord {
    uint32_t m_js_key = 0;
    uint32_t m_hook_mask = 0;
    std::string m_engine_id;  ///< empty for a stage
};

HookRecord& record_of(void* user_data) noexcept {
    return *static_cast<HookRecord*>(user_data);
}

uintptr_t address(const void* pointer) noexcept {
    return reinterpret_cast<uintptr_t>(pointer);
}

anira_status call_stage(void* user_data, uint32_t phase, uintptr_t a, uintptr_t b) noexcept {
    return static_cast<anira_status>(
        anira_em_js_stage(record_of(user_data).m_js_key, phase, a, b, k_stage_thrown));
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

// ---- the stage trampolines: one per slot shape; the four phases share one, by ctx->phase ----

anira_status ANIRA_CALL stage_phase(const anira_stage_ctx* ctx,
                                    void* prepared,
                                    void* user_data) noexcept {
    return call_stage(user_data, ctx->phase, address(ctx), address(prepared));
}

void ANIRA_CALL stage_reset(const anira_stage_ctx* ctx, void* prepared, void* user_data) noexcept {
    static_cast<void>(call_stage(user_data, ANIRA_PHASE_RESET, address(ctx), address(prepared)));
}

anira_status ANIRA_CALL stage_prepare(const anira_prepare_info* info,
                                      void* user_data,
                                      void** out_prepared) noexcept {
    return call_stage(user_data, ANIRA_PHASE_PREPARE, address(info), address(out_prepared));
}

void ANIRA_CALL stage_unprepare(void* prepared, void* user_data) noexcept {
    static_cast<void>(call_stage(user_data, ANIRA_PHASE_UNPREPARE, address(prepared), 0));
}

anira_status ANIRA_CALL stage_init(const anira_init_info* info, void* user_data) noexcept {
    return call_stage(user_data, ANIRA_PHASE_INIT, address(info), 0);
}

void ANIRA_CALL stage_release(void* user_data) noexcept {
    const std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(user_data));
    if ((record->m_hook_mask & ANIRA_EM_HOOK_RELEASE) != 0) {
        static_cast<void>(call_stage(user_data, ANIRA_PHASE_RELEASE, 0, 0));
    }
}

// ---- the engine trampolines ----

// What a JavaScript engine learns of its model entry, the C engine's load record read once:
// built by the descriptor's load on the control thread from the row of anira_engine_load_info
// (its path, copied, and its bytes, borrowed: the loaded model holds them while it lives, and
// this record goes with its unload), and handed to every process and reset call as their
// `prepared` pointer (the descriptor's prepare returns it). The layout is the package's
// (web/src/registry.ts engineModel reads it): four 32-bit words on wasm32.
struct EngineModel {
    uint32_t m_row = 0;
    const void* m_bytes = nullptr;  ///< the entry's bytes, or NULL
    uint32_t m_size = 0;            ///< their size, 0 without
    const char* m_path = nullptr;   ///< into m_path_storage, or NULL for an entry without one
    std::string m_path_storage;
};
static_assert(sizeof(void*) == 4 && offsetof(EngineModel, m_row) == 0 &&
                  offsetof(EngineModel, m_bytes) == 4 && offsetof(EngineModel, m_size) == 8 &&
                  offsetof(EngineModel, m_path) == 12,
              "registry.ts engineModel reads these four words");

anira_status ANIRA_CALL engine_load(const anira_engine_load_info* info,
                                    void* /*user_data*/,
                                    void** out_loaded) noexcept {
    try {
        auto model = std::make_unique<EngineModel>();
        model->m_row = info->row;
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
        *out_loaded = model.release();
        return ANIRA_OK;
    } catch (...) { return ANIRA_ERROR_OUT_OF_MEMORY; }
}

void ANIRA_CALL engine_unload(void* loaded, void* /*user_data*/) noexcept {
    delete static_cast<EngineModel*>(loaded);  // NOLINT(cppcoreguidelines-owning-memory)
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

// Fills a stage descriptor of trampolines over a new record; throws std::bad_alloc only.
void fill_stage_desc(anira_stage_desc& desc,
                     uint32_t js_key,
                     uint32_t hook_mask,
                     uint32_t stage_flags) {
    auto record = std::make_unique<HookRecord>();
    record->m_js_key = js_key;
    record->m_hook_mask = hook_mask;
    desc = ANIRA_STAGE_DESC_INIT;
    if ((hook_mask & ANIRA_EM_HOOK_PRE) != 0) { desc.pre_process = &stage_phase; }
    if ((hook_mask & ANIRA_EM_HOOK_POST) != 0) { desc.post_process = &stage_phase; }
    if ((hook_mask & ANIRA_EM_HOOK_BEFORE) != 0) { desc.before_inference = &stage_phase; }
    if ((hook_mask & ANIRA_EM_HOOK_AFTER) != 0) { desc.after_inference = &stage_phase; }
    if ((hook_mask & ANIRA_EM_HOOK_RESET) != 0) { desc.reset = &stage_reset; }
    if ((hook_mask & ANIRA_EM_HOOK_PREPARE) != 0) { desc.prepare = &stage_prepare; }
    if ((hook_mask & ANIRA_EM_HOOK_UNPREPARE) != 0) { desc.unprepare = &stage_unprepare; }
    if ((hook_mask & ANIRA_EM_HOOK_INIT) != 0) { desc.init = &stage_init; }
    desc.release = &stage_release;
    // The caller's stated promise, as given: anira_pipeline_add_stage judges the bits.
    desc.flags = stage_flags;
    desc.user_data = record.release();
}

// Fills an engine descriptor of trampolines over a new record; throws std::bad_alloc only.
void fill_engine_desc(anira_engine_desc& desc,
                      uint32_t js_key,
                      uint32_t hook_mask,
                      uint32_t engine_flags) {
    auto record = std::make_unique<HookRecord>();
    record->m_js_key = js_key;
    record->m_hook_mask = hook_mask;
    desc = ANIRA_ENGINE_DESC_INIT;
    desc.process = &engine_process;
    // The model record of every call (no crossing: the engine runs in the Workers' scopes,
    // and reads the record there, registry.ts engineModel).
    desc.load = &engine_load;
    desc.unload = &engine_unload;
    desc.prepare = &engine_prepare;
    desc.unprepare = &engine_unprepare;
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_RESET) != 0) { desc.reset = &engine_reset; }
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_QUERY) != 0) { desc.query = &engine_query; }
    desc.release = &engine_release;
    // As given: anira_custom_engine_create judges the bits.
    desc.flags = engine_flags;
    desc.user_data = record.release();
}

}  // namespace

anira_status ANIRA_CALL anira_em_stage_desc_init(anira_stage_desc* desc,
                                                 uint32_t js_key,
                                                 uint32_t hook_mask,
                                                 uint32_t stage_flags) ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_stage_hooks) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_stage_desc(*desc, js_key, hook_mask, stage_flags);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_em_pipeline_add_js_stage(anira_pipeline* pipeline,
                                                       uint32_t js_key,
                                                       uint32_t hook_mask,
                                                       uint32_t stage_flags,
                                                       anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE((hook_mask & ~k_stage_hooks) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "JavaScript stage: hook_mask 0x%x carries a bit no ANIRA_EM_HOOK_* "
                       "defines (they are 0x%x)",
                       hook_mask,
                       k_stage_hooks);
    anira_stage_desc desc = ANIRA_STAGE_DESC_INIT;
    fill_stage_desc(desc, js_key, hook_mask, stage_flags);
    // A refused addition never calls release: until it succeeds the record is ours.
    std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(desc.user_data));
    const anira_status added = anira_pipeline_add_stage(pipeline, &desc, err);
    if (added == ANIRA_OK) { static_cast<void>(record.release()); }
    return added;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_em_engine_desc_init(anira_engine_desc* desc,
                                                  uint32_t js_key,
                                                  uint32_t hook_mask,
                                                  uint32_t engine_flags) ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_engine_hooks) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_engine_desc(*desc, js_key, hook_mask, engine_flags);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_em_pipeline_register_js_engine(anira_pipeline* pipeline,
                                                             const char* engine_id,
                                                             uint32_t js_key,
                                                             uint32_t hook_mask,
                                                             uint32_t engine_flags,
                                                             anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE((hook_mask & ~k_engine_hooks) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "JavaScript engine: hook_mask 0x%x carries a bit no "
                       "ANIRA_EM_ENGINE_HOOK_* defines (they are 0x%x)",
                       hook_mask,
                       k_engine_hooks);
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    fill_engine_desc(desc, js_key, hook_mask, engine_flags);
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
