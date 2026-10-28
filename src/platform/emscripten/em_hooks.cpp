// anira/anira_em.h: the Emscripten companion of the C ABI. The EM_JS thunks into the hook
// object Module.anira of the instance that runs the C call, the stage and engine trampolines
// that call them, the heap record behind the descriptors' user_data, the JavaScript log sink
// and the per-instance stack and thread-local block. Compiled into anira on the Wasm binary
// format only (CMakeLists.txt); the header is empty everywhere else.
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/anira_em.h>
#include <anira/utils/Logger.h>
#include <emscripten/em_js.h>
#include <emscripten/em_macros.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

#include "../../capi/capi_internal.h"

using anira::capi::translate_exception;

// wasm-ld synthesizes it for a shared-memory link: copies the TLS segment's initial image into
// the block and makes it the calling instance's __tls_base.
extern "C" void __wasm_init_tls(void* block);  // NOLINT(bugprone-reserved-identifier)

// The thunks. Each catches what the hook throws, reports it once on the console (not through the
// anira log, which may be what threw) and returns the failure status the caller passes, so that
// no JavaScript exception unwinds through a C frame. A hook that returns nothing is ANIRA_OK.
EM_JS_DEPS(anira_em_hooks, "$UTF8ToString");

EM_JS(int,
      anira_em_js_stage,
      (uint32_t key, uint32_t phase, uintptr_t a, uintptr_t b, int failure),
      {
          try {
              return Module.anira.stage(key >>> 0, phase, a >>> 0, b >>> 0) | 0;
          } catch (e) {
              console.error("anira: the JavaScript stage hook (key " + (key >>> 0) + ", phase " +
                                phase + ") threw:",
                            e);
              return failure;
          }
      });

EM_JS(int,
      anira_em_js_engine,
      (uint32_t key, uint32_t phase, uintptr_t a, uintptr_t b, uintptr_t engine_id, int failure),
      {
          try {
              return Module.anira.engine(key >>> 0, phase, a >>> 0, b >>> 0) | 0;
          } catch (e) {
              console.error("anira: the JavaScript engine hook (" +
                                (engine_id ? UTF8ToString(engine_id) : "key " + (key >>> 0)) +
                                ", phase " + phase + ") threw:",
                            e);
              return failure;
          }
      });

EM_JS(void, anira_em_js_log, (uintptr_t record), {
    try {
        Module.anira.log(record >>> 0);
    } catch (e) { console.error("anira: the JavaScript log hook threw:", e); }
});

namespace {

constexpr uint32_t k_stage_hooks = ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_POST | ANIRA_EM_HOOK_BEFORE |
                                   ANIRA_EM_HOOK_AFTER | ANIRA_EM_HOOK_PREPARE |
                                   ANIRA_EM_HOOK_RELEASE | ANIRA_EM_HOOK_RESET |
                                   ANIRA_EM_HOOK_UNPREPARE | ANIRA_EM_HOOK_INIT;
constexpr uint32_t k_engine_hooks = ANIRA_EM_ENGINE_HOOK_RESET | ANIRA_EM_ENGINE_HOOK_QUERY;
// What the hook of a stage returns when it throws: the phase fails, its chunk delivers zeros.
constexpr anira_status k_stage_thrown = ANIRA_ERROR_UNKNOWN;
constexpr anira_status k_engine_thrown = ANIRA_ERROR_ENGINE;
constexpr size_t k_stack_align = 16;
constexpr size_t k_default_stack_bytes = size_t{4} * 1024 * 1024;

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

size_t stack_bytes(size_t bytes) noexcept {
    const size_t requested = bytes == 0 ? k_default_stack_bytes : bytes;
    return (requested + k_stack_align - 1) / k_stack_align * k_stack_align;
}

size_t tls_align() noexcept {
    const size_t align = __builtin_wasm_tls_align();
    return align > k_stack_align ? align : k_stack_align;
}

// The thread-local block sits above the stack's top, where a growing stack never reaches.
void* tls_block(void* stack_base, size_t bytes) noexcept {
    const size_t align = tls_align();
    const uintptr_t top = address(stack_base) + stack_bytes(bytes);
    const uintptr_t block = (top + align - 1) / align * align;
    return reinterpret_cast<void*>(block);  // NOLINT(performance-no-int-to-ptr)
}

// Fills a stage descriptor of trampolines over a new record; throws std::bad_alloc only.
void fill_stage_desc(anira_stage_desc& desc, uint32_t js_key, uint32_t hook_mask) {
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
    // The bit is the hook's real-time promise for its phase: JavaScript cannot write the flags.
    if ((hook_mask & (ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_POST | ANIRA_EM_HOOK_RESET)) != 0) {
        desc.flags |= ANIRA_STAGE_FLAG_REALTIME_PRE_POST;
    }
    if ((hook_mask & (ANIRA_EM_HOOK_BEFORE | ANIRA_EM_HOOK_AFTER)) != 0) {
        desc.flags |= ANIRA_STAGE_FLAG_REALTIME_HOOKS;
    }
    desc.user_data = record.release();
}

// Fills an engine descriptor of trampolines over a new record; throws std::bad_alloc only.
void fill_engine_desc(anira_engine_desc& desc, uint32_t js_key, uint32_t hook_mask) {
    auto record = std::make_unique<HookRecord>();
    record->m_js_key = js_key;
    record->m_hook_mask = hook_mask;
    desc = ANIRA_ENGINE_DESC_INIT;
    desc.process = &engine_process;
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_RESET) != 0) { desc.reset = &engine_reset; }
    if ((hook_mask & ANIRA_EM_ENGINE_HOOK_QUERY) != 0) { desc.query = &engine_query; }
    desc.release = &engine_release;
    desc.user_data = record.release();
}

}  // namespace

uint32_t ANIRA_CALL anira_em_hooks_version(void) ANIRA_NOEXCEPT {
    return ANIRA_EM_HOOKS_VERSION;
}

anira_status ANIRA_CALL anira_em_stage_desc_init(anira_stage_desc* desc,
                                                 uint32_t js_key,
                                                 uint32_t hook_mask) ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_stage_hooks) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_stage_desc(*desc, js_key, hook_mask);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_em_pipeline_add_js_stage(anira_pipeline* pipeline,
                                                       uint32_t js_key,
                                                       uint32_t hook_mask,
                                                       anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE((hook_mask & ~k_stage_hooks) == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "JavaScript stage: hook_mask 0x%x carries a bit no ANIRA_EM_HOOK_* "
                       "defines (they are 0x%x)",
                       hook_mask,
                       k_stage_hooks);
    anira_stage_desc desc = ANIRA_STAGE_DESC_INIT;
    fill_stage_desc(desc, js_key, hook_mask);
    // A refused addition never calls release: until it succeeds the record is ours.
    std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(desc.user_data));
    const anira_status added = anira_pipeline_add_stage(pipeline, &desc, err);
    if (added == ANIRA_OK) { static_cast<void>(record.release()); }
    return added;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_em_engine_desc_init(anira_engine_desc* desc,
                                                  uint32_t js_key,
                                                  uint32_t hook_mask) ANIRA_NOEXCEPT try {
    if (desc == nullptr || (hook_mask & ~k_engine_hooks) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    fill_engine_desc(*desc, js_key, hook_mask);
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
    fill_engine_desc(desc, js_key, hook_mask);
    // A refused create never calls release: until it succeeds the record is ours.
    std::unique_ptr<HookRecord> record(static_cast<HookRecord*>(desc.user_data));
    if (engine_id != nullptr) { record->m_engine_id = engine_id; }
    desc.flags = engine_flags;
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
    if (enable != 0) {
        if (s_log_hook_sink == 0) {
            s_log_hook_sink = anira::detail::add_log_sink(&log_hook, nullptr, ANIRA_LOG_DEBUG);
        }
    } else if (s_log_hook_sink != 0) {
        anira::detail::remove_log_sink(s_log_hook_sink);
        s_log_hook_sink = 0;
    }
} catch (...) { anira::capi::report_void_failure(__func__); }

void* ANIRA_CALL anira_em_alloc_thread_stack(size_t bytes) ANIRA_NOEXCEPT {
    const size_t tls_bytes = __builtin_wasm_tls_size();
    const size_t total = stack_bytes(bytes) + tls_align() + tls_bytes;
    // A multiple of the alignment, as aligned_alloc wants; free() returns it.
    return std::aligned_alloc(k_stack_align,
                              (total + k_stack_align - 1) / k_stack_align * k_stack_align);
}

void ANIRA_CALL anira_em_free_thread_stack(void* base) ANIRA_NOEXCEPT {
    std::free(base);  // NOLINT(cppcoreguidelines-no-malloc)
}

anira_status ANIRA_CALL anira_em_init_instance(void* stack_base, size_t bytes) ANIRA_NOEXCEPT {
    if (stack_base == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    __wasm_init_tls(tls_block(stack_base, bytes));
    return ANIRA_OK;
}
