// The Wasm Workers of InferenceThread::start on WebAssembly: see em_worker.h.
#include "em_worker.h"

#include <anira/utils/Logger.h>
#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>
#include <emscripten/eventloop.h>
#include <emscripten/wasm_worker.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr size_t k_align = 16;

constexpr uint32_t k_booting = 0;
constexpr uint32_t k_entered = 1;
constexpr uint32_t k_cancelled = 2;

// How often the reaper looks at the cancelled workers, and how long a cancelled worker may take
// to finish its boot before it is terminated where it stands (its region then leaked).
constexpr double k_reap_interval_ms = 100.0;
constexpr double k_reap_timeout_ms = 10000.0;

// What the worker's bootstrap reads: the entry, its argument, the module to import first, the
// state the entry and a cancel race on, and the word the bootstrap stores 1 to once the entry
// returned or was skipped. Owned by the main thread, freed with the worker's region; it outlives
// the thread object of the entry's argument, which a cancelled worker never touches.
struct Boot {
    void (*m_enter)(void*) = nullptr;
    void* m_arg = nullptr;
    std::string m_module_url;
    std::atomic<uint32_t> m_state{k_booting};
    std::atomic<int32_t> m_left{0};
};

// Called back by the bootstrap once the module and the registry's boot are done: the entry,
// unless cancel_boot() won the race (then the argument may be gone already).
void enter_booted(void* boot) {
    Boot& b = *static_cast<Boot*>(boot);
    uint32_t expected = k_booting;
    if (!b.m_state.compare_exchange_strong(expected, k_entered, std::memory_order::acq_rel)) {
        return;
    }
    b.m_enter(b.m_arg);
}

void reap(void* user_data);

// A main-thread timer. emscripten_set_timeout's callback ends in the runtime's maybeExit, which
// leaves the runtime alive because the module links noExitRuntime (cmake/build-wasm.cmake).
void arm_reaper() {
    emscripten_set_timeout(&reap, k_reap_interval_ms, nullptr);
}

// The cancelled workers that have not finished their boot yet, on the main thread.
struct Cancelled {
    anira::em::Worker m_worker;
    double m_since = 0.0;
};
std::vector<Cancelled> s_cancelled;
bool s_reaper_armed = false;

// The one place a cancelled worker's memory goes: a main-thread timer, every 100 ms while any
// remain. A worker that stored its left word is terminated (idle by then: its boot is over)
// and its region and record freed; one still booting after 10 s is terminated where it stands
// and its region left allocated, since it may be inside a C frame.
void reap(void* /*user_data*/) {
    s_reaper_armed = false;
    const double now = emscripten_get_now();
    std::vector<Cancelled> waiting;
    for (Cancelled& c : s_cancelled) {
        if (anira::em::has_left(c.m_worker)) {
            anira::em::terminate_worker(c.m_worker, true);
        } else if (now - c.m_since > k_reap_timeout_ms) {
            ANIRA_LOG_WARNING(anira::log_group::k_scheduler,
                              "inference thread: a cancelled Wasm Worker did not finish its boot "
                              "(its module's import or an engine's init) within %.0f ms; it is "
                              "terminated where it stands and its %zu-byte region is leaked, "
                              "since it may be inside a C frame",
                              k_reap_timeout_ms,
                              anira::em::k_worker_stack_bytes);
            anira::em::terminate_worker(c.m_worker, false);
        } else {
            waiting.push_back(c);
        }
    }
    s_cancelled = std::move(waiting);
    if (!s_cancelled.empty()) {
        s_reaper_armed = true;
        arm_reaper();
    }
}

size_t round_up(size_t bytes, size_t align) {
    return (bytes + align - 1) / align * align;
}

}  // namespace

// The worker's first JavaScript: import the module (the engines and stages it defines on the
// scope's registry, globalThis[Symbol.for('anira.registry')]), await the registry's boot over
// this instance (its JavaScript engines' init), then call the entry through the table, and
// once it returned store 1 to `left` from JavaScript, after the last C frame of the worker. The
// import and the boot are asynchronous: the entry runs from a later task of the worker.
EM_JS_DEPS(anira_em_worker, "$getWasmTableEntry,$UTF8ToString");

EM_JS(void,
      anira_em_js_boot_worker,
      (const char* module_url, uintptr_t enter, void* boot, int32_t* left),
      {
          var url = module_url ? UTF8ToString(module_url) : "";
          // A template literal: Vite's dev server rewrites a dynamic import of a bare variable
          // into one through its client, which a worklet scope cannot load
          // (cmake/wasm-glue-rewrite.cmake). clang-format would split `${url}` into `$ { url }`,
          // which is no longer an interpolation.
          // clang-format off
          var loaded = url ? import(`${url}`) : Promise.resolve();
          // clang-format on
          loaded
              .then(function() {
                  var registry = globalThis[Symbol.for ('anira.registry')];
                  return registry && registry.boot ? registry.boot(Module) : undefined;
              })
              .catch(function(e) {
                  console.error('anira: the inference Worker module ' + url +
                                    ' failed to load; its JavaScript engines are missing:',
                                e);
              })
              .then(function() {
                  try {
                      getWasmTableEntry(enter)(boot);
                  } finally { Atomics.store(HEAP32, left >> 2, 1); }
              });
      });

namespace {

// The worker's entry, posted by start_worker.
void boot_worker(int boot) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr) the posted argument is the record's address
    auto* b = reinterpret_cast<Boot*>(static_cast<uintptr_t>(boot));
    anira_em_js_boot_worker(b->m_module_url.empty() ? nullptr : b->m_module_url.c_str(),
                            reinterpret_cast<uintptr_t>(&enter_booted),
                            b,
                            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                            reinterpret_cast<int32_t*>(&b->m_left));
}

}  // namespace

namespace anira::em {

static_assert(sizeof(std::atomic<int32_t>) == sizeof(int32_t) &&
                  std::atomic<int32_t>::is_always_lock_free,
              "the bootstrap stores to the word through Atomics.store");

Worker start_worker(const std::string& module_url, void (*enter)(void*), void* arg) {
    const size_t tls_align =
        __builtin_wasm_tls_align() > k_align ? __builtin_wasm_tls_align() : k_align;
    const size_t bytes = round_up(k_worker_stack_bytes + __builtin_wasm_tls_size(), tls_align);
    void* region = std::aligned_alloc(tls_align, bytes);
    if (region == nullptr) { return {}; }
    auto* boot = new Boot();  // NOLINT(cppcoreguidelines-owning-memory)
    boot->m_enter = enter;
    boot->m_arg = arg;
    boot->m_module_url = module_url;
    const emscripten_wasm_worker_t id = emscripten_create_wasm_worker(region, bytes);
    if (id == 0) {
        delete boot;        // NOLINT(cppcoreguidelines-owning-memory)
        std::free(region);  // NOLINT(cppcoreguidelines-no-malloc)
        return {};
    }
    emscripten_wasm_worker_post_function_vi(id,
                                            &boot_worker,
                                            static_cast<int>(reinterpret_cast<uintptr_t>(boot)));
    return Worker{id, region, boot};
}

bool cancel_boot(Worker& worker) {
    if (worker.m_id == 0) { return false; }
    auto& boot = *static_cast<Boot*>(worker.m_boot);
    uint32_t expected = k_booting;
    if (!boot.m_state.compare_exchange_strong(expected, k_cancelled, std::memory_order::acq_rel)) {
        return false;
    }
    s_cancelled.push_back(Cancelled{worker, emscripten_get_now()});
    worker = Worker{};
    if (!s_reaper_armed) {
        s_reaper_armed = true;
        arm_reaper();
    }
    return true;
}

bool has_left(const Worker& worker) {
    return worker.m_boot != nullptr &&
           static_cast<const Boot*>(worker.m_boot)->m_left.load(std::memory_order::acquire) != 0;
}

void terminate_worker(Worker& worker, bool free_memory) {
    if (worker.m_id == 0) { return; }
    emscripten_terminate_wasm_worker(worker.m_id);
    if (free_memory) {
        std::free(worker.m_region);                // NOLINT(cppcoreguidelines-no-malloc)
        delete static_cast<Boot*>(worker.m_boot);  // NOLINT(cppcoreguidelines-owning-memory)
    }
    worker = Worker{};
}

}  // namespace anira::em
