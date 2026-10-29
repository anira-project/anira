// The Wasm Workers of InferenceThread::start on WebAssembly: see em_worker.h.
#include "em_worker.h"

#include <emscripten/em_js.h>
#include <emscripten/wasm_worker.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

constexpr size_t k_align = 16;

// What the worker's bootstrap reads: the entry, its argument, the module to import first and
// the word it stores 1 to once the entry returned. Freed with the worker's region.
struct Boot {
    void (*m_enter)(void*) = nullptr;
    void* m_arg = nullptr;
    std::string m_module_url;
    std::atomic<int32_t>* m_left = nullptr;
};

// Called back by the bootstrap once the module and the registry's boot are done.
void enter_booted(void* boot) {
    const Boot& b = *static_cast<const Boot*>(boot);
    b.m_enter(b.m_arg);
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
                            reinterpret_cast<int32_t*>(b->m_left));
}

}  // namespace

namespace anira::em {

static_assert(sizeof(std::atomic<int32_t>) == sizeof(int32_t) &&
                  std::atomic<int32_t>::is_always_lock_free,
              "the bootstrap stores to the word through Atomics.store");

Worker start_worker(const std::string& module_url,
                    void (*enter)(void*),
                    void* arg,
                    std::atomic<int32_t>* left) {
    const size_t tls_align =
        __builtin_wasm_tls_align() > k_align ? __builtin_wasm_tls_align() : k_align;
    const size_t bytes = round_up(k_worker_stack_bytes + __builtin_wasm_tls_size(), tls_align);
    void* region = std::aligned_alloc(tls_align, bytes);
    if (region == nullptr) { return {}; }
    auto* boot = new Boot{enter, arg, module_url, left};  // NOLINT(cppcoreguidelines-owning-memory)
    const emscripten_wasm_worker_t id = emscripten_create_wasm_worker(region, bytes);
    if (id == 0) {
        delete boot;        // NOLINT(cppcoreguidelines-owning-memory)
        std::free(region);  // NOLINT(cppcoreguidelines-no-malloc)
        return {};
    }
    left->store(0, std::memory_order::release);
    emscripten_wasm_worker_post_function_vi(id,
                                            &boot_worker,
                                            static_cast<int>(reinterpret_cast<uintptr_t>(boot)));
    return Worker{id, region, boot};
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
