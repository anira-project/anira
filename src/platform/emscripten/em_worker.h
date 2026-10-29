// The Wasm Workers anira starts on WebAssembly (InferenceThread::start): a worker on the module's
// shared memory with a stack and thread-local block of its own, which first imports the
// configured ES module (the JavaScript engines and stages it defines on the scope's registry),
// awaits the registry's boot, then calls the entry. Compiled on the Wasm binary format only.
#ifndef ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H
#define ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace anira::em {

/// A Wasm Worker of start_worker and the memory it runs on.
struct Worker {
    int m_id = 0;              ///< emscripten_wasm_worker_t; 0: none
    void* m_region = nullptr;  ///< its stack and thread-local block
    void* m_boot = nullptr;    ///< the bootstrap's record
};

/// The stack of an inference Worker.
inline constexpr size_t k_worker_stack_bytes = size_t{4} * 1024 * 1024;

/// Starts a Wasm Worker that imports `module_url` (none when empty), awaits the boot of the
/// scope's registry and then calls `enter(arg)` on the worker. A module that fails to load is
/// reported on the console and the entry is called all the same. Once the entry returned, the
/// worker's JavaScript stores 1 to `left`, after its last C frame: from then on the worker
/// runs no C code and its region may go. Returns a worker of id 0 when the worker or its
/// region cannot be created. Main thread.
Worker start_worker(const std::string& module_url,
                    void (*enter)(void*),
                    void* arg,
                    std::atomic<int32_t>* left);

/// Terminates the worker. With `free_memory` its region and record are freed too, which is
/// legal only once the worker runs no C code any more (its `left` is 1); without, they are
/// left allocated (a worker terminated while it may still run C code). A worker of id 0 is a
/// no-op.
void terminate_worker(Worker& worker, bool free_memory);

}  // namespace anira::em

#endif  // ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H
