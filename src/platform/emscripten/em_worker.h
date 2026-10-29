// The Wasm Workers anira starts on WebAssembly (InferenceThread::start): a worker on the module's
// shared memory with a stack and thread-local block of its own, which first imports the
// configured ES module (the JavaScript engines and stages it defines on the scope's registry),
// awaits the registry's boot, then calls the entry. Compiled on the Wasm binary format only.
#ifndef ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H
#define ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H

#include <cstddef>
#include <string>

namespace anira::em {

/// A Wasm Worker of start_worker and the memory it runs on.
struct Worker {
    int m_id = 0;              ///< emscripten_wasm_worker_t; 0: none
    void* m_region = nullptr;  ///< its stack and thread-local block
    void* m_boot = nullptr;    ///< the bootstrap's record: the entry, the cancel, the left word
};

/// The stack of an inference Worker.
inline constexpr size_t k_worker_stack_bytes = size_t{4} * 1024 * 1024;

/// Starts a Wasm Worker that imports `module_url` (none when empty), awaits the boot of the
/// scope's registry and then calls `enter(arg)` on the worker, unless cancel_boot() came first.
/// A module that fails to load is reported on the console and the entry is called all the same.
/// Once the entry returned (or was skipped), the worker's JavaScript stores 1 to the record's
/// left word, after its last C frame: from then on the worker runs no C code and its region may
/// go (has_left). Returns a worker of id 0 when the worker or its region cannot be created.
/// Main thread.
Worker start_worker(const std::string& module_url, void (*enter)(void*), void* arg);

/// Cancels a worker that has not called its entry yet: true, and the worker is handed to the
/// reaper (and `worker` cleared), which frees its region and record once the worker has
/// finished its boot and stored its left word, terminating it only then. A worker is never
/// terminated in the middle of its boot (a module import or an engine's init in flight): a
/// browser reports that as an error (Firefox: NS_ERROR_FAILURE, "uncaught exception"). False
/// when the entry already runs (stop the loop, wait for has_left, then terminate_worker).
/// Main thread.
bool cancel_boot(Worker& worker);

/// Whether the worker's entry returned and it runs no C code any more.
bool has_left(const Worker& worker);

/// Terminates the worker. With `free_memory` its region and record are freed too, which is
/// legal only once has_left(); without, they are left allocated (a worker terminated while it
/// may still run C code). A worker of id 0 is a no-op. Main thread.
void terminate_worker(Worker& worker, bool free_memory);

}  // namespace anira::em

#endif  // ANIRA_PLATFORM_EMSCRIPTEN_EM_WORKER_H
