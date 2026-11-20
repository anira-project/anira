/*
 * The node test of the Backoff's last phase on WebAssembly (src/utils/Backoff.h, a private
 * header: this gate compiles it alone, without the library). Every inference thread there is a
 * Wasm Worker, and a Wasm Worker is no pthread: it has no thread record, and the libc's sleep,
 * which runs the calling pthread's queue of proxied calls first, aborts on one in a build with
 * Emscripten's assertions (this gate is one). A Worker of this test runs a Backoff through its
 * two spin phases and pauses in the last one: every pause must return, and wait. The main
 * thread, which has a record, does the same through the libc's sleep. Every failed check prints
 * one line; exit code 0 on success.
 */
#include <emscripten/emscripten.h>
#include <emscripten/wasm_worker.h>

#include <atomic>
#include <cstdio>

#include "utils/Backoff.h"

namespace {

using anira::detail::Backoff;

constexpr int k_last_phase_pauses = 16;
constexpr unsigned k_stack_bytes = 1024u * 1024u;
constexpr double k_worker_bound_ms = 10000.0;
// A pause of the last phase waits Backoff::k_sleep (100 us). The bound is half of that per
// pause: a wait may end late, by as much as the machine likes, and never early.
constexpr double k_least_ms = k_last_phase_pauses * 0.05;

int s_failures = 0;
// The Worker's result, written before the flag.
double s_worker_ms = 0.0;
std::atomic<bool> s_worker_done{false};

void check(bool ok, const char* what) {
    if (!ok) {
        printf("FAILED: %s\n", what);
        ++s_failures;
    }
}

// A Backoff run through its spin phases, then paused in its last phase: the milliseconds those
// pauses took, or a negative number when the spin phases did not end in the last phase.
double last_phase_ms() {
    Backoff backoff;
    for (int i = 0; i < Backoff::k_immediate + Backoff::k_paused; ++i) { backoff.pause(); }
    if (!backoff.in_last_phase()) { return -1.0; }
    const double start = emscripten_get_now();
    for (int i = 0; i < k_last_phase_pauses; ++i) { backoff.pause(); }
    return emscripten_get_now() - start;
}

void on_worker() {
    s_worker_ms = last_phase_ms();
    s_worker_done.store(true, std::memory_order::release);
}

}  // namespace

int main() {
    const emscripten_wasm_worker_t worker = emscripten_malloc_wasm_worker(k_stack_bytes);
    check(worker != 0, "the test's Wasm Worker");
    emscripten_wasm_worker_post_function_v(worker, &on_worker);
    // A Worker loads the module's glue before it runs anything: wait for its result, bounded.
    const double posted = emscripten_get_now();
    while (!s_worker_done.load(std::memory_order::acquire) &&
           emscripten_get_now() - posted < k_worker_bound_ms) {}
    const bool done = s_worker_done.load(std::memory_order::acquire);
    check(done, "the Wasm Worker came back from the last phase of its Backoff");
    if (done) {
        printf("a Wasm Worker: %d pauses of the last phase took %.3f ms\n",
               k_last_phase_pauses,
               s_worker_ms);
        check(s_worker_ms >= k_least_ms, "a Wasm Worker's pause of the last phase waits");
    }
    emscripten_terminate_wasm_worker(worker);

    const double main_ms = last_phase_ms();
    printf("the main thread: %d pauses of the last phase took %.3f ms\n",
           k_last_phase_pauses,
           main_ms);
    check(main_ms >= k_least_ms, "the main thread's pause of the last phase waits");

    printf("%s: %d failed check(s)\n", s_failures == 0 ? "PASSED" : "FAILED", s_failures);
    return s_failures == 0 ? 0 : 1;
}
