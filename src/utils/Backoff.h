/*
 * The backoff of anira's polling loops, private to the library: a few immediate retries, then
 * retries with a CPU pause hint between them, then a yield and a 100 us sleep before every
 * further retry. The inference thread's wait for work (InferenceThread::exponential_backoff,
 * ANIRA_WAIT_SPIN_BACKOFF), the engine room's wait for a free instance of a loaded model
 * (engine::Loaded::claim_and_run, for a caller that runs a shared call directly) and the spin
 * locks of the scheduler's wait lists (InstanceWaitList) and of the JavaScript engines' list of
 * loaded models (em_hooks.cpp) poll with it. Inference and control threads only: the last phase
 * sleeps, so the audio thread never waits here.
 *
 * On WebAssembly the last phase of a Wasm Worker (every inference thread there is one) is a
 * wait on a word nobody notifies, with the sleep's time as its timeout, instead of the yield
 * and the sleep: see pause().
 */
#ifndef ANIRA_UTILS_BACKOFF_H
#define ANIRA_UTILS_BACKOFF_H

#include <chrono>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include <immintrin.h>
#endif

#if defined(__EMSCRIPTEN__) && defined(__wasm_atomics__)
#include <pthread.h>

#include <cstdint>
#endif

namespace anira::detail {

/// A short busy-wait hint to the CPU: a few pause instructions where the architecture has one.
inline void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    _mm_pause();
    _mm_pause();
#elif __aarch64__
    // ISB instruction is better than WFE
    // https://stackoverflow.com/questions/70810121/why-does-hintspin-loop-use-isb-on-aarch64
    // Still on linux it maxes out the CPU, so the last phase of Backoff sleeps
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
    asm volatile("isb sy");
#elif __arm__
    asm volatile("yield");
    asm volatile("yield");
    asm volatile("yield");
    asm volatile("yield");
#endif
}

/// The wait between two retries of a polling loop, one pause() per failed retry: nothing for
/// the first `immediate` retries, cpu_pause() for the next `paused`, then a yield and a 100 us
/// sleep for every retry after that.
class Backoff {
public:
    /// The inference thread's schedule: 4 immediate retries, then 32 paused ones (about
    /// 100 ns each).
    static constexpr int k_immediate = 4;
    static constexpr int k_paused = 32;
    static constexpr std::chrono::microseconds k_sleep{100};

    constexpr Backoff() noexcept = default;
    constexpr Backoff(int immediate, int paused) noexcept
        : m_immediate(immediate), m_paused(paused) {}

    /// Whether the retries have reached the last phase, where pause() yields and sleeps: the
    /// inference thread parks on its wake word there instead on WebAssembly.
    [[nodiscard]] bool in_last_phase() const noexcept { return m_step >= m_immediate + m_paused; }

    /// Waits before the next retry, by the phase the retries so far have reached.
    void pause() noexcept {
        if (m_step < m_immediate) {
            ++m_step;
            return;
        }
        if (m_step < m_immediate + m_paused) {
            ++m_step;
            cpu_pause();
            return;
        }
        // The sleep is what matters: without it the thread consumes a whole core, the pause
        // hints (ISB, WFE) included, and on Linux a thread spinning like that was seen to be
        // suspended by the OS for a while now and then (missing samples).
#if defined(__EMSCRIPTEN__) && defined(__wasm_atomics__)
        // WebAssembly: a Wasm Worker is no pthread. It has no thread record (pthread_self() is
        // null), and this libc's sleep is a pthread's: before it waits it runs the queue of
        // calls proxied to the calling thread, by that record. A build with Emscripten's
        // assertions aborts the Worker there (assert(pthread_self()), proxying.c); one without
        // finds no queue for a thread without a record and goes on to sleep. So a thread
        // without a record waits here itself, on a word of its own that nobody notifies, for
        // the sleep's time (what emscripten_wasm_worker_sleep does); no yield, which does
        // nothing on such a thread. The audio worklet has no record either and may not wait at
        // all: it never backs off (see the file comment). A thread with a record (the main
        // thread, a host's pthread) sleeps through the libc below, as everywhere.
        if (pthread_self() == pthread_t{}) {
            int32_t unnotified = 0;
            __builtin_wasm_memory_atomic_wait32(&unnotified,
                                                0,
                                                std::chrono::nanoseconds(k_sleep).count());
            return;
        }
#endif
        std::this_thread::yield();
        std::this_thread::sleep_for(k_sleep);
    }

private:
    int m_immediate = k_immediate;
    int m_paused = k_paused;
    int m_step = 0;
};

}  // namespace anira::detail

#endif  // ANIRA_UTILS_BACKOFF_H
