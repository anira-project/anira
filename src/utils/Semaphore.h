#ifndef ANIRA_SEMAPHORE_H
#define ANIRA_SEMAPHORE_H

#include <chrono>
#include <cstdint>

/**
 * Decide which semaphore implementation backs anira::Semaphore.
 *
 * std::counting_semaphore (and std::binary_semaphore) are annotated in libc++
 * as available only on macOS 11.0+ / iOS 14.0+. Using them with an older
 * deployment target is a compile-time availability error. On those targets we
 * fall back to moodycamel::LightweightSemaphore, which works on every platform.
 *
 * Define ANIRA_USE_LIGHTWEIGHT_SEMAPHORE=1 manually to force the fallback.
 */
#if !defined(ANIRA_USE_LIGHTWEIGHT_SEMAPHORE)
#if defined(__APPLE__)
#include <Availability.h>
#if (defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED < 110000) || \
    (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED < 140000)
#define ANIRA_USE_LIGHTWEIGHT_SEMAPHORE 1
#else
#define ANIRA_USE_LIGHTWEIGHT_SEMAPHORE 0
#endif
#else
#define ANIRA_USE_LIGHTWEIGHT_SEMAPHORE 0
#endif
#endif

#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
// lightweightsemaphore.h is not standalone: it relies on <cassert> and the
// MOODYCAMEL_DELETE_FUNCTION macro that concurrentqueue.h pulls in first.
#include <concurrentqueue.h>
#include <lightweightsemaphore.h>
#else
#include <semaphore>
#endif

namespace anira {

/**
 * @brief Thin semaphore wrapper that mirrors the std::semaphore interface.
 *
 * Backed either by std::binary_semaphore (modern platforms) or
 * moodycamel::LightweightSemaphore (older Apple deployment targets), selected
 * at compile time. Call sites use a single uniform API regardless of backend.
 */
class Semaphore {
public:
    explicit Semaphore(int initial_count = 0) : m_semaphore(initial_count) {}

    /** Increment the counter and unblock a waiter (std::semaphore::release). */
    void release() {
#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
        m_semaphore.signal();
#else
        m_semaphore.release();
#endif
    }

    /** Block until the counter is positive, then decrement (std::semaphore::acquire). */
    void acquire() {
#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
        m_semaphore.wait();
#else
        m_semaphore.acquire();
#endif
    }

    /** Try to decrement without blocking (std::semaphore::try_acquire). */
    bool try_acquire() {
#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
        return m_semaphore.tryWait();
#else
        return m_semaphore.try_acquire();
#endif
    }

    /** Try to decrement, blocking until at most wait_until (std::semaphore::try_acquire_until). */
    bool try_acquire_until(std::chrono::steady_clock::time_point wait_until) {
#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
        // LightweightSemaphore takes a relative timeout in microseconds; a
        // negative value would mean "wait forever", so clamp an elapsed
        // deadline to a non-blocking poll instead.
        auto usecs = std::chrono::duration_cast<std::chrono::microseconds>(
                         wait_until - std::chrono::steady_clock::now())
                         .count();
        if (usecs < 0) { usecs = 0; }
        return m_semaphore.wait(static_cast<std::int64_t>(usecs));
#else
        return m_semaphore.try_acquire_until(wait_until);
#endif
    }

private:
#if ANIRA_USE_LIGHTWEIGHT_SEMAPHORE
    moodycamel::LightweightSemaphore m_semaphore;
#else
    std::binary_semaphore m_semaphore;
#endif
};

}  // namespace anira

#endif  // ANIRA_SEMAPHORE_H
