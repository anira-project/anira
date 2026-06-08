#include "gtest/gtest.h"
#include <anira/anira.h>

#include <chrono>

using namespace anira;

// These tests exercise whichever backend anira::Semaphore selected at compile
// time: std::binary_semaphore on macOS 11.0+/iOS 14+/Linux/Windows, or
// moodycamel::LightweightSemaphore on older Apple deployment targets. The
// observable behavior is expected to be identical for both.

// A freshly constructed semaphore with count 0 has nothing to acquire.
TEST(SemaphoreTest, TryAcquireOnEmptyFails) {
    Semaphore sem{0};
    EXPECT_FALSE(sem.try_acquire());
}

// release() makes exactly one acquire succeed; the next try_acquire fails.
TEST(SemaphoreTest, ReleaseThenTryAcquireSucceedsOnce) {
    Semaphore sem{0};
    sem.release();
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());
}

// A semaphore constructed with an initial count is immediately acquirable.
TEST(SemaphoreTest, InitialCountIsAcquirable) {
    Semaphore sem{1};
    EXPECT_TRUE(sem.try_acquire());
    EXPECT_FALSE(sem.try_acquire());
}

// Blocking acquire() returns promptly when a permit is already available.
TEST(SemaphoreTest, AcquireReturnsWhenPermitAvailable) {
    Semaphore sem{0};
    sem.release();
    sem.acquire(); // must not deadlock
    SUCCEED();
}

// try_acquire_until on an empty semaphore returns false and actually waits
// for approximately the requested timeout before giving up.
TEST(SemaphoreTest, TryAcquireUntilTimesOut) {
    Semaphore sem{0};
    constexpr auto timeout = std::chrono::milliseconds(20);

    const auto start = std::chrono::steady_clock::now();
    const bool acquired = sem.try_acquire_until(start + timeout);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(acquired);
    // It should have waited at least most of the timeout. Allow a small slack
    // to tolerate timer granularity / early wakeups across platforms.
    EXPECT_GE(elapsed, timeout - std::chrono::milliseconds(5));
}

// try_acquire_until succeeds immediately when a permit is available, well
// before the deadline elapses.
TEST(SemaphoreTest, TryAcquireUntilSucceedsImmediately) {
    Semaphore sem{0};
    sem.release();

    const auto start = std::chrono::steady_clock::now();
    const bool acquired = sem.try_acquire_until(start + std::chrono::seconds(5));
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(acquired);
    EXPECT_LT(elapsed, std::chrono::seconds(1));
}

// An already-elapsed deadline must not block: it behaves like a non-blocking
// poll and returns promptly (exercises the clamp-to-zero path in the wrapper).
TEST(SemaphoreTest, TryAcquireUntilPastDeadlineDoesNotBlock) {
    Semaphore sem{0};
    const auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    const auto start = std::chrono::steady_clock::now();
    const bool acquired = sem.try_acquire_until(past);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(acquired);
    EXPECT_LT(elapsed, std::chrono::milliseconds(100));
}
