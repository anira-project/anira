// anira::HighPriorityThread is deprecated (it is a thin wrapper over
// thl::core::Thread kept for one minor release), but it is still public API and
// still shipped, so its start/stop/should_exit contract is under test until it
// is removed. The deprecation warning is suppressed here on purpose: exercising
// a deprecated class is exactly what this file is for.

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <anira/system/HighPriorityThread.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

namespace {

class CountingThread : public anira::HighPriorityThread {
public:
    // stop() belongs in the *derived* destructor. ~HighPriorityThread() also calls
    // it, but a base destructor runs after the derived part is gone, so a worker
    // still inside run() would touch members that no longer exist — UBSan reports
    // exactly that ("member access within address ... which does not point to an
    // object of type CountingThread").
    ~CountingThread() override { stop(); }

    void run() override {
        m_entered = true;
        while (!should_exit()) {
            m_iterations.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        *m_left = true;
    }

    std::atomic<bool> m_entered{false};
    // Shared, not a member flag: DestructorStopsARunningThread reads it after the
    // thread object is gone, and a pointer into the destroyed object would dangle.
    std::shared_ptr<std::atomic<bool>> m_left = std::make_shared<std::atomic<bool>>(false);
    std::atomic<int> m_iterations{0};
};

// Spins until the predicate holds or the deadline passes, so the test does not
// depend on how quickly the OS schedules the new thread.
template <typename Predicate>
bool wait_for(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

}  // namespace

TEST(HighPriorityThread, StartRunsAndStopJoins) {
    CountingThread thread;
    EXPECT_FALSE(thread.is_running());

    thread.start();
    EXPECT_TRUE(wait_for([&thread] { return thread.m_entered.load(); }));
    EXPECT_TRUE(thread.is_running());
    EXPECT_TRUE(wait_for([&thread] { return thread.m_iterations.load() > 0; }));

    thread.stop();
    // stop() joins, so run() has returned by the time it does.
    EXPECT_TRUE(thread.m_left->load());
    EXPECT_FALSE(thread.is_running());
    EXPECT_TRUE(thread.should_exit());
}

// A thread left running is joined by destruction, through the derived
// destructor's stop().
TEST(HighPriorityThread, DerivedDestructorStopsARunningThread) {
    std::shared_ptr<std::atomic<bool>> left;
    {
        CountingThread thread;
        thread.start();
        EXPECT_TRUE(wait_for([&thread] { return thread.m_entered.load(); }));
        left = thread.m_left;
    }
    // The object is gone; the flag was set before the destructor returned.
    EXPECT_TRUE(left->load());
}

// stop() on a thread that was never started must be a no-op, not a crash or a
// hang on a join of nothing.
TEST(HighPriorityThread, StopWithoutStartIsANoOp) {
    CountingThread thread;
    thread.stop();
    EXPECT_FALSE(thread.is_running());
    EXPECT_FALSE(thread.m_entered.load());
}

// elevate_priority() ignores both of its parameters and acts on the calling
// thread. It must not throw or abort when the platform refuses the request
// (an unprivileged CI runner typically does).
TEST(HighPriorityThread, ElevatePriorityOnTheCallingThread) {
    EXPECT_NO_THROW(anira::HighPriorityThread::elevate_priority(std::thread::native_handle_type{}));
    EXPECT_NO_THROW(
        anira::HighPriorityThread::elevate_priority(std::thread::native_handle_type{}, true));
}

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
