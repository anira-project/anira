#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <tanh/core/threading/Thread.h>

TEST(Benchmark, Advanced) {
    // Elevate this thread's priority for more consistent timing
    thl::core::Thread::set_current_priority(thl::core::ThreadPriority::RealTime);

    benchmark::RunSpecifiedBenchmarks();
}