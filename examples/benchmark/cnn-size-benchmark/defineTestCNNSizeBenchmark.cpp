#include <anira/anira.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

TEST(Benchmark, CNNSize) {
    // Elevate this thread's priority for more consistent timing
    thl::core::Thread::set_current_priority(thl::core::ThreadPriority::RealTime);

    benchmark::RunSpecifiedBenchmarks();
}