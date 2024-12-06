#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <anira/anira.h>

TEST(Benchmark, CNNSize){
#if __linux__ || __APPLE__
    pthread_t self = pthread_self();
#elif WIN32
    HANDLE self = GetCurrentThread();
#endif
    anira::HighPriorityThread::elevate_priority(self, true);

    benchmark::RunSpecifiedBenchmarks();
}