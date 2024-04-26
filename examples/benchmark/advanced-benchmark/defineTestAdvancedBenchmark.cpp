#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <anira/anira.h>

TEST(Benchmark, Advanced){
#if __linux__
    pthread_t self = pthread_self();
#elif WIN32
    HANDLE self = GetCurrentThread();
#endif
    anira::system::RealtimeThread::elevateToRealTimePriority(self, true);

    benchmark::RunSpecifiedBenchmarks();
}