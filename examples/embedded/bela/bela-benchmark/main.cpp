#include <benchmark/benchmark.h>
#include <anira/anira.h>

int main(int argc, char** argv) {
    // Initialize benchmark
    benchmark::Initialize(&argc, argv);

    pthread_t self = pthread_self();

    anira::RealtimeThread::elevateToRealTimePriority(self, true);

    // Run benchmark
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}