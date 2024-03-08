#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

TEST(Benchmark, All){
    // LockMemory();
    benchmark::RunSpecifiedBenchmarks();
    std::cout << "--------------------------------------------------------------" << std::endl;
}