#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "MyConfig.h"
#include "MyPrePostProcessor.h"

// TODO Make sure that benchmarks also work when HOST_BUFFER_SIZE % MODEL_INPUT_SIZE != 0

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define STARTING_BUFFER_SIZE 2048
#define STOPPING_BUFFER_SIZE 8192

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;
MyPrePostProcessor myPrePostProcessor;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_LIBTORCH_BACKEND)(benchmark::State& state) {
    constructInferenceHandler(myPrePostProcessor, myConfig);
    setInferenceBackend(anira::LIBTORCH);

    // the buffer size is set to the first value of the state range
    prepareInferenceHandler(anira::HostAudioConfig(1, getBufferSize(), 44100));

    int iteration = 0;

    for (auto _ : state) {
        pushSamplesInBuffer(anira::HostAudioConfig(1, getBufferSize(), 44100));

        bool init = isInitializing();
        int prevNumReceivedSamples = getNumReceivedSamples();

        auto start = std::chrono::high_resolution_clock::now();
        
        inferenceHandler->process(getArrayOfWritePointers(), getBufferSize());

        if (init) {
            while (getNumReceivedSamples() < prevNumReceivedSamples + getBufferSize()){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }
        else {
            while (getNumReceivedSamples() < prevNumReceivedSamples){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << state.name() << "/" << state.range(0) << "/iteration:" << iteration << "/repetition:" << getRepetition() << "\t\t\t" << elapsedTimeMS.count() << std::endl;
        iteration++;
    }
    repetitionStep();

    std::cout << "\n------------------------------------------------------------------------------------------------\n" << std::endl;
}

// /* ============================================================ *
//  * ================== BENCHMARK REGISTRATION ================== *
//  * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_LIBTORCH_BACKEND)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->RangeMultiplier(2)->Range(STARTING_BUFFER_SIZE, STOPPING_BUFFER_SIZE)
->ComputeStatistics("min", anira::benchmark::calculateMin)
->ComputeStatistics("max", anira::benchmark::calculateMax)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return anira::benchmark::calculatePercentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();