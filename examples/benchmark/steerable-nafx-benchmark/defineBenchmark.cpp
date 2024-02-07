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

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_LIBTORCH_BACKEND)(::benchmark::State& state) {

    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(myPrePostProcessor, myConfig);
    m_inferenceHandler->prepare(anira::HostAudioConfig(1, getBufferSize(), 44100));
    m_inferenceHandler->setInferenceBackend(anira::LIBTORCH);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(1, getBufferSize());

    int iteration = 0;

    for (auto _ : state) {
        pushRandomSamplesInBuffer(anira::HostAudioConfig(1, getBufferSize(), 44100));

        initializeIteration();

        auto start = std::chrono::high_resolution_clock::now();
        
        m_inferenceHandler->process(m_buffer->getArrayOfWritePointers(), getBufferSize());

        while (!bufferHasBeenProcessed()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        interationStep(start, end, iteration, state);
    }
    repetitionStep();
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