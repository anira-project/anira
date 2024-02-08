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

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig(1, getBufferSize(), 48000);
    anira::InferenceBackend inferenceBackend = anira::LIBTORCH;

    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(myPrePostProcessor, myConfig);
    m_inferenceHandler->prepare(hostAudioConfig);
    m_inferenceHandler->setInferenceBackend(inferenceBackend);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(hostAudioConfig.hostChannels, hostAudioConfig.hostBufferSize);

    initializeRepetition(myConfig, hostAudioConfig, inferenceBackend);

    for (auto _ : state) {
        pushRandomSamplesInBuffer(hostAudioConfig);

        initializeIteration();

        auto start = std::chrono::high_resolution_clock::now();
        
        m_inferenceHandler->process(m_buffer->getArrayOfWritePointers(), getBufferSize());

        while (!bufferHasBeenProcessed()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        interationStep(start, end, state);
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