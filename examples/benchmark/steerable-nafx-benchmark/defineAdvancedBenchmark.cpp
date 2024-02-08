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
#define BUFFER_SIZES {2048, 4096, 8192}
#define SAMPLE_RATE 44100

// define the buffer sizes to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    std::vector<int> bufferSizes = BUFFER_SIZES;
    for (int i = 0; i < bufferSizes.size(); ++i)
        for (int j = 0; j < 3; ++j)
            b->Args({bufferSizes[i], j});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;
MyPrePostProcessor myPrePostProcessor;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig(1, getBufferSize(), SAMPLE_RATE);
    anira::InferenceBackend inferenceBackend;

    if (state.range(1) == 0)
        inferenceBackend = anira::LIBTORCH;
    else if (state.range(1) == 1)
        inferenceBackend = anira::ONNX;
    else if (state.range(1) == 2)
        inferenceBackend = anira::TFLITE;

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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_ADVANCED)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->Apply(Arguments)
->ComputeStatistics("min", anira::benchmark::calculateMin)
->ComputeStatistics("max", anira::benchmark::calculateMax)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return anira::benchmark::calculatePercentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();