#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/models/stateful-rnn/StatefulRNNPrePostProcessor.h"


/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> bufferSizes = {2048, 4096, 8192};
std::vector<anira::InferenceConfig> inferenceConfigs = {cnnConfig, hybridNNConfig, statefulRNNConfig};

// define the buffer sizes to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < bufferSizes.size(); ++i)
        for (int j = 0; j < inferenceConfigs.size(); ++j)
            b->Args({bufferSizes[i], j});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig = {1, (size_t) getBufferSize(), SAMPLE_RATE};
    anira::InferenceBackend inferenceBackend = anira::NONE;

    // TODO: Why is this necessary?
    anira::PrePostProcessor *myPrePostProcessor;
    if (state.range(1) == 0) {
        myPrePostProcessor = new CNNPrePostProcessor();
    } else if (state.range(1) == 1) {
        myPrePostProcessor = new HybridNNPrePostProcessor();
    } else if (state.range(1) == 2) {
        myPrePostProcessor = new StatefulRNNPrePostProcessor();
    }

    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(*myPrePostProcessor, inferenceConfigs[state.range(1)]);
    m_inferenceHandler->prepare(hostAudioConfig);
    m_inferenceHandler->setInferenceBackend(inferenceBackend);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(hostAudioConfig.hostChannels, hostAudioConfig.hostBufferSize);

    initializeRepetition(inferenceConfigs[state.range(1)], hostAudioConfig, inferenceBackend);

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

    delete myPrePostProcessor;
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