#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../extras/models/cnn/advanced-configs/CNNAdvancedConfigs.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/advanced-configs/HybridNNAdvancedConfigs.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/stateful-rnn/advanced-configs/StatefulRNNAdvancedConfigs.h"
#include "../../../extras/models/stateful-rnn/StatefulRNNPrePostProcessor.h"

// TODO Make sure that benchmarks also work when HOST_BUFFER_SIZE % MODEL_INPUT_SIZE != 0

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> bufferSizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
std::vector<anira::InferenceBackend> inferenceBackends = {anira::LIBTORCH, anira::ONNX, anira::TFLITE};
std::vector<AdvancedInferenceConfigs> advancedInferenceConfigs = {cnnAdvancedConfigs, hybridNNAdvancedConfigs, statefulRNNAdvancedConfigs};

// define the buffer sizes to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < bufferSizes.size(); ++i)
        for (int j = 0; j < advancedInferenceConfigs.size(); ++j)
            for (int k = 0; k < inferenceBackends.size(); ++k)
                // ONNX backend does not support stateful RNN
                if (!(j == 2 && k == 1))
                    b->Args({bufferSizes[i], j, k});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig = {1, (size_t) getBufferSize(), SAMPLE_RATE};

    AdvancedInferenceConfigs currentAdvancedInferenceConfigs = advancedInferenceConfigs[state.range(1)];
    anira::InferenceConfig inferenceConfig;

    for (auto advancedConfig : currentAdvancedInferenceConfigs) {
        if (advancedConfig.bufferSize == getBufferSize()) {
            inferenceConfig = advancedConfig.config;
        }
    }

    anira::PrePostProcessor *myPrePostProcessor;

    if (state.range(1) == 0) {
        myPrePostProcessor = new CNNPrePostProcessor();
        static_cast<CNNPrePostProcessor*>(myPrePostProcessor)->config = inferenceConfig;
    } else if (state.range(1) == 1) {
        myPrePostProcessor = new HybridNNPrePostProcessor();
        static_cast<HybridNNPrePostProcessor*>(myPrePostProcessor)->config = inferenceConfig;
    } else if (state.range(1) == 2) {
        myPrePostProcessor = new StatefulRNNPrePostProcessor();
    }

    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(*myPrePostProcessor, inferenceConfig);
    m_inferenceHandler->prepare(hostAudioConfig);
    m_inferenceHandler->setInferenceBackend(inferenceBackends[state.range(2)]);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(hostAudioConfig.hostChannels, hostAudioConfig.hostBufferSize);

    initializeRepetition(inferenceConfig, hostAudioConfig, inferenceBackends[state.range(2)]);

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