#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../extras/models/cnn/CnnConfig.h"
#include "../../../extras/models/cnn/CnnPrePostProcessor.h"
#include "../../../extras/models/stateless-rnn/StatelessLstmConfig.h"
#include "../../../extras/models/stateless-rnn/StatelessLstmPrePostProcessor.h"
#include "../../../extras/models/stateful-rnn/StatefulLstmConfig.h"
#include "../../../extras/models/stateful-rnn/StatefulLstmPrePostProcessor.h"

// TODO Make sure that benchmarks also work when HOST_BUFFER_SIZE % MODEL_INPUT_SIZE != 0

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> bufferSizes = {2048, 4096, 8192};
std::vector<anira::InferenceBackend> inferenceBackends = {anira::LIBTORCH, anira::ONNX, anira::TFLITE};
std::vector<anira::InferenceConfig> inferenceConfigs = {cnnConfig, statelessRnnConfig, statefulRnnConfig};

// define the buffer sizes to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < bufferSizes.size(); ++i)
        for (int j = 0; j < inferenceBackends.size(); ++j)
            for (int k = 0; k < inferenceConfigs.size(); ++k)
                b->Args({bufferSizes[i], j, k});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig(1, getBufferSize(), SAMPLE_RATE);

    // TODO: Why is this necessary?
    anira::PrePostProcessor *myPrePostProcessor;
    if (state.range(2) == 0) {
        myPrePostProcessor = new CnnPrePostProcessor();
    } else if (state.range(2) == 1) {
        myPrePostProcessor = new StatelessLstmPrePostProcessor();
    } else if (state.range(2) == 2) {
        myPrePostProcessor = new StatefulLstmPrePostProcessor();
    }

    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(*myPrePostProcessor, inferenceConfigs[state.range(2)]);
    m_inferenceHandler->prepare(hostAudioConfig);
    m_inferenceHandler->setInferenceBackend(inferenceBackends[state.range(1)]);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(hostAudioConfig.hostChannels, hostAudioConfig.hostBufferSize);

    initializeRepetition(inferenceConfigs[state.range(2)], hostAudioConfig, inferenceBackends[state.range(1)]);

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