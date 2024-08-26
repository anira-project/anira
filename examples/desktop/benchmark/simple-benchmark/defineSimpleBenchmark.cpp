#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../../extras/desktop/models/cnn/CNNConfig.h"
#include "../../../../extras/desktop/models/cnn/CNNPrePostProcessor.h"
#include "../../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../../extras/desktop/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../../extras/desktop/models/stateful-rnn/StatefulRNNPrePostProcessor.h"


/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define BUFFER_SIZE 2048
#define SAMPLE_RATE 44100

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

CNNPrePostProcessor myPrePostProcessor;
// HybridNNPrePostProcessor myPrePostProcessor;
// StatefulRNNPrePostProcessor myPrePostProcessor;

anira::InferenceConfig myConfig = cnnConfig;
// anira::InferenceConfig myConfig = hybridNNConfig;
// anira::InferenceConfig myConfig = statefulRNNConfig;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {

    // The buffer size return in getBufferSize() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig hostAudioConfig = {1, (size_t) getBufferSize(), SAMPLE_RATE};
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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->Arg(BUFFER_SIZE)
->UseManualTime();