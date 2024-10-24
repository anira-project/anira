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

CNNPrePostProcessor my_pp_processor;
// HybridNNPrePostProcessor my_pp_processor;
// StatefulRNNPrePostProcessor my_pp_processor;

anira::InferenceConfig my_inference_config = cnn_config;
// anira::InferenceConfig my_inference_config = hybridnn_config;
// anira::InferenceConfig my_inference_config = rnn_config;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {

    // The buffer size return in get_buffer_size() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig host_config = {1, (size_t) get_buffer_size(), SAMPLE_RATE};
    anira::InferenceBackend inference_backend = anira::LIBTORCH;

    m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor, my_inference_config);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backend);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(host_config.m_host_channels, host_config.m_host_buffer_size);

    initialize_repetition(my_inference_config, host_config, inference_backend);

    for (auto _ : state) {
        push_random_samples_in_buffer(host_config);

        initialize_iteration();

        auto start = std::chrono::high_resolution_clock::now();
        
        m_inference_handler->process(m_buffer->get_array_of_write_pointers(), get_buffer_size());

        while (!buffer_processed()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }
        
        auto end = std::chrono::high_resolution_clock::now();

        interation_step(start, end, state);
    }
    repetition_step();
}

// /* ============================================================ *
//  * ================== BENCHMARK REGISTRATION ================== *
//  * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->Arg(BUFFER_SIZE)
->UseManualTime();