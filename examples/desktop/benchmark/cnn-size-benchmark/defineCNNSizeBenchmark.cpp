#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../../extras/desktop/models/cnn/advanced-configs/CNNAdvancedConfigs.h"
#include "../../../../extras/desktop/models/cnn/medium-cnn-advanced-config/Medium_CNNAdvancedConfigs.h"
#include "../../../../extras/desktop/models/cnn/small-cnn-advanced-config/Small_CNNAdvancedConfigs.h"
#include "../../../../extras/desktop/models/cnn/CNNPrePostProcessor.h"


/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
std::vector<anira::InferenceBackend> inference_backends = {anira::LIBTORCH, anira::ONNX, anira::TFLITE};
std::vector<AdvancedInferenceConfigs> advanced_inference_configs = {cnn_advanced_configs, medium_cnn_advanced_configs, small_cnn_advanced_configs};

// define the buffer sizes, backends and model configs to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i)
        for (int j = 0; j < advanced_inference_configs.size(); ++j)
            for (int k = 0; k < inference_backends.size(); ++k)
                b->Args({buffer_sizes[i], j, k});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in get_buffer_size() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig host_config = {1, (size_t) get_buffer_size(), SAMPLE_RATE};

    AdvancedInferenceConfigs current_advanced_inference_configs = advanced_inference_configs[state.range(1)];
    anira::InferenceConfig inference_config;

    for (auto advanced_config : current_advanced_inference_configs) {
        if (advanced_config.buffer_size == get_buffer_size()) {
            inference_config = advanced_config.config;
        }
    }

    anira::PrePostProcessor *my_pp_processor;

    my_pp_processor = new CNNPrePostProcessor();
    static_cast<CNNPrePostProcessor*>(my_pp_processor)->config = inference_config;

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*my_pp_processor, inference_config);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backends[state.range(2)]);

    m_buffer = std::make_unique<anira::AudioBuffer<float>>(host_config.m_host_channels, host_config.m_host_buffer_size);

    initialize_repetition(inference_config, host_config, inference_backends[state.range(2)]);

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

    delete my_pp_processor;
}

// /* ============================================================ *
//  * ================== BENCHMARK REGISTRATION ================== *
//  * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_ADVANCED)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->Apply(Arguments)
->ComputeStatistics("min", anira::benchmark::calculate_min)
->ComputeStatistics("max", anira::benchmark::calculate_max)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return anira::benchmark::calculate_percentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();