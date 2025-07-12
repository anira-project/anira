#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/Medium_CNNConfig.h"
#include "../../../extras/models/cnn/Small_CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"


/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 5
#define NUM_REPETITIONS 2
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
std::vector<anira::InferenceBackend> inference_backends = {
#ifdef USE_LIBTORCH    
    anira::LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    anira::ONNX,
#endif
#ifdef USE_TFLITE
    anira::TFLITE,
#endif
    anira::CUSTOM
};
std::vector<anira::InferenceConfig> inference_configs = {cnn_config, medium_cnn_config, small_cnn_config};
anira::InferenceConfig inference_config;

void adapt_cnn_config(anira::InferenceConfig& inference_config, int buffer_size, int model_size);

// define the buffer sizes, backends and model configs to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i)
        for (int j = 0; j < inference_configs.size(); ++j)
            for (int k = 0; k < inference_backends.size(); ++k)
                b->Args({buffer_sizes[i], j, k});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_CNNSIZE)(::benchmark::State& state) {

    // The buffer size return in get_buffer_size() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig host_config = {(size_t) get_buffer_size(), SAMPLE_RATE};

    inference_config = inference_configs[state.range(1)];
    adapt_cnn_config(inference_config, get_buffer_size(), state.range(1));

    anira::PrePostProcessor *my_pp_processor;

    my_pp_processor = new CNNPrePostProcessor(inference_config);

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*my_pp_processor, inference_config);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backends[state.range(2)]);

    m_buffer = std::make_unique<anira::Buffer<float>>(inference_config.get_preprocess_input_channels()[0], host_config.m_host_buffer_size);

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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_CNNSIZE)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->Apply(Arguments)
->ComputeStatistics("min", anira::calculate_min)
->ComputeStatistics("max", anira::calculate_max)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return anira::calculate_percentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();

void adapt_cnn_config(anira::InferenceConfig& inference_config, int buffer_size, int model_size) {
    int receptive_field;
    if (model_size == 0) receptive_field = 13332;
    else if (model_size == 1) receptive_field = 1332;
    else if (model_size == 2) receptive_field = 132;

    int input_size = buffer_size + receptive_field;
    int output_size = buffer_size;

#ifdef USE_LIBTORCH
        inference_config.set_tensor_input_shape({{1, 1, input_size}}, anira::LIBTORCH);
        inference_config.set_tensor_output_shape({{1, 1, output_size}}, anira::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        inference_config.set_tensor_input_shape({{1, 1, input_size}}, anira::ONNX);
        inference_config.set_tensor_output_shape({{1, 1, output_size}}, anira::ONNX);
#endif
#ifdef USE_TFLITE
        inference_config.set_tensor_input_shape({{1, input_size, 1}}, anira::TFLITE);
        inference_config.set_tensor_output_shape({{1, output_size, 1}}, anira::TFLITE);
#endif
    inference_config.set_preprocess_input_size(std::vector<size_t>{static_cast<size_t>(input_size - receptive_field)});
}