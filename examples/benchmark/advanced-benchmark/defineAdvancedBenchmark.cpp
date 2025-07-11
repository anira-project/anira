#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "ClearCustomProcessor.h"


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
std::vector<anira::InferenceConfig> inference_configs = {cnn_config, hybridnn_config, rnn_config};
anira::InferenceConfig inference_config;

void adapt_config(anira::InferenceConfig& inference_config, int buffer_size, int model);

// define the buffer sizes, backends and model configs to be used in the benchmark and the backends to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i)
        for (int j = 0; j < inference_configs.size(); ++j)
            for (int k = 0; k < inference_backends.size(); ++k)
                // ONNX backend does not support stateful RNN
                if (!(j == 2 && k == 1))
                    b->Args({buffer_sizes[i], j, k});
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in get_buffer_size() is populated by state.range(0) param of the google benchmark
    anira::HostAudioConfig host_config = {(size_t) get_buffer_size(), SAMPLE_RATE};

    inference_config = inference_configs[state.range(1)];

    adapt_config(inference_config, get_buffer_size(), state.range(1));

    anira::PrePostProcessor *my_pp_processor;

    if (state.range(1) == 0) {
        my_pp_processor = new CNNPrePostProcessor();
        static_cast<CNNPrePostProcessor*>(my_pp_processor)->m_inference_config = inference_config;
    } else if (state.range(1) == 1) {
        my_pp_processor = new HybridNNPrePostProcessor();
        static_cast<HybridNNPrePostProcessor*>(my_pp_processor)->m_inference_config = inference_config;
    } else if (state.range(1) == 2) {
        my_pp_processor = new anira::PrePostProcessor();
    }

    ClearCustomProcessor clear_custom_processor(inference_config);

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*my_pp_processor, inference_config, clear_custom_processor);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backends[state.range(2)]);

    m_buffer = std::make_unique<anira::Buffer<float>>(inference_config.m_num_audio_channels[anira::Input], host_config.m_host_buffer_size);

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
->ComputeStatistics("min", anira::calculate_min)
->ComputeStatistics("max", anira::calculate_max)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return anira::calculate_percentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();

void adapt_config(anira::InferenceConfig& inference_config, int buffer_size, int model) {

    if (model == 0) {
        int receptive_field = 13332;
        int input_size = buffer_size + receptive_field;
        int output_size = buffer_size;

#ifdef USE_LIBTORCH
        inference_config.set_input_shape({{1, 1, input_size}}, anira::LIBTORCH);
        inference_config.set_output_shape({{1, 1, output_size}}, anira::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        inference_config.set_input_shape({{1, 1, input_size}}, anira::ONNX);
        inference_config.set_output_shape({{1, 1, output_size}}, anira::ONNX);
#endif
#ifdef USE_TFLITE
        inference_config.set_input_shape({{1, input_size, 1}}, anira::TFLITE);
        inference_config.set_output_shape({{1, output_size, 1}}, anira::TFLITE);
#endif
        inference_config.m_input_sizes[0] = input_size;
        inference_config.m_output_sizes[0] = output_size;
    } else if (model == 1) {
#ifdef USE_LIBTORCH
        inference_config.set_input_shape({{buffer_size, 1, 150}}, anira::LIBTORCH);
        inference_config.set_output_shape({{buffer_size, 1}}, anira::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        inference_config.set_input_shape({{buffer_size, 1, 150}}, anira::ONNX);
        inference_config.set_output_shape({{buffer_size, 1}}, anira::ONNX);
#endif
#ifdef USE_TFLITE
        std::string model_data = inference_config.get_model_path(anira::TFLITE);
        size_t pos = model_data.find("256");
        if (pos != std::string::npos) {
            model_data.replace(pos, 3, std::to_string(buffer_size));
        }
        inference_config.set_model_path(model_data, anira::TFLITE);
        inference_config.set_input_shape({{buffer_size, 150, 1}}, anira::TFLITE);
        inference_config.set_output_shape({{buffer_size, 1}}, anira::TFLITE);
#endif
        inference_config.m_input_sizes[0] = buffer_size * 150;
        inference_config.m_output_sizes[0] = buffer_size;
    } else if (model == 2) {
#ifdef USE_LIBTORCH
        inference_config.set_input_shape({{buffer_size, 1, 1}}, anira::LIBTORCH);
        inference_config.set_output_shape({{buffer_size, 1, 1}}, anira::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        inference_config.set_input_shape({{buffer_size, 1, 1}}, anira::ONNX);
        inference_config.set_output_shape({{buffer_size, 1, 1}}, anira::ONNX);
#endif
#ifdef USE_TFLITE
        inference_config.set_input_shape({{1, buffer_size, 1}}, anira::TFLITE);
        inference_config.set_output_shape({{1, buffer_size, 1}}, anira::TFLITE);
#endif
        inference_config.m_input_sizes[0] = buffer_size;
        inference_config.m_output_sizes[0] = buffer_size;
    }
}