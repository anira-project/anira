#include <anira/anira.h>
#include <anira/benchmark.h>
#include <anira/compat/v3_to_v2.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/model_files.h"
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
    anira::InferenceBackend::LIBTORCH,
#endif
#ifdef USE_ONNXRUNTIME
    anira::InferenceBackend::ONNX,
#endif
#ifdef USE_TFLITE
    anira::InferenceBackend::TFLITE,
#endif
#ifdef USE_LITERT
    anira::InferenceBackend::LITERT,
#endif
#ifdef USE_EXECUTORCH
    anira::InferenceBackend::EXECUTORCH,
#endif
    anira::InferenceBackend::CUSTOM};
// The three models: the steerable-nafx CNN, GuitarLSTM and the stateful LSTM. Their shapes
// follow the host buffer (the CNN's hop, GuitarLSTM's batch count, the LSTM's chunk), which the
// fixed windows of the configuration files cannot, so the benchmark builds each model config
// in code with the builders of extras/models and bridges it per buffer size; the contract
// comes from the model's contract file. inference_config outlives the handler of one run.
constexpr int k_num_models = 3;
anira::InferenceConfig inference_config;

/// Builds the model config of `model` at `buffer_size` into inference_config and returns the
/// host geometry (the buffer at SAMPLE_RATE) through the model's contract.
anira::HostConfig configure(int model, int buffer_size);

// define the buffer sizes, backends and model configs to be used in the benchmark and the backends
// to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i) {
        for (int j = 0; j < k_num_models; ++j) {
            for (int k = 0; k < inference_backends.size(); ++k) {
                const bool is_stateful_rnn = (j == 2);
                bool skip = false;
#ifdef USE_ONNXRUNTIME
                // ONNX backend does not support the stateful RNN
                skip = skip ||
                       (is_stateful_rnn && inference_backends[k] == anira::InferenceBackend::ONNX);
#endif
#ifdef USE_EXECUTORCH
                // The stateful RNN .pte is exported at a fixed 2048-sample chunk and
                // cannot follow the varying buffer size
                skip = skip || (is_stateful_rnn &&
                                inference_backends[k] == anira::InferenceBackend::EXECUTORCH);
#endif
                if (!skip) { b->Args({buffer_sizes[i], j, k}); }
            }
        }
    }
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {
    // The buffer size (state.range(0) of the google benchmark, read through get_buffer_size())
    // and the model (state.range(1)) of this run.
    anira::HostConfig host_config = configure(static_cast<int>(state.range(1)), get_buffer_size());

    anira::PrePostProcessor* my_pp_processor;

    if (state.range(1) == 0) {
        my_pp_processor = new CNNPrePostProcessor(inference_config);
    } else if (state.range(1) == 1) {
        my_pp_processor = new HybridNNPrePostProcessor(inference_config);
    } else if (state.range(1) == 2) {
        my_pp_processor = new anira::PrePostProcessor(inference_config);
    }

    ClearCustomProcessor clear_custom_processor(inference_config);

    // Only report errors, so the log output of the backends does not pollute the
    // benchmark results.
    const anira::CoreConfig core_config =
        anira::v3compat::to_core_config(anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*my_pp_processor,
                                                                    inference_config,
                                                                    clear_custom_processor,
                                                                    core_config);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backends[state.range(2)]);

    m_buffer =
        std::make_unique<anira::Buffer<float>>(inference_config.get_preprocess_input_channels()[0],
                                               host_config.m_buffer_size);

    initialize_repetition(inference_config, host_config, inference_backends[state.range(2)]);

    for (auto _ : state) {
        push_random_samples_in_buffer(host_config);

        initialize_iteration();

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        m_inference_handler->process(m_buffer->get_array_of_write_pointers(), get_buffer_size());

        while (!buffer_processed()) { std::this_thread::sleep_for(std::chrono::nanoseconds(10)); }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        interation_step(start, end, state);
    }
    repetition_step(NUM_REPETITIONS);

    delete my_pp_processor;
}

// /* ============================================================ *
//  * ================== BENCHMARK REGISTRATION ================== *
//  * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_ADVANCED)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITERATIONS)
    ->Repetitions(NUM_REPETITIONS)
    ->Apply(Arguments)
    ->ComputeStatistics("min", anira::calculate_min)
    ->ComputeStatistics("max", anira::calculate_max)
    ->ComputeStatistics("percentile",
                        [](const std::vector<double>& v) -> double {
                            return anira::calculate_percentile(v, PERCENTILE);
                        })
    ->DisplayAggregatesOnly(false)
    ->UseManualTime();

anira::HostConfig configure(int model, int buffer_size) {
    // Every engine of this build, plus ANIRA_ENGINE_NONE so the custom placeholder entry below
    // survives the candidate filter.
    std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_NONE);
    anira::ModelConfig model_config;
    const char* contract_json = nullptr;
    if (model == 0) {
        // The CNN: the hop follows the buffer size, the receptive field stays.
        model_config = cnn_model_config(buffer_size);
        contract_json = k_cnn_contract_json;
    } else if (model == 1) {
        // GuitarLSTM: one window per buffer sample; the TensorFlow export is per batch count.
        model_config = hybridnn_model_config(buffer_size);
        contract_json = k_hybridnn_contract_json;
    } else {
        // The stateful LSTM: one chunk per buffer. ONNX Runtime does not support it and the
        // ExecuTorch export runs only at its exported 2048-sample chunk, so neither is a
        // candidate (Arguments() skips those rows as well).
        model_config = rnn_model_config(buffer_size);
        contract_json = k_rnn_contract_json;
        std::erase(candidates, ANIRA_ENGINE_ONNXRUNTIME);
        std::erase(candidates, ANIRA_ENGINE_EXECUTORCH);
    }
    // The custom backend needs no model file, but the benchmark fixture resolves a model name
    // via get_model_path(CUSTOM): a placeholder entry for the 2.x custom engine.
    model_config.add_model_path("anira.v2.custom", "custom-placeholder");
    anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    inference_config = anira::v3compat::to_inference_config(model_config, contract, candidates);
    const auto block = static_cast<uint32_t>(buffer_size);
    contract.hard_geometry(block, block, SAMPLE_RATE);
    return anira::v3compat::to_host_config(contract, model_config);
}
