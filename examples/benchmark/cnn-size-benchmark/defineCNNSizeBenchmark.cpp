#include <anira/anira.h>
#include <anira/benchmark.h>
#include <anira/compat/v3_to_v2.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <array>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/model_files.h"

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
// The three sizes of the steerable-nafx CNN. The hop follows the host buffer, which the fixed
// window of a configuration file cannot, so the benchmark builds the model config in code with
// the CNN builder of extras/models and bridges it per buffer size; the contract comes from the
// size's contract file. inference_config outlives the handler of one run.
constexpr std::array<CnnSize, 3> k_sizes{CnnSize::Full, CnnSize::Medium, CnnSize::Small};
constexpr std::array<const char*, 3> k_contracts{k_cnn_contract_json,
                                                 k_medium_cnn_contract_json,
                                                 k_small_cnn_contract_json};
anira::InferenceConfig inference_config;

/// Builds the CNN of `model_size` at `buffer_size` into inference_config and returns the host
/// geometry (the buffer at SAMPLE_RATE) through the size's contract.
anira::HostConfig configure(int model_size, int buffer_size);

// define the buffer sizes, backends and model configs to be used in the benchmark and the backends
// to be used
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i) {
        for (int j = 0; j < k_sizes.size(); ++j) {
            for (int k = 0; k < inference_backends.size(); ++k) {
                b->Args({buffer_sizes[i], j, k});
            }
        }
    }
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_CNNSIZE)(::benchmark::State& state) {
    // The buffer size (state.range(0) of the google benchmark, read through get_buffer_size())
    // and the model size (state.range(1)) of this run.
    anira::HostConfig host_config = configure(static_cast<int>(state.range(1)), get_buffer_size());

    anira::PrePostProcessor* my_pp_processor;

    my_pp_processor = new CNNPrePostProcessor(inference_config);

    // Only report errors, so the log output of the backends does not pollute the
    // benchmark results.
    const anira::CoreConfig core_config =
        anira::v3compat::to_core_config(anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));

    m_inference_handler =
        std::make_unique<anira::InferenceHandler>(*my_pp_processor, inference_config, core_config);
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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_CNNSIZE)
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

anira::HostConfig configure(int model_size, int buffer_size) {
    // Every engine of this build, plus ANIRA_ENGINE_NONE so the custom placeholder entry below
    // survives the candidate filter.
    std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_NONE);
    anira::ModelConfig model_config =
        cnn_model_config(buffer_size, k_sizes[static_cast<size_t>(model_size)]);
    // The custom backend needs no model file, but the benchmark fixture resolves a model name
    // via get_model_path(CUSTOM): a placeholder entry for the 2.x custom engine.
    model_config.add_model_path("anira.v2.custom", "custom-placeholder");
    anira::ContractHandle contract =
        anira::ContractHandle::from_file(k_contracts[static_cast<size_t>(model_size)]);
    inference_config = anira::v3compat::to_inference_config(model_config, contract, candidates);
    const auto block = static_cast<uint32_t>(buffer_size);
    contract.hard_geometry(block, block, SAMPLE_RATE);
    return anira::v3compat::to_host_config(contract, model_config);
}
