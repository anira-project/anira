#include <anira/anira.h>
#include <anira/benchmark.h>
#include <anira/compat/v3_to_v2.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "../../../extras/models/cnn/CNNPrePostProcessor.h"
#include "../../../extras/models/hybrid-nn/HybridNNPrePostProcessor.h"
#include "../../../extras/models/model_files.h"

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 5
#define NUM_REPETITIONS 2
#define BUFFER_SIZE 2048
#define SAMPLE_RATE 44100

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

// The model to benchmark: its model file and contract file (extras/models/model_files.h),
// loaded with the 3.x API and bridged to the 2.x runtime classes the fixture still takes.
// Pick another pair (and the matching pre/post processor) to benchmark another model:
//   k_cnn_model_json / k_cnn_contract_json with CNNPrePostProcessor
//   k_rnn_model_json / k_rnn_contract_json, k_gain_model_json / k_gain_contract_json or
//   k_stereo_gain_model_json / k_stereo_gain_contract_json with anira::PrePostProcessor
anira::ModelConfig my_model_config = anira::ModelConfig::from_file(k_hybridnn_model_json);
anira::ContractHandle my_contract = anira::ContractHandle::from_file(k_hybridnn_contract_json);
anira::InferenceConfig my_inference_config =
    anira::v3compat::to_inference_config(my_model_config,
                                         my_contract,
                                         anira::v3compat::enabled_engines());
HybridNNPrePostProcessor my_pp_processor(my_inference_config);

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {
    // The host geometry: the swept buffer size (state.range(0) of the google benchmark, read
    // through get_buffer_size()) at SAMPLE_RATE, through the contract.
    const auto block = static_cast<uint32_t>(get_buffer_size());
    my_contract.hard_geometry(block, block, SAMPLE_RATE);
    anira::HostConfig host_config = anira::v3compat::to_host_config(my_contract, my_model_config);
    anira::InferenceBackend inference_backend = anira::InferenceBackend::ONNX;

    // Only report errors, so the log output of the backends does not pollute the
    // benchmark results.
    const anira::ContextConfig context_config =
        anira::v3compat::to_context_config(anira::MachineConfig{}.log_level(ANIRA_LOG_ERROR));

    m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor,
                                                                    my_inference_config,
                                                                    context_config);
    m_inference_handler->prepare(host_config);
    m_inference_handler->set_inference_backend(inference_backend);

    m_buffer = std::make_unique<anira::Buffer<float>>(
        my_inference_config.get_preprocess_input_channels()[0],
        host_config.m_buffer_size);

    initialize_repetition(my_inference_config, host_config, inference_backend);

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
}

// /* ============================================================ *
//  * ================== BENCHMARK REGISTRATION ================== *
//  * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITERATIONS)
    ->Repetitions(NUM_REPETITIONS)
    ->Arg(BUFFER_SIZE)
    ->UseManualTime();