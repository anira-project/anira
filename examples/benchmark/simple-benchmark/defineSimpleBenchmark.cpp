#include <ProcessBlockFixture.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "../../../extras/models/hybrid-nn/HybridNNProcessor.h"
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
// loaded with the 3.x API, and the processor that chunks its input (HybridNNProcessor builds the
// model's windows). Pick another pair to benchmark another model:
//   k_cnn_model_json / k_cnn_contract_json, k_rnn_model_json / k_rnn_contract_json,
//   k_gain_model_json / k_gain_contract_json or k_stereo_gain_model_json /
//   k_stereo_gain_contract_json, each without a processor (anira's default fill)
const anira::ModelConfig my_model_config = anira::ModelConfig::from_file(k_hybridnn_model_json);
anira::ContractHandle my_contract = anira::ContractHandle::from_file(k_hybridnn_contract_json);

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {
    // The host blocks: the swept buffer size (state.range(0) of the google benchmark, read
    // through get_buffer_size()) at SAMPLE_RATE, through the contract.
    const auto block = static_cast<double>(get_buffer_size());
    my_contract.hard_host_blocks(block, block, SAMPLE_RATE);

    // Only report errors, so the log output of the engines does not pollute the benchmark
    // results.
    m_context = std::make_unique<anira::Context>(anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));
    anira::Pipeline pipeline;
    pipeline.add_inference(my_model_config);
    pipeline.add_processor(std::make_shared<HybridNNProcessor>());
    m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
    const anira::PlanReport report = m_inference_handler->prepare(my_contract);
    const uint32_t plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}).value();
    m_inference_handler->set_plan(plan);

    initialize_repetition(my_model_config, my_contract.hard(), plan);

    for (auto _ : state) {
        push_random_samples_in_buffer();

        initialize_iteration();

        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        process_buffer();

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
