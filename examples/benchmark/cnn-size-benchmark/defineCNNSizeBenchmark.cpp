#include <ProcessBlockFixture.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "../../../extras/models/BypassEngine.h"
#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/model_files.h"

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 5
#define NUM_REPETITIONS 2
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
// The three sizes of the steerable-nafx CNN. The hop follows the host buffer, which the fixed
// window of a configuration file cannot, so the benchmark builds the model config in code with
// the CNN builder of extras/models per buffer size; the contract comes from the size's contract
// file. The CNN needs no stage: anira's default fill is its receptive-field window.
constexpr std::array<CnnSize, 3> k_sizes{CnnSize::Full, CnnSize::Medium, CnnSize::Small};
constexpr std::array<const char*, 3> k_contracts{k_cnn_contract_json,
                                                 k_medium_cnn_contract_json,
                                                 k_small_cnn_contract_json};

/// The engines a run benchmarks: every engine of this build, and the bypass engine
/// (BypassEngine.h, the newest samples of the window to the output) as the baseline without a
/// model.
const std::vector<anira::EngineRef>& benchmark_engines() {
    static const std::vector<anira::EngineRef> engines = [] {
        std::vector<anira::EngineRef> list;
        for (const anira::BackendId& row : anira::enabled_engines()) {
            list.push_back({.kind = static_cast<anira::EngineKind>(row.engine)});
        }
        list.push_back({.kind = ANIRA_ENGINE_CUSTOM, .id = k_bypass_engine_id});
        return list;
    }();
    return engines;
}

/// The one candidate of a run: the engine benchmarked, on the CPU path, so the handler loads
/// that engine's model alone.
anira::BackendId candidate_of(const anira::EngineRef& engine) {
    return {.struct_size = sizeof(anira::BackendId),
            .engine = static_cast<uint32_t>(engine.kind),
            .provider = ANIRA_PROVIDER_CPU,
            .engine_id = engine.id.empty() ? nullptr : engine.id.data()};
}

// define the buffer sizes, model sizes and engines to be used in the benchmark
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i) {
        for (int j = 0; j < k_sizes.size(); ++j) {
            for (int k = 0; k < benchmark_engines().size(); ++k) {
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
    // The buffer size (state.range(0) of the google benchmark, read through get_buffer_size()),
    // the model size (state.range(1)) and the engine (state.range(2)) of this run.
    const auto size = static_cast<size_t>(state.range(1));
    const anira::EngineRef engine = benchmark_engines()[static_cast<size_t>(state.range(2))];
    anira::ModelConfig model_config = cnn_model_config(get_buffer_size(), k_sizes[size]);
    model_config.add_model(k_bypass_engine_id);  // the bypass needs no model file
    anira::ContractHandle contract = anira::ContractHandle::from_file(k_contracts[size]);
    const auto block = static_cast<double>(get_buffer_size());
    contract.hard_geometry(block, block, SAMPLE_RATE);

    // Only report errors, so the log output of the engines does not pollute the benchmark
    // results.
    m_context = std::make_unique<anira::Context>(anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));
    const anira::Pipeline pipeline{anira::stage::Inference(model_config, {candidate_of(engine)})
                                       .engine(std::make_shared<BypassEngine>())};
    m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
    const anira::PlanReport report = m_inference_handler->prepare(contract);
    const uint32_t plan = ProcessBlockFixture::plan_of(report, engine.kind, engine.id);
    m_inference_handler->set_plan(plan);

    initialize_repetition(model_config, contract.hard(), plan);

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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_CNNSIZE)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITERATIONS)
    ->Repetitions(NUM_REPETITIONS)
    ->Apply(Arguments)
    ->ComputeStatistics("min", anira::benchmark::calculate_min)
    ->ComputeStatistics("max", anira::benchmark::calculate_max)
    ->ComputeStatistics("percentile",
                        [](const std::vector<double>& v) -> double {
                            return anira::benchmark::calculate_percentile(v, PERCENTILE);
                        })
    ->DisplayAggregatesOnly(false)
    ->UseManualTime();
