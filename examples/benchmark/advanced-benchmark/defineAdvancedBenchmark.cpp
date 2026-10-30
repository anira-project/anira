#include <ProcessBlockFixture.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNProcessor.h"
#include "../../../extras/models/model_files.h"
#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "ClearEngine.h"

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 5
#define NUM_REPETITIONS 2
#define PERCENTILE 0.999
#define SAMPLE_RATE 44100

std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
// The three models: the steerable-nafx CNN, GuitarLSTM and the stateful LSTM. Their shapes
// follow the host buffer (the CNN's hop, GuitarLSTM's batch count, the LSTM's chunk), which the
// fixed windows of the configuration files cannot, so the benchmark builds each model config
// in code with the builders of extras/models per buffer size; the contract comes from the
// model's contract file.
constexpr int k_num_models = 3;

/// The engines a run benchmarks: every engine of this build, and the clear engine
/// (ClearEngine.h, zeros without a model) as the baseline of anira's own path.
const std::vector<anira::EngineRef>& benchmark_engines() {
    static const std::vector<anira::EngineRef> engines = [] {
        std::vector<anira::EngineRef> list;
        for (const anira::BackendId& row : anira::enabled_engines()) {
            list.push_back({.kind = static_cast<anira::EngineKind>(row.engine)});
        }
        list.push_back({.kind = ANIRA_ENGINE_CUSTOM, .id = k_clear_engine_id});
        return list;
    }();
    return engines;
}

/// The model config of `model` at `buffer_size`, with the clear engine's entry, and its
/// contract file.
anira::ModelConfig model_config_of(int model, int buffer_size, const char*& contract_json);

/// The one candidate of a run: the engine benchmarked, on the CPU path, so the handler loads
/// that engine's model alone.
anira::BackendId candidate_of(const anira::EngineRef& engine) {
    return {.struct_size = sizeof(anira::BackendId),
            .engine = static_cast<uint32_t>(engine.kind),
            .provider = ANIRA_PROVIDER_CPU,
            .engine_id = engine.id.empty() ? nullptr : engine.id.data()};
}

// define the buffer sizes, models and engines to be used in the benchmark
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i) {
        for (int j = 0; j < k_num_models; ++j) {
            for (int k = 0; k < benchmark_engines().size(); ++k) {
                const bool is_stateful_rnn = (j == 2);
                const anira::EngineKind engine = benchmark_engines()[k].kind;
                // ONNX Runtime does not support the stateful RNN; its ExecuTorch export is fixed
                // at a 2048-sample chunk and cannot follow the varying buffer size.
                const bool skip = is_stateful_rnn && (engine == ANIRA_ENGINE_ONNXRUNTIME ||
                                                      engine == ANIRA_ENGINE_EXECUTORCH);
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
    // The buffer size (state.range(0) of the google benchmark, read through get_buffer_size()),
    // the model (state.range(1)) and the engine (state.range(2)) of this run.
    const int model = static_cast<int>(state.range(1));
    const anira::EngineRef engine = benchmark_engines()[static_cast<size_t>(state.range(2))];
    const char* contract_json = nullptr;
    const anira::ModelConfig model_config =
        model_config_of(model, get_buffer_size(), contract_json);
    anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    const auto block = static_cast<double>(get_buffer_size());
    contract.hard_geometry(block, block, SAMPLE_RATE);

    // Only report errors, so the log output of the engines does not pollute the benchmark
    // results.
    m_context = std::make_unique<anira::Context>(anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));
    anira::Pipeline pipeline;
    pipeline.add_engine(std::make_shared<ClearEngine>());
    pipeline.add_inference(model_config, {candidate_of(engine)});
    if (model == 1) {
        // GuitarLSTM's windows; the CNN and the LSTM take anira's default fill.
        pipeline.add_processor(std::make_shared<HybridNNProcessor>());
    }
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

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_ADVANCED)
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

anira::ModelConfig model_config_of(int model, int buffer_size, const char*& contract_json) {
    anira::ModelConfig model_config;
    if (model == 0) {
        // The CNN: the hop follows the buffer size, the receptive field stays.
        model_config = cnn_model_config(buffer_size);
        contract_json = k_cnn_contract_json;
    } else if (model == 1) {
        // GuitarLSTM: one window per buffer sample; the TensorFlow export is per batch count.
        model_config = hybridnn_model_config(buffer_size);
        contract_json = k_hybridnn_contract_json;
    } else {
        // The stateful LSTM: one chunk per buffer (Arguments() skips ONNX Runtime and
        // ExecuTorch, which cannot run it at every size).
        model_config = rnn_model_config(buffer_size);
        contract_json = k_rnn_contract_json;
    }
    model_config.add_model(k_clear_engine_id);  // the baseline needs no model file
    return model_config;
}
