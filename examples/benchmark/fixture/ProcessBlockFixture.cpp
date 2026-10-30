#include "ProcessBlockFixture.h"

#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <iostream>
#include <ratio>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace anira::benchmark {

namespace {

/// The name a plan's engine prints under: the built-in engine's, a custom engine's id.
std::string engine_name(const anira_plan_info& plan) {
    switch (plan.engine) {
        case ANIRA_ENGINE_LIBTORCH: return "libtorch";
        case ANIRA_ENGINE_ONNXRUNTIME: return "onnx";
        case ANIRA_ENGINE_TFLITE: return "tflite";
        case ANIRA_ENGINE_LITERT: return "litert";
        case ANIRA_ENGINE_EXECUTORCH: return "executorch";
        case ANIRA_ENGINE_CUSTOM: return plan.engine_id != nullptr ? plan.engine_id : "custom";
        default: return "unknown";
    }
}

/// The file name of the model entry a plan runs: the first entry of the plan's engine, the
/// engine's id when the entry has no source (a custom engine that needs no model).
std::string model_name(const anira::ModelConfig& model, const anira_plan_info& plan) {
    for (uint32_t i = 0; i < model.model_count(); ++i) {
        const anira::EngineRef engine = model.model_engine(i);
        if (static_cast<uint32_t>(engine.kind) != plan.engine) { continue; }
        if (plan.engine == ANIRA_ENGINE_CUSTOM &&
            (plan.engine_id == nullptr || engine.id != plan.engine_id)) {
            continue;
        }
        const std::string path(model.model_path(i));
        if (path.empty()) { return engine_name(plan); }
        // the part after the last / or \ (a path of any platform)
        return path.substr(path.find_last_of("/\\") + 1);
    }
    return "unknown_model_path";
}

/// The extent of the Channel axis of a spec, 1 without one.
size_t channels_of(const anira::SpecView& spec) {
    for (uint32_t axis = 0; axis < spec.ndim(); ++axis) {
        if (spec.axis(axis).tag == ANIRA_AXIS_CHANNEL) {
            return static_cast<size_t>(spec.axis(axis).extent);
        }
    }
    return 1;
}

}  // namespace

ProcessBlockFixture::ProcessBlockFixture() = default;

ProcessBlockFixture::~ProcessBlockFixture() = default;

void ProcessBlockFixture::initialize_iteration() {
    m_prev_num_received_samples = 0;
    m_inference_handler->get_available_samples(0, 0, &m_prev_num_received_samples);
}

void ProcessBlockFixture::initialize_repetition(const anira::ModelConfig& model,
                                                const anira::Hard& hard,
                                                uint32_t plan,
                                                bool sleep_after_repetition) {
    m_sleep_after_repetition = sleep_after_repetition;
    if (m_sleep_after_repetition) {
        m_runtime_last_repetition = std::chrono::duration<double, std::milli>(0);
    }
    m_iteration = 0;

    // The buffer: the anchor input's channels (the first input) times one block.
    const size_t channels = model.input_count() > 0 ? channels_of(model.input_spec(0)) : 1;
    const auto samples = static_cast<size_t>(hard.block_max);
    m_planes.assign(channels, std::vector<float>(samples, 0.F));
    m_plane_pointers.clear();
    for (std::vector<float>& plane : m_planes) { m_plane_pointers.push_back(plane.data()); }
    const std::array<int64_t, 2> extents{static_cast<int64_t>(channels),
                                         static_cast<int64_t>(samples)};
    m_block =
        anira::Tensor::from_host_planar<float>(std::span<float* const>(m_plane_pointers), extents);

    const std::vector<anira_plan_info> plans = m_inference_handler->plan_report().plans();
    const anira_plan_info info = plan < plans.size() ? plans[plan] : anira_plan_info{};
    const std::string key = model.to_json() + "|" + std::to_string(hard.block_max) + "|" +
                            std::to_string(hard.rate) + "|" + std::to_string(plan);
    if (key != m_run_key) {
        m_run_key = key;
        m_repetition = 0;
        m_rtf_sum = 0.0;
        m_rtf_max = 0.0;
        m_num_underruns = 0;
        m_num_iterations_total = 0;
        m_engine_name = engine_name(info);
        m_model_name = model_name(model, info);
        m_sample_rate = hard.rate;
        std::cout << "\n---------------------------------------------------------------------------"
                     "-------------------------------------------------------------"
                  << '\n';
        std::cout << "Model: " << m_model_name << " | Backend: " << m_engine_name
                  << " | Host Sample Rate: " << std::fixed << std::setprecision(0) << hard.rate
                  << " Hz | Host Buffer Size: " << hard.block_max << " = " << std::fixed
                  << std::setprecision(4) << hard.block_max * 1000.0 / hard.rate << " ms" << '\n';
        std::cout << "-----------------------------------------------------------------------------"
                     "-----------------------------------------------------------\n"
                  << '\n';
    }
}

bool ProcessBlockFixture::buffer_processed() {
    size_t available = 0;
    m_inference_handler->get_available_samples(0, 0, &available);
    return available >= m_prev_num_received_samples;
}

void ProcessBlockFixture::push_random_samples_in_buffer() {
    for (std::vector<float>& plane : m_planes) {
        for (float& sample : plane) { sample = random_sample(); }
    }
}

anira_status ProcessBlockFixture::process_buffer() noexcept {
    return m_inference_handler->process(m_block, 0, m_block, 0);
}

int ProcessBlockFixture::get_buffer_size() {
    return m_buffer_size;
}

int ProcessBlockFixture::get_repetition() {
    return m_repetition;
}

void ProcessBlockFixture::interation_step(const std::chrono::steady_clock::time_point& start,
                                          const std::chrono::steady_clock::time_point& end,
                                          ::benchmark::State& state) {
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());

    auto elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    m_runtime_last_repetition += elapsed_time_ms;

    double const buffer_period_seconds = static_cast<double>(m_buffer_size) / m_sample_rate;
    double const rtf = elapsed_seconds.count() / buffer_period_seconds;
    m_benchmark_name = state.name();
    m_rtf_sum += rtf;
    m_rtf_max = std::max(m_rtf_max, rtf);
    m_num_iterations_total++;
    if (rtf > 1.0) { m_num_underruns++; }

    std::cout << "SingleIteration/" << state.name() << "/" << m_model_name << "/" << m_engine_name
              << "/" << state.range(0) << "/iteration:" << m_iteration
              << "/repetition:" << m_repetition << "\t\t\t" << std::fixed << std::setprecision(4)
              << elapsed_time_ms.count() << " ms | RTF: " << rtf << (rtf > 1.0 ? " [underrun]" : "")
              << '\n';
    m_iteration++;
}

void ProcessBlockFixture::repetition_step(int total_repetitions) {
    m_repetition += 1;
    std::cout << "\n-------------------------------------------------------------------------------"
                 "---------------------------------------------------------\n"
              << '\n';
    if (total_repetitions > 0 && m_repetition == total_repetitions && m_num_iterations_total > 0) {
        std::cout << "\033[31mSummary/" << m_benchmark_name << "/" << m_model_name << "/"
                  << m_engine_name << "/" << m_buffer_size << "\t\t\t" << std::fixed
                  << std::setprecision(4) << "rtf_mean: " << m_rtf_sum / m_num_iterations_total
                  << " | rtf_max: " << m_rtf_max << " | underruns: " << m_num_underruns << "/"
                  << m_num_iterations_total << "\033[0m" << '\n';
        std::cout
            << "\n-------------------------------------------------------------------------------"
               "---------------------------------------------------------\n"
            << '\n';
    }
}

void ProcessBlockFixture::SetUp(const ::benchmark::State& state) {
    if (m_buffer_size != static_cast<int>(state.range(0))) {
        m_buffer_size = static_cast<int>(state.range(0));
    }
}

void ProcessBlockFixture::TearDown(const ::benchmark::State& /*state*/) {
    m_inference_handler.reset();
    m_context.reset();

    if (m_sleep_after_repetition) { std::this_thread::sleep_for(m_runtime_last_repetition); }
}

}  // namespace anira::benchmark
