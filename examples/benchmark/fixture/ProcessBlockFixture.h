#ifndef ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
#define ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H

#include <benchmark/benchmark.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace anira::benchmark {

/**
 * @brief Benchmark fixture for measuring the inference of one model in a real-time audio setting
 *
 * The ProcessBlockFixture is a Google Benchmark fixture over the anira.hpp handler: a benchmark
 * body creates the Context, the Pipeline and the InferenceHandler (m_context,
 * m_inference_handler), prepares it with the Hard contract of the swept buffer size, selects
 * the plan to measure (PlanReport::find) with InferenceHandler::set_plan, and calls
 * initialize_repetition. Each iteration then pushes one buffer of random samples through
 * InferenceHandler::process (process_buffer) and waits until the inference of that buffer has
 * been collected (buffer_processed), which is the time the iteration reports.
 *
 * It reports, per iteration, the time and the real-time factor (the time over the buffer's
 * duration), marking an underrun when the factor exceeds 1, and after the final repetition a
 * summary line per benchmark instance (the mean and the largest factor, the underruns).
 *
 * The fixture is part of the examples (examples/benchmark/fixture, the static target
 * anira_benchmark_fixture), not of the library: copy it into a project of your own to benchmark
 * your models, or link the target from a build of this repository.
 *
 * @note Use it with the BENCHMARK_DEFINE_F / BENCHMARK_REGISTER_F macros of Google Benchmark;
 *       the handler and the buffer are static members, shared by the instances of one benchmark.
 */
class ProcessBlockFixture : public ::benchmark::Fixture {
public:
    ProcessBlockFixture();
    ~ProcessBlockFixture() override;

    /**
     * @brief Records how many samples the output ring holds before the iteration's buffer
     */
    void initialize_iteration();

    /**
     * @brief Starts a repetition over the prepared handler: sizes the input buffer (the anchor
     * input's channels times the buffer size, as one planar float32 tensor), and when the model,
     * the contract, the plan or the plan's backend (its engine and provider) changed, resets the
     * statistics and prints the header of the run
     *
     * @param model The model config the handler runs (for the model's name)
     * @param hard The Hard contract it was prepared with (the buffer size and the sample rate:
     *        host_block_max and rate)
     * @param plan The plan of the handler's report that is measured (set with set_plan)
     * @param sleep_after_repetition Whether TearDown sleeps as long as the repetition took, so
     *        the system cools down between repetitions
     */
    void initialize_repetition(const anira::ModelConfig& model,
                               const anira::Hard& hard,
                               uint32_t plan,
                               bool sleep_after_repetition = true);

    /**
     * @brief Whether the inference of the iteration's buffer has been collected: the output ring
     * holds at least what it held before the iteration
     */
    bool buffer_processed();

    /**
     * @brief Fills the input buffer with random samples in [-1, 1)
     */
    void push_random_samples_in_buffer();

    /**
     * @brief One block through the handler, in place: InferenceHandler::process over the planar
     * tensor of the input buffer
     *
     * @return The status of the Hard entry
     */
    anira_status process_buffer() noexcept;

    /// The buffer size of the running benchmark (state.range(0)).
    int get_buffer_size();
    /// The repetition of the running benchmark.
    int get_repetition();

    /**
     * @brief Records the time of one iteration, prints it with its real-time factor and adds it
     * to the statistics of the benchmark instance
     */
    void interation_step(const std::chrono::steady_clock::time_point& start,
                         const std::chrono::steady_clock::time_point& end,
                         ::benchmark::State& state);

    /**
     * @brief Ends a repetition; after the last one (total_repetitions, 0 for none) prints the
     * summary of the benchmark instance
     */
    void repetition_step(int total_repetitions = 0);

    /// The context of the benchmark's handler; created by the benchmark body.
    inline static std::unique_ptr<anira::Context> m_context = nullptr;
    /// The handler being benchmarked; created by the benchmark body, destroyed in TearDown.
    inline static std::unique_ptr<anira::InferenceHandler> m_inference_handler = nullptr;

private:
    int m_buffer_size = 0;                 ///< Current buffer size being benchmarked
    int m_repetition = 0;                  ///< Current repetition number
    bool m_sleep_after_repetition = true;  ///< Whether to sleep after repetitions for system
                                           ///< cooldown
    int m_iteration = 0;                   ///< Current iteration number within a repetition
    std::chrono::duration<double, std::milli> m_runtime_last_repetition =
        std::chrono::duration<double, std::milli>(0);  ///< Runtime of the last repetition
    size_t m_prev_num_received_samples = 0;  ///< Samples the output ring held at the start of
                                             ///< the current iteration
    double m_rtf_sum = 0.0;                  ///< Sum of the real-time factors of all iterations
                                             ///< of the current benchmark instance
    double m_rtf_max = 0.0;                  ///< Largest real-time factor of all iterations of
                                             ///< the current benchmark instance
    int m_num_underruns = 0;                 ///< Iterations of the current benchmark instance
                                             ///< that took longer than the host buffer period
    int m_num_iterations_total = 0;          ///< Iterations of the current benchmark instance
                                             ///< across all repetitions
    std::string m_benchmark_name;            ///< Name of the currently running benchmark
    std::string m_model_name;                ///< File name of the model being benchmarked
    std::string m_engine_name;               ///< Name of the engine of the measured plan
    std::string m_run_key;                   ///< The model, contract, plan and backend of the
                                             ///< last run
    double m_sample_rate = 0.0;              ///< Host sample rate of the current run
    // The input buffer: one plane per channel, and the planar tensor over them.
    std::vector<std::vector<float>> m_planes;
    std::vector<float*> m_plane_pointers;
    anira::Tensor m_block{};

    // NOLINTNEXTLINE(readability-identifier-naming) - overrides ::benchmark::Fixture::SetUp
    void SetUp(const ::benchmark::State& state) override;
    // NOLINTNEXTLINE(readability-identifier-naming) - overrides ::benchmark::Fixture::TearDown
    void TearDown(const ::benchmark::State& state) override;
};

/// One sample in [-1, 1).
inline float random_sample() {
    return -1.F + static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX) / 2.F);
}

/// The nearest-rank percentile of the values (an input value).
inline double calculate_percentile(const std::vector<double>& v, double percentile) {
    if (v.empty()) { throw std::invalid_argument("Input vector is empty."); }
    std::vector<double> sorted_data = v;
    std::ranges::sort(sorted_data);
    // Nearest rank: truncating the index is what makes the result an input value.
    const size_t n = sorted_data.size();
    const auto percentile_index = static_cast<size_t>(percentile * static_cast<double>(n - 1));
    return sorted_data[percentile_index];
}

inline double calculate_min(const std::vector<double>& v) {
    return *(std::ranges::min_element(v));
}

inline double calculate_max(const std::vector<double>& v) {
    return *(std::ranges::max_element(v));
}

}  // namespace anira::benchmark

#endif  // ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
