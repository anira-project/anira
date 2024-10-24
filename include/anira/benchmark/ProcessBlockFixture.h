#ifndef ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
#define ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H

#include <benchmark/benchmark.h>
#include <iomanip>
#include "../anira.h"
#include "helperFunctions.h"

namespace anira {
namespace benchmark {

class ANIRA_API ProcessBlockFixture : public ::benchmark::Fixture {
public:

    ProcessBlockFixture();
    ~ProcessBlockFixture();

    void initialize_iteration();
    void initialize_repetition(const InferenceConfig& inference_config, const HostAudioConfig& host_config, const InferenceBackend& inference_backend, bool sleep_after_repetition = true);
    bool buffer_processed();
    void push_random_samples_in_buffer(anira::HostAudioConfig host_config);
    int get_buffer_size();
    int get_repetition();

#if defined(_WIN32) || defined(__APPLE__)
        void interation_step(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, ::benchmark::State& state);
#else
        void interation_step(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end, ::benchmark::State& state);
#endif

    void repetition_step();

    inline static std::unique_ptr<anira::InferenceHandler> m_inference_handler = nullptr;
    inline static std::unique_ptr<anira::AudioBuffer<float>> m_buffer = nullptr;

private:
    int m_buffer_size = 0;
    int m_repetition = 0;
    bool m_sleep_after_repetition = true;
    int m_iteration = 0;
    std::chrono::duration<double, std::milli> m_runtime_last_repetition = std::chrono::duration<double, std::milli>(0);
    int m_prev_num_received_samples = 0;
    std::string m_model_name;
    std::string m_inference_backend_name;
    InferenceBackend m_inference_backend;
    InferenceConfig m_inference_config;
    HostAudioConfig m_host_config;

    void SetUp(const ::benchmark::State& state);
    void TearDown(const ::benchmark::State& state);
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H