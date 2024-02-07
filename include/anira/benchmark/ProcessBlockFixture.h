#ifndef ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
#define ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H

#include <benchmark/benchmark.h>
#include "../anira.h"
#include "helperFunctions.h"

namespace anira {
namespace benchmark {

class ProcessBlockFixture : public ::benchmark::Fixture {
public:

    ProcessBlockFixture();
    ~ProcessBlockFixture();

    void initializeIteration();
    bool bufferHasBeenProcessed();
    void pushRandomSamplesInBuffer(anira::HostAudioConfig hostAudioConfig);
    int getBufferSize();
    int getRepetition();
    void interationStep(std::chrono::_V2::system_clock::time_point start, std::chrono::_V2::system_clock::time_point end, int& iteration, ::benchmark::State& state);
    void repetitionStep();

    inline static std::unique_ptr<anira::InferenceHandler> m_inferenceHandler = nullptr;
    inline static std::unique_ptr<anira::AudioBuffer<float>> m_buffer = nullptr;

private:
    inline static std::unique_ptr<int> m_bufferSize = nullptr;
    inline static std::unique_ptr<int> m_repetition = nullptr;
    bool m_init = false;
    int m_prev_num_received_samples = 0;

    void SetUp(const ::benchmark::State& state);
    void TearDown(const ::benchmark::State& state);
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H