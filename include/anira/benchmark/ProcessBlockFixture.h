#ifndef ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
#define ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H

#include <benchmark/benchmark.h>
#include "../anira.h"
#include "helperFunctions.h"

namespace anira {
namespace benchmark {

class ProcessBlockFixture : public ::benchmark::Fixture {
public:

    ProcessBlockFixture() {
        bufferSize = std::make_unique<int>(0);
        repetition = std::make_unique<int>(0);
    }
    ~ProcessBlockFixture() {
        bufferSize.reset(); // buffersize and repetetion don't need to be reset when the plugin is reset
        repetition.reset();
    }

    void constructInferenceHandler(PrePostProcessor& prePostProcessor, InferenceConfig& config) {
        inferenceHandler = std::make_unique<InferenceHandler>(prePostProcessor, config);
    }

    void prepareInferenceHandler(anira::HostAudioConfig hostAudioConfig) {
        inferenceHandler->prepare(hostAudioConfig);
    }

    void pushSamplesInBuffer(anira::HostAudioConfig hostAudioConfig) {
        for (size_t channel = 0; channel < hostAudioConfig.hostChannels; channel++) {
            for (size_t sample = 0; sample < hostAudioConfig.hostBufferSize; sample++) {
                buffer->setSample(channel, sample, randomSample());
            }
        }
    }

    void setInferenceBackend(anira::InferenceBackend inferenceBackend) {
        inferenceHandler->setInferenceBackend(inferenceBackend);
    }

    bool isInitializing() {
        return inferenceHandler->getInferenceManager().isInitializing();
    }

    int getNumReceivedSamples() {
        return inferenceHandler->getInferenceManager().getNumReceivedSamples();
    }

    int getBufferSize() {
        return *bufferSize;
    }

    float** getArrayOfWritePointers() {
        return buffer->getArrayOfWritePointers();
    }

    int getRepetition() {
        return *repetition;
    }

    void repetitionStep() {
        *repetition += 1;
    }

    inline static std::unique_ptr<anira::InferenceHandler> inferenceHandler = nullptr;

private:
    inline static std::unique_ptr<int> bufferSize = nullptr;
    inline static std::unique_ptr<anira::AudioBuffer<float>> buffer = nullptr;
    inline static std::unique_ptr<int> repetition = nullptr;

    void SetUp(const ::benchmark::State& state) {
        buffer = std::make_unique<AudioBuffer<float>>(1, *bufferSize);
        if (*bufferSize != (int) state.range(0)) {
            *bufferSize = (int) state.range(0);
            std::cout << "\n------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Sample Rate 44100 Hz | Buffer Size " << *bufferSize << " = " << (float) * bufferSize * 1000.f/44100.f << " ms" << std::endl;
            std::cout << "------------------------------------------------------------------------------------------------\n" << std::endl;
            buffer->initialize(1, (int) *bufferSize);
            repetition.reset(new int(0));
        }
    }
    void TearDown(const ::benchmark::State& state) {
        buffer.reset();
        inferenceHandler.reset();
    }
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H