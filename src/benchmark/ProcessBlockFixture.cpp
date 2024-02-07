#include <anira/benchmark/ProcessBlockFixture.h>

namespace anira {
namespace benchmark {

ProcessBlockFixture::ProcessBlockFixture() {
    m_bufferSize = std::make_unique<int>(0);
    m_repetition = std::make_unique<int>(0);
}
ProcessBlockFixture::~ProcessBlockFixture() {
    m_bufferSize.reset(); // buffersize and repetetion don't need to be reset when the plugin is reset
    m_repetition.reset();
}

void ProcessBlockFixture::initializeIteration() {
    m_init = m_inferenceHandler->getInferenceManager().isInitializing();
    m_prev_num_received_samples = m_inferenceHandler->getInferenceManager().getNumReceivedSamples();
}

bool ProcessBlockFixture::bufferHasBeenProcessed() {
    if (m_init) {
        return m_inferenceHandler->getInferenceManager().getNumReceivedSamples() >= m_prev_num_received_samples + *m_bufferSize;
    }
    else {
        return m_inferenceHandler->getInferenceManager().getNumReceivedSamples() >= m_prev_num_received_samples;
    }
}

void ProcessBlockFixture::pushRandomSamplesInBuffer(anira::HostAudioConfig hostAudioConfig) {
    for (size_t channel = 0; channel < hostAudioConfig.hostChannels; channel++) {
        for (size_t sample = 0; sample < hostAudioConfig.hostBufferSize; sample++) {
            m_buffer->setSample(channel, sample, randomSample());
        }
    }
}

int ProcessBlockFixture::getBufferSize() {
    return *m_bufferSize;
}

int ProcessBlockFixture::getRepetition() {
    return *m_repetition;
}

void ProcessBlockFixture::interationStep(std::chrono::_V2::system_clock::time_point start, std::chrono::_V2::system_clock::time_point end, int& iteration, ::benchmark::State& state) {
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());

    auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    std::cout << state.name() << "/" << state.range(0) << "/iteration:" << iteration << "/repetition:" << getRepetition() << "\t\t\t" << elapsedTimeMS.count() << std::endl;
    iteration++;
}

void ProcessBlockFixture::repetitionStep() {
    *m_repetition += 1;
    std::cout << "\n------------------------------------------------------------------------------------------------\n" << std::endl;
}

void ProcessBlockFixture::SetUp(const ::benchmark::State& state) {
    if (*m_bufferSize != (int) state.range(0)) {
        *m_bufferSize = (int) state.range(0);
        std::cout << "\n------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Sample Rate 44100 Hz | Buffer Size " << *m_bufferSize << " = " << (float) * m_bufferSize * 1000.f/44100.f << " ms" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------\n" << std::endl;
        m_repetition.reset(new int(0));
    }
}

void ProcessBlockFixture::TearDown(const ::benchmark::State& state) {
    m_buffer.reset();
    m_inferenceHandler.reset();
}

} // namespace benchmark
} // namespace anira