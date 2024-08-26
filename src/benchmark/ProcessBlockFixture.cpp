#include <anira/benchmark/ProcessBlockFixture.h>

namespace anira {
namespace benchmark {

ProcessBlockFixture::ProcessBlockFixture() {
    // A new instance of ProcessBlockFixture is created for each benchmark that has been defined and registered
    m_bufferSize = 0;
    m_repetition = 0;
}

ProcessBlockFixture::~ProcessBlockFixture() {
}

void ProcessBlockFixture::initializeIteration() {
    m_prev_num_received_samples = m_inferenceHandler->getInferenceManager().getNumReceivedSamples();
}

void ProcessBlockFixture::initializeRepetition(const InferenceConfig& inferenceConfig, const HostAudioConfig& hostAudioConfig, const InferenceBackend& inferenceBackend, bool sleep_after_repetition) {
    m_sleep_after_repetition = sleep_after_repetition;
    if (m_sleep_after_repetition) {
        m_runtime_last_repetition = std::chrono::duration<double, std::milli>(0);
    }
    m_iteration = 0;

    if (m_inferenceBackend != inferenceBackend || m_inferenceConfig != inferenceConfig || m_hostAudioConfig != hostAudioConfig) {
        m_repetition = 0;
        if (m_inferenceBackend != inferenceBackend || m_inferenceConfig != inferenceConfig) {
            m_inferenceBackend = inferenceBackend;
            m_inferenceConfig = inferenceConfig;
            std::string path;
            switch (m_inferenceBackend)
            {
#ifdef USE_LIBTORCH
            case anira::LIBTORCH:
                m_inference_backend_name = "libtorch";
                path = m_inferenceConfig.m_model_path_torch;
                break;
#endif
#ifdef USE_ONNXRUNTIME
            case anira::ONNX:
                m_inference_backend_name = "onnx";
                path = m_inferenceConfig.m_model_path_onnx;
                break;
#endif
#ifdef USE_TFLITE
            case anira::TFLITE:
                m_inference_backend_name = "tflite";
                path = m_inferenceConfig.m_model_path_tflite;
                break;
#endif
            case anira::NONE:
                m_inference_backend_name = "none";
                path = "no_model";
                break;
            default:
                m_inference_backend_name = "unknown";
                path = "unknown_model_path";
                break;
            }
            // find the last of any instance of / or \ and take the substring from there to the end
            m_model_name = path.substr(path.find_last_of("/\\") + 1);
        }
        if (m_hostAudioConfig != hostAudioConfig) {
            m_hostAudioConfig = hostAudioConfig;
        }
        std::cout << "\n----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Model: " << m_model_name << " | Backend: " << m_inference_backend_name << " | Sample Rate: " << std::fixed << std::setprecision(0) << m_hostAudioConfig.hostSampleRate << " Hz | Buffer Size: " << m_hostAudioConfig.hostBufferSize << " = " << std::fixed << std::setprecision(4) << (float) m_hostAudioConfig.hostBufferSize * 1000.f/m_hostAudioConfig.hostSampleRate << " ms" << std::endl;
        std::cout << "----------------------------------------------------------------------------------------------------------------------------------------\n" << std::endl;
    }

}

bool ProcessBlockFixture::bufferHasBeenProcessed() {
    return m_inferenceHandler->getInferenceManager().getNumReceivedSamples() >= m_prev_num_received_samples;
}

void ProcessBlockFixture::pushRandomSamplesInBuffer(anira::HostAudioConfig hostAudioConfig) {
    for (size_t channel = 0; channel < hostAudioConfig.hostChannels; channel++) {
        for (size_t sample = 0; sample < hostAudioConfig.hostBufferSize; sample++) {
            m_buffer->setSample(channel, sample, randomSample());
        }
    }
}

int ProcessBlockFixture::getBufferSize() {
    return m_bufferSize;
}

#if defined(_WIN32) || defined(__APPLE__)
void ProcessBlockFixture::interationStep(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, ::benchmark::State& state) {
#else
void ProcessBlockFixture::interationStep(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end, ::benchmark::State& state) {
#endif
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());

    auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    m_runtime_last_repetition += elapsedTimeMS;

    std::cout << "SingleIteration/" << state.name() << "/" << m_model_name << "/" << m_inference_backend_name << "/" << state.range(0) << "/iteration:" << m_iteration << "/repetition:" << m_repetition << "\t\t\t" << std::fixed << std::setprecision(4) << elapsedTimeMS.count() << " ms" << std::endl;
    m_iteration++;
}

void ProcessBlockFixture::repetitionStep() {
    m_repetition += 1;
    std::cout << "\n----------------------------------------------------------------------------------------------------------------------------------------\n" << std::endl;
}

void ProcessBlockFixture::SetUp(const ::benchmark::State& state) {
    if (m_bufferSize != (int) state.range(0)) {
        m_bufferSize = (int) state.range(0);
    }
}

void ProcessBlockFixture::TearDown(const ::benchmark::State& state) {
    m_buffer.reset();
    m_inferenceHandler.reset();

    if (m_sleep_after_repetition) {
        std::this_thread::sleep_for(m_runtime_last_repetition);
    }
}

} // namespace benchmark
} // namespace anira