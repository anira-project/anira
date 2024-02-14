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
    m_init = m_inferenceHandler->getInferenceManager().isInitializing();
    m_prev_num_received_samples = m_inferenceHandler->getInferenceManager().getNumReceivedSamples();
}

void ProcessBlockFixture::initializeRepetition(const InferenceConfig& inferenceConfig, const HostAudioConfig& hostAudioConfig, const InferenceBackend& inferenceBackend) {
    m_iteration = 0;
    if (m_inferenceBackend != inferenceBackend || m_inferenceConfig != &inferenceConfig || m_hostAudioConfig != hostAudioConfig) {
        m_repetition = 0;
        if (m_inferenceConfig != &inferenceConfig) {
            m_inferenceConfig = &inferenceConfig;
            std::string path;
            if (m_inferenceBackend == anira::LIBTORCH) {
                path = m_inferenceConfig->m_model_path_torch;
            } else if (m_inferenceBackend == anira::ONNX) {
                path = m_inferenceConfig->m_model_path_onnx;
            } else if (m_inferenceBackend == anira::TFLITE) {
                path = m_inferenceConfig->m_model_path_tflite;
            } else {
                path = "unknown_model_path";
            }
            // find the last of any instance of / or \ and take the substring from there to the end
            m_model_name = path.substr(path.find_last_of("/\\") + 1);
        }
        if (m_inferenceBackend != inferenceBackend) {
            m_inferenceBackend = inferenceBackend;
            std::string path;
            if (m_inferenceBackend == anira::LIBTORCH) {
                m_inference_backend_name = "libtorch";
                path = m_inferenceConfig->m_model_path_torch;
            } else if (m_inferenceBackend == anira::ONNX) {
                m_inference_backend_name = "onnx";
                path = m_inferenceConfig->m_model_path_onnx;
            } else if (m_inferenceBackend == anira::TFLITE) {
                m_inference_backend_name = "tflite";
                path = m_inferenceConfig->m_model_path_tflite;
            } else {
                m_inference_backend_name = "unknown_backend";
                path = "unknown_model_path";
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
    if (m_init) {
        // if we have init the process output does not get called so, we never pop samples out
        // when asking for the number of received samples, we dont wait since the time bufferInSec time we give to the request is zero 
        return m_inferenceHandler->getInferenceManager().getNumReceivedSamples() >= m_prev_num_received_samples + m_bufferSize;
    }
    else {
        // when init is finished we allready anticipate that we take out samples from the buffer and just wait for the buffer to be filled again
        // therefore it makes no difference if the buffer gets filled while waiting for the semaphore or in this while loop
        // TODO: A problem could occur is when init is set to false at start and the wait_for_semaphore time is too short so no samples are returned 
        // At the moment this is not possible since init is allways set to true at the start, but this shall be changed in the future, so we can do realreal time processing
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
}

} // namespace benchmark
} // namespace anira