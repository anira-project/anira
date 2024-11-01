#include <anira/benchmark/ProcessBlockFixture.h>

namespace anira {
namespace benchmark {

ProcessBlockFixture::ProcessBlockFixture() {
    // A new instance of ProcessBlockFixture is created for each benchmark that has been defined and registered
    m_buffer_size = 0;
    m_repetition = 0;
}

ProcessBlockFixture::~ProcessBlockFixture() {
}

void ProcessBlockFixture::initialize_iteration() {
    m_prev_num_received_samples = m_inference_handler->get_inference_manager().get_num_received_samples();
}

void ProcessBlockFixture::initialize_repetition(const InferenceConfig& inference_config, const HostAudioConfig& host_config, const InferenceBackend& inference_backend, bool sleep_after_repetition) {
    m_sleep_after_repetition = sleep_after_repetition;
    if (m_sleep_after_repetition) {
        m_runtime_last_repetition = std::chrono::duration<double, std::milli>(0);
    }
    m_iteration = 0;

    if (m_inference_backend != inference_backend || m_inference_config != inference_config || m_host_config != host_config) {
        m_repetition = 0;
        if (m_inference_backend != inference_backend || m_inference_config != inference_config) {
            m_inference_backend = inference_backend;
            m_inference_config = inference_config;
            std::string path;
            switch (m_inference_backend)
            {
#ifdef USE_LIBTORCH
            case anira::LIBTORCH:
                m_inference_backend_name = "libtorch";
                path = m_inference_config.m_model_path_torch;
                break;
#endif
#ifdef USE_ONNXRUNTIME
            case anira::ONNX:
                m_inference_backend_name = "onnx";
                path = m_inference_config.m_model_path_onnx;
                break;
#endif
#ifdef USE_TFLITE
            case anira::TFLITE:
                m_inference_backend_name = "tflite";
                path = m_inference_config.m_model_path_tflite;
                break;
#endif
            case anira::CUSTOM:
                m_inference_backend_name = "custom";
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
        if (m_host_config != host_config) {
            m_host_config = host_config;
        }
        std::cout << "\n----------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Model: " << m_model_name << " | Backend: " << m_inference_backend_name << " | Sample Rate: " << std::fixed << std::setprecision(0) << m_host_config.m_host_sample_rate << " Hz | Buffer Size: " << m_host_config.m_host_buffer_size << " = " << std::fixed << std::setprecision(4) << (float) m_host_config.m_host_buffer_size * 1000.f/m_host_config.m_host_sample_rate << " ms" << std::endl;
        std::cout << "----------------------------------------------------------------------------------------------------------------------------------------\n" << std::endl;
    }

}

bool ProcessBlockFixture::buffer_processed() {
    return m_inference_handler->get_inference_manager().get_num_received_samples() >= m_prev_num_received_samples;
}

void ProcessBlockFixture::push_random_samples_in_buffer(anira::HostAudioConfig host_config) {
    for (size_t channel = 0; channel < host_config.m_host_channels; channel++) {
        for (size_t sample = 0; sample < host_config.m_host_buffer_size; sample++) {
            m_buffer->set_sample(channel, sample, random_sample());
        }
    }
}

int ProcessBlockFixture::get_buffer_size() {
    return m_buffer_size;
}

int ProcessBlockFixture::get_repetition() {
    return m_repetition;
}

#if defined(_WIN32) || defined(__APPLE__)
void ProcessBlockFixture::interation_step(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, ::benchmark::State& state) {
#else
void ProcessBlockFixture::interation_step(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end, ::benchmark::State& state) {
#endif
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());

    auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

    m_runtime_last_repetition += elapsed_time_ms;

    std::cout << "SingleIteration/" << state.name() << "/" << m_model_name << "/" << m_inference_backend_name << "/" << state.range(0) << "/iteration:" << m_iteration << "/repetition:" << m_repetition << "\t\t\t" << std::fixed << std::setprecision(4) << elapsed_time_ms.count() << " ms" << std::endl;
    m_iteration++;
}

void ProcessBlockFixture::repetition_step() {
    m_repetition += 1;
    std::cout << "\n----------------------------------------------------------------------------------------------------------------------------------------\n" << std::endl;
}

void ProcessBlockFixture::SetUp(const ::benchmark::State& state) {
    if (m_buffer_size != (int) state.range(0)) {
        m_buffer_size = (int) state.range(0);
    }
}

void ProcessBlockFixture::TearDown(const ::benchmark::State& state) {
    m_buffer.reset();
    m_inference_handler.reset();

    if (m_sleep_after_repetition) {
        std::this_thread::sleep_for(m_runtime_last_repetition);
    }
}

} // namespace benchmark
} // namespace anira