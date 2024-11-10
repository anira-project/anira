#include <anira/scheduler/InferenceManager.h>

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor, const ContextConfig& context_config) :
    m_context(Context::get_instance(context_config)),
    m_session(m_context->create_session(pp_processor, inference_config, custom_processor)),
    m_inference_config(inference_config)
{
}

InferenceManager::~InferenceManager() {
    m_context->release_session(m_session);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    m_session->m_currentBackend.store(new_inference_backend, std::memory_order_relaxed);
}

InferenceBackend InferenceManager::get_backend() const {
    return m_session->m_currentBackend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(HostAudioConfig new_config) {
    m_spec = new_config;

    m_context->prepare(m_session, m_spec);

    m_inference_counter.store(0);

    m_init_samples = calculate_latency();
    for (size_t i = 0; i < m_inference_config.m_num_audio_channels[Output]; ++i) {
        for (size_t j = 0; j < m_init_samples; ++j) {
            m_session->m_receive_buffer.push_sample(i, 0.f);
        }
    }
}

void InferenceManager::process(const float* const* input_data, float* const* output_data, size_t num_samples) {
    process_input(input_data, num_samples);

    m_context->new_data_submitted(m_session);
    double time_in_sec = static_cast<double>(num_samples) / m_spec.m_host_sample_rate;
    m_context->new_data_request(m_session, time_in_sec);

    process_output(output_data, num_samples);
}

void InferenceManager::process_input(const float* const* input_data, size_t num_samples) {
    for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[Input]; ++channel) {
        for (size_t sample = 0; sample < num_samples; ++sample) {
            m_session->m_send_buffer.push_sample(channel, input_data[channel][sample]);
        }
    }
}

void InferenceManager::process_output(float* const* output_data, size_t num_samples) {    
    while (m_inference_counter.load() > 0) {
        if (m_session->m_receive_buffer.get_available_samples(0) >= 2 * (size_t) num_samples) {
            for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[Output]; ++channel) {
                for (size_t sample = 0; sample < num_samples; ++sample) {
                    m_session->m_receive_buffer.pop_sample(channel);
                }
            }
            m_inference_counter.fetch_sub(1);
#ifndef BELA
            std::cout << "[WARNING] Catch up samples in session: " << m_session->m_session_id << "!" << std::endl;
#else
            printf("[WARNING] Catch up samples in session: %d!\n", m_session->m_session_id);
#endif
        }
        else {
            break;
        }
    }
    if (m_session->m_receive_buffer.get_available_samples(0) >= (size_t) num_samples) {
        for (size_t channel = 0; channel < m_inference_config.m_num_audio_channels[Output]; ++channel) {
            for (size_t sample = 0; sample < num_samples; ++sample) {
                output_data[channel][sample] = m_session->m_receive_buffer.pop_sample(channel);
            }
        }
    } else {
        clear_data(output_data, num_samples, m_inference_config.m_num_audio_channels[Output]);
        m_inference_counter.fetch_add(1);
#ifndef BELA
            std::cout << "[WARNING] Missing samples in session: " << m_session->m_session_id << "!" << std::endl;
#else
            printf("[WARNING] Missing samples in session: %d!\n", m_session->m_session_id);
#endif
    }
}

void InferenceManager::clear_data(float* const* data, size_t num_samples, size_t num_channels) {
    for (size_t channel = 0; channel < num_channels; ++channel) {
        for (size_t sample = 0; sample < num_samples; ++sample) {
            data[channel][sample] = 0.f;
        }
    }
}

int InferenceManager::get_latency() const {
    return m_init_samples;
}

const Context& InferenceManager::get_context() const {
    return *m_context;
}

size_t InferenceManager::get_num_received_samples() const {
    m_context->new_data_request(m_session, 0); // TODO: Check if process_output call is better here
    return m_session->m_receive_buffer.get_available_samples(0);
}

int InferenceManager::get_missing_blocks() const {
    return m_inference_counter.load();
}

int InferenceManager::get_session_id() const {
    return m_session->m_session_id;
}

void InferenceManager::exec_inference() const {
    m_context->exec_inference();
}

int InferenceManager::calculate_latency() {
    // First calculate some universal values
    int num_output_samples = m_inference_config.m_output_sizes[m_inference_config.m_index_audio_data[Output]] / m_inference_config.m_num_audio_channels[Output];
    float host_buffer_time = (float) m_spec.m_host_buffer_size * 1000.f / (float) m_spec.m_host_sample_rate;
#ifdef USE_CONTROLLED_BLOCKING
    float wait_time = m_inference_config.m_wait_in_process_block * host_buffer_time;
#else
    float wait_time = 0.f;
#endif

    // Then caclulate the different parts of the latency
    int buffer_adaptation = calculate_buffer_adaptation(m_spec.m_host_buffer_size, num_output_samples);

    int max_possible_inferences = max_num_inferences(m_spec.m_host_buffer_size, num_output_samples);
    float total_inference_time_after_wait = (max_possible_inferences * m_inference_config.m_max_inference_time) - wait_time;
    int num_buffers_for_max_inferences = std::ceil(total_inference_time_after_wait / host_buffer_time);
    int inference_caused_latency = num_buffers_for_max_inferences * m_spec.m_host_buffer_size;

    int model_caused_latency = m_inference_config.m_internal_latency;

    // Add it all together
    return buffer_adaptation + inference_caused_latency + model_caused_latency;
}


int InferenceManager::calculate_buffer_adaptation(int host_buffer_size, int num_output_samples) {
    int res = 0;
    for (int i = host_buffer_size; i < leat_common_multiple(host_buffer_size, num_output_samples) ; i+=host_buffer_size) {
        res = std::max<int>(res, i%num_output_samples);
    }
    return res;
}

int InferenceManager::max_num_inferences(int host_buffer_size, int num_output_samples) {
    float samples_in_buffer = host_buffer_size;
    int res = (int) (samples_in_buffer / (float) num_output_samples);
    res = std::max<int>(res, 1);
    int num_inferences = 0;
    for (int i = host_buffer_size; i < leat_common_multiple(host_buffer_size, num_output_samples) ; i+=host_buffer_size) {
        num_inferences = (int) (samples_in_buffer / (float) num_output_samples);
        res = std::max<int>(res, num_inferences);
        samples_in_buffer += host_buffer_size - num_inferences * num_output_samples;
    }
    return res;
}

int InferenceManager::greatest_common_divisor(int a, int b) {
    if (b == 0) {
        return a;
    } else {
        return greatest_common_divisor(b, a % b);
    }
}

int InferenceManager::leat_common_multiple(int a, int b) {
    return a * b / greatest_common_divisor(a, b);
}

} // namespace anira