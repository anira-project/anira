#include <anira/scheduler/InferenceManager.h>

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& config, BackendBase& none_processor) :
    m_inference_thread_pool(InferenceThreadPool::get_instance(config)),
    m_session(m_inference_thread_pool->create_session(pp_processor, config, none_processor)),
    m_inference_config(config)
{
}

InferenceManager::~InferenceManager() {
    m_inference_thread_pool->release_session(m_session, m_inference_config);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    m_session.m_currentBackend.store(new_inference_backend, std::memory_order_relaxed);
}

InferenceBackend InferenceManager::get_backend() {
    return m_session.m_currentBackend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(HostAudioConfig new_config) {
    m_spec = new_config;

    m_inference_thread_pool->prepare(m_session, m_spec);

    m_inference_counter = 0;

    m_init_samples = calculate_latency();
    for (size_t i = 0; i < m_spec.m_host_channels; ++i) {
        for (size_t j = 0; j < m_init_samples; ++j) {
            m_session.m_receive_buffer.push_sample(i, 0.f);
        }
    }
}

void InferenceManager::process(float ** input_buffer, size_t input_samples) {
    process_input(input_buffer, input_samples);

    m_inference_thread_pool->new_data_submitted(m_session);
    double time_in_sec = static_cast<double>(input_samples) / m_spec.m_host_sample_rate;
    m_inference_thread_pool->new_data_request(m_session, time_in_sec);

    process_output(input_buffer, input_samples);
}

void InferenceManager::process_input(float ** input_buffer, size_t input_samples) {
    for (size_t channel = 0; channel < m_spec.m_host_channels; ++channel) {
        for (size_t sample = 0; sample < input_samples; ++sample) {
            m_session.m_send_buffer.push_sample(0, input_buffer[channel][sample]);
        }
    }
}

void InferenceManager::process_output(float ** input_buffer, size_t input_samples) {    
    while (m_inference_counter > 0) {
        if (m_session.m_receive_buffer.get_available_samples(0) >= 2 * (size_t) input_samples) {
            for (size_t channel = 0; channel < m_spec.m_host_channels; ++channel) {
                for (size_t sample = 0; sample < input_samples; ++sample) {
                    m_session.m_receive_buffer.pop_sample(channel);
                }
            }
            m_inference_counter--;
#ifndef BELA
            std::cout << "[WARNING] Catch up samples!" << std::endl;
#else
            printf("[WARNING] Catch up samples!");
#endif
        }
        else {
            break;
        }
    }
    if (m_session.m_receive_buffer.get_available_samples(0) >= (size_t) input_samples) {
        for (size_t channel = 0; channel < m_spec.m_host_channels; ++channel) {
            for (size_t sample = 0; sample < input_samples; ++sample) {
                input_buffer[channel][sample] = m_session.m_receive_buffer.pop_sample(channel);
            }
        }
    }
    else {
        clear_buffer(input_buffer, input_samples);
        m_inference_counter++;
#ifndef BELA
            std::cout << "[WARNING] Missing samples!" << std::endl;
#else
            printf("[WARNING] Missing samples!\n");
#endif
    }
}

void InferenceManager::clear_buffer(float ** input_buffer, size_t input_samples) {
    for (size_t channel = 0; channel < m_spec.m_host_channels; ++channel) {
        for (size_t sample = 0; sample < input_samples; ++sample) {
            input_buffer[channel][sample] = 0.f;
        }
    }
}

int InferenceManager::get_latency() const {
    return m_init_samples;
}

InferenceThreadPool& InferenceManager::get_inference_thread_pool() {
    return *m_inference_thread_pool;
}

size_t InferenceManager::get_num_received_samples() {
    m_inference_thread_pool->new_data_request(m_session, 0); // TODO: Check if process_output call is better here
    return m_session.m_receive_buffer.get_available_samples(0);
}

int InferenceManager::get_missing_blocks() {
    return m_inference_counter.load();
}

int InferenceManager::get_session_id() const {
    return m_session.m_session_id;
}

int InferenceManager::calculate_latency() {
    // First calculate some universal values
    int model_output_size = m_inference_config.m_new_model_output_size;
    float host_buffer_time = (float) m_spec.m_host_buffer_size * 1000.f / (float) m_spec.m_host_sample_rate;
#ifndef USE_SEMAPHORE
    if (m_inference_config.m_wait_in_process_block != 0.f) {
        std::cout << "[WARNING] Using a wait time in process block of 0 ms. Wait time is not supported without semaphores." << std::endl;
        m_inference_config.m_wait_in_process_block = 0.f;
    }
#endif
    float wait_time = m_inference_config.m_wait_in_process_block * host_buffer_time;

    // Then caclulate the different parts of the latency
    int buffer_adaptation = calculate_buffer_adaptation(m_spec.m_host_buffer_size, model_output_size);

    int max_possible_inferences = max_num_inferences(m_spec.m_host_buffer_size, model_output_size);
    float total_inference_time_after_wait = (max_possible_inferences * m_inference_config.m_max_inference_time) - wait_time;
    int num_buffers_for_max_inferences = std::ceil(total_inference_time_after_wait / host_buffer_time);
    int inference_caused_latency = num_buffers_for_max_inferences * m_spec.m_host_buffer_size;

    int model_caused_latency = m_inference_config.m_model_latency;

    // Add it all together
    return buffer_adaptation + inference_caused_latency + model_caused_latency;
}


int InferenceManager::calculate_buffer_adaptation(int host_buffer_size, int model_output_size) {
    int res = 0;
    for (int i = host_buffer_size; i < leat_common_multiple(host_buffer_size, model_output_size) ; i+=host_buffer_size) {
        res = std::max<int>(res, i%model_output_size);
    }
    return res;
}

int InferenceManager::max_num_inferences(int host_buffer_size, int model_output_size) {
    float samples_in_buffer = host_buffer_size;
    int res = (int) (samples_in_buffer / (float) model_output_size);
    res = std::max<int>(res, 1);
    int num_inferences = 0;
    for (int i = host_buffer_size; i < leat_common_multiple(host_buffer_size, model_output_size) ; i+=host_buffer_size) {
        num_inferences = (int) (samples_in_buffer / (float) model_output_size);
        res = std::max<int>(res, num_inferences);
        samples_in_buffer += host_buffer_size - num_inferences * model_output_size;
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