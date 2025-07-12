#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/Logger.h>

namespace anira {

InferenceManager::InferenceManager(PrePostProcessor& pp_processor, InferenceConfig& inference_config, BackendBase* custom_processor, const ContextConfig& context_config) :
    m_context(Context::get_instance(context_config)),
    m_session(m_context->create_session(pp_processor, inference_config, custom_processor)),
    m_inference_config(inference_config),
    m_pp_processor(pp_processor) {
}

InferenceManager::~InferenceManager() {
    m_context->release_session(m_session);
}

void InferenceManager::set_backend(InferenceBackend new_inference_backend) {
    m_session->m_current_backend.store(new_inference_backend, std::memory_order_relaxed);
}

InferenceBackend InferenceManager::get_backend() const {
    return m_session->m_current_backend.load(std::memory_order_relaxed);
}

void InferenceManager::prepare(HostAudioConfig new_config) {
    m_spec = new_config;

    m_context->prepare(m_session, m_spec);

    m_inference_counter.store(0);

    m_init_samples.clear();
    m_init_samples = calculate_latency();
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                for (size_t k = 0; k < m_init_samples[i] - m_inference_config.get_internal_latency()[i]; ++k) {
                    m_session->m_receive_buffer[i].push_sample(j, 0.f);
                }
            }
        }
    }
}

void InferenceManager::process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples) {
    process_input(input_data, num_input_samples);

    m_context->new_data_submitted(m_session);
    double time_in_sec = static_cast<double>(num_input_samples[m_spec.m_input_tensor_index]) / m_spec.m_host_input_sample_rate;
    m_context->new_data_request(m_session, time_in_sec);

    process_output(output_data, num_output_samples);
}

void InferenceManager::push_data(const float* const* const* input_data, size_t* num_input_samples) {
    process_input(input_data, num_input_samples);
    m_context->new_data_submitted(m_session);
}

void InferenceManager::pop_data(float* const* const* output_data, size_t* num_output_samples) {
    m_context->new_data_request(m_session, 0.0);
    process_output(output_data, num_output_samples);
}

void InferenceManager::process_input(const float* const* const* input_data, size_t* num_samples) {
    for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_input_shape().size(); ++tensor_index) {
        if (m_inference_config.get_preprocess_input_size()[tensor_index] > 0) {
            for (size_t channel = 0; channel < m_inference_config.get_preprocess_input_channels()[tensor_index]; ++channel) {
                for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                    m_session->m_send_buffer[tensor_index].push_sample(channel, input_data[tensor_index][channel][sample]);
                }
            }
        } else {
            for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                m_pp_processor.set_input(input_data[tensor_index][0][sample], tensor_index, sample); // Non-streamable parameters have no channel count
            }
        }
    }
}

void InferenceManager::process_output(float* const* const* output_data, size_t* num_samples) {
    bool enough_samples = true;
    while (m_inference_counter.load() > 0) {
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                if (m_session->m_receive_buffer[i].get_available_samples(0) < 2 * (size_t) num_samples[i]) {
                    enough_samples = false;
                    break;
                }
            }
        }
        if (enough_samples) {
            for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
                if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                    for (size_t channel = 0; channel < m_inference_config.get_postprocess_output_channels()[i]; ++channel) {
                        for (size_t sample = 0; sample < num_samples[i]; ++sample) {
                            m_session->m_receive_buffer[i].pop_sample(channel);
                        }
                    }
                }
                m_inference_counter.fetch_sub(1);
                LOG_INFO << "[WARNING] Catch up samples in session: " << m_session->m_session_id << "!" << std::endl;
            }
        } else {
            break; // Exit the loop if not enough samples to catch up
        }
    }
    enough_samples = true;
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            if (m_session->m_receive_buffer[i].get_available_samples(0) < (size_t) num_samples[i]) {
                enough_samples = false;
                break;
            }
        }
    }
    if (enough_samples) {
        for (size_t tensor_index = 0; tensor_index < m_inference_config.get_tensor_output_shape().size(); ++tensor_index) {
            if (m_inference_config.get_postprocess_output_size()[tensor_index] > 0) {
                for (size_t channel = 0; channel < m_inference_config.get_postprocess_output_channels()[tensor_index]; ++channel) {
                    for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                        output_data[tensor_index][channel][sample] = m_session->m_receive_buffer[tensor_index].pop_sample(channel);
                    }
                }
            } else {
                for (size_t sample = 0; sample < num_samples[tensor_index]; ++sample) {
                    output_data[tensor_index][0][sample] = m_pp_processor.get_output(tensor_index, sample); // Non-streamable parameters have no channel count
                }
            }
        }
    } else {
        clear_data(output_data, num_samples, m_inference_config.get_postprocess_output_channels());
        m_inference_counter.fetch_add(1);
        LOG_INFO << "[WARNING] Missing samples in session: " << m_session->m_session_id << "!" << std::endl;
    }
}

void InferenceManager::clear_data(float* const* const* data, size_t* num_samples, const std::vector<size_t>& num_channels) {
    for (size_t i = 0; i < num_channels.size(); ++i) {
        if (num_channels[i] <= 0) {
            for (size_t sample = 0; sample < num_samples[i]; ++sample) {
                data[i][0][sample] = 0.f; // Non-streamable parameters have no channel count
            }
        } else {
            for (size_t channel = 0; channel < num_channels[i]; ++channel) {
                for (size_t sample = 0; sample < num_samples[i]; ++sample) {
                    data[i][channel][sample] = 0.f;
                }
            }
        }
    }
}

std::vector<int> InferenceManager::get_latency() const {
    return m_init_samples;
}

const Context& InferenceManager::get_context() const {
    return *m_context;
}

size_t InferenceManager::get_num_received_samples(size_t tensor_index, size_t channel) const {
    m_context->new_data_request(m_session, 0.);
    return m_session->m_receive_buffer[tensor_index].get_available_samples(channel);
}

int InferenceManager::get_missing_blocks() const {
    return m_inference_counter.load();
}

int InferenceManager::get_session_id() const {
    return m_session->m_session_id;
}

void InferenceManager::set_non_realtime(bool is_non_realtime) const {
    m_session->m_is_non_real_time = is_non_realtime;
}

std::vector<int> InferenceManager::calculate_latency() {
    std::vector<int> result;
    int max_possible_inferences = max_num_inferences(m_spec.m_max_host_input_size, m_inference_config.get_preprocess_input_size()[m_spec.m_input_tensor_index]);
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] <= 0) {
            result.push_back(0);
        } else {
            int postprocess_output_size = m_inference_config.get_postprocess_output_size()[i];
            int host_output_size;
            float sample_rate;
            if (m_spec.m_output_tensor_index == i) {
                host_output_size = (int) m_spec.m_max_host_output_size;
                sample_rate = (float) m_spec.m_host_output_sample_rate;
            } else {
                int given_host_output_size = (int) m_spec.m_max_host_output_size;
                int given_sample_rate = (float) m_spec.m_host_output_sample_rate;
                float ratio = (float) m_inference_config.get_postprocess_output_size()[i] / (float) m_inference_config.get_postprocess_output_size()[m_spec.m_output_tensor_index];
                host_output_size = (int) std::ceil(given_host_output_size * ratio);
                sample_rate = given_sample_rate * ratio;
            }
            float host_buffer_time = (float) host_output_size * 1000.f / sample_rate;
#ifdef USE_CONTROLLED_BLOCKING
            float wait_time = m_inference_config.m_wait_in_process_block * host_buffer_time;
#else
            float wait_time = 0.f;
#endif
            // Calculate the different parts of the latency
            int buffer_adaptation = calculate_buffer_adaptation(host_output_size, postprocess_output_size);
            float total_inference_time_after_wait = (max_possible_inferences * m_inference_config.m_max_inference_time) - wait_time;
            int num_buffers_for_max_inferences = std::ceil(total_inference_time_after_wait / host_buffer_time);
            int inference_caused_latency = num_buffers_for_max_inferences * host_output_size;
            int model_caused_latency = m_inference_config.get_internal_latency()[i];
            // Add it all together
            result.push_back(buffer_adaptation + inference_caused_latency + model_caused_latency);
        }
    }
    return result;
}

int InferenceManager::calculate_buffer_adaptation(int host_buffer_size, int postprocess_output_size) {
    int res = 0;
    for (int i = host_buffer_size; i < least_common_multiple(host_buffer_size, postprocess_output_size) ; i+=host_buffer_size) {
        res = std::max<int>(res, i%postprocess_output_size);
    }
    return res;
}

int InferenceManager::max_num_inferences(int host_buffer_size, int postprocess_input_size) {
    float samples_in_buffer = host_buffer_size;
    int res = (int) (samples_in_buffer / (float) postprocess_input_size);
    res = std::max<int>(res, 1);
    int num_inferences = 0;
    for (int i = host_buffer_size; i < least_common_multiple(host_buffer_size, postprocess_input_size) ; i+=host_buffer_size) {
        num_inferences = (int) (samples_in_buffer / (float) postprocess_input_size);
        res = std::max<int>(res, num_inferences);
        samples_in_buffer += host_buffer_size - num_inferences * postprocess_input_size;
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

int InferenceManager::least_common_multiple(int a, int b) {
    return a * b / greatest_common_divisor(a, b);
}

} // namespace anira