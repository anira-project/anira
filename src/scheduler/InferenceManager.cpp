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

void InferenceManager::prepare(HostConfig new_config, std::vector<long> custom_latency) {
    m_host_config = new_config;

    m_context->prepare_session(m_session, m_host_config, custom_latency);

    m_missing_samples.clear();
    m_missing_samples.resize(m_inference_config.get_tensor_output_shape().size(), 0);
}

size_t* InferenceManager::process(const float* const* const* input_data, size_t* num_input_samples, float* const* const* output_data, size_t* num_output_samples) {
    process_input(input_data, num_input_samples);

    m_context->new_data_submitted(m_session);
    double time_in_sec = static_cast<double>(num_input_samples[m_host_config.m_tensor_index]) / m_host_config.m_sample_rate;
    m_context->new_data_request(m_session, time_in_sec);

    return process_output(output_data, num_output_samples);
}

void InferenceManager::push_data(const float* const* const* input_data, size_t* num_input_samples) {
    process_input(input_data, num_input_samples);
    m_context->new_data_submitted(m_session);
}

size_t* InferenceManager::pop_data(float* const* const* output_data, size_t* num_output_samples) {
    m_context->new_data_request(m_session, 0.0);
    return process_output(output_data, num_output_samples);
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

size_t* InferenceManager::process_output(float* const* const* output_data, size_t* num_samples) {
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            int missing_samples_before = m_missing_samples[i];
            while (m_missing_samples[i]) {
                if (m_session->m_receive_buffer[i].get_available_samples(0) > num_samples[i]) {
                    for (size_t channel = 0; channel < m_inference_config.get_postprocess_output_channels()[i]; ++channel) {
                        m_session->m_receive_buffer[i].pop_sample(channel);
                    }
                    m_missing_samples[i]--;
                } else {
                    break; // Exit the loop if not enough samples to pop
                }
            }
            if (missing_samples_before - m_missing_samples[i] > 0) {
                LOG_INFO << "[WARNING] Catch up missing samples: " << missing_samples_before - m_missing_samples[i] << " in session: " << m_session->m_session_id << " for tensor index: " << i << "!" << std::endl;
            }
        }
    }
    bool enough_samples = true;
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
        return num_samples;
    } else {
        clear_data(output_data, num_samples, m_inference_config.get_postprocess_output_channels());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                m_missing_samples[i] += num_samples[i];
                LOG_INFO << "[WARNING] Missing samples: " << m_missing_samples[i] << " in session: " << m_session->m_session_id << " for tensor index: " << i << "!" << std::endl;
            }
            num_samples[i] = 0; // Set num_samples to 0 if not enough samples are available
        }
        return num_samples; // Return the updated num_samples
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

std::vector<unsigned int> InferenceManager::get_latency() const {
    return m_session->m_latency;
}

const Context& InferenceManager::get_context() const {
    return *m_context;
}

size_t InferenceManager::get_num_received_samples(size_t tensor_index, size_t channel) const {
    m_context->new_data_request(m_session, 0.);
    return m_session->m_receive_buffer[tensor_index].get_available_samples(channel);
}

int InferenceManager::get_session_id() const {
    return m_session->m_session_id;
}

void InferenceManager::set_non_realtime(bool is_non_realtime) const {
    m_session->m_is_non_real_time = is_non_realtime;
}

void InferenceManager::reset() {
    m_context->reset_session(m_session);
    for (size_t& missing_samples : m_missing_samples) {
        missing_samples = 0; // Reset missing samples to zero
    }
}

} // namespace anira