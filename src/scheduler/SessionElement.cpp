#include <anira/scheduler/SessionElement.h>

namespace anira {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& pp_processor, InferenceConfig& inference_config) :
    m_session_id(newSessionID),
    m_pp_processor(pp_processor),
    m_inference_config(inference_config),
    m_default_processor(m_inference_config),
    m_custom_processor(&m_default_processor)
{
}

SessionElement::ThreadSafeStruct::ThreadSafeStruct(std::vector<size_t> tensor_input_size, std::vector<size_t> tensor_output_size) {
    m_tensor_input_data.clear();
    m_tensor_output_data.clear();
    for (size_t i = 0; i < tensor_input_size.size(); ++i) {
        m_tensor_input_data.emplace_back(1, tensor_input_size[i]);
    }
    for (size_t i = 0; i < tensor_output_size.size(); ++i) {
        m_tensor_output_data.emplace_back(1, tensor_output_size[i]);
    }
}

void SessionElement::clear() {
    for (auto& buffer : m_send_buffer) {
        buffer.clear_with_positions();
    }
    for (auto& buffer : m_receive_buffer) {
        buffer.clear_with_positions();
    }
    m_time_stamps.clear();
    m_inference_queue.clear();
}

void SessionElement::prepare(const HostAudioConfig& host_config) {
    m_host_config = host_config;

    // Calculate the latency, number of structs needed and the sizes of the send and receive buffers
    m_latency.clear();
    m_latency = calculate_latency(host_config);
    m_num_structs = calculate_num_structs(host_config);
    m_send_buffer_size.clear();
    m_receive_buffer_size.clear();
    m_send_buffer_size = calculate_send_buffer_sizes(host_config);
    m_receive_buffer_size = calculate_receive_buffer_sizes(host_config);

    // Resize the send and receive buffers
    m_send_buffer.clear();
    m_receive_buffer.clear();
    m_send_buffer.resize(m_inference_config.get_tensor_input_shape().size());
    m_receive_buffer.resize(m_inference_config.get_tensor_output_shape().size());

    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_send_buffer_size[i] > 0) {
            m_send_buffer[i].initialize_with_positions(m_inference_config.get_preprocess_input_channels()[i], m_send_buffer_size[i]);
        } else {
            m_send_buffer[i].clear_with_positions();
        }
    }
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_receive_buffer_size[i] > 0) {
            m_receive_buffer[i].initialize_with_positions(m_inference_config.get_postprocess_output_channels()[i], m_receive_buffer_size[i]);
        } else {
            m_receive_buffer[i].clear_with_positions();
        }
    }

    // Push back 0.f for latency
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_latency[i] > 0) {
            for (size_t j = 0; j < m_inference_config.get_postprocess_output_channels()[i]; ++j) {
                for (size_t k = 0; k < m_latency[i] - m_inference_config.get_internal_latency()[i]; ++k) {
                    m_receive_buffer[i].push_sample(j, 0.f);
                }
            }
        }
    }
    
    // Create the thread-safe structs for the inference queue
    m_inference_queue.clear();

    std::vector<size_t> tensor_input_size = m_inference_config.get_tensor_input_size();
    std::vector<size_t> tensor_output_size = m_inference_config.get_tensor_output_size();

    for (int i = 0; i < m_num_structs; ++i) {
        m_inference_queue.emplace_back(std::make_unique<ThreadSafeStruct>(tensor_input_size, tensor_output_size));
    }

    m_time_stamps.clear();
    m_time_stamps.reserve(m_num_structs);
}

template <typename T> void SessionElement::set_processor(std::shared_ptr<T>& processor) {
#ifdef USE_LIBTORCH
    if (std::is_same<T, LibtorchProcessor>::value) {
        m_libtorch_processor = std::dynamic_pointer_cast<LibtorchProcessor>(processor);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (std::is_same<T, OnnxRuntimeProcessor>::value) {
        m_onnx_processor = std::dynamic_pointer_cast<OnnxRuntimeProcessor>(processor);
    }
#endif
#ifdef USE_TFLITE
    if (std::is_same<T, TFLiteProcessor>::value) {
        m_tflite_processor = std::dynamic_pointer_cast<TFLiteProcessor>(processor);
    }
#endif
}

size_t SessionElement::calculate_num_structs(const HostAudioConfig& host_config) const {
    // Now calculate the number of structs necessary to keep the inference queues filled
    float max_inference_time_in_samples = m_inference_config.m_max_inference_time * host_config.m_sample_rate / 1000;
    int new_samples_needed_for_inference = m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
    int max_possible_inferences = 0;
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] > 0) {
            float ratio_host_input = host_config.m_max_buffer_size / m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            float host_input_size = m_inference_config.get_preprocess_input_size()[i] * ratio_host_input;
            max_possible_inferences = std::max(max_possible_inferences, max_num_inferences(host_input_size, m_inference_config.get_preprocess_input_size()[i]));
        }
    }
    int structs_per_max_inference_time = std::ceil((float) max_inference_time_in_samples / (float) new_samples_needed_for_inference);
    // We need to multiply the number of structs per max inference time with the maximum possible inferences, because all can run in parallel
    int n_structs = (int) (max_possible_inferences + structs_per_max_inference_time * max_possible_inferences);
    return n_structs;
}

std::vector<unsigned int> SessionElement::calculate_latency(const HostAudioConfig& host_config) {
    std::vector<float> result_float;
    std::vector<unsigned int> result;
    float max_possible_inferences = 0.f;
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] > 0) {
            float ratio_host_input = host_config.m_max_buffer_size / m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            float host_input_size = m_inference_config.get_preprocess_input_size()[i] * ratio_host_input;
            max_possible_inferences = std::max(max_possible_inferences, (float) max_num_inferences(host_input_size, m_inference_config.get_preprocess_input_size()[i]));
        }
    }
    for (size_t i = 0; i < m_inference_config.get_postprocess_output_size().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] <= 0) {
            result_float.push_back(0);
        } else {
            float ratio_host_input = host_config.m_max_buffer_size / m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            float host_output_size = m_inference_config.get_postprocess_output_size()[i] * ratio_host_input;
            float ratio_input_output = (float) m_inference_config.get_postprocess_output_size()[i] / (float) m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            float sample_rate = host_config.m_sample_rate * ratio_input_output;
            float host_buffer_time = host_output_size * 1000.f / sample_rate;
#ifdef USE_CONTROLLED_BLOCKING
            float wait_time = m_inference_config.m_wait_in_process_block * host_buffer_time;
#else
            float wait_time = 0.f;
#endif
            // Calculate the different parts of the latency
            int buffer_adaptation = calculate_buffer_adaptation(host_output_size, m_inference_config.get_postprocess_output_size()[i]);
            float total_inference_time_after_wait = (max_possible_inferences * m_inference_config.m_max_inference_time) - wait_time;
            float num_buffers_for_max_inferences = std::ceil(total_inference_time_after_wait / host_buffer_time);
            int inference_caused_latency = std::ceil(num_buffers_for_max_inferences * host_output_size);
            int model_caused_latency = m_inference_config.get_internal_latency()[i];
            // Add it all together
            result_float.push_back(buffer_adaptation + inference_caused_latency);
        }
    }
    if (result_float.size() > 1) {
        float latency_ratio = 0.f;
        for (size_t i = 0; i < result_float.size(); ++i) {
            // check because otherwise we would divide by zero
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                latency_ratio = std::max<float>(latency_ratio, (float) result_float[i] / m_inference_config.get_postprocess_output_size()[i]);
            }
        }
        for (size_t i = 0; i < result_float.size(); ++i) {
            if (m_inference_config.get_postprocess_output_size()[i] > 0) {
                result_float[i] = std::ceil(latency_ratio) * m_inference_config.get_postprocess_output_size()[i];
                result.push_back(std::ceil(result_float[i] + m_inference_config.get_internal_latency()[i])); // Add the internal latency to the result
            } else {
                result.push_back(0); // If no output size, just return 0
            }
        }
    }
    else {
        result.push_back(std::ceil(result_float[0] + m_inference_config.get_internal_latency()[0])); // If only one output size, just return the calculated value
    }
    return result;
}

int SessionElement::calculate_buffer_adaptation(float host_buffer_size, int postprocess_output_size) const {
    int res = 0;
    for (float i = host_buffer_size; i < least_common_multiple(std::floor(host_buffer_size), postprocess_output_size); i+=host_buffer_size) {
        float remainder = std::fmod(i, (float)postprocess_output_size);
        res = std::max<int>(res, std::ceil(remainder));
    }
    // We do not want special handling of float buffer sizes as the user must then only pop data if he pushed enough for an int buffersize
    return res;
}

int SessionElement::max_num_inferences(float host_buffer_size, int postprocess_input_size) const {
    float samples_in_buffer = host_buffer_size;
    int res = (int) (samples_in_buffer / (float) postprocess_input_size);
    res = std::max<int>(res, 1);
    int num_inferences = 0;
    for (float i = samples_in_buffer; i < least_common_multiple(std::floor(host_buffer_size), postprocess_input_size); i+=host_buffer_size) {
        num_inferences = (int) (samples_in_buffer / (float) postprocess_input_size);
        res = std::max<int>(res, num_inferences);
        samples_in_buffer += host_buffer_size - num_inferences * postprocess_input_size;
    }
    // Here we handle the maximum number of inferences that can be done with a float buffer size
    if (std::fmod(host_buffer_size, 1.f) > 1e-6f) {
        samples_in_buffer = host_buffer_size;
        float remainder = 0.f;
        do {
            num_inferences = (int) (samples_in_buffer / (float) postprocess_input_size);
            res = std::max<int>(res, num_inferences);
            remainder = std::fmod(samples_in_buffer, 1.f);
            samples_in_buffer += host_buffer_size - num_inferences * postprocess_input_size;
        } while (remainder > std::fmod(samples_in_buffer, 1.f));
    }
    return res;
}

int SessionElement::greatest_common_divisor(int a, int b) const {
    if (b == 0) {
        return a;
    } else {
        return greatest_common_divisor(b, a % b);
    }
}

int SessionElement::least_common_multiple(int a, int b) const {
    return a * b / greatest_common_divisor(a, b);
}

std::vector<size_t> SessionElement::calculate_send_buffer_sizes(const HostAudioConfig& host_config) const {
    std::vector<size_t> send_buffer_sizes;

    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); ++i) {
        if (m_inference_config.get_preprocess_input_size()[i] > 0) {
            float ratio_host_input = host_config.m_max_buffer_size / m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            int host_input_size = std::ceil(m_inference_config.get_preprocess_input_size()[i] * ratio_host_input);
            int preprocess_input_size = m_inference_config.get_preprocess_input_size()[i];
            int buffer_adaptation = calculate_buffer_adaptation(host_input_size, preprocess_input_size);
            int past_samples_needed = std::max(static_cast<int>(m_inference_config.get_tensor_input_size()[i]/m_inference_config.get_preprocess_input_channels()[i]) - preprocess_input_size, 0);
            send_buffer_sizes.push_back(host_input_size + buffer_adaptation + past_samples_needed + host_input_size); // 2 host_input_size because of not full buffers
        } else {
            send_buffer_sizes.push_back(0);
        }
    }
    return send_buffer_sizes;
}

std::vector<size_t> SessionElement::calculate_receive_buffer_sizes(const HostAudioConfig& host_config) const {
    std::vector<size_t> receive_buffer_sizes;
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); ++i) {
        if (m_inference_config.get_postprocess_output_size()[i] > 0) {
            float ratio_host_input = host_config.m_max_buffer_size / m_inference_config.get_preprocess_input_size()[host_config.m_tensor_index];
            int host_output_size = std::ceil(m_inference_config.get_postprocess_output_size()[i] * ratio_host_input);
            int postprocess_output_size = m_inference_config.get_postprocess_output_size()[i];
            int new_samples = std::ceil(m_num_structs * postprocess_output_size);
            receive_buffer_sizes.push_back(new_samples + host_output_size); // Add host_output_size to account for the not full buffers
        } else {
            receive_buffer_sizes.push_back(0);
        }
    }
    return receive_buffer_sizes;
}

#ifdef USE_LIBTORCH
template void SessionElement::set_processor<LibtorchProcessor>(std::shared_ptr<LibtorchProcessor>& processor);
#endif
#ifdef USE_ONNXRUNTIME
template void SessionElement::set_processor<OnnxRuntimeProcessor>(std::shared_ptr<OnnxRuntimeProcessor>& processor);
#endif
#ifdef USE_TFLITE
template void SessionElement::set_processor<TFLiteProcessor>(std::shared_ptr<TFLiteProcessor>& processor);
#endif

} // namespace anira