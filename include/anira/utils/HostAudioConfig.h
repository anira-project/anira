#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>
#include <functional>
#include "../InferenceConfig.h"

namespace anira {

struct ANIRA_API HostAudioConfig {
    HostAudioConfig() = default;
    HostAudioConfig(float max_host_input_size, float host_input_sample_rate, 
                    bool allow_smaller_buffers = false, size_t input_tensor_index = 0)
        : m_buffer_size(max_host_input_size),
          m_sample_rate(host_input_sample_rate),
          m_allow_smaller_buffers(allow_smaller_buffers),
          m_tensor_index(input_tensor_index) {}

    float m_buffer_size = 0; // Maximum size of the input buffer
    float m_sample_rate = 0.0; // Sample rate of the input
    bool m_allow_smaller_buffers = false; // Whether to allow smaller buffer sizes
    size_t m_tensor_index = 0; // Index of the tensor in the session element's send buffer

    bool operator==(const HostAudioConfig& other) const {
        return std::abs(m_buffer_size - other.m_buffer_size) < 1e-6
            && std::abs(m_sample_rate - other.m_sample_rate) < 1e-6
            && m_allow_smaller_buffers == other.m_allow_smaller_buffers
            && m_tensor_index == other.m_tensor_index;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }

    float get_relative_buffer_size(const InferenceConfig& inference_config, size_t tensor_index, bool input = true) const {
        float ratio_buffer_size = m_buffer_size / inference_config.get_preprocess_input_size()[m_tensor_index];
        if (input) {
            return inference_config.get_preprocess_input_size()[tensor_index] * ratio_buffer_size;
        } else {
            return inference_config.get_postprocess_output_size()[tensor_index] * ratio_buffer_size;
        }
    }

    float get_relative_sample_rate(const InferenceConfig& inference_config, size_t tensor_index, bool input = true) const {
        float ratio_sample_rate = m_sample_rate / inference_config.get_preprocess_input_size()[m_tensor_index];
        if (input) {
            return inference_config.get_preprocess_input_size()[tensor_index] * ratio_sample_rate;
        } else {
            return inference_config.get_postprocess_output_size()[tensor_index] * ratio_sample_rate;
        }
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H