#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>
#include <functional>

namespace anira {

struct ANIRA_API HostAudioConfig {
    HostAudioConfig() = default;
    HostAudioConfig(size_t max_host_input_size, double host_input_sample_rate,
                    size_t max_host_output_size, double host_output_sample_rate,
                    size_t input_tensor_index = 0, size_t output_tensor_index = 0)
        : m_max_host_input_size(max_host_input_size),
          m_host_input_sample_rate(host_input_sample_rate),
          m_max_host_output_size(max_host_output_size),
          m_host_output_sample_rate(host_output_sample_rate),
          m_input_tensor_index(input_tensor_index),
          m_output_tensor_index(output_tensor_index) {
            m_max_host_buffer_size = m_max_host_input_size; // Default to max input size
            m_host_sample_rate = m_host_input_sample_rate; // Default to input sample rate
            if (m_max_host_output_size > m_max_host_input_size) {
                m_max_host_buffer_size = m_max_host_output_size; // Use max output size if larger
                m_host_sample_rate = m_host_output_sample_rate; // Use output sample rate if larger
            }
            if (std::abs((double)m_max_host_input_size / m_host_input_sample_rate -
                         (double)m_max_host_output_size / m_host_output_sample_rate) > 1e-6) {
                throw std::invalid_argument("max_host_input_size / host_input_sample_rate must equal max_host_output_size / host_output_sample_rate");
            }
          }

    HostAudioConfig(size_t host_buffer_size, double host_sample_rate)
        : HostAudioConfig(host_buffer_size, host_sample_rate,
                        host_buffer_size, host_sample_rate) {}
          
    size_t m_max_host_input_size = 0; // Maximum size of the input buffer
    double m_host_input_sample_rate = 0.0; // Sample rate of the input
    size_t m_max_host_output_size = 0; // Maximum size of the output buffer
    double m_host_output_sample_rate = 0.0; // Sample rate of the output
    size_t m_input_tensor_index = 0; // Index of the input tensor in the session element's send buffer
    size_t m_output_tensor_index = 0; // Index of the output tensor in the session element's receive buffer
    size_t m_max_host_buffer_size = 0; // Maximum size of the host buffer
    double m_host_sample_rate = 0.0; // Sample rate of the host buffers

    bool operator==(const HostAudioConfig& other) const {
        return m_max_host_output_size == other.m_max_host_output_size
            && std::abs(m_host_output_sample_rate - other.m_host_output_sample_rate) < 1e-6
            && m_max_host_input_size == other.m_max_host_input_size
            && std::abs(m_host_input_sample_rate - other.m_host_input_sample_rate) < 1e-6
            && m_input_tensor_index == other.m_input_tensor_index
            && m_output_tensor_index == other.m_output_tensor_index
            && m_max_host_buffer_size == other.m_max_host_buffer_size
            && std::abs(m_host_sample_rate - other.m_host_sample_rate) < 1e-6;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H