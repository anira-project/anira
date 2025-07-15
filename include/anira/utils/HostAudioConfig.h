#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>
#include <functional>

namespace anira {

struct ANIRA_API HostAudioConfig {
    HostAudioConfig() = default;
    HostAudioConfig(float max_host_input_size, float host_input_sample_rate,
                    size_t input_tensor_index = 0)
        : m_max_buffer_size(max_host_input_size),
          m_sample_rate(host_input_sample_rate),
          m_tensor_index(input_tensor_index) {}

    float m_max_buffer_size = 0; // Maximum size of the input buffer
    float m_sample_rate = 0.0; // Sample rate of the input
    size_t m_tensor_index = 0; // Index of the tensor in the session element's send buffer

    bool operator==(const HostAudioConfig& other) const {
        return std::abs(m_max_buffer_size - other.m_max_buffer_size) < 1e-6
            && std::abs(m_sample_rate - other.m_sample_rate) < 1e-6
            && m_tensor_index == other.m_tensor_index;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H