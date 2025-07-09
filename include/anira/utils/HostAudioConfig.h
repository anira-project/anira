#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>
#include <functional>

namespace anira {

struct ANIRA_API HostAudioConfig {
    HostAudioConfig() = default;
    HostAudioConfig(size_t host_buffer_size, double host_sample_rate) : m_host_buffer_size(host_buffer_size), m_host_sample_rate(host_sample_rate) {}
    size_t m_host_buffer_size;
    double m_host_sample_rate;

    bool operator==(const HostAudioConfig& other) const {
        return m_host_buffer_size == other.m_host_buffer_size
            && std::abs(m_host_sample_rate - other.m_host_sample_rate) < 1e-6;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H