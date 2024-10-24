#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>
#include <functional>

namespace anira {

struct ANIRA_API HostAudioConfig {
    size_t m_host_channels;
    size_t m_host_buffer_size;
    double m_host_sample_rate;
    std::function<bool(int)> hostThreadSubmitTaskCallback;

    bool operator==(const HostAudioConfig& other) const {
        return m_host_channels == other.m_host_channels && m_host_buffer_size == other.m_host_buffer_size && m_host_sample_rate == other.m_host_sample_rate;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H