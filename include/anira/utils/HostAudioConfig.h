#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>

namespace anira {

struct ANIRA_API HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;

    bool operator==(const HostAudioConfig& other) const {
        return hostChannels == other.hostChannels && hostBufferSize == other.hostBufferSize && hostSampleRate == other.hostSampleRate;
    }

    bool operator!=(const HostAudioConfig& other) const {
        return !(*this == other);
    }
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H