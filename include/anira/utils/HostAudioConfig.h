#ifndef ANIRA_HOSTAUDIOCONFIG_H
#define ANIRA_HOSTAUDIOCONFIG_H

#include <cstddef>

namespace anira {

struct HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;
};

} // namespace anira

#endif //ANIRA_HOSTAUDIOCONFIG_H