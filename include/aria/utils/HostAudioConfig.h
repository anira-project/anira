#ifndef ARIA_HOSTAUDIOCONFIG_H
#define ARIA_HOSTAUDIOCONFIG_H

#include <cstddef>

struct HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;
};

#endif //ARIA_HOSTAUDIOCONFIG_H
