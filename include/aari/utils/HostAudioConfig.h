#ifndef AARI_HOSTAUDIOCONFIG_H
#define AARI_HOSTAUDIOCONFIG_H

#include <cstddef>

struct HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;
};

#endif //AARI_HOSTAUDIOCONFIG_H
