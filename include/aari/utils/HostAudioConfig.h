#ifndef AARI_HOSTAUDIOCONFIG_H
#define AARI_HOSTAUDIOCONFIG_H

#include <cstddef>

namespace aari {

struct HostAudioConfig {
    size_t hostChannels;
    size_t hostBufferSize;
    double hostSampleRate;
};

} // namespace aari

#endif //AARI_HOSTAUDIOCONFIG_H