#ifndef ANIRA_BACKENDBASE_H
#define ANIRA_BACKENDBASE_H

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"

namespace anira {

class ANIRA_API BackendBase {
public:
    BackendBase(InferenceConfig& config);
    virtual void prepareToPlay() = 0;
    virtual void processBlock(AudioBufferF& input, AudioBufferF& output) = 0;

protected:
    static void returnAudio(AudioBufferF& input, AudioBufferF& output);

protected:
    InferenceConfig& inferenceConfig;
};

} // namespace anira

#endif //ANIRA_BACKENDBASE_H
