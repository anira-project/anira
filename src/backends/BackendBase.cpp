#include <anira/backends/BackendBase.h>

namespace anira {
BackendBase::BackendBase(InferenceConfig &config) : inferenceConfig(config) {

}

void BackendBase::returnAudio(AudioBufferF &input, AudioBufferF &output) {
    auto equalChannels = input.getNumChannels() == output.getNumChannels();
    auto equalSamples = input.getNumSamples() == output.getNumSamples();

    if (equalChannels && equalSamples) {
        for (int channel = 0; channel < input.getNumChannels(); ++channel) {
            auto writePtr = output.getWritePointer(0);
            auto readPtr = input.getReadPointer(0);

            for (int i = 0; i < output.getNumSamples(); ++i) {
                writePtr[i] = readPtr[i];
            }
        }
    }
}
}