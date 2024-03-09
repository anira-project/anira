#include <anira/backends/BackendBase.h>

namespace anira {
BackendBase::BackendBase(InferenceConfig &config) : inferenceConfig(config) {

}

void BackendBase::prepareToPlay() {

}

void BackendBase::processBlock(AudioBufferF &input, AudioBufferF &output) {
    auto equalChannels = input.getNumChannels() == output.getNumChannels();
    auto sampleDiff = input.getNumSamples() - output.getNumSamples();

    if (equalChannels && sampleDiff == 0) {
        for (int channel = 0; channel < input.getNumChannels(); ++channel) {
            auto writePtr = output.getWritePointer(channel);
            auto readPtr = input.getReadPointer(channel);

            for (size_t i = 0; i < output.getNumSamples(); ++i) {
                writePtr[i] = readPtr[i];
            }
        }
    }
    else {
        output.clear();
    }
}

}