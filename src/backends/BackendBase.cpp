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
            auto writePtr = output.getWritePointer(0);
            auto readPtr = input.getReadPointer(0);

            for (int i = 0; i < output.getNumSamples(); ++i) {
                writePtr[i] = readPtr[i];
            }
        }
    }
    else {
        output.clear();
    }
}

}