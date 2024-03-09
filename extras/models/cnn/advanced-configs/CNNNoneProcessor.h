#ifndef ANIRA_CNN_NONE_PROCESSOR_H
#define ANIRA_CNN_NONE_PROCESSOR_H

#include <anira/anira.h>

class CNNNoneProcessor : public anira::BackendBase {
public:
    CNNNoneProcessor(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void processBlock(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
        auto equalChannels = input.getNumChannels() == output.getNumChannels();
        auto sampleDiff = input.getNumSamples() - output.getNumSamples();

        if (equalChannels && sampleDiff >= 0) {
            for (int channel = 0; channel < input.getNumChannels(); ++channel) {
                auto writePtr = output.getWritePointer(0);
                auto readPtr = input.getReadPointer(0);

                for (int i = 0; i < output.getNumSamples(); ++i) {
                    writePtr[i] = readPtr[i+sampleDiff];
                }
            }
        }
    }
};

#endif // ANIRA_CNN_NONE_PROCESSOR_H