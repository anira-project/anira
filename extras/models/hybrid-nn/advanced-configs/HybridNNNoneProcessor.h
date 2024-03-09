#ifndef ANIRA_HYBRID_NN_NONE_PROCESSOR_H
#define ANIRA_HYBRID_NN_NONE_PROCESSOR_H

#include <anira/anira.h>

class HybridNNNoneProcessor : public anira::BackendBase {
public:
    HybridNNNoneProcessor(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void processBlock(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
        auto equalChannels = input.getNumChannels() == output.getNumChannels();
        auto sampleDiff = input.getNumSamples() - output.getNumSamples();

        if (equalChannels && sampleDiff >= 0) {
            for (size_t channel = 0; channel < input.getNumChannels(); ++channel) {
                auto writePtr = output.getWritePointer(channel);
                auto readPtr = input.getReadPointer(channel);

                for (size_t batch = 0; batch < inferenceConfig.m_batch_size; ++batch) {
                    size_t baseIdx = batch * inferenceConfig.m_model_input_size_backend;
                    writePtr[batch] = readPtr[inferenceConfig.m_model_input_size_backend - 1 + baseIdx];
                }
            }
        }
    }
};

#endif // ANIRA_HYBRID_NN_NONE_PROCESSOR_H