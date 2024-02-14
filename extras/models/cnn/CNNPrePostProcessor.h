#ifndef ANIRA_CNNPREPOSTPROCESSOR_H
#define ANIRA_CNNPREPOSTPROCESSOR_H

#include "CNNConfig.h"
#include <anira/anira.h>

class CNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size);
    };

private:
    anira::InferenceConfig config = cnnConfig;
};

#endif //ANIRA_CNNPREPOSTPROCESSOR_H
