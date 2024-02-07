#ifndef MYPREPOSTPROCESSOR_H
#define MYPREPOSTPROCESSOR_H

#include <anira/anira.h>

#include "MyConfig.h"

class MyPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        popSamplesFromBuffer(input, output, myConfig.m_model_input_size, myConfig.m_model_input_size_backend-myConfig.m_model_input_size);
    };
};

#endif // MYPREPOSTPROCESSOR_H