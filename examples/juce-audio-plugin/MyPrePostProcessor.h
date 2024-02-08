#ifndef NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H
#define NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H

#include <anira/PrePostProcessor.h>
#include "Configs.h"

class MyPrePostProcessor : public anira::PrePostProcessor
{
public:
#if MODEL_TO_USE == 1
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < config.m_batch_size; batch++) {
            size_t baseIdx = batch * config.m_model_input_size_backend;
            popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size, baseIdx);
        }
    };
#elif MODEL_TO_USE == 2
    virtual void preProcess(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend currentInferenceBackend) override {
        popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size);
    };
#elif MODEL_TO_USE == 3
    // The third model uses the default preProcess method
#endif // MODEL_TO_USE
};

#endif // NN_INFERENCE_TEMPLATE_MYPREPOSTPROCESSOR_H