#ifndef ANIRA_STATELESSLSTMPREPOSTPROCESSOR_H
#define ANIRA_STATELESSLSTMPREPOSTPROCESSOR_H

#include "StatelessLstmConfig.h"
#include <anira/anira.h>

class StatelessLstmPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        for (size_t batch = 0; batch < config.m_batch_size; batch++) {
            size_t baseIdx = batch * config.m_model_input_size_backend;
            popSamplesFromBuffer(input, output, config.m_model_input_size, config.m_model_input_size_backend-config.m_model_input_size, baseIdx);
        }
    };
    
private:
    anira::InferenceConfig config = statelessRnnConfig;
};

#endif //ANIRA_STATELESSLSTMPREPOSTPROCESSOR_H
