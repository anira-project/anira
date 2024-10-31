#ifndef ANIRA_CNNPREPOSTPROCESSOR_H
#define ANIRA_CNNPREPOSTPROCESSOR_H

#include "CNNConfig.h"
#include <anira/anira.h>

class CNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void pre_process(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input, output, config.m_output_sizes[config.m_index_audio_data[1]], config.m_input_sizes[config.m_index_audio_data[0]]-config.m_output_sizes[config.m_index_audio_data[1]]);
    };

    anira::InferenceConfig config = cnn_config;
};

#endif //ANIRA_CNNPREPOSTPROCESSOR_H
