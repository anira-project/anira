#ifndef ANIRA_CNNPREPOSTPROCESSOR_H
#define ANIRA_CNNPREPOSTPROCESSOR_H

#include "CNNConfig.h"
#include <anira/anira.h>

class CNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    using anira::PrePostProcessor::PrePostProcessor;

    virtual void pre_process(anira::RingBuffer& input, anira::BufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        pop_samples_from_buffer(input, output, m_inference_config.get_tensor_output_size()[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]], m_inference_config.get_tensor_input_size()[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]]-m_inference_config.get_tensor_output_size()[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]]);
    }
};

#endif //ANIRA_CNNPREPOSTPROCESSOR_H
