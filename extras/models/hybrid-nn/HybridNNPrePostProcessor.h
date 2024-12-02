#ifndef ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
#define ANIRA_HYBRIDNNPREPOSTPROCESSOR_H

#include "HybridNNConfig.h"
#include <anira/anira.h>

class HybridNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void pre_process(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        int64_t num_batches = 0;
        int64_t num_input_samples = 0;
        int64_t num_output_samples = 0;
#ifdef USE_LIBTORCH
        if (current_inference_backend == anira::LIBTORCH) {
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][2];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
        }
#endif
#ifdef USE_ONNXRUNTIME
        if (current_inference_backend == anira::ONNX) {
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][2];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
        }
#endif
#ifdef USE_TFLITE
        if (current_inference_backend == anira::TFLITE) {
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][1];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
        }
#endif 
        else if (current_inference_backend == anira::CUSTOM) {
#if USE_LIBTORCH
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][2];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::LIBTORCH)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
#elif USE_ONNXRUNTIME
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][2];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::ONNX)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
#elif USE_TFLITE
            num_batches = m_inference_config.get_input_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][0];
            num_input_samples = m_inference_config.get_input_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Input]][1];
            num_output_samples = m_inference_config.get_output_shape(anira::InferenceBackend::TFLITE)[m_inference_config.m_index_audio_data[anira::IndexAudioData::Output]][1];
#endif
        }

        if (
#ifdef USE_LIBTORCH
            current_inference_backend != anira::LIBTORCH &&
#endif
#ifdef USE_ONNXRUNTIME
            current_inference_backend != anira::ONNX &&
#endif
#ifdef USE_TFLITE
            current_inference_backend != anira::TFLITE &&
#endif
            current_inference_backend != anira::CUSTOM) {
            throw std::runtime_error("Invalid inference backend");
        }
            
        for (size_t batch = 0; batch < (size_t) num_batches; batch++) {
            size_t base_index = batch * (size_t) num_input_samples;
            pop_samples_from_buffer(input, output, (size_t) num_output_samples, (size_t) (num_input_samples-num_output_samples), base_index);
        }
    }
    
    anira::InferenceConfig m_inference_config = hybridnn_config;
};

#endif //ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
