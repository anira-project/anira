#ifndef ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
#define ANIRA_HYBRIDNNPREPOSTPROCESSOR_H

#include "HybridNNConfig.h"
#include <anira/anira.h>

class HybridNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    virtual void preProcess(anira::RingBuffer& input, anira::AudioBufferF& output, [[maybe_unused]] anira::InferenceBackend currentInferenceBackend) override {
        int64_t num_batches;
        int64_t num_input_samples;
        int64_t num_output_samples;
        if (currentInferenceBackend == anira::LIBTORCH) {
            num_batches = config.m_model_input_shape_torch[0];
            num_input_samples = config.m_model_input_shape_torch[2];
            num_output_samples = config.m_model_output_shape_torch[1];
        } else if (currentInferenceBackend == anira::ONNX) {
            num_batches = config.m_model_input_shape_onnx[0];
            num_input_samples = config.m_model_input_shape_onnx[2];
            num_output_samples = config.m_model_output_shape_onnx[1];
        } else if (currentInferenceBackend == anira::TFLITE) {
            num_batches = config.m_model_input_shape_tflite[0];
            num_input_samples = config.m_model_input_shape_tflite[1];
            num_output_samples = config.m_model_output_shape_tflite[1];
        } else if (currentInferenceBackend == anira::NONE) {
#if USE_LIBTORCH
            num_batches = config.m_model_input_shape_torch[0];
            num_input_samples = config.m_model_input_shape_torch[2];
            num_output_samples = config.m_model_output_shape_torch[1];
#elif USE_ONNXRUNTIME
            num_batches = config.m_model_input_shape_onnx[0];
            num_input_samples = config.m_model_input_shape_onnx[2];
            num_output_samples = config.m_model_output_shape_onnx[1];
#elif USE_TFLITE
            num_batches = config.m_model_input_shape_tflite[0];
            num_input_samples = config.m_model_input_shape_tflite[1];
            num_output_samples = config.m_model_output_shape_tflite[1];
#endif
        } else {
            throw std::runtime_error("Invalid inference backend");
        }
            
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t baseIdx = batch * num_input_samples;
            popSamplesFromBuffer(input, output, num_output_samples, num_input_samples-num_output_samples, baseIdx);
        }
    };
    
    anira::InferenceConfig config = hybridNNConfig;
};

#endif //ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
