#ifndef ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
#define ANIRA_HYBRIDNNPREPOSTPROCESSOR_H

#include "HybridNNConfig.h"
#include <anira/anira.h>

class HybridNNPrePostProcessor : public anira::PrePostProcessor
{
public:
    using anira::PrePostProcessor::PrePostProcessor;

    virtual void pre_process(std::vector<anira::RingBuffer>& input, std::vector<anira::BufferF>& output, [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        int64_t num_batches = 0;
        int64_t num_input_samples = 0;
        int64_t num_output_samples = 0;
#ifdef USE_LIBTORCH
        if (current_inference_backend == anira::InferenceBackend::LIBTORCH) {
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::LIBTORCH)[0][1];
        }
#endif
#ifdef USE_ONNXRUNTIME
        if (current_inference_backend == anira::InferenceBackend::ONNX) {
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::ONNX)[0][1];
        }
#endif
#ifdef USE_TFLITE
        if (current_inference_backend == anira::InferenceBackend::TFLITE) {
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][1];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::TFLITE)[0][1];
        }
#endif 
        else if (current_inference_backend == anira::InferenceBackend::CUSTOM) {
#if USE_LIBTORCH
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::LIBTORCH)[0][1];
#elif USE_ONNXRUNTIME
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::ONNX)[0][1];
#elif USE_TFLITE
            num_batches = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][1];
            num_output_samples = m_inference_config.get_tensor_output_shape(anira::InferenceBackend::TFLITE)[0][1];
#endif
        }

        if (
#ifdef USE_LIBTORCH
            current_inference_backend != anira::InferenceBackend::LIBTORCH &&
#endif
#ifdef USE_ONNXRUNTIME
            current_inference_backend != anira::InferenceBackend::ONNX &&
#endif
#ifdef USE_TFLITE
            current_inference_backend != anira::InferenceBackend::TFLITE &&
#endif
            current_inference_backend != anira::InferenceBackend::CUSTOM) {
            throw std::runtime_error("Invalid inference backend");
        }
            
        for (size_t batch = 0; batch < (size_t) num_batches; batch++) {
            size_t base_index = batch * (size_t) num_input_samples;
            pop_samples_from_buffer(input[0], output[0], (size_t) num_output_samples, (size_t) (num_input_samples-num_output_samples), base_index);
        }
    }
};

#endif //ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
