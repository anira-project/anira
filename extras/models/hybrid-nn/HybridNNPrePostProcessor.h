#ifndef ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
#define ANIRA_HYBRIDNNPREPOSTPROCESSOR_H

#include <anira/anira.h>

#include "HybridNNConfig.h"

class HybridNNPrePostProcessor : public anira::PrePostProcessor {
public:
    using anira::PrePostProcessor::PrePostProcessor;

    virtual void pre_process(
        std::vector<anira::RingBuffer>& input,
        std::vector<anira::BufferF>& output,
        [[maybe_unused]] anira::InferenceBackend current_inference_backend) override {
        int64_t num_batches = 0;
        int64_t num_input_samples = 0;
        int64_t num_output_samples = 0;

        bool channels_last = false;
        anira::InferenceBackend channels_last_backend = anira::InferenceBackend::CUSTOM;
#ifdef USE_TFLITE
        if (current_inference_backend == anira::InferenceBackend::TFLITE) {
            channels_last = true;
            channels_last_backend = anira::InferenceBackend::TFLITE;
        }
#endif
#ifdef USE_LITERT
        if (current_inference_backend == anira::InferenceBackend::LITERT) {
            channels_last = true;
            channels_last_backend = anira::InferenceBackend::LITERT;
        }
#endif
        if (channels_last) {
            num_batches = m_inference_config.get_tensor_input_shape(channels_last_backend)[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape(channels_last_backend)[0][1];
            num_output_samples =
                m_inference_config.get_tensor_output_shape(channels_last_backend)[0][1];
        } else {
            num_batches = m_inference_config.get_tensor_input_shape()[0][0];
            num_input_samples = m_inference_config.get_tensor_input_shape()[0][2];
            num_output_samples = m_inference_config.get_tensor_output_shape()[0][1];
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
#ifdef USE_LITERT
            current_inference_backend != anira::InferenceBackend::LITERT &&
#endif
            current_inference_backend != anira::InferenceBackend::CUSTOM) {
            throw std::runtime_error("Invalid inference backend");
        }

        for (size_t batch = 0; batch < (size_t)num_batches; batch++) {
            size_t base_index = batch * (size_t)num_input_samples;
            pop_samples_from_buffer(input[0],
                                    output[0],
                                    (size_t)num_output_samples,
                                    (size_t)(num_input_samples - num_output_samples),
                                    base_index);
        }
    }
};

#endif  // ANIRA_HYBRIDNNPREPOSTPROCESSOR_H
