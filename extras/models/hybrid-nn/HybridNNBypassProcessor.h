#ifndef ANIRA_HYBRID_NN_BYPASS_PROCESSOR_H
#define ANIRA_HYBRID_NN_BYPASS_PROCESSOR_H

#include <anira/anira.h>

class HybridNNBypassProcessor : public anira::BackendBase {
public:
    HybridNNBypassProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(std::vector<anira::BufferF> &input, std::vector<anira::BufferF> &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        size_t num_batches;
        size_t num_input_samples;

#if USE_TFLITE
        num_batches = (size_t) m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][0];
        num_input_samples = (size_t) m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[0][1];
#else
        num_batches = (size_t) m_inference_config.get_tensor_input_shape()[0][0];
        num_input_samples = (size_t) m_inference_config.get_tensor_input_shape()[0][2];
#endif

        for (size_t channel = 0; channel < input[0].get_num_channels(); ++channel) {
            float* write_ptr = output[0].get_write_pointer(channel);
            const float* read_ptr = input[0].get_read_pointer(channel);

            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t base_index = batch * num_input_samples;
                write_ptr[batch] = read_ptr[num_input_samples - 1 + base_index];
            }
        }
    }
};

#endif // ANIRA_HYBRID_NN_BYPASS_PROCESSOR_H