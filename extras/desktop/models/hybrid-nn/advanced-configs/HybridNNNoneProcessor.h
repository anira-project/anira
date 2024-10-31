#ifndef ANIRA_HYBRID_NN_NONE_PROCESSOR_H
#define ANIRA_HYBRID_NN_NONE_PROCESSOR_H

#include <anira/anira.h>

class HybridNNNoneProcessor : public anira::BackendBase {
public:
    HybridNNNoneProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::AudioBufferF &input, anira::AudioBufferF &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        auto equal_channels = input.get_num_channels() == output.get_num_channels();
        auto sample_diff = input.get_num_samples() - output.get_num_samples();
        int64_t num_batches;
        int64_t num_input_samples;
#if USE_LIBTORCH
        num_batches = m_inference_config.m_model_input_shape_torch[m_inference_config.m_index_audio_data[0]][0];
        num_input_samples = m_inference_config.m_model_input_shape_torch[m_inference_config.m_index_audio_data[0]][2];
#elif USE_ONNXRUNTIME
        num_batches = m_inference_config.m_model_input_shape_onnx[m_inference_config.m_index_audio_data[0]][0];
        num_input_samples = m_inference_config.m_model_input_shape_onnx[m_inference_config.m_index_audio_data[0]][2];
#elif USE_TFLITE
        num_batches = m_inference_config.m_model_input_shape_tflite[m_inference_config.m_index_audio_data[0]][0];
        num_input_samples = m_inference_config.m_model_input_shape_tflite[m_inference_config.m_index_audio_data[0]][1];
#endif

        if (equal_channels && sample_diff >= 0) {
            for (size_t channel = 0; channel < input.get_num_channels(); ++channel) {
                auto write_ptr = output.get_write_pointer(channel);
                auto read_ptr = input.get_read_pointer(channel);

                for (size_t batch = 0; batch < (size_t) num_batches; ++batch) {
                    size_t base_index = batch * num_input_samples;
                    write_ptr[batch] = read_ptr[num_input_samples - 1 + base_index];
                }
            }
        }
    }
};

#endif // ANIRA_HYBRID_NN_NONE_PROCESSOR_H