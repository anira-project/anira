#ifndef ANIRA_HYBRID_NN_NONE_PROCESSOR_H
#define ANIRA_HYBRID_NN_NONE_PROCESSOR_H

#include <anira/anira.h>

class HybridNNNoneProcessor : public anira::BackendBase {
public:
    HybridNNNoneProcessor(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void processBlock(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
        auto equalChannels = input.getNumChannels() == output.getNumChannels();
        auto sampleDiff = input.getNumSamples() - output.getNumSamples();
        int64_t num_batches;
        int64_t num_input_samples;
#if USE_LIBTORCH
        num_batches = inferenceConfig.m_model_input_shape_torch[0];
        num_input_samples = inferenceConfig.m_model_input_shape_torch[2];
#elif USE_ONNXRUNTIME
        num_batches = inferenceConfig.m_model_input_shape_onnx[0];
        num_input_samples = inferenceConfig.m_model_input_shape_onnx[2];
#elif USE_TFLITE
        num_batches = inferenceConfig.m_model_input_shape_tflite[0];
        num_input_samples = inferenceConfig.m_model_input_shape_tflite[1];
#endif

        if (equalChannels && sampleDiff >= 0) {
            for (size_t channel = 0; channel < input.getNumChannels(); ++channel) {
                auto writePtr = output.getWritePointer(channel);
                auto readPtr = input.getReadPointer(channel);

                for (size_t batch = 0; batch < num_batches; ++batch) {
                    size_t baseIdx = batch * num_input_samples;
                    writePtr[batch] = readPtr[num_input_samples - 1 + baseIdx];
                }
            }
        }
    }
};

#endif // ANIRA_HYBRID_NN_NONE_PROCESSOR_H