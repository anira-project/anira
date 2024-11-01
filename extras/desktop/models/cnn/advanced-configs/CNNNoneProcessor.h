#ifndef ANIRA_CNN_NONE_PROCESSOR_H
#define ANIRA_CNN_NONE_PROCESSOR_H

#include <anira/anira.h>

class CNNNoneProcessor : public anira::BackendBase {
public:
    CNNNoneProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::AudioBufferF &input, anira::AudioBufferF &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        auto equal_channels = input.get_num_channels() == output.get_num_channels();
        auto sample_diff = input.get_num_samples() - output.get_num_samples();

        if (equal_channels && sample_diff >= 0) {
            for (size_t channel = 0; channel < input.get_num_channels(); ++channel) {
                auto write_ptr = output.get_write_pointer(channel);
                auto read_ptr = input.get_read_pointer(channel);

                for (size_t i = 0; i < output.get_num_samples(); ++i) {
                    write_ptr[i] = read_ptr[i+sample_diff];
                }
            }
        }
    }
};

#endif // ANIRA_CNN_NONE_PROCESSOR_H