#ifndef ANIRA_CNN_CUSTOM_PROCESSOR_H
#define ANIRA_CNN_CUSTOM_PROCESSOR_H

#include <anira/anira.h>

class CNNBypassProcessor : public anira::BackendBase {
public:
    CNNBypassProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::BufferF &input, anira::BufferF &output, [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        auto sample_diff = input.get_num_samples() - output.get_num_samples();

        for (size_t channel = 0; channel < input.get_num_channels(); ++channel) {
            auto write_ptr = output.get_write_pointer(channel);
            auto read_ptr = input.get_read_pointer(channel);

            for (size_t i = 0; i < output.get_num_samples(); ++i) {
                write_ptr[i] = read_ptr[i+sample_diff];
            }
        }
    }
};

#endif // ANIRA_CNN_CUSTOM_PROCESSOR_H