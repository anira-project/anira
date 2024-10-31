#include <anira/backends/BackendBase.h>

namespace anira {

BackendBase::BackendBase(InferenceConfig& inference_config) : m_inference_config(inference_config) {
}

void BackendBase::prepare() {

}

void BackendBase::process(AudioBufferF &input, AudioBufferF &output) {
    auto equal_channels = input.get_num_channels() == output.get_num_channels();
    auto sample_diff = input.get_num_samples() - output.get_num_samples();

    if (equal_channels && sample_diff == 0) {
        for (int channel = 0; channel < input.get_num_channels(); ++channel) {
            auto write_ptr = output.get_write_pointer(channel);
            auto read_ptr = input.get_read_pointer(channel);

            for (size_t i = 0; i < output.get_num_samples(); ++i) {
                write_ptr[i] = read_ptr[i];
            }
        }
    }
    else {
        output.clear();
    }
}

}