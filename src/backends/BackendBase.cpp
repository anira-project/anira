#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace anira {

BackendBase::BackendBase(InferenceConfig& inference_config)
    : m_inference_config(inference_config) {}

void BackendBase::prepare() {}

// The session parameter is passed by value to match the virtual signature declared in the header
// (BackendBase.h), which is out of scope to change here.
void BackendBase::process(std::vector<BufferF>& input,
                          std::vector<BufferF>& output,
                          [[maybe_unused]] std::shared_ptr<SessionElement>
                              session) {  // NOLINT(performance-unnecessary-value-param)
    // Tensor counts may differ (e.g. stateful models with more inputs than
    // outputs): roundtrip the pairwise-matching tensors and zero any extra
    // outputs. Indexing output[] by the input count read out of bounds.
    for (size_t i = input.size(); i < output.size(); ++i) { output[i].clear(); }
    for (size_t i = 0; i < std::min(input.size(), output.size()); ++i) {
        bool const equal_channels = input[i].get_num_channels() == output[i].get_num_channels();
        auto sample_diff = input[i].get_num_samples() - output[i].get_num_samples();
        if (equal_channels && sample_diff == 0) {
            for (int channel = 0; channel < input[i].get_num_channels(); ++channel) {
                auto write_ptr = output[i].get_write_pointer(channel);
                auto read_ptr = input[i].get_read_pointer(channel);

                for (size_t j = 0; j < output[i].get_num_samples(); ++j) {
                    write_ptr[j] = read_ptr[j];
                }
            }
        } else {
            output[i].clear();
        }
    }
}

}  // namespace anira