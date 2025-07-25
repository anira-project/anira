#ifndef ANIRA_PREPOSTPROCESSOR_H
#define ANIRA_PREPOSTPROCESSOR_H

#include "utils/RingBuffer.h"
#include "utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"
#include "InferenceConfig.h"
#include <atomic>
#include <vector>
#include <cassert>

namespace anira {

class ANIRA_API PrePostProcessor
{
public:
    PrePostProcessor() = delete;
    PrePostProcessor(InferenceConfig& inference_config);
    ~PrePostProcessor() = default;

    virtual void pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output, [[maybe_unused]] InferenceBackend current_inference_backend);
    virtual void post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output, [[maybe_unused]] InferenceBackend current_inference_backend);

    void set_input(const float& input, size_t i, size_t j);
    void set_output(const float& output, size_t i, size_t j);
    float get_input(size_t i, size_t j);
    float get_output(size_t i, size_t j);

    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_samples);
    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples);
    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples, size_t offset);
    void push_samples_to_buffer(const BufferF& input, RingBuffer& output, size_t num_samples);

protected:
    InferenceConfig& m_inference_config;
    
private:
    std::vector<MemoryBlock<std::atomic<float>>> m_inputs;
    std::vector<MemoryBlock<std::atomic<float>>> m_outputs;
};

} // namespace anira

#endif //ANIRA_PREPOSTPROCESSOR_H