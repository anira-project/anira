#ifndef ANIRA_PREPOSTPROCESSOR_H
#define ANIRA_PREPOSTPROCESSOR_H

#include "utils/RingBuffer.h"
#include "utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"
#include "InferenceConfig.h"

namespace anira {

class ANIRA_API PrePostProcessor
{
public:
    PrePostProcessor();
    PrePostProcessor(InferenceConfig& inference_config);
    ~PrePostProcessor() = default;

    virtual void pre_process(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend current_inference_backend);
    virtual void post_process(AudioBufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend current_inference_backend);

    // TODO: implement these functions
    // bool set_input(const std::vector<float>& input, size_t index);
    // bool set_output(const std::vector<float>& output, size_t index);

    // std::vector<float> get_input(size_t index);
    // std::vector<float> get_output(size_t index);

protected:
    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples, int offset);

    void push_samples_to_buffer(const AudioBufferF& input, RingBuffer& output);

public:
    std::vector<MemoryBlock<std::atomic<float>>> m_inputs;
    std::vector<MemoryBlock<std::atomic<float>>> m_outputs;

private:
    InferenceConfig m_inference_config;
};

} // namespace anira

#endif //ANIRA_PREPOSTPROCESSOR_H