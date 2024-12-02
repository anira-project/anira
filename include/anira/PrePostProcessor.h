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
    PrePostProcessor(); 
    PrePostProcessor(InferenceConfig& inference_config);
    ~PrePostProcessor() = default;

    virtual void pre_process(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend current_inference_backend);
    virtual void post_process(AudioBufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend current_inference_backend);

    void set_input(const float& input, size_t i, size_t j);
    void set_output(const float& output, size_t i, size_t j);
    float get_input(size_t i, size_t j);
    float get_output(size_t i, size_t j);

protected:
    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, size_t num_new_samples, size_t num_old_samples);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, size_t num_new_samples, size_t num_old_samples, size_t offset);

    void push_samples_to_buffer(const AudioBufferF& input, RingBuffer& output);

private:
    std::vector<MemoryBlock<std::atomic<float>>> m_inputs;
    std::vector<MemoryBlock<std::atomic<float>>> m_outputs;

    std::array<size_t, 2> m_index_audio_data;
};

} // namespace anira

#endif //ANIRA_PREPOSTPROCESSOR_H