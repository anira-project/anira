#ifndef ANIRA_PREPOSTPROCESSOR_H
#define ANIRA_PREPOSTPROCESSOR_H

#include "utils/RingBuffer.h"
#include "utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"

namespace anira {

class ANIRA_API PrePostProcessor
{
public:
    PrePostProcessor() = default;
    ~PrePostProcessor() = default;

    virtual void pre_process(RingBuffer& input, AudioBufferF& output, [[maybe_unused]] InferenceBackend current_inference_backend);
    virtual void post_process(AudioBufferF& input, RingBuffer& output, [[maybe_unused]] InferenceBackend current_inference_backend);

protected:
    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples);

    void pop_samples_from_buffer(RingBuffer& input, AudioBufferF& output, int num_new_samples, int num_old_samples, int offset);

    void push_samples_to_buffer(const AudioBufferF& input, RingBuffer& output);
};

} // namespace anira

#endif //ANIRA_PREPOSTPROCESSOR_H