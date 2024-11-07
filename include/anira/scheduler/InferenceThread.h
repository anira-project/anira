#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#ifdef USE_SEMAPHORE
    #include <semaphore>
#else
    #include <atomic>
#endif
#include <memory>
#include <vector>

#include "../system/HighPriorityThread.h"
#include "../utils/AudioBuffer.h"
#include "SessionElement.h"
#include "concurrentqueue.h"
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {
    
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
#ifdef USE_SEMAPHORE
    InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference);
#else
    InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference);
#endif
    ~InferenceThread() = default;

    bool execute();

private:
    void run() override;

    void do_inference(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> thread_safe_struct);
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);
    void exponential_backoff(std::array<int, 2> iterations);

private:

    moodycamel::ConcurrentQueue<InferenceData>& m_next_inference;
    InferenceData m_inference_data;

    int m_last_session_index = 0;
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H