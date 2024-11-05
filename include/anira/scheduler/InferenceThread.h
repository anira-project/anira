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
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {
    
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
#ifdef USE_SEMAPHORE
    InferenceThread(std::counting_semaphore<UINT16_MAX>& global_counter, std::vector<std::shared_ptr<SessionElement>>& sessions);
#else
    InferenceThread(std::atomic<int>& global_counter, std::vector<std::shared_ptr<SessionElement>>& sessions);
#endif
    ~InferenceThread() = default;

    bool execute();

private:
    void run() override;

    bool tryInference(std::shared_ptr<SessionElement> session);
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);
    void exponential_backoff(std::array<int, 2> iterations);

private:
#ifdef USE_SEMAPHORE
    std::counting_semaphore<UINT16_MAX>& m_global_counter;
#else
    std::atomic<int>& m_global_counter;
#endif
    std::vector<std::shared_ptr<SessionElement>>& m_sessions;
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H