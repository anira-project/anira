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
#include "../backends/BackendBase.h"
#include "SessionElement.h"
#include "../utils/AudioBuffer.h"

namespace anira {
    
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
#ifdef USE_SEMAPHORE
    InferenceThread(std::counting_semaphore<UINT16_MAX>& global_counter, std::vector<std::shared_ptr<SessionElement>>& sessions);
#else
    InferenceThread(std::atomic<int>& global_counter, std::vector<std::shared_ptr<SessionElement>>& sessions);
#endif
    ~InferenceThread() = default;

    void run() override;

private:
    bool tryInference(std::shared_ptr<SessionElement> session);
    void inference(std::shared_ptr<SessionElement> session, AudioBufferF& input, AudioBufferF& output);

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