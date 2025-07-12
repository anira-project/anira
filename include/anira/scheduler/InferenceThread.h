#ifndef ANIRA_INFERENCETHREAD_H
#define ANIRA_INFERENCETHREAD_H

#include <atomic>
#include <memory>
#include <vector>

#include "../system/HighPriorityThread.h"
#include "../utils/Buffer.h"
#include "SessionElement.h"
#include <concurrentqueue.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif

namespace anira {
    
class ANIRA_API InferenceThread : public HighPriorityThread {
public:
    InferenceThread(moodycamel::ConcurrentQueue<InferenceData>& next_inference);
    ~InferenceThread() override;

    bool execute();

private:
    void run() override;

    void do_inference(std::shared_ptr<SessionElement> session, std::shared_ptr<SessionElement::ThreadSafeStruct> thread_safe_struct);
    void inference(std::shared_ptr<SessionElement> session, std::vector<BufferF>& input, std::vector<BufferF>& output);
    void exponential_backoff(std::array<int, 2> iterations);

private:
    moodycamel::ConcurrentQueue<InferenceData>& m_next_inference;
    InferenceData m_inference_data;
 };

} // namespace anira

#endif //ANIRA_INFERENCETHREAD_H