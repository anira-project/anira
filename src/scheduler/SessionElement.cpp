#include <anira/scheduler/SessionElement.h>

namespace anira {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& ppP, InferenceConfig& config) :
    sessionID(newSessionID),
    prePostProcessor(ppP),
    inferenceConfig(config)
{
    const size_t batch_size = inferenceConfig.m_batch_size;
    const size_t model_input_size = inferenceConfig.m_model_input_size_backend;
    const size_t model_output_size = inferenceConfig.m_model_output_size_backend;

    for (int i = 0; i < 5000; ++i) {
        inferenceQueue.emplace_back(std::make_unique<ThreadSafeStruct>(batch_size, model_input_size, model_output_size));
    }
}

    SessionElement::ThreadSafeStruct::ThreadSafeStruct(size_t batch_size, size_t model_input_size,
                                                       size_t model_output_size) {
        processedModelInput.initialize(1, batch_size * model_input_size);
        rawModelOutput.initialize(1, batch_size * model_output_size);
    }
} // namespace anira