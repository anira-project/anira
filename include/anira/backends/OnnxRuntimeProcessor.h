#ifndef ANIRA_ONNXRUNTIMEPROCESSOR_H
#define ANIRA_ONNXRUNTIMEPROCESSOR_H

#ifdef USE_ONNXRUNTIME

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <onnxruntime_cxx_api.h>

namespace anira {

class ANIRA_API OnnxRuntimeProcessor : private BackendBase {
public:
    OnnxRuntimeProcessor(InferenceConfig& config);
    ~OnnxRuntimeProcessor();

    void prepare() override;
    void process(AudioBufferF& input, AudioBufferF& output) override;

private:
    Ort::Env m_env;
    Ort::MemoryInfo m_memory_info;
    Ort::AllocatorWithDefaultOptions m_ort_alloc;
    Ort::SessionOptions m_session_options;
    std::unique_ptr<Ort::Session> m_session;

    size_t m_input_size;
    size_t m_output_size;

    std::vector<float> m_input_data;
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    std::unique_ptr<Ort::AllocatedStringPtr> m_input_name;
    std::unique_ptr<Ort::AllocatedStringPtr> m_output_name;

    std::array<const char *, 1> m_input_names;
    std::array<const char *, 1> m_output_names;
};

} // namespace anira

#endif
#endif //ANIRA_ONNXRUNTIMEPROCESSOR_H