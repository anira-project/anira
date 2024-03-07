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

    void prepareToPlay() override;
    void processBlock(AudioBufferF& input, AudioBufferF& output) override;

private:
    Ort::Env env;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    size_t inputSize;
    size_t outputSize;

    std::vector<float> inputData;
    std::vector<Ort::Value> inputTensor;
    std::vector<Ort::Value> outputTensor;

    std::unique_ptr<Ort::AllocatedStringPtr> inputName;
    std::unique_ptr<Ort::AllocatedStringPtr> outputName;

    std::array<const char *, 1> inputNames;
    std::array<const char *, 1> outputNames;
};

} // namespace anira

#endif
#endif //ANIRA_ONNXRUNTIMEPROCESSOR_H