#ifndef AARI_ONNXRUNTIMEPROCESSOR_H
#define AARI_ONNXRUNTIMEPROCESSOR_H

#ifdef USE_ONNXRUNTIME

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <onnxruntime_cxx_api.h>

namespace aari {

class OnnxRuntimeProcessor {
public:
    OnnxRuntimeProcessor(InferenceConfig& config);
    ~OnnxRuntimeProcessor();

    void prepareToPlay();
    void processBlock(AudioBufferF& input, AudioBufferF& output);

private:
    InferenceConfig& inferenceConfig;

    Ort::Env env;
    Ort::MemoryInfo memory_info;
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<int64_t> inputShape;
    std::array<const char *, 1> inputNames;

    std::array<const char *, 1> outputNames;
    // Define output tensor vector
    std::vector<Ort::Value> outputTensors;
};

} // namespace aari

#endif
#endif //AARI_ONNXRUNTIMEPROCESSOR_H