#ifndef ANIRA_LIBTORCHPROCESSOR_H
#define ANIRA_LIBTORCHPROCESSOR_H

#ifdef USE_LIBTORCH

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <torch/script.h>
#include <stdlib.h>

namespace anira {

class LibtorchProcessor {
public:
    LibtorchProcessor(InferenceConfig& config);
    ~LibtorchProcessor();

    void prepareToPlay();
    void processBlock(AudioBufferF& input, AudioBufferF& output);

private:
    InferenceConfig& inferenceConfig;

    torch::jit::script::Module module;

    at::Tensor inputTensor;
    at::Tensor outputTensor;
    std::vector<torch::jit::IValue> inputs;
};

} // namespace anira

#endif
#endif //ANIRA_LIBTORCHPROCESSOR_H