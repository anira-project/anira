#ifndef ANIRA_LIBTORCHPROCESSOR_H
#define ANIRA_LIBTORCHPROCESSOR_H

#ifdef USE_LIBTORCH

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include "BackendBase.h"
#include <torch/script.h>
#include <stdlib.h>

namespace anira {

class ANIRA_API LibtorchProcessor : private BackendBase {
public:
    LibtorchProcessor(InferenceConfig& config);
    ~LibtorchProcessor();

    void prepareToPlay() override;
    void processBlock(AudioBufferF& input, AudioBufferF& output) override;

private:
    torch::jit::script::Module module;

    torch::Tensor inputTensor;
    torch::Tensor outputTensor;

    std::vector<torch::jit::IValue> inputs;

};

} // namespace anira

#endif
#endif //ANIRA_LIBTORCHPROCESSOR_H