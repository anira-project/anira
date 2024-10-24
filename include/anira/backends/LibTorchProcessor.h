#ifndef ANIRA_LIBTORCHPROCESSOR_H
#define ANIRA_LIBTORCHPROCESSOR_H

#ifdef USE_LIBTORCH

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include "BackendBase.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <stdlib.h>

namespace anira {

class ANIRA_API LibtorchProcessor : private BackendBase {
public:
    LibtorchProcessor(InferenceConfig& config);
    ~LibtorchProcessor();

    void prepare() override;
    void process(AudioBufferF& input, AudioBufferF& output) override;

private:
    torch::jit::script::Module m_module;

    torch::Tensor m_input_tensor;
    torch::Tensor m_output_tensor;

    std::vector<torch::jit::IValue> m_inputs;

};

} // namespace anira

#endif
#endif //ANIRA_LIBTORCHPROCESSOR_H