#ifndef ANIRA_LIBTORCHPROCESSOR_H
#define ANIRA_LIBTORCHPROCESSOR_H

#ifdef USE_LIBTORCH

// Avoid min/max macro conflicts on Windows for LibTorch compatibility
#ifdef _WIN32
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#endif

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include "BackendBase.h"
#include "../scheduler/SessionElement.h"
#include <stdlib.h>
#include <memory>

// LibTorch headers trigger many warnings; disabling for cleaner build logs
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wall"
#endif

#include <torch/script.h>
#include <torch/torch.h>

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace anira {

class ANIRA_API LibtorchProcessor : public BackendBase {
public:
    LibtorchProcessor(InferenceConfig& inference_config);
    ~LibtorchProcessor();

    void prepare() override;
    void process(AudioBufferF& input, AudioBufferF& output, std::shared_ptr<SessionElement> session) override;

private:
    struct Instance {
        Instance(InferenceConfig& inference_config);
        void prepare();
        void process(AudioBufferF& input, AudioBufferF& output, std::shared_ptr<SessionElement> session);

        torch::jit::script::Module m_module;

        std::vector<MemoryBlock<float>> m_input_data;

        std::vector<c10::IValue> m_inputs;
        c10::IValue m_outputs;

        InferenceConfig& m_inference_config;
        std::atomic<bool> m_processing {false};
    };
    
    std::vector<std::shared_ptr<Instance>> m_instances;
};

} // namespace anira

#endif
#endif //ANIRA_LIBTORCHPROCESSOR_H