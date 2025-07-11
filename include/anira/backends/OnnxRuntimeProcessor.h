#ifndef ANIRA_ONNXRUNTIMEPROCESSOR_H
#define ANIRA_ONNXRUNTIMEPROCESSOR_H

#ifdef USE_ONNXRUNTIME

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/Buffer.h"
#include "../scheduler/SessionElement.h"

#include <onnxruntime_cxx_api.h>

namespace anira {

class ANIRA_API OnnxRuntimeProcessor : public BackendBase {
public:
    OnnxRuntimeProcessor(InferenceConfig& inference_config);
    ~OnnxRuntimeProcessor();

    void prepare() override;
    void process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session) override;

private:
    struct Instance {
        Instance(InferenceConfig& inference_config);
        ~Instance();

        void prepare();
        void process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session);

        Ort::MemoryInfo m_memory_info;
        Ort::Env m_env;
        Ort::AllocatorWithDefaultOptions m_ort_alloc;
        Ort::SessionOptions m_session_options;

        std::unique_ptr<Ort::Session> m_session;

        std::vector<MemoryBlock<float>> m_input_data;
        std::vector<Ort::Value> m_inputs;
        std::vector<Ort::Value> m_outputs;

        std::vector<Ort::AllocatedStringPtr> m_input_name;
        std::vector<Ort::AllocatedStringPtr> m_output_name;

        std::vector<const char *> m_output_names;
        std::vector<const char *> m_input_names;

        InferenceConfig& m_inference_config;
        std::atomic<bool> m_processing {false};
    };

    std::vector<std::shared_ptr<Instance>> m_instances;
};

} // namespace anira

#endif
#endif //ANIRA_ONNXRUNTIMEPROCESSOR_H