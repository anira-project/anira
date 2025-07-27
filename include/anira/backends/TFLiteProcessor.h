#ifndef ANIRA_TFLITEPROCESSOR_H
#define ANIRA_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/Buffer.h"
#include "../scheduler/SessionElement.h"
#include <tensorflow/lite/c_api.h>
#include <memory>

namespace anira {

class ANIRA_API TFLiteProcessor : public BackendBase {
public:
    TFLiteProcessor(InferenceConfig& inference_config);
    ~TFLiteProcessor();

    void prepare() override;
    void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session) override;

private:
    struct Instance {
        Instance(InferenceConfig& inference_config);
        ~Instance();
        
        void prepare();
        void process(std::vector<BufferF>& input, std::vector<BufferF>& output, std::shared_ptr<SessionElement> session);

        TfLiteModel* m_model;
        TfLiteInterpreterOptions* m_options;
        TfLiteInterpreter* m_interpreter;

        std::vector<MemoryBlock<float>> m_input_data;

        std::vector<TfLiteTensor*> m_inputs;
        std::vector<const TfLiteTensor*> m_outputs;

        InferenceConfig& m_inference_config;
        std::atomic<bool> m_processing {false};

#if DOXYGEN
        // Placeholder for Doxygen documentation
        // Since Doxygen does not find classes structures nested in std::shared_ptr
        MemoryBlock<float>* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
    };

    std::vector<std::shared_ptr<Instance>> m_instances;

#if DOXYGEN
    Instance* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif
#endif //ANIRA_TFLITEPROCESSOR_H