#ifndef ANIRA_TFLITEPROCESSOR_H
#define ANIRA_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <tensorflow/lite/c_api.h>
#include <memory>

namespace anira {

class ANIRA_API TFLiteProcessor : public BackendBase {
public:
    TFLiteProcessor(InferenceConfig& inference_config);
    ~TFLiteProcessor();

    void prepare() override;
    void process(AudioBufferF& input, AudioBufferF& output) override;

private:
    struct Instance {
        Instance(InferenceConfig& inference_config);
        ~Instance();
        
        void prepare();
        void process(AudioBufferF& input, AudioBufferF& output);

        TfLiteModel* m_model;
        TfLiteInterpreterOptions* m_options;
        TfLiteInterpreter* m_interpreter;

        TfLiteTensor* m_input_tensor;
        const TfLiteTensor* m_output_tensor;

        InferenceConfig& m_inference_config;
        std::atomic<bool> m_processing {false};
    };

    std::vector<std::shared_ptr<Instance>> m_instances;
};

} // namespace anira

#endif
#endif //ANIRA_TFLITEPROCESSOR_H