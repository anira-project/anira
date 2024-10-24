#ifndef ANIRA_TFLITEPROCESSOR_H
#define ANIRA_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "BackendBase.h"
#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <tensorflow/lite/c_api.h>

namespace anira {

class ANIRA_API TFLiteProcessor : private BackendBase {
public:
    TFLiteProcessor(InferenceConfig& config);
    ~TFLiteProcessor();

    void prepare() override;
    void process(AudioBufferF& input, AudioBufferF& output) override;

private:
    TfLiteModel* m_model;
    TfLiteInterpreterOptions* m_options;
    TfLiteInterpreter* m_interpreter;

    TfLiteTensor* m_input_tensor;
    const TfLiteTensor* m_output_tensor;
};

} // namespace anira

#endif
#endif //ANIRA_TFLITEPROCESSOR_H