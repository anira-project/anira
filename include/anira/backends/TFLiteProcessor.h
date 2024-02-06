#ifndef ANIRA_TFLITEPROCESSOR_H
#define ANIRA_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <tensorflow/lite/c_api.h>

namespace anira {

class TFLiteProcessor {
public:
    TFLiteProcessor(InferenceConfig& config);
    ~TFLiteProcessor();

    void prepareToPlay();
    void processBlock(AudioBufferF& input, AudioBufferF& output);

private:
    InferenceConfig& inferenceConfig;

    TfLiteModel* model;
    TfLiteInterpreterOptions* options;
    TfLiteInterpreter* interpreter;

    TfLiteTensor* inputTensor;
    const TfLiteTensor* outputTensor;
};

} // namespace anira

#endif
#endif //ANIRA_TFLITEPROCESSOR_H