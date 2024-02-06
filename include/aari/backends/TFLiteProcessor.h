#ifndef AARI_TFLITEPROCESSOR_H
#define AARI_TFLITEPROCESSOR_H

#ifdef USE_TFLITE

#include "../InferenceConfig.h"
#include "../utils/AudioBuffer.h"
#include <tensorflow/lite/c_api.h>

namespace aari {

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

} // namespace aari

#endif
#endif //AARI_TFLITEPROCESSOR_H