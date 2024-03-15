/* ==========================================================================

Minimal TensorFlow Lite example from https://www.tensorflow.org/lite/guide/inference
Licence: Apache 2.0

========================================================================== */

#include <cstdio>
#include <iostream>
#include <array>
#include <tensorflow/lite/c_api.h>

#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/cnn/CNNConfig.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

#define MODEL_TO_USE 3

void minimal_inference(anira::InferenceConfig config) {
    std::cout << "Minimal TensorFlow-Lite example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << config.m_model_path_tflite << std::endl;

    // Load model
    TfLiteModel* model = TfLiteModelCreateFromFile(config.m_model_path_tflite.c_str());

    // Create the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    // Limit inference to one thread
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    TfLiteInterpreterResizeInputTensor(interpreter, 0, config.m_model_input_shape_tflite.data(), config.m_model_input_shape_tflite.size());

    // Allocate memory for all tensors
    TfLiteInterpreterAllocateTensors(interpreter);

    // Get input tensor
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    for (int i = 0; i < TfLiteTensorNumDims(inputTensor); ++i) {
        std::cout << "Input shape " << i <<": " << TfLiteTensorDim(inputTensor, i) << '\n';
    }

    // Fill input tensor with data
    const int inputSize = config.m_batch_size * config.m_model_input_size_backend;
    std::vector<float> inputData;
    for (int i = 0; i < inputSize; i++) {
        inputData.push_back(i * 0.000001f);
    }
    TfLiteTensorCopyFromBuffer(inputTensor, inputData.data(), inputSize * sizeof(float));

    // Execute inference.
    TfLiteInterpreterInvoke(interpreter);

    // Get output tensor
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Extract the output tensor data
    const int outputSize = config.m_batch_size * config.m_model_output_size_backend;
    std::vector<float> outputData;
    outputData.reserve(outputSize);

    TfLiteTensorCopyToBuffer(outputTensor, outputData.data(), outputSize * sizeof(float));

    for (int i = 0; i < TfLiteTensorNumDims(outputTensor); ++i) {
        std::cout << "Output shape " << i << ": " << TfLiteTensorDim(outputTensor, i) << '\n';
    }

    for (int i = 0; i < outputSize; i++) {
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

int main(int argc, const char* argv[]) {

    std::vector<anira::InferenceConfig> modelsToInference = {hybridNNConfig, cnnConfig, statefulRNNConfig};

    for (int i = 0; i < modelsToInference.size(); ++i) {
        minimal_inference(modelsToInference[i]);
    }

    return 0;
}