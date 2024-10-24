/* ==========================================================================

Minimal TensorFlow Lite example from https://www.tensorflow.org/lite/guide/inference
Licence: Apache 2.0

========================================================================== */

#include <cstdio>
#include <iostream>
#include <array>
#include <tensorflow/lite/c_api.h>

#include "../../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../../extras/desktop/models/cnn/CNNConfig.h"

#define TFLITE_MINIMAL_CHECK(x)                              \
    if (!(x)) {                                                \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

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

    std::vector<int> input_shape(config.m_model_input_shape_tflite.begin(), config.m_model_input_shape_tflite.end());
    TfLiteInterpreterResizeInputTensor(interpreter, 0, input_shape.data(), input_shape.size());

    // Allocate memory for all tensors
    TfLiteInterpreterAllocateTensors(interpreter);

    // Get input tensor
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    for (int i = 0; i < TfLiteTensorNumDims(input_tensor); ++i) {
        std::cout << "Input shape " << i <<": " << TfLiteTensorDim(input_tensor, i) << '\n';
    }

    // Fill input tensor with data
    std::vector<float> input_data;
    for (int i = 0; i < config.m_new_model_input_size; i++) {
        input_data.push_back(i * 0.000001f);
    }
    TfLiteTensorCopyFromBuffer(input_tensor, input_data.data(), config.m_new_model_input_size * sizeof(float));

    // Execute inference.
    TfLiteInterpreterInvoke(interpreter);

    // Get output tensor
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    // Extract the output tensor data
    std::vector<float> output_data;
    output_data.reserve(config.m_new_model_output_size);

    TfLiteTensorCopyToBuffer(output_tensor, output_data.data(), config.m_new_model_output_size * sizeof(float));

    for (int i = 0; i < TfLiteTensorNumDims(output_tensor); ++i) {
        std::cout << "Output shape " << i << ": " << TfLiteTensorDim(output_tensor, i) << '\n';
    }

    for (int i = 0; i < config.m_new_model_output_size; i++) {
        std::cout << "Output data [" << i << "]: " << output_data[i] << std::endl;
    }

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}

int main(int argc, const char* argv[]) {

    std::vector<anira::InferenceConfig> models_to_inference = {hybridnn_config, cnn_config, rnn_config};

    for (int i = 0; i < models_to_inference.size(); ++i) {
        minimal_inference(models_to_inference[i]);
    }

    return 0;
}