/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <iostream>
#include <onnxruntime_cxx_api.h>

#include "../../../extras/models/stateful-rnn/StatefulLstmConfig.h"
#include "../../../extras/models/stateless-rnn/StatelessLstmConfig.h"
#include "../../../extras/models/cnn/CnnConfig.h"

void minimal_inference(anira::InferenceConfig config) {

    std::cout << "Minimal OnnxRuntime example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << config.m_model_path_onnx << std::endl;

    // Define environment that holds logging state used by all other objects.
    // Note: One Env must be created before using any other Onnxruntime functionality.
    Ort::Env env;
    // Define memory info for input and output tensors for CPU usage
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // Define allocator
    Ort::AllocatorWithDefaultOptions ort_alloc;

    // Limit inference to one thread
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load the model and create InferenceSession
#ifdef _WIN32
    std::wstring modelWideStr = std::wstring(config.m_model_path_onnx.begin(), config.m_model_path_onnx.end());
    const wchar_t* modelWideCStr = modelWideStr.c_str();
    Ort::Session session(env, modelWideCStr, session_options);
#else
    Ort::Session session(env, config.m_model_path_onnx.c_str(), Ort::SessionOptions{ nullptr });
#endif

    // Define input data
    const int inputSize = config.m_batch_size * config.m_model_input_size_backend;
    std::vector<float> inputData;
    for (int i = 0; i < inputSize; i++) {
        inputData.push_back(i * 0.000001f);
    }

    // Define the shape of input tensor
    std::vector<int64_t> inputShape = config.m_model_input_shape_onnx;

    // Create input tensor object from input data values and shape
    const Ort::Value inputTensor = Ort::Value::CreateTensor<float>  (memory_info,
                                                                    inputData.data(),
                                                                    inputSize,
                                                                    inputShape.data(),
                                                                    inputShape.size());

    for (int i = 0; i < inputTensor.GetTensorTypeAndShapeInfo().GetShape().size(); ++i) {
        std::cout << "Input shape " << i << ": " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[i] << std::endl;
    }

    // Get input and output names from model
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char *, 1> inputNames = {(char*) inputName.get()};
    const std::array<const char *, 1> outputNames = {(char*) outputName.get()};

    // Define output tensor vector
    std::vector<Ort::Value> outputTensors;

    try {
        // Run inference
        outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, inputNames.size(), outputNames.data(), outputNames.size());
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
    }

    for (int i = 0; i < outputTensors[0].GetTensorTypeAndShapeInfo().GetShape().size(); ++i) {
        std::cout << "Output shape " << i << ": " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[i] << std::endl;
    }

    // Define output vector
    int outputSize = config.m_batch_size * config.m_model_output_size_backend;
    const float* outputData = outputTensors[0].GetTensorData<float>();

    // Extract the output tensor data
    for (int i = 0; i < outputSize; i++) {
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }
}

int main(int argc, const char* argv[]) {

    std::vector<anira::InferenceConfig> modelsToInference = {statelessRnnConfig, cnnConfig, statefulRnnConfig};

    for (int i = 0; i < modelsToInference.size(); ++i) {
        minimal_inference(modelsToInference[i]);
    }

    return 0;
}