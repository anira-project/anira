/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <iostream>
#include <onnxruntime_cxx_api.h>

#include "../../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../../extras/desktop/models/cnn/CNNConfig.h"

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
    std::vector<float> input_data;
    for (int i = 0; i < config.m_new_model_input_size; i++) {
        input_data.push_back(i * 0.000001f);
    }

    // Define the shape of input tensor
    std::vector<int64_t> input_shape = config.m_model_input_shape_onnx;

    // Create input tensor object from input data values and shape
    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>  (memory_info,
                                                                    input_data.data(),
                                                                    config.m_new_model_input_size,
                                                                    input_shape.data(),
                                                                    input_shape.size());

    for (int i = 0; i < input_tensor.GetTensorTypeAndShapeInfo().GetShape().size(); ++i) {
        std::cout << "Input shape " << i << ": " << input_tensor.GetTensorTypeAndShapeInfo().GetShape()[i] << std::endl;
    }

    // Get input and output names from model
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char *, 1> input_names = {(char*) input_name.get()};
    const std::array<const char *, 1> output_names = {(char*) output_name.get()};

    // Define output tensor vector
    std::vector<Ort::Value> output_tensors;

    try {
        // Run inference
        output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
    }
    catch (Ort::Exception &e) {
        std::cout << e.what() << std::endl;
    }

    for (int i = 0; i < output_tensors[0].GetTensorTypeAndShapeInfo().GetShape().size(); ++i) {
        std::cout << "Output shape " << i << ": " << output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[i] << std::endl;
    }

    // Define output vector
    const float* output_data = output_tensors[0].GetTensorData<float>();

    // Extract the output tensor data
    for (int i = 0; i < config.m_new_model_output_size; i++) {
        std::cout << "Output data [" << i << "]: " << output_data[i] << std::endl;
    }
}

int main(int argc, const char* argv[]) {

    std::vector<anira::InferenceConfig> models_to_inference = {hybridnn_config, cnn_config, rnn_config};

    for (int i = 0; i < models_to_inference.size(); ++i) {
        minimal_inference(models_to_inference[i]);
    }

    return 0;
}