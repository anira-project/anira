/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <onnxruntime_cxx_api.h>
#include <iostream>

int main(int argc, char* argv[]) {

    std::cout << "Minimal OnnxRuntime example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

#if MODEL_TO_USE == 1
    const int batchSize = 2;
    const int modelInputSize = 150;
    const int modelOutputSize = 1;
#elif MODEL_TO_USE == 2
    const int batchSize = 1;
    const int modelInputSize = 15380;
    const int modelOutputSize = 2048;
#elif MODEL_TO_USE == 3
    const int batchSize = 1;
    const int modelInputSize = 2048;
    const int modelOutputSize = 2048;
#endif

    bool tflite = true;
    bool libtorch = true;

    if (tflite) {

        std::cout << "Tensorflow model converted to onnx:" << std::endl;

#if MODEL_TO_USE == 1
        std::string filepath = GUITARLSTM_MODELS_PATH_TENSORFLOW;
        std::string modelpath = filepath + "model_0/model_0-tflite-minimal.onnx";
#elif MODEL_TO_USE == 2
        std::string filepath = STEERABLENAFX_MODELS_PATH_TENSORFLOW;
        std::string modelpath = filepath + "model_0/steerable-nafx-tflite-2048.onnx";
#elif MODEL_TO_USE == 3
        std::string filepath = STATEFULLSTM_MODELS_PATH_TENSORFLOW;
        std::string modelpath = filepath + "model_0/stateful-lstm-tflite.onnx";
#endif

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
        std::wstring modelWideStr = std::wstring(modelpath.begin(), modelpath.end());
        const wchar_t* modelWideCStr = modelWideStr.c_str();
        Ort::Session session(env, modelWideCStr, session_options);
#else
        Ort::Session session(env, modelpath.c_str(), Ort::SessionOptions{ nullptr });
#endif

        // Define input data
        const int inputSize = batchSize * modelInputSize;
        float inputData[inputSize];
        for (int i = 0; i < inputSize; i++) {
            inputData[i] = i * 0.000001f;
        }

        // Define the shape of input tensor
        std::array<int64_t, 3> inputShape = {batchSize, modelInputSize, 1};

        // Create input tensor object from input data values and shape
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>  (memory_info,
                                                                        inputData,
                                                                        inputSize,
                                                                        inputShape.data(),
                                                                        inputShape.size());

        std::cout << "Input shape 0: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[0] << '\n';
        std::cout << "Input shape 1: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[1] << '\n';
        std::cout << "Input shape 2: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[2] << '\n';

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
        
        std::cout << "Output shape 0: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[0] << std::endl;
        std::cout << "Output shape 1: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1] << std::endl;
        std::cout << "Output shape 2: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[2] << std::endl;

        // Define output vector
        int outputSize = batchSize * modelOutputSize;
        const float* outputData = outputTensors[0].GetTensorData<float>();

        // Extract the output tensor data
        for (int i = 0; i < outputSize; i++) {
            std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
        }
    }

    if (libtorch) {

        std::cout << "PyTorch model converted to onnx:" << std::endl;

#if MODEL_TO_USE == 1
        std::string filepath = GUITARLSTM_MODELS_PATH_PYTORCH;
        std::string modelpath = filepath + "model_0/model_0-libtorch-minimal.onnx";
#elif MODEL_TO_USE == 2
        std::string filepath = STEERABLENAFX_MODELS_PATH_PYTORCH;
        std::string modelpath = filepath + "model_0/steerable-nafx-libtorch-2048.onnx";
#elif MODEL_TO_USE == 3
        std::string filepath = STATEFULLSTM_MODELS_PATH_PYTORCH;
        std::string modelpath = filepath + "model_0/stateful-lstm-libtorch.onnx";
#endif

        // Define environment that holds logging state used by all other objects.
        // Note: One Env must be created before using any other Onnxruntime functionality.
        Ort::Env env;
        // Define memory info for input and output tensors for CPU usage
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // Define allocator
        Ort::AllocatorWithDefaultOptions ort_alloc;

        // Load the model and create InferenceSession
#ifdef _WIN32
        std::wstring modelWideStr = std::wstring(modelpath.begin(), modelpath.end());
        const wchar_t* modelWideCStr = modelWideStr.c_str();
        Ort::Session session(env, modelWideCStr, Ort::SessionOptions{nullptr });
#else
        Ort::Session session(env, modelpath.c_str(), Ort::SessionOptions{ nullptr });
#endif

        // Define input data
        const int inputSize = batchSize * modelInputSize;
        float inputData[inputSize];
        for (int i = 0; i < inputSize; i++) {
            inputData[i] = i * 0.001f;
        }

        // Define the shape of input tensor
#if MODEL_TO_USE == 1 || MODEL_TO_USE == 2
        std::array<int64_t, 3> inputShape = {batchSize, 1, modelInputSize};
#elif MODEL_TO_USE == 3
        std::array<int64_t, 3> inputShape = {modelInputSize, batchSize, 1};
#endif

        // Create input tensor object from input data values and shape
        const Ort::Value inputTensor = Ort::Value::CreateTensor<float>  (memory_info,
                                                                        inputData,
                                                                        inputSize,
                                                                        inputShape.data(),
                                                                        inputShape.size());

        std::cout << "Input shape 0: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[0] << '\n';
        std::cout << "Input shape 1: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[1] << '\n';
        std::cout << "Input shape 2: " << inputTensor.GetTensorTypeAndShapeInfo().GetShape()[2] << '\n';

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
        
        std::cout << "Output shape 0: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[0] << std::endl;
        std::cout << "Output shape 1: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1] << std::endl;
        std::cout << "Output shape 2: " << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[2] << std::endl;

        // Define output vector
        int outputSize = batchSize * modelOutputSize;
        const float* outputData = outputTensors[0].GetTensorData<float>();

        // Extract the output tensor data
        for (int i = 0; i < outputSize; i++) {
            std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
        }
    }
}