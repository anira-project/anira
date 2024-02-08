/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

#if WIN32
    _putenv("OMP_NUM_THREADS=1");
    _putenv("MKL_NUM_THREADS=1");
#else
    putenv("OMP_NUM_THREADS=1");
    putenv("MKL_NUM_THREADS=1");
#endif

#if MODEL_TO_USE == 1
    std::string filepath = GUITARLSTM_MODELS_PATH_PYTORCH;
    std::string modelpath = filepath + "model_0/model_0-minimal.pt";

    const int batchSize = 2;
    const int modelInputSize = 150;
    const int modelOutputSize = 1;
#elif MODEL_TO_USE == 2
    std::string filepath = STEERABLENAFX_MODELS_PATH_PYTORCH;
    std::string modelpath = filepath + "model_0/steerable-nafx-2048.pt";

    const int batchSize = 1;
    const int modelInputSize = 15380;
    const int modelOutputSize = 2048;
#elif MODEL_TO_USE == 3
    std::string filepath = STATEFULLSTM_MODELS_PATH_PYTORCH;
    std::string modelpath = filepath + "model_0/stateful-lstm.pt";

    const int batchSize = 1;
    const int modelInputSize = 2048;
    const int modelOutputSize = 2048;
#endif

    // Load model
    torch::jit::Module module;
    try {
        module = torch::jit::load(modelpath);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cout << e.what() << std::endl;
        return -1;
    }

    // Fill input tensor with data
    const int inputSize = batchSize * modelInputSize;
    float inputData[inputSize];
    for (int i = 0; i < inputSize; i++) {
        inputData[i] = i * 0.000001f;
    }

    // Create input tensor object from input data values and reshape
#if MODEL_TO_USE == 1 || MODEL_TO_USE == 2
    torch::Tensor inputTensor = torch::from_blob(&inputData, { batchSize, 1, modelInputSize });
#elif MODEL_TO_USE == 3
    torch::Tensor inputTensor = torch::from_blob(&inputData, { modelInputSize, batchSize, 1 });
#endif

    std::cout << "Input shape 0: " << inputTensor.sizes()[0] << '\n';
    std::cout << "Input shape 1: " << inputTensor.sizes()[1] << '\n';
    std::cout << "Input shape 2: " << inputTensor.sizes()[2] << '\n';

    // Create IValue vector for input of interpreter
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    // Execute inference
    at::Tensor outputTensor = module.forward(inputs).toTensor();

    std::cout << "Output shape 0: " << outputTensor.sizes()[0] << '\n';
    std::cout << "Output shape 1: " << outputTensor.sizes()[1] << '\n';
    std::cout << "Output shape 2: " << outputTensor.sizes()[2] << '\n';

    // Extract the output tensor data
    const int outputSize = batchSize * modelOutputSize;
    float outputData[outputSize];

#if MODEL_TO_USE == 1
    for (int i = 0; i < outputSize; i++) {
        outputData[i] = outputTensor[i][0].item().toFloat();
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }
#elif MODEL_TO_USE == 2
    for (int i = 0; i < outputSize; i++) {
        outputData[i] = outputTensor[0][0][i].item().toFloat();
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }
#elif MODEL_TO_USE == 3
    for (int i = 0; i < outputSize; i++) {
        outputData[i] = outputTensor[i][0][0].item().toFloat();
        std::cout << "Output data [" << i << "]: " << outputData[i] << std::endl;
    }
#endif

    return 0;
}