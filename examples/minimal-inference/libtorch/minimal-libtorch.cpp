/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

#include "../../../extras/models/stateful-rnn/StatefulLstmConfig.h"
#include "../../../extras/models/stateless-rnn/StatelessLstmConfig.h"
#include "../../../extras/models/cnn/CnnConfig.h"

void minimal_inference(anira::InferenceConfig config) {
    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << config.m_model_path_torch << std::endl;

#if WIN32
    _putenv("OMP_NUM_THREADS=1");
    _putenv("MKL_NUM_THREADS=1");
#else
    putenv("OMP_NUM_THREADS=1");
    putenv("MKL_NUM_THREADS=1");
#endif

    // Load model
    torch::jit::Module module;
    try {
        module = torch::jit::load(config.m_model_path_torch);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cout << e.what() << std::endl;
    }

    // Fill input tensor with data
    const int inputSize = config.m_batch_size * config.m_model_input_size_backend;
    std::vector<float> inputData;
    for (int i = 0; i < inputSize; i++) {
        inputData.push_back(i * 0.000001f);
    }

    // Create input tensor object from input data values and reshape
    std::vector<int64_t> shape = config.m_model_input_shape_torch;
    torch::Tensor inputTensor = torch::from_blob(inputData.data(), shape);

    for (int i = 0; i < inputTensor.sizes().size(); ++i) {
        std::cout << "Input shape " << i << ": " << inputTensor.sizes()[i] << '\n';
    }

    // Create IValue vector for input of interpreter
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    // Execute inference
    at::Tensor outputTensor = module.forward(inputs).toTensor();

    for (int i = 0; i < outputTensor.sizes().size(); ++i) {
        std::cout << "Output shape " << i << ": " << outputTensor.sizes()[i] << '\n';
    }

    auto flatOutputTensor = outputTensor.reshape({-1});

    const int outputSize = config.m_batch_size * config.m_model_output_size_backend;
    std::vector<float> outputData;

    // Copy the data to the outputData vector
    for (int i = 0; i < outputSize; ++i) {
        outputData.push_back(flatOutputTensor[i].item().toFloat());
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