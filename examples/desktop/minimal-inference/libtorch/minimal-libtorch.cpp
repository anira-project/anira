/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

#include "../../../../extras/desktop/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../../extras/desktop/models/hybrid-nn/HybridNNConfig.h"
#include "../../../../extras/desktop/models/cnn/CNNConfig.h"

void minimal_inference(anira::InferenceConfig config) {
    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << config.m_model_path_torch << std::endl;

    std::string omp_num_threads = "OMP_NUM_THREADS=1";
    std::string mkl_num_threads = "MKL_NUM_THREADS=1";

#if WIN32
    _putenv(omp_num_threads.data());
    _putenv(mkl_num_threads.data());
#else
    putenv(omp_num_threads.data());
    putenv(mkl_num_threads.data());
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
    std::vector<float> input_data;
    for (int i = 0; i < config.m_new_model_input_size; i++) {
        input_data.push_back(i * 0.000001f);
    }

    // Create input tensor object from input data values and reshape
    std::vector<int64_t> shape = config.m_model_input_shape_torch;
    torch::Tensor input_tensor = torch::from_blob(input_data.data(), shape);

    for (int i = 0; i < input_tensor.sizes().size(); ++i) {
        std::cout << "Input shape " << i << ": " << input_tensor.sizes()[i] << '\n';
    }

    // Create IValue vector for input of interpreter
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Execute inference
    torch::Tensor output_tensor = module.forward(inputs).toTensor();

    for (int i = 0; i < output_tensor.sizes().size(); ++i) {
        std::cout << "Output shape " << i << ": " << output_tensor.sizes()[i] << '\n';
    }

    // Flatten the output tensor
    output_tensor = output_tensor.view({-1});

    std::vector<float> output_data;

    // Copy the data to the output_data vector
    for (int i = 0; i < config.m_new_model_output_size; ++i) {
        output_data.push_back(output_tensor[i].item().toFloat());
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