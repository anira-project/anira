/* ==========================================================================

Minimal ExecuTorch example using the executorch::extension::Module C++ API.

Unlike the other minimal-inference examples this one is deliberately
standalone (it does not link anira): anira embeds its own copy of the
ExecuTorch runtime, and a second copy linked into the same process must stay
fully isolated from it — see this example's CMakeLists.txt.

Licence: Apache 2.0

========================================================================== */

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "executorch/extension/module/module.h"
#include "executorch/extension/tensor/tensor.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/evalue.h"

#define EXECUTORCH_MINIMAL_CHECK(x)                              \
    if ((x) != executorch::runtime::Error::Ok) {                 \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

static void print_shape(const char* label, const std::vector<executorch::aten::SizesType>& shape) {
    std::cout << label << " shape: [";
    for (size_t j = 0; j < shape.size(); ++j) {
        std::cout << shape[j];
        if (j < shape.size() - 1) { std::cout << ", "; }
    }
    std::cout << "]" << std::endl;
}

int main(int argc, const char* argv[]) {
    // The GuitarLSTM example model, exported ahead-of-time from the same PyTorch
    // weights as the LibTorch model (batched [256, 1, 150] -> [256, 1] interface). The
    // other minimal examples read the path and the shapes from the model file
    // (extras/models/hybrid-nn/hybridnn.model.json) through anira; this one cannot link
    // anira (see above), so it spells them out.
    const std::string model_path = ANIRA_EXTRAS_MODELS_DIR
        "/hybrid-nn/GuitarLSTM/pytorch-version/models/model_0/GuitarLSTM-executorch.pte";
    const std::vector<executorch::aten::SizesType> input_shape = {256, 1, 150};
    const std::vector<executorch::aten::SizesType> output_shape = {256, 1};

    std::cout << "Minimal ExecuTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << model_path << std::endl;

    // Load the .pte program and its 'forward' method
    executorch::extension::Module module(model_path);
    EXECUTORCH_MINIMAL_CHECK(module.load_forward());

    // Fill the input tensor's host memory with some data
    size_t input_size = 1;
    for (const auto dim : input_shape) { input_size *= static_cast<size_t>(dim); }
    std::vector<float> input_data(input_size);
    for (size_t i = 0; i < input_size; ++i) { input_data[i] = static_cast<float>(i) * 0.000001f; }
    const auto input_tensor = executorch::extension::from_blob(input_data.data(),
                                                               input_shape,
                                                               executorch::aten::ScalarType::Float);
    print_shape("Input", input_shape);

    // Execute inference
    const auto result = module.forward(*input_tensor);
    EXECUTORCH_MINIMAL_CHECK(result.error());

    // Read back and print the output data
    print_shape("Output", output_shape);
    size_t output_size = 1;
    for (const auto dim : output_shape) { output_size *= static_cast<size_t>(dim); }
    const float* output_data = (*result)[0].toTensor().const_data_ptr<float>();
    for (size_t j = 0; j < output_size; ++j) {
        std::cout << "Output data [" << j << "]: " << output_data[j] << std::endl;
    }

    return 0;
}
