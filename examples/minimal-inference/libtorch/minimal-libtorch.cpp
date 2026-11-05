/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "../../../extras/models/model_files.h"
#include "../minimal_model.h"

// m_ prefix is not used to indicate member variables it is used to be compatible with code in the
// LibTorch adapter (src/engines/LibTorchAdapter.cpp)

void minimal_inference(const MinimalModel& info) {
    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << info.m_path << std::endl;

    torch::set_num_threads(1);

    // Load model
    torch::jit::script::Module m_module;
    try {
        m_module = torch::jit::load(info.m_path);
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] error loading the model\n";
        std::cerr << e.what() << std::endl;
    }

    // Fill a Buffer with some data
    std::vector<float> input(info.m_input_sizes[0]);
    for (int i = 0; i < info.m_input_sizes[0]; ++i) { input[i] = i * 0.000001f; }

    // Create IValue vector for input of interpreter
    std::vector<c10::IValue> m_inputs;
    std::vector<std::vector<float>> m_input_data;

    // Create input tensors
    m_inputs.resize(info.m_input_shapes.size());
    m_input_data.resize(info.m_input_shapes.size());
    for (size_t i = 0; i < info.m_input_shapes.size(); i++) {
        m_input_data[i].resize(info.m_input_sizes[i]);
        if (i == 0) { m_input_data[i].swap(input); }
        m_inputs[i] = torch::from_blob(m_input_data[i].data(), info.m_input_shapes[i]);
    }

    // Get the shapes of the input tensors
    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": " << m_inputs[i].toTensor().sizes() << '\n';
    }

    // Execute inference
    c10::IValue m_outputs = m_module.forward(m_inputs);

    std::vector<std::vector<float>> m_output_data;

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    if (m_outputs.isTuple()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < info.m_output_shapes.size(); i++) {
            std::cout << "Output size " << i << ": "
                      << m_outputs.toTuple()->elements()[i].toTensor().sizes() << '\n';
        }
        m_output_data.resize(info.m_output_shapes.size());
        for (size_t i = 0; i < info.m_output_shapes.size(); i++) {
            m_output_data[i].resize(info.m_output_sizes[i]);
            for (size_t j = 0; j < info.m_output_sizes[i]; j++) {
                m_output_data[i][j] =
                    m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensorList()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < info.m_output_shapes.size(); i++) {
            std::cout << "Output size " << i << ": " << m_outputs.toTensorList().get(i).sizes()
                      << '\n';
        }
        m_output_data.resize(info.m_output_shapes.size());
        for (size_t i = 0; i < info.m_output_shapes.size(); i++) {
            m_output_data[i].resize(info.m_output_sizes[i]);
            for (size_t j = 0; j < info.m_output_sizes[i]; j++) {
                m_output_data[i][j] =
                    m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensor()) {
        std::cout << "Output is a tensor" << std::endl;
        std::cout << "Output size: " << m_outputs.toTensor().sizes() << '\n';
        m_output_data.resize(1);
        m_output_data[0].resize(info.m_output_sizes[0]);
        for (size_t i = 0; i < info.m_output_sizes[0]; i++) {
            m_output_data[0][i] = m_outputs.toTensor().view({-1}).data_ptr<float>()[i];
        }
    }

    // Print output data
    for (int i = 0; i < m_output_data.size(); i++) {
        for (int j = 0; j < m_output_data[i].size(); j++) {
            std::cout << "Output data [" << i << "][" << j << "]: " << m_output_data[i][j]
                      << std::endl;
        }
    }
}

int main(int argc, const char* argv[]) {
    // The bundled model files (extras/models/model_files.h), read back with the 3.x API
    // (minimal_model.h): the path of the entry for this example's engine and the shapes the
    // engine is fed. A file without an entry for the engine is skipped.
    const std::array<const char*, 5> models_to_inference = {
        k_hybridnn_model_json,
        k_cnn_model_json,
        k_rnn_model_json,
        k_gain_model_json,
        k_stereo_gain_model_json,
    };

    for (const char* model_json : models_to_inference) {
        const std::optional<MinimalModel> model = minimal_model(model_json, ANIRA_ENGINE_LIBTORCH);
        if (!model.has_value()) {
            std::cout << model_json << ": no entry for this engine" << std::endl;
            continue;
        }
        minimal_inference(*model);
    }

    return 0;
}
