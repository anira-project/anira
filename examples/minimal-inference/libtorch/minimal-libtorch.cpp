/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <anira/compat/v3_to_v2.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <iostream>
#include <memory>
#include <utility>

#include "../../../extras/models/model_files.h"
#include "../../../include/anira/utils/Buffer.h"
#include "../../../include/anira/utils/MemoryBlock.h"

// m_ prefix is not used to indicate member variables it is used to be compatible with code in the
// LibTorchProcessor class

void minimal_inference(anira::InferenceConfig m_inference_config) {
    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: "
              << m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH) << std::endl;

    torch::set_num_threads(1);

    // Load model
    torch::jit::script::Module m_module;
    try {
        m_module =
            torch::jit::load(m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH));
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] error loading the model\n";
        std::cerr << e.what() << std::endl;
    }

    // Fill a Buffer with some data
    anira::BufferF input(1, m_inference_config.get_tensor_input_size()[0]);
    for (int i = 0; i < m_inference_config.get_tensor_input_size()[0]; ++i) {
        input.set_sample(0, i, i * 0.000001f);
    }

    // Create IValue vector for input of interpreter
    std::vector<c10::IValue> m_inputs;
    std::vector<anira::MemoryBlock<float>> m_input_data;

    // Create input tensors
    m_inputs.resize(m_inference_config.get_tensor_input_shape().size());
    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        if (i != 0) {
            m_input_data[i].clear();
        } else {
            m_input_data[i].swap_data(input.get_memory_block());
            input.reset_channel_ptr();
        }
        m_inputs[i] = torch::from_blob(
            m_input_data[i].data(),
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[i]);
    }

    // Get the shapes of the input tensors
    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": " << m_inputs[i].toTensor().sizes() << '\n';
    }

    // Execute inference
    c10::IValue m_outputs = m_module.forward(m_inputs);

    std::vector<anira::MemoryBlock<float>> m_output_data;

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    if (m_outputs.isTuple()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            std::cout << "Output size " << i << ": "
                      << m_outputs.toTuple()->elements()[i].toTensor().sizes() << '\n';
        }
        m_output_data.resize(m_inference_config.get_tensor_output_shape().size());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                m_output_data[i][j] =
                    m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensorList()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            std::cout << "Output size " << i << ": " << m_outputs.toTensorList().get(i).sizes()
                      << '\n';
        }
        m_output_data.resize(m_inference_config.get_tensor_output_shape().size());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                m_output_data[i][j] =
                    m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j];
            }
        }
    } else if (m_outputs.isTensor()) {
        std::cout << "Output is a tensor" << std::endl;
        std::cout << "Output size: " << m_outputs.toTensor().sizes() << '\n';
        m_output_data.resize(1);
        m_output_data[0].resize(m_inference_config.get_tensor_output_size()[0]);
        for (size_t i = 0; i < m_inference_config.get_tensor_output_size()[0]; i++) {
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
    // The bundled models: a model file and a contract file each (extras/models/model_files.h),
    // loaded with the 3.x API and bridged to the 2.x InferenceConfig this example reads its
    // model path and tensor shapes from. The candidate list keeps the entries of this build's
    // engines only.
    const std::array<std::pair<const char*, const char*>, 5> models_to_inference = {{
        {k_hybridnn_model_json, k_hybridnn_contract_json},
        {k_cnn_model_json, k_cnn_contract_json},
        {k_rnn_model_json, k_rnn_contract_json},
        {k_gain_model_json, k_gain_contract_json},
        {k_stereo_gain_model_json, k_stereo_gain_contract_json},
    }};

    for (const auto& [model_json, contract_json] : models_to_inference) {
        const anira::ModelConfig model_config = anira::ModelConfig::from_file(model_json);
        const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
        minimal_inference(anira::v3compat::to_inference_config(model_config,
                                                               contract,
                                                               anira::v3compat::enabled_engines()));
    }

    return 0;
}
