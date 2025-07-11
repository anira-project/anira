/* ==========================================================================

Minimal LibTorch example from https://pytorch.org/tutorials/advanced/cpp_export.html
Licence: modified BSD

========================================================================== */

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

#include "../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../extras/models/cnn/CNNConfig.h"
#include "../../../extras/models/model-pool/SimpleGainConfig.h"
#include "../../../extras/models/model-pool/SimpleStereoGainConfig.h"

#include "../../../include/anira/utils/MemoryBlock.h"
#include "../../../include/anira/utils/Buffer.h"

// m_ prefix is not used to indicate member variables it is used to be compatible with code in the LibTorchProcessor class

void minimal_inference(anira::InferenceConfig m_inference_config) {
    std::cout << "Minimal LibTorch example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH) << std::endl;

    torch::set_num_threads(1);

    // Load model
    torch::jit::script::Module m_module;
    try {
        m_module = torch::jit::load(m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH));
    }
    catch (const c10::Error& e) {
        std::cerr << "[ERROR] error loading the model\n";
        std::cerr << e.what() << std::endl;
    }

    // Fill an Buffer with some data
    anira::BufferF input(1, m_inference_config.get_tensor_input_size()[m_inference_config.m_index_audio_data[anira::Input]]);
    for(int i = 0; i < m_inference_config.get_tensor_input_size()[m_inference_config.m_index_audio_data[anira::Input]]; ++i) {
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
        if (i != m_inference_config.m_index_audio_data[anira::Input]) {
            m_input_data[i].clear();
        } else {
            m_input_data[i].swap_data(input.get_memory_block());
            input.reset_channel_ptr();
        }
        m_inputs[i] = torch::from_blob(m_input_data[i].data(), m_inference_config.get_tensor_input_shape(anira::InferenceBackend::LIBTORCH)[i]);
    }


    // Get the shapes of the input tensors
    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": " << m_inputs[i].toTensor().sizes() << '\n';
    }

    // Execute inference
    c10::IValue m_outputs = m_module.forward(m_inputs);

    std::vector<anira::MemoryBlock<float>> m_output_data;

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    if(m_outputs.isTuple()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            std::cout << "Output size " << i << ": " << m_outputs.toTuple()->elements()[i].toTensor().sizes() << '\n';
        }
        m_output_data.resize(m_inference_config.get_tensor_output_shape().size());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                m_output_data[i][j] = m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j];
            }
        }
    } else if(m_outputs.isTensorList()) {
        std::cout << "Output is a tensor list" << std::endl;
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            std::cout << "Output size " << i << ": " << m_outputs.toTensorList().get(i).sizes() << '\n';
        }
        m_output_data.resize(m_inference_config.get_tensor_output_shape().size());
        for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
            m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                m_output_data[i][j] = m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j];
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
            std::cout << "Output data [" << i << "][" << j << "]: " << m_output_data[i][j] << std::endl;
        }
    }
}

int main(int argc, const char* argv[]) {

    std::vector<anira::InferenceConfig> models_to_inference = {hybridnn_config, cnn_config, rnn_config, gain_config, stereo_gain_config};

    for (int i = 0; i < models_to_inference.size(); ++i) {
        minimal_inference(models_to_inference[i]);
    }

    return 0;
}