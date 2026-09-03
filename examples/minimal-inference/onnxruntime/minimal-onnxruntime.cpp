/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <anira/compat/v3_to_v2.h>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../../../extras/models/model_files.h"
#include "../../../include/anira/utils/Buffer.h"
#include "../../../include/anira/utils/MemoryBlock.h"

namespace {
// Render a tensor shape as "d0, d1, ...". std::vector has no operator<< of its own;
// relying on one only works when another dependency (e.g. libtorch) happens to pull
// one in, which is not the case for every backend configuration.
std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string result;
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i + 1 < shape.size()) { result += ", "; }
    }
    return result;
}
}  // namespace

void minimal_inference(anira::InferenceConfig m_inference_config) {
    std::cout << "Minimal OnnxRuntime example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << m_inference_config.get_model_path(anira::InferenceBackend::ONNX)
              << std::endl;

    // Define environment that holds logging state used by all other objects.
    // Note: One Env must be created before using any other OnnxRuntime functionality.
    Ort::Env m_env;
    // Define memory info for input and output tensors for CPU usage
    Ort::MemoryInfo m_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    // Define allocator
    Ort::AllocatorWithDefaultOptions m_ort_alloc;

    // Limit inference to one thread
    Ort::SessionOptions m_session_options;
    m_session_options.SetIntraOpNumThreads(1);

    // Load the model and create InferenceSession
#ifdef _WIN32
    std::wstring modelWideStr =
        std::wstring(m_inference_config.get_model_path(anira::InferenceBackend::ONNX).begin(),
                     m_inference_config.get_model_path(anira::InferenceBackend::ONNX).end());
    const wchar_t* modelWideCStr = modelWideStr.c_str();
    Ort::Session m_session(m_env, modelWideCStr, m_session_options);
#else
    Ort::Session m_session(m_env,
                           m_inference_config.get_model_path(anira::InferenceBackend::ONNX).c_str(),
                           Ort::SessionOptions{nullptr});
#endif

    // Fill an Buffer with some data
    anira::BufferF input(1, m_inference_config.get_tensor_input_size()[0]);
    for (int i = 0; i < m_inference_config.get_tensor_input_size()[0]; ++i) {
        input.set_sample(0, i, i * 0.000001f);
    }

    std::vector<anira::MemoryBlock<float>> m_input_data;
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    m_inputs.clear();
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        if (i != 0) {
            m_input_data[i].clear();
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                m_input_data[i].data(),
                m_input_data[i].size(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i]
                    .size()));
        } else {
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                input.data(),
                input.get_num_samples(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i]
                    .size()));
        }
    }

    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": ["
                  << shape_to_string(m_inputs[i].GetTensorTypeAndShapeInfo().GetShape()) << "]"
                  << std::endl;
    }

    // Get input and output names from model
    std::vector<Ort::AllocatedStringPtr> m_input_name;
    std::vector<Ort::AllocatedStringPtr> m_output_name;
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;

    m_input_names.resize(m_session.GetInputCount());
    m_output_names.resize(m_session.GetOutputCount());

    for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
        m_input_name.emplace_back(m_session.GetInputNameAllocated(i, m_ort_alloc));
        m_input_names[i] = m_input_name[i].get();
    }
    for (size_t i = 0; i < m_session.GetOutputCount(); ++i) {
        m_output_name.emplace_back(m_session.GetOutputNameAllocated(i, m_ort_alloc));
        m_output_names[i] = m_output_name[i].get();
    }

    try {
        m_outputs = m_session.Run(Ort::RunOptions{nullptr},
                                  m_input_names.data(),
                                  m_inputs.data(),
                                  m_input_names.size(),
                                  m_output_names.data(),
                                  m_output_names.size());
    } catch (Ort::Exception& e) { std::cerr << e.what() << std::endl; }

    for (int i = 0; i < m_outputs.size(); ++i) {
        std::cout << "Output shape " << i << ": ["
                  << shape_to_string(m_outputs[i].GetTensorTypeAndShapeInfo().GetShape()) << "]"
                  << std::endl;
    }

    std::vector<anira::MemoryBlock<float>> m_output_data;
    m_output_data.resize(m_outputs.size());

    for (size_t i = 0; i < m_outputs.size(); i++) {
        const auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();
        m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);

        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
            std::cout << "Output data [" << i << "][" << j << "]: " << output_read_ptr[j]
                      << std::endl;
            m_output_data[i][j] = output_read_ptr[j];
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
