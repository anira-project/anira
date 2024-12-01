/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <iostream>
#include <onnxruntime_cxx_api.h>

#include "../../../../extras/models/stateful-rnn/StatefulRNNConfig.h"
#include "../../../../extras/models/hybrid-nn/HybridNNConfig.h"
#include "../../../../extras/models/cnn/CNNConfig.h"
#include "../../../../extras/models/model-pool/SimpleGainConfig.h"
#include "../../../../extras/models/model-pool/SimpleStereoGainConfig.h"

#include "../../../../include/anira/utils/MemoryBlock.h"
#include "../../../../include/anira/utils/AudioBuffer.h"

void minimal_inference(anira::InferenceConfig m_inference_config) {

    std::cout << "Minimal OnnxRuntime example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << m_inference_config.get_model_path(anira::InferenceBackend::ONNX) << std::endl;

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
    std::wstring modelWideStr = std::wstring(m_inference_config.get_model_path(anira::InferenceBackend::ONNX).begin(), m_inference_config.get_model_path(anira::InferenceBackend::ONNX).end());
    const wchar_t* modelWideCStr = modelWideStr.c_str();
    Ort::Session m_session(m_env, modelWideCStr, m_session_options);
#else
    Ort::Session m_session(m_env, m_inference_config.get_model_path(anira::InferenceBackend::ONNX).c_str(), Ort::SessionOptions{ nullptr });
#endif

    // Fill an AudioBuffer with some data
    anira::AudioBufferF input(1, m_inference_config.m_input_sizes[m_inference_config.m_index_audio_data[anira::Input]]);
    for(int i = 0; i < m_inference_config.m_input_sizes[m_inference_config.m_index_audio_data[anira::Input]]; ++i) {
        input.set_sample(0, i, i * 0.000001f);
    }

    std::vector<anira::MemoryBlock<float>> m_input_data;
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    m_input_data.resize(m_inference_config.m_input_sizes.size());
    m_inputs.clear();
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        m_input_data[i].resize(m_inference_config.m_input_sizes[i]);
        if (i != m_inference_config.m_index_audio_data[anira::Input]) {
            m_input_data[i].clear();
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                m_input_data[i].data(),
                m_input_data[i].size(),
                m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[i].size()
            ));
        } else {
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                input.data(),
                input.get_num_samples(),
                m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                m_inference_config.get_input_shape(anira::InferenceBackend::ONNX)[i].size()
            ));
        }
    }

    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": [" << m_inputs[i].GetTensorTypeAndShapeInfo().GetShape() << "]" << std::endl;
    }

    // Get input and output names from model
    std::vector<Ort::AllocatedStringPtr> m_input_name;
    std::vector<Ort::AllocatedStringPtr> m_output_name;
    std::vector<const char *> m_input_names;
    std::vector<const char *> m_output_names;

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
        m_outputs = m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
    } catch (Ort::Exception &e) {
        std::cerr << e.what() << std::endl;
    }

    for (int i = 0; i < m_outputs.size(); ++i) {
        std::cout << "Output shape " << i << ": [" << m_outputs[i].GetTensorTypeAndShapeInfo().GetShape() << "]" << std::endl;
    }

    std::vector<anira::MemoryBlock<float>> m_output_data;
    m_output_data.resize(m_outputs.size());

    for (size_t i = 0; i < m_outputs.size(); i++) {
        const auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();
        m_output_data[i].resize(m_inference_config.m_output_sizes[i]);

        for (size_t j = 0; j < m_inference_config.m_output_sizes[i]; j++) {
            std::cout << "Output data [" << i << "][" << j << "]: " << output_read_ptr[j] << std::endl;
            m_output_data[i][j] = output_read_ptr[j];
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