/* ==========================================================================

Minimal OnnxRuntime example from https://onnxruntime.ai
Licence: MIT

========================================================================== */

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../../extras/models/model_files.h"
#include "../minimal_model.h"

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

void minimal_inference(const MinimalModel& info) {
    std::cout << "Minimal OnnxRuntime example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: " << info.m_path << std::endl;

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
    std::wstring modelWideStr = std::wstring(info.m_path.begin(), info.m_path.end());
    const wchar_t* modelWideCStr = modelWideStr.c_str();
    Ort::Session m_session(m_env, modelWideCStr, m_session_options);
#else
    Ort::Session m_session(m_env, info.m_path.c_str(), Ort::SessionOptions{nullptr});
#endif

    // Fill an Buffer with some data
    std::vector<float> input(info.m_input_sizes[0]);
    for (int i = 0; i < info.m_input_sizes[0]; ++i) { input[i] = i * 0.000001f; }

    std::vector<std::vector<float>> m_input_data;
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;

    m_input_data.resize(info.m_input_shapes.size());
    m_inputs.clear();
    for (size_t i = 0; i < info.m_input_shapes.size(); i++) {
        m_input_data[i].resize(info.m_input_sizes[i]);
        if (i != 0) {
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(m_memory_info,
                                                                  m_input_data[i].data(),
                                                                  m_input_data[i].size(),
                                                                  info.m_input_shapes[i].data(),
                                                                  info.m_input_shapes[i].size()));
        } else {
            m_inputs.emplace_back(Ort::Value::CreateTensor<float>(m_memory_info,
                                                                  input.data(),
                                                                  input.size(),
                                                                  info.m_input_shapes[i].data(),
                                                                  info.m_input_shapes[i].size()));
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

    std::vector<std::vector<float>> m_output_data;
    m_output_data.resize(m_outputs.size());

    for (size_t i = 0; i < m_outputs.size(); i++) {
        const auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();
        m_output_data[i].resize(info.m_output_sizes[i]);

        for (size_t j = 0; j < info.m_output_sizes[i]; j++) {
            std::cout << "Output data [" << i << "][" << j << "]: " << output_read_ptr[j]
                      << std::endl;
            m_output_data[i][j] = output_read_ptr[j];
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
        const std::optional<MinimalModel> model =
            minimal_model(model_json, ANIRA_ENGINE_ONNXRUNTIME);
        if (!model.has_value()) {
            std::cout << model_json << ": no entry for this engine" << std::endl;
            continue;
        }
        minimal_inference(*model);
    }

    return 0;
}
