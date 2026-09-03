/* ==========================================================================

Minimal TensorFlow Lite example from https://www.tensorflow.org/lite/guide/inference
Licence: Apache 2.0

========================================================================== */

#include <anira/compat/v3_to_v2.h>
#include <tensorflow/lite/c_api.h>

#include <array>
#include <cstdio>
#include <iostream>
#include <utility>

#include "../../../extras/models/model_files.h"
#include "../../../include/anira/utils/Buffer.h"
#include "../../../include/anira/utils/MemoryBlock.h"

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x)) {                                                  \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

void minimal_inference(anira::InferenceConfig m_inference_config) {
    std::cout << "Minimal TensorFlow-Lite example:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Using model: "
              << m_inference_config.get_model_path(anira::InferenceBackend::TFLITE) << std::endl;

    // Load model
    TfLiteModel* m_model;
    m_model = TfLiteModelCreateFromFile(
        m_inference_config.get_model_path(anira::InferenceBackend::TFLITE).c_str());

    // Create the interpreter
    TfLiteInterpreterOptions* m_options;
    TfLiteInterpreter* m_interpreter;
    m_options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m_options, 1);
    m_interpreter = TfLiteInterpreterCreate(m_model, m_options);

    // This is necessary when we have dynamic input shapes, it should be done before allocating
    // tensors obviously
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        std::vector<int> input_shape;
        std::vector<int64_t> input_shape64 =
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[i];
        for (size_t j = 0; j < input_shape64.size(); j++) {
            input_shape.push_back((int)input_shape64[j]);
        }
        TfLiteInterpreterResizeInputTensor(m_interpreter,
                                           i,
                                           input_shape.data(),
                                           static_cast<int32_t>(input_shape.size()));
    }

    // Allocate memory for all tensors
    TfLiteInterpreterAllocateTensors(m_interpreter);

    // Fill an Buffer with some data
    anira::BufferF input(1, m_inference_config.get_tensor_input_size()[0]);
    for (int i = 0; i < m_inference_config.get_tensor_input_size()[0]; ++i) {
        input.set_sample(0, i, i * 0.000001f);
    }

    // Get input tensors and prepare input data
    std::vector<TfLiteTensor*> m_inputs;
    std::vector<anira::MemoryBlock<float>> m_input_data;

    m_inputs.resize(m_inference_config.get_tensor_input_shape().size());
    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        m_inputs[i] = TfLiteInterpreterGetInputTensor(m_interpreter, i);
        if (i != 0) {
            m_input_data[i].clear();
        } else {
            m_input_data[i].swap_data(input.get_memory_block());
            input.reset_channel_ptr();
        }
        TfLiteTensorCopyFromBuffer(m_inputs[i],
                                   m_input_data[i].data(),
                                   m_inference_config.get_tensor_input_size()[i] * sizeof(float));
    }

    for (int i = 0; i < m_inputs.size(); ++i) {
        std::cout << "Input shape " << i << ": [";
        for (int j = 0; j < TfLiteTensorNumDims(m_inputs[i]); ++j) {
            std::cout << TfLiteTensorDim(m_inputs[i], j);
            if (j < TfLiteTensorNumDims(m_inputs[i]) - 1) { std::cout << ", "; }
        }
        std::cout << "]" << std::endl;
    }

    // Execute inference
    TfLiteInterpreterInvoke(m_interpreter);

    // Get output tensors
    std::vector<const TfLiteTensor*> m_outputs;

    m_outputs.resize(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
        m_outputs[i] = TfLiteInterpreterGetOutputTensor(m_interpreter, i);
    }

    for (int i = 0; i < m_outputs.size(); ++i) {
        std::cout << "Output shape " << i << ": [";
        for (int j = 0; j < TfLiteTensorNumDims(m_outputs[i]); ++j) {
            std::cout << TfLiteTensorDim(m_outputs[i], j);
            if (j < TfLiteTensorNumDims(m_outputs[i]) - 1) { std::cout << ", "; }
        }
        std::cout << "]" << std::endl;
    }

    // Extract the output tensor data
    std::vector<anira::MemoryBlock<float>> m_output_data;
    m_output_data.resize(m_inference_config.get_tensor_output_shape().size());

    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
        float* output_data = (float*)TfLiteTensorData(m_outputs[i]);
        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
            m_output_data[i].resize(m_inference_config.get_tensor_output_size()[i]);
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                m_output_data[i][j] = output_data[j];
            }
        }
    }

    // Print output data
    for (int i = 0; i < m_output_data.size(); i++) {
        for (int j = 0; j < m_output_data[i].size(); j++) {
            std::cout << "Output data [" << i << "][" << j << "]: " << m_output_data[i][j]
                      << std::endl;
        }
    }

    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(m_interpreter);
    TfLiteInterpreterOptionsDelete(m_options);
    TfLiteModelDelete(m_model);
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
