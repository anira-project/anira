#include <anira/InferenceConfig.h>
#include <anira/backends/BackendBase.h>
#include <anira/backends/TFLiteProcessor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <tensorflow/lite/core/c/c_api.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <comdef.h>
#endif

namespace anira {

TFLiteProcessor::TFLiteProcessor(InferenceConfig& inference_config)
    : BackendBase(inference_config) {
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

TFLiteProcessor::~TFLiteProcessor() = default;

void TFLiteProcessor::prepare() {
    for (auto& instance : m_instances) { instance->prepare(); }
}

void TFLiteProcessor::process(std::vector<BufferF>& input,
                              std::vector<BufferF>& output,
                              std::shared_ptr<SessionElement> session) {
    while (true) {
        for (auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output, session);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

TFLiteProcessor::Instance::Instance(InferenceConfig& inference_config)
    : m_inference_config(inference_config) {
    if (inference_config.is_model_binary(anira::InferenceBackend::TFLITE)) {
        const anira::ModelData* model_data =
            m_inference_config.get_model_data(anira::InferenceBackend::TFLITE);
        assert(model_data && "Model data not found for binary model!");
        m_model = TfLiteModelCreate(model_data->m_data, model_data->m_size);
    } else {
        std::string const modelpath =
            m_inference_config.get_model_path(anira::InferenceBackend::TFLITE);
        m_model = TfLiteModelCreateFromFile(modelpath.c_str());
    }

    m_options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(m_options, 1);
    m_interpreter = TfLiteInterpreterCreate(m_model, m_options);

    // This is necessary when we have dynamic input shapes, it should be done before allocating
    // tensors obviously
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        std::vector<int> input_shape;
        std::vector<int64_t> const input_shape64 =
            m_inference_config.get_tensor_input_shape(anira::InferenceBackend::TFLITE)[i];
        input_shape.reserve(input_shape64.size());
        for (long j : input_shape64) { input_shape.push_back((int)j); }
        TfLiteInterpreterResizeInputTensor(m_interpreter,
                                           static_cast<int32_t>(i),
                                           input_shape.data(),
                                           static_cast<int32_t>(input_shape.size()));
    }

    TfLiteInterpreterAllocateTensors(m_interpreter);

    m_inputs.resize(m_inference_config.get_tensor_input_shape().size());
    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        m_inputs[i] = TfLiteInterpreterGetInputTensor(m_interpreter, static_cast<int32_t>(i));
    }

    m_outputs.resize(m_inference_config.get_tensor_output_shape().size());
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
        m_outputs[i] = TfLiteInterpreterGetOutputTensor(m_interpreter, static_cast<int32_t>(i));
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        TfLiteInterpreterInvoke(m_interpreter);
    }
}

TFLiteProcessor::Instance::~Instance() {
    TfLiteInterpreterDelete(m_interpreter);
    TfLiteInterpreterOptionsDelete(m_options);
    TfLiteModelDelete(m_model);
}

void TFLiteProcessor::Instance::prepare() {
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].clear();
    }
}

void TFLiteProcessor::Instance::process(std::vector<BufferF>& input,
                                        std::vector<BufferF>& output,
                                        const std::shared_ptr<SessionElement>&) {
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].swap_data(input[i].get_memory_block());
        input[i].reset_channel_ptr();
        // TODO: Check if we can find a solution to avoid copying the data
        TfLiteTensorCopyFromBuffer(m_inputs[i],
                                   m_input_data[i].data(),
                                   m_inference_config.get_tensor_input_size()[i] * sizeof(float));
    }

    // Run inference
    TfLiteInterpreterInvoke(m_interpreter);

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    for (size_t i = 0; i < m_inference_config.get_tensor_output_shape().size(); i++) {
        auto* output_read_ptr = (float*)TfLiteTensorData(m_outputs[i]);
        for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
            output[i].get_memory_block()[j] = output_read_ptr[j];
        }
    }
}

}  // namespace anira