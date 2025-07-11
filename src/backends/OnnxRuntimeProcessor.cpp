#include <anira/backends/OnnxRuntimeProcessor.h>
#include <anira/utils/Logger.h>

namespace anira {

OnnxRuntimeProcessor::OnnxRuntimeProcessor(InferenceConfig& inference_config) : BackendBase(inference_config)
{
    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

OnnxRuntimeProcessor::~OnnxRuntimeProcessor() {
}

void OnnxRuntimeProcessor::prepare() {
    for(auto& instance : m_instances) {
        instance->prepare();
    }
}

void OnnxRuntimeProcessor::process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session) {
    while (true) {
        for(auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output, session);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

OnnxRuntimeProcessor::Instance::Instance(InferenceConfig& inference_config) : m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
                                                                    m_inference_config(inference_config)
{
    m_session_options.SetIntraOpNumThreads(1);

    // Check if the model is binary
    if (m_inference_config.is_model_binary(anira::InferenceBackend::ONNX)) {
        const anira::ModelData* model_data = m_inference_config.get_model_data(anira::InferenceBackend::ONNX);
        assert(model_data && "Model data not found for binary model!");

        // Load model from binary data
        m_session = std::make_unique<Ort::Session>(m_env, model_data->m_data, model_data->m_size, m_session_options);
    } else {
        // Load model from file path
#ifdef _WIN32
        std::string modelpath_str = m_inference_config.get_model_path(anira::InferenceBackend::ONNX);
        std::wstring modelpath = std::wstring(modelpath_str.begin(), modelpath_str.end());
#else
        std::string modelpath = m_inference_config.get_model_path(anira::InferenceBackend::ONNX);
#endif
        m_session = std::make_unique<Ort::Session>(m_env, modelpath.c_str(), m_session_options);
    }
    
    m_input_names.resize(m_session->GetInputCount());
    m_output_names.resize(m_session->GetOutputCount());
    m_input_name.clear();
    m_output_name.clear();

    for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
        m_input_name.emplace_back(m_session->GetInputNameAllocated(i, m_ort_alloc));
        m_input_names[i] = m_input_name[i].get();
    }
    for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
        m_output_name.emplace_back(m_session->GetOutputNameAllocated(i, m_ort_alloc));
        m_output_names[i] = m_output_name[i].get();
    }

    m_input_data.resize(m_inference_config.get_tensor_input_shape().size());
    m_inputs.clear();
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        m_input_data[i].resize(m_inference_config.get_tensor_input_size()[i]);
        m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                m_input_data[i].data(),
                m_input_data[i].size(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].size()
        ));
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        try {
            m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
        } catch (Ort::Exception &e) {
            LOG_ERROR << e.what() << std::endl;
        }
    }
}

OnnxRuntimeProcessor::Instance::~Instance() {
    // Reseting the session here is very important otherwise new models might not be loaded correctly
    m_session.reset();
}

void OnnxRuntimeProcessor::Instance::prepare() {
    for (auto & i : m_input_data) {
        i.clear();
    }
}

void OnnxRuntimeProcessor::Instance::process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session) {
    for (size_t i = 0; i < m_inference_config.get_tensor_input_shape().size(); i++) {
        if (i != m_inference_config.m_index_audio_data[Input]) {
            for (size_t j = 0; j < m_input_data[i].size(); j++) {
                m_input_data[i][j] = session->m_pp_processor.get_input(i, j);
            }
        } else {
            m_inputs[i] = Ort::Value::CreateTensor<float>(
                    m_memory_info,
                    input.data(),
                    input.get_num_samples() * input.get_num_channels(),
                    m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].data(),
                    m_inference_config.get_tensor_input_shape(anira::InferenceBackend::ONNX)[i].size()
            );
        }
    }

    try {
        m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
    } catch (Ort::Exception &e) {
        LOG_ERROR << e.what() << std::endl;
    }

    for (size_t i = 0; i < m_outputs.size(); i++) {
        const auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();
        if (i != m_inference_config.m_index_audio_data[Output]) {
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                session->m_pp_processor.set_output(output_read_ptr[j], i, j);
            }
        } else {
            for (size_t j = 0; j < m_inference_config.get_tensor_output_size()[i]; j++) {
                output.get_memory_block()[j] = output_read_ptr[j];
            }
        }
    }
}

} // namespace anira
