#include <anira/backends/OnnxRuntimeProcessor.h>

namespace anira {

OnnxRuntimeProcessor::OnnxRuntimeProcessor(InferenceConfig& inference_config) : BackendBase(inference_config)
{
    for (size_t i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
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

void OnnxRuntimeProcessor::process(AudioBufferF& input, AudioBufferF& output) {
    while (true) {
        for(auto& instance : m_instances) {
            if (!(instance->m_processing.exchange(true))) {
                instance->process(input, output);
                instance->m_processing.exchange(false);
                return;
            }
        }
    }
}

OnnxRuntimeProcessor::Instance::Instance(InferenceConfig& inference_config) : m_inference_config(inference_config),
                                                                    m_memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{
    m_session_options.SetIntraOpNumThreads(1);

    if (m_session == nullptr) {
#ifdef _WIN32
        std::string modelpath_str = m_inference_config.m_model_path_onnx;
        std::wstring modelpath = std::wstring(modelpath_str.begin(), modelpath_str.end());
#else
        std::string modelpath = m_inference_config.m_model_path_onnx;
#endif
        m_session = std::make_unique<Ort::Session>(m_env, modelpath.c_str(), m_session_options);

        m_input_name = std::make_unique<Ort::AllocatedStringPtr>(m_session->GetInputNameAllocated(0, m_ort_alloc));
        m_output_name = std::make_unique<Ort::AllocatedStringPtr>(m_session->GetOutputNameAllocated(0, m_ort_alloc));
        m_input_names = {(char*) m_input_name->get()};
        m_output_names = {(char*) m_output_name->get()};

        m_input_size = m_inference_config.m_new_model_input_size;
        m_output_size = m_inference_config.m_new_model_output_size;
    }

    std::vector<int64_t> input_shape = m_inference_config.m_model_input_shape_onnx;
    m_input_data.resize(m_input_size, 0.0f);

    std::vector<int64_t> output_shape = m_inference_config.m_model_output_shape_onnx;
    m_output_data.resize(m_output_size, 0.0f);

    m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_input_data.data(),
            m_input_size,
            input_shape.data(),
            input_shape.size()
    ));

    m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
            m_memory_info,
            m_output_data.data(),
            m_output_size,
            output_shape.data(),
            output_shape.size()
    ));
}

void OnnxRuntimeProcessor::Instance::prepare() {
    if (m_inference_config.m_warm_up) {
        AudioBufferF input(1, m_input_size);
        AudioBufferF output(1, m_output_size);
        process(input, output);
    }
}

void OnnxRuntimeProcessor::Instance::process(AudioBufferF& input, AudioBufferF& output) {
    auto input_write_ptr = m_inputs[0].GetTensorMutableData<float>();
    auto input_read_ptr = input.get_read_pointer(0);
    for (size_t i = 0; i < m_input_size; i++) {
        input_write_ptr[i] = input_read_ptr[i];
    }

    try {
        m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
    }
    catch (Ort::Exception &e) {
        std::cerr << e.what() << std::endl;
    }

    auto output_write_ptr = output.get_write_pointer(0);
    auto output_read_ptr = m_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i < m_output_size; ++i) {
        output_write_ptr[i] = output_read_ptr[i];
    }
}

} // namespace anira