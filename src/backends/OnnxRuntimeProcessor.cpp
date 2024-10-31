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

void OnnxRuntimeProcessor::process(AudioBufferF& input, AudioBufferF& output, std::shared_ptr<SessionElement> session) {
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
    }

    m_input_data.resize(m_inference_config.m_input_sizes.size());
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        m_input_data[i].resize(m_inference_config.m_input_sizes[i]);
        m_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                m_input_data[i].data(),
                m_input_data[i].size(),
                m_inference_config.m_model_input_shape_onnx[i].data(),
                m_inference_config.m_model_input_shape_onnx[i].size()
        ));
    }

    m_output_data.resize(m_inference_config.m_output_sizes.size());
    for (size_t i = 0; i < m_inference_config.m_output_sizes.size(); i++) {
        m_output_data[i].resize(m_inference_config.m_output_sizes[i]);
        m_outputs.emplace_back(Ort::Value::CreateTensor<float>(
                m_memory_info,
                m_output_data[i].data(),
                m_output_data[i].size(),
                m_inference_config.m_model_output_shape_onnx[i].data(),
                m_inference_config.m_model_output_shape_onnx[i].size()
        ));
    }

}

void OnnxRuntimeProcessor::Instance::prepare() {
    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        try {
            m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
        }
        catch (Ort::Exception &e) {
            std::cerr << e.what() << std::endl;
        }
    }
}

void OnnxRuntimeProcessor::Instance::process(AudioBufferF& input, AudioBufferF& output, std::shared_ptr<SessionElement> session) {
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        if (i != m_inference_config.m_index_audio_data[Input]) {
            for (size_t j = 0; j < m_input_data[i].size(); j++) {
                m_input_data[i][j] = session->m_pp_processor.m_inputs[i][j].load();
            }
        } else {
            for (size_t j = 0; j < m_input_data[i].size(); j++) {
                m_input_data[i][j] = input.get_sample(0, j);
            }
            // TODO avoid copy operation, but swapping does not work
            // m_input_data[i].swap_data(input.get_memory_block());
            // input.reset_channel_ptr();
        }
    }

    try {
        m_outputs = m_session->Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_input_names.size(), m_output_names.data(), m_output_names.size());
    }
    catch (Ort::Exception &e) {
        std::cerr << e.what() << std::endl;
    }

    for (size_t i = 0; i < m_outputs.size(); i++) {
        // TODO check why m_output_data cant be used
        //auto output_read_ptr = m_output_data[i].data();
        auto output_read_ptr = m_outputs[i].GetTensorMutableData<float>();

        if (i != m_inference_config.m_index_audio_data[Output]) {
            for (size_t j = 0; j < m_output_data[i].size(); j++) {
                session->m_pp_processor.m_outputs[i][j].store(output_read_ptr[j]);
            }
        } else {
            for (size_t j = 0; j < m_output_data[i].size(); j++) {
                output.get_memory_block()[j] = output_read_ptr[j];
            }
        }
    }
}

} // namespace anira