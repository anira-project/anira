#include <anira/backends/LibTorchProcessor.h>
#include <anira/utils/Logger.h>

namespace anira {

LibtorchProcessor::LibtorchProcessor(InferenceConfig& inference_config) : BackendBase(inference_config) {
    torch::set_num_threads(1);

    for (unsigned int i = 0; i < m_inference_config.m_num_parallel_processors; ++i) {
        m_instances.emplace_back(std::make_shared<Instance>(m_inference_config));
    }
}

LibtorchProcessor::~LibtorchProcessor() {
}

void LibtorchProcessor::prepare() {
    for(auto& instance : m_instances) {
        instance->prepare();
    }
}

void LibtorchProcessor::process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session) { 
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

LibtorchProcessor::Instance::Instance(InferenceConfig& inference_config) : m_inference_config(inference_config) {
    try {
        m_module = torch::jit::load(m_inference_config.get_model_path(anira::InferenceBackend::LIBTORCH));
    }
    catch (const c10::Error& e) {
        LOG_ERROR << "[ERROR] error loading the model\n";
        LOG_ERROR << e.what() << std::endl;
    }
    m_inputs.resize(m_inference_config.m_input_sizes.size());
    m_input_data.resize(m_inference_config.m_input_sizes.size());
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        m_input_data[i].resize(m_inference_config.m_input_sizes[i]);
        m_inputs[i] = torch::from_blob(m_input_data[i].data(), m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[i]);
    }

    for (size_t i = 0; i < m_inference_config.m_warm_up; i++) {
        m_outputs = m_module.forward(m_inputs);
    }
}

void LibtorchProcessor::Instance::prepare() {
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        m_input_data[i].clear();
    }
}

void LibtorchProcessor::Instance::process(BufferF& input, BufferF& output, std::shared_ptr<SessionElement> session) {
    for (size_t i = 0; i < m_inference_config.m_input_sizes.size(); i++) {
        if (i != m_inference_config.m_index_audio_data[Input]) {
            for (size_t j = 0; j < m_input_data[i].size(); j++) {
                m_input_data[i][j] = session->m_pp_processor.get_input(i, j);
            }
        } else {
            m_input_data[i].swap_data(input.get_memory_block());
            input.reset_channel_ptr();
        }
        // This is necessary because the tensor data pointers seem to change from inference to inference
        m_inputs[i] = torch::from_blob(m_input_data[i].data(), m_inference_config.get_input_shape(anira::InferenceBackend::LIBTORCH)[i]);
    }

    // Run inference
    m_outputs = m_module.forward(m_inputs);

    // We need to copy the data because we cannot access the data pointer ref of the tensor directly
    if(m_outputs.isTuple()) {
        for (size_t i = 0; i < m_inference_config.m_output_sizes.size(); i++) {
            if (i != m_inference_config.m_index_audio_data[Output]) {
                for (size_t j = 0; j < m_inference_config.m_output_sizes[i]; j++) {
                    session->m_pp_processor.set_output(m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j], i, j);
                }
            } else {
                for (size_t j = 0; j < m_inference_config.m_output_sizes[i]; j++) {
                    output.get_memory_block()[j] = m_outputs.toTuple()->elements()[i].toTensor().view({-1}).data_ptr<float>()[j];
                }
            }
        }
    } else if(m_outputs.isTensorList()) {
        for (size_t i = 0; i < m_inference_config.m_output_sizes.size(); i++) {
            if (i != m_inference_config.m_index_audio_data[Output]) {
                for (size_t j = 0; j < m_inference_config.m_output_sizes[i]; j++) {
                    session->m_pp_processor.set_output(m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j], i, j);
                }
            } else {
                for (size_t j = 0; j < m_inference_config.m_output_sizes[i]; j++) {
                    output.get_memory_block()[j] = m_outputs.toTensorList().get(i).view({-1}).data_ptr<float>()[j];
                }
            }
        }
    } else if (m_outputs.isTensor()) {
        for (size_t i = 0; i < m_inference_config.m_output_sizes[0]; i++) {
            output.get_memory_block()[i] = m_outputs.toTensor().view({-1}).data_ptr<float>()[i];
        }
    }
}

} // namespace anira