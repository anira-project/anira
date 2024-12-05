#include <anira/InferenceConfig.h>

namespace anira {

InferenceConfig::InferenceConfig (
        std::vector<ModelData> model_data,
        std::vector<TensorShape> tensor_shape,
        float max_inference_time,
        unsigned int internal_latency,
        unsigned int warm_up,
        std::array<size_t, 2> index_audio_data,
        std::array<size_t, 2> num_audio_channels,
        bool session_exclusive_processor,
        unsigned int num_parallel_processors
#ifdef USE_CONTROLLED_BLOCKING
        , float wait_in_process_block
#endif
        ) :
        m_model_data(model_data),
        m_tensor_shape(tensor_shape),
        m_max_inference_time(max_inference_time),
        m_internal_latency(internal_latency),
        m_warm_up(warm_up),
        m_index_audio_data(index_audio_data),
        m_num_audio_channels(num_audio_channels),
        m_session_exclusive_processor(session_exclusive_processor),
        m_num_parallel_processors(num_parallel_processors)
#ifdef USE_CONTROLLED_BLOCKING
        , m_wait_in_process_block(wait_in_process_block)
#endif
{
    assert((m_tensor_shape.size() > 0 && "At least one tensor shape must be provided."));
    for (size_t i = 0; i < m_model_data.size(); ++i) {
        bool success = false;
        for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
            if (!m_tensor_shape[j].m_universal) {
                if (m_model_data[i].m_backend == m_tensor_shape[j].m_backend){
                    success = true;
                    break;
                }
            }
        }
        if (!success) {
            for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
                if (m_tensor_shape[j].m_universal) {
                    TensorShape tensor_shape = m_tensor_shape[j];
                    tensor_shape.m_backend = m_model_data[i].m_backend;
                    m_tensor_shape.push_back(tensor_shape);
                    break;
                } 
                assert((j < m_tensor_shape.size() - 1 && "No tensor shape provided for model."));
            }
        }
    }
    m_input_sizes.resize(m_tensor_shape[0].m_input_shape.size());
    for(int i = 0; i < m_tensor_shape[0].m_input_shape.size(); ++i) {
        m_input_sizes[i] = 1;
        for(int j = 0; j < m_tensor_shape[0].m_input_shape[i].size(); ++j) {
            m_input_sizes[i] *= (int) m_tensor_shape[0].m_input_shape[i][j];
        }
    }
    m_output_sizes.resize(m_tensor_shape[0].m_output_shape.size());
    for(int i = 0; i < m_tensor_shape[0].m_output_shape.size(); ++i) {
        m_output_sizes[i] = 1;
        for(int j = 0; j < m_tensor_shape[0].m_output_shape[i].size(); ++j) {
            m_output_sizes[i] *= (int) m_tensor_shape[0].m_output_shape[i][j];
        }
    }
    if (m_session_exclusive_processor) {
        m_num_parallel_processors = 1;
    }
    if (m_num_parallel_processors < 1) {
        m_num_parallel_processors = 1;
        std::cout << "[WARNING] Number of parellel processors must be at least 1. Setting to 1." << std::endl;
    }
}

void InferenceConfig::set_input_sizes(const std::vector<size_t>& input_sizes) {
    m_input_sizes = input_sizes;
}

void InferenceConfig::set_output_sizes(const std::vector<size_t>& output_sizes) {
    m_output_sizes = output_sizes;
}

std::string InferenceConfig::get_model_path(InferenceBackend backend) {
    for (int i = 0; i < m_model_data.size(); ++i) {
        if (m_model_data[i].m_backend == backend) {
            return std::string((char*) m_model_data[i].m_data, m_model_data[i].m_size);
        }
    }
    assert((false && "No model path found for backend."));
    return "";
}

TensorShapeList InferenceConfig::get_input_shape() {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_universal) {
            return m_tensor_shape[i].m_input_shape;
        }
    }
    return m_tensor_shape[0].m_input_shape;
}

TensorShapeList InferenceConfig::get_output_shape() {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_universal) {
            return m_tensor_shape[i].m_output_shape;
        }
    }
    return m_tensor_shape[0].m_output_shape;
}

TensorShapeList InferenceConfig::get_input_shape(InferenceBackend backend) {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_backend == backend) {
            for (int j = 0; j < m_tensor_shape[i].m_input_shape[0].size(); ++j) {
            }
            return m_tensor_shape[i].m_input_shape;
        }
    }
    assert((false && "No input shape found for backend."));
    return {};
}

TensorShapeList InferenceConfig::get_output_shape(InferenceBackend backend) {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_backend == backend) {
            return m_tensor_shape[i].m_output_shape;
        }
    }
    assert((false && "No output shape found for backend."));
    return {};
}

void InferenceConfig::set_model_path(const std::string& model_path, InferenceBackend backend) {
    for (int i = 0; i < m_model_data.size(); ++i) {
        if (m_model_data[i].m_backend == backend) {
            if (!m_model_data[i].m_is_binary) {
                free(m_model_data[i].m_data);
                m_model_data[i].m_data = malloc(model_path.size() * sizeof(char));
                memcpy(m_model_data[i].m_data, model_path.c_str(), model_path.size());
                m_model_data[i].m_size = model_path.size();
            }
            return;
        }
    }
    assert((false && "No model path found for backend."));
}

void InferenceConfig::set_input_shape(const TensorShapeList& input_shape, InferenceBackend backend) {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_backend == backend) {
            m_tensor_shape[i].m_input_shape = input_shape;
            return;
        }
    }
    assert((false && "No tensor shape found for backend."));
}

void InferenceConfig::set_output_shape(const TensorShapeList& output_shape, InferenceBackend backend) {
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        if (m_tensor_shape[i].m_backend == backend) {
            m_tensor_shape[i].m_output_shape = output_shape;
            return;
        }
    }
    assert((false && "No tensor shape found for backend."));
}

} // namespace anira