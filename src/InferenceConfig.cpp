#include <anira/InferenceConfig.h>
#include <anira/utils/Logger.h>

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
    update_tensor_shapes();

    if (m_session_exclusive_processor) {
        m_num_parallel_processors = 1;
    }
    if (m_num_parallel_processors < 1) {
        m_num_parallel_processors = 1;
        LOG_INFO << "[WARNING] Number of parellel processors must be at least 1. Setting to 1." << std::endl;
    }
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

// Check if the model is binary
bool InferenceConfig::is_model_binary(InferenceBackend backend) const {
    for (const auto& model : m_model_data) {
        if (model.m_backend == backend) {
            return model.m_is_binary;
        }
    }
    return false; // Default to false if no model is found
}
// Get binary model data
const ModelData* InferenceConfig::get_model_data(InferenceBackend backend) const {
    for (const auto& model : m_model_data) {
        if (model.m_backend == backend) {
            return &model;
        }
    }
    return nullptr; // No model data found
}

TensorShapeList InferenceConfig::get_tensor_input_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_input_shape;
}

TensorShapeList InferenceConfig::get_tensor_output_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_output_shape;
}

std::vector<size_t> InferenceConfig::get_tensor_input_size(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_input_size;
}

std::vector<size_t> InferenceConfig::get_tensor_output_size(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_output_size;
}

std::vector<size_t> InferenceConfig::get_preprocess_input_channels(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_preprocess_input_channels;
}

std::vector<size_t> InferenceConfig::get_preprocess_output_channels(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_preprocess_output_channels;
}

std::vector<size_t> InferenceConfig::get_preprocess_input_size(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_preprocess_input_size;
}

std::vector<size_t> InferenceConfig::get_postprocess_output_size(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_postprocess_output_size;
}

void InferenceConfig::set_tensor_input_shape(const TensorShapeList& input_shape, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_tensor_input_shape = input_shape;
            shape.m_tensor_input_size.clear();
            shape.m_preprocess_input_channels.clear();
            shape.m_preprocess_input_size.clear();
        }
    }
    update_tensor_shapes();
    return;
}

void InferenceConfig::set_tensor_output_shape(const TensorShapeList& output_shape, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_tensor_output_shape = output_shape;
            shape.m_tensor_output_size.clear();
            shape.m_preprocess_output_channels.clear();
            shape.m_postprocess_output_size.clear();
        }
    }
    update_tensor_shapes();
    return;
}


void InferenceConfig::set_preprocess_input_channels(const std::vector<size_t>& input_channels, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_preprocess_input_channels = input_channels;
            shape.m_preprocess_input_size.clear();
        }
    }
    update_tensor_shapes();
    return;
}

void InferenceConfig::set_preprocess_output_channels(const std::vector<size_t>& output_channels, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_preprocess_output_channels = output_channels;
            shape.m_postprocess_output_size.clear();
        }
    }
    update_tensor_shapes();
    return;
}

void InferenceConfig::set_preprocess_input_size(const std::vector<size_t>& preprocess_input_size, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_preprocess_input_size = preprocess_input_size;
        }
    }
    update_tensor_shapes();
    return;
}

void InferenceConfig::set_postprocess_output_size(const std::vector<size_t>& postprocess_output_size, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend || backend == InferenceBackend::UNIVERSAL) {
            shape.m_postprocess_output_size = postprocess_output_size;
        }
    }
    update_tensor_shapes();
    return;
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

const TensorShape& InferenceConfig::get_tensor_shape(InferenceBackend backend) const {
    if (backend == InferenceBackend::UNIVERSAL) {
        for (int i = 0; i < m_tensor_shape.size(); ++i) {
            if (m_tensor_shape[i].m_backend == InferenceBackend::UNIVERSAL) {
                return m_tensor_shape[i];
            }
        }
    } else {
        for (int i = 0; i < m_tensor_shape.size(); ++i) {
            if (m_tensor_shape[i].m_backend == backend) {
                return m_tensor_shape[i];
            }
        }
        assert((false && "No tensor shape found for backend."));
    }
    return m_tensor_shape[0]; // Fallback to the first tensor shape
}

void InferenceConfig::update_tensor_shapes() {
    assert((m_tensor_shape.size() > 0 && "At least one tensor shape must be provided."));
    for (size_t i = 0; i < m_model_data.size(); ++i) {
        bool success = false;
        for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
            if (m_tensor_shape[j].m_backend != InferenceBackend::UNIVERSAL) {
                if (m_model_data[i].m_backend == m_tensor_shape[j].m_backend){
                    success = true;
                    break;
                }
            }
        }
        if (!success) {
            for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
                if (m_tensor_shape[j].m_backend == InferenceBackend::UNIVERSAL) {
                    TensorShape tensor_shape = m_tensor_shape[j];
                    tensor_shape.m_backend = m_model_data[i].m_backend;
                    m_tensor_shape.push_back(tensor_shape);
                    break;
                } 
                assert((j < m_tensor_shape.size() - 1 && "No tensor shape provided for model."));
            }
        }
    }
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_tensor_input_shape.size() != shape.m_tensor_input_size.size()) {
            shape.m_tensor_input_size.clear();
            for (const auto& input_shape : shape.m_tensor_input_shape) {
                size_t tensor_input_size = 1;
                for (int64_t dim : input_shape) {
                    if (dim < 1) {
                        LOG_ERROR << "Invalid dimension in input shape: " << dim << ". Input dimensions must be positive." << std::endl;
                        throw std::invalid_argument("Invalid dimension in input shape.");
                    }
                    tensor_input_size *= (size_t) dim;
                }
                shape.m_tensor_input_size.push_back(tensor_input_size);
            }
        }
        if (shape.m_tensor_output_shape.size() != shape.m_tensor_output_size.size()) {
            shape.m_tensor_output_size.clear();
            for (const auto& output_shape : shape.m_tensor_output_shape) {
                size_t tensor_output_size = 1;
                for (int64_t dim : output_shape) {
                    if (dim < 1) {
                        LOG_ERROR << "Invalid dimension in output shape: " << dim << ". Output dimensions must be positive." << std::endl;
                        throw std::invalid_argument("Invalid dimension in output shape.");
                    }
                    tensor_output_size *= (size_t) dim;
                }
                shape.m_tensor_output_size.push_back(tensor_output_size);
            }
        }
        if (shape.m_preprocess_input_channels.size() != shape.m_tensor_input_size.size()) {
            shape.m_preprocess_input_channels.clear();
            for (size_t i = 0; i < shape.m_tensor_input_shape.size(); ++i) {
                shape.m_preprocess_input_channels.push_back(1); // Default to 1 channel if not specified
            }
        }
        if (shape.m_preprocess_output_channels.size() != shape.m_tensor_output_size.size()) {
            shape.m_preprocess_output_channels.clear();
            for (size_t i = 0; i < shape.m_tensor_output_shape.size(); ++i) {
                shape.m_preprocess_output_channels.push_back(1); // Default to 1 channel if not specified
            }
        }
        if (shape.m_preprocess_input_size.size() != shape.m_tensor_input_shape.size()) {
            shape.m_preprocess_input_size.clear();
            for (size_t i = 0; i < shape.m_tensor_input_shape.size(); ++i) {
                const auto& input_shape = shape.m_tensor_input_shape[i];
                const size_t num_channels = shape.m_preprocess_input_channels[i];
                size_t length = 1;
                for (int64_t dim : input_shape) {
                    if (dim < 1) {
                        LOG_ERROR << "Invalid dimension in input shape: " << dim << ". Input dimensions must be positive, when preprocess_input_size is not specified." << std::endl;
                        throw std::invalid_argument("Invalid dimension in input shape.");
                    }
                    length *= (size_t) dim;
                }
                length /= num_channels; // Adjust length by number of channels
                shape.m_preprocess_input_size.push_back(length);
            }
        }
        if (shape.m_postprocess_output_size.size() != shape.m_tensor_output_shape.size()) {
            shape.m_postprocess_output_size.clear();
            for (size_t i = 0; i < shape.m_tensor_output_shape.size(); ++i) {
                const auto& output_shape = shape.m_tensor_output_shape[i];
                const size_t num_channels = shape.m_preprocess_output_channels[i];
                size_t length = 1;
                for (int64_t dim : output_shape) {
                    if (dim < 1) {
                        LOG_ERROR << "Invalid dimension in output shape: " << dim << ". Output dimensions must be positive, when postprocess_output_size is not specified." << std::endl;
                        throw std::invalid_argument("Invalid dimension in output shape.");
                    }
                    length *= (size_t) dim;
                }
                length /= num_channels; // Adjust length by number of channels
                shape.m_postprocess_output_size.push_back(length);
            }
        }
    }
    assert((m_tensor_shape.size() > 0 && "At least one tensor shape must be provided."));
    assert((m_tensor_shape[0].m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
    assert((m_tensor_shape[0].m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
    assert((m_tensor_shape[0].m_tensor_input_shape.size() == m_tensor_shape[0].m_preprocess_input_channels.size() && "Input shape size must match input channels size."));
    assert((m_tensor_shape[0].m_tensor_output_shape.size() == m_tensor_shape[0].m_preprocess_output_channels.size() && "Output shape size must match output channels size."));
    assert((m_tensor_shape[0].m_preprocess_input_size.size() == m_tensor_shape[0].m_tensor_input_shape.size() && "Length for preprocessing must match input shape size."));
    assert((m_tensor_shape[0].m_postprocess_output_size.size() == m_tensor_shape[0].m_tensor_output_shape.size() && "Length for postprocessing must match output shape size."));
    assert((m_tensor_shape[0].m_tensor_input_shape.size() == m_tensor_shape[0].m_tensor_input_size.size() && "Input shape size must match tensor input size."));
    assert((m_tensor_shape[0].m_tensor_output_shape.size() == m_tensor_shape[0].m_tensor_output_size.size() && "Output shape size must match tensor output size."));
}

} // namespace anira
