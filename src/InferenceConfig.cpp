#include <anira/InferenceConfig.h>
#include <anira/utils/Logger.h>

namespace anira {

InferenceConfig::InferenceConfig (
        std::vector<ModelData> model_data,
        std::vector<TensorShape> tensor_shape,
        ProcessingSpec processing_spec,
        float max_inference_time,
        unsigned int warm_up,
        bool session_exclusive_processor,
        unsigned int num_parallel_processors
#ifdef USE_CONTROLLED_BLOCKING
        , float wait_in_process_block
#endif
        ) :
        m_model_data(model_data),
        m_tensor_shape(tensor_shape),
        m_max_inference_time(max_inference_time),
        m_processing_spec(processing_spec),
        m_warm_up(warm_up),
        m_session_exclusive_processor(session_exclusive_processor),
        m_num_parallel_processors(num_parallel_processors)
#ifdef USE_CONTROLLED_BLOCKING
        , m_wait_in_process_block(wait_in_process_block)
#endif
{
    update_processing_spec();

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

TensorShapeList InferenceConfig::get_tensor_input_shape() const {
    TensorShapeList input_shapes;
    for (const auto& shape : m_tensor_shape) {
        if (shape.is_universal()) {
            return shape.m_tensor_input_shape; // Return universal input shape if available
        }
    }
    return m_tensor_shape[0].m_tensor_input_shape; // Fallback to the first tensor shape
}

TensorShapeList InferenceConfig::get_tensor_output_shape() const {
    TensorShapeList output_shapes;
    for (const auto& shape : m_tensor_shape) {
        if (shape.is_universal()) {
            return shape.m_tensor_output_shape; // Return universal output shape if available
        }
    }
    return m_tensor_shape[0].m_tensor_output_shape; // Fallback to the first tensor shape
}

TensorShapeList InferenceConfig::get_tensor_input_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_input_shape;
}

TensorShapeList InferenceConfig::get_tensor_output_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_output_shape;
}

std::vector<size_t> InferenceConfig::get_tensor_input_size() const {
    return m_processing_spec.m_tensor_input_size;
}

std::vector<size_t> InferenceConfig::get_tensor_output_size() const {
    return m_processing_spec.m_tensor_output_size;
}

std::vector<size_t> InferenceConfig::get_preprocess_input_channels() const {
    return m_processing_spec.m_preprocess_input_channels;
}

std::vector<size_t> InferenceConfig::get_postprocess_output_channels() const {
    return m_processing_spec.m_postprocess_output_channels;
}

std::vector<size_t> InferenceConfig::get_preprocess_input_size() const {
    return m_processing_spec.m_preprocess_input_size;
}

std::vector<size_t> InferenceConfig::get_postprocess_output_size() const {
    return m_processing_spec.m_postprocess_output_size;
}

std::vector<size_t> InferenceConfig::get_internal_latency() const {
    return m_processing_spec.m_internal_latency;
}

void InferenceConfig::set_tensor_input_shape(const TensorShapeList& input_shape) {
    for (TensorShape& shape : m_tensor_shape) {
        shape.m_tensor_input_shape = input_shape;
        m_processing_spec.m_tensor_input_size.clear();
        m_processing_spec.m_preprocess_input_channels.clear();
        m_processing_spec.m_preprocess_input_size.clear();
    }
    clear_processing_spec();
    update_processing_spec();
    return;
}

void InferenceConfig::set_tensor_output_shape(const TensorShapeList& output_shape) {
    for (TensorShape& shape : m_tensor_shape) {
        shape.m_tensor_output_shape = output_shape;
        m_processing_spec.m_tensor_output_size.clear();
        m_processing_spec.m_postprocess_output_channels.clear();
        m_processing_spec.m_postprocess_output_size.clear();
    }
    clear_processing_spec();
    update_processing_spec();
    return;
}

void InferenceConfig::set_tensor_input_shape(const TensorShapeList& input_shape, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) {
            shape.m_tensor_input_shape = input_shape;
        }
    }
    return;
}

void InferenceConfig::set_tensor_output_shape(const TensorShapeList& output_shape, InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) {
            shape.m_tensor_output_shape = output_shape;
        }
    }
    return;
}


void InferenceConfig::set_preprocess_input_channels(const std::vector<size_t>& input_channels) {
    m_processing_spec.m_preprocess_input_channels = input_channels;
    return;
}

void InferenceConfig::set_preprocess_output_channels(const std::vector<size_t>& output_channels) {
    m_processing_spec.m_postprocess_output_channels = output_channels;
    return;
}

void InferenceConfig::set_preprocess_input_size(const std::vector<size_t>& preprocess_input_size) {
    m_processing_spec.m_preprocess_input_size = preprocess_input_size;
    return;
}

void InferenceConfig::set_postprocess_output_size(const std::vector<size_t>& postprocess_output_size) {
    m_processing_spec.m_postprocess_output_size = postprocess_output_size;
    return;
}

void InferenceConfig::set_internal_latency(const std::vector<size_t>& internal_latency) {
    m_processing_spec.m_internal_latency = internal_latency;
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
    for (const TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) {
            return shape;
        }
    }
    for (const TensorShape& shape : m_tensor_shape) {
        if (shape.is_universal()) {
            return shape;
        }
    }
    LOG_ERROR << "No tensor shape found for backend: " << static_cast<int>(backend) << ". Returning the first tensor shape." << std::endl;
    return m_tensor_shape[0]; // Fallback to the first tensor shape
}

void InferenceConfig::clear_processing_spec() {
    m_processing_spec.m_preprocess_input_channels.clear();
    m_processing_spec.m_postprocess_output_channels.clear();
    m_processing_spec.m_preprocess_input_size.clear();
    m_processing_spec.m_postprocess_output_size.clear();
    m_processing_spec.m_internal_latency.clear();
    m_processing_spec.m_tensor_input_size.clear();
    m_processing_spec.m_tensor_output_size.clear();
}

void InferenceConfig::update_processing_spec() {
    assert((m_tensor_shape.size() > 0 && "At least one tensor shape must be provided."));
    for (size_t i = 0; i < m_model_data.size(); ++i) {
        bool success = false;
        for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
            if (!m_tensor_shape[j].is_universal()) {
                if (m_model_data[i].m_backend == m_tensor_shape[j].m_backend){
                    success = true;
                    break;
                }
            }
        }
        if (!success) {
            for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
                if (m_tensor_shape[j].is_universal()) {
                    TensorShape tensor_shape = m_tensor_shape[j];
                    tensor_shape.m_backend = m_model_data[i].m_backend;
                    m_tensor_shape.push_back(tensor_shape);
                    break;
                } 
                assert((j < m_tensor_shape.size() - 1 && "No tensor shape provided for model."));
            }
        }
    }

    m_processing_spec.m_tensor_input_size.clear();
    m_processing_spec.m_tensor_output_size.clear();
    for (int i = 0; i < m_tensor_shape.size(); ++i) {
        TensorShape& shape = m_tensor_shape[i];
        std::vector<size_t> input_size(m_tensor_shape[i].m_tensor_input_shape.size(), 1);
        std::vector<size_t> output_size(m_tensor_shape[i].m_tensor_output_shape.size(), 1);
        if (shape.m_tensor_input_shape.size() < 1) {
            LOG_ERROR << "No input shape provided for backend: " << static_cast<int>(shape.m_backend) << ". At least one input shape must be provided." << std::endl;
            throw std::invalid_argument("No input shape provided.");
        }
        if (shape.m_tensor_output_shape.size() < 1) {
            LOG_ERROR << "No output shape provided for backend: " << static_cast<int>(shape.m_backend) << ". At least one output shape must be provided." << std::endl;
            throw std::invalid_argument("No output shape provided.");
        }
        for (int j = 0; j < shape.m_tensor_input_shape.size(); ++j) {
            for (auto& dim : shape.m_tensor_input_shape[j]) {
                if (dim < 1) {
                    LOG_ERROR << "Invalid dimension in input shape: " << dim << ". Input dimensions must be positive." << std::endl;
                    throw std::invalid_argument("Invalid dimension in input shape.");
                }
                input_size[j] *= (size_t) dim;
            }
        }
        for (int j = 0; j < shape.m_tensor_output_shape.size(); ++j) {
            for (auto& dim : shape.m_tensor_output_shape[j]) {
                if (dim < 1) {
                    LOG_ERROR << "Invalid dimension in output shape: " << dim << ". Output dimensions must be positive." << std::endl;
                    throw std::invalid_argument("Invalid dimension in output shape.");
                }
                output_size[j] *= (size_t) dim;
            }
        }
        if (i == 0) {
            m_processing_spec.m_tensor_input_size = input_size;
            m_processing_spec.m_tensor_output_size = output_size;
            if (m_processing_spec.m_preprocess_input_channels.size() != input_size.size()) {
                m_processing_spec.m_preprocess_input_channels.clear();
                for (size_t j = 0; j < input_size.size(); ++j) {
                    m_processing_spec.m_preprocess_input_channels.push_back(1); // Default to 1 channel if not specified
                }
            }
            if (m_processing_spec.m_postprocess_output_channels.size() != output_size.size()) {
                m_processing_spec.m_postprocess_output_channels.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    m_processing_spec.m_postprocess_output_channels.push_back(1); // Default to 1 channel if not specified
                }
            }
            if (m_processing_spec.m_preprocess_input_size.size() != input_size.size()) {
                m_processing_spec.m_preprocess_input_size.clear();
                for (size_t j = 0; j < input_size.size(); ++j) {
                    size_t length = input_size[j];
                    if (m_processing_spec.m_preprocess_input_channels.size() > j) {
                        length /= m_processing_spec.m_preprocess_input_channels[j]; // Adjust length by number of channels
                    }
                    m_processing_spec.m_preprocess_input_size.push_back(length);
                }
            }
            if (m_processing_spec.m_postprocess_output_size.size() != output_size.size()) {
                m_processing_spec.m_postprocess_output_size.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    size_t length = output_size[j];
                    if (m_processing_spec.m_postprocess_output_channels.size() > j) {
                        length /= m_processing_spec.m_postprocess_output_channels[j]; // Adjust length by number of channels
                    }
                    m_processing_spec.m_postprocess_output_size.push_back(length);
                }
            }
            if (m_processing_spec.m_internal_latency.size() != output_size.size()) {
                m_processing_spec.m_internal_latency.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    m_processing_spec.m_internal_latency.push_back(0); // Default to 0 latency if not specified
                }
            }
        } else {
            if (m_processing_spec.m_tensor_input_size != input_size) {
                LOG_ERROR << "Input size mismatch for backend: " << static_cast<int>(shape.m_backend) << ". All backends must have the same input size." << std::endl;
                throw std::invalid_argument("Input size mismatch.");
            }
            if (m_processing_spec.m_tensor_output_size != output_size) {
                LOG_ERROR << "Output size mismatch for backend: " << static_cast<int>(shape.m_backend) << ". All backends must have the same output size." << std::endl;
                throw std::invalid_argument("Output size mismatch.");
            }
        }
    }
    if (m_processing_spec.m_preprocess_input_channels.size() != m_processing_spec.m_tensor_input_size.size()) {
        LOG_ERROR << "Preprocess input channels size mismatch. Must match the number of input tensors." << std::endl;
        throw std::invalid_argument("Preprocess input channels size mismatch.");
    }
    if (m_processing_spec.m_postprocess_output_channels.size() != m_processing_spec.m_tensor_output_size.size()) {
        LOG_ERROR << "Postprocess output channels size mismatch. Must match the number of output tensors." << std::endl;
        throw std::invalid_argument("Postprocess output channels size mismatch.");
    }
    if (m_processing_spec.m_preprocess_input_size.size() != m_processing_spec.m_tensor_input_size.size()) {
        LOG_ERROR << "Preprocess input size mismatch. Must match the number of input tensors." << std::endl;
        throw std::invalid_argument("Preprocess input size mismatch.");
    }
    if (m_processing_spec.m_postprocess_output_size.size() != m_processing_spec.m_tensor_output_size.size()) {
        LOG_ERROR << "Postprocess output size mismatch. Must match the number of output tensors." << std::endl;
        throw std::invalid_argument("Postprocess output size mismatch.");
    }
    if (m_processing_spec.m_internal_latency.size() != m_processing_spec.m_tensor_output_size.size()) {
        LOG_ERROR << "Internal latency size mismatch. Must match the number of output tensors." << std::endl;
        throw std::invalid_argument("Internal latency size mismatch.");
    }
    for (size_t i = 0; i < m_processing_spec.m_tensor_input_size.size(); ++i) {
        if (m_processing_spec.m_preprocess_input_size[i] == 0) {
            if (m_processing_spec.m_preprocess_input_channels[i] != 1) {
                LOG_ERROR << "For non-streamable tensors (preprocess_input_size[" << i << "] == 0), the number of channels must be 1." << std::endl;
                throw std::invalid_argument("Invalid number of channels for non-streamable tensor.");
            }
        }
        if (m_processing_spec.m_postprocess_output_size[i] == 0) {
            if (m_processing_spec.m_postprocess_output_channels[i] != 1) {
                LOG_ERROR << "For non-streamable tensors (postprocess_output_size[" << i << "] == 0), the number of channels must be 1." << std::endl;
                throw std::invalid_argument("Invalid number of channels for non-streamable tensor.");
            }
        }
    }
}

} // namespace anira
