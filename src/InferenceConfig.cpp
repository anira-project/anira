#include <anira/InferenceConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace anira {

unsigned int InferenceConfig::Defaults::m_num_parallel_processors =
    (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2 : 1;

InferenceConfig::InferenceConfig(std::vector<ModelData> model_data,
                                 std::vector<TensorShape> tensor_shape,
                                 ProcessingSpec processing_spec,
                                 float max_inference_time,
                                 unsigned int warm_up,
                                 bool session_exclusive_processor,
                                 float blocking_ratio,
                                 unsigned int num_parallel_processors)
    : m_model_data(std::move(std::move(model_data)))
    , m_tensor_shape(std::move(std::move(tensor_shape)))
    , m_max_inference_time(max_inference_time)
    , m_processing_spec(std::move(std::move(processing_spec)))
    , m_warm_up(warm_up)
    , m_session_exclusive_processor(session_exclusive_processor)
    , m_blocking_ratio(blocking_ratio)
    , m_num_parallel_processors(num_parallel_processors) {
    if (m_max_inference_time <= 0.f) {
        throw std::invalid_argument("max_inference_time must be greater than 0, got " +
                                    std::to_string(m_max_inference_time));
    }

    update_processing_spec();

    if (m_session_exclusive_processor) { m_num_parallel_processors = 1; }
    if (m_num_parallel_processors < 1) {
        m_num_parallel_processors = 1;
        ANIRA_LOG_WARNING(log_group::k_config,
                          "Number of parellel processors must be at least 1. Setting to 1.");
    }
}

std::string InferenceConfig::get_model_path(InferenceBackend backend) {
    for (auto& i : m_model_data) {
        if (i.m_backend == backend) { return {(char*)i.m_data, i.m_size}; }
    }
    assert((false && "No model path found for backend."));
    return "";
}

std::string InferenceConfig::get_model_function(InferenceBackend backend) const {
    for (const auto& model : m_model_data) {
        if (model.m_backend == backend) { return model.m_model_function; }
    }
    return "";  // Return empty string if no model function is found
}

// Check if the model is binary
bool InferenceConfig::is_model_binary(InferenceBackend backend) const {
    for (const auto& model : m_model_data) {
        if (model.m_backend == backend) { return model.m_is_binary; }
    }
    return false;  // Default to false if no model is found
}
// Get binary model data
const ModelData* InferenceConfig::get_model_data(InferenceBackend backend) const {
    for (const auto& model : m_model_data) {
        if (model.m_backend == backend) { return &model; }
    }
    return nullptr;  // No model data found
}

const TensorShapeList& InferenceConfig::get_tensor_input_shape() const {
    for (const auto& shape : m_tensor_shape) {
        if (shape.is_universal()) {
            return shape.m_tensor_input_shape;  // Return universal input shape if available
        }
    }
    return m_tensor_shape[0].m_tensor_input_shape;  // Fallback to the first tensor shape
}

const TensorShapeList& InferenceConfig::get_tensor_output_shape() const {
    for (const auto& shape : m_tensor_shape) {
        if (shape.is_universal()) {
            return shape.m_tensor_output_shape;  // Return universal output shape if available
        }
    }
    return m_tensor_shape[0].m_tensor_output_shape;  // Fallback to the first tensor shape
}

const TensorShapeList& InferenceConfig::get_tensor_input_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_input_shape;
}

const TensorShapeList& InferenceConfig::get_tensor_output_shape(InferenceBackend backend) const {
    return get_tensor_shape(backend).m_tensor_output_shape;
}

const std::vector<size_t>& InferenceConfig::get_tensor_input_size() const {
    return m_processing_spec.m_tensor_input_size;
}

const std::vector<size_t>& InferenceConfig::get_tensor_output_size() const {
    return m_processing_spec.m_tensor_output_size;
}

const std::vector<size_t>& InferenceConfig::get_preprocess_input_channels() const {
    return m_processing_spec.m_preprocess_input_channels;
}

const std::vector<size_t>& InferenceConfig::get_postprocess_output_channels() const {
    return m_processing_spec.m_postprocess_output_channels;
}

const std::vector<size_t>& InferenceConfig::get_preprocess_input_size() const {
    return m_processing_spec.m_preprocess_input_size;
}

const std::vector<size_t>& InferenceConfig::get_postprocess_output_size() const {
    return m_processing_spec.m_postprocess_output_size;
}

const std::vector<size_t>& InferenceConfig::get_internal_model_latency() const {
    return m_processing_spec.m_internal_model_latency;
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

void InferenceConfig::set_tensor_input_shape(const TensorShapeList& input_shape,
                                             InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) { shape.m_tensor_input_shape = input_shape; }
    }
    return;
}

void InferenceConfig::set_tensor_output_shape(const TensorShapeList& output_shape,
                                              InferenceBackend backend) {
    for (TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) { shape.m_tensor_output_shape = output_shape; }
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

void InferenceConfig::set_postprocess_output_size(
    const std::vector<size_t>& postprocess_output_size) {
    m_processing_spec.m_postprocess_output_size = postprocess_output_size;
    return;
}

void InferenceConfig::set_internal_model_latency(
    const std::vector<size_t>& internal_model_latency) {
    m_processing_spec.m_internal_model_latency = internal_model_latency;
    return;
}

void InferenceConfig::set_model_path(const std::string& model_path, InferenceBackend backend) {
    for (auto& i : m_model_data) {
        if (i.m_backend == backend) {
            if (!i.m_is_binary) {
                free(i.m_data);
                i.m_data = malloc(model_path.size() * sizeof(char));
                std::memcpy(i.m_data, model_path.c_str(), model_path.size());
                i.m_size = model_path.size();
            }
            return;
        }
    }
    assert((false && "No model path found for backend."));
}

const TensorShape& InferenceConfig::get_tensor_shape(InferenceBackend backend) const {
    for (const TensorShape& shape : m_tensor_shape) {
        if (shape.m_backend == backend) { return shape; }
    }
    for (const TensorShape& shape : m_tensor_shape) {
        if (shape.is_universal()) { return shape; }
    }
    ANIRA_LOG_ERROR(log_group::k_config,
                    "No tensor shape found for backend: %d. Returning the first tensor shape.",
                    static_cast<int>(backend));
    return m_tensor_shape[0];  // Fallback to the first tensor shape
}

void InferenceConfig::clear_processing_spec() {
    m_processing_spec.m_preprocess_input_channels.clear();
    m_processing_spec.m_postprocess_output_channels.clear();
    m_processing_spec.m_preprocess_input_size.clear();
    m_processing_spec.m_postprocess_output_size.clear();
    m_processing_spec.m_internal_model_latency.clear();
    m_processing_spec.m_tensor_input_size.clear();
    m_processing_spec.m_tensor_output_size.clear();
}

void InferenceConfig::update_processing_spec() {
    assert((m_tensor_shape.size() > 0 && "At least one tensor shape must be provided."));
    for (auto& i : m_model_data) {
        bool success = false;
        for (auto& j : m_tensor_shape) {
            if (!j.is_universal()) {
                if (i.m_backend == j.m_backend) {
                    success = true;
                    break;
                }
            }
        }
        if (!success) {
            for (size_t j = 0; j < m_tensor_shape.size(); ++j) {
                if (m_tensor_shape[j].is_universal()) {
                    TensorShape tensor_shape = m_tensor_shape[j];
                    tensor_shape.m_backend = i.m_backend;
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
            throw std::invalid_argument("no input shape provided for backend " +
                                        std::to_string(static_cast<int>(shape.m_backend)) +
                                        "; at least one input shape is required");
        }
        if (shape.m_tensor_output_shape.size() < 1) {
            throw std::invalid_argument("no output shape provided for backend " +
                                        std::to_string(static_cast<int>(shape.m_backend)) +
                                        "; at least one output shape is required");
        }
        for (int j = 0; j < shape.m_tensor_input_shape.size(); ++j) {
            for (auto& dim : shape.m_tensor_input_shape[j]) {
                if (dim < 1) {
                    throw std::invalid_argument("invalid dimension " + std::to_string(dim) +
                                                " in input shape " + std::to_string(j) +
                                                "; dimensions must be positive");
                }
                input_size[j] *= (size_t)dim;
            }
        }
        for (int j = 0; j < shape.m_tensor_output_shape.size(); ++j) {
            for (auto& dim : shape.m_tensor_output_shape[j]) {
                if (dim < 1) {
                    throw std::invalid_argument("invalid dimension " + std::to_string(dim) +
                                                " in output shape " + std::to_string(j) +
                                                "; dimensions must be positive");
                }
                output_size[j] *= (size_t)dim;
            }
        }
        if (i == 0) {
            m_processing_spec.m_tensor_input_size = input_size;
            m_processing_spec.m_tensor_output_size = output_size;
            if (m_processing_spec.m_preprocess_input_channels.size() != input_size.size()) {
                m_processing_spec.m_preprocess_input_channels.clear();
                for (size_t j = 0; j < input_size.size(); ++j) {
                    m_processing_spec.m_preprocess_input_channels.push_back(1);  // Default to 1
                                                                                 // channel if not
                                                                                 // specified
                }
            }
            if (m_processing_spec.m_postprocess_output_channels.size() != output_size.size()) {
                m_processing_spec.m_postprocess_output_channels.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    m_processing_spec.m_postprocess_output_channels.push_back(1);  // Default to 1
                                                                                   // channel if not
                                                                                   // specified
                }
            }
            if (m_processing_spec.m_preprocess_input_size.size() != input_size.size()) {
                m_processing_spec.m_preprocess_input_size.clear();
                for (size_t j = 0; j < input_size.size(); ++j) {
                    size_t length = input_size[j];
                    if (m_processing_spec.m_preprocess_input_channels.size() > j) {
                        length /= m_processing_spec.m_preprocess_input_channels[j];  // Adjust
                                                                                     // length by
                                                                                     // number of
                                                                                     // channels
                    }
                    m_processing_spec.m_preprocess_input_size.push_back(length);
                }
            }
            if (m_processing_spec.m_postprocess_output_size.size() != output_size.size()) {
                m_processing_spec.m_postprocess_output_size.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    size_t length = output_size[j];
                    if (m_processing_spec.m_postprocess_output_channels.size() > j) {
                        length /= m_processing_spec.m_postprocess_output_channels[j];  // Adjust
                                                                                       // length by
                                                                                       // number of
                                                                                       // channels
                    }
                    m_processing_spec.m_postprocess_output_size.push_back(length);
                }
            }
            if (m_processing_spec.m_internal_model_latency.size() != output_size.size()) {
                m_processing_spec.m_internal_model_latency.clear();
                for (size_t j = 0; j < output_size.size(); ++j) {
                    m_processing_spec.m_internal_model_latency.push_back(0);  // Default to 0
                                                                              // latency if not
                                                                              // specified
                }
            }
        } else {
            if (m_processing_spec.m_tensor_input_size != input_size) {
                throw std::invalid_argument("input size mismatch for backend " +
                                            std::to_string(static_cast<int>(shape.m_backend)) +
                                            "; all backends must have the same input size");
            }
            if (m_processing_spec.m_tensor_output_size != output_size) {
                throw std::invalid_argument("output size mismatch for backend " +
                                            std::to_string(static_cast<int>(shape.m_backend)) +
                                            "; all backends must have the same output size");
            }
        }
    }
    if (m_processing_spec.m_preprocess_input_channels.size() !=
        m_processing_spec.m_tensor_input_size.size()) {
        throw std::invalid_argument(
            "preprocess_input_channels has " +
            std::to_string(m_processing_spec.m_preprocess_input_channels.size()) +
            " entries; the model has " +
            std::to_string(m_processing_spec.m_tensor_input_size.size()) + " input tensors");
    }
    if (m_processing_spec.m_postprocess_output_channels.size() !=
        m_processing_spec.m_tensor_output_size.size()) {
        throw std::invalid_argument(
            "postprocess_output_channels has " +
            std::to_string(m_processing_spec.m_postprocess_output_channels.size()) +
            " entries; the model has " +
            std::to_string(m_processing_spec.m_tensor_output_size.size()) + " output tensors");
    }
    if (m_processing_spec.m_preprocess_input_size.size() !=
        m_processing_spec.m_tensor_input_size.size()) {
        throw std::invalid_argument(
            "preprocess_input_size has " +
            std::to_string(m_processing_spec.m_preprocess_input_size.size()) +
            " entries; the model has " +
            std::to_string(m_processing_spec.m_tensor_input_size.size()) + " input tensors");
    }
    if (m_processing_spec.m_postprocess_output_size.size() !=
        m_processing_spec.m_tensor_output_size.size()) {
        throw std::invalid_argument(
            "postprocess_output_size has " +
            std::to_string(m_processing_spec.m_postprocess_output_size.size()) +
            " entries; the model has " +
            std::to_string(m_processing_spec.m_tensor_output_size.size()) + " output tensors");
    }
    if (m_processing_spec.m_internal_model_latency.size() !=
        m_processing_spec.m_tensor_output_size.size()) {
        throw std::invalid_argument(
            "internal_model_latency has " +
            std::to_string(m_processing_spec.m_internal_model_latency.size()) +
            " entries; the model has " +
            std::to_string(m_processing_spec.m_tensor_output_size.size()) + " output tensors");
    }
    for (size_t i = 0; i < m_processing_spec.m_tensor_input_size.size(); ++i) {
        if (m_processing_spec.m_preprocess_input_size[i] == 0) {
            if (m_processing_spec.m_preprocess_input_channels[i] != 1) {
                throw std::invalid_argument(
                    "input tensor " + std::to_string(i) +
                    " is non-streamable (preprocess_input_size 0) but has " +
                    std::to_string(m_processing_spec.m_preprocess_input_channels[i]) +
                    " channels; a non-streamable tensor has exactly 1");
            }
        }
    }
    for (size_t i = 0; i < m_processing_spec.m_tensor_output_size.size(); ++i) {
        if (m_processing_spec.m_postprocess_output_size[i] == 0) {
            if (m_processing_spec.m_postprocess_output_channels[i] != 1) {
                throw std::invalid_argument(
                    "output tensor " + std::to_string(i) +
                    " is non-streamable (postprocess_output_size 0) but has " +
                    std::to_string(m_processing_spec.m_postprocess_output_channels[i]) +
                    " channels; a non-streamable tensor has exactly 1");
            }
        }
    }
}

}  // namespace anira
