#ifndef ANIRA_INFERENCECONFIG_H
#define ANIRA_INFERENCECONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include <iostream>
#include <cassert>
#include <cstring>
#include <anira/utils/InferenceBackend.h>
#include "anira/system/AniraWinExports.h"

namespace anira {

enum IndexAudioData : size_t {
    Input = 0,
    Output = 1
};

typedef std::vector<std::vector<int64_t>> TensorShapeList;

struct ModelData {
    ModelData(void* data, size_t size, InferenceBackend backend, bool is_binary = true) : m_data(data), m_size(size), m_backend(backend), m_is_binary(is_binary) {}
    ModelData(std::string model_path, InferenceBackend backend, bool is_binary = false) : m_size(model_path.size()), m_backend(backend), m_is_binary(is_binary) {
        m_data = malloc(sizeof(char) * model_path.size());
        memcpy(m_data, model_path.data(), model_path.size());
        m_size = model_path.size();
    }

    ModelData(const ModelData& other) 
        : m_size(other.m_size), m_backend(other.m_backend), m_is_binary(other.m_is_binary) {
        if (!m_is_binary) {
            m_data = malloc(sizeof(char) * other.m_size);
            memcpy(m_data, other.m_data, other.m_size);
        } else {
            m_data = other.m_data;
        }
    }

    ModelData& operator=(const ModelData& other) {
        if (this != &other) {
            if (!m_is_binary) {
                free(m_data);
                m_data = malloc(other.m_size);
                memcpy(m_data, other.m_data, other.m_size);
            } else {
                m_data = other.m_data;
            }
            m_size = other.m_size;
            m_backend = other.m_backend;
            m_is_binary = other.m_is_binary;
        }
        return *this;
    }

    ~ModelData() {
        if (!m_is_binary) {
            free(m_data);
        }
    }
    
    void* m_data;
    size_t m_size;
    InferenceBackend m_backend;
    bool m_is_binary;

    bool operator==(const ModelData& other) const {
        return
            m_data == other.m_data &&
            m_size == other.m_size &&
            m_backend == other.m_backend &&
            m_is_binary == other.m_is_binary;
    }

    bool operator!=(const ModelData& other) const {
        return !(*this == other);
    }
};

struct TensorShape {
    TensorShapeList m_tensor_input_shape;
    TensorShapeList m_tensor_output_shape;
    std::vector<size_t> m_tensor_input_size;
    std::vector<size_t> m_tensor_output_size;
    std::vector<size_t> m_preprocess_input_channels;
    std::vector<size_t> m_preprocess_output_channels;
    std::vector<size_t> m_preprocess_input_size;
    std::vector<size_t> m_postprocess_output_size;
    InferenceBackend m_backend;

    TensorShape() = delete;

    TensorShape(TensorShapeList input_shape, std::vector<size_t> input_channels, std::vector<size_t> preprocess_input_size, TensorShapeList output_shape, std::vector<size_t> output_channels, std::vector<size_t> postprocess_output_size, InferenceBackend backend = InferenceBackend::UNIVERSAL) :
        m_tensor_input_shape(input_shape),
        m_tensor_output_shape(output_shape),
        m_preprocess_input_channels(input_channels),
        m_preprocess_output_channels(output_channels),
        m_preprocess_input_size(preprocess_input_size),
        m_postprocess_output_size(postprocess_output_size),
        m_backend(backend) {
        for (const auto& shape : m_tensor_input_shape) {
            size_t tensor_input_size = 1;
            for (int64_t dim : shape) {
                if (dim < 1) {
                    std::cerr << "Invalid dimension in input shape: " << dim << ". Input dimensions must be positive." << std::endl;
                    throw std::invalid_argument("Invalid dimension in input shape.");
                }
                tensor_input_size *= (size_t) dim;
            }
            m_tensor_input_size.push_back(tensor_input_size);
        }
        for (const auto& shape : m_tensor_output_shape) {
            size_t tensor_output_size = 1;
            for (int64_t dim : shape) {
                if (dim < 1) {
                    std::cerr << "Invalid dimension in output shape: " << dim << ". Output dimensions must be positive." << std::endl;
                    throw std::invalid_argument("Invalid dimension in output shape.");
                }
                tensor_output_size *= (size_t) dim;
            }
            m_tensor_output_size.push_back(tensor_output_size);
        }

        if (m_preprocess_input_channels.size() == 0) {
            for (size_t i = 0; i < m_tensor_input_shape.size(); ++i) {
                m_preprocess_input_channels.push_back(1); // Default to 1 channel if not specified
            }
        }
        if (m_preprocess_output_channels.size() == 0) {
            for (size_t i = 0; i < m_tensor_output_shape.size(); ++i) {
                m_preprocess_output_channels.push_back(1); // Default to 1 channel if not specified
            }
        }
        if (m_preprocess_input_size.size() == 0) {
            for (size_t i = 0; i < m_tensor_input_shape.size(); ++i) {
                const auto& shape = m_tensor_input_shape[i];
                const size_t num_channels = m_preprocess_input_channels[i];
                size_t length = 1;
                for (int64_t dim : shape) {
                    if (dim < 1) {
                        std::cerr << "Invalid dimension in input shape: " << dim << ". Input dimensions must be positive, when preprocess_input_size is not specified." << std::endl;
                        throw std::invalid_argument("Invalid dimension in input shape.");
                    }
                    length *= (size_t) dim;
                }
                length /= num_channels; // Adjust length by number of channels
                m_preprocess_input_size.push_back(length);
            }
        }
        if (m_postprocess_output_size.size() == 0) {
            for (size_t i = 0; i < m_tensor_output_shape.size(); ++i) {
                const auto& shape = m_tensor_output_shape[i];
                const size_t num_channels = m_preprocess_output_channels[i];
                size_t length = 1;
                for (int64_t dim : shape) {
                    if (dim < 1) {
                        std::cerr << "Invalid dimension in output shape: " << dim << ". Output dimensions must be positive, when postprocess_output_size is not specified." << std::endl;
                        throw std::invalid_argument("Invalid dimension in output shape.");
                    }
                    length *= (size_t) dim;
                }
                length /= num_channels; // Adjust length by number of channels
                m_postprocess_output_size.push_back(length);
            }
        }
        assert((m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
        assert((m_tensor_input_shape.size() == m_preprocess_input_channels.size() && "Input shape size must match input channels size."));
        assert((m_tensor_output_shape.size() == m_preprocess_output_channels.size() && "Output shape size must match output channels size."));
        assert((m_preprocess_input_size.size() == m_tensor_input_shape.size() && "Length for preprocessing must match input shape size."));
        assert((m_postprocess_output_size.size() == m_tensor_output_shape.size() && "Length for postprocessing must match output shape size."));
    }

    TensorShape(TensorShapeList input_shape, std::vector<size_t> input_channels, TensorShapeList output_shape, std::vector<size_t> output_channels, InferenceBackend backend = InferenceBackend::UNIVERSAL) :
        TensorShape(input_shape, input_channels, {}, output_shape, output_channels, {}, backend) {
    }

    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape, InferenceBackend backend = InferenceBackend::UNIVERSAL) :
        TensorShape(input_shape, {}, {}, output_shape, {}, {}, backend) {
    }

    bool operator==(const TensorShape& other) const {
        return
            m_tensor_input_shape == other.m_tensor_input_shape &&
            m_tensor_output_shape == other.m_tensor_output_shape &&
            m_tensor_input_size == other.m_tensor_input_size &&
            m_tensor_output_size == other.m_tensor_output_size &&
            m_preprocess_input_channels == other.m_preprocess_input_channels &&
            m_preprocess_output_channels == other.m_preprocess_output_channels &&
            m_preprocess_input_size == other.m_preprocess_input_size &&
            m_postprocess_output_size == other.m_postprocess_output_size &&
            m_backend == other.m_backend;
    }

    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
};


class ANIRA_API InferenceConfig {

public:
    InferenceConfig() = default;

    InferenceConfig(
            std::vector<ModelData> model_data,
            std::vector<TensorShape> tensor_shape,
            float max_inference_time, // in ms per inference
            unsigned int internal_latency = 0, // in samples per inference
            unsigned int warm_up = 0, // number of warm up inferences
            bool session_exclusive_processor = false,
            unsigned int num_parallel_processors = (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2 : 1
#ifdef USE_CONTROLLED_BLOCKING
            , float wait_in_process_block = 0.f
#endif
            );

    std::string get_model_path(InferenceBackend backend);
    const ModelData* get_model_data(InferenceBackend backend) const;
    bool is_model_binary(InferenceBackend backend) const;
    TensorShapeList get_tensor_input_shape(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    TensorShapeList get_tensor_output_shape(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_tensor_input_size(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_tensor_output_size(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_preprocess_input_channels(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_postprocess_output_channels(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_preprocess_input_size(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    std::vector<size_t> get_postprocess_output_size(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    void set_tensor_input_shape(const TensorShapeList& input_shape, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_tensor_output_shape(const TensorShapeList& output_shape, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_preprocess_input_channels(const std::vector<size_t>& input_channels, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_preprocess_output_channels(const std::vector<size_t>& output_channels, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_preprocess_input_size(const std::vector<size_t>& preprocess_input_size, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_postprocess_output_size(const std::vector<size_t>& postprocess_output_size, InferenceBackend backend = InferenceBackend::UNIVERSAL);
    void set_model_path(const std::string& model_path, InferenceBackend backend);

    std::vector<ModelData> m_model_data;
    std::vector<TensorShape> m_tensor_shape;
    float m_max_inference_time;
    std::vector<unsigned int> m_internal_latency;
    unsigned int m_warm_up;
    bool m_session_exclusive_processor;
    unsigned int m_num_parallel_processors;

#ifdef USE_CONTROLLED_BLOCKING
    float m_wait_in_process_block;
#endif
    
    bool operator==(const InferenceConfig& other) const {
        return
            m_model_data == other.m_model_data &&
            m_tensor_shape == other.m_tensor_shape &&
            std::abs(m_max_inference_time - other.m_max_inference_time) < 1e-6 &&
            m_internal_latency == other.m_internal_latency &&
            m_warm_up == other.m_warm_up &&
            m_session_exclusive_processor == other.m_session_exclusive_processor &&
            m_num_parallel_processors == other.m_num_parallel_processors
#ifdef USE_CONTROLLED_BLOCKING
            && std::abs(m_wait_in_process_block - other.m_wait_in_process_block) < 1e-6
#endif 
            ;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

private:
    const TensorShape& get_tensor_shape(InferenceBackend backend = InferenceBackend::UNIVERSAL) const;
    void update_tensor_shapes();
};

} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H
