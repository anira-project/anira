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
    TensorShapeList m_input_shape;
    TensorShapeList m_output_shape;
    InferenceBackend m_backend;
    bool m_universal = false;

    TensorShape() = delete;
    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape, InferenceBackend backend) :
        m_input_shape(input_shape),
        m_output_shape(output_shape),
        m_backend(backend) {
        assert((m_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_output_shape.size() > 0 && "At least one output shape must be provided."));
    }

    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape) :
        m_input_shape(input_shape),
        m_output_shape(output_shape),
        m_universal(true) {
        assert((m_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_output_shape.size() > 0 && "At least one output shape must be provided."));
    }

    bool operator==(const TensorShape& other) const {
        return
            m_input_shape == other.m_input_shape &&
            m_output_shape == other.m_output_shape &&
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
            std::array<size_t, 2> index_audio_data = {0, 0}, // input and output index of audio data vector of tensors
            std::array<size_t, 2> num_audio_channels = {1, 1}, // input and output number of audio channels
            bool session_exclusive_processor = false,
            unsigned int num_parallel_processors = ((int) std::thread::hardware_concurrency() / 2 > 0) ? (unsigned int) std::thread::hardware_concurrency() / 2 : 1
#ifdef USE_SEMAPHORE
            , float wait_in_process_block = 0.f
#endif
            );

    ~InferenceConfig() = default;

    void set_input_sizes(const std::vector<size_t>& input_sizes);
    void set_output_sizes(const std::vector<size_t>& output_sizes);
    std::string get_model_path(InferenceBackend backend);
    TensorShapeList get_input_shape(InferenceBackend backend);
    TensorShapeList get_output_shape(InferenceBackend backend);
    void set_model_path(const std::string& model_path, InferenceBackend backend);
    void set_input_shape(const TensorShapeList& input_shape, InferenceBackend backend);
    void set_output_shape(const TensorShapeList& output_shape, InferenceBackend backend);

    std::vector<ModelData> m_model_data;
    std::vector<TensorShape> m_tensor_shape;
    float m_max_inference_time;
    unsigned int m_internal_latency;
    unsigned int m_warm_up;
    std::array<size_t, 2> m_index_audio_data;
    std::array<size_t, 2> m_num_audio_channels;
    bool m_session_exclusive_processor;
    size_t m_num_parallel_processors;

#ifdef USE_SEMAPHORE
    float m_wait_in_process_block;
#endif
    
    std::vector<size_t> m_input_sizes;
    std::vector<size_t> m_output_sizes;

    bool operator==(const InferenceConfig& other) const {
        return
            m_model_data == other.m_model_data &&
            m_tensor_shape == other.m_tensor_shape &&
            std::abs(m_max_inference_time - other.m_max_inference_time) < 1e-6 &&
            m_internal_latency == other.m_internal_latency &&
            m_warm_up == other.m_warm_up &&
            m_index_audio_data == other.m_index_audio_data &&
            m_num_audio_channels == other.m_num_audio_channels &&
            m_session_exclusive_processor == other.m_session_exclusive_processor &&
            m_num_parallel_processors == other.m_num_parallel_processors &&
#ifdef USE_SEMAPHORE
            std::abs(m_wait_in_process_block - other.m_wait_in_process_block) < 1e-6 &&
#endif
            m_input_sizes == other.m_input_sizes &&
            m_output_sizes == other.m_output_sizes;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

};

} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H