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

typedef std::vector<std::vector<int64_t>> TensorShapeList;

struct ModelData {
    ModelData(void* data, size_t size, InferenceBackend backend, const std::string& model_function = "", bool is_binary = true) : m_data(data), m_size(size), m_backend(backend), m_model_function(std::move(model_function)), m_is_binary(is_binary) {
        assert((m_size > 0 && "Model data size must be greater than zero."));
        assert((m_data != nullptr && "Model data pointer cannot be null."));
        if (!m_model_function.empty()) {
            if (backend == InferenceBackend::LIBTORCH) {
                m_model_function = model_function; // For LIBTORCH, we can specify a function name
            } else {
                std::cerr << "Model function is only applicable for LIBTORCH backend." << std::endl;
            }
        }
        if (!is_binary) {
            m_data = malloc(sizeof(char) * size);
            memcpy(m_data, data, size);
        } else {
            if (backend != InferenceBackend::ONNX) {
                std::cerr << "Binary model is only supported for ONNX backend." << std::endl;
            }
        }
    }
    ModelData(const std::string& model_path, InferenceBackend backend, const std::string& model_function = "", bool is_binary = false) 
        : ModelData((void*) model_path.data(), model_path.size(), backend, std::move(model_function), is_binary) {}

    ModelData(const ModelData& other) 
        : m_size(other.m_size), m_backend(other.m_backend), m_model_function(other.m_model_function), m_is_binary(other.m_is_binary) {
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
    std::string m_model_function; // Function name in the model, if applicable
    bool m_is_binary;

    bool operator==(const ModelData& other) const {
        return
            m_data == other.m_data &&
            m_size == other.m_size &&
            m_backend == other.m_backend &&
            m_model_function == other.m_model_function &&
            m_is_binary == other.m_is_binary;
    }

    bool operator!=(const ModelData& other) const {
        return !(*this == other);
    }
};

struct TensorShape {
    TensorShapeList m_tensor_input_shape;
    TensorShapeList m_tensor_output_shape;
    InferenceBackend m_backend;
    bool m_universal = false;

    TensorShape() = delete;

    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape) :
        m_tensor_input_shape(input_shape),
        m_tensor_output_shape(output_shape) {
        assert((m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
        m_universal = true;
    }

    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape, InferenceBackend backend) :
        m_tensor_input_shape(input_shape),
        m_tensor_output_shape(output_shape),
        m_backend(backend) {
        assert((m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
    }

    bool is_universal() const {
        return m_universal;
    }

    bool operator==(const TensorShape& other) const {
        return
            m_tensor_input_shape == other.m_tensor_input_shape &&
            m_tensor_output_shape == other.m_tensor_output_shape &&
            m_backend == other.m_backend &&
            m_universal == other.m_universal;
    }

    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
};

struct ProcessingSpec {
    std::vector<size_t> m_preprocess_input_channels;
    std::vector<size_t> m_postprocess_output_channels;
    std::vector<size_t> m_preprocess_input_size;
    std::vector<size_t> m_postprocess_output_size;
    std::vector<size_t> m_internal_latency;
    std::vector<size_t> m_tensor_input_size;
    std::vector<size_t> m_tensor_output_size;

    ProcessingSpec() = default;

    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels,
            std::vector<size_t> preprocess_input_size,
            std::vector<size_t> postprocess_output_size,
            std::vector<size_t> internal_latency) :
        m_preprocess_input_channels(std::move(preprocess_input_channels)),
        m_postprocess_output_channels(std::move(preprocess_output_channels)),
        m_preprocess_input_size(std::move(preprocess_input_size)),
        m_postprocess_output_size(std::move(postprocess_output_size)),
        m_internal_latency(std::move(internal_latency)) {}

    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels) :
        ProcessingSpec(std::move(preprocess_input_channels), std::move(preprocess_output_channels), {}, {}, {}) {}
    
    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels,
            std::vector<size_t> preprocess_input_size,
            std::vector<size_t> postprocess_output_size) :
        ProcessingSpec(std::move(preprocess_input_channels), std::move(preprocess_output_channels), std::move(preprocess_input_size), std::move(postprocess_output_size), {}) {}

    bool operator==(const ProcessingSpec& other) const {
        return
            m_preprocess_input_channels == other.m_preprocess_input_channels &&
            m_postprocess_output_channels == other.m_postprocess_output_channels &&
            m_preprocess_input_size == other.m_preprocess_input_size &&
            m_postprocess_output_size == other.m_postprocess_output_size &&
            m_internal_latency == other.m_internal_latency;
    }

    bool operator!=(const ProcessingSpec& other) const {
        return !(*this == other);
    }
};

class ANIRA_API InferenceConfig {

public:
    struct Defaults
    {
        static constexpr unsigned int m_warm_up = 0;
        static constexpr bool m_session_exclusive_processor = false;
        static constexpr float m_blocking_ratio = 0.f;
        inline static unsigned int m_num_parallel_processors =
            (std::thread::hardware_concurrency() / 2 > 0)
                ? std::thread::hardware_concurrency() / 2
                : 1;
    };

    InferenceConfig() = default;

    InferenceConfig(
            std::vector<ModelData> model_data,
            std::vector<TensorShape> tensor_shape,
            ProcessingSpec processing_spec,
            float max_inference_time, // in ms per inference
            unsigned int warm_up = Defaults::m_warm_up, // number of warm up inferences
            bool session_exclusive_processor = Defaults::m_session_exclusive_processor,
            float blocking_ratio = Defaults::m_blocking_ratio,
            unsigned int num_parallel_processors = Defaults::m_num_parallel_processors
            );
    
    InferenceConfig(
            std::vector<ModelData> model_data,
            std::vector<TensorShape> tensor_shape,
            float max_inference_time, // in ms per inference
            unsigned int warm_up = Defaults::m_warm_up, // number of warm up inferences
            bool session_exclusive_processor = Defaults::m_session_exclusive_processor,
            float blocking_ratio = Defaults::m_blocking_ratio,
            unsigned int num_parallel_processors = Defaults::m_num_parallel_processors
            ) :
        InferenceConfig(
            std::move(model_data),
            std::move(tensor_shape),
            ProcessingSpec(),
            max_inference_time,
            warm_up,
            session_exclusive_processor,
            blocking_ratio,
            num_parallel_processors
            ) {}

    std::string get_model_path(InferenceBackend backend);
    const ModelData* get_model_data(InferenceBackend backend) const;
    std::string get_model_function(InferenceBackend backend) const;
    bool is_model_binary(InferenceBackend backend) const;
    TensorShapeList get_tensor_input_shape() const;
    TensorShapeList get_tensor_output_shape() const;
    TensorShapeList get_tensor_input_shape(InferenceBackend backend) const;
    TensorShapeList get_tensor_output_shape(InferenceBackend backend) const;
    std::vector<size_t> get_tensor_input_size() const;
    std::vector<size_t> get_tensor_output_size() const;
    std::vector<size_t> get_preprocess_input_channels() const;
    std::vector<size_t> get_postprocess_output_channels() const;
    std::vector<size_t> get_preprocess_input_size() const;
    std::vector<size_t> get_postprocess_output_size() const;
    std::vector<size_t> get_internal_latency() const;
    void set_tensor_input_shape(const TensorShapeList& input_shape);
    void set_tensor_output_shape(const TensorShapeList& output_shape);
    void set_tensor_input_shape(const TensorShapeList& input_shape, InferenceBackend backend);
    void set_tensor_output_shape(const TensorShapeList& output_shape, InferenceBackend backend);
    void set_preprocess_input_channels(const std::vector<size_t>& input_channels);
    void set_preprocess_output_channels(const std::vector<size_t>& output_channels);
    void set_preprocess_input_size(const std::vector<size_t>& preprocess_input_size);
    void set_postprocess_output_size(const std::vector<size_t>& postprocess_output_size);
    void set_internal_latency(const std::vector<size_t>& internal_latency);
    void set_model_path(const std::string& model_path, InferenceBackend backend);

    std::vector<ModelData> m_model_data;
    std::vector<TensorShape> m_tensor_shape;
    ProcessingSpec m_processing_spec;
    float m_max_inference_time;
    unsigned int m_warm_up;
    bool m_session_exclusive_processor;
    float m_blocking_ratio;
    unsigned int m_num_parallel_processors;
    
    bool operator==(const InferenceConfig& other) const {
        return
            m_model_data == other.m_model_data &&
            m_tensor_shape == other.m_tensor_shape &&
            std::abs(m_max_inference_time - other.m_max_inference_time) < 1e-6 &&
            m_processing_spec == other.m_processing_spec &&
            m_warm_up == other.m_warm_up &&
            m_session_exclusive_processor == other.m_session_exclusive_processor &&
            std::abs(m_blocking_ratio - other.m_blocking_ratio) < 1e-6 &&
            m_num_parallel_processors == other.m_num_parallel_processors
            ;
    }

    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

    const TensorShape& get_tensor_shape(InferenceBackend backend) const;
    void clear_processing_spec();
    void update_processing_spec();
};

} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H
