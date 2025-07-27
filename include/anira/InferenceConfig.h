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

/**
 * @brief Container for neural network model data and metadata
 * 
 * The ModelData struct encapsulates all information necessary to load and identify
 * a neural network model for inference. It supports both binary model data (loaded
 * from files) and string-based model paths, along with backend-specific metadata.
 * 
 * @see InferenceBackend, InferenceConfig
 */
struct ANIRA_API ModelData {
    /**
     * @brief Constructs ModelData with binary data or model path
     * 
     * @param data Pointer to model data (binary data) or model path string
     * @param size Size of the data in bytes (for binary) or string length (for paths)
     * @param backend The inference backend that will use this model
     * @param model_function Optional function name within the model (LibTorch only)
     * @param is_binary Whether the data represents binary model data (true) or a file path (false)
     */
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
    
    /**
     * @brief Constructs ModelData from a file path string
     * 
     * Convenience constructor for creating ModelData from a model file path.
     * The path string is copied internally and managed by the ModelData instance.
     * 
     * @param model_path Path to the model file on disk
     * @param backend The inference backend that will use this model
     * @param model_function Optional function name within the model (LibTorch only)
     * @param is_binary Whether to treat the path as binary data (typically false for file paths)
     */
    ModelData(const std::string& model_path, InferenceBackend backend, const std::string& model_function = "", bool is_binary = false) 
        : ModelData((void*) model_path.data(), model_path.size(), backend, std::move(model_function), is_binary) {}

    /**
     * @brief Copy constructor with proper memory management
     * 
     * Creates a deep copy of the ModelData, ensuring independent memory management
     * for non-binary data while sharing binary data references safely.
     * 
     * @param other The ModelData instance to copy from
     */
    ModelData(const ModelData& other) 
        : m_size(other.m_size), m_backend(other.m_backend), m_model_function(other.m_model_function), m_is_binary(other.m_is_binary) {
        if (!m_is_binary) {
            m_data = malloc(sizeof(char) * other.m_size);
            memcpy(m_data, other.m_data, other.m_size);
        } else {
            m_data = other.m_data;
        }
    }

    /**
     * @brief Assignment operator with proper memory management
     * 
     * Safely assigns one ModelData to another, handling memory deallocation
     * and reallocation as needed for non-binary data.
     * 
     * @param other The ModelData instance to assign from
     * @return Reference to this ModelData instance
     */
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

    /**
     * @brief Destructor with automatic memory cleanup
     * 
     * Automatically frees allocated memory for non-binary data to prevent memory leaks.
     * Binary data is assumed to be managed externally and is not freed.
     */
    ~ModelData() {
        if (!m_is_binary) {
            free(m_data);
        }
    }
    
    void* m_data;                    ///< Pointer to model data (binary data or string data)
    size_t m_size;                   ///< Size of the model data in bytes
    InferenceBackend m_backend;      ///< Target inference backend for this model
    std::string m_model_function;    ///< Function name within the model (LibTorch specific)
    bool m_is_binary;                ///< Whether the data represents binary model data

    /**
     * @brief Equality comparison operator
     * 
     * @param other The ModelData instance to compare with
     * @return true if all members are equal, false otherwise
     */
    bool operator==(const ModelData& other) const {
        return
            m_data == other.m_data &&
            m_size == other.m_size &&
            m_backend == other.m_backend &&
            m_model_function == other.m_model_function &&
            m_is_binary == other.m_is_binary;
    }

    /**
     * @brief Inequality comparison operator
     * 
     * @param other The ModelData instance to compare with
     * @return true if any members are not equal, false otherwise
     */
    bool operator!=(const ModelData& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Defines input and output tensor shapes for neural network models
 * 
 * The TensorShape struct specifies the dimensional structure of tensors used by
 * neural network models. It supports both universal shapes (backend-agnostic) and
 * backend-specific shapes for models that require different tensor layouts across
 * different inference engines.
 * 
 * @warning All tensor shapes must have at least one input and one output tensor
 * defined to ensure proper model configuration.
 * 
 * @see TensorShapeList, InferenceBackend, ModelData
 */
struct ANIRA_API TensorShape {
    TensorShapeList m_tensor_input_shape;   ///< List of input tensor shapes (each shape is a vector of dimensions)
    TensorShapeList m_tensor_output_shape;  ///< List of output tensor shapes (each shape is a vector of dimensions)
    InferenceBackend m_backend;             ///< Target backend for backend-specific shapes
    bool m_universal = false;               ///< Whether this shape configuration is universal (backend-agnostic)

    /**
     * @brief Default constructor is deleted to prevent uninitialized instances
     */
    TensorShape() = delete;

    /**
     * @brief Constructs a universal TensorShape that works across all backends
     * 
     * Creates a TensorShape configuration that can be used with any inference backend.
     * This is useful when all backends can use the same tensor layout.
     * 
     * @param input_shape List of input tensor shapes, where each shape is a vector of dimensions
     * @param output_shape List of output tensor shapes, where each shape is a vector of dimensions
     */
    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape) :
        m_tensor_input_shape(input_shape),
        m_tensor_output_shape(output_shape) {
        assert((m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
        m_universal = true;
    }

    /**
     * @brief Constructs a backend-specific TensorShape
     * 
     * Creates a TensorShape configuration that is optimized for a specific inference backend.
     * This allows different tensor layouts for different backends when models are optimized
     * differently for each inference engine.
     * 
     * @param input_shape List of input tensor shapes, where each shape is a vector of dimensions
     * @param output_shape List of output tensor shapes, where each shape is a vector of dimensions
     * @param backend The specific inference backend this shape configuration targets
     */
    TensorShape(TensorShapeList input_shape, TensorShapeList output_shape, InferenceBackend backend) :
        m_tensor_input_shape(input_shape),
        m_tensor_output_shape(output_shape),
        m_backend(backend) {
        assert((m_tensor_input_shape.size() > 0 && "At least one input shape must be provided."));
        assert((m_tensor_output_shape.size() > 0 && "At least one output shape must be provided."));
    }

    /**
     * @brief Checks if this tensor shape configuration is universal
     * 
     * @return true if the configuration works across all backends, false if backend-specific
     */
    bool is_universal() const {
        return m_universal;
    }

    /**
     * @brief Equality comparison operator
     * 
     * @param other The TensorShape instance to compare with
     * @return true if all members are equal, false otherwise
     */
    bool operator==(const TensorShape& other) const {
        return
            m_tensor_input_shape == other.m_tensor_input_shape &&
            m_tensor_output_shape == other.m_tensor_output_shape &&
            m_backend == other.m_backend &&
            m_universal == other.m_universal;
    }

    /**
     * @brief Inequality comparison operator
     * 
     * @param other The TensorShape instance to compare with
     * @return true if any members are not equal, false otherwise
     */
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Specification for preprocessing and postprocessing parameters
 * 
 * The ProcessingSpec struct defines the processing pipeline configuration for transforming
 * data between the host application and neural network inference.
 * 
 * @par Streamable vs Non-Streamable Tensors:
 * - **Streamable tensors**: Time-varying data (e.g., audio) that flows continuously
 *   - Have non-zero preprocess_input_size and postprocess_output_size
 *   - Managed through ring buffers for real-time processing
 * - **Non-streamable tensors**: Static parameters or control values
 *   - Have zero preprocess_input_size or postprocess_output_size
 *   - Stored in thread-safe internal storage
 * 
 * @see InferenceConfig, TensorShape, PrePostProcessor
 */
struct ANIRA_API ProcessingSpec {
    std::vector<size_t> m_preprocess_input_channels;   ///< Number of input channels for each input tensor
    std::vector<size_t> m_postprocess_output_channels; ///< Number of output channels for each output tensor
    std::vector<size_t> m_preprocess_input_size;       ///< Samples count required for preprocessing for each input tensor (0 = non-streamable)
    std::vector<size_t> m_postprocess_output_size;     ///< Samples count after the postprocessing for each output tensor (0 = non-streamable)
    std::vector<size_t> m_internal_model_latency;      ///< Internal latency in samples for each output tensor
    std::vector<size_t> m_tensor_input_size;           ///< Total size (elements) of each input tensor (computed from shape)
    std::vector<size_t> m_tensor_output_size;          ///< Total size (elements) of each output tensor (computed from shape)

    /**
     * @brief Default constructor creating an empty processing specification
     */
    ProcessingSpec() = default;

    /**
     * @brief Constructs a complete ProcessingSpec with all parameters
     * 
     * @param preprocess_input_channels Number of input channels for each input tensor
     * @param preprocess_output_channels Number of output channels for each output tensor  
     * @param preprocess_input_size Samples count required for preprocessing for each input tensor (0 = non-streamable)
     * @param postprocess_output_size Samples count after the postprocessing for each output tensor (0 = non-streamable)
     * @param internal_model_latency Internal model latency in samples for each output tensor
     */
    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels,
            std::vector<size_t> preprocess_input_size,
            std::vector<size_t> postprocess_output_size,
            std::vector<size_t> internal_model_latency) :
        m_preprocess_input_channels(std::move(preprocess_input_channels)),
        m_postprocess_output_channels(std::move(preprocess_output_channels)),
        m_preprocess_input_size(std::move(preprocess_input_size)),
        m_postprocess_output_size(std::move(postprocess_output_size)),
        m_internal_model_latency(std::move(internal_model_latency)) {}

    /**
     * @brief Constructs a minimal ProcessingSpec with only channel information
     * 
     * Creates a processing specification with only input and output channel counts.
     * Other parameters are left empty and will be computed automatically by InferenceConfig.
     * 
     * @param preprocess_input_channels Number of input channels for each input tensor
     * @param preprocess_output_channels Number of output channels for each output tensor
     */
    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels) :
        ProcessingSpec(std::move(preprocess_input_channels), std::move(preprocess_output_channels), {}, {}, {}) {}
    
    /**
     * @brief Constructs a ProcessingSpec with channel and size information
     * 
     * Creates a processing specification with input/output channels and buffer sizes.
     * Internal model latency defaults to zero for all tensors.
     * 
     * @param preprocess_input_channels Number of input channels for each input tensor
     * @param preprocess_output_channels Number of output channels for each output tensor
     * @param preprocess_input_size Samples count required for preprocessing for each input tensor (0 = non-streamable)
     * @param postprocess_output_size Samples count after the postprocessing for each output tensor (0 = non-streamable)
     */
    ProcessingSpec(
            std::vector<size_t> preprocess_input_channels,
            std::vector<size_t> preprocess_output_channels,
            std::vector<size_t> preprocess_input_size,
            std::vector<size_t> postprocess_output_size) :
        ProcessingSpec(std::move(preprocess_input_channels), std::move(preprocess_output_channels), std::move(preprocess_input_size), std::move(postprocess_output_size), {}) {}

    /**
     * @brief Equality comparison operator
     * 
     * @param other The ProcessingSpec instance to compare with
     * @return true if all members are equal, false otherwise
     */
    bool operator==(const ProcessingSpec& other) const {
        return
            m_preprocess_input_channels == other.m_preprocess_input_channels &&
            m_postprocess_output_channels == other.m_postprocess_output_channels &&
            m_preprocess_input_size == other.m_preprocess_input_size &&
            m_postprocess_output_size == other.m_postprocess_output_size &&
            m_internal_model_latency == other.m_internal_model_latency;
    }

    /**
     * @brief Inequality comparison operator
     * 
     * @param other The ProcessingSpec instance to compare with
     * @return true if any members are not equal, false otherwise
     */
    bool operator!=(const ProcessingSpec& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Complete configuration for neural network inference operations
 * 
 * The InferenceConfig struct serves as the central configuration hub for all aspects
 * of neural network inference in anira. It combines model data, tensor specifications,
 * processing parameters, and performance settings into a single, cohesive configuration.
 * 
 * @par Usage Examples:
 * ```cpp
 * // Simple mono audio effect
 * InferenceConfig config(
 *     {ModelData("model.onnx", ONNX)},
 *     {TensorShape({{1, 512}}, {{1, 512}})},
 *     ProcessingSpec({1}, {1}, {512}, {512}),
 *     10.0f  // 10ms max inference time
 * );
 * 
 * // Multi-input model with control parameters
 * InferenceConfig config(
 *     {ModelData("model.onnx", ONNX)},
 *     {TensorShape({{1, 512}, {1, 4}}, {{1, 512}})}, // Audio + 4 parameters
 *     ProcessingSpec({1, 1}, {1}, {512, 0}, {512}),  // Second input non-streamable
 *     15.0f
 * );
 * ```
 * 
 * @see ModelData, TensorShape, ProcessingSpec, InferenceHandler
 */
struct ANIRA_API InferenceConfig {

public:
    /**
     * @brief Default values for inference configuration parameters
     * 
     * This nested struct provides sensible default values for optional InferenceConfig
     * parameters, ensuring consistent behavior across different usage scenarios.
     */
    struct Defaults
    {
        static constexpr unsigned int m_warm_up = 0;                        ///< Default number of warm-up inferences (0 = no warm-up)
        static constexpr bool m_session_exclusive_processor = false;        ///< Default session exclusivity (false = shared processors)
        static constexpr float m_blocking_ratio = 0.f;                     ///< Default blocking ratio (0.0 = non-blocking)
        
        /// Default number of parallel processors (half of available hardware threads, minimum 1)
        inline static unsigned int m_num_parallel_processors =
            (std::thread::hardware_concurrency() / 2 > 0)
                ? std::thread::hardware_concurrency() / 2
                : 1;
    };

    /**
     * @brief Default constructor creating an empty configuration
     * 
     * Creates an uninitialized InferenceConfig that must be properly configured
     * before use. All member variables are default-initialized.
     */
    InferenceConfig() = default;

    /**
     * @brief Constructs a complete InferenceConfig with processing specification
     * 
     * Creates a fully configured InferenceConfig suitable for immediate use.
     * This constructor includes explicit processing specifications for fine-grained
     * control over the inference pipeline.
     * 
     * @param model_data Vector of model data for different backends
     * @param tensor_shape Vector of tensor shape configurations
     * @param processing_spec Processing specification defining channels, sizes, and latencies
     * @param max_inference_time Maximum allowed inference time in milliseconds per inference
     * @param warm_up Number of warm-up inferences to perform during initialization
     * @param session_exclusive_processor Whether to use exclusive processor sessions
     * @param blocking_ratio Ratio controlling blocking behavior (0.0-1.0)
     * @param num_parallel_processors Number of parallel inference processors to use
     */
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
    
    /**
     * @brief Constructs a simplified InferenceConfig with automatic processing specification
     * 
     * Creates an InferenceConfig with automatic processing specification generation.
     * The ProcessingSpec will be computed automatically from the provided tensor shapes,
     * using default values for buffer sizes and channel counts.
     * 
     * @param model_data Vector of model data for different backends
     * @param tensor_shape Vector of tensor shape configurations
     * @param max_inference_time Maximum allowed inference time in milliseconds per inference
     * @param warm_up Number of warm-up inferences to perform during initialization
     * @param session_exclusive_processor Whether to use exclusive processor sessions
     * @param blocking_ratio Ratio controlling blocking behavior (0.0-1.0)
     * @param num_parallel_processors Number of parallel inference processors to use
     */
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

    // ========================================
    // Model Data Access Methods
    // ========================================
    
    /**
     * @brief Gets the model file path for a specific backend
     * @param backend The target inference backend
     * @return Model file path as string
     */
    std::string get_model_path(InferenceBackend backend);
    
    /**
     * @brief Gets the model data structure for a specific backend
     * @param backend The target inference backend
     * @return Pointer to ModelData, or nullptr if not found
     */
    const ModelData* get_model_data(InferenceBackend backend) const;
    
    /**
     * @brief Gets the model function name for a specific backend
     * @param backend The target inference backend
     * @return Model function name (LibTorch specific)
     */
    std::string get_model_function(InferenceBackend backend) const;
    
    /**
     * @brief Checks if the model data is binary for a specific backend
     * @param backend The target inference backend
     * @return true if model data is binary, false if it's a file path
     */
    bool is_model_binary(InferenceBackend backend) const;

    // ========================================
    // Tensor Shape Access Methods
    // ========================================
    
    /**
     * @brief Gets universal input tensor shapes
     * @return List of input tensor shapes (universal across backends)
     */
    TensorShapeList get_tensor_input_shape() const;
    
    /**
     * @brief Gets universal output tensor shapes
     * @return List of output tensor shapes (universal across backends)
     */
    TensorShapeList get_tensor_output_shape() const;
    
    /**
     * @brief Gets input tensor shapes for a specific backend
     * @param backend The target inference backend
     * @return List of input tensor shapes for the specified backend
     */
    TensorShapeList get_tensor_input_shape(InferenceBackend backend) const;
    
    /**
     * @brief Gets output tensor shapes for a specific backend
     * @param backend The target inference backend
     * @return List of output tensor shapes for the specified backend
     */
    TensorShapeList get_tensor_output_shape(InferenceBackend backend) const;

    // ========================================
    // Processing Specification Access Methods
    // ========================================
    
    /**
     * @brief Gets total size (element count) for each input tensor
     * @return Vector of input tensor sizes
     */
    std::vector<size_t> get_tensor_input_size() const;
    
    /**
     * @brief Gets total size (element count) for each output tensor
     * @return Vector of output tensor sizes
     */
    std::vector<size_t> get_tensor_output_size() const;
    
    /**
     * @brief Gets number of input channels for each input tensor
     * @return Vector of input channel counts
     */
    std::vector<size_t> get_preprocess_input_channels() const;
    
    /**
     * @brief Gets number of output channels for each output tensor
     * @return Vector of output channel counts
     */
    std::vector<size_t> get_postprocess_output_channels() const;
    
    /**
     * @brief Gets samples count required for preprocessing for each input tensor
     * @return Vector of input buffer sizes (0 = non-streamable)
     */
    std::vector<size_t> get_preprocess_input_size() const;
    
    /**
     * @brief Gets samples count after the postprocessing for each output tensor
     * @return Vector of output buffer sizes (0 = non-streamable)
     */
    std::vector<size_t> get_postprocess_output_size() const;
    
    /**
     * @brief Gets internal model latency for each output tensor
     * @return Vector of latency values in samples
     */
    std::vector<size_t> get_internal_model_latency() const;

    // ========================================
    // Configuration Modification Methods
    // ========================================
    
    /**
     * @brief Sets universal input tensor shapes
     * @param input_shape New input tensor shapes for all backends
     */
    void set_tensor_input_shape(const TensorShapeList& input_shape);
    
    /**
     * @brief Sets universal output tensor shapes
     * @param output_shape New output tensor shapes for all backends
     */
    void set_tensor_output_shape(const TensorShapeList& output_shape);
    
    /**
     * @brief Sets input tensor shapes for a specific backend
     * @param input_shape New input tensor shapes
     * @param backend Target inference backend
     */
    void set_tensor_input_shape(const TensorShapeList& input_shape, InferenceBackend backend);
    
    /**
     * @brief Sets output tensor shapes for a specific backend
     * @param output_shape New output tensor shapes
     * @param backend Target inference backend
     */
    void set_tensor_output_shape(const TensorShapeList& output_shape, InferenceBackend backend);
    
    /**
     * @brief Sets input channel counts for preprocessing
     * @param input_channels New input channel counts
     */
    void set_preprocess_input_channels(const std::vector<size_t>& input_channels);
    
    /**
     * @brief Sets output channel counts for postprocessing
     * @param output_channels New output channel counts
     */
    void set_preprocess_output_channels(const std::vector<size_t>& output_channels);
    
    /**
     * @brief Sets the required samples count for preprocessing each input tensor
     * @param preprocess_input_size New samples count required for preprocessing (0 = non-streamable)
     */
    void set_preprocess_input_size(const std::vector<size_t>& preprocess_input_size);
    
    /**
     * @brief Sets the required samples count after postprocessing each output tensor
     * @param postprocess_output_size New samples count after postprocessing (0 = non-streamable)
     */
    void set_postprocess_output_size(const std::vector<size_t>& postprocess_output_size);
    
    /**
     * @brief Sets internal model latency values
     * @param internal_model_latency New latency values in samples
     */
    void set_internal_model_latency(const std::vector<size_t>& internal_model_latency);
    
    /**
     * @brief Sets model path for a specific backend
     * @param model_path New model file path
     * @param backend Target inference backend
     */
    void set_model_path(const std::string& model_path, InferenceBackend backend);

    // ========================================
    // Public Member Variables
    // ========================================
    
    std::vector<ModelData> m_model_data;           ///< Model data for different inference backends
    std::vector<TensorShape> m_tensor_shape;       ///< Tensor shape configurations for inputs and outputs
    ProcessingSpec m_processing_spec;              ///< Processing specification for preprocessing and postprocessing
    float m_max_inference_time;                    ///< Maximum allowed inference time in milliseconds
    unsigned int m_warm_up;                        ///< Number of warm-up inferences to perform
    bool m_session_exclusive_processor;            ///< Whether to use exclusive processor sessions
    float m_blocking_ratio;                        ///< Blocking ratio for real-time control (0.0-1.0)
    unsigned int m_num_parallel_processors;        ///< Number of parallel inference processors
    
    /**
     * @brief Equality comparison operator
     * 
     * Compares all configuration parameters using appropriate epsilon values
     * for floating-point comparisons.
     * 
     * @param other The InferenceConfig instance to compare with
     * @return true if all parameters are equal within tolerance, false otherwise
     */
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

    /**
     * @brief Inequality comparison operator
     * 
     * @param other The InferenceConfig instance to compare with
     * @return true if any parameters are not equal, false otherwise
     */
    bool operator!=(const InferenceConfig& other) const {
        return !(*this == other);
    }

    // ========================================
    // Advanced Configuration Methods
    // ========================================
    
    /**
     * @brief Gets tensor shape configuration for a specific backend
     * 
     * Retrieves the TensorShape object that matches the specified backend,
     * or falls back to universal shapes if no backend-specific shape is found.
     * 
     * @param backend The target inference backend
     * @return Reference to the matching TensorShape configuration
     * @throws std::runtime_error if no matching shape configuration is found
     */
    const TensorShape& get_tensor_shape(InferenceBackend backend) const;
    
    /**
     * @brief Clears all processing specification parameters
     * 
     * Resets all processing specification vectors to empty state.
     * This is useful before reconfiguring the processing pipeline.
     */
    void clear_processing_spec();
    
    /**
     * @brief Updates and validates the processing specification
     * 
     * Automatically computes tensor sizes from shapes, validates consistency
     * between tensor configurations and processing parameters, and fills in
     * default values where needed. This method should be called after any
     * changes to tensor shapes or processing parameters.
     * 
     * @throws std::invalid_argument if tensor shapes and processing parameters are inconsistent
     * @throws std::invalid_argument if tensor dimensions are invalid (non-positive)
     * 
     * @note This method is automatically called by InferenceConfig constructors
     */
    void update_processing_spec();

private:
#if DOXYGEN
    // Placeholder to include in Doxygen diagram
    // Since Doxygen does not find classes structures nested in std::vectors or std::shared_ptr
    ModelData* __doxygen_force_0;
    TensorShape* __doxygen_force_1;
#endif
};

} // namespace anira

#endif //ANIRA_INFERENCECONFIG_H
