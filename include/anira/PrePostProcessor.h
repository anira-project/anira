#ifndef ANIRA_PREPOSTPROCESSOR_H
#define ANIRA_PREPOSTPROCESSOR_H

#include "utils/RingBuffer.h"
#include "utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"
#include "InferenceConfig.h"
#include <atomic>
#include <vector>
#include <cassert>

namespace anira {

/**
 * @brief Abstract base class for preprocessing and postprocessing data for neural network inference
 * 
 * The PrePostProcessor class handles the transformation of data between the host application
 * and neural network inference engines. It provides default implementations for common use cases
 * and serves as a base class for custom preprocessing implementations.
 * 
 * The class supports two types of tensor data:
 * - **Streamable tensors**: Time-varying signals that flow continuously through ring buffers
 * - **Non-streamable tensors**: Static parameters or control values stored in thread-safe internal storage
 * 
 * @par Key Features:
 * - Thread-safe handling of non-streamable tensor data using atomic operations
 * - Helper methods for efficient buffer manipulation
 * - Support for multiple input/output tensors with different characteristics
 * - Real-time safe operations suitable for processing
 * 
 * @par Usage:
 * For models that operate in the time domain with simple input/output shapes, the default
 * implementation can be used directly. For custom preprocessing requirements (frequency domain
 * transforms, custom windowing, multi-tensor operations), inherit from this class and override
 * the pre_process() and post_process() methods.
 * 
 * @warning All methods are designed to be real-time safe and should not perform memory allocation
 * or other blocking operations when called from the audio thread.
 * 
 * @see InferenceConfig, RingBuffer, BufferF
 */
class ANIRA_API PrePostProcessor
{
public:
    /**
     * @brief Default constructor is deleted to prevent uninitialized instances
     */
    PrePostProcessor() = delete;
    
    /**
     * @brief Constructs a PrePostProcessor with the given inference configuration
     * 
     * Initializes internal storage for non-streamable tensors based on the configuration.
     * Streamable tensors (those with preprocess_input_size > 0 or postprocess_output_size > 0)
     * do not require internal storage as they use ring buffers directly.
     * 
     * @param inference_config Reference to the inference configuration containing tensor specifications
     */
    PrePostProcessor(InferenceConfig& inference_config);
    
    /**
     * @brief Default destructor
     */
    virtual ~PrePostProcessor() = default;

    /**
     * @brief Transforms input data from ring buffers to inference tensors
     * 
     * This method is called before neural network inference to prepare input data.
     * For streamable tensors, it extracts samples from ring buffers.
     * For non-streamable tensors, it retrieves values from internal storage.
     * 
     * @param input Vector of input ring buffers containing data from the host application
     * @param output Vector of output tensors that will be fed to the inference engine
     * @param current_inference_backend Currently active inference backend (for backend-specific processing)
     * 
     * @note This method is called from the audio thread and must be real-time safe
     * @see pop_samples_from_buffer(), get_input()
     */
    virtual void pre_process(std::vector<RingBuffer>& input, std::vector<BufferF>& output, [[maybe_unused]] InferenceBackend current_inference_backend);
    
    /**
     * @brief Transforms inference results to output ring buffers
     * 
     * This method is called after neural network inference to process the results.
     * For streamable tensors, it pushes samples to ring buffers.
     * For non-streamable tensors, it stores values in internal storage.
     * 
     * @param input Vector of input tensors containing inference results
     * @param output Vector of output ring buffers that will be read by the host application
     * @param current_inference_backend Currently active inference backend (for backend-specific processing)
     * 
     * @note This method is called from the audio thread and must be real-time safe
     * @see push_samples_to_buffer(), set_output()
     */
    virtual void post_process(std::vector<BufferF>& input, std::vector<RingBuffer>& output, [[maybe_unused]] InferenceBackend current_inference_backend);

    /**
     * @brief Sets a non-streamable input value in thread-safe storage
     * 
     * Used to store control parameters or static values that don't change sample-by-sample.
     * The data is stored using atomic operations for thread safety.
     * 
     * @param input The value to store
     * @param i Tensor index (which input tensor)
     * @param j Sample index within the tensor
     * 
     * @warning Only use for tensors where preprocess_input_size == 0
     * @see get_input()
     */
    void set_input(const float& input, size_t i, size_t j);
    
    /**
     * @brief Sets a non-streamable output value in thread-safe storage
     * 
     * Used to store control parameters or static values from inference results.
     * The data is stored using atomic operations for thread safety.
     * 
     * @param output The value to store
     * @param i Tensor index (which output tensor)
     * @param j Sample index within the tensor
     * 
     * @warning Only use for tensors where postprocess_output_size == 0
     * @see get_output()
     */
    void set_output(const float& output, size_t i, size_t j);
    
    /**
     * @brief Retrieves a non-streamable input value from thread-safe storage
     * 
     * Used to read control parameters or static values in a thread-safe manner.
     * 
     * @param i Tensor index (which input tensor)
     * @param j Sample index within the tensor
     * @return The stored input value
     * 
     * @warning Only use for tensors where preprocess_input_size == 0
     * @see set_input()
     */
    float get_input(size_t i, size_t j);
    
    /**
     * @brief Retrieves a non-streamable output value from thread-safe storage
     * 
     * Used to read inference results or control parameters in a thread-safe manner.
     * 
     * @param i Tensor index (which output tensor)
     * @param j Sample index within the tensor
     * @return The stored output value
     * 
     * @warning Only use for tensors where postprocess_output_size == 0
     * @see set_output()
     */
    float get_output(size_t i, size_t j);

    /**
     * @brief Extracts samples from a ring buffer to an output tensor
     * 
     * Pops the specified number of samples from the ring buffer and writes them
     * to the output tensor. For multi-channel inputs, samples are interleaved
     * in the output buffer (channel 0 samples first, then channel 1, etc.).
     * 
     * @param input Source ring buffer
     * @param output Destination tensor buffer
     * @param num_samples Number of samples to extract per channel
     * 
     * @note Real-time safe operation
     */
    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_samples);
    
    /**
     * @brief Extracts samples with overlapping windows from a ring buffer
     * 
     * Combines new samples with previously extracted samples to create overlapping windows.
     * This is useful for models that require context from previous inference steps.
     * 
     * @param input Source ring buffer
     * @param output Destination tensor buffer
     * @param num_new_samples Number of new samples to extract per channel
     * @param num_old_samples Number of samples to retain from previous extraction
     * 
     * @note Real-time safe operation
     * @see pop_samples_from_buffer(RingBuffer&, BufferF&, size_t, size_t, size_t)
     */
    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples);
    
    /**
     * @brief Extracts samples with overlapping windows and offset
     * 
     * Advanced version that allows specifying an offset in the output buffer where
     * the extracted samples should be written. Useful for batched processing or
     * complex tensor layouts.
     * 
     * @param input Source ring buffer
     * @param output Destination tensor buffer
     * @param num_new_samples Number of new samples to extract per channel
     * @param num_old_samples Number of samples to retain from previous extraction
     * @param offset Starting position in the output buffer for writing samples
     * 
     * @note Real-time safe operation
     */
    void pop_samples_from_buffer(RingBuffer& input, BufferF& output, size_t num_new_samples, size_t num_old_samples, size_t offset);
    
    /**
     * @brief Writes samples from a tensor to a ring buffer
     * 
     * Pushes samples from the input tensor to the ring buffer. For multi-channel outputs,
     * assumes samples are interleaved in the input buffer (channel 0 samples first,
     * then channel 1, etc.).
     * 
     * @param input Source tensor buffer
     * @param output Destination ring buffer
     * @param num_samples Number of samples to write per channel
     * 
     * @note Real-time safe operation
     */
    void push_samples_to_buffer(const BufferF& input, RingBuffer& output, size_t num_samples);

protected:
    /**
     * @brief Reference to the inference configuration
     * 
     * Provides access to tensor shapes, sizes, and processing parameters needed
     * for proper data transformation between buffers and neural network tensors.
     */
    InferenceConfig& m_inference_config;
    
private:
    /**
     * @brief Thread-safe storage for non-streamable input tensors
     * 
     * Vector of atomic float arrays for storing input parameters that don't change
     * sample-by-sample (e.g., control parameters, static values).
     */
    std::vector<MemoryBlock<std::atomic<float>>> m_inputs;
    
    /**
     * @brief Thread-safe storage for non-streamable output tensors
     * 
     * Vector of atomic float arrays for storing output parameters that don't change
     * sample-by-sample (e.g., peak values, analysis results).
     */
    std::vector<MemoryBlock<std::atomic<float>>> m_outputs;

#if DOXYGEN
    // Placeholder for Doxygen documentation
    // Since Doxygen does not find classes structures nested in std::vectors
    MemoryBlock<std::atomic<float>>* __doxygen_force_0; ///< Placeholder for Doxygen documentation
#endif
};

} // namespace anira

#endif //ANIRA_PREPOSTPROCESSOR_H