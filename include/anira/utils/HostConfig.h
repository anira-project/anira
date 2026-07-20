#ifndef ANIRA_HOSTCONFIG_H
#define ANIRA_HOSTCONFIG_H

#include <cstddef>
#include <functional>

#include "../InferenceConfig.h"

namespace anira {

/**
 * @brief Configuration structure for host system parameters
 *
 * The HostConfig struct encapsulates the host system's configuration parameters
 * that are needed for proper integration with neural network inference processing.
 * It defines the buffer characteristics, sample rate, and processing constraints
 * that the inference system must adapt to.
 *
 * The struct provides utility methods for calculating relative buffer sizes and
 * sample rates when working with multiple tensors that may have different
 * processing requirements or dimensions.
 *
 * @note This struct is designed to be lightweight and suitable for frequent
 *       copying and comparison operations in real-time contexts.
 */
struct ANIRA_API HostConfig {
    /**
     * @brief Default constructor that creates an empty host configuration
     *
     * Initializes all parameters to default values (zero buffer size, zero sample rate).
     * The configuration must be properly initialized before use in audio processing.
     */
    HostConfig() = default;

    /**
     * @brief Constructor that initializes host configuration with specified parameters
     *
     * Creates a host configuration with the specified audio system parameters.
     * This constructor allows full customization of the audio host environment.
     *
     * @param host_buffer_size Buffer size of the host
     * @param host_sample_rate Sample rate of the host
     * @param allow_smaller_buffers Whether to allow processing of buffers smaller than the host
     * buffer size (default: false)
     * @param input_tensor_index Index of the primary input tensor for buffer size calculations
     * (default: 0)
     */
    HostConfig(float host_buffer_size,
               float host_sample_rate,
               bool allow_smaller_buffers = false,
               size_t input_tensor_index = 0)
        : m_buffer_size(host_buffer_size)
        , m_sample_rate(host_sample_rate)
        , m_allow_smaller_buffers(allow_smaller_buffers)
        , m_tensor_index(input_tensor_index) {}

    float m_buffer_size = 0;               ///< Maximum size of the input buffer from the host
    float m_sample_rate = 0.0;             ///< Sample rate of the host in Hz
    bool m_allow_smaller_buffers = false;  ///< Whether to allow processing of buffers smaller than
                                           ///< the maximum size
    size_t m_tensor_index = 0;  ///< Index of the tensor used as reference for buffer size
                                ///< calculations

    /**
     * @brief Equality comparison operator
     *
     * Compares two HostConfig instances for equality using appropriate tolerance
     * for floating-point comparisons. All member variables must match within
     * acceptable precision for the configs to be considered equal.
     *
     * @param other The HostConfig instance to compare with
     * @return True if both configurations are equivalent, false otherwise
     *
     * @note Floating-point comparisons use a tolerance of 1e-6 to handle
     *       precision issues in floating-point arithmetic.
     */
    bool operator==(const HostConfig& other) const {
        return std::abs(m_buffer_size - other.m_buffer_size) < 1e-6 &&
               std::abs(m_sample_rate - other.m_sample_rate) < 1e-6 &&
               m_allow_smaller_buffers == other.m_allow_smaller_buffers &&
               m_tensor_index == other.m_tensor_index;
    }

    /**
     * @brief Inequality comparison operator
     *
     * Compares two HostConfig instances for inequality by negating the equality operator.
     *
     * @param other The HostConfig instance to compare with
     * @return True if the configurations are different, false if they are equivalent
     */
    bool operator!=(const HostConfig& other) const { return !(*this == other); }

    /**
     * @brief Calculates the relative buffer size for a specific tensor
     *
     * Computes the appropriate buffer size for a given tensor based on the ratio
     * between this host configuration's buffer size and the reference tensor's size.
     * This is useful when working with multiple tensors that may have different
     * dimensional requirements while maintaining proportional scaling.
     *
     * The calculation uses the reference tensor (m_tensor_index) to establish a
     * scaling ratio, then applies this ratio to the target tensor's dimensions.
     *
     * @param inference_config The inference configuration containing tensor dimension information
     * @param tensor_index The index of the tensor to calculate the buffer size for
     * @param input Whether to calculate for input tensors (true) or output tensors (false)
     * @return The calculated relative buffer size for the specified tensor
     *
     * @note The returned value maintains the proportional relationship between
     *       different tensor sizes based on the host buffer configuration.
     */
    float get_relative_buffer_size(const InferenceConfig& inference_config,
                                   size_t tensor_index,
                                   bool input = true) const {
        float const ratio_buffer_size = m_buffer_size / get_reference_size(inference_config);
        if (input) {
            return static_cast<float>(inference_config.get_preprocess_input_size()[tensor_index]) *
                   ratio_buffer_size;
        } else {
            return static_cast<float>(
                       inference_config.get_postprocess_output_size()[tensor_index]) *
                   ratio_buffer_size;
        }
    }

    /**
     * @brief Calculates the relative sample rate for a specific tensor
     *
     * Computes the appropriate sample rate for a given tensor based on the ratio
     * between this host configuration's sample rate and the reference tensor's size.
     * This is useful when different tensors represent audio data at different
     * effective sample rates due to processing or downsampling.
     *
     * The calculation uses the reference tensor (m_tensor_index) to establish a
     * scaling ratio, then applies this ratio to the target tensor's dimensions
     * to determine the effective sample rate.
     *
     * @param inference_config The inference configuration containing tensor dimension information
     * @param tensor_index The index of the tensor to calculate the sample rate for
     * @param input Whether to calculate for input tensors (true) or output tensors (false)
     * @return The calculated relative sample rate for the specified tensor
     *
     * @note This method is useful for handling models that process audio at
     *       different effective sample rates or with different temporal resolutions.
     */
    float get_relative_sample_rate(const InferenceConfig& inference_config,
                                   size_t tensor_index,
                                   bool input = true) const {
        float const ratio_sample_rate = m_sample_rate / get_reference_size(inference_config);
        if (input) {
            return static_cast<float>(inference_config.get_preprocess_input_size()[tensor_index]) *
                   ratio_sample_rate;
        } else {
            return static_cast<float>(
                       inference_config.get_postprocess_output_size()[tensor_index]) *
                   ratio_sample_rate;
        }
    }

    /**
     * @brief Size of the reference stream used for relative buffer/rate scaling
     *
     * The reference is the input tensor selected by m_tensor_index. If that
     * tensor is non-streamable (preprocess size 0) — e.g. generator-style
     * models whose inputs are all control parameters — the reference falls
     * back to the first streamable input, then to the first streamable
     * output. Without the fallback the ratio computations above divide by
     * zero, and the resulting inf ends up in SessionElement::prepare()'s
     * smaller-buffer countdown loop, which never terminates (inf - 1 == inf).
     *
     * @param inference_config The inference configuration containing tensor sizes
     * @return The reference tensor's streamable size, or 1.0f if no tensor is
     *         streamable (ratios are meaningless then, but stay finite)
     */
    float get_reference_size(const InferenceConfig& inference_config) const {
        const auto& input_sizes = inference_config.get_preprocess_input_size();
        if (m_tensor_index < input_sizes.size() && input_sizes[m_tensor_index] > 0) {
            return static_cast<float>(input_sizes[m_tensor_index]);
        }
        for (const auto size : input_sizes) {
            if (size > 0) { return static_cast<float>(size); }
        }
        for (const auto size : inference_config.get_postprocess_output_size()) {
            if (size > 0) { return static_cast<float>(size); }
        }
        return 1.0f;
    }
};

}  // namespace anira

#endif  // ANIRA_HOSTCONFIG_H