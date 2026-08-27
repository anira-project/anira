#ifndef ANIRA_HOSTCONFIG_H
#define ANIRA_HOSTCONFIG_H

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>

#include "../InferenceConfig.h"

namespace anira {

/**
 * @brief The resolved reference stream of a HostConfig
 *
 * Names the streamable tensor whose preprocess (input) or postprocess (output)
 * size is the unit in which HostConfig::m_buffer_size and
 * HostConfig::m_sample_rate are stated. Produced once by
 * HostConfig::resolve_reference() and stored by the session for the real-time
 * path, which never re-resolves it.
 */
struct ANIRA_API ReferenceStream {
    bool m_is_input = true;  ///< True if the reference tensor is an input, false if an output
    size_t m_index = 0;      ///< Index of the reference tensor within its input/output list
};

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
 * @par Reference stream
 * The buffer size and sample rate are stated in samples of one streamable tensor,
 * the reference stream. It is either selected explicitly (m_tensor_index together
 * with m_tensor_is_input) or, with the default m_tensor_index == k_first_streamable,
 * resolved automatically as the first streamable input tensor and, if no input is
 * streamable (a generator model whose inputs are all control parameters), the first
 * streamable output tensor. An explicit reference that is out of range or not
 * streamable is an error: resolve_reference() throws std::invalid_argument, and so
 * does prepare(). There is no silent fallback.
 *
 * @note This struct is designed to be lightweight and suitable for frequent
 *       copying and comparison operations in real-time contexts.
 */
struct ANIRA_API HostConfig {
    /**
     * @brief Sentinel for m_tensor_index: resolve the reference stream automatically
     *
     * The reference is the first streamable input tensor, or the first streamable
     * output tensor if no input is streamable. See resolve_reference().
     */
    static constexpr size_t k_first_streamable = static_cast<size_t>(-1);

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
     * @param host_buffer_size Buffer size of the host, in samples of the reference stream
     * @param host_sample_rate Sample rate of the host, in samples of the reference stream per
     * second
     * @param allow_smaller_buffers Whether to allow processing of buffers smaller than the host
     * buffer size (default: false)
     * @param tensor_index Index of the reference tensor (default: k_first_streamable, i.e. the
     * first streamable input, else the first streamable output)
     * @param tensor_is_input Whether tensor_index refers to an input (true, default) or an output
     * tensor (false); ignored when tensor_index is k_first_streamable
     */
    HostConfig(float host_buffer_size,
               float host_sample_rate,
               bool allow_smaller_buffers = false,
               size_t tensor_index = k_first_streamable,
               bool tensor_is_input = true)
        : m_buffer_size(host_buffer_size)
        , m_sample_rate(host_sample_rate)
        , m_allow_smaller_buffers(allow_smaller_buffers)
        , m_tensor_index(tensor_index)
        , m_tensor_is_input(tensor_is_input) {}

    float m_buffer_size = 0;               ///< Maximum buffer size of the host, in samples of the
                                           ///< reference stream
    float m_sample_rate = 0.0;             ///< Sample rate of the host in Hz, in samples of the
                                           ///< reference stream per second
    bool m_allow_smaller_buffers = false;  ///< Whether to allow processing of buffers smaller than
                                           ///< the maximum size
    size_t m_tensor_index = k_first_streamable;  ///< Index of the reference tensor, or
                                                 ///< k_first_streamable to resolve it automatically
    bool m_tensor_is_input = true;  ///< Whether m_tensor_index refers to an input (true) or an
                                    ///< output (false) tensor; ignored while m_tensor_index is
                                    ///< k_first_streamable

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
               m_tensor_index == other.m_tensor_index &&
               m_tensor_is_input == other.m_tensor_is_input;
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
     * @brief Resolves the reference stream against an inference configuration
     *
     * With m_tensor_index == k_first_streamable the reference is the first input tensor
     * with a non-zero preprocess_input_size, or, if there is none, the first output
     * tensor with a non-zero postprocess_output_size. Otherwise m_tensor_index names a
     * tensor in the input list (m_tensor_is_input == true) or the output list
     * (m_tensor_is_input == false), which must exist and be streamable.
     *
     * @param inference_config The inference configuration providing the tensor sizes
     * @return The resolved reference stream
     * @throws std::invalid_argument if an explicit reference is out of range or not
     *         streamable, or if no tensor on either side is streamable
     */
    ReferenceStream resolve_reference(const InferenceConfig& inference_config) const {
        const std::vector<size_t>& input_sizes = inference_config.get_preprocess_input_size();
        const std::vector<size_t>& output_sizes = inference_config.get_postprocess_output_size();

        if (m_tensor_index == k_first_streamable) {
            for (size_t i = 0; i < input_sizes.size(); ++i) {
                if (input_sizes[i] > 0) {
                    return ReferenceStream{.m_is_input = true, .m_index = i};
                }
            }
            for (size_t i = 0; i < output_sizes.size(); ++i) {
                if (output_sizes[i] > 0) {
                    return ReferenceStream{.m_is_input = false, .m_index = i};
                }
            }
            throw std::invalid_argument(
                "HostConfig: no streamable tensor on either side; the reference stream needs at "
                "least one input with preprocess_input_size > 0 or one output with "
                "postprocess_output_size > 0.");
        }

        const std::vector<size_t>& sizes = m_tensor_is_input ? input_sizes : output_sizes;
        const std::string side = m_tensor_is_input ? "input" : "output";
        if (m_tensor_index >= sizes.size()) {
            throw std::invalid_argument("HostConfig: reference tensor " + side + "[" +
                                        std::to_string(m_tensor_index) +
                                        "] is out of range (the model has " +
                                        std::to_string(sizes.size()) + " " + side + " tensors).");
        }
        if (sizes[m_tensor_index] == 0) {
            throw std::invalid_argument(
                "HostConfig: reference tensor " + side + "[" + std::to_string(m_tensor_index) +
                "] is not streamable (its " +
                (m_tensor_is_input ? "preprocess_input_size" : "postprocess_output_size") +
                " is 0).");
        }
        return ReferenceStream{.m_is_input = m_tensor_is_input, .m_index = m_tensor_index};
    }

    /**
     * @brief Size of the reference stream in samples per inference
     *
     * The preprocess_input_size (input reference) or postprocess_output_size (output
     * reference) of the tensor returned by resolve_reference(). All relative buffer
     * size and sample rate calculations scale against this value.
     *
     * @param inference_config The inference configuration providing the tensor sizes
     * @return The reference tensor's streamable size
     * @throws std::invalid_argument if the reference stream cannot be resolved
     */
    float get_reference_size(const InferenceConfig& inference_config) const {
        const ReferenceStream reference = resolve_reference(inference_config);
        if (reference.m_is_input) {
            return static_cast<float>(
                inference_config.get_preprocess_input_size()[reference.m_index]);
        }
        return static_cast<float>(
            inference_config.get_postprocess_output_size()[reference.m_index]);
    }

    /**
     * @brief Calculates the relative buffer size for a specific tensor
     *
     * Computes the appropriate buffer size for a given tensor based on the ratio
     * between this host configuration's buffer size and the reference stream's size.
     * This is useful when working with multiple tensors that may have different
     * dimensional requirements while maintaining proportional scaling.
     *
     * The calculation uses the reference stream (see resolve_reference()) to establish
     * a scaling ratio, then applies this ratio to the target tensor's dimensions.
     *
     * @param inference_config The inference configuration containing tensor dimension information
     * @param tensor_index The index of the tensor to calculate the buffer size for
     * @param input Whether to calculate for input tensors (true) or output tensors (false)
     * @return The calculated relative buffer size for the specified tensor
     * @throws std::invalid_argument if the reference stream cannot be resolved
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
     * between this host configuration's sample rate and the reference stream's size.
     * This is useful when different tensors represent audio data at different
     * effective sample rates due to processing or downsampling.
     *
     * The calculation uses the reference stream (see resolve_reference()) to establish
     * a scaling ratio, then applies this ratio to the target tensor's dimensions
     * to determine the effective sample rate.
     *
     * @param inference_config The inference configuration containing tensor dimension information
     * @param tensor_index The index of the tensor to calculate the sample rate for
     * @param input Whether to calculate for input tensors (true) or output tensors (false)
     * @return The calculated relative sample rate for the specified tensor
     * @throws std::invalid_argument if the reference stream cannot be resolved
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
};

}  // namespace anira

#endif  // ANIRA_HOSTCONFIG_H
