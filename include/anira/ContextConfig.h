#ifndef ANIRA_CONTEXTCONFIG_H
#define ANIRA_CONTEXTCONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include "anira/utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"

namespace anira {

/**
 * @brief Configuration structure for the inference context and threading behavior
 * 
 * The ContextConfig struct controls global settings for the anira inference system,
 * including thread pool management and available inference backends. This configuration
 * is shared across all inference sessions within a single context instance.
 * 
 * @par Usage Examples:
 * @code
 * // Use default configuration (half of available CPU cores)
 * anira::ContextConfig default_config;
 * 
 * // Specify custom thread count
 * anira::ContextConfig custom_config(4);
 * 
 * // Use with InferenceHandler
 * anira::InferenceHandler handler(pp_processor, inference_config, custom_config);
 * @endcode
 * 
 * @note This configuration affects global behavior and should be set once during
 * application initialization. Changing context configuration during runtime
 * requires recreating the context and all associated sessions.
 * 
 * @see Context, InferenceHandler, InferenceBackend
 */
struct ANIRA_API ContextConfig {
    /**
     * @brief Constructs a ContextConfig with specified thread count
     * 
     * Initializes the context configuration with the given number of inference threads
     * and automatically populates the list of available backends based on compile-time
     * feature flags.
     * 
     * @param num_threads Number of background inference threads to create
     *                   Default: Half of available CPU cores (minimum 1)
     * 
     * @note The constructor automatically detects and registers available inference
     * backends based on compile-time definitions (USE_LIBTORCH, USE_ONNXRUNTIME, USE_TFLITE)
     */
    ContextConfig(
            unsigned int num_threads = (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2 : 1) :
            m_num_threads(num_threads)
    {
#ifdef USE_LIBTORCH
        m_enabled_backends.push_back(InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        m_enabled_backends.push_back(InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
        m_enabled_backends.push_back(InferenceBackend::TFLITE);
#endif
    }

    /**
     * @brief Number of background inference threads
     * 
     * Controls the size of the thread pool used for neural network inference.
     * These threads run at high priority to minimize inference latency and are
     * shared across all inference sessions within the context.
     * 
     * @note This value is set during construction and cannot be changed without
     * recreating the context. All inference sessions using this context will
     * share the same thread pool.
     */
    unsigned int m_num_threads;
    
    /**
     * @brief Version string of the anira library
     * 
     * Contains the version of the anira library that was used to create this
     * configuration. This is useful for debugging, logging, and ensuring
     * compatibility when serializing/deserializing configurations.
     * 
     * @note This field is automatically populated with the ANIRA_VERSION
     * macro during construction and should not be modified manually.
     */
    std::string m_anira_version = ANIRA_VERSION;
    
    /**
     * @brief List of available inference backends
     * 
     * Contains all inference backends that were detected as available during
     * compilation. This list is automatically populated in the constructor
     * based on compile-time feature flags:
     * 
     * - InferenceBackend::LIBTORCH (if USE_LIBTORCH is defined)
     * - InferenceBackend::ONNX (if USE_ONNXRUNTIME is defined)
     * - InferenceBackend::TFLITE (if USE_TFLITE is defined)
     * - InferenceBackend::CUSTOM (always available)
     * 
     * @note The CUSTOM backend is not automatically added to this list but is
     * always available for use with custom backend implementations.
     */
    std::vector<InferenceBackend> m_enabled_backends;
    
private:
    /**
     * @brief Equality comparison operator
     * 
     * Compares two ContextConfig instances for equality by checking all member
     * variables. This is used internally for configuration validation and
     * context management.
     * 
     * @param other The ContextConfig to compare against
     * @return true if all configuration parameters match, false otherwise
     **/
    bool operator==(const ContextConfig& other) const {
        return
            m_num_threads == other.m_num_threads &&
            m_anira_version == other.m_anira_version &&
            m_enabled_backends == other.m_enabled_backends;
    }

    /**
     * @brief Inequality comparison operator
     * 
     * Compares two ContextConfig instances for inequality. This is the logical
     * inverse of the equality operator.
     * 
     * @param other The ContextConfig to compare against
     * @return true if any configuration parameters differ, false otherwise
     * 
     **/
    bool operator!=(const ContextConfig& other) const {
        return !(*this == other);
    }

};

} // namespace anira

#endif //ANIRA_CONTEXTCONFIG_H