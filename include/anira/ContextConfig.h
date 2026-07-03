#ifndef ANIRA_CONTEXTCONFIG_H
#define ANIRA_CONTEXTCONFIG_H

#include <array>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "anira/system/AniraWinExports.h"
#include "anira/utils/InferenceBackend.h"

namespace anira {

/**
 * @brief Idle-wait strategy of the inference threads
 *
 * Controls how inference threads wait for new work when the global inference
 * queue is empty:
 *
 * - SpinBackoff: Exponential backoff — a short hot-spin phase, then a
 *   yield/sleep polling loop (~100 us period). Lowest possible pickup latency
 *   when work arrives within microseconds of the thread going idle, at the
 *   cost of continuous polling syscalls and CPU wakeups while idle.
 * - Blocking: Threads block on the queue's semaphore (futex) and are woken
 *   directly by the enqueue. No idle CPU usage and immediate wakeup, at the
 *   cost of one semaphore signal on the submitting thread (a bounded,
 *   non-blocking syscall when a consumer is asleep) and a scheduler wakeup
 *   latency of typically a few microseconds.
 *
 * @note Blocking is not available on WebAssembly builds, where inference loops
 * are driven cooperatively by JS Workers and blocking is not possible. There,
 * Context::get_instance and the JSON config loader coerce Blocking to
 * SpinBackoff and log a warning.
 *
 * @note All sessions in a process share one inference thread pool, so only one
 * strategy can be in effect per process: the one of the first-created context.
 * A later ContextConfig requesting a different strategy is ignored and
 * reported with a warning. Both strategies produce identical results — they
 * differ only in idle CPU usage and work-pickup latency.
 */
enum class WaitStrategy { SpinBackoff, Blocking };

/**
 * @brief Returns a human-readable name for a WaitStrategy value
 */
inline const char* to_string(WaitStrategy wait_strategy) {
    return wait_strategy == WaitStrategy::Blocking ? "blocking" : "spin_backoff";
}

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
     * @param num_threads Number of background inference threads to create.
     *                   Default: half of available CPU cores (minimum 1).
     *                   Pass 0 to opt out of the auto-managed pool and supply your
     *                   own threads via Context::make_inference_thread() (required on
     *                   WebAssembly, optional on native). When the Context singleton
     *                   already exists, num_threads == 0 leaves any existing pool
     *                   untouched — it signals "no preference," not "shrink to zero."
     * @param wait_strategy How idle inference threads wait for new work.
     *                   Default: WaitStrategy::SpinBackoff (see WaitStrategy for the
     *                   trade-offs). Must be identical across all ContextConfigs in
     *                   a process, since all sessions share one thread pool.
     *
     * @note The constructor automatically detects and registers available inference
     * backends based on compile-time definitions (USE_LIBTORCH, USE_ONNXRUNTIME, USE_TFLITE)
     */
    ContextConfig(unsigned int num_threads = (std::thread::hardware_concurrency() / 2 > 0)
                                                 ? std::thread::hardware_concurrency() / 2
                                                 : 1,
                  WaitStrategy wait_strategy = WaitStrategy::SpinBackoff)
        : m_num_threads(num_threads), m_wait_strategy(wait_strategy) {
#ifdef USE_LIBTORCH
        m_enabled_backends.push_back(InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        m_enabled_backends.push_back(InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
        m_enabled_backends.push_back(InferenceBackend::TFLITE);
#endif
#ifdef USE_LITERT
        m_enabled_backends.push_back(InferenceBackend::LITERT);
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
     * @brief Idle-wait strategy of the inference threads
     *
     * Determines whether idle inference threads poll the inference queue with
     * an exponential-backoff spin loop or block on the queue's semaphore until
     * work is enqueued. See WaitStrategy for the trade-offs.
     *
     * @note All sessions in a process share one inference thread pool, so only
     * one strategy can be in effect per process: the one of the first-created
     * context. A later ContextConfig requesting a different strategy is
     * ignored and reported with a warning. On WebAssembly builds, Blocking is
     * coerced to SpinBackoff with a warning (see WaitStrategy).
     */
    WaitStrategy m_wait_strategy = WaitStrategy::SpinBackoff;

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
        return m_num_threads == other.m_num_threads && m_wait_strategy == other.m_wait_strategy &&
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
    bool operator!=(const ContextConfig& other) const { return !(*this == other); }
};

}  // namespace anira

#endif  // ANIRA_CONTEXTCONFIG_H