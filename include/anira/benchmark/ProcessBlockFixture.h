#ifndef ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H
#define ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H

#include <benchmark/benchmark.h>
#include <iomanip>
#include "../utils/helperFunctions.h"
#include "../anira.h"

namespace anira {
namespace benchmark {

/**
 * @brief Benchmark fixture class for testing neural network inference performance in audio processing contexts
 * 
 * The ProcessBlockFixture provides a specialized Google Benchmark fixture for measuring the performance
 * of neural network inference operations in real-time audio processing scenarios. It manages test setup,
 * data preparation, timing measurements, and cleanup for benchmarking different inference backends,
 * model configurations, and buffer sizes.
 * 
 * This fixture supports:
 * - Performance testing across different inference backends (ONNX, LibTorch, TensorFlow Lite, Custom)
 * - Benchmarking with various audio buffer sizes and sample rates
 * - Detailed timing measurements and reporting for each iteration and repetition
 * - Automated test data generation and buffer management
 * - Cross-platform timing precision handling
 * 
 * @note This class inherits from Google Benchmark's Fixture class and should be used with
 *       the BENCHMARK_F macro for defining benchmark tests.
 */
class ANIRA_API ProcessBlockFixture : public ::benchmark::Fixture {
public:

    /**
     * @brief Default constructor that initializes the benchmark fixture
     * 
     * Sets up initial state for benchmarking, including buffer size and repetition counters.
     * A new instance is created for each benchmark that has been defined and registered.
     */
    ProcessBlockFixture();
    
    /**
     * @brief Destructor that cleans up benchmark resources
     */
    ~ProcessBlockFixture();

    /**
     * @brief Initializes the current benchmark iteration
     * 
     * Prepares the benchmark for a new iteration by capturing the current state
     * of processed samples. This method should be called at the beginning of each
     * benchmark iteration to establish a baseline for measuring progress.
     */
    void initialize_iteration();
    
    /**
     * @brief Initializes a new benchmark repetition with specified configurations
     * 
     * Sets up the benchmark environment for a new repetition, including inference
     * configuration, audio settings, and backend selection. Handles switching between
     * different configurations and provides detailed logging of the test setup.
     * 
     * @param inference_config Reference to the inference configuration containing model settings
     * @param host_config Audio host configuration with sample rate and buffer size settings
     * @param inference_backend The backend type to use for inference (ONNX, LibTorch, etc.)
     * @param sleep_after_repetition Whether to sleep after each repetition to allow system cooldown (default: true)
     */
    void initialize_repetition(const InferenceConfig& inference_config, const HostConfig& host_config, const InferenceBackend& inference_backend, bool sleep_after_repetition = true);
    
    /**
     * @brief Checks if the current audio buffer has been fully processed
     * 
     * Determines whether the inference handler has processed all samples that were
     * pushed since the last iteration initialization.
     * 
     * @return True if the buffer has been processed, false otherwise
     */
    bool buffer_processed();
    
    /**
     * @brief Fills the audio buffer with random sample data
     * 
     * Generates random audio samples and populates the internal buffer according to
     * the specified host configuration. Used to create test data for benchmarking.
     * 
     * @param host_config Audio host configuration specifying buffer size and channel count
     */
    void push_random_samples_in_buffer(anira::HostConfig host_config);
    
    /**
     * @brief Gets the current buffer size being used for benchmarking
     * 
     * @return The buffer size in samples
     */
    int get_buffer_size();
    
    /**
     * @brief Gets the current repetition number
     * 
     * @return The number of the current benchmark repetition
     */
    int get_repetition();

/**
     * @brief Records timing information for a single benchmark iteration (Windows/macOS)
     * 
     * Measures and records the elapsed time for a single benchmark iteration using
     * steady_clock for timing. Updates benchmark state and provides detailed logging
     * of iteration results.
     * 
     * @param start The start time point of the iteration
     * @param end The end time point of the iteration  
     * @param state Reference to the benchmark state for recording results
     */
        void interation_step(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, ::benchmark::State& state);

    /**
     * @brief Finalizes the current benchmark repetition
     * 
     * Performs cleanup and logging tasks at the end of a benchmark repetition.
     * Increments the repetition counter and provides visual separation in the output.
     */
    void repetition_step();

    /**
     * @brief Static shared pointer to the inference handler used across benchmark instances
     * 
     * This shared inference handler is used by all benchmark instances to perform
     * neural network inference operations. It's managed as a static member to avoid
     * repeated initialization costs during benchmarking.
     */
    inline static std::unique_ptr<anira::InferenceHandler> m_inference_handler = nullptr;
    
    /**
     * @brief Static shared pointer to the audio buffer used for benchmark data
     * 
     * This shared buffer holds audio data for benchmarking operations. It's managed
     * as a static member to enable efficient data sharing across benchmark instances.
     */
    inline static std::unique_ptr<anira::Buffer<float>> m_buffer = nullptr;

private:
    int m_buffer_size = 0;                    ///< Current buffer size being benchmarked
    int m_repetition = 0;                     ///< Current repetition number
    bool m_sleep_after_repetition = true;     ///< Whether to sleep after repetitions for system cooldown
    int m_iteration = 0;                      ///< Current iteration number within a repetition
    std::chrono::duration<double, std::milli> m_runtime_last_repetition = std::chrono::duration<double, std::milli>(0); ///< Runtime of the last repetition
    int m_prev_num_received_samples = 0;      ///< Number of samples received at the start of current iteration
    std::string m_model_name;                 ///< Name of the model being benchmarked
    std::string m_inference_backend_name;     ///< Name of the inference backend being used
    InferenceBackend m_inference_backend;     ///< Current inference backend configuration
    InferenceConfig m_inference_config;       ///< Current inference configuration
    HostConfig m_host_config;                 ///< Current host audio configuration

    /**
     * @brief Sets up the benchmark fixture before each benchmark run
     * 
     * Google Benchmark framework callback that prepares the fixture state
     * before benchmark execution begins. Updates buffer size based on benchmark parameters.
     * 
     * @param state The benchmark state containing runtime parameters
     */
    void SetUp(const ::benchmark::State& state);
    
    /**
     * @brief Cleans up the benchmark fixture after each benchmark run
     * 
     * Google Benchmark framework callback that performs cleanup operations
     * after benchmark execution completes. Resets shared resources and optionally
     * sleeps for system cooldown.
     * 
     * @param state The benchmark state containing runtime parameters
     */
    void TearDown(const ::benchmark::State& state);
};

} // namespace benchmark
} // namespace anira

#endif // ANIRA_BENCHMARK_PROCESSBLOCKFIXTURE_H