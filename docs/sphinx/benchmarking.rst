Benchmarking Guide
==================

Overview
--------

anira facilitates the measurement of neural network's real-time performance by providing a custom benchmark fixture class within the `Google Benchmark <https://github.com/google/benchmark>`_ framework - the :cpp:class:`anira::benchmark::ProcessBlockFixture`. This fixture class constructs a static instance of the :cpp:class:`anira::InferenceHandler` class and measures the runtimes of several consecutive calls to the :cpp:func:`anira::InferenceHandler::process` method.

The fixture is designed to:

- Measure runtime performance across multiple iterations
- Compare first inference with subsequent ones to detect warm-up requirements
- Test different configurations (buffer sizes, inference backends)
- Estimate maximum inference times for real-time constraints

.. note::
    To use anira's benchmarking capabilities, you should first become familiar with the main anira usage patterns. The benchmarking tools build upon the same :cpp:class:`anira::InferenceHandler`, :cpp:struct:`anira::InferenceConfig`, and :cpp:class:`anira::PrePostProcessor` classes described in the :doc:`usage` section. The benchmarking fixture is a specialized tool that extends the standard usage patterns to measure performance in a controlled manner.

Prerequisites
-------------

Before using the benchmarking tools, ensure that:

1. anira was built with the ``ANIRA_BUILD_BENCHMARK`` option set to ``ON``
2. You have a working :cpp:struct:`anira::InferenceConfig` and :cpp:class:`anira::PrePostProcessor` setup
3. Your model files are accessible and properly configured

In the cmake configuration, the build system automatically links Google Benchmark and Google Test libraries when anira is built with benchmarking support.

Single Configuration Benchmarking
----------------------------------

1. Define the Benchmark Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating your benchmark using the ``BENCHMARK_DEFINE_F`` macro. The fixture handles the creation and management of the :cpp:class:`anira::InferenceHandler` instance:

.. code-block:: cpp
    :caption: benchmark.cpp

    #include <gtest/gtest.h>
    #include <benchmark/benchmark.h>
    #include <anira/anira.h>
    #include <anira/benchmark.h>

    typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

    // Configure your inference setup (same as in regular usage)
    anira::InferenceConfig my_inference_config(
        // ... your model configuration
    );
    anira::PrePostProcessor my_pp_processor(my_inference_config);

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {
        // Define the host configuration for the benchmark
        anira::HostConfig host_config(BUFFER_SIZE, SAMPLE_RATE);
        anira::InferenceBackend inference_backend = anira::InferenceBackend::ONNX;

        // Create and prepare the InferenceHandler instance
        m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor, my_inference_config);
        m_inference_handler->prepare(host_config);
        m_inference_handler->set_inference_backend(inference_backend);

        // Create the input buffer
        m_buffer = std::make_unique<anira::Buffer<float>>(
            my_inference_config.get_preprocess_input_channels()[0], 
            host_config.m_buffer_size
        );

        // Initialize the repetition (enables configuration tracking and optional sleep)
        initialize_repetition(my_inference_config, host_config, inference_backend, true);

.. note::
    The :cpp:func:`initialize_repetition` method sets up the benchmark fixture, allowing you to track configuration changes and optionally sleep between repetitions for thermal stability. The first parameter is the inference configuration, the second is the host configuration, the third is the inference backend, and the fourth controls whether to sleep after each repetition. The sleep duration is equal to the time taken to process all iterations, allowing for thermal cooldown between repetitions.

2. Measure Process Method Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the main measurement loop using the Google Benchmark framework's state control:

.. code-block:: cpp
    :caption: benchmark.cpp

        // Main benchmark loop
        for (auto _ : state) {
            // Fill buffer with random samples in range [-1.0, 1.0]
            push_random_samples_in_buffer(host_config);

            // Initialize iteration tracking
            initialize_iteration();

            // Begin timing measurement
            auto start = std::chrono::high_resolution_clock::now();
            
            // Process the buffer (this triggers inference)
            m_inference_handler->process(m_buffer->get_array_of_write_pointers(), get_buffer_size());

            // Wait for processing completion (inference is asynchronous)
            while (!buffer_processed()) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(10));
            }
            
            // End timing measurement
            auto end = std::chrono::high_resolution_clock::now();

            // Record the measured runtime
            interation_step(start, end, state);
        }
        
        // Clean up after all iterations complete
        repetition_step();
    }

.. note::
    The :cpp:func:`anira::InferenceHandler::process` method operates asynchronously. To ensure accurate timing measurements, you must wait for the :cpp:func:`buffer_processed` method to return ``true`` before stopping the timer. This guarantees that the measured time includes the complete processing duration, not just the time to initiate processing.

3. Register the Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

Configure and register your benchmark with the Google Benchmark framework:

.. code-block:: cpp
    :caption: benchmark.cpp

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(NUM_ITERATIONS)
        ->Repetitions(NUM_REPETITIONS)
        ->UseManualTime();

The key parameters are:

- **Unit**: Specify the time unit for results (e.g., ``benchmark::kMillisecond``)
- **Iterations**: Number of :cpp:func:`anira::InferenceHandler::process` calls per repetition
- **Repetitions**: Number of times to repeat the entire benchmark
- **UseManualTime**: Required since we manually measure processing time

4. CMake Configuration
~~~~~~~~~~~~~~~~~~~~~~

Set up your CMake project to build and test the benchmark:

.. code-block:: cmake

    project(benchmark_project)

    # Enable benchmarking in anira
    set(ANIRA_BUILD_BENCHMARK ON)
    add_subdirectory(anira)

    # Create benchmark executable
    add_executable(benchmark_target benchmark.cpp)
    target_link_libraries(benchmark_target anira::anira)

5. Run the Benchmark
~~~~~~~~~~~~~~~~~~~~

You can then simply execute your benchmark executable:

.. code-block:: bash

    ./build/benchmark_target

Or use Google Test to integrate it with your test suite:

Create Unit Test Integration
----------------------------

Write a Google Test case
~~~~~~~~~~~~~~~~~~~~~~~~

Integrate the benchmark with Google Test for easy execution:

.. code-block:: cpp
    :caption: test.cpp

    #include <benchmark/benchmark.h>
    #include <gtest/gtest.h>
    #include <anira/anira.h>

    TEST(Benchmark, Simple) {
        // Elevate process priority for more consistent timing
    #if __linux__ || __APPLE__
        pthread_t self = pthread_self();
    #elif WIN32
        HANDLE self = GetCurrentThread();
    #endif
        anira::HighPriorityThread::elevate_priority(self, true);

        // Execute the benchmark
        benchmark::RunSpecifiedBenchmarks();
    }

Integrate via CMake
~~~~~~~~~~~~~~~~~~~

Set up your CMake project to include the benchmark and find the test:

.. code-block:: cmake

    project(benchmark_project)

    # Enable benchmarking in anira
    set(ANIRA_BUILD_BENCHMARK ON)
    add_subdirectory(anira)

    # Create benchmark executable
    add_executable(benchmark_target benchmark.cpp)
    target_link_libraries(benchmark_target anira::anira)

    # Add Google Test support
    enable_testing()
    gtest_discover_tests(benchmark_target)

Run the Test
~~~~~~~~~~~~

Execute your benchmark using CTest:

.. code-block:: bash

    # Run all tests with verbose output
    ctest -VV

    # Run specific benchmark test
    ctest -R Benchmark.Simple -VV

    # For long-running benchmarks, increase timeout
    ctest --timeout 100000 -VV

.. note::
    Test outputs are stored in the ``Testing`` directory of your build folder. Use the ``-VV`` flag to see detailed benchmark results in the console.

Multiple Configuration Benchmarking
------------------------------------

For comprehensive performance analysis, you can benchmark multiple configurations by passing arguments to your benchmark functions.

Single Argument Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~

Test different buffer sizes by passing arguments during registration:

.. code-block:: cpp
    :caption: benchmark.cpp

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)(::benchmark::State& state) {
        // Use state.range(0) to get the buffer size argument
        anira::HostConfig host_config = {(size_t) state.range(0), SAMPLE_RATE};
        anira::InferenceBackend inference_backend = anira::InferenceBackend::ONNX;

        m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor, my_inference_config);
        m_inference_handler->prepare(host_config);
        m_inference_handler->set_inference_backend(inference_backend);

        m_buffer = std::make_unique<anira::Buffer<float>>(
            my_inference_config.get_preprocess_input_channels()[0], 
            host_config.m_buffer_size
        );

        initialize_repetition(my_inference_config, host_config, inference_backend);

        // ... measurement loop (same as single configuration)
    }

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(50)
        ->Repetitions(10)
        ->UseManualTime()
        ->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

.. warning::
    Currently, the :cpp:class:`anira::benchmark::ProcessBlockFixture` requires buffer sizes that are multiples of the model output size. The :cpp:func:`buffer_processed` function may not return ``true`` for other buffer sizes.

Multiple Argument Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex configuration testing, define argument combinations using a custom function:

.. code-block:: cpp
    :caption: benchmark.cpp

    // Define test configurations
    std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<anira::InferenceBackend> inference_backends = {
        anira::InferenceBackend::LIBTORCH, anira::InferenceBackend::ONNX, anira::InferenceBackend::TFLITE, anira::InferenceBackend::CUSTOM
    };
    std::vector<anira::InferenceConfig> inference_configs = {
        cnn_config, hybridnn_config, rnn_config
    };

    // Define argument combinations
    static void Arguments(::benchmark::internal::Benchmark* b) {
        for (int i = 0; i < buffer_sizes.size(); ++i) {
            for (int j = 0; j < inference_configs.size(); ++j) {
                for (int k = 0; k < inference_backends.size(); ++k) {
                    // Skip incompatible combinations (e.g., ONNX + stateful RNN)
                    if (!(j == 2 && k == 1)) {
                        b->Args({buffer_sizes[i], j, k});
                    }
                }
            }
        }
    }

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)(::benchmark::State& state) {
        // Extract configuration from arguments
        anira::HostConfig host_config = {(size_t) state.range(0), SAMPLE_RATE};
        anira::InferenceConfig& inference_config = inference_configs[state.range(1)];
        anira::InferenceBackend inference_backend = inference_backends[state.range(2)];

        // Setup with selected configuration
        anira::PrePostProcessor pp_processor(inference_config);
        m_inference_handler = std::make_unique<anira::InferenceHandler>(pp_processor, inference_config);
        m_inference_handler->prepare(host_config);
        m_inference_handler->set_inference_backend(inference_backend);

        m_buffer = std::make_unique<anira::Buffer<float>>(
            inference_config.get_preprocess_input_channels()[0], 
            host_config.m_buffer_size
        );

        initialize_repetition(inference_config, host_config, inference_backend);

        // ... measurement loop
    }

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(NUM_ITERATIONS)
        ->Repetitions(NUM_REPETITIONS)
        ->UseManualTime()
        ->Apply(Arguments);

Specialized Benchmarking Scenarios
-----------------------------------

Benchmarking Without Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To measure only pre/post-processing overhead without actual inference:

.. code-block:: cpp
    :caption: benchmark.cpp

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_NO_INFERENCE)(::benchmark::State& state) {
        // ... setup code ...
        
        // Use CUSTOM backend with default processor (performs roundtrip without inference)
        m_inference_handler->set_inference_backend(anira::InferenceBackend::CUSTOM);
        
        // ... measurement loop ...
    }

This configuration measures the overhead of anira's processing pipeline without the neural network inference step.

Benchmarking Custom Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom inference backend implementations:

.. code-block:: cpp
    :caption: benchmark.cpp

    // Define your custom processor class first
    class MyCustomProcessor : public anira::BackendBase {
        // ... implement your custom inference logic ...
    };

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_CUSTOM_INFERENCE)(::benchmark::State& state) {
        // ... setup code ...
        
        // Register your custom processor
        // (implementation depends on your custom backend design)
        
        m_inference_handler->set_inference_backend(anira::InferenceBackend::CUSTOM);
        
        // ... measurement loop ...
    }

Best Practices
--------------

1. **Consistent Environment**: Run benchmarks on a dedicated system with minimal background processes
2. **Thermal Management**: Use the sleep option in :cpp:func:`initialize_repetition` for thermal stability
3. **Multiple Repetitions**: Use sufficient repetitions to account for system variability
4. **Priority Elevation**: Always elevate process priority for consistent timing measurements
5. **Warm-up Analysis**: Compare first vs. subsequent iterations to identify warm-up requirements
6. **Configuration Coverage**: Test realistic buffer sizes and configurations for your target use case

Interpreting Results
--------------------

Benchmark results include:

- **Mean Processing Time**: Average time per process call
- **Standard Deviation**: Timing variability indicator  
- **Min/Max Times**: Best and worst case performance
- **Iterations/Repetitions**: Statistical confidence measures

Use these metrics to:

- Verify real-time constraints are met
- Compare backend performance
- Identify optimal buffer sizes
- Detect performance regressions

