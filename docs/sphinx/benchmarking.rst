Benchmarking Guide
==================

This guide shows you how to use anira's benchmarking capabilities to measure neural network's real-time performance using the custom benchmark fixture class within the Google Benchmark framework.

Overview
--------

anira facilitates the measurement of neural network's real-time performance, by providing a custom benchmark fixture class within the `Google Benchmark <https://github.com/google/benchmark>`_ framework - the ``anira::benchmark::ProcessBlockFixture``.

Within this fixture class, a static instance of the ``anira::InferenceHandler`` class is constructed. The fixture class is designed to measure the runtimes of several consecutive calls to the ``process`` method of this instance. The number of calls to the ``process`` method can be configured by the user and is defined as iterations, as it is done in the Google Benchmark framework.

After all iterations are completed, the fixture will destroy the ``anira::InferenceHandler`` instance, freeing all threads and resources that have been used. This whole process can be repeated for a given number of repetitions as well as for different configurations.

In this way, the user can reliably compare the first inference with the subsequent ones and find out if the chosen inference backend needs some warm-up time. In addition, the user can compare runtimes for different configurations, such as different host buffer sizes and inference backends, and get an estimate of the maximum inference time.

Prerequisites
-------------

To use anira's benchmarking capabilities, you should first become familiar with the :doc:`usage` guide. This guide will show you how to create the necessary classes, configure the inference backend, and prepare anira for real-time audio processing.

To use the benchmarking tools within anira, please follow the steps below. First, you'll find a step-by-step guide on benchmarking a single configuration, followed by instructions on extending the benchmarks to multiple configurations.

Since the ``anira::benchmark::ProcessBlockFixture`` is a Google Benchmark fixture, you can use all the features of the Google Benchmark framework to further customize your benchmark setup. Please refer to the `Google Benchmark documentation <https://github.com/google/benchmark/blob/main/docs/user_guide.md>`_ for more information.

Single Configuration Benchmarking
----------------------------------

Step 1: Set Up the InferenceHandler and Input Buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we can start to define the benchmark, we need to create an ``anira::InferenceConfig`` instance and an ``anira::PrePostProcessor`` instance. This is done in the same way as described in the :doc:`usage` guide.

After that we can start to define the benchmark with the ``BENCHMARK_DEFINE_F`` macro. The first argument is the fixture class and the second argument is the name of the benchmark. The following code snippet shows how to use the ``anira::benchmark::ProcessBlockFixture`` and how to create and prepare a static ``anira::InferenceHandler`` class member within the fixture class.

We will also create a static ``anira::Buffer`` member, which will be used later as an input buffer. Finally, we will initialize the repetition. This will allow the anira fixture class to keep track of all configurations used and print options such as model path and buffer size in the benchmark log.

.. note::
   This code is only run once per repetition, not for every iteration. It is also not measured by the benchmark.

.. code-block:: cpp

   // benchmark.cpp

   #include <gtest/gtest.h>
   #include <benchmark/benchmark.h>
   #include <anira/anira.h>
   #include <anira/benchmark.h>

   typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

   anira::InferenceConfig my_inference_config(
       ...
   );
   anira::PrePostProcessor my_pp_processor(my_inference_config);

   BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {

       // Define the host configuration that shall be used / simulated for the benchmark
       anira::HostConfig host_config(BUFFER_SIZE, SAMPLE_RATE);
       anira::InferenceBackend inference_backend = anira::ONNX;

       // Create a static InferenceHandler instance, prepare and select backend
       m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor, my_inference_config);
       m_inference_handler->prepare(host_config);
       m_inference_handler->set_inference_backend(inference_backend);

       // Create a static Buffer instance
       m_buffer = std::make_unique<anira::Buffer<float>>(my_inference_config.get_preprocess_input_channels()[0], host_config.m_buffer_size);

       // Initialize the repetition
       initialize_repetition(my_inference_config, host_config, inference_backend, true);

.. note::
   In the ``initialize_repetition`` function, we can use the fourth argument to specify whether we want to sleep after a repetition. This can be useful if we want to give the system some time to cool down after a repetition. The time the fixture will sleep after a repetition is equal to the time it took to process all the iterations.

Step 2: Measure the Runtime of the Process Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the ``anira::InferenceHandler`` is prepared and the ``anira::Buffer`` is created, we can start to measure and record the runtime of the ``process`` method. For this we will use the ``state`` object that is passed to the benchmark function. The ``state`` object is used by the Google Benchmark framework to control the benchmark.

First we push random samples in the range of -1.f and 1.f into the ``anira::Buffer`` and initialize the iteration. Then we measure the runtime of the ``process`` method by calling it and waiting for the result. We have to wait for the result because the processing of the buffer is not done in the same thread as the call to the ``process`` function.

Then we update the fixture with the measured runtime. Finally, when all iterations are done, the ``anira::InferenceHandler`` and the ``anira::Buffer`` will be reset and if the repetition was initialized with the sleep after a repetition option, the fixture will sleep.

.. code-block:: cpp

   // benchmark.cpp (continued)
       for (auto _ : state) {
           // Fill the buffer with random samples
           push_random_samples_in_buffer(host_config);

           // Initialize the iteration
           initialize_iteration();

           // Here we start the actual measurement of the runtime
           auto start = std::chrono::high_resolution_clock::now();
           
           // Process the buffer
           m_inference_handler->process(m_buffer->get_array_of_write_pointers(), get_buffer_size());

           // Wait for the result
           while (!buffer_processed()) {
               std::this_thread::sleep_for(std::chrono::nanoseconds (10));
           }
           
           // End of the measurement
           auto end = std::chrono::high_resolution_clock::now();

           // Update the fixture with the measured runtime
           interation_step(start, end, state);
       }
       // Repetition is done, reset the InferenceHandler and the Buffer
       repetition_step();
   }

Step 3: Register the Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the benchmark is defined, we need to register it with the Google Benchmark framework. This is done by calling the ``BENCHMARK_REGISTER_F`` macro. The first argument is the fixture class, the second argument is the name of the benchmark. The name of the benchmark is used to identify it in the test log.

Here we also define which time unit we want to use for the benchmark and the number of iterations and repetitions. Finally, we need to specify that we want to use manual timing, since we are measuring the runtime of the ``process`` method ourselves.

.. code-block:: cpp

   BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
   ->Unit(benchmark::kMillisecond)
   ->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
   ->UseManualTime();

Multiple Configuration Benchmarking
------------------------------------

To benchmark multiple configurations, we can use Google Benchmark's parameterized benchmarks. This allows us to test different buffer sizes, sample rates, inference backends, and other parameters in a systematic way.

Setting Up Parameter Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, define the parameter ranges you want to test:

.. code-block:: cpp

   // Define buffer sizes to test
   std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024};
   
   // Define sample rates to test
   std::vector<double> sample_rates = {44100.0, 48000.0, 96000.0};
   
   // Define inference backends to test
   std::vector<anira::InferenceBackend> backends = {
       anira::InferenceBackend::ONNX,
       anira::InferenceBackend::LIBTORCH,
       anira::InferenceBackend::TFLITE
   };

Parameterized Benchmark Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a parameterized benchmark that tests all combinations:

.. code-block:: cpp

   BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_PARAMETERIZED)(::benchmark::State& state) {
       // Extract parameters from state
       int buffer_size = state.range(0);
       double sample_rate = state.range(1);
       anira::InferenceBackend backend = static_cast<anira::InferenceBackend>(state.range(2));

       // Set up configuration
       anira::HostConfig host_config(buffer_size, sample_rate);

       // Create and prepare InferenceHandler
       m_inference_handler = std::make_unique<anira::InferenceHandler>(my_pp_processor, my_inference_config);
       m_inference_handler->prepare(host_config);
       m_inference_handler->set_inference_backend(backend);

       // Create buffer
       m_buffer = std::make_unique<anira::Buffer<float>>(
           my_inference_config.get_preprocess_input_channels()[0], 
           buffer_size
       );

       // Initialize repetition
       initialize_repetition(my_inference_config, host_config, backend, true);

       // Benchmark loop
       for (auto _ : state) {
           push_random_samples_in_buffer(host_config);
           initialize_iteration();

           auto start = std::chrono::high_resolution_clock::now();
           m_inference_handler->process(m_buffer->get_array_of_write_pointers(), buffer_size);
           
           while (!buffer_processed()) {
               std::this_thread::sleep_for(std::chrono::nanoseconds(10));
           }
           
           auto end = std::chrono::high_resolution_clock::now();
           interation_step(start, end, state);
       }
       
       repetition_step();
   }

Registering Parameterized Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register the parameterized benchmark with all parameter combinations:

.. code-block:: cpp

   // Register with parameter combinations
   BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_PARAMETERIZED)
   ->ArgsProduct({
       benchmark::CreateRange(64, 1024, 2),  // Buffer sizes: 64, 128, 256, 512, 1024
       {44100, 48000, 96000},                // Sample rates
       {static_cast<int>(anira::InferenceBackend::ONNX),
        static_cast<int>(anira::InferenceBackend::LIBTORCH),
        static_cast<int>(anira::InferenceBackend::TFLITE)}  // Backends
   })
   ->Unit(benchmark::kMillisecond)
   ->Iterations(100)
   ->Repetitions(5)
   ->UseManualTime();

Advanced Benchmarking Features
------------------------------

Custom Metrics
~~~~~~~~~~~~~~

You can add custom metrics to track additional performance indicators:

.. code-block:: cpp

   // In your benchmark loop
   for (auto _ : state) {
       // ... benchmark code ...
       
       // Add custom counters
       state.counters["Latency_Samples"] = inference_handler->get_latency();
       state.counters["CPU_Usage"] = get_cpu_usage();
       state.counters["Memory_Usage"] = get_memory_usage();
   }

Warmup Iterations
~~~~~~~~~~~~~~~~~

For more accurate measurements, especially with neural networks that may have initialization overhead:

.. code-block:: cpp

   // Add warmup iterations before the main benchmark
   for (int warmup = 0; warmup < 10; ++warmup) {
       push_random_samples_in_buffer(host_config);
       m_inference_handler->process(m_buffer->get_array_of_write_pointers(), buffer_size);
       while (!buffer_processed()) {
           std::this_thread::sleep_for(std::chrono::nanoseconds(10));
       }
   }

Running Benchmarks
------------------

Compile and run your benchmarks:

.. code-block:: bash

   # Build with benchmark support
   cmake . -B build -DCMAKE_BUILD_TYPE=Release -DANIRA_WITH_BENCHMARK=ON
   cmake --build build --config Release

   # Run benchmarks
   ./build/your_benchmark_executable

Output and Analysis
~~~~~~~~~~~~~~~~~~

The benchmark will produce output showing:

- Mean execution time per iteration
- Standard deviation
- Minimum and maximum times
- Custom metrics you've added
- Configuration parameters for each test

Example output:

.. code-block:: text

   Run on (8 X 3000 MHz CPU s)
   CPU Caches:
     L1 Data 32 KiB (x4)
     L1 Instruction 32 KiB (x4)
     L2 Unified 256 KiB (x4)
     L3 Unified 8192 KiB (x1)
   -------------------------------------------------------------------
   Benchmark                           Time           CPU Iterations
   -------------------------------------------------------------------
   BM_SIMPLE/256/48000/ONNX        2.34 ms      2.34 ms        100
   BM_SIMPLE/512/48000/ONNX        4.21 ms      4.21 ms        100
   BM_SIMPLE/256/48000/LIBTORCH    3.12 ms      3.12 ms        100

Best Practices
--------------

1. **Consistent Environment**: Run benchmarks on a consistent system configuration
2. **Multiple Repetitions**: Use multiple repetitions to account for system variance
3. **Isolation**: Close other applications to minimize interference
4. **Warmup**: Include warmup iterations for neural network models
5. **Statistical Analysis**: Use the statistical output to understand performance variance
6. **Documentation**: Document your benchmark configurations and system specifications

This benchmarking framework allows you to systematically evaluate the real-time performance of different neural network models and configurations, helping you optimize your audio processing pipeline for production use.
