Benchmarking Guide
==================

Overview
--------

anira's benchmark examples measure a neural network's real-time performance with a custom benchmark fixture class within the `Google Benchmark <https://github.com/google/benchmark>`_ framework - ``anira::benchmark::ProcessBlockFixture``. The fixture holds a static :cpp:class:`anira::InferenceHandler` and measures the runtimes of several consecutive calls to its :cpp:func:`anira::InferenceHandler::process` method.

The fixture is designed to:

- Measure runtime performance across multiple iterations
- Compare first inference with subsequent ones to detect warm-up requirements
- Test different configurations (buffer sizes, engines)
- Estimate maximum inference times for real-time constraints

The fixture is part of the examples, not of the library: it lives in ``examples/benchmark/fixture/`` (``ProcessBlockFixture.h`` and ``ProcessBlockFixture.cpp``, the static target ``anira_benchmark_fixture``) and is written over ``anira/anira.hpp`` alone. The benchmark examples of this repository link the target; a project of your own copies the two files (see *CMake Configuration* below). anira itself does not link Google Benchmark, and its installed package does not look for it.

.. note::
    To use the fixture, you should first become familiar with the main anira usage patterns. The benchmarks build upon the same :cpp:class:`anira::InferenceHandler`, :cpp:class:`anira::ModelConfig`, :cpp:class:`anira::ContractHandle` and :cpp:class:`anira::Pipeline` described in the :doc:`usage` section. The fixture is a specialized tool that extends the standard usage patterns to measure performance in a controlled manner.

Prerequisites
-------------

Before using the benchmarking tools, ensure that:

1. you build the benchmark examples of this repository with the ``ANIRA_WITH_BENCHMARK`` option set to ``ON`` (it builds ``examples/benchmark`` and the fixture, also without ``ANIRA_WITH_EXAMPLES``), or your project carries a copy of the fixture and links Google Benchmark and Google Test itself
2. you have a working model configuration and contract (:doc:`usage` section 1)
3. your model files are accessible and properly configured

Single Configuration Benchmarking
----------------------------------

1. Define the Benchmark Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating your benchmark using the ``BENCHMARK_DEFINE_F`` macro. The benchmark body creates the context, the pipeline and the :cpp:class:`anira::InferenceHandler` into the fixture's static members, prepares it with the contract of the buffer size and selects the plan to measure:

.. code-block:: cpp
    :caption: benchmark.cpp

    #include <ProcessBlockFixture.h>
    #include <benchmark/benchmark.h>
    #include <gtest/gtest.h>

    #include <anira/anira.hpp>

    typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

    // Configure your inference setup (same as in regular usage): the model file and the
    // contract file
    const anira::ModelConfig my_model_config = anira::ModelConfig::from_file("model.json");
    anira::ContractHandle my_contract = anira::ContractHandle::from_file("contract.json");

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {
        // The host blocks: the buffer size at SAMPLE_RATE, on the contract
        const auto block = static_cast<double>(get_buffer_size());
        my_contract.hard_host_blocks(block, block, SAMPLE_RATE);

        // Only report errors, so the log output of the engines does not pollute the
        // benchmark results (section 3.1 of the usage guide)
        m_context = std::make_unique<anira::Context>(
            anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));

        // Create and prepare the InferenceHandler instance, then pick the plan to measure
        anira::Pipeline pipeline;
        pipeline.add_inference(my_model_config);
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        const anira::PlanReport report = m_inference_handler->prepare(my_contract);
        const uint32_t plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}).value();
        m_inference_handler->set_plan(plan);

        // Initialize the repetition (sizes the buffer, tracks the configuration, optional sleep)
        initialize_repetition(my_model_config, my_contract.hard(), plan, true);

.. note::
    The ``initialize_repetition`` method sets up the benchmark fixture over the prepared handler: it sizes the input buffer (the anchor input's channels times the buffer size, one planar float32 tensor), tracks configuration changes, and optionally sleeps between repetitions for thermal stability. The first parameter is the model configuration, the second the Hard contract the handler was prepared with (its ``host_block_max`` and ``rate``), the third the plan of the report that is measured, and the fourth controls whether to sleep after each repetition. The sleep duration is equal to the time taken to process all iterations, allowing for thermal cooldown between repetitions. :cpp:func:`anira::PlanReport::find` returns the index of the report's first plan on an engine (a custom engine by its id), ``std::nullopt`` when the report has none; a pipeline with one candidate has one plan, plan 0.

2. Measure Process Method Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the main measurement loop using the Google Benchmark framework's state control:

.. code-block:: cpp
    :caption: benchmark.cpp

        // Main benchmark loop
        for (auto _ : state) {
            // Fill buffer with random samples in range [-1.0, 1.0)
            push_random_samples_in_buffer();

            // Initialize iteration tracking
            initialize_iteration();

            // Begin timing measurement
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

            // Process the buffer (this triggers inference)
            process_buffer();

            // Wait for processing completion (inference is asynchronous)
            while (!buffer_processed()) {
                std::this_thread::sleep_for(std::chrono::nanoseconds(10));
            }

            // End timing measurement
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            // Record the measured runtime
            interation_step(start, end, state);
        }

        // Clean up after all iterations complete
        repetition_step(NUM_REPETITIONS);
    }

.. note::
    ``process_buffer()`` is :cpp:func:`anira::InferenceHandler::process` over the planar tensor of the input buffer, in place, and returns its status. It operates asynchronously: to ensure accurate timing measurements, you must wait for the ``buffer_processed`` method to return ``true`` before stopping the timer. This guarantees that the measured time includes the complete processing duration, not just the time to initiate processing.

.. note::
    Each recorded iteration also prints its real-time factor (RTF) — the measured runtime divided by the host buffer period — and marks iterations that exceeded the buffer period with ``[underrun]``. When the total number of repetitions is passed to ``repetition_step``, the fixture additionally prints a ``Summary/...`` line after the final repetition, reporting ``rtf_mean``, ``rtf_max`` and the underrun count over all iterations of all repetitions of the current benchmark instance. Calling ``repetition_step`` without an argument disables the summary line.

3. Register the Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

Configure and register your benchmark with the Google Benchmark framework; the argument is the buffer size ``get_buffer_size()`` reads:

.. code-block:: cpp
    :caption: benchmark.cpp

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(NUM_ITERATIONS)
        ->Repetitions(NUM_REPETITIONS)
        ->Arg(BUFFER_SIZE)
        ->UseManualTime();

The key parameters are:

- **Unit**: Specify the time unit for results (e.g., ``benchmark::kMillisecond``)
- **Iterations**: Number of :cpp:func:`anira::InferenceHandler::process` calls per repetition
- **Repetitions**: Number of times to repeat the entire benchmark
- **Arg**: The buffer size of the run (``state.range(0)``)
- **UseManualTime**: Required since we manually measure processing time

4. CMake Configuration
~~~~~~~~~~~~~~~~~~~~~~

Inside this repository, ``ANIRA_WITH_BENCHMARK`` builds the benchmark examples and the fixture target, which a benchmark links beside anira:

.. code-block:: cmake

    add_executable(benchmark_target benchmark.cpp)
    target_link_libraries(benchmark_target anira::anira anira_benchmark_fixture)

A project of your own copies ``ProcessBlockFixture.h`` and ``ProcessBlockFixture.cpp`` from ``examples/benchmark/fixture/`` and builds them with Google Benchmark next to its benchmark (the fixture needs nothing of anira but ``anira/anira.hpp``):

.. code-block:: cmake

    project(benchmark_project)

    add_subdirectory(anira)  # or find_package(anira REQUIRED) for an installed anira

    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(benchmark)

    # Create benchmark executable with the copied fixture
    add_executable(benchmark_target benchmark.cpp
        fixture/ProcessBlockFixture.cpp fixture/ProcessBlockFixture.h)
    target_include_directories(benchmark_target PRIVATE fixture)
    target_compile_features(benchmark_target PRIVATE cxx_std_20)
    target_link_libraries(benchmark_target anira::anira benchmark::benchmark)

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
    #include <tanh/core/threading/Thread.h>

    TEST(Benchmark, Simple) {
        // Elevate this thread's priority for more consistent timing
        thl::core::Thread::set_current_priority(thl::core::ThreadPriority::RealTime);

        // Execute the benchmark
        benchmark::RunSpecifiedBenchmarks();
    }

Integrate via CMake
~~~~~~~~~~~~~~~~~~~

Set up your CMake project to include the benchmark and find the test. The benchmark is a Google Test executable, so it links ``gtest_main`` (``GTest::gtest_main`` from your project's Google Test; this repository's fixture target links it for its own benchmarks, and anira itself links no test library):

.. code-block:: cmake

    # Create benchmark executable
    add_executable(benchmark_target benchmark.cpp test.cpp)
    target_link_libraries(benchmark_target anira::anira anira_benchmark_fixture GTest::gtest_main)

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

The benchmarks of this repository are such tests: ``cmake --preset desktop-benchmark-tests-debug``, then ``ctest --preset desktop-benchmark-tests-debug -R '^Benchmark' -j 1`` (one at a time, so they do not compete for the cores).

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
        // state.range(0) is the buffer size argument: the host blocks of this run
        const auto block = static_cast<double>(get_buffer_size());
        my_contract.hard_host_blocks(block, block, SAMPLE_RATE);

        m_context = std::make_unique<anira::Context>(anira::ContextConfig{});
        anira::Pipeline pipeline;
        pipeline.add_inference(my_model_config);
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        const anira::PlanReport report = m_inference_handler->prepare(my_contract);
        const uint32_t plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}).value();
        m_inference_handler->set_plan(plan);

        initialize_repetition(my_model_config, my_contract.hard(), plan);

        // ... measurement loop (same as single configuration)
    }

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(50)
        ->Repetitions(10)
        ->UseManualTime()
        ->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

.. warning::
    Currently, the fixture requires buffer sizes that are multiples of the model output size. The ``buffer_processed`` function may not return ``true`` for other buffer sizes.

Multiple Argument Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex configuration testing, define argument combinations using a custom function:

.. code-block:: cpp
    :caption: benchmark.cpp

    // Define test configurations: the buffer sizes, and the engines of this build plus a
    // custom engine (examples/benchmark/advanced-benchmark measures its ClearEngine, which
    // runs no model, as the baseline)
    std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const std::vector<anira::BackendId>& benchmark_backends();  // anira::enabled_engines() + the clear engine
    // One model per index. The shapes follow the buffer size, which the fixed windows of the
    // configuration files cannot, so the benchmark builds each model config in code (the
    // builders of extras/models/cnn/CNNConfig.h and friends) and loads the model's contract
    // file (see examples/benchmark/advanced-benchmark)
    constexpr int num_models = 3;
    struct ModelAndContract {
        anira::ModelConfig m_model_config;
        anira::ContractHandle m_contract;
    };
    ModelAndContract model_and_contract(int model, int buffer_size);

    // Define argument combinations
    static void Arguments(::benchmark::internal::Benchmark* b) {
        for (int i = 0; i < buffer_sizes.size(); ++i) {
            for (int j = 0; j < num_models; ++j) {
                for (int k = 0; k < benchmark_backends().size(); ++k) {
                    // Skip incompatible combinations (e.g., ONNX + stateful RNN)
                    if (!(j == 2 && benchmark_backends()[k].engine == ANIRA_ENGINE_ONNXRUNTIME)) {
                        b->Args({buffer_sizes[i], j, k});
                    }
                }
            }
        }
    }

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)(::benchmark::State& state) {
        // Extract configuration from arguments
        const anira::BackendId backend = benchmark_backends()[state.range(2)];
        auto [model, contract] = model_and_contract(state.range(1), get_buffer_size());
        contract.hard_host_blocks(get_buffer_size(), get_buffer_size(), SAMPLE_RATE);

        // Setup with selected configuration: the one backend as the single candidate
        m_context = std::make_unique<anira::Context>(anira::ContextConfig{});
        anira::Pipeline pipeline;
        pipeline.add_engine(std::make_shared<ClearEngine>()).add_inference(model, {backend});
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        m_inference_handler->prepare(contract);

        initialize_repetition(model, contract.hard(), 0);  // the one candidate: plan 0

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

To measure only pre/post-processing overhead without actual inference, run a custom engine that runs no model: the advanced benchmark's ``ClearEngine`` (``org.anira.examples.clear``, ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``) zeroes the outputs, and the extras' ``BypassEngine`` (``org.anira.examples.bypass``) passes the input through:

.. code-block:: cpp
    :caption: benchmark.cpp

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_NO_INFERENCE)(::benchmark::State& state) {
        // ... setup code: the model config names the engine (model.add_model(k_clear_engine_id)) ...
        anira::Pipeline pipeline;
        pipeline.add_engine(std::make_shared<ClearEngine>());
        pipeline.add_inference(model);
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        const anira::PlanReport report = m_inference_handler->prepare(hard);
        m_inference_handler->set_plan(
            report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = k_clear_engine_id}})
                .value());

        // ... measurement loop ...
    }

This configuration measures the overhead of anira's processing pipeline without the neural network inference step.

Benchmarking Custom Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A custom engine is benchmarked like a built-in one: it is an :cpp:class:`anira::Engine` added to the pipeline (``Pipeline::add_engine``, or ``anira_custom_engine_create`` and ``anira_pipeline_add_engine`` in C; :doc:`custom_engines`), named by a model entry, and its plan is selected by its id, as in the scenario above.

Best Practices
--------------

1. **Consistent Environment**: Run benchmarks on a dedicated system with minimal background processes
2. **Thermal Management**: Use the sleep option in ``initialize_repetition`` for thermal stability
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
- **Real-Time Factor (RTF)**: Measured runtime divided by the host buffer period, printed per iteration. An RTF above ``1.0`` means the block took longer to process than the audio it represents — such iterations are marked ``[underrun]``.
- **Underruns**: Number of iterations that exceeded the buffer period, reported in the ``Summary/...`` line per benchmark instance (see above).

Use these metrics to:

- Verify real-time constraints are met
- Compare engine performance
- Identify optimal buffer sizes
- Detect performance regressions

.. note::
    The RTF compares a single blocking round-trip (``process()`` call plus waiting for the result) against one buffer period. This is the strictest real-time criterion: anira's scheduler tolerates inference latencies of multiple buffer periods in exchange for added latency, so an RTF above ``1.0`` does not necessarily mean audible dropouts in a real host — see :doc:`latency` for details.
