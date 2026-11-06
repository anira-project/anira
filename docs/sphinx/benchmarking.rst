Benchmarking Guide
==================

anira's benchmark examples measure a model's real-time performance with a fixture for `Google
Benchmark <https://github.com/google/benchmark>`_, ``anira::benchmark::ProcessBlockFixture``: it
holds a static :cpp:class:`anira::InferenceHandler` and times consecutive calls of its ``process``.
The fixture is part of the examples, not of the library: it lives in ``examples/benchmark/fixture/``
(``ProcessBlockFixture.h`` / ``.cpp``, the static target ``anira_benchmark_fixture``), written over
``anira/anira.hpp`` alone. anira itself does not link Google Benchmark, and its installed package
does not look for it. The benchmarks use the objects of :doc:`usage`: a model config, a contract, a
pipeline and a handler.

Inside this repository, ``ANIRA_WITH_BENCHMARK=ON`` builds the benchmark examples and the fixture
(also without ``ANIRA_WITH_EXAMPLES``); a project of your own copies the two fixture files and links
Google Benchmark (and Google Test) itself (:ref:`benchmarking-cmake`).

.. _benchmarking-one:

One benchmark
-------------

The body creates the context, the pipeline and the handler into the fixture's static members,
prepares it with the contract for the buffer size, selects the plan to measure, and times each
``process`` until its result is back:

.. code-block:: cpp
    :caption: benchmark.cpp

    #include <ProcessBlockFixture.h>
    #include <benchmark/benchmark.h>
    #include <gtest/gtest.h>

    #include <anira/anira.hpp>

    typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

    const anira::ModelConfig my_model_config = anira::ModelConfig::from_file("model.json");
    anira::ContractHandle my_contract = anira::ContractHandle::from_file("contract.json");

    BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {
        // The host blocks: the buffer size (the registered argument) at SAMPLE_RATE
        const auto block = static_cast<double>(get_buffer_size());
        my_contract.hard_host_blocks(block, block, SAMPLE_RATE);

        // Only report errors, so the engines' log output does not pollute the results
        m_context = std::make_unique<anira::Context>(
            anira::ContextConfig{}.log_level(ANIRA_LOG_ERROR));

        anira::Pipeline pipeline;
        pipeline.add_inference(my_model_config);
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        const anira::PlanReport report = m_inference_handler->prepare(my_contract);
        const uint32_t plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}}).value();
        m_inference_handler->set_plan(plan);

        // Size the input buffer, track the configuration, sleep between repetitions (true)
        initialize_repetition(my_model_config, my_contract.hard(), plan, true);

        for (auto _ : state) {
            push_random_samples_in_buffer();      // random samples in [-1.0, 1.0)
            initialize_iteration();
            const auto start = std::chrono::steady_clock::now();
            process_buffer();                     // InferenceHandler::process, in place
            while (!buffer_processed()) {         // the inference runs on another thread
                std::this_thread::sleep_for(std::chrono::nanoseconds(10));
            }
            const auto end = std::chrono::steady_clock::now();
            interation_step(start, end, state);   // records the time and its RTF
        }
        repetition_step(NUM_REPETITIONS);         // the total enables the Summary line
    }

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(NUM_ITERATIONS)      // process calls per repetition
        ->Repetitions(NUM_REPETITIONS)    // repetitions of the whole benchmark
        ->Arg(BUFFER_SIZE)                // the buffer size, state.range(0)
        ->UseManualTime();                // the loop times itself

- ``initialize_repetition(model, hard, plan, sleep)`` sets the fixture up over the prepared handler:
  it sizes the input buffer (the anchor input's channels times the buffer size, one planar float32
  tensor), resets the statistics and prints the run's header when the model, the contract (its
  ``host_block_max`` and ``host_rate``), the plan or its backend changed, and with ``sleep`` sleeps
  after each repetition as long as it took, for thermal stability.
- :cpp:func:`anira::PlanReport::find` returns the first plan on an engine (a custom one by its id),
  ``std::nullopt`` when the report has none; a pipeline with one candidate has one plan, plan 0.
- ``process_buffer()`` returns ``process``'s status. The inference runs asynchronously, so the loop
  waits for ``buffer_processed()`` before it stops the clock: the time is the whole round trip.

.. _benchmarking-output:

What it prints
--------------

The run's header names the model (its file), the engine (a custom engine's id), the host rate and
buffer, and what prepare found for the plan (:ref:`usage-plan-report`): the budget (the measured
figure, or the contract's under an explicit budget) and the warm-up's outcome (``measured over N
inferences,`` then ``stable``, ``fixed`` or ``capped before stable``; ``the model's figure, measured
at an earlier prepare``; or ``explicit, no warm-up``), and the plan's real-time factor of its
parallel inferences with the verdict. Nothing is held whatever the verdict: a benchmark measures.

.. code-block:: text

    Model: model.onnx | Backend: onnx | Host Sample Rate: 44100 Hz | Host Buffer Size: 2048 = 46.4399 ms
    Budget 7.94 ms (measured over 6 inferences, stable); real-time factor 0.171 of 4: comfortable

Each iteration then prints its time and its **real-time factor** (RTF: the time over the buffer
period), marked ``[underrun]`` above 1.0. With the total of repetitions passed to
``repetition_step``, a ``Summary/...`` line after the last one reports ``rtf_mean``, ``rtf_max`` and
the underruns over every iteration of the benchmark instance; ``repetition_step()`` without an
argument prints none. Google Benchmark adds its mean, median and standard deviation per
configuration; the bundled benchmarks add the minimum, the maximum and a percentile with
``ComputeStatistics`` and the fixture's ``calculate_min`` / ``calculate_max`` /
``calculate_percentile``.

.. note::
    The iteration RTF is a single blocking round trip against one buffer period, the strictest
    criterion: anira's scheduler absorbs inference times of several buffer periods in exchange for
    latency (:doc:`latency`), so an RTF above 1.0 is not necessarily a dropout in a real host. The
    header's factor is the plan's own, over the hop period, the one the verdict judges.

.. _benchmarking-cmake:

CMake
-----

Inside this repository a benchmark links the fixture target beside anira:

.. code-block:: cmake

    add_executable(benchmark_target benchmark.cpp)
    target_link_libraries(benchmark_target anira::anira anira_benchmark_fixture)

A project of your own copies ``ProcessBlockFixture.h`` and ``.cpp`` and builds them with Google
Benchmark (the fixture needs nothing of anira but ``anira/anira.hpp``). As a Google Test executable
it fetches Google Test and links ``GTest::gtest_main``, and the test below calls tanh-lib's
thread API, so the project fetches tanh-lib's core and links ``tanh::Core`` itself: a tanh-lib of
its own, which never meets anira's private copy (``thl::anira``, :ref:`host-tanh-lib`).

.. code-block:: cmake

    project(benchmark_project)

    add_subdirectory(anira)  # or find_package(anira REQUIRED) for an installed anira

    include(FetchContent)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5)
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(benchmark)

    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)   # GTest::gtest_main

    # tanh-lib's core component, for the thread priority of test.cpp
    FetchContent_Declare(tanh-lib
        GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
        GIT_TAG v0.5.0)
    set(TANH_BUILD_STATE OFF)
    set(TANH_BUILD_DSP OFF)
    set(TANH_BUILD_MODULATION OFF)
    set(TANH_BUILD_AUDIO_IO OFF)
    set(TANH_WITH_TESTS OFF)
    set(TANH_WITH_EXAMPLES OFF)
    FetchContent_MakeAvailable(tanh-lib)

    add_executable(benchmark_target benchmark.cpp test.cpp
        fixture/ProcessBlockFixture.cpp fixture/ProcessBlockFixture.h)
    target_include_directories(benchmark_target PRIVATE fixture)
    target_compile_features(benchmark_target PRIVATE cxx_std_20)
    target_link_libraries(benchmark_target anira::anira benchmark::benchmark GTest::gtest_main tanh::Core)

    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(benchmark_target)

Running it
----------

Run the executable directly (``./build/benchmark_target``), or wrap the benchmarks in a Google Test
case that raises the thread's priority first, for steadier timings:

.. code-block:: cpp
    :caption: test.cpp

    #include <benchmark/benchmark.h>
    #include <gtest/gtest.h>
    #include <tanh/core/threading/Thread.h>

    TEST(Benchmark, Simple) {
        thl::core::Thread::set_current_priority(thl::core::ThreadPriority::RealTime);
        benchmark::RunSpecifiedBenchmarks();
    }

.. code-block:: bash

    ctest -R Benchmark.Simple -VV           # -VV shows the results; they are kept in Testing/
    ctest --timeout 100000 -VV              # long benchmarks need a longer timeout

The benchmarks of this repository are such tests: ``cmake --preset desktop-benchmark-tests-debug``,
then ``ctest --preset desktop-benchmark-tests-debug -R '^Benchmark' -j 1`` (one at a time, so they do
not compete for the cores).

Several configurations
----------------------

A list of arguments sweeps the buffer size; each run prepares the handler for its own
(``BM_MULTIPLE_BUFFER_SIZES`` is ``BM_SIMPLE``'s body under another name):

.. code-block:: cpp

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(50)
        ->Repetitions(10)
        ->UseManualTime()
        ->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

.. warning::
    The fixture needs buffer sizes that are multiples of the model's output hop; for other sizes
    ``buffer_processed`` may not return ``true``.

Argument tuples select the model and the backend too. ``examples/benchmark/advanced-benchmark``
sweeps buffer sizes, three models (built in code, since their shapes follow the buffer size, with the
builders of ``extras/models/cnn/CNNConfig.h`` and friends, and their contract files) and every engine
of the build plus a custom engine as the baseline:

.. code-block:: cpp

    std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const std::vector<anira::BackendId>& benchmark_backends();  // anira::enabled_engines() + the clear engine
    constexpr int num_models = 3;
    struct ModelAndContract {
        anira::ModelConfig m_model_config;
        anira::ContractHandle m_contract;
    };
    ModelAndContract model_and_contract(int model, int buffer_size);

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
        const anira::BackendId backend = benchmark_backends()[state.range(2)];
        auto [model, contract] = model_and_contract(state.range(1), get_buffer_size());
        contract.hard_host_blocks(get_buffer_size(), get_buffer_size(), SAMPLE_RATE);

        // The one backend as the single candidate: plan 0
        m_context = std::make_unique<anira::Context>(anira::ContextConfig{});
        anira::Pipeline pipeline;
        pipeline.add_engine(std::make_shared<ClearEngine>()).add_inference(model, {backend});
        m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
        m_inference_handler->prepare(contract);
        initialize_repetition(model, contract.hard(), 0);

        // ... measurement loop
    }

    BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)
        ->Unit(benchmark::kMillisecond)
        ->Iterations(NUM_ITERATIONS)
        ->Repetitions(NUM_REPETITIONS)
        ->UseManualTime()
        ->Apply(Arguments);

Without inference, and custom engines
-------------------------------------

To measure the pipeline's own overhead, run an engine that runs no model: the advanced benchmark's
``ClearEngine`` (``org.anira.examples.clear``, ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``) zeroes the
outputs, the extras' ``BypassEngine`` (``org.anira.examples.bypass``) passes the input through. Its
plan is selected by its id:

.. code-block:: cpp

    // the model config names the engine: model.add_model(k_clear_engine_id)
    anira::Pipeline pipeline;
    pipeline.add_engine(std::make_shared<ClearEngine>());
    pipeline.add_inference(model);
    m_inference_handler = std::make_unique<anira::InferenceHandler>(*m_context, pipeline);
    const anira::PlanReport report = m_inference_handler->prepare(hard);
    m_inference_handler->set_plan(
        report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = k_clear_engine_id}})
            .value());

Any custom engine is benchmarked the same way: an :cpp:class:`anira::Engine` added to the pipeline,
named by a model entry, its plan selected by its id (:doc:`custom_engines`).

Run on a quiet machine, with enough repetitions for its variability, the priority raised and the
sleep between repetitions on; measure the buffer sizes and engines your host will use. Since prepare
warms the model up (under any contract but an explicit budget with ``ANIRA_WARMUP_NONE``), the first
iteration is no longer a cold start: what the warm-up found is in the header.
