# anira Benchmark User Guide

## Preface

To use anira's benchmarking capabilities, you should first become familiar with the [anira usage guide](anira-usage.md). This guide will show you how to create the necessary classes, configure the inference backend, and prepare anira for real-time audio processing.

anira facilitates the measurement of neural network's real-time performance, by providing a custom benchmark fixture class within the [Google Benchmark](https://github.com/google/benchmark) framework - the `anira::benchmark::ProcessBlockFixture`. Within this fixture class, a static instance of the `anira::InferenceHandler` class is constructed. The fixture class is designed to measure the runtimes of several consecutive calls to the `process` method of this instance. The number of calls to the `process` method can be configured by the user and is defined as iterations, as it is done in the Google Benchmark framework. After all iterations are completed, the fixture will destroy the `anira::InferenceHandler` instance, freeing all threads and resources that have been used. This whole process can be repeated for a given number of repetitions as well as for different configurations. In this way, the user can reliably compare the first inference with the subsequent ones and find out if the chosen inference backend needs some warm-up time. In addition, the user can compare runtimes for different configurations, such as different host buffer sizes and inference backends, and get an estimate of the maximum inference time.

To use the benchmarking tools within anira, please follow the steps below. First, you'll find a step-by-step guide on benchmarking a single configuration, followed by instructions on extending the benchmarks to multiple configurations. Since the `anira::benchmark::ProcessBlockFixture` is a Google Benchmark fixture, you can use all the features of the Google Benchmark framework to further customize your benchmark setup. Please refer to the [Google Benchmark documentation](https://github.com/google/benchmark/blob/main/docs/user_guide.md) for more information.

## Single Configuration Benchmarking

### Step 1: Start Defining the Benchmark by Setting Up the InferenceHandler and Input Buffer

Before we can start to define the benchmark, we need to create an `anira::InferenceConfig` instance and an `anira::PrePostProcessor` instance. This is done in the same way as described in the [anira usage guide](anira-usage.md).

After that we can start to define the benchmark with the `BENCHMARK_DEFINE_F` macro. The first argument is the fixture class and the second argument is the name of the benchmark. The following code snippet shows how to use the `anira::benchmark::ProcessBlockFixture` and how to create and prepare a static `anira::InferenceHandler` class member within the fixture class. We will also create a static `anira::Buffer` member, which will be used later as an input buffer. Finally, we will initialize the repetition. This will allow the anira fixture class to keep track of all configurations used and print options such as model path and buffer size in the benchmark log. Note that this code is only run once per repetition, not for every iteration. It is also not measured by the benchmark.

```cpp
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
```

Note: In the `initialize_repetition` function, we can use the fourth argument to specify whether we want to sleep after a repetition. This can be useful if we want to give the system some time to cool down after a repetition. The time the fixture will sleep after a repetition is equal to the time it took to process all the iterations.

### Step 2: Measure the Runtime of the Process Method

After the `anira::InferenceHandler` is prepared and the `anira::Buffer` is created, we can start to measure and record the runtime of the `process` method. For this we will use the `state` object that is passed to the benchmark function. The `state` object is used by the Google Benchmark framework to control the benchmark.

First we push random samples in the range of -1.f and 1.f into the `anira::Buffer` and initialize the iteration. Then we measure the runtime of the `process` method by calling it and waiting for the result. We have to wait for the result because the processing of the buffer is not done in the same thread as the call to the `process` function. Then we update the fixture with the measured runtime. Finally, when all iterations are done, the `anira::InferenceHandler` and the `anira::Buffer` will be reset and if the repetition was initialized with the sleep after a repetition option, the fixture will sleep.

```cpp
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
```

### Step 3: Register the Benchmark

Once the benchmark is defined, we need to register it with the Google Benchmark framework. This is done by calling the `BENCHMARK_REGISTER_F` macro. The first argument is the fixture class, the second argument is the name of the benchmark. The name of the benchmark is used to identify it in the test log. Here we also define which time unit we want to use for the benchmark and the number of iterations and repetitions. Finally, we need to specify that we want to use manual timing, since we are measuring the runtime of the `process` method ourselves.

```cpp
BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->UseManualTime();
```

### Step 4: Define the Benchmark as a Unit Test

To run the benchmark as a unit test, we first need to define it as such. For this we use the [Google Test](https://github.com/google/googletest) framework. With the macro `TEST` we define a new test-case. The first argument is the name of the group of tests and the second argument is the name of the test-case. Before running the benchmark in the test-case, we increase the operation system-wide priority of the process.

```cpp
// test.cpp

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <anira/anira.h>

// Define a new test case
TEST(Benchmark, Simple){
    // Elevate the priority of the process
    #if __linux__ || __APPLE__
        pthread_t self = pthread_self();
    #elif WIN32
        HANDLE self = GetCurrentThread();
    #endif
    anira::HighPriorityThread::elevate_priority(self, true);

    // Run the benchmark
    benchmark::RunSpecifiedBenchmarks();
}
```

### Step 5: Building the CMake Target and Adding the Test to CTest

To build the benchmark, we create a new cmake project. The executable of this project should contain the target sources `benchmark.cpp` (where the benchmark is defined and registered) and `test.cpp` (where the test case is defined). The target must link against the `anira` library. Please note that the `anira` library must be built with the `ANIRA_BUILD_BENCHMARK` option set to `ON`. The Google Benchmark and Google Test libraries will then be linked automatically by the `anira` library. Finally, we will use the `gtest_discover_tests` function to let cmake discover the test case and add it to ctest.

```cmake
# CMakeLists.txt

# New project
project(benchmark_project)

# Add the anira library as a subdirectory
set(ANIRA_BUILD_BENCHMARK ON)
add_subdirectory(anira)

# Add the benchmark target
add_executable(benchmark_target benchmark.cpp test.cpp)

# Link the anira library
target_link_libraries(benchmark_target anira)

# Add the test to the ctest
gtest_discover_tests(benchmark_target)
```

### Step 6: Run the Benchmark

After building the project, you can run the benchmark by either running the executable or by running ctest. The ctest runs the benchmark as a unit test and prints the results to the console. To run the ctest, change to the build directory and run the following command:

```bash
ctest -VV

# Or to run the test case directly
ctest -R Benchmark.Simple -VV
```

Note: The `-VV` flag prints the test-case output to the console. If you want to change the test timeout for long-running benchmarks, you can do so by passing the `--timeout 100000` flag to the ctest command. The output log of the tests is stored in the `Testing` directory of the build directory.

## Multiple Configuration Benchmarking

### Passing Single Arguments

To benchmark multiple configurations, we can pass arguments to the defined benchmark function. This is done when we register the benchmark with the Google Benchmark framework. The arguments are passed as integers to the `Arg` template argument of the `BENCHMARK_REGISTER_F` macro. The arguments are then available in the benchmark via the `state` object. The following code snippet shows how to pass arguments that define the buffer size to the benchmark function. For more information on passing arguments, see the [Google Benchmark documentation] (https://github.com/google/benchmark/blob/main/docs/user_guide.md#passing-arguments).

```cpp
// benchmark.cpp

...

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)(::benchmark::State& state) {
    
    ...

    // Here we use the state object to pass the variable buffer size
    anira::HostConfig host_config = {(size_t) state.range(0), SAMPLE_RATE};

    m_inference_handler->prepare(host_config);

    ...
}

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_BUFFER_SIZES)
->Unit(benchmark::kMillisecond)
->Iterations(50)->Repetitions(10)
->UseManualTime()
// Define the buffer sizes that shall be used for the benchmark
->Arg(512)->Arg(1024)->Arg(2048);
```

Note: At present, the `anira::benchmark::ProcessBlockFixture` does not support benchmarks where the buffer size is not a multiple of the model output size, as the `buffer_processed` function will never return true. We plan to add support for this in the future.

### Passing Multiple Arguments

Multiple arguments can be passed to the benchmark function using the `Args` template argument of the `BENCHMARK_REGISTER_F` macro.The arguments are passed as tuples to the `Args' template argument (e.g., `->Args(64, 512)`). The arguments are then available in the benchmark via the `state.range(0)` and `state.range(1)` objects.

Another way to pass multiple arguments is to define the arguments in a separate function and pass the function when registering the benchmark. The following code snippet shows how to pass multiple arguments to the benchmark function.

```cpp
// benchmark.cpp

// Define the InferenceConfigs and PrePostProcessors as global variables, respectively
...

// Define the arguments as vectors
std::vector<int> buffer_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
std::vector<anira::InferenceBackend> inference_backends = {anira::LIBTORCH, anira::ONNX, anira::TFLITE, anira::CUSTOM};
std::vector<anira::InferenceConfig> inference_configs = {cnn_config, hybridnn_config, rnn_config};
anira::InferenceConfig inference_config;

// define the arguments function
static void Arguments(::benchmark::internal::Benchmark* b) {
    for (int i = 0; i < buffer_sizes.size(); ++i)
        for (int j = 0; j < inference_configs.size(); ++j)
            for (int k = 0; k < inference_backends.size(); ++k)
                // ONNX backend does not support stateful RNN
                if (!(j == 2 && k == 1))
                    b->Args({buffer_sizes[i], j, k});
}

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)(::benchmark::State& state) {

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ADVANCED)(::benchmark::State& state) {

    // The buffer size return in get_buffer_size() is populated by state.range(0) param of the google benchmark
    anira::HostConfig host_config = {(size_t) get_buffer_size(), SAMPLE_RATE};

    // Use state.range(1) to pass a distinct InferenceConfig and its respective PrePostProcessor
    anira::InferenceConfig inference_config = inference_configs[state.range(1)];
    anira::PrePostProcessor pp_processor = pp_processors[state.range(1)];

    m_inference_handler = std::make_unique<anira::InferenceHandler>(*my_pp_processor, inference_config);
    m_inference_handler->prepare(host_config);

    // Use state.range(2) to select the inference backend
    m_inference_handler->set_inference_backend(inference_backends[state.range(2)]);

    m_buffer = std::make_unique<anira::Buffer<float>>(inference_config.get_preprocess_input_channels()[0], host_config.m_buffer_size);

    initialize_repetition(inference_config, host_config, inference_backends[state.range(2)]);

    ...
}

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_MULTIPLE_CONFIGURATIONS)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->UseManualTime()
// Apply all arguments via the Arguments function
->Apply(Arguments);
```

## Benchmarking anira Without Inference

If you want to benchmark anira without inference, just measuring the runtime of the pre- and post-processing stages and the runtime of the `process` method, you can use the `anira::benchmark::ProcessBlockFixture` in the same way as described above. The only difference is that you have to set the inference backend to `anira::CUSTOM`. As the default custom processor is doing a roundtrip.

## Benchmarking anira With Custom Inference

If you want to benchmark anira with a custom inference backend, define the custom inference backend as described in the [anira usage guide](anira-usage.md). You can then select the custom inference backend by setting the inference backend to `anira::CUSTOM` and passing the custom inference backend to the `anira::InferenceHandler` constructor.
