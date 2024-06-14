# anira benchmark usage guide

## Preface

# anira benchmark usage guide

## Preface

To use anira's benchmarking capabilities, you should first become familiar with the [anira usage guide](anira-usage.md). This guide will show you how to create the necessary classes, configure the inference backend, and prepare anira for real-time audio processing.

Anira provides a custom benchmark fixture class within the [Google Benchmark](https://github.com/google/benchmark) framework - the `anira::benchmark::ProcessBlockFixture`. Within this fixture class, a static instance of the `anira::InferenceHandler` class is constructed. The fixture class is designed to measure the runtimes of several consecutive calls to the `process` method of this instance. The number of calls to the `process` method can be configured by the user and is defined as iterations, as it is done in the Google Benchmark framework. After all iterations are completed, the fixture will destroy the `anira::InferenceHandler` instance, freeing all threads and resources that have been used. This whole process can be repeated for a given number of repetitions as well as for different configurations. In this way, the user can reliably compare the first inference with the subsequent ones and find out if the chosen inference backend needs some warm-up time. In addition, the user can compare runtimes for different configurations, such as different host buffer sizes and inference backends.

To use the benchmarking tools within aniras, please follow the steps below. First there is a step-by-step guide on how to benchmark a single configuration and then a guide on how to extend the benchmarks to multiple configurations. Since the `anira::benchmark::ProcessBlockFixture` is a Google Benchmark fixture, you can use all the features of the Google Benchmark framework to further customize your benchmark setup. Please refer to the [Google Benchmark documentation] for more information.(https://github.com/google/benchmark/blob/main/docs/user_guide.md).

## Single Configuration Benchmarking

### Step 1: Start defining the benchmark and setting up the InferenceHandler and Audiobuffer to be processed

Before we can start to define the benchmark, we need to create an `anira::InferenceConfig` instance and an `anira::PrePostProcessor` instance. This is done in the same way as described in the [anira usage guide](anira-usage.md).

After that we can start to define the benchmark with the `BENCHMARK_DEFINE_F` macro. The first argument is the fixture class and the second argument is the name of the benchmark. The following code snippet shows how to use the `anira::benchmark::ProcessBlockFixture` and how to create and prepare a static `anira::InferenceHandler` class member within the fixture class. We will also create a static `anira::AudioBuffer` member, which will be used later as an input buffer. Finally, we will initialize the iteration. This will allow the anira fixture class to keep track of all configurations used and print options such as model path and buffer size in the benchmark log. Note that this code is only run once per repetition, not for every iteration. It is also not measured by the benchmark.

```cpp
// benchmark.cpp

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include <anira/anira.h>
#include <anira/benchmark.h>

anira::PrepostProcessor myPrePostProcessor;
anira::InferenceConfig myConfig(
    ...
);

typedef anira::benchmark::ProcessBlockFixture ProcessBlockFixture;

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_SIMPLE)(::benchmark::State& state) {

    // Create a static InferenceHandler instance
    m_inferenceHandler = std::make_unique<anira::InferenceHandler>(myPrePostProcessor, myConfig);
    // Define the host audio configuration that shall be used / simulated for the benchmark
    anira::HostAudioConfig hostAudioConfig(2, 512, 48000);
    // Prepare the InferenceHandler instance
    m_inferenceHandler->prepare(hostAudioConfig);
    // Select the inference backend
    m_inferenceHandler->setInferenceBackend(anira::LIBTORCH);

    // Create a static AudioBuffer instance
    m_buffer = std::make_unique<anira::AudioBuffer<float>>(hostAudioConfig.hostChannels, hostAudioConfig.hostBufferSize);

    // Initialize the repetition and define with a bool whether to sleep after a repetition
    initializeRepetition(myConfig, hostAudioConfig, inferenceBackend, true);

```
Note: In the `initializeRepetition` function, we can use the fourth argument to specify whether we want to sleep after a repetition. This can be useful if we want to give the system some time to cool down after a repetition. The time the fixture will sleep after a repetition is equal to the time it took to process all the iterations.

### Step 2: Measure the runtime of the process method

After the `anira::InferenceHandler` is prepared and the `anira::AudioBuffer` is created, we can start to measure and record the runtime of the `process` method. For this we will use the `state` object that is passed to the benchmark function. The `state` object is used by the Google Benchmark framework to control the benchmark.

First we push random samples in the range of -1.f and 1.f into the `anira::AudioBuffer` and initialize the iteration. Then we measure the runtime of the `process` method by calling it and waiting for the result. We have to wait for the result because the processing of the buffer is not done in the same thread as the call to the `process` function. Then we update the fixture with the measured runtime. Finally, when all iterations are done, the `anira::InferenceHandler` and the `anira::Audiobuffer` will be reset and if the repetition was initialized with the sleep after a repetition option, the fixture will sleep.

```cpp
    for (auto _ : state) {
        
        // Initialize the iteration
        initializeIteration();

        // Fill the buffer with random samples
        pushRandomSamplesInBuffer(hostAudioConfig);

        // Here we start the actual measurement of the runtime
        auto start = std::chrono::high_resolution_clock::now();

        // Process the buffer
        m_inferenceHandler->process(m_buffer->getArrayOfWritePointers(), getBufferSize());

        // Wait for the result
        while (!bufferHasBeenProcessed()) {
            std::this_thread::sleep_for(std::chrono::nanoseconds (10));
        }

        // End of the measurement
        auto end = std::chrono::high_resolution_clock::now();

        // Update the fixture with the measured runtime
        interationStep(start, end, state);
    }

    // Repetition is done, reset the InferenceHandler and the AudioBuffer
    repetitionStep();
}
```

### Step 3: Register the benchmark

Once the benchmark is defined, we need to register it with the Google Benchmark framework. This is done by calling the `BENCHMARK_REGISTER_F` macro. The first argument is the fixture class, the second argument is the name of the benchmark. The name of the benchmark is used to identify it in the test log. Here we also define which units we want to use for the benchmark and the number of iterations and repetitions. Finally, we need to specify that we want to use manual timing, since we are measuring the runtime of the `process` method ourselves.

```cpp
BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_SIMPLE)
->Unit(benchmark::kMillisecond)
->Iterations(50)->Repetitions(10)
->UseManualTime();
```

### Step 4: Define the benchmark as a unit test

To run the benchmark as a ctest unit test, we first need to define it as such. For this we use the [Google Test](https://github.com/google/googletest) framework. With the macro `TEST` we define a new test-case. The first argument is the name of the group of tests and the second argument is the name of the test-case. Before running the benchmark in the test-case, we increase the priority of the process.

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
    anira::RealtimeThread::elevateToRealTimePriority(self, true);

    // Run the benchmark
    benchmark::RunSpecifiedBenchmarks();
}
```

### Step 5: Building the cmake target and adding the test to ctest

To build the benchmark, we create a new cmake project. The executable of this project should contain the target sources `benchmark.cpp` (where the benchmark is defined and registered) and `test.cpp` (where the test case is defined). The target must link against the `anira` library. Please note that the `anira` library must be built with the `ANIRA_BUILD_BENCHMARK` option set to `ON`. The Google Benchmark and Google Test libraries will then be linked automatically by the `anira` library. Finally, we will use the `gtest_discover_tests` function to let cmake discover the test case and add it to ctest.

```cmake
# CMakeLists.txt

# New project
project(benchmark_project)

# Add the benchmark target
add_executable(benchmark_target benchmark.cpp test.cpp)

# Link the anira library
target_link_libraries(benchmark_target anira)

# Add the test to the ctest
gtest_discover_tests(benchmark_target)
```

### Step 6: Run the benchmark

After building the project, you can run the benchmark by either running the executable or by running ctest. The ctest runs the benchmark as a unit test and prints the results to the console. To run the ctest, change to the build directory and run the following command:

```bash
ctest -VV

# or to run the test case directly
ctest -R Benchmark.Simple -VV
```

Note: The `-VV` flag prints the test-case output to the console. If you want to change the test timeout for long-running benchmarks, you can do so by passing the `--timeout 100000` flag to the ctest command. The output log of the tests is stored in the `Testing` directory of the build directory.
