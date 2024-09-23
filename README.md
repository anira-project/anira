![anira Logo](/docs/img/anira-logo.png)

![Build Status](https://github.com/anira-project/anira/actions/workflows/build.yml/badge.svg)
--------------------------------------------------------------------------------

**anira** is a high-performance library designed to enable easy real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, [LibTorch](https://github.com/pytorch/pytorch/), [ONNXRuntime](https://github.com/microsoft/onnxruntime/), and [Tensorflow Lite](https://github.com/tensorflow/tensorflow/), anira bridges the gap between advanced neural network architectures and real-time audio processing.

## Features

- **Real-time Safe Execution**: Ensures deterministic runtimes suitable for real-time audio applications
- **Thread Pool Management**: Utilizes a static thread pool to avoid oversubscription and enables efficient parallel inference
- **Built-in Benchmarking**: Includes tools for evaluating the real-time performance of neural networks
- **Comprehensive Inference Engine Support**: Integrates common inference engines, LibTorch, ONNXRuntime, and TensorFlow Lite
- **Flexible Neural Network Integration**: Supports a variety of neural network models, including stateful and stateless models
- **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows

## Usage

An extensive anira usage guide can be found [here](docs/anira-usage.md).

The basic usage of anira is as follows:

```cpp
#include <anira/anira.h>

// Create a model configuration struct for your neural network
anira::InferenceConfig myNNConfig(
    "path/to/your/model.onnx (or *.pt, *.tflite)", // Model path
    {2048, 1, 150}, // Input shape
    {2048, 1}, // Output shape
    42.66f // Maximum inference time in ms
);

// Create a pre- and post-processor instance
anira::PrePostProcessor myPrePostProcessor;

// Create an InferenceHandler instance
anira::InferenceHandler inferenceHandler(myPostProcessor, myNNConfig);

// Create a HostAudioConfig instance containing the host config infos
anira::HostAudioConfig audioConfig {
    1, // currently only mono is supported
    bufferSize,
    sampleRate
};

// Allocate memory for audio processing
inferenceHandler.prepare(audioConfig);

// Select the inference backend
inferenceHandler.selectInferenceBackend(anira::LIBTORCH);

// Optionally get the latency of the inference process in samples
int latencyInSamples = inferenceHandler.getLatency();

// Real-time safe audio processing in process callback of your application
processBlock(float** audioData, int numSamples) {
    inferenceHandler.process(audioData, numSamples);
}
// audioData now contains the processed audio samples
```

## Install

### CMake

anira can be easily integrated into your CMake project. Either add anira as a submodule or download the pre-built binaries from the [releases page](https://github.com/anira-project/anira/releases/latest).

#### Add as a git submodule

```bash
# Add anira repo as a submodule
git submodule add https://github.com/anira-project/anira.git modules/anira
```

In your CMakeLists.txt, add anira as a subdirectory and link your target to the anira library:

```cmake
# Setup your project and target
project(your_project)
add_executable(your_target main.cpp ...)

# Add anira as a subdirectory
add_subdirectory(modules/anira)

#Link your target to the anira library
target_link_libraries(your_target anira::anira)
```

#### With pre-built binaries

Download the pre-built binaries from your operating system and architecture from the [releases page](https://github.com/anira-project/anira/releases/latest).

```cmake
# Setup your project and target
project(your_project)
add_executable(your_target main.cpp ...)

# Add the path to the anira library as cmake prefix path and find the package
list(APPEND CMAKE_PREFIX_PATH "path/to/anira")
find_package(anira REQUIRED)

# Link your target to the anira library
target_link_libraries(your_target anira::anira)
```

### Build from source

You can also build anira from source using CMake. All dependencies are automatically installed during the build process.

```bash
git clone https://github.com/anira-project/anira
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target anira
```

### Build options

By default, all three inference engines are installed. You can disable specific backends as needed:

- LibTorch: ```-DANIRA_WITH_LIBTORCH=OFF```
- OnnxRuntime: ```-DANIRA_WITH_ONNXRUNTIME=OFF```
- Tensrflow Lite. ```-DANIRA_WITH_TFLITE=OFF```

The method of thread synchronization can be chosen between hard real-time safe raw atomic operations and an option with semaphores. The option with semaphores allows the use of `wait_in_process_block` in the `InferenceConfig` class. The default is the raw atomic operations. To enable the semaphore option, use the following flag:

- Use semaphores for thread synchronization: ```-DANIRA_WITH_SEMAPHORES=ON```

Moreover the following options are available:

- Build anira with benchmark capabilities: ```-DANIRA_WITH_BENCHMARK=ON```
- Build example applications and populate example neural models: ```-DANIRA_WITH_EXAMPLES=ON```

## Documentation

For using anira to inference your custom models, check out the [extensive usage guide](docs/anira-usage.md). If you want to use anira for benchmarking, check out the [benchmarking guide](docs/benchmark-usage.md) and the section below.
Detailed documentation on anira's API and will be available soon in our upcoming wiki.

## Benchmark capabilities

anira allows users to benchmark and compare the inference performance of different neural network models, backends, and audio configurations. The benchmarking capabilities can be enabled during the build process by setting the ```-DANIRA_WITH_BENCHMARK=ON``` flag. The benchmarks are implemented using the [Google Benchmark](https://github.com/google/benchmark) and [Google Test](https://github.com/google/googletest) libraries. Both libraries are automatically linked with the anira library in the build process when benchmarking is enabled. To provide a reproducible and easy-to-use benchmarking environment, anira provides a custom Google benchmark fixture `anira::benchmark::ProcessBlockFixture` that is used to define benchmarks. This fixture offers many useful functions for setting up and running benchmarks. For more information on how to use the benchmarking capabilities, check out the [benchmarking guide](docs/benchmark-usage.md).

## Examples

### Build in examples

- [Simple JUCE Audio Plugin](examples/juce-audio-plugin/): Demonstrates how to use anira in a real-time audio JUCE / VST3-Plugin.
- [Benchmark](examples/benchmark/): Demonstrates how to use anira for benchmarking of different neural network models, backends and audio configurations.
- [Minimal Inference](examples/minimal-inference/): Demonstrates how minimal inference applications can be implemented in all three backends.

### Other examples

- [nn-inference-template](https://github.com/Torsion-Audio/nn-inference-template): Another more JUCE / VST3-Plugin that uses anira for real-time safe neural network inference. This plugin is more complex than the simple JUCE Audio Plugin example and has a more appealing GUI.

## Real-time safety

anira's real-time safety is checked in [this](https://github.com/anira-project/anira-rt-principle-check) repository with the [radsan](https://github.com/realtime-sanitizer/radsan) sanitizer.

## Citation

If you use anira in your research or project, please cite our work:

```cite
@software{anira2024ackvaschulz,
  author = {Valentin Ackva and Fares Schulz},
  title = {anira: an architecture for neural network inference in real-time audio application},
  url = {https://github.com/anira-project/anira},
  version = {x.x.x},
  year = {2024},
}
```

## Contributors

- [Valentin Ackva](https://github.com/vackva)
- [Fares Schulz](https://github.com/faressc)

## License
This project is licensed under [Apache-2.0](LICENSE).
