![anira Logo](/docs/img/anira-logo.png)

![Build Status](https://github.com/anira-project/anira/actions/workflows/build.yml/badge.svg)
--------------------------------------------------------------------------------

**anira** is a high-performance library designed to enable easy real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, [LibTorch](https://github.com/pytorch/pytorch/), [ONNXRuntime](https://github.com/microsoft/onnxruntime/), and [Tensorflow Lite](https://github.com/tensorflow/tensorflow/), anira bridges the gap between advanced neural network architectures and real-time audio processing. In the [paper](https://doi.org/10.1109/IS262782.2024.10704099) you can find more information about the architecture and the design decisions of **anira**, as well as extensive performance evaluations with the built-in benchmarking capabilities.

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

anira::InferenceConfig inference_config(
        {{"path/to/your/model.onnx", anira::InferenceBackend::ONNX}}, // Model path
        {{{256, 1, 150}}, {{256, 1}}},  // Input, Output shape
        5.33f // Maximum inference time in ms
);

// Create a pre- and post-processor instance
anira::PrePostProcessor pp_processor;

// Create an InferenceHandler instance
anira::InferenceHandler inference_handler(pp_processor, inference_config);

// Pass the host audio configuration and allocate memory for audio processing
inference_handler.prepare({buffer_size, sample_rate});

// Select the inference backend
inference_handler.set_inference_backend(anira::ONNX);

// Optionally get the latency of the inference process in samples
int latency_in_samples = inference_handler.get_latency();

// Real-time safe audio processing in process callback of your application
process(float** audio_data, int num_samples) {
    inference_handler.process(audio_data, num_samples);
}
// audio_data now contains the processed audio samples
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

To allow a controversial approach of controlled blocking in the audio callback to further reduce latency, a flag can be set to allow the use of a semaphore. The semaphore is not 100% real-time safe, but it allows the use of the `wait_in_process_block` option in the `InferenceConfig` class. We only recommend that you use this option if you are not spawning multiple instances of the `InferenceHandler' in serial. By default we use a real-time safe raw atomic operation. To enable the semaphore option, use the following flag:

- Use semaphores for thread synchronization: ```-DANIRA_WITH_SEMAPHORES=ON```

Moreover the following options are available:

- Build anira with benchmark capabilities: ```-DANIRA_WITH_BENCHMARK=ON```
- Build example applications, plugins and populate example neural models: ```-DANIRA_WITH_EXAMPLES=ON```
- Build a Bela example application: ```-DANIRA_WITH_BELA_EXAMPLE=ON```
- Build anira with tests: ```-DANIRA_WITH_TESTS=ON```


## Documentation

For using anira to inference your custom models, check out the [extensive usage guide](docs/anira-usage.md). If you want to use anira for benchmarking, check out the [benchmarking guide](docs/benchmark-usage.md) and the section below.
Detailed documentation on anira's API and will be available soon in our upcoming wiki.

## Benchmark capabilities

anira allows users to benchmark and compare the inference performance of different neural network models, backends, and audio configurations. The benchmarking capabilities can be enabled during the build process by setting the ```-DANIRA_WITH_BENCHMARK=ON``` flag. The benchmarks are implemented using the [Google Benchmark](https://github.com/google/benchmark) and [Google Test](https://github.com/google/googletest) libraries. Both libraries are automatically linked with the anira library in the build process when benchmarking is enabled. To provide a reproducible and easy-to-use benchmarking environment, anira provides a custom Google benchmark fixture `anira::benchmark::ProcessBlockFixture` that is used to define benchmarks. This fixture offers many useful functions for setting up and running benchmarks. For more information on how to use the benchmarking capabilities, check out the [benchmarking guide](docs/benchmark-usage.md).

## Examples

### Build in examples

- [Simple JUCE Audio Plugin](examples/desktop/juce-audio-plugin/): Demonstrates how to use anira in a real-time audio JUCE / VST3-Plugin.
- [CLAP Plugin Example](examples/desktop/clap-audio-plugin/): Demonstrates how to use anira in a real-time clap plugin utilizing host provided threads instead of anira's thread pool.
- [Benchmark](examples/desktop/benchmark/): Demonstrates how to use anira for benchmarking of different neural network models, backends and audio configurations.
- [Minimal Inference](examples/desktop/minimal-inference/): Demonstrates how minimal inference applications can be implemented in all three backends.
- [Bela Example](examples/embedded/bela/bela-inference/): Demonstrates how to use anira in a real-time audio application on the Bela platform.
- [Bela Benchmark](examples/embedded/bela/bela-benchmark/): Demonstrates how to use anira for benchmarking on the Bela platform.

### Other examples

- [nn-inference-template](https://github.com/Torsion-Audio/nn-inference-template): Another more JUCE / VST3-Plugin that uses anira for real-time safe neural network inference. This plugin is more complex than the simple JUCE Audio Plugin example as it has a more appealing GUI.

## Real-time safety

anira's real-time safety is checked in [this](https://github.com/anira-project/anira-rt-principle-check) repository with the [rtsan](https://github.com/realtime-sanitizer/rtsan) sanitizer.

## Citation

If you use anira in your research or project, please cite either the [paper](https://doi.org/10.1109/IS262782.2024.10704099) our the software itself:

```cite
@inproceedings{ackvaschulz2024anira,
    author={Ackva, Valentin and Schulz, Fares},
    booktitle={2024 IEEE 5th International Symposium on the Internet of Sounds (IS2)},
    title={ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications}, 
    year={2024},
    volume={},
    number={},
    pages={1-10},
    publisher={IEEE},
    doi={10.1109/IS262782.2024.10704099}
}

@software{ackvaschulz2024anira,
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
