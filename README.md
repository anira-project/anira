# ![anira Logo](https://raw.githubusercontent.com/anira-project/anira/main/docs/img/anira-logo.png)

![build_test](https://github.com/anira-project/anira/actions/workflows/build_test.yml/badge.svg)
![build_benchmark](https://github.com/anira-project/anira/actions/workflows/build_benchmark.yml/badge.svg)
![build_examples](https://github.com/anira-project/anira/actions/workflows/build_examples.yml/badge.svg)
![build_docs](https://github.com/anira-project/anira/actions/workflows/build_docs_and_deploy.yml/badge.svg)
![on_tag](https://github.com/anira-project/anira/actions/workflows/on_tag.yml/badge.svg)

---

**Anira** is a high-performance library designed to enable easy real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, [LibTorch](https://github.com/pytorch/pytorch/), [ONNXRuntime](https://github.com/microsoft/onnxruntime/), and [Tensorflow Lite](https://github.com/tensorflow/tensorflow/), anira bridges the gap between advanced neural network architectures and real-time audio processing. In the [paper](https://doi.org/10.1109/IS262782.2024.10704099) you can find more information about the architecture and the design decisions of **anira**, as well as extensive performance evaluations with the built-in benchmarking capabilities.

## Documentation

An extensive documentation of anira can be found at [https://anira-project.github.io/anira/](https://anira-project.github.io/anira/).

<!-- Features -->

## Features

- **Real-time Safe Execution**: Ensures deterministic runtimes suitable for real-time audio applications
- **Thread Pool Management**: Utilizes a static thread pool to avoid oversubscription and enables efficient parallel inference
- **Minimal Latency**: Designed to minimize latency while maintaining real-time safety
- **Built-in Benchmarking**: Includes tools for evaluating the real-time performance of neural networks
- **Comprehensive Inference Engine Support**: Integrates common inference engines, LibTorch, ONNXRuntime, and TensorFlow Lite
- **Flexible Neural Network Integration**: Supports a variety of neural network models, including stateful and stateless models
- **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows

## Usage

The basic usage of anira is as follows:

```cpp
#include <anira/anira.h>

anira::InferenceConfig inference_config(
        {{"path/to/your/model.onnx", anira::InferenceBackend::ONNX}}, // Model path
        {{{256, 1, 1}}, {{256, 1}}},  // Input, Output shape
        5.33f // Maximum inference time in ms
);

// Create a pre- and post-processor instance
anira::PrePostProcessor pp_processor(inference_config);

// Create an InferenceHandler instance
anira::InferenceHandler inference_handler(pp_processor, inference_config);

// Pass the host configuration and allocate memory for audio processing
inference_handler.prepare({buffer_size, sample_rate});

// Select the inference backend
inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

// Optionally get the latency of the inference process in samples
unsigned int latency_in_samples = inference_handler.get_latency();

// Real-time safe audio processing in process callback of your application
process(float** audio_data, int num_samples) {
    inference_handler.process(audio_data, num_samples);
}
// audio_data now contains the processed audio samples
```

## Installation

### C++ Library

Anira can be easily integrated into your CMake project. You can either add anira as a submodule, download the pre-built binaries from the [releases page](https://github.com/anira-project/anira/releases/latest), or build from source.

#### Option 1: Add as Git Submodule (Recommended)

```bash
# Add anira repo as a submodule
git submodule add https://github.com/anira-project/anira.git modules/anira
```

In your `CMakeLists.txt`:

```cmake
# Setup your project and target
project(your_project)
add_executable(your_target main.cpp ...)

# Add anira as a subdirectory
add_subdirectory(modules/anira)

# Link your target to the anira library
target_link_libraries(your_target anira::anira)
```

#### Option 2: Use Pre-built Binaries

Download pre-built binaries from the [releases page](https://github.com/anira-project/anira/releases/latest).

In your `CMakeLists.txt`:

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

#### Option 3: Build from Source

```bash
git clone https://github.com/anira-project/anira.git
cd anira
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target anira
cmake --install build --prefix /path/to/install/directory
```

### C++ Build Options

By default, LibTorch, ONNXRuntime and LiteRT are enabled. You can disable specific backends as needed:

- LibTorch: ``-DANIRA_WITH_LIBTORCH=OFF``
- OnnxRuntime: ``-DANIRA_WITH_ONNXRUNTIME=OFF``
- LiteRT (`LiteRt*` C API): ``-DANIRA_WITH_LITERT=OFF`` — runs `.tflite` models through LiteRT's native CompiledModel runtime. Enabled by default; it is the modern TensorFlow-Lite-family backend.
- TensorFlow Lite (legacy `TfLite*` C API): ``-DANIRA_WITH_TFLITE=ON`` — the **same runtime** as LiteRT exposed through the older C API, so the two are **mutually exclusive**. To use it, disable LiteRT: ``-DANIRA_WITH_LITERT=OFF -DANIRA_WITH_TFLITE=ON``.

Pre-built backend binaries are downloaded at configure time from the
[anira-project/backends](https://github.com/anira-project/backends) release pinned by
`ANIRA_BACKENDS_VERSION` and verified against `cmake/backends_lock.cmake`. Linkage and source are configurable:

- Linkage: ``-DANIRA_BACKEND_LINKAGE=auto|shared|static`` (``auto`` follows ``BUILD_SHARED_LIBS``; static anira → static backends). Per-engine override: ``-DANIRA_<ENGINE>_LINKAGE=...`` where `<ENGINE>` is `LIBTORCH|ONNXRUNTIME|TFLITE|LITERT`. LibTorch is shared-only.
- Backends release tag: ``-DANIRA_BACKENDS_VERSION=v2.1.1`` (regenerate the lockfile when bumping: ``cmake -DTAG=<tag> -P cmake/UpdateBackendsLock.cmake``).
- Bring your own backend (no fork): ``-DANIRA_<ENGINE>_ROOTDIR=/path/to/prebuilt`` (a tree with `include/` + `lib/`), or a custom source via ``-DANIRA_<ENGINE>_URL=... -DANIRA_<ENGINE>_SHA256=...``.

Moreover, the following options are available:

- Build anira with benchmark capabilities: ``-DANIRA_WITH_BENCHMARK=ON``
- Build example applications, plugins and populate example neural models: ``-DANIRA_WITH_EXAMPLES=ON``
- Build anira with tests: ``-DANIRA_WITH_TESTS=ON``
- Build anira with documentation: ``-DANIRA_WITH_DOCS=ON``
- Disable the logging system: ``-DANIRA_WITH_LOGGING=OFF``

### Anira Web (Web / JavaScript)

Anira is available as the `@anira-project/anira` package for use in web applications:

```bash
# npm
npm install @anira-project/anira

# pnpm
pnpm add @anira-project/anira

# yarn
yarn add @anira-project/anira
```

#### Building @anira-project/anira from source

If you want to build the WASM module and JavaScript bindings yourself, you need to provide your own [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html). The CMake presets expect the `EMSDK` environment variable to be set to the root of your emsdk installation.

```bash
git clone https://github.com/anira-project/anira.git
cd anira

export EMSDK=/path/to/your/emsdk

# Configure and build the WASM module (release)
cmake --preset web-prod
cmake --build --preset web-prod

# Build the JavaScript package
cd web
npm install
npm run build
```

For packaging it locally, use
```bash
npm pack
```
in the `web` folder, which will create a `.tgz` file that can be installed with npm or yarn.

Then install the package in your project:

```bash
npm install path/to/anira/web/anira-project-anira-x.x.x.tgz
```

A debug preset is also available via `cmake --preset web` / `cmake --build --preset web`.

## Examples

### Build in examples

- [Simple JUCE Audio Plugin](https://github.com/anira-project/anira/tree/main/examples/juce-audio-plugin/): Demonstrates how to use anira in a real-time audio JUCE / VST3-Plugin.
- [CLAP Plugin Example](https://github.com/anira-project/anira/tree/main/examples/clap-audio-plugin/): Demonstrates how to use anira in a real-time clap plugin.
- [Benchmark](https://github.com/anira-project/anira/tree/main/examples/benchmark/): Demonstrates how to use anira for benchmarking of different neural network models, backends and audio configurations.
- [Minimal Inference](https://github.com/anira-project/anira/tree/main/examples/minimal-inference/): Demonstrates how minimal inference applications can be implemented in all three backends.

### Other examples

- [nn-inference-template](https://github.com/Torsion-Audio/nn-inference-template): Another more JUCE / VST3-Plugin that uses anira for real-time safe neural network inference. This plugin is more complex than the simple JUCE Audio Plugin example as it has a more appealing GUI.

## Real-time safety

anira's real-time safety is checked in [this](https://github.com/anira-project/anira-rt-principle-check) repository with the [rtsan](https://github.com/realtime-sanitizer/rtsan) sanitizer.

## Citation

If you use anira in your research or project, please cite either the [paper](https://doi.org/10.1109/IS262782.2024.10704099) or the software itself:

```bibtex
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

- [Fares Schulz](https://github.com/faressc)
- [Valentin Ackva](https://github.com/vackva)

## Sponsor
<img src="docs/img/bmftr-funding.png" alt="Funded by the German Federal Ministry of Research, Technology and Space (BMFTR)" width="200">

## License

This project is licensed under [Apache-2.0](https://github.com/anira-project/anira/tree/main/LICENSE).
