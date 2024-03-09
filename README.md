![anira Logo](/docs/img/anira-logo.png)

--------------------------------------------------------------------------------

> **Note: This project is still a work in progress. We are actively developing and refining its features and documentation.**

**anira** is a high-performance library designed to enable the real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, [LibTorch](https://github.com/pytorch/pytorch/), [ONNXRuntime](https://github.com/microsoft/onnxruntime/), and [Tensorflow Lite](https://github.com/tensorflow/tensorflow/), anira bridges the gap between advanced neural network architectures and real-time audio processing.

## Features

- **Real-time Safe Execution**: Ensures deterministic runtimes suitable for real-time audio applications
- **Thread Pool Management**: Utilizes a static thread pool to avoid oversubscription and enables efficient parallel inference
- **Cross-Platform Compatibility**: Works seamlessly on MacOS, Linux, and Windows
- **Flexible Neural Network Integration**: Supports a variety of neural network models, including stateful and stateless models
- **Comprehensive Inference Engine Support**: Integrates common inference engines, LibTorch, ONNXRuntime, and TensorFlow Lite
- **Built-in Benchmarking**: Includes tools for evaluating the real-time performance of neural networks

### General usage

| Class             | Description                                                                                                                   |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `InferenceHandler` | Manages audio processing within real-time threads by routing audio input to inference threads and updating the buffer with processed audio. |
| `PrePostProcessor` | Facilitates model-specific pre- and post-processing. Implement this class to customize processing steps before and after inference. |
| `InferenceConfig`  | A configuration struct for defining model specifics, such as input/output sample counts, model details, batch size, and more. |

For more details checkout: [anira usage](docs/anira-usage.md)

### Install
Build anira as shared library
```bash
git clone https://github.com/tu-studio/anira
cmake . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release --target anira
```
**Note:** The CMake build automatically installs all dependencies for the following targets:
 - Benchmarks (disable with ```-DANIRA_WITH_BENCHMARK=OFF```)
 - Example neural models (disable with ```-DANIRA_WITH_EXTRAS=OFF```)
 - JUCE plugin example (disable with ```-DCANIRA_WITH_EXTRAS=OFF```)
 - Minimum inference example (disable with ```-DCANIRA_WITH_EXTRAS=OFF```)

**Moreover** by default, all three inference engines are installed. You can:
- Disable this automatism with ```-DANIRA_BACKEND_ALL=OFF```
- Enable specific backends as needed:
  - LibTorch: ```-DANIRA_WITH_LIBTORCH=ON```
  - OnnxRuntime: ```-DANIRA_WITH_ONNXRUNTIME=ON```
  - Tensrflow Lite. ```-DANIRA_WITH_TFLITE=ON```

### Documentation
Detailed documentation on **anira**'s API and how to integrate custom neural networks will be available soon in our upcoming wiki. In the meantime, check out the examples below.

### Examples
- [juce-plugin-example](https://github.com/tu-studio/anira/tree/main/examples/juce-audio-plugin): Demonstrates how to use **anira** with three different neural models in a real-time audio application.
- [anira-rt-principle-check](https://github.com/tu-studio/anira-rt-principle-check): Shows how to integrate **anira** as a submodule with CMake.
  
### Citation
If you use Anira in your research or project, please cite our software: 
```
@software{anira2024ackvaschulz,
  author = {Fares Schulz and Valentin Ackva},
  title = {anira: an architecture for neural network inference in real-time audio application},
  url = {https://github.com/tu-studio/anira},
  version = {0.0.1-alpha},
  year = {2024},
}
```

## Contributors
- [Valentin Ackva](https://github.com/vackva)
- [Fares Schulz](https://github.com/faressc)

## License
This project is licensed under [Apache-2.0](LICENSE).
