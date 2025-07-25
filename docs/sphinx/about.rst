About
=====

**anira** is a C++ library designed to streamline the development of real-time audio applications that integrate neural network inference. It provides a high-performance, real-time safe execution environment for neural networks, ensuring deterministic runtimes that meet the demands of professional audio processing.

Neural network inference in anira is powered by industry-standard engines, including LibTorch, ONNXRuntime, and TensorFlow Lite. The library offers a unified interface to these engines through the :cpp:class:`anira::InferenceHandler` class, delegating inference execution to a static thread pool. This architecture maintains real-time safety by executing inference outside the audio thread, ensuring applications remain responsive and deterministic. Additionally, anira leverages multiple CPU cores for efficient parallel inference.

Anira is optimized to minimize latency and supports predefined tensor shapes for neural networks. A key feature is the intelligent adaptation of host audio buffers to neural network input and output tensors, with automatic calculation of the minimum required latency based on the :cpp:struct:`anira::InferenceConfig` struct. If neural network processing exceeds the latency threshold, anira compensates for missing frames to maintain smooth and synchronized audio processing.

Model inputs and outputs can be preprocessed and postprocessed using built-in functionality. For custom data handling, developers can extend the :cpp:class:`anira::PrePostProcessor` class to implement specialized logic. This flexibility ensures neural network models receive properly formatted data and that results are correctly integrated into the audio processing pipeline.

The library supports a wide range of neural network architectures, including both stateful and stateless models, with single or multiple input and output tensors. Tensors are categorized as streamable (for time-varying data like audio signals) or non-streamable (for static parameters requiring asynchronous updates). Since version 2.0, anira supports input and output tensors with varying sizes and sampling rates, enabling more complex processing scenarios and greater architectural flexibility.

Anira also features built-in benchmarking tools, allowing developers to evaluate neural network performance within the same environment as their audio applications. This is essential for optimizing applications to meet real-time processing requirements.

While anira is primarily focused on audio processing, it is also suitable for other real-time applications such as robotics and computer vision, where both streamable and non-streamable data processing are required. Its design principles and real-time safety features make it a versatile tool for developers across various domains.