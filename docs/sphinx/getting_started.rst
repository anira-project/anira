Getting Started
===============

This guide will help you get started with anira for neural network inference in your audio applications.

Prerequisites
-------------

Before using anira, ensure you have:

- A C++ compiler with C++17 support
- CMake (version 3.14 or higher)
- One of the supported neural network model formats:
    - ONNX model files (.onnx)
    - PyTorch model files (.pt, .pth or .ts)
    - TensorFlow Lite model files (.tflite)

Installation
------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Installation
   :end-before: ## Examples

Basic Usage Example
-------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Usage
   :end-before: ## Installation

Using Different Backends
------------------------

Anira supports multiple backends that can be selected at runtime. Use the :cpp:func:`anira::InferenceHandler::set_inference_backend` method to switch between them:

.. code-block:: cpp
    :linenos:

    // Set the inference backend to ONNX
    inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

    // Set the inference backend to PyTorch
    inference_handler.set_inference_backend(anira::InferenceBackend::PYTORCH);

    // Set the inference backend to TensorFlow Lite
    inference_handler.set_inference_backend(anira::InferenceBackend::TFLITE);

    // You can also provide and select a custom backend if needed
    inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);

Multi Tensor Processing Example
-------------------------------

Some neural network models require multiple input tensors or produce multiple output tensors. For example, a model might need both audio data and control parameters as inputs, or output both processed audio and confidence scores. Anira provides flexible methods to handle such models through its multi-tensor processing capabilities.

An important distinction in multi-tensor processing is between **streamable** and **non-streamable** tensors:

- **Streamable tensors**: Contain data that varies over time (e.g., audio samples, time-series data). They can have multiple channels.
- **Non-streamable tensors**: Contain static parameters or metadata (e.g., control parameters, configuration values, global settings). Only one channel is allowed.

Here's how to configure and process multi-tensor models with anira:

.. code-block:: cpp
    :linenos:

    #include <anira/anira.h>

    // Configure a model with multiple inputs and outputs
    anira::InferenceConfig multi_tensor_config(
            {{"path/to/your/multi_tensor_model.onnx", anira::InferenceBackend::ONNX}},
            {{{1, 1, 2048}, {1, 1, 4}},     // Two inputs: audio (2048 samples) + control params (4 values)
             {{1, 1, 2048}, {1, 1, 1}}},    // Two outputs: processed audio (2048 samples) + confidence (1 value)
            anira::ProcessingSpec(          // Optional processing specification
                {1, 1},        // Input channels per tensor: [audio_channels, control_channels]
                {1, 1},        // Output channels per tensor: [audio_channels, confidence_channels]  
                {2048, 0},     // Input sizes: [streamable_audio, non_streamable_params]
                {2048, 0}      // Output sizes: [streamable_audio, non_streamable_confidence]
            ),
            10.0f // Maximum inference time in ms
    );

    // Create pre- and post-processor and inference handler
    anira::PrePostProcessor pp_processor(multi_tensor_config);
    anira::InferenceHandler inference_handler(pp_processor, multi_tensor_config);

    // Prepare for processing
    inference_handler.prepare({buffer_size, sample_rate});
    inference_handler.set_inference_backend(anira::InferenceBackend::ONNX);

    // Optionally get the latency of the inference process in samples
    std::vector<unsigned int> all_latencies = inference_handler.get_latency_vector();

Next step is the real-time processing of audio data and control parameters. The following examples demonstrate how to set inputs, process the data, and retrieve outputs. We have the following inputs and outputs:
    - ``audio_input``: A pointer to a pointer of floats (``float**``) with shape [num_channels][num_samples].
    - ``audio_output``: A pointer to a pointer of floats (``float**``) with shape [num_channels][num_samples].
    - ``control_params``: A pointer to float (``float*``) containing 4 control values.
    - ``confidence_output``: A pointer to float (``float*``) used to receive the confidence score.
    - ``num_samples``: The number of audio samples to process.

Method 1: Individual Tensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp
    :linenos:
    
    // Step 1: Set non-streamable control parameters (tensor index 1)
    for (size_t i = 0; i < 4; ++i) {
        pp_processor.set_input(control_params[i], 1, i);  // tensor_index=1, sample_index=i
    }

    // Step 2: Process streamable audio data (tensor index 0)
    inference_handler.process(audio_input, num_samples, 0); // Process audio data in tensor 0
    // audio_input now contains processed audio data

    // Step 3: Retrieve non-streamable confidence output (tensor index 1)
    *confidence_output = pp_processor.get_output(1, 0);  // Get confidence from tensor 1, sample 0

Method 2: Multi-Tensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp
    :linenos:

    // Allocate memory for input data and output data (not in the real-time callback)
    const float* const* const* input_data = new const float* const*[2];
    float* const* const* output_data = new float* const*[2];

    // Prepare input data structure: [tensor_index][channel][sample]
    input_data[0] = audio_input;                            // Tensor 0: streamable audio data
    input_data[1] = (const float* const*) &control_params;  // Tensor 1: non-streamable control params

    // Prepare output data structure: [tensor_index][channel][sample]  
    output_data[0] = audio_output;                          // Tensor 0: processed audio output
    output_data[1] = (float* const*) &confidence_output;    // Tensor 1: confidence score output

    // Specify number of samples for each tensor
    size_t input_samples[2] = {num_samples, 4};             // Audio: num_samples, Control: 4 values
    size_t output_samples[2] = {num_samples, 1};            // Audio: num_samples, Confidence: 1 value

    // Process all tensors in one call
    size_t* processed_samples = inference_handler.process(
        input_data, input_samples, output_data, output_samples);

    // Clean up the allocated memory after processing
    delete[] input_data;
    delete[] output_data;

Key Points for Multi-Tensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Tensor Organization and Indexing**

- **Tensor indexing**: Tensors are indexed starting from 0, following the order specified in the ``TensorShape`` configuration
- **Data structure**: Multi-tensor data uses a 3D array structure: ``[tensor_index][channel][sample]``

**Streamable vs Non-Streamable Tensors**

- **Streamable tensors**: Time-varying data (audio, time-series) that flows continuously through the processing pipeline
- **Non-streamable tensors**: Static parameters or metadata that is updated asynchronously
- **Configuration**: Set processing sizes to 0 for non-streamable tensors in the ``ProcessingSpec``

**Processing Methods**

- **Individual tensor processing**: Use the tensor index parameter in :cpp:func:`anira::InferenceHandler::process` for processing specific tensors separately
- **Simultaneous processing**: Pass all tensors at once using the multi-tensor version of :cpp:func:`anira::InferenceHandler::process`
- **Push/Pop workflow**: Use :cpp:func:`anira::InferenceHandler::push_data` and :cpp:func:`anira::InferenceHandler::pop_data` for granular control over data flow

**Non-Streamable Data Access**

- **Setting inputs**: Use :cpp:func:`anira::PrePostProcessor::set_input` to provide non-streamable input data
- **Getting outputs**: Use :cpp:func:`anira::PrePostProcessor::get_output` to retrieve non-streamable output data
- **Real-time safety**: These methods are designed for real-time use with pre-allocated internal buffers

.. note::
    Streamable tensors can not be accessed with the :cpp:func:`anira::PrePostProcessor::set_input` and :cpp:func:`anira::PrePostProcessor::get_output` methods.

.. note::
    Non-streamable tensors will allways have a single channel and a latency of 0 samples, as they are not time-varying.

.. tip::
    When designing multi-tensor models, consider separating time-varying audio data (streamable) from control parameters (non-streamable).

Next Steps
----------

- Check the :doc:`usage` page for more detailed usage instructions
- See the :doc:`examples` page for complete example applications
- Review the :doc:`architecture` to understand anira's design
- Try the :doc:`benchmarking` tools to evaluate your models' performance
