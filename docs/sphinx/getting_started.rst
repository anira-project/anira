Getting Started
===============

This guide will help you get started with anira for neural network inference in your audio applications.

Prerequisites
-------------

Before using anira, ensure you have:

- A C++20 compiler (anira's C headers compile as C11 and C++17; the C++ builders of ``anira/anira.hpp`` and the examples need C++20)
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

    // Set the inference backend to LibTorch (PyTorch)
    inference_handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);

    // Set the inference backend to LiteRT (default TensorFlow Lite family backend)
    inference_handler.set_inference_backend(anira::InferenceBackend::LITERT);

    // Set the inference backend to the legacy TensorFlow Lite backend
    // (only available when built with -DANIRA_WITH_TFLITE=ON; mutually exclusive with LiteRT)
    inference_handler.set_inference_backend(anira::InferenceBackend::TFLITE);

    // Set the inference backend to ExecuTorch (.pte programs exported with torch.export)
    inference_handler.set_inference_backend(anira::InferenceBackend::EXECUTORCH);

    // A custom engine is registered on the pipeline under its id (see "Custom Engines"); the
    // 2.x handler of this pre-release selects the BackendBase it was constructed with as CUSTOM
    inference_handler.set_inference_backend(anira::InferenceBackend::CUSTOM);

Multi Tensor Processing Example
-------------------------------

Some neural network models take several input tensors or produce several output tensors: audio
and a control vector in, processed audio and a confidence score out, or a hidden state that the
model returns and expects back on its next call. Every tensor of the model file has a **role**,
and the role says how the tensor travels between your code and the model:

- **Streamed** (``"role": "streamed"``, the default): data that varies over time, with exactly
  one Time axis. You hand the handler blocks of samples of any size; anira buffers them, forms
  the model's window (hop plus context) and hands the result back the same way. A Channel axis
  of any extent maps onto the ring's channels: a stereo stream is one tensor of two channels.
- **Static** (``"role": "static"``): one whole value per inference, no Time axis. Control
  parameters, an embedding, a class vector, a confidence score. You set an input whole, in the
  spec's shape and dtype, and read an output whole; a Channel axis is just an axis of that shape
  and may have any extent. The value set last is the one every later inference sees, and the
  value read is the one the latest completed inference produced.
- **State** (``"role": "state"``): a pair, an output the model expects back in one of its inputs
  on the next inference (a recurrent network's hidden state, a streaming convolution's cache).
  The input names the output it is fed from (``"state_source"``); anira feeds and captures the
  pair itself, and your code never sees it. Section 3.4 of the :doc:`usage` guide has the
  bundled ``StatefulAccumulatorNetwork`` as the worked example.
- **Buffer** (``"role": "buffer"``): a whole submitted buffer is one model tensor, with no
  windowing: an image, a whole utterance. This is the tensor of the *Async* contract; under a
  *Hard* contract, the real-time contract this page uses, a Buffer spec is refused at prepare in
  this pre-release.

The model file below names every tensor and gives the two control tensors the ``static``
role; the two streamed tensors carry a window:

.. code-block:: json
    :caption: multi_tensor.model.json

    {
      "models": [ { "engine": "onnxruntime", "path": "multi_tensor_model.onnx" } ],
      "inputs": [
        { "name": "audio_in", "axes": [["batch", 1], ["channel", 1], ["time", 2048]],
          "window": { "min": 2048, "max": 2048 } },
        { "name": "control", "role": "static", "axes": [["batch", 1], ["channel", 1], ["any", 4]] }
      ],
      "outputs": [
        { "name": "audio_out", "axes": [["batch", 1], ["channel", 1], ["time", 2048]],
          "window": { "min": 2048, "max": 2048 } },
        { "name": "confidence", "role": "static", "axes": [["batch", 1], ["channel", 1], ["any", 1]] }
      ]
    }

A tensor is addressed by its **slot**: its position in the model file's ``inputs`` list or
``outputs`` list, counted from 0, whatever its role. Here ``audio_in`` is input slot 0,
``control`` input slot 1, ``audio_out`` output slot 0 and ``confidence`` output slot 1. The
configuration is built with the C++ builders of ``anira/anira.hpp`` and the model runs on the C
handler of ``anira/abi/handler.h``, the 3.x runtime of this pre-release:

.. code-block:: cpp
    :linenos:

    #include <anira/anira.hpp>

    #include <chrono>

    // The model file, a Hard contract with a 10 ms budget, the context and the pipeline
    anira::ModelConfig model_config = anira::ModelConfig::from_file("multi_tensor.model.json");
    anira::ContractHandle contract{anira::Hard{
        .budget = ANIRA_BUDGET_EXPLICIT, .budget_value = std::chrono::milliseconds(10),
        .warmup = ANIRA_WARMUP_FIXED, .warmup_iterations = 2}};
    anira::ContextConfig context_config;
    anira::Context context(context_config);
    anira::Pipeline pipeline{anira::stage::Inference(model_config)};   // every engine of the build

    // The handler, prepared with the host geometry: block size and sample rate complete the contract
    anira_handler* handler = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (ANIRA_FAILED(anira_handler_create(context.native(), pipeline.native(), &handler, &err))) {
        /* err.message says why */
    }
    contract.hard_geometry(buffer_size, buffer_size, sample_rate);
    if (ANIRA_FAILED(anira_handler_prepare(handler, contract.native(), &err))) {
        /* err.message names the tensor and the rule */
    }

    // Optionally the latency of the stream in samples (a Static output reports 0)
    const uint32_t latency = anira_handler_get_latency(handler, 0);

In the real-time callback every value crosses as an ``anira_tensor``, a plain descriptor of
your memory: the audio blocks as planar tensors over the host's channel pointers, the Static
values whole, in the spec's shape. With ``audio_input`` and ``audio_output`` as ``float**``
arrays of ``num_channels`` pointers, ``control_params`` as four floats and ``confidence`` as one:

.. code-block:: cpp
    :linenos:

    // Step 1: the control vector, whole, in the spec's shape [1][1][4]. Every inference
    // submitted from now on sees it; set it again whenever it changes.
    const int64_t control_shape[3] = {1, 1, 4};
    anira_tensor control;
    anira_tensor_init_host(&control, control_params, ANIRA_DTYPE_F32, 3, control_shape);
    anira_handler_set_static_input(handler, 1, &control);

    // Step 2: the audio block in and out of slot 0: [channels][samples], planar
    const int64_t block_shape[2] = {static_cast<int64_t>(num_channels), static_cast<int64_t>(num_samples)};
    anira_tensor in_block;
    anira_tensor out_block;
    anira_tensor_init_host_planar(&in_block, audio_input, num_channels, ANIRA_DTYPE_F32, 2, block_shape);
    anira_tensor_init_host_planar(&out_block, audio_output, num_channels, ANIRA_DTYPE_F32, 2, block_shape);
    size_t delivered = 0;
    anira_handler_process(handler, &in_block, 0, &out_block, 0, &delivered);
    // audio_output holds delivered samples per channel (zeros for the first latency samples)

    // Step 3: the confidence the latest completed inference produced, whole, shape [1][1][1]
    const int64_t confidence_shape[3] = {1, 1, 1};
    anira_tensor confidence;
    anira_tensor_init_host(&confidence, &confidence_value, ANIRA_DTYPE_F32, 3, confidence_shape);
    anira_handler_get_static_output(handler, 1, &confidence);

Key Points for Multi-Tensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**One slot space**

- A slot is the tensor's position in the model file's ``inputs`` or ``outputs`` list, from 0,
  State tensors included; every handler entry, the latency vector and the plan report use it.
- The role decides which entries take a slot: a Streamed slot the block calls, a Static slot
  the two Static entries, a State slot none (anira owns it), a Buffer slot the Async contract's.

**Streamed tensors**

- ``anira_handler_process(h, &in, in_slot, &out, out_slot, &delivered)`` moves one block in
  and one block out, any block size, one Streamed slot per side; ``anira_handler_push_data`` and
  ``anira_handler_pop_data`` split the two halves for hosts that produce and consume at
  different rates; the ``_multi`` forms take one tensor per slot of a side in one call.
- A block is ``[channels][samples]``, planar over the host's channel pointers or one packed
  block; the dtype is the ring's, float32 unless the contract declares another.

**Static tensors**

- ``anira_handler_set_static_input(h, slot, &tensor)`` and
  ``anira_handler_get_static_output(h, slot, &tensor)`` take the whole tensor in the spec's
  shape and dtype; "fits" is a check, never a clamp. Both are real-time safe and legal from
  create on, so a value can be set before prepare.
- A Static tensor reports a latency of 0: it is not time-varying.

**State tensors**

- Declared in the model file, run by anira: fed ahead of the inference, captured behind it, kept
  across a plan switch, re-initialised by ``anira_handler_reset``. Nothing to call.

.. note::
    The 2.x handler of the sections above addresses a non-streamable tensor as channel 0 of a
    ``float***`` block; the C handler takes it whole, in the spec's shape (:doc:`usage`
    section 3.2, *Static tensors*). The :doc:`migration` page maps the 2.x calls onto the C entries.

Next Steps
----------

- Check the :doc:`usage` page for more detailed usage instructions
- See the :doc:`examples` page for complete example applications
- Review the :doc:`architecture` to understand anira's design
- Try the :doc:`benchmarking` tools to evaluate your models' performance
