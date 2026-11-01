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

The installed package
~~~~~~~~~~~~~~~~~~~~~

An installed anira (``cmake --install``, or a release zip) carries the library and its public
headers, and nothing of its private dependencies:

.. code-block:: text

    include/anira/anira.h            the C umbrella (every header of anira/abi/)
    include/anira/anira_all.h        the C umbrella plus the platform companions
    include/anira/anira_em.h         the WebAssembly companion (empty on every other target)
    include/anira/anira.hpp          the C++20 face, header-only over the C entries
    include/anira/compat/v2.hpp      the 2.x source-compatible face, header-only
    include/anira/abi/*.h            the C ABI, one header per area; abi/draft/ the draft entries
    lib/libanira.so.0 (libanira.0.dylib, bin/anira-0.dll + lib/anira-0.lib, or lib/libanira.a)
    include/anira-engines/<engine>/  and the engine libraries, for code that calls an engine itself
    lib/cmake/anira/                 the CMake package: find_package(anira) → anira::anira

The shared library is named after the ABI major (``0`` through the 3.0.0 pre-releases, ``3`` from
v3.0.0 on), exports the ``anira_*`` C entries and nothing else, and carries tanh-lib's core
inside it; the queue and the JSON library are header-only and private. No C++ header of anira's
core is installed. ``find_package(anira)`` looks up nothing but ``Threads`` and the engines the
build was made with.

Linking anira from C
~~~~~~~~~~~~~~~~~~~~

The C ABI is C11. A C project links ``anira::anira`` like a C++ one; with a static anira, which
is C++ inside, the executable links through the C++ driver:

.. code-block:: cmake

    project(my_host LANGUAGES C CXX)
    find_package(anira REQUIRED)
    add_executable(my_host main.c)
    target_link_libraries(my_host PRIVATE anira::anira)
    set_target_properties(my_host PROPERTIES LINKER_LANGUAGE CXX)  # static anira only

.. code-block:: c

    #include <anira/anira.h>

    int main(void) {
        anira_error err = ANIRA_ERROR_INIT;
        if (ANIRA_FAILED(anira_check_abi(ANIRA_ABI_VERSION))) { return 1; }  /* header == library */
        anira_context_config* config = NULL;
        anira_context* context = NULL;
        anira_context_config_create(&config, &err);
        if (ANIRA_FAILED(anira_context_create(config, &context, &err))) { return 1; }
        /* ... a model config, a pipeline, a handler, prepared with a Hard contract ... */
        anira_context_destroy(context);
        anira_context_config_destroy(config);
        return ANIRA_FAILED(anira_shutdown()) ? 1 : 0;
    }

``test/contracts/install/main_c.c`` is a complete one: a registered engine, a handler prepared
under a Hard contract and one block through ``anira_handler_process``.

Basic Usage Example
-------------------

.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: ## Usage
   :end-before: ## Installation

Using Different Backends
------------------------

Anira supports multiple engines; which ones a build carries depends on its options
(``anira::enabled_engines()`` lists them). ``prepare`` returns the plan report, one plan per
model entry the build can run, and :cpp:func:`anira::InferenceHandler::set_plan` switches between
them at runtime, also while audio runs:

.. code-block:: cpp
    :linenos:

    const anira::PlanReport report = inference_handler.prepare(contract);

    // The plan on ONNX Runtime; the other engines are ANIRA_ENGINE_LIBTORCH (PyTorch),
    // ANIRA_ENGINE_LITERT (the default TensorFlow Lite family backend), ANIRA_ENGINE_TFLITE
    // (the legacy TensorFlow Lite backend, -DANIRA_WITH_TFLITE=ON, mutually exclusive with
    // LiteRT) and ANIRA_ENGINE_EXECUTORCH (.pte programs exported with torch.export)
    if (auto plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}})) {
        inference_handler.set_plan(*plan);
    }

    // A custom engine is added to the pipeline under its id (see "Custom Engines"), and its
    // plan is found by that id: report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
    // .engine_id = "com.example.engine"}})

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
configuration is built with the C++ builders of ``anira/anira.hpp``, and the model runs on
:cpp:class:`anira::InferenceHandler`, the C++ face of the C handler of ``anira/abi/handler.h``:

.. code-block:: cpp
    :linenos:

    #include <anira/anira.hpp>

    #include <chrono>

    using namespace std::chrono_literals;

    // The model file, the context and the pipeline (the default candidate set)
    anira::ModelConfig model_config = anira::ModelConfig::from_file("multi_tensor.model.json");
    anira::Context context{anira::ContextConfig{}};
    anira::Pipeline pipeline;
    pipeline.add_inference(model_config);
    anira::InferenceHandler inference_handler(context, pipeline);

    // Prepared with a Hard contract: the host's block size and sample rate, a 10 ms budget
    inference_handler.prepare(anira::Hard{.host_block_min = static_cast<double>(buffer_size),
                                          .host_block_max = static_cast<double>(buffer_size),
                                          .host_rate = sample_rate,
                                          .budget = 10ms,
                                          .warmup = ANIRA_WARMUP_FIXED,
                                          .warmup_iterations = 2});

    // Optionally the latency of the stream in samples (a Static output reports 0)
    const uint32_t latency = inference_handler.get_latency(0);

Both throw ``anira::Error`` with the status and the reason (``what()`` names the tensor and the
rule). In the real-time callback every value crosses as an :cpp:class:`anira::Tensor`, a plain
descriptor of your memory: the audio blocks as planar tensors over the host's channel pointers,
the Static values whole, in the spec's shape. With ``audio_input`` and ``audio_output`` as
``float**`` arrays of ``num_channels`` pointers, ``control_params`` as four floats and
``confidence_value`` as one, the tensors are built once at prepare:

.. code-block:: cpp
    :linenos:

    // The audio blocks: [channels][samples], planar over the pointer arrays
    const std::array<int64_t, 2> block_shape{static_cast<int64_t>(num_channels),
                                             static_cast<int64_t>(max_block)};
    anira::Tensor in_block = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(audio_input, num_channels), block_shape);
    anira::Tensor out_block = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(audio_output, num_channels), block_shape);
    // The Static values, whole, in the specs' shapes [1][1][4] and [1][1][1]
    const anira::Tensor control = anira::Tensor::from_host(
        control_params, ANIRA_DTYPE_F32, std::array<int64_t, 3>{1, 1, 4});
    const anira::Tensor confidence = anira::Tensor::from_host(
        &confidence_value, ANIRA_DTYPE_F32, std::array<int64_t, 3>{1, 1, 1});

and every block runs through them:

.. code-block:: cpp
    :linenos:

    // Step 1: the control vector. Every inference submitted from now on sees it; set it
    // again whenever it changes.
    inference_handler.set_static_input(1, control);

    // Step 2: the audio block in and out of slot 0
    in_block.shape[1] = out_block.shape[1] = static_cast<int64_t>(num_samples);
    size_t delivered = 0;
    inference_handler.process(in_block, 0, out_block, 0, &delivered);
    // audio_output holds delivered samples per channel (zeros for the first latency samples)

    // Step 3: the confidence the latest completed inference produced
    inference_handler.get_static_output(1, confidence);

Key Points for Multi-Tensor Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**One slot space**

- A slot is the tensor's position in the model file's ``inputs`` or ``outputs`` list, from 0,
  State tensors included; every handler entry, the latency vector and the plan report use it.
- The role decides which entries take a slot: a Streamed slot the block calls, a Static slot
  the two Static entries, a State slot none (anira owns it), a Buffer slot the Async contract's.

**Streamed tensors**

- ``process(in, in_slot, out, out_slot, &delivered)`` moves one block in and one block out,
  any block size, one Streamed slot per side; ``push_data`` and ``pop_data`` split the two
  halves for hosts that produce and consume at different rates; the forms over every slot take
  one tensor per slot of a side in one call.
- A block is ``[channels][samples]``, planar over the host's channel pointers or one packed
  block; the dtype is the ring's, float32 unless the contract declares another.

**Static tensors**

- ``set_static_input(slot, tensor)`` and ``get_static_output(slot, tensor)`` take the whole
  tensor in the spec's shape and dtype; "fits" is a check, never a clamp. Both are real-time
  safe and legal from create on, so a value can be set before prepare.
- A Static tensor reports a latency of 0: it is not time-varying.

**State tensors**

- Declared in the model file, run by anira: fed ahead of the inference, captured behind it, kept
  across a plan switch, re-initialised by ``reset()``. Nothing to call.

.. note::
    Every member above is one C entry of ``anira/abi/handler.h`` (``anira_handler_process``,
    ``anira_handler_set_static_input``, ...; :doc:`usage` section 3.2) for a host in C or
    another language. A 2.x handler addressed a non-streamable tensor as channel 0 of a
    ``float***`` block; the :doc:`migration` page maps the 2.x calls onto the 3.x members.

Next Steps
----------

- Check the :doc:`usage` page for more detailed usage instructions
- See the :doc:`examples` page for complete example applications
- Review the :doc:`architecture` to understand anira's design
- Try the :doc:`benchmarking` tools to evaluate your models' performance
