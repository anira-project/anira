Examples
========

This section provides examples and demonstrations of how to use anira in various contexts, from simple audio processing to complex real-time plugins.

Built-in Examples
-----------------

anira comes with several built-in examples that demonstrate different use cases and integration patterns. These examples are available when building with ``-DANIRA_WITH_EXAMPLES=ON``.

Model configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``extras/models/``

Every bundled model ships its configuration as files next to its model directory, in the format
section 1.5 of the :doc:`usage` guide describes: a *model file* (``<model>.model.json``: the
model's exports per engine with paths relative to the file, the tensor specs, the state) and a
*contract file* (``<model>.contract.json``: the per-inference budget and the warm-up; the host
geometry is patched in at prepare). ``extras/models/model_files.h`` names them for the examples
and the tests. Covered are the steerable-nafx CNN in three sizes (``cnn/``), GuitarLSTM
(``hybrid-nn/``), the stateful LSTM (``stateful-rnn/``), SimpleGainNetwork in mono and stereo
(``model-pool/``) and RAVE funk drum as the whole model, its encoder and its decoder
(``third-party/ircam-acids/``, LibTorch only).

Every example loads a model the same way, in three lines, and the runtime takes it from there:

.. code-block:: cpp

    anira::ModelConfig model_config = anira::ModelConfig::from_file(k_cnn_model_json);
    anira::ContractHandle contract = anira::ContractHandle::from_file(k_cnn_contract_json);
    anira::InferenceConfig inference_config = anira::v3compat::to_inference_config(
        model_config, contract, anira::v3compat::enabled_engines());

The files set no instance ceiling, so one processor per engine runs; the plugins pass no
machine configuration either, since the library default (half the hardware threads) is what
they want. The one place a model is built in code is the benchmark sweeps: ``CNNConfig.h``,
``HybridNNConfig.h`` and ``StatefulRNNConfig.h`` build the same configurations with the hop,
the batch count or the chunk following the host buffer, which the fixed windows of the files
cannot; a test keeps each builder equal to its file at the default size.

JUCE Audio Plugin
~~~~~~~~~~~~~~~~~

**Location**: ``examples/juce-audio-plugin/``

This example demonstrates how to integrate anira into a JUCE-based VST3 plugin for real-time audio processing. It shows:

- Setting up anira within a JUCE plugin architecture
- Managing real-time constraints in an audio plugin context
- Handling parameter changes and state management
- Building and deploying a VST3 plugin with neural network inference

Key files:

- ``PluginProcessor.h/cpp``: Main plugin processor with anira integration
- ``PluginParameters.h/cpp``: Parameter management
- ``CMakeLists.txt``: Build configuration for JUCE plugin

**Building**:

.. code-block:: bash

    cmake . -B build -DANIRA_WITH_EXAMPLES=ON
    cmake --build build --target anira-juce-plugin-example_VST3

.. note::
    The JUCE plugin example runs one of the bundled models, picked with the ``MODEL_TO_USE``
    cache variable (``-DMODEL_TO_USE=2``; the default is 1): 0 the steerable-nafx CNN, 1 the
    same CNN with every file compiled into the plugin, 2 GuitarLSTM, 3 the stateful LSTM, 4
    and 5 SimpleGainNetwork in mono and stereo, 6 RAVE funk drum, 7 RAVE as an encoder and a
    decoder handler. Every variant is configured by the three lines above; what changes is the
    pair of files and, for the CNN and GuitarLSTM, the custom pre/post processor. Variant 1
    reads nothing from disk: the model file, the contract file and the four exports go into
    JUCE's ``BinaryData``, the model config is loaded from the embedded text
    (``ModelConfig::from_json``) and each entry's source is swapped for the embedded bytes of
    its engine (``set_model_bytes``, ``ANIRA_BYTES_BORROW``), so the description of the model
    stays in the file. Variant 7's decoder anchors on its audio output
    (``rave_funk_drum_decoder.model.json``), so both handlers prepare with the host's block and
    rate. A 2.x configuration file goes through the same loaders (:ref:`migration-json`).

CLAP Plugin Example
~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/clap-audio-plugin/``

Demonstrates anira integration with the CLAP (CLever Audio Plugin) format:

- CLAP plugin architecture with anira
- Real-time audio processing with neural networks
- Modern plugin format implementation

Key files:

- ``anira-clap-demo.h/cpp``: Main CLAP plugin implementation
- ``anira-clap-demo-pluginentry.cpp``: Plugin entry point

**Building**:

.. code-block:: bash

    cmake . -B build -DANIRA_WITH_EXAMPLES=ON
    cmake --build build --target anira-clap-demo

Benchmark Examples
~~~~~~~~~~~~~~~~~~

**Location**: ``examples/benchmark/``

Three different benchmark examples showing various benchmarking scenarios:

Simple Benchmark
^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/simple-benchmark/``

Basic benchmarking setup demonstrating:

- Single configuration benchmarking
- Basic performance measurement
- Simple benchmark fixture usage

CNN Size Benchmark
^^^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/cnn-size-benchmark/``

Benchmarks different CNN model sizes to evaluate:

- Performance scaling with model complexity
- Memory usage patterns
- Optimal model size selection for real-time constraints

Advanced Benchmark
^^^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/advanced-benchmark/``

Comprehensive benchmarking suite featuring:

- Multiple configuration testing
- Parameterized benchmarks
- Statistical analysis
- Performance comparison across backends

Minimal Inference Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/minimal-inference/``

These examples show the minimal code required to perform inference with each backend supported by anira. They read the model path and the tensor shapes from anira's model files (through the bridge to the 2.x :cpp:struct:`anira::InferenceConfig`) and then drive the engine's own API directly, without the anira runtime. The ExecuTorch example is the exception: it does not link anira (anira embeds its own copy of the ExecuTorch runtime, which a second copy must stay isolated from), so it spells the path and the shapes out.

External Examples
-----------------

Neural Network Inference Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Repository**: `nn-inference-template <https://github.com/Torsion-Audio/nn-inference-template>`_

A more comprehensive JUCE/VST3 plugin template that uses anira for real-time safe neural network inference. This plugin is more complex than the simple JUCE Audio Plugin example and features:

- Professional GUI implementation
- Advanced parameter management
- State saving and loading
- Real-world plugin architecture patterns
- Production-ready code structure

This template serves as an excellent starting point for developing commercial audio plugins with neural network processing.
