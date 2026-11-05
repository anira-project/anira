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
blocks are patched in at prepare). ``extras/models/model_files.h`` names them for the examples
and the tests. Covered are the steerable-nafx CNN in three sizes (``cnn/``), GuitarLSTM
(``hybrid-nn/``), the stateful LSTM (``stateful-rnn/``), SimpleGainNetwork in mono and stereo,
the stereo StatefulAccumulatorNetwork with its declared State pair, TypedGainNetwork, the gain
model's tensors at ``int16``, ``int8`` and ``float16`` on every engine
(``typed_gain_<dtype>.model.json``; the ``_unplanned`` files run an ExecuTorch export whose
inputs and outputs the memory plan leaves to the caller, so the engine binds them in place), and,
for the tests, an ONNX model per dtype anira names and a TFLite model with quantised int8 I/O
(``model-pool/``), and RAVE funk drum as the whole model, its encoder and its decoder
(``third-party/ircam-acids/``, LibTorch only).

Every example loads a model the same way, in a few lines, and the runtime takes it from there:

.. code-block:: cpp

    anira::ModelConfig model_config = anira::ModelConfig::from_file(k_cnn_model_json);
    anira::ContractHandle contract = anira::ContractHandle::from_file(k_cnn_contract_json);
    anira::Pipeline pipeline;
    pipeline.add_inference(model_config);
    anira::InferenceHandler handler(context, pipeline);

The files set no instance ceiling, so the instance count is the inference pool's
(``ANIRA_INSTANCES_AUTO``); the plugins pass no context configuration either, since the library
default (half the hardware threads) is what they want. Every plugin also carries the bypass
engine of ``extras/models/BypassEngine.h`` (``org.anira.examples.bypass``, a custom engine that
needs no model file and delays the input by the latency anira reports) behind its "BYPASS"
backend choice, and GuitarLSTM runs with ``HybridNNProcessor`` (``extras/models/hybrid-nn/``), the
processor that builds the model's overlapping windows; the CNN and the other models take anira's
default fill. The one place a model is built in code is the benchmark sweeps: ``CNNConfig.h``,
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
    decoder handler. Every variant is configured by the lines above; what changes is the
    pair of files and, for GuitarLSTM, the ``HybridNNProcessor`` of the pipeline. The "Backend Type"
    parameter stores its choice in an atomic, and ``processBlock`` applies it: the plan of the
    ``PlanReport`` that ``prepare`` returned that runs the chosen engine
    (``PlanReport::find``, then ``InferenceHandler::set_plan``; nothing changes when the build or
    the model has no such plan). The gain of variants 4 and 5 is the model's Static input
    (``set_static_input``). Variant 1
    reads nothing from disk: the model file, the contract file and the four exports go into
    JUCE's ``BinaryData``, the model config is loaded from the embedded text
    (``ModelConfig::from_json``) and each entry's source is swapped for the embedded file of the
    same name as its path (``set_model_bytes``, ``ANIRA_BYTES_BORROW``), one loop over the entries,
    so the description of the model stays in the file. Variant 7's decoder anchors on its audio output
    (``rave_funk_drum_decoder.model.json``), so both handlers prepare with the host's block range and
    rate; the latent frames pass from the encoder's ``pop_data`` to the decoder's ``push_data``
    as one planar tensor. A 2.x configuration file goes through the same loaders (:ref:`migration-json`).

Both plugins declare the host's real block range on the Hard contract, since neither host promises
a fixed block. JUCE's ``prepareToPlay(sampleRate, samplesPerBlock)`` names the largest block only,
so the JUCE plugin prepares with ``hard_host_blocks(1, samplesPerBlock, sampleRate)``; CLAP's
``activate(sampleRate, minFrameCount, maxFrameCount)`` names both ends, so the CLAP plugin prepares
with ``hard_host_blocks(max(1, minFrameCount), maxFrameCount, sampleRate)``. Every
``processBlock`` / ``process`` call carries its own block. The price is latency: anira sizes it for
every block size of the range (at 48 kHz and a 512-sample maximum the CNN reports 4602 samples
instead of 3584, SimpleGainNetwork 1024 instead of 512, GuitarLSTM 767 instead of 512), which the
plugins report to the host (``setLatencySamples``, the CLAP latency extension) for compensation. A
fixed contract under a host that changes its block size misses blocks instead. The figure covers
every block size of the range held from call to call; a host that changes the size between calls
can still miss a block, rarely: a full block right after a short one that completed a hop. At
48 kHz, 480 calls of 1 to 512 samples paced in real time missed 1 block on SimpleGainNetwork and
2 on GuitarLSTM under the range, 314 and 159 under the fixed contract
(`issue #228 <https://github.com/anira-project/anira/issues/228>`_).

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

Three different benchmark examples showing various benchmarking scenarios, on the Google
Benchmark fixture of ``examples/benchmark/fixture/`` (``ProcessBlockFixture``, the static target
``anira_benchmark_fixture``, not part of the library; :doc:`benchmarking`). They build with
``-DANIRA_WITH_BENCHMARK=ON`` and run as CTest tests (``ctest -R '^Benchmark' -j 1``):

Simple Benchmark
^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/simple-benchmark/``

Basic benchmarking setup demonstrating:

- Single configuration benchmarking (GuitarLSTM with ``HybridNNProcessor`` on ONNX Runtime)
- Basic performance measurement
- Simple benchmark fixture usage

CNN Size Benchmark
^^^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/cnn-size-benchmark/``

Benchmarks different CNN model sizes to evaluate:

- Performance scaling with model complexity
- Memory usage patterns
- Optimal model size selection for real-time constraints (every engine of the build, and the
  bypass engine as the baseline)

Advanced Benchmark
^^^^^^^^^^^^^^^^^^

**Location**: ``examples/benchmark/advanced-benchmark/``

Comprehensive benchmarking suite featuring:

- Multiple configuration testing
- Parameterized benchmarks
- Statistical analysis
- Performance comparison across engines, with the clear engine of ``ClearEngine.h``
  (``org.anira.examples.clear``, zeros without a model) as the baseline of anira's own path

Minimal Inference Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``examples/minimal-inference/``

These examples show the minimal code required to perform inference with each engine supported by anira. They read the model path and the tensor shapes from anira's model files (``minimal_model.h``, over the read-back getters of :cpp:class:`anira::ModelConfig`) and then drive the engine's own API directly, without the anira runtime. The ExecuTorch example is the exception: it does not link anira (anira embeds its own copy of the ExecuTorch runtime, which a second copy must stay isolated from), so it spells the path and the shapes out.

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
