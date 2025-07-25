Examples
========

This section provides examples and demonstrations of how to use anira in various contexts, from simple audio processing to complex real-time plugins.

Overview
--------

The examples in this documentation are organized into several categories to help you get started with anira quickly and understand different integration patterns:

**Quick Start Examples**
    Simple, focused examples that demonstrate core concepts and get you running quickly.

**Built-in Examples** 
    Complete, production-ready examples included with the anira source code that showcase real-world integration patterns.

**External Examples**
    Community and official examples hosted in separate repositories that demonstrate advanced usage patterns.

Quick Start Examples
--------------------

.. toctree::
    :maxdepth: 1

    simple

Built-in Examples
-----------------

anira comes with several built-in examples that demonstrate different use cases and integration patterns. These examples are available when building with ``-DANIRA_WITH_EXAMPLES=ON``.

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

These examples show the minimal code required to perform inference with each backend supported by anira. They do not use the anira library, but show how to use the underlying libraries directly.

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
