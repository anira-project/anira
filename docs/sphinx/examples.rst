Examples
========

This section provides examples and demonstrations of how to use anira in various contexts, from simple audio processing to complex real-time plugins.

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

These examples show the minimal code required to get inference running with each backend:

LibTorch Example
^^^^^^^^^^^^^^^^

**Location**: ``examples/minimal-inference/libtorch/``

Minimal setup for LibTorch backend:

.. code-block:: cpp

   #include <anira/anira.h>

   int main() {
       // Model configuration
       anira::InferenceConfig config({
           {"model.pt", anira::LIBTORCH}
       }, {
           {{{1, 1, 2048}}, {{1, 1, 2048}}}
       }, 10.0f);

       // Create processor and handler
       anira::PrePostProcessor processor(config);
       anira::InferenceHandler handler(processor, config);

       // Prepare for processing
       anira::HostAudioConfig host_config(512, 48000.0);
       handler.prepare(host_config);
       handler.set_inference_backend(anira::LIBTORCH);

       // Process audio
       float* audio_data[2] = {/* your audio data */};
       handler.process(audio_data, 512);

       return 0;
   }

ONNX Runtime Example
^^^^^^^^^^^^^^^^^^^^

**Location**: ``examples/minimal-inference/onnxruntime/``

Minimal setup for ONNX Runtime backend:

.. code-block:: cpp

   #include <anira/anira.h>

   int main() {
       // Model configuration
       anira::InferenceConfig config({
           {"model.onnx", anira::ONNX}
       }, {
           {{{1, 1, 2048}}, {{1, 1, 2048}}}
       }, 8.0f);

       // Create processor and handler
       anira::PrePostProcessor processor(config);
       anira::InferenceHandler handler(processor, config);

       // Prepare for processing
       anira::HostAudioConfig host_config(256, 44100.0);
       handler.prepare(host_config);
       handler.set_inference_backend(anira::ONNX);

       // Process audio
       float* audio_data[1] = {/* your audio data */};
       handler.process(audio_data, 256);

       return 0;
   }

TensorFlow Lite Example
^^^^^^^^^^^^^^^^^^^^^^^

**Location**: ``examples/minimal-inference/tensorflow-lite/``

Minimal setup for TensorFlow Lite backend:

.. code-block:: cpp

   #include <anira/anira.h>

   int main() {
       // Model configuration
       anira::InferenceConfig config({
           {"model.tflite", anira::TFLITE}
       }, {
           {{{1, 2048, 1}}, {{1, 2048, 1}}}
       }, 5.0f);

       // Create processor and handler
       anira::PrePostProcessor processor(config);
       anira::InferenceHandler handler(processor, config);

       // Prepare for processing
       anira::HostAudioConfig host_config(1024, 96000.0);
       handler.prepare(host_config);
       handler.set_inference_backend(anira::TFLITE);

       // Process audio
       float* audio_data[2] = {/* your audio data */};
       handler.process(audio_data, 1024);

       return 0;
   }

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

Usage Patterns and Best Practices
----------------------------------

Common Integration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Audio Effect Plugin**

   Use anira to create audio effects that apply neural network processing to incoming audio:

   .. code-block:: cpp

      class AudioEffectProcessor {
          anira::InferenceHandler m_inference_handler;
          
      public:
          void processBlock(AudioBuffer<float>& buffer) {
              float** channelData = buffer.getArrayOfWritePointers();
              m_inference_handler.process(channelData, buffer.getNumSamples());
          }
      };

2. **Real-time Analysis**

   Use anira for real-time audio analysis with neural networks:

   .. code-block:: cpp

      class AudioAnalyzer {
          anira::InferenceHandler m_analyzer;
          
      public:
          void analyzeAudio(const float** input, int numSamples) {
              // Push audio for analysis
              m_analyzer.push_data(input, numSamples);
              
              // Get analysis results
              float analysisResults[NUM_FEATURES];
              m_analyzer.pop_data(&analysisResults, NUM_FEATURES);
          }
      };

3. **Multi-Model Processing**

   Use multiple anira instances for different processing stages:

   .. code-block:: cpp

      class MultiStageProcessor {
          anira::InferenceHandler m_preprocessor;
          anira::InferenceHandler m_mainProcessor;
          anira::InferenceHandler m_postprocessor;
          
      public:
          void processAudio(float** audio, int numSamples) {
              m_preprocessor.process(audio, numSamples);
              m_mainProcessor.process(audio, numSamples);
              m_postprocessor.process(audio, numSamples);
          }
      };

Performance Optimization Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Buffer Size Selection**

   Choose buffer sizes that balance latency and computational efficiency:

   .. code-block:: cpp

      // For low-latency applications
      anira::HostAudioConfig lowLatency(64, 48000.0);
      
      // For high-throughput applications
      anira::HostAudioConfig highThroughput(1024, 48000.0);

2. **Thread Pool Configuration**

   Optimize thread pool size based on your system:

   .. code-block:: cpp

      // Conservative approach (half of available cores)
      anira::ContextConfig config(std::thread::hardware_concurrency() / 2);
      
      // Aggressive approach (more threads for I/O bound models)
      anira::ContextConfig config(std::thread::hardware_concurrency());

3. **Memory Pre-allocation**

   Always call ``prepare()`` during initialization, not during real-time processing:

   .. code-block:: cpp

      void initialize() {
          // Good: allocate during initialization
          m_inference_handler.prepare(host_config);
      }
      
      void processAudio(float** audio, int samples) {
          // Good: no allocation during real-time processing
          m_inference_handler.process(audio, samples);
      }

Troubleshooting Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Audio Dropouts**

   If experiencing audio dropouts, check:
   
   - Maximum inference time setting
   - Buffer size configuration
   - System CPU load
   - Thread priority settings

2. **Latency Issues**

   To minimize latency:
   
   - Use smaller buffer sizes
   - Optimize model inference time
   - Consider the ``blocking_ratio`` parameter carefully

3. **Memory Issues**

   For memory optimization:
   
   - Pre-allocate all buffers during initialization
   - Use appropriate tensor shapes
   - Monitor memory usage in benchmarks

These examples provide a solid foundation for integrating anira into your audio processing applications, whether you're building simple effects or complex multi-stage processing systems.
