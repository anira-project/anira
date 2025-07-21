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
  - PyTorch model files (.pt)
  - TensorFlow Lite model files (.tflite)

Installation
------------

There are several ways to integrate anira into your project:

Option 1: Add as Git Submodule (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Add anira repo as a submodule
   git submodule add https://github.com/anira-project/anira.git modules/anira

In your CMakeLists.txt:

.. code-block:: cmake

   # Setup your project and target
   project(your_project)
   add_executable(your_target main.cpp ...)

   # Add anira as a subdirectory
   add_subdirectory(modules/anira)

   # Link your target to the anira library
   target_link_libraries(your_target anira::anira)

Option 2: Use Pre-built Binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download pre-built binaries from the `releases page <https://github.com/anira-project/anira/releases/latest>`_.

In your CMakeLists.txt:

.. code-block:: cmake

   # Setup your project and target
   project(your_project)
   add_executable(your_target main.cpp ...)

   # Add the path to the anira library as cmake prefix path and find the package
   list(APPEND CMAKE_PREFIX_PATH "path/to/anira")
   find_package(anira REQUIRED)

   # Link your target to the anira library
   target_link_libraries(your_target anira::anira)

Option 3: Build from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/anira-project/anira.git
   cd anira
   cmake . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release --target anira
   cmake --install build --prefix /path/to/install/directory

Basic Usage Example
-------------------

Here's a minimal example to get you started with anira:

.. code-block:: cpp

   #include <anira/anira.h>

   // Step 1: Create an inference configuration
   anira::InferenceConfig config(
       {{"path/to/model.onnx", anira::InferenceBackend::ONNX}},  // Model path and backend
       {{{256, 1, 128}}, {{256, 1}}},                            // Input and output tensor shapes
       5.0f                                                      // Maximum inference time (ms)
   );

   // Step 2: Create a pre/post processor
   anira::PrePostProcessor processor;

   // Step 3: Create the inference handler
   anira::InferenceHandler handler(processor, config);

   // Step 4: Prepare for audio processing
   anira::HostAudioConfig hostConfig(512, 48000);  // buffer size, sample rate
   handler.prepare(hostConfig);
   handler.set_inference_backend(anira::InferenceBackend::ONNX);

   // Step 5: Process audio in your audio callback
   void processAudio(float** audioBuffer, int numSamples) {
       // Process audio in-place (input and output use the same buffer)
       handler.process(audioBuffer, numSamples);
   }

Handling Multiple Channels
--------------------------

For multi-channel audio processing:

.. code-block:: cpp

   // Process audio with separate input and output buffers
   void processMultiChannel(float** inputBuffer, float** outputBuffer, int numChannels, int numSamples) {
       for (int ch = 0; ch < numChannels; ch++) {
           handler.process(inputBuffer[ch], outputBuffer[ch], numSamples);
       }
   }

Using Different Backends
------------------------

anira supports multiple backends that can be selected at runtime:

.. code-block:: cpp

   // Check if a backend is available
   if (handler.is_backend_available(anira::InferenceBackend::LIBTORCH)) {
       handler.set_inference_backend(anira::InferenceBackend::LIBTORCH);
   } else if (handler.is_backend_available(anira::InferenceBackend::ONNX)) {
       handler.set_inference_backend(anira::InferenceBackend::ONNX);
   } else if (handler.is_backend_available(anira::InferenceBackend::TFLITE)) {
       handler.set_inference_backend(anira::InferenceBackend::TFLITE);
   }

Loading Models from Memory
--------------------------

For applications that need to embed models or load them from unconventional sources:

.. code-block:: cpp

   // Load model data from memory (binary data)
   std::vector<uint8_t> modelData = loadModelFromMemory();
   
   anira::InferenceConfig config(
       {{modelData, anira::InferenceBackend::ONNX}},  // Model data and backend
       {{{256, 1, 128}}, {{256, 1}}},                 // Input and output tensor shapes
       5.0f                                           // Maximum inference time (ms)
   );

Next Steps
----------

- Check the :doc:`usage` page for more detailed usage instructions
- See the :doc:`examples` page for complete example applications
- Review the :doc:`architecture` to understand anira's design
- Try the :doc:`benchmarking` tools to evaluate your models' performance
