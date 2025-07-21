anira Documentation
===================

.. image:: ../img/anira-logo.png
   :alt: anira Logo
   :align: center
   :width: 800px

|build_test| |build_benchmark| |build_examples| |on_tag|

.. |build_test| image:: https://github.com/anira-project/anira/actions/workflows/build_test.yml/badge.svg
   :target: https://github.com/anira-project/anira/actions/workflows/build_test.yml

.. |build_benchmark| image:: https://github.com/anira-project/anira/actions/workflows/build_benchmark.yml/badge.svg
   :target: https://github.com/anira-project/anira/actions/workflows/build_benchmark.yml

.. |build_examples| image:: https://github.com/anira-project/anira/actions/workflows/build_examples.yml/badge.svg
   :target: https://github.com/anira-project/anira/actions/workflows/build_examples.yml

.. |build_docs| image:: https://github.com/anira-project/anira/actions/workflows/build_docs_and_deploy.yml/badge.svg
   :target: https://github.com/anira-project/anira/actions/workflows/build_docs_and_deploy.yml

.. |on_tag| image:: https://github.com/anira-project/anira/actions/workflows/on_tag.yml/badge.svg
   :target: https://github.com/anira-project/anira/actions/workflows/on_tag.yml

**anira** is a high-performance library designed to enable easy real-time safe integration of neural network inference within audio applications. Compatible with multiple inference backends, `LibTorch <https://github.com/pytorch/pytorch/>`_, `ONNXRuntime <https://github.com/microsoft/onnxruntime/>`_, and `Tensorflow Lite <https://github.com/tensorflow/tensorflow/>`_, anira bridges the gap between advanced neural network architectures and real-time audio processing. In the `paper <https://doi.org/10.1109/IS262782.2024.10704099>`_ you can find more information about the architecture and the design decisions of **anira**, as well as extensive performance evaluations with the built-in benchmarking capabilities.

Features
========

- **Real-time Safe Execution**: Ensures deterministic runtimes suitable for real-time audio applications
- **Thread Pool Management**: Utilizes a static thread pool to avoid oversubscription and enables efficient parallel inference
- **Built-in Benchmarking**: Includes tools for evaluating the real-time performance of neural networks
- **Comprehensive Inference Engine Support**: Integrates common inference engines, LibTorch, ONNXRuntime, and TensorFlow Lite
- **Flexible Neural Network Integration**: Supports a variety of neural network models, including stateful and stateless models
- **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows

Quick Start
===========

The basic usage of anira is as follows:

.. code-block:: cpp

   #include <anira/anira.h>

   anira::InferenceConfig inference_config(
           {{"path/to/your/model.onnx", anira::InferenceBackend::ONNX}}, // Model path
           {{{256, 1, 150}}, {{256, 1}}},  // Input, Output shape
           5.33f // Maximum inference time in ms
   );

   // Create a pre- and post-processor instance
   anira::PrePostProcessor pp_processor;

   // Create an InferenceHandler instance
   anira::InferenceHandler inference_handler(pp_processor, inference_config);

   // Pass the host audio configuration and allocate memory for audio processing
   inference_handler.prepare({buffer_size, sample_rate});

   // Select the inference backend
   inference_handler.set_inference_backend(anira::ONNX);

   // Optionally get the latency of the inference process in samples
   unsigned int latency_in_samples = inference_handler.get_latency();

   // Real-time safe audio processing in process callback of your application
   process(float** audio_data, int num_samples) {
       inference_handler.process(audio_data, num_samples);
   }
   // audio_data now contains the processed audio samples

Installation
============

CMake Integration
-----------------

anira can be easily integrated into your CMake project. Either add anira as a submodule or download the pre-built binaries from the `releases page <https://github.com/anira-project/anira/releases/latest>`_.

Add as a git submodule
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Add anira repo as a submodule
   git submodule add https://github.com/anira-project/anira.git modules/anira

In your CMakeLists.txt, add anira as a subdirectory and link your target to the anira library:

.. code-block:: cmake

   # Setup your project and target
   project(your_project)
   add_executable(your_target main.cpp ...)

   # Add anira as a subdirectory
   add_subdirectory(modules/anira)

   #Link your target to the anira library
   target_link_libraries(your_target anira::anira)

With pre-built binaries
~~~~~~~~~~~~~~~~~~~~~~~

Download the pre-built binaries from your operating system and architecture from the `releases page <https://github.com/anira-project/anira/releases/latest>`_.

.. code-block:: cmake

   # Setup your project and target
   project(your_project)
   add_executable(your_target main.cpp ...)

   # Add the path to the anira library as cmake prefix path and find the package
   list(APPEND CMAKE_PREFIX_PATH "path/to/anira")
   find_package(anira REQUIRED)

   # Link your target to the anira library
   target_link_libraries(your_target anira::anira)

Build from source
~~~~~~~~~~~~~~~~~

You can also build anira from source using CMake. All dependencies are automatically installed during the build process.

.. code-block:: bash

   git clone https://github.com/anira-project/anira
   cmake . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release --target anira

Build options
~~~~~~~~~~~~~

By default, all three inference engines are installed. You can disable specific backends as needed:

- LibTorch: ``-DANIRA_WITH_LIBTORCH=OFF``
- OnnxRuntime: ``-DANIRA_WITH_ONNXRUNTIME=OFF``
- Tensrflow Lite: ``-DANIRA_WITH_TFLITE=OFF``

Moreover, the following options are available:

- Build anira with benchmark capabilities: ``-DANIRA_WITH_BENCHMARK=ON``
- Build example applications, plugins and populate example neural models: ``-DANIRA_WITH_EXAMPLES=ON``
- Build anira with tests: ``-DANIRA_WITH_TESTS=ON``
- Build anira with documentation: ``-DANIRA_WITH_DOCS=ON``

Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   getting_started
   usage
   architecture
   benchmarking
   examples
   troubleshooting
   contributing

.. toctree::
   :maxdepth: 1
   :caption: API Documentation:
   :glob:

   api/*

Citation
========

If you use anira in your research or project, please cite either the `paper <https://doi.org/10.1109/IS262782.2024.10704099>`_ or the software itself:

.. code-block:: bibtex

   @inproceedings{ackvaschulz2024anira,
       author={Ackva, Valentin and Schulz, Fares},
       booktitle={2024 IEEE 5th International Symposium on the Internet of Sounds (IS2)},
       title={ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications},
       year={2024},
       volume={},
       number={},
       pages={1-10},
       publisher={IEEE},
       doi={10.1109/IS262782.2024.10704099}
   }

   @software{ackvaschulz2024anira,
       author = {Valentin Ackva and Fares Schulz},
       title = {anira: an architecture for neural network inference in real-time audio application},
       url = {https://github.com/anira-project/anira},
       version = {x.x.x},
       year = {2024},
   }

License
=======

This project is licensed under `Apache-2.0 <https://github.com/anira-project/anira/blob/main/LICENSE>`_.

