Architecture
============

This page describes the high-level architecture of the anira library, its core components, and how they interact with each other.

System Overview
---------------

anira is designed with real-time audio applications in mind, focusing on deterministic performance and thread safety. The architecture consists of several key components working together to provide neural network inference capabilities that can be safely used within audio processing contexts.

.. code-block:: text

    +-----------------------------------+
    |         InferenceHandler          |
    |   (Main user-facing interface)    |
    +----------------+------------------+
                     |
                     v
    +----------------+------------------+
    |        InferenceConfig           |
    | (Model paths, shapes, settings)   |
    +----------------+------------------+
                     |
                     v
    +----------------+------------------+
    |      PrePostProcessor            |
    |  (Format conversion, buffering)   |
    +----------------+------------------+
                     |
                     v
    +----------------+------------------+
    |         InferenceManager         |
    |    (Thread pool coordination)    |
    +----------------+------------------+
                     |
                     v
    +----------------+------------------+       +----------------------+
    |         Backend Processors       | <---> |   Inference Engines  |
    | (LibTorch, ONNX, TensorFlow Lite)|       | (External libraries) |
    +-----------------------------------+       +----------------------+

Key Design Principles
---------------------

1. **Real-time Safety**
   
   * No dynamic memory allocation during audio processing
   * Deterministic performance guarantees
   * Thread-safety with atomic operations
   * Pre-allocated buffers and resources

2. **Flexibility**
   
   * Support for multiple inference backends
   * Configurable thread management
   * Customizable pre/post-processing
   * Support for both stateful and stateless models

3. **Performance**
   
   * Efficient buffer management with zero-copy where possible
   * Thread pool to prevent oversubscription
   * Optimized tensor operations
   * Benchmarking tools for performance analysis

Component Responsibilities
--------------------------

InferenceHandler
~~~~~~~~~~~~~~~~

The primary interface for users, handling the overall integration of neural network inference into audio processing workflows.

* Manages the audio processing lifecycle
* Provides real-time safe process methods
* Handles buffer management
* Reports latency information

InferenceConfig
~~~~~~~~~~~~~~~

Stores configuration data for models and processing parameters.

* Model paths and backend selection
* Input and output tensor shapes
* Maximum inference time limits
* Memory management settings

PrePostProcessor
~~~~~~~~~~~~~~~~

Handles data formatting between audio buffers and neural network tensors.

* Converts audio data to model input format
* Converts model outputs back to audio format
* Manages intermediate buffers

InferenceManager
~~~~~~~~~~~~~~~~

Coordinates the thread pool and inference scheduling.

* Manages worker threads
* Schedules inference tasks
* Handles synchronization between audio and inference threads

Backend Processors
~~~~~~~~~~~~~~~~~~

Backend-specific implementations for different inference engines.

* LibTorchProcessor - PyTorch C++ API integration
* OnnxRuntimeProcessor - ONNX Runtime integration
* TFLiteProcessor - TensorFlow Lite integration

Data Flow
---------

1. **Audio Input:** The host application provides audio data to the InferenceHandler
2. **Pre-processing:** The PrePostProcessor converts audio data to tensors
3. **Scheduling:** The InferenceManager schedules the inference task
4. **Inference:** A backend processor executes the neural network model
5. **Post-processing:** The PrePostProcessor converts results back to audio
6. **Audio Output:** The processed audio is returned to the host application

Threading Model
---------------

anira employs a multi-threaded architecture with careful synchronization:

* **Audio Thread:** Real-time thread from the host application, never blocked
* **Inference Threads:** Worker threads performing the actual model inference
* **Synchronization:** Lock-free communication with atomic operations and ring buffers

The system avoids blocking operations in the audio thread and uses a carefully designed thread pool to prevent CPU oversubscription.
