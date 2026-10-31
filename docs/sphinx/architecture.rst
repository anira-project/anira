Architecture
============

This page describes the high-level architecture of the anira library, its core components, and how they interact with each other.

System Overview
---------------

anira is designed with real-time audio applications in mind, focusing on deterministic performance and thread safety. The architecture consists of several key components working together to provide neural network inference capabilities that can be safely used within audio processing contexts.

.. code-block:: text

     +-----------------------------------+       +----------------------+
     |         InferenceHandler          | ----> |       Context        |
     |   (Main user-facing interface)    |       | (Thread pool, logs)  |
     +----------------+------------------+       +----------------------+
                      |
                      v
     +----------------+------------------+
     |     Pipeline: inference stage     |
     |  (ModelConfig, candidates) and a  |
     |  Processor (custom pre-/post-)    |
     +----------------+------------------+
                      |
                      v
     +----------------+------------------+
     |         InferenceManager          |
     |  (Rings, chunks, scheduling)      |
     +----------------+------------------+
                      |
                      v
     +----------------+------------------+       +----------------------+
     |  Engine Adapters / custom Engine  | <---> |   Inference Engines  |
     | (LibTorch, ONNX, TensorFlow Lite) |       | (External libraries) |
     +-----------------------------------+       +----------------------+

Every box is reachable from C11 through ``anira/abi/`` (``anira_handler``, ``anira_context``, ``anira_pipeline``, ``anira_processor_desc``, ``anira_engine_desc``); ``anira/anira.hpp`` is the header-only C++20 face over those entries, and the Hard contract (``anira::ContractHandle``) tells the handler how the stream runs.

Key Design Principles
---------------------

1. **Real-time Safety**

    * No dynamic memory allocation during audio processing
    * Deterministic performance guarantees
    * Thread-safety with atomic operations
    * Pre-allocated buffers and resources

2. **Flexibility**

    * Support for multiple inference engines and providers
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

:cpp:class:`anira::InferenceHandler`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary interface for users, handling the overall integration of neural network inference into audio processing workflows (``anira_handler`` of ``anira/abi/handler.h``).

* Manages the audio processing lifecycle: created on a context from a pipeline, prepared with the contract
* Provides the real-time safe Hard entries (``process``, ``push_data``, ``pop_data``) over host tensors
* Selects among the plans of its report
* Reports latency information

:cpp:class:`anira::Context`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The view of the process-wide core every handler runs on (``anira_context`` of ``anira/abi/context.h``).

* The inference thread pool and the wait strategy of its threads
* Logging: the level, the drain of the real-time log queue
* What this machine can run: the backends and providers of its capabilities

:cpp:class:`anira::Pipeline`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What a handler runs: the inference stage, the model configuration (:cpp:class:`anira::ModelConfig`: the model entries, the tensor specs, the state) with its candidate backends and custom engines, and at most one custom processor (:cpp:class:`anira::Processor`).

* Model entries and the engine or provider each runs on
* Input and output tensor specs: roles, axes, windows, latencies
* The processor's phases convert host blocks to model tensors and back where anira's default fill does not fit (:doc:`custom_preprocessing`)

:cpp:class:`anira::InferenceManager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinates the rings and the inference scheduling of one handler (internal).

* Moves the host's blocks into the rings and the chunks out of them
* Schedules inference tasks on the context's thread pool
* Handles synchronization between audio and inference threads

Engine Adapters
~~~~~~~~~~~~~~~

Engine-specific adapters.

* The LibTorch adapter (``src/engines/LibTorchAdapter.cpp``, internal) - PyTorch C++ API integration
* The ONNX Runtime adapter (``src/engines/OnnxRuntimeAdapter.cpp``, internal) - ONNX Runtime integration
* The TensorFlow Lite and LiteRT adapters (``src/engines/TFLiteAdapter.cpp``, ``LiteRtAdapter.cpp``, internal) - TensorFlow Lite and LiteRT integration
* The ExecuTorch adapter (``src/engines/ExecuTorchAdapter.cpp``, internal) - ExecuTorch integration
* :cpp:class:`anira::Engine` (``anira_engine_desc`` of ``anira/abi/engine.h`` in C) - a custom engine registered on the pipeline under its id, run through the same adapter interface as the five above (:doc:`custom_engines`)

Data Flow
---------

1. **Audio Input:** The host application provides audio data to the InferenceHandler as host tensors
2. **Pre-processing:** The processor (or anira's default fill) moves the host blocks into model tensors
3. **Scheduling:** The InferenceManager schedules the inference task
4. **Inference:** The engine's adapter runs the model
5. **Post-processing:** The processor (or anira's default) moves the results back into the output rings
6. **Audio Output:** The processed audio is returned to the host application

Threading Model
---------------

anira employs a multi-threaded architecture with careful synchronization:

* **Audio Thread:** Real-time thread from the host application, never blocked
* **Inference Threads:** Worker threads performing the actual model inference
* **Synchronization:** Lock-free communication with atomic operations and ring buffers

The system avoids blocking operations in the audio thread and uses a carefully designed thread pool to prevent CPU oversubscription.

How idle inference threads wait for new work is configurable on the context (``context.threads(n, wait)``, ``anira_context_config_set_threads``): either an exponential-backoff polling loop (``ANIRA_WAIT_SPIN_BACKOFF``, the default) or a blocking wait on the shared inference queue's semaphore (``ANIRA_WAIT_BLOCKING``), where the enqueue from the audio thread wakes a sleeping worker directly with a single bounded, non-blocking semaphore signal. See the :doc:`usage` chapter for the trade-offs.
