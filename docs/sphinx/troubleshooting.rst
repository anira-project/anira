Troubleshooting & FAQ
=====================

This section addresses common issues and questions that may arise when using anira.

Frequently Asked Questions
--------------------------

General
~~~~~~~

What is anira?
^^^^^^^^^^^^^^

Anira is a high-performance library designed for real-time neural network inference in audio applications. It provides a consistent API across multiple inference backends with a focus on deterministic performance suitable for audio processing.

Which platforms are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Anira supports macOS, Linux, and Windows platforms. It has been tested on x86_64, ARM64, and ARM7 architectures.

Which neural network frameworks are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Anira currently supports three inference backends:
    - LibTorch
    - ONNX Runtime
    - TensorFlow Lite

.. note::
    Custom backends can be integrated as needed.

Is anira free and open source?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, anira is open source and available under the Apache-2.0 license.

Technical Questions
~~~~~~~~~~~~~~~~~~~

How does anira ensure real-time safety?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Anira ensures real-time safety through several mechanisms:
    - No dynamic memory allocation during audio processing
    - Static thread pool to avoid oversubscription
    - Lock-free communication between audio and inference threads
    - Pre-allocation of all required resources
    - Consistent timing checks and fallback mechanisms

What's the minimum latency I can achieve?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The minimum achievable latency depends on several factors, including model complexity, hardware performance, and audio buffer size. Anira is optimized for low-latency operation and, in ideal conditions, can return inference results within the same audio processing cycle—effectively achieving zero added latency.

Can I use multiple models simultaneously?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, you can use multiple models simultaneously by creating separate :cpp:class:`anira::InferenceHandler` instances, each with its own model configuration. All handlers can share the same thread pool, enabling efficient parallel processing of multiple models.

Troubleshooting
---------------

Compilation Issues
~~~~~~~~~~~~~~~~~~

Missing Backend Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: CMake fails to find LibTorch, ONNX Runtime, or TensorFlow Lite.

**Solution**: You can disable specific backends using CMake options:
    - `-DANIRA_WITH_LIBTORCH=OFF`
    - `-DANIRA_WITH_ONNXRUNTIME=OFF`
    - `-DANIRA_WITH_TFLITE=OFF`

Alternatively, you can specify custom paths to these dependencies if they are installed in non-standard locations.

Compilation Errors with C++ Standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Compiler errors related to C++ standard compatibility.

**Solution**: Anira requires C++17 or later. Ensure your compiler supports C++17.

Runtime Issues
~~~~~~~~~~~~~~

Audio Glitches or Dropouts
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Audio processing experiences dropouts or glitches during inference.

**Solutions**:
    1. Increase the maximum inference time in your :cpp:struct:`anira::InferenceConfig` to allow more time for model processing.
    2. Reduce the complexity of your neural network model
    3. Increase audio buffer size (though this increases latency)
    4. Check if other processes are consuming CPU resources
    5. Use `anira::benchmark` tools to identify performance bottlenecks

Model Loading Failures
^^^^^^^^^^^^^^^^^^^^^^

**Issue**: "Failed to load model" or similar errors.

**Solutions**:
    1. Verify the model file exists at the specified path
    2. Check that the model format is compatible with the selected backend
    3. Ensure tensor shapes in your :cpp:struct:`anira::InferenceConfig` match the model's expected shapes
    4. Try a different backend if available

Thread Priority Issues
^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Thread priority settings fail, particularly on Linux.

**Solution**: On Linux, you may need to set the `rtprio` limit for your user. Add the following to `/etc/security/limits.conf`:

.. code-block:: conf

    your_username - rtprio 99

Log out and back in for the changes to take effect.

Unexpected Results or Crashes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Inference produces incorrect outputs or crashes.

**Solutions**:
    1. Validate tensor shapes in your :cpp:struct:`anira::InferenceConfig` match your model's expectations
    2. Ensure your pre/post-processing logic correctly handles the data format
    3. Try using a different backend to rule out backend-specific issues
    4. Check that your model works correctly outside of anira use the minimal inference example provided in the :doc:`examples` section.

.. note::
    If you continue to experience issues feel free to file an issue on the [GitHub repository](https://github.com/anira-project/anira/issues).

