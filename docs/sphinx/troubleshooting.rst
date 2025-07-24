Troubleshooting & FAQ
=====================

This section addresses common issues and questions that may arise when using anira.

Frequently Asked Questions
--------------------------

General
~~~~~~~

What is anira?
^^^^^^^^^^^^^^
anira is a high-performance library designed for real-time neural network inference in audio applications. It provides a consistent API across multiple inference backends with a focus on deterministic performance suitable for audio processing.

Which platforms are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
anira supports macOS, Linux, and Windows platforms. It has been tested on x86_64, ARM64, and ARM7 architectures.

Which neural network frameworks are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
anira currently supports three inference backends:
- PyTorch (LibTorch)
- ONNX Runtime
- TensorFlow Lite

Is anira free and open source?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, anira is open source and available under the Apache-2.0 license.

Technical Questions
~~~~~~~~~~~~~~~~~~~

How does anira ensure real-time safety?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
anira ensures real-time safety through several mechanisms:
- No dynamic memory allocation during audio processing
- Static thread pool to avoid oversubscription
- Lock-free communication between audio and inference threads
- Pre-allocation of all required resources
- Consistent timing checks and fallback mechanisms

What's the minimum latency I can achieve?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The minimum latency depends on your model complexity, hardware, and audio buffer size. anira is designed to minimize latency while maintaining real-time safety. You can use the built-in benchmarking tools to measure the actual latency for your specific use case.

Can I use multiple models simultaneously?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes, you can create multiple InferenceHandler instances, each with its own model configuration. However, be mindful of CPU resources as multiple models may compete for processing power.

How do I convert my TensorFlow/PyTorch model to work with anira?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can convert:
- TensorFlow models to TensorFlow Lite using the TF Lite converter
- PyTorch models can be used directly with LibTorch or exported to ONNX
- Most models can be converted to ONNX format which works with the ONNX Runtime backend

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

**Solution**: anira requires C++17 or later. Ensure your compiler supports C++17 and add the following to your CMakeLists.txt:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

Runtime Issues
~~~~~~~~~~~~~~

Audio Glitches or Dropouts
^^^^^^^^^^^^^^^^^^^^^^^^^^
**Issue**: Audio processing experiences dropouts or glitches during inference.

**Solutions**:
1. Increase the maximum inference time in `InferenceConfig`
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
3. Ensure tensor shapes in `InferenceConfig` match the model's expected shapes
4. Try a different backend if available

Thread Priority Issues
^^^^^^^^^^^^^^^^^^^^^^
**Issue**: Thread priority settings fail, particularly on Linux.

**Solution**: On Linux, you may need to set the `rtprio` limit for your user. Add the following to `/etc/security/limits.conf`:
```
your_username - rtprio 99
```
Log out and back in for the changes to take effect.

Unexpected Results or Crashes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Issue**: Inference produces incorrect outputs or crashes.

**Solutions**:
1. Validate tensor shapes in `InferenceConfig` match your model's expectations
2. Ensure your pre/post-processing logic correctly handles the data format
3. Try using a different backend to rule out backend-specific issues
4. Check that your model works correctly outside of anira (e.g., in PyTorch or TensorFlow directly)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Improving Inference Speed
^^^^^^^^^^^^^^^^^^^^^^^^^
1. **Use Smaller Models**: Consider model pruning, quantization, or distillation
2. **Optimize Tensor Shapes**: Reshape tensors to utilize hardware optimizations
3. **Backend Selection**: Benchmark your model across available backends to find the fastest
4. **Buffer Size Tuning**: Experiment with different audio buffer sizes
5. **Thread Priority**: Ensure inference threads run at appropriate priority levels

Reducing Latency
^^^^^^^^^^^^^^^^
1. **Decrease Buffer Size**: Use smaller audio buffer sizes where possible
2. **Model Optimization**: Use simpler models or quantized versions
3. **Backend Selection**: Some backends may offer lower latency for specific models
4. **Thread Priority**: Properly configure thread priorities for inference tasks

Additional Resources
--------------------

If you continue to experience issues:

1. Check the `examples` directory for working examples
2. Review the detailed API documentation
3. File an issue on the [GitHub repository](https://github.com/anira-project/anira/issues)
4. Consult the [paper](https://doi.org/10.1109/IS262782.2024.10704099) for technical details on anira's design
