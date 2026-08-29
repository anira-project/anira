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
    - `-DANIRA_WITH_LITERT=OFF`
    - `-DANIRA_WITH_EXECUTORCH=OFF`

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

Wait Strategy Mismatch
^^^^^^^^^^^^^^^^^^^^^^

**Issue**: The log shows ``[WARNING] ContextConfig wait strategy mismatch``.

All anira instances in a process share one inference thread pool, and the pool's threads wait for work according to the :cpp:enum:`anira::WaitStrategy` of the *first* :cpp:struct:`anira::ContextConfig` the context was created with. A later instance that requests a different strategy has no effect — the warning tells you the originally configured strategy stays active. This is harmless (both strategies produce identical results), but the requested idle-CPU/latency characteristic is not the one in effect.

**Solution**: Use the same ``wait_strategy`` in every :cpp:struct:`anira::ContextConfig` (and in the ``context_config`` block of every JSON configuration file) that the process loads.

Thread Priority Issues
^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Thread priority settings fail, particularly on Linux.

**Solution**: On Linux, you may need to set the `rtprio` limit for your user. Add the following to `/etc/security/limits.conf`:

.. code-block::

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

Host application ships its own backend runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: A plugin embedding anira crashes (or fails to instantiate with an
"OrtGetApiBase resolved to an ONNX Runtime that does not support the API
version" error) inside a specific host application, but works in the
standalone build and in other hosts. Ableton Live 12, for example, bundles its
own ONNX Runtime dylib for its built-in AI features.

**Cause**: If backend symbols are exported from the plugin binary, the dynamic
linker can bind them across module boundaries — ELF interposition on Linux,
weak-symbol coalescing on macOS (e.g. the ORT C++ header's
``Ort::Global<void>::api_``). The plugin's backend calls then resolve against
the host's (typically older) runtime and the first API call crashes the host.

**Solutions**:
    1. Use anira's build system to link the prebuilt backend archives: ``anira_target_link_static_backend`` links the on-demand archives (ONNX Runtime, TFLite, LiteRT, mobile ExecuTorch) hidden (``-load_hidden`` on macOS, ``--exclude-libs`` on Linux/Android), the desktop ExecuTorch archives get ``--exclude-libs`` on ELF, and anira itself is compiled with hidden symbol visibility. A static ``libanira`` is compiled without any export decoration (``ANIRA_API`` is empty under ``ANIRA_STATIC``, which the CMake package defines for consumers), so a plugin that embeds it with hidden visibility exports nothing of anira. A shared ``libanira`` instead pins its export table to namespace ``anira`` at link time (``cmake/anira.map`` on ELF, ``cmake/anira.exports`` on macOS): compiler visibility alone cannot hide what a header stamps default-visibility itself — libstdc++'s ``std::`` instantiations, LibTorch's ``C10_API`` typeinfo, or the default-visibility ExecuTorch desktop archives. anira's ONNX Runtime processor additionally verifies at startup that ``OrtGetApiBase()`` resolved to a compatible runtime and throws a descriptive error instead of crashing the host.
    2. If your plugin's own translation units include backend headers (e.g. ``onnxruntime_cxx_api.h``), compile them with hidden visibility too (``CXX_VISIBILITY_PRESET hidden``, ``VISIBILITY_INLINES_HIDDEN ON``).
    3. Restrict your plugin's exports to its entry points — e.g. on macOS ``-Wl,-exported_symbols_list`` with only ``_bundleEntry``/``_bundleExit``/``_GetPluginFactory`` for a VST3, on Linux a version script (``-Wl,--version-script``) with the same names under ``global:`` and ``local: *;``. This also covers any other statically linked dependency and is *required* on macOS when your plugin links a static anira with the ExecuTorch backend: its desktop archives are force-loaded, and ld64 has no hidden variant of ``-force_load``, so nothing short of an export list keeps ``executorch::``/``xnn_*`` out of the plugin's export table. Verify with ``nm -gU your_plugin`` (macOS) or ``nm -D --defined-only your_plugin.so`` (Linux): no ``Ort``/backend symbols should appear. anira runs the same check on its own binaries in CTest (``anira_exports``, ``cmake/CheckExports.cmake``).

.. _plugin-library-unload:

Host crashes when unloading the plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: A plugin embedding anira works, but the host crashes when the last
instance is removed or when the host quits.

**Cause**: Hosts unload a plugin's shared library (``dlclose`` / ``FreeLibrary``)
once the last instance is gone; any thread still running inside it then executes
unmapped code. anira holds the required invariant — *no anira thread exists once the
last* :cpp:class:`anira::InferenceHandler` *is destroyed* — by construction: the
inference threads exist exactly while handler instances exist, and destroying the
last one stops and joins them before its destructor returns. The context's state is
never destroyed while the library is loaded (calling into anira is valid at any time,
even from late-running static destructors) and is reclaimed at unload.

**Solutions**:
    1. **Host unloads with a live instance** (a host bug, but it happens): on Linux
       and macOS a library-unload hook calls :cpp:func:`anira::Context::shutdown`
       automatically. On Windows nothing that runs at ``DLL_PROCESS_DETACH`` may join a
       thread (loader lock), so call ``anira::Context::shutdown()`` from your module-exit
       entry point — CLAP ``deinit``, VST3 ``ExitDll`` — as ``examples/clap-audio-plugin``
       does. It is idempotent and cheap when there is nothing to do.
    2. **You manage inference threads yourself** (``ContextConfig(0)`` +
       :cpp:func:`anira::Context::make_inference_thread`): stop them before your library
       can be unloaded; the hook only joins anira's own pool.
    3. **The library silently never unloads (GCC/Linux)**: glibc never unloads an object
       that defines an ``STB_GNU_UNIQUE`` symbol, which GCC emits for exported inline
       statics and template statics. anira compiles with ``-fno-gnu-unique``; add the
       same flag plus hidden visibility (``CXX_VISIBILITY_PRESET hidden``,
       ``VISIBILITY_INLINES_HIDDEN ON``) to your plugin's translation units, and verify
       with ``nm -DC your_plugin.so | grep ' u '`` (should print nothing).
    4. **macOS never unloads your plugin** (the opposite problem, and harmless): dyld
       does not unload images that use thread-local storage, which the statically linked
       backend archives (ONNX Runtime, LiteRT, ExecuTorch) do. Such a plugin — or a
       ``libanira.dylib`` with a static backend inside — stays mapped until the host
       quits and cannot crash this way; anira's ``test/unload`` skips its unmapped
       assertions in that configuration.

The scenario is covered by anira's host-shaped ``test/unload`` test, which loads a
plugin-shaped module from an executable that does not link anira, unloads it and
checks that it was really unmapped.

.. note::
    If you continue to experience issues feel free to file an issue on the [GitHub repository](https://github.com/anira-project/anira/issues).

