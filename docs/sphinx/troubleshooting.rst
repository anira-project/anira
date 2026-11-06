Troubleshooting & FAQ
=====================

Questions
---------

Which platforms are supported?
   macOS, Linux and Windows (x86_64 and arm64), Android, iOS, and the browser through WebAssembly
   (:doc:`web-api/index`). The engines each platform ships are in README's platform table
   (:doc:`getting_started`).

Which engines are supported?
   ONNX Runtime, LibTorch, LiteRT (or the legacy TensorFlow Lite runtime, mutually exclusive with
   it) and ExecuTorch, plus any engine you register (:doc:`custom_engines`), and on the web the
   ONNX Runtime Web engine.

How does anira stay real-time safe?
   The ``ANIRA_NONBLOCKING`` entries never allocate, lock or wait: the audio thread copies into and
   out of preallocated rings, runs the processor's ``pre_process`` / ``post_process`` (which must
   keep the processor's real-time promise) and enqueues chunks into a lock-free queue. The
   inferences run on the context's inference threads, one pool per process for every handler. Only
   a ``_wait`` twin waits, and only when the host calls it (:doc:`architecture`).

What latency can I get?
   The figure anira computes from the host blocks, the model, the contract's inference budget and
   wait ratio, and the instance count (:doc:`latency`); a declared latency replaces it. A ``_wait``
   entry with a wait ratio can collect a result in the callback that submitted it, at the price of
   waiting on the audio thread.

Can I run several models at once?
   Yes: one handler per model. All handlers of a process share the one inference pool.

Is anira free and open source?
   Yes, under the Apache-2.0 license.

Troubleshooting
---------------

Compilation Issues
~~~~~~~~~~~~~~~~~~

Engines fail to download or configure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Turn off what you do not need (``-DANIRA_WITH_LIBTORCH=OFF``, ``-DANIRA_WITH_ONNXRUNTIME=OFF``,
``-DANIRA_WITH_LITERT=OFF``, ``-DANIRA_WITH_EXECUTORCH=OFF``; the legacy TensorFlow Lite engine,
``ANIRA_WITH_TFLITE``, is off by default and needs LiteRT off), point an engine at a prebuilt tree
with ``-DANIRA_<ENGINE>_ROOTDIR=/path`` (``include/`` and ``lib/``), or build offline with
``-DANIRA_BACKENDS_SKIP_REMOTE_CHECK=ON`` (README, *C++ Build Options*). An engine that does not
ship the linkage of the build (LibTorch in a static build, ExecuTorch in a shared one) is disabled
with a warning.

Compiler errors about the C++ standard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

anira and ``anira/anira.hpp`` need C++20; the C headers compile as C11 and C++17.

Runtime Issues
~~~~~~~~~~~~~~

Audio Glitches or Dropouts
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: Audio processing experiences dropouts or glitches during inference.

**Solutions**:
    1. Read the plan's verdict (the plan report, logged at prepare; :ref:`usage-plan-report`): a ``MARGINAL`` or ``NOT_REALTIME`` plan does not keep up on this machine (below).
    2. Pin an explicit budget above the measured one (``anira::Hard::budget``, ``"budget": {"ms": x}`` in the contract file): more latency, more headroom for load the warm-up could not see.
    3. Reduce the complexity of your neural network model
    4. Increase audio buffer size (though this increases latency)
    5. Check if other processes are consuming CPU resources
    6. Use the benchmark fixture of ``examples/benchmark/fixture/`` (:doc:`benchmarking`) to identify performance bottlenecks

``ANIRA_MISSED`` from a Hard entry, or ``anira_handler_rt_error`` reporting ``ANIRA_ERROR_CAPACITY``, tells the two kinds apart: a missed block (an inference late for its block) and a dropped hop (every inference slot busy, the hop never inferred). Both are filled by the contract's miss policy at their own stream positions, a late result is discarded and never delivered, and the output returns to exactly the reported latency by itself once the inferences keep up again (:doc:`latency`, *When the inference does not keep up*). A host that pops faster than it pushes keeps missing: every sample that arrives is late for a pop it already made.

A model too slow for the stream
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not an error: the prepare succeeds, the plan's ``rt_verdict`` reads ``ANIRA_RT_NOT_REALTIME``, a
Warning says so, the session is sized for one host block's burst and the stream misses blocks under
the miss policy. Hold the plan and ask the user ("Try nonetheless", :ref:`usage-plan-report`), or
pick another engine or provider, a smaller model, a larger block. An offline render is never held.

Prepare fails with ``ANIRA_ERROR_INVALID_STATE`` about the warm-up
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The contract's warm-up pass (the default) runs on the inference threads, so a prepare needs one in
its loop. Natively that is a context of ``threads(0)`` with no thread started: start an
:cpp:class:`anira::InferenceThread` (``anira_inference_thread_start`` returns once it is in its loop)
or run ``run_loop`` on a thread of your own before the prepare. On the web it is a prepare before
``await context.ready()``. The same status names a pass inference that did not complete within 10 s.
An explicit budget with ``ANIRA_WARMUP_NONE`` needs no thread (:ref:`usage-budget-warmup`).

Missed blocks right after a JUCE standalone starts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A JUCE standalone logs a few ``Missing samples`` warnings at start-up, then runs clean. Its
wrapper starts the audio device before ``AudioProcessorPlayer::setProcessor`` calls
``prepareToPlay`` under the lock the audio callback takes, so the first ``processBlock`` calls
arrive back to back after the prepare and outrun the first inferences. Plugin hosts resume at their
real-time pace and do not show it. Nothing to do: the stream stays time-aligned. When you chain
handlers by hand (one's ``pop_data`` into another's ``push_data``), treat a pop that returns
``ANIRA_MISSED`` like a delivered block, or the chain stays out of step.

A model does not load
^^^^^^^^^^^^^^^^^^^^^

The status and message name the engine, the path and the engine's own text (:doc:`logging`).
Check that the file exists (a path in a model file resolves relative to that file), that its format
is the entry's engine's, and that the tensor specs match the export's shapes; the plan report says
how each slot was bound. Try another engine or provider. The minimal-inference examples run a file
on an engine without anira (:doc:`examples`).

``wait strategy mismatch`` in the log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All handlers of a process share one pool, which waits with the strategy of the first context; a
later context asking for another one is ignored with this warning. The results are the same, but
the idle-CPU and pickup behaviour stays the first context's. Use one wait strategy in every
context config and context file of the process.

Thread priority fails on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Allow real-time priorities for your user in ``/etc/security/limits.conf`` and log in again:

.. code-block:: text

    your_username - rtprio 99

Wrong output or crashes
^^^^^^^^^^^^^^^^^^^^^^^

Check the tensor specs and the bindings in the plan report against the export, the processor
against the hop rule (:doc:`custom_preprocessing`), and the same model on another engine.

The log is empty, or a record is missing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

anira hands a failure back to the caller and does not log it; the log carries what nobody could
receive and the real-time refusals (:doc:`logging`).

- Two switches, both off by default: ``ANIRA_LOG_FLAG_TRACE_FAILURES`` on the context config logs
  every failed status too, and a sink of your own (``anira_log_fn``,
  ``anira::ContextConfig::log_sink``) receives every record, to keep the last ones for your crash
  handler.
- Real-time records wait in a queue for the drain thread (10 ms by default) or a failing
  control-path call, and anira installs no crash handler: a crash inside that window loses them.
  Lower ``drain_interval_ms`` in a debug build, or pump ``anira_drain_log`` from your crash handler
  if your sink is async-signal-safe. On the web the page pumps ``anira.drainLog()``.
- Where the records go on each platform: :doc:`logging`, *Sinks*.

Plugins
~~~~~~~

A plugin that crashes inside one host, a host that ships its own engine runtime or tanh-lib, and a
crash when the host unloads the plugin: :doc:`plugins`.

.. note::
    If a problem remains, file an issue on `GitHub <https://github.com/anira-project/anira/issues>`_.
