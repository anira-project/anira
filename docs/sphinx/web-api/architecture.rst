Architecture
============

``Anira Web`` is the WebAssembly distribution of anira: the same C++
library, compiled to WASM and wrapped in a
TypeScript API. The TypeScript layer's job is to spread that WASM
module across the browser threads anira needs in order to run
real-time inference — a main thread, the audio worklet thread, and one
or more inference worker threads.

.. code-block:: text

   ┌──────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
   │   Main thread    │    │ Inference worker(s)│    │ Audio worklet thread │
   │                  │    │   (Web Worker(s))  │    │                      │
   │  Setup           │    │                    │    │  AudioWorklet-       │
   │                  │    │  Model inference   │    │   Processor          │
   │  Configuration   │◀──▶│  (WASM ONNX,       │◀──▶│  Real-time           │
   │                  │    │   onnxruntime-web, │    │   process()          │
   │  UI control      │    │   or custom)       │    │  Pre/Post-processor  │
   └──────────────────┘    └────────────────────┘    └──────────────────────┘
            ▲                       ▲                          ▲
            └───────────────────────┴──────────────────────────┘
                       Shared WebAssembly memory

The number of inference workers is up to you. Each call to
``aniraWeb.spinUpInferenceWorker()`` spawns a new Web Worker hosting an
``InferenceThread`` — the same primitive anira uses for its desktop
thread pool. One worker is enough for simple models on most machines; spawn more if
you see audio dropouts, so anira can run inference on multiple batches
in parallel.

All threads share a single ``WebAssembly.Memory`` instance, so
configuration objects, ring buffers, and tensor data live at the same
heap addresses everywhere. Cross-thread coordination uses message
passing for setup and atomics on shared memory for the real-time path.

Main Thread
-----------

The main thread is where you set up anira. Calling
``await AniraWeb.create()`` instantiates the WASM module and returns
the ``aniraWeb`` factory; from there you wire up the model, inference
configuration, and pre/post-processing the same way you would in C++.

The main thread also owns your UI. Non-streamable tensor values
written from here — through ``setInput`` and similar APIs — reach the
model without blocking the audio path, so a slider or toggle can
update the model from frame to frame.

Inference Worker
----------------

``await aniraWeb.spinUpInferenceWorker()`` starts a Web Worker that
owns inference execution. Pulling inference off the audio thread is
what keeps the audio worklet's ``process`` callback real-time-safe
even when a forward pass takes longer than one audio block.

The worker hosts the inference engine itself, regardless of where that
engine actually runs. ``Anira Web`` ships with two built-in engines:
ONNX Runtime compiled into the WASM module, and ``onnxruntime-web`` on
the JavaScript side (:js:class:`ONNXRuntimeWebBackend`). User-written JS backends
also run on this worker. See :doc:`custom_inference_backends`.

You can also replace the worker entry point itself:

.. code-block:: typescript

   await aniraWeb.spinUpInferenceWorker(
     new URL('./customInferenceWorker.ts', import.meta.url)
   )

``spinUpInferenceWorker()`` returns an ``InferenceWorker`` handle.
When you're done with a worker — for instance when reconfiguring or
unloading the model — call ``worker.stop()`` to halt its inference
thread, terminate the underlying ``Worker``, and remove it from
``aniraWeb.getActiveWorkers()``. Workers spun up but never stopped
stay alive for the lifetime of the page.

Audio Worklet Thread
--------------------

The browser's ``AudioWorkletGlobalScope`` runs the audio callback. Anira
ships with a default worklet that handles the common case: a
single-tensor model with in-place stereo or mono I/O. To install it:

.. code-block:: typescript

   await aniraWeb.registerAudioWorkletForContext(audioContext)
   const node = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor
   )

For models that need more — multi-tensor I/O, a custom processing
buffer size, ``AudioParam`` integration, or a JS pre/post processor —
you provide a custom worklet file. See :doc:`custom_audio_worklets`.

.. note::
   :js:class:`JSPrePostProcessor` subclasses are constructed on the
   audio worklet thread, not on the main thread. Pre- and
   post-processing run in the real-time callback, so the JS object that
   implements them must live where that callback runs.

Three Customization Axes
------------------------

Most extension work falls into one of three independent categories,
each with its own page:

1. :doc:`custom_audio_worklets` — extend
   :js:class:`AniraAudioWorkletBase` for multi-tensor models
   (``processMulti``), custom ``maxBufferSize``, ``AudioParam``
   integration, or to host a custom :js:class:`JSPrePostProcessor`.
2. :doc:`custom_pre_post_processing` — subclass
   :js:class:`JSPrePostProcessor` to run JavaScript before and after
   inference (windowing, normalization, parameter clamping, etc.).
3. :doc:`custom_inference_backends` — replace the WASM-side runtime with
   a JavaScript backend. Built-in options
   (:js:class:`JSBackendBase`, :js:class:`ONNXRuntimeWebBackend`) and
   user-written backends both run on the inference worker.

Custom pre/post processing **requires** a custom worklet (because the
subclass must be instantiated on the audio thread); custom worklets and
custom backends are otherwise independent and can be combined freely.

The JS ↔ WASM Bridge
--------------------

C++ objects live in WASM-managed memory and are referenced by raw
numeric pointers. The TypeScript wrappers (``InferenceHandler``,
``PrePostProcessor``, ``BufferF``, …) extend :js:class:`BaseWrapper`,
which holds two fields per instance: ``ptr`` (the C++ pointer) and
``wasmInstance`` (the Emscripten module). Every wrapper method
forwards into a ``_<class>_<method>`` C export with ``this.ptr`` as
the first argument.

Most wrapper APIs accept the union type
``PossiblePointer<T> = T | number``, so you can pass either a wrapper
instance *or* a raw numeric pointer. This avoids forcing an allocation
just to call into the next wrapper — when you already have a pointer
in hand (for example from a worklet message or another wrapper's
``getPointer()``), pass it directly.

Helpers, all exported from the package root:

+--------------------------------+----------------------------------------------------------+
| Helper                         | Role                                                     |
+================================+==========================================================+
| ``resolvePtr(value)``          | Coerce a ``PossiblePointer`` to a number — returns       |
|                                | ``value`` if it's already numeric, ``value.getPointer()``|
|                                | if it's a wrapper instance. Use this inside hot loops or |
|                                | when calling raw WASM exports yourself.                  |
+--------------------------------+----------------------------------------------------------+
| ``instance.getPointer()``      | Return the wrapper's underlying C++ pointer as a         |
|                                | number. Symmetric to ``resolvePtr`` for the              |
|                                | wrapper-to-pointer direction.                            |
+--------------------------------+----------------------------------------------------------+
| ``instance.wrapPointer(Cls,    | Build a wrapper of class ``Cls`` around an existing      |
| ptr)``                         | pointer, reusing ``this.wasmInstance``. Skips the C++    |
|                                | constructor — useful for viewing C++ objects you don't   |
|                                | own.                                                     |
+--------------------------------+----------------------------------------------------------+
| ``Cls.createFromPointer(       | Static counterpart of ``wrapPointer`` for cases where    |
| module, ptr)``                 | you don't have an existing wrapper handy (e.g.           |
|                                | reconstructing a ``JSPrePostProcessor`` subclass on the  |
|                                | worklet thread from ``state.prePostProcessorPtr``).      |
+--------------------------------+----------------------------------------------------------+

.. _lifecycle-and-cleanup:

Lifecycle and Cleanup
---------------------

Wrapper instances expose a ``destroy()`` method that frees the
underlying C++ object via the corresponding ``_<class>_destroy`` C
export. JavaScript has no destructors, so the GC won't call this for
you — the C++ memory only goes away when ``destroy()`` runs. For a
long-lived page that loads a model once and keeps inferring, the leak
is harmless (the module stays alive for the session anyway); for apps
that swap models, recreate handlers, or run under a test harness,
``destroy()`` is what you call.

Not every wrapper needs ``destroy()``, though. Whether a wrapper is
*owning* depends on how it was created:

* A wrapper from ``new SomeClass(...)`` runs the TS constructor,
  which calls ``_<class>_create`` and stashes the fresh C++ pointer.
  This wrapper is the only handle to that C++ object — calling
  ``destroy()`` on it frees the object.
* A wrapper from ``wrapPointer`` or ``createFromPointer`` skips the
  TS constructor entirely; it's a view over a C++ object that was
  allocated by somebody else (typically another wrapper, or the
  inference worker). **Don't call ``destroy()`` on these** — doing so
  would free a C++ object that other code is still pointing at.

For example, ``InferenceConfig.getTensorInputShape()`` returns a
``TensorShapeList`` view; the underlying storage belongs to the
``InferenceConfig``, so you destroy the config, not the view.

When you do tear down a full setup, free the handler before the
config and processor it references — the handler holds pointers back
into them, so freeing them first leaves it with dangling references:

.. code-block:: typescript

   inferenceHandler.destroy()   // first: holds refs into pp + config
   ppProcessor.destroy()
   inferenceConfig.destroy()    // last among the three
   // ProcessingSpec, VectorModelData, VectorTensorShape, etc. can
   // then be destroyed in any order.
