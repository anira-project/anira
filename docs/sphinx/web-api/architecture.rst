Architecture
============

``Anira Web`` is anira compiled to WebAssembly and the TypeScript runtime that spreads the
module across the browser threads real-time inference needs: the main thread, the audio
worklet thread, and one or more inference Worker threads.

.. code-block:: text

   ┌──────────────────┐    ┌────────────────────┐    ┌──────────────────────┐
   │   Main thread    │    │ Inference Worker(s)│    │ Audio worklet thread │
   │                  │    │                    │    │                      │
   │  configs, the    │    │  run_loop of an    │    │  one handler process │
   │  pipeline, the   │    │  inference thread: │    │  per render quantum  │
   │  handler, the    │◀──▶│  the engine call   │◀──▶│  stage pre/post/     │
   │  log pump, UI    │    │  (built-in ORT or  │    │  reset hooks         │
   │  stage init/prep │    │  a JS engine),     │    │                      │
   │                  │    │  before/after hooks│    │                      │
   └──────────────────┘    └────────────────────┘    └──────────────────────┘
            ▲                       ▲                          ▲
            └───────────────────────┴──────────────────────────┘
                       One shared WebAssembly.Memory

Module Instances on One Memory
------------------------------

Every thread instantiates the same module (``createAniraWasm``) on one shared
``WebAssembly.Memory``, so a config, a handler, a ring or a tensor lives at the same heap
address everywhere and a pointer is passed between threads as a number. The main instance is
created by ``AniraWeb.create()``. Every other instance (each inference Worker, the worklet) runs
on a stack and a thread-local block of its own: the main thread allocates both
(``_anira_em_alloc_thread_stack``), the instance's first C call installs them
(``_anira_em_init_instance``), so the ``thread_local`` variables of anira, tanh-lib and the
runtime are per instance, as they are per thread natively. The main instance installs a block
of its own the same way (the module has no ``main()``, so none exists at instantiation, and
the runtime's exception state lives there). Setup crosses threads by ``postMessage``; the
real-time path uses only the shared memory and its atomics.

JavaScript Reaches anira Through the C ABI
------------------------------------------

The module's exports are the C ABI: ``aniraWeb.module._anira_context_create(...)`` is the
same call a C host makes (:doc:`../usage`), with pointers as numbers. Records are read and
written only through the generated offsets of ``abi/layout.ts`` (``layout.anira_tensor``,
``layout.anira_engine_ctx``, …), enums and defines come from ``abi/enums.ts``, and an
``anira_error`` is read by :js:func:`callWithError`, which throws an :js:class:`AniraError`.
The :js:class:`Heap` holds the views over the memory and the ``>>> 0`` rule: a pointer, a
``size_t`` or a ``uint32_t`` an export returns arrives as a signed 32-bit number, so it is
made unsigned before use.

The Hook Object and the Trampolines
-----------------------------------

What runs in JavaScript (a stage, an engine, the log) is reached through the Emscripten
companion ``anira/anira_em.h``. Each instance has one hook object, ``Module.anira``
(``{ version, stage, engine, log }``), installed before instantiation over that instance's
:js:class:`HookRegistry`, which maps a ``js_key`` to the stage or engine object. A JavaScript
stage or engine is a descriptor whose slots are C trampolines compiled into anira: each calls
the hook of the instance that runs the C call, so

* ``preProcess``, ``postProcess`` and ``reset`` of a stage run in the worklet (under a Hard
  contract the driving thread is the audio thread),
* ``beforeInference`` / ``afterInference`` and an engine's ``process`` run on the inference
  Worker that dequeued the job,
* a stage's ``init``, ``prepare``, ``unprepare`` and ``release`` run on the main thread.

Every request to a Worker or the worklet carries a request id its reply and its error echo,
so concurrent requests on one target never answer or fail each other. The same ``js_key``
therefore names one object per instance: :js:meth:`AniraWeb.addJSStage`
puts the main-thread object into the main registry and has every Worker construct the class
for its own registry; the worklet constructs it from its ``stageClasses``. A stage released
on the main thread (its last carrier died) is dropped from every instance's registry. What an
instance could never serve is refused at registration: a stage with main-thread bits and no
main-thread object, an engine with the query bit, a worklet configured without the stage its
handler's pipeline runs there. A hook that throws
is caught in its trampoline, reported once with ``console.error``, and fails the call
(``ANIRA_ERROR_UNKNOWN`` for a stage, ``ANIRA_ERROR_ENGINE`` for an engine): the chunk
delivers zeros and the handler's real-time error is latched, as for a failing C callback.

Main Thread
-----------

The main thread builds the configuration with raw calls: a context config and context (no
pool threads on the web: the Workers bring them), a model config (in code or from JSON), a
pipeline with the inference stage and, optionally, a JavaScript stage and JavaScript engines,
the handler, the contract, ``anira_handler_prepare``. It also owns the log: there is no drain
thread on the web, so ``startLogPump`` drains the real-time queue on an interval, and
:js:meth:`AniraWeb.setLogHook` routes the records of every instance to one callback.

Inference Workers
-----------------

``await aniraWeb.spinUpInferenceWorker(contextPtr)`` creates a user-driven inference thread of
the context's core on the main instance (``_anira_inference_thread_create``) and a Worker that
runs it: the Worker instantiates the module, enters its stack, registers the JavaScript
engines and Worker-side stages (an engine's ``init()`` is awaited), and calls
``_anira_inference_thread_run_loop`` inside its message handler. Spin up more Workers to run
more inferences at once.

While the queue is empty the loop parks on the wake word (``memory.atomic.wait32``, at most
5 ms at a time) instead of spinning a core; every enqueue and every stop notifies it.

The stop protocol has no message: ``_anira_inference_thread_stop`` requests the exit and the
main thread polls ``_anira_inference_thread_has_exited`` (a shared-memory flag) on a
``setTimeout`` until the Worker left the loop; only then is
``_anira_inference_thread_destroy`` legal. :js:class:`InferenceThread` and
``InferenceWorker.stop()`` implement it. A Worker handles no message while it is inside the
loop, so ``AniraWeb`` stops the thread, sends a registration, and starts it again whenever an
engine or stage is added later.

The engine a Worker calls is either the ONNX Runtime compiled into the module
(``ANIRA_ENGINE_ONNXRUNTIME`` on ``ANIRA_PROVIDER_CPU``, from model bytes) or a JavaScript
engine: :js:class:`ONNXRuntimeWebBackend` (``org.anira.onnxruntime-web``) or your own
:js:class:`JSBackend` subclass, see :doc:`custom_inference_backends`.

Audio Worklet Thread
--------------------

The worklet streams through a prepared handler: one ``_anira_handler_process`` per render
quantum over planar input and output tensors it builds once. The handler's Hard contract
declares ``block_min = block_max =`` the render quantum (128 frames); a call above the declared
block is refused, so the worklet builds its tensors at configure for the declared
quantum and, when the browser renders another one, outputs silence and reports it
(``onQuantumChange``): the host re-prepares the handler with the new block, then the worklet
rebuilds its tensors in its message handler, never in ``process()``, which allocates
nothing. The worklet
never waits unless asked to (``waitMs``, ``_anira_handler_process_wait``). See
:doc:`custom_audio_worklets`.

Customization
-------------

1. :doc:`custom_audio_worklets` — extend :js:class:`AniraAudioWorkletBase` to stream more
   slots, add ``AudioParam`` handling or host a JavaScript stage's pre/post/reset.
2. :doc:`custom_pre_post_processing` — a JavaScript stage (``anira_em_pipeline_add_js_stage``)
   that owns the pre/post phases or the hooks around the inference.
3. :doc:`custom_inference_backends` — a JavaScript engine
   (``anira_em_pipeline_register_js_engine``) run by the inference Workers.

Lifecycle and Cleanup
---------------------

C objects are freed by their ``_destroy`` entries, never by the garbage collector. Tear down in
the reverse order of creation: dispose the worklet (``await aniraWeb.disposeAudioWorklet(node)``:
the worklet frees its tensors and stops calling into the module, then its stack is freed),
stop the Workers (``await worker.stop()``, which destroys the thread and frees the Worker's
stack once the Worker said it destroyed its engines), then ``_anira_handler_destroy``,
``_anira_pipeline_destroy``, ``_anira_contract_destroy``, ``_anira_model_config_destroy``,
``_anira_context_destroy`` and ``_anira_context_config_destroy``. A JavaScript stage's
``release`` runs when its last carrier dies, a JavaScript engine's C object when the pipeline
and the handlers holding it are gone; ``unregisterJSEngine`` removes the Worker-side object.
