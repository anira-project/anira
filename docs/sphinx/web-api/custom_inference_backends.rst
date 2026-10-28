Custom Inference Backends
=========================

A JavaScript engine is a class the inference Workers call in place of a built-in engine: the
web twin of a C custom engine (:doc:`../custom_engines`). On the C side it is an
``anira_engine_desc`` whose ``process`` (and, opted in, ``reset`` and ``query``) are trampolines
into the hook object of the Worker that runs the inference, created under a reverse-URI id and
added to the pipeline by ``_anira_em_pipeline_register_js_engine``. A model entry names the
engine by that id; the pipeline runs it as ``ANIRA_ENGINE_CUSTOM``. Writing one takes three
steps: the class, a worker file that knows it, the registration.

Step 1: Implement the Engine
----------------------------

Subclass :js:class:`JSBackend`. ``init()`` loads what the engine needs; the Worker awaits it
before it reports the engine registered, so the engine is ready before the host prepares the
handler. ``process(ctx, prepared)`` runs once per inference on the Worker: ``ctx`` points at the
call's ``anira_engine_ctx``, whose input and output tensors ``this.inputs(ctx)`` /
``this.outputs(ctx)`` read through ``layout.ts`` (``_anira_tensor_data_f32``,
``_anira_tensor_num_elements``). Return an ``anira_status``; a thrown exception fails the
inference (``ANIRA_ERROR_ENGINE``, the chunk delivers zeros).

.. code-block:: typescript

   // gain-engine.ts
   import { JSBackend, anira_status } from '@anira-project/anira'

   export class GainEngine extends JSBackend {
     override process(ctx: number): number {
       const f32 = this.env.heap.f32
       const [input] = this.inputs(ctx)
       const [output] = this.outputs(ctx)
       if (!input || !output || input.f32Index === 0 || output.f32Index === 0) {
         return anira_status.ANIRA_ERROR_NOT_SUPPORTED // not host float32
       }
       for (let i = 0; i < output.numElements; i++) {
         f32[output.f32Index + i] = 0.5 * f32[input.f32Index + i]
       }
       return anira_status.ANIRA_OK
     }
   }

``this.env`` holds the Worker's module and heap, the engine id, and the model config and entry
index the engine was registered with; :js:class:`ONNXRuntimeWebBackend` reads its entry with
``_anira_model_config_model_path`` (a URL it fetches) or ``_anira_model_config_model_bytes``.

Step 2: A Worker File That Knows the Class
------------------------------------------

The bundled worker knows ``ONNXRuntimeWebBackend`` only. Ship a worker file that hands your
classes to anira's worker runtime:

.. code-block:: typescript

   // gain-worker.ts
   import { setupInferenceWorker } from '@anira-project/anira'
   import { GainEngine } from './gain-engine'

   setupInferenceWorker({ GainEngine })

Step 3: Register It
-------------------

On the main thread, register the engine on the pipeline before ``_anira_handler_create`` and
spin the Workers up from your file:

.. code-block:: typescript

   const jsKey = await anira.registerJSEngine(pipeline, {
     engineId: 'org.example.gain', // the model entry's "engine"
     className: 'GainEngine',
     modelConfigPtr: model,
     engineFlags: 0, // ANIRA_ENGINE_FLAG_*: what the engine promises
   })
   const worker = await anira.spinUpInferenceWorker(
     context,
     new URL('./gain-worker.ts', import.meta.url)
   )

The model entry: ``{ "engine": "org.example.gain", "path": "unused.bin" }`` (an engine that
loads nothing from its entry may name any path; anira never opens a registered engine's
path). ``engineFlags`` carries the descriptor flags JavaScript cannot write itself:
``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``, ``ANIRA_ENGINE_FLAG_REALTIME_SAFE``,
``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``, ``ANIRA_ENGINE_FLAG_STATE_ALIAS`` (:doc:`../custom_engines`
says what each promises). A registered engine has no providers list, so it serves
``ANIRA_PROVIDER_CPU`` alone; ``hookMask`` adds ``ANIRA_EM_ENGINE_HOOK_RESET`` for an engine
that keeps per-stream state of an exclusive handler.

Engines registered after a Worker started reach it too: ``registerJSEngine`` stops that
Worker's thread, sends the registration, awaits the engine's ``init()`` and starts the thread
again. ``unregisterJSEngine(jsKey)`` removes the Worker-side objects (their ``destroy()`` runs);
the C engine stays with the pipeline and the handlers that hold it.

The Lower Level
---------------

``registerJSEngine`` is ``_anira_em_pipeline_register_js_engine(pipeline, engineId, jsKey,
hookMask, engineFlags, err)`` plus the Worker registrations. A C or JavaScript caller that
needs a providers list fills the descriptor with ``_anira_em_engine_desc_init(desc, jsKey,
hookMask, engineFlags)``, sets the list, and calls ``_anira_custom_engine_create`` and
``_anira_pipeline_add_engine`` itself; the query hook (``ANIRA_EM_ENGINE_HOOK_QUERY``) then runs
on the main thread, so the main instance's :js:class:`HookRegistry` needs the engine object too
(``anira.anira.hooks.engines.set(jsKey, engine)``).
