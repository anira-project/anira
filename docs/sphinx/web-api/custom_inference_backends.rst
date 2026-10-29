Custom Inference Engines
========================

An inference engine in TypeScript runs on the context's inference Workers, called synchronously
from each Worker's loop; it is the counterpart of a C ``anira_engine_desc``
(:doc:`../custom_engines`). The module's built-in ONNX Runtime engine needs none of this: a model
entry ``"engine": "onnxruntime"`` runs on it as natively.

The Worker Module
-----------------

Every inference Worker imports the context's worker module before it enters its loop, and
awaits the boot of its scope's registry: the ``init`` of every engine the module defined. Define
the engine there:

.. literalinclude:: ../../../web/examples/stage-and-engine/double-engine.ts
   :language: ts

and name the module in the context, bundled into one file (with Vite,
``import workerModule from './double-engine.ts?worker&url'``):

.. code-block:: ts

   const context = Context.create(anira, { threads: 2, workerModule })

``init`` (optional, may be async) loads what the engine needs: a model fetched by URL, a runtime
instantiated. ``process(ctx, prepared)`` runs once per inference: ``engineTensors(env, ctx)``
gives the call's input and output tensors (``elements``, and ``f32``, a view of a host float32
tensor's data); it fills the outputs and returns an ``anira_status``. ``reset`` (with
``hooks: ANIRA_EM_ENGINE_HOOK_RESET`` in ``registerEngine``) runs at a new stream of an exclusive
handler, on the Workers too; ``query`` (``ANIRA_EM_ENGINE_HOOK_QUERY``) answers what the engine
can run here and runs on the main thread, so a page whose engine has one imports the module there
as well. Each Worker builds its own instance, so an engine object is never shared between
threads.

``process`` must be synchronous: the Worker's event loop does not run while it is in its loop, so
an engine whose inference is a promise (onnxruntime-web's ``session.run()``, a WebGPU readback)
cannot run there yet. :ref:`web-realms` says what an engine can reach once the Worker runs.

Registering It
--------------

The model entry names the engine by its reverse-URI id, and the pipeline registers it under that
id:

.. code-block:: ts

   const model = await ModelConfig.fromJson(anira, JSON.stringify({
     models: [{ engine: 'org.example.double', path: 'no-model-file' }],
     // ...
   }))
   const pipeline = Pipeline.create(anira)
     .addInference(model)
     .registerEngine('org.example.double', { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })

``flags`` states the engine's promises (``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``: it reads no model
file). Register before ``Handler.create``: a handler copies the pipeline.

ONNX Runtime Web
----------------

``@anira-project/anira/engines/onnxruntime-web`` wraps onnxruntime-web's own WebAssembly build
as such an engine: its ``init`` loads onnxruntime-web's binary (``dist/wasm/
ort-wasm-simd-threaded.wasm``) and creates a session from the model, its ``process`` runs one
float32 ORT tensor per model tensor of the call:

.. code-block:: ts

   // the worker module
   import { defineEngine } from '@anira-project/anira/scope'
   import { ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb } from '@anira-project/anira/engines/onnxruntime-web'

   defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb({ model: new URL('./model.onnx', import.meta.url) }))

The page then registers ``ONNXRUNTIME_WEB_ENGINE_ID`` and names it in the model entry.
