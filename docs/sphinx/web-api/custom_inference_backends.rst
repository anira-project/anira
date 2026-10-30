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

``init`` (optional, may be async) loads what the engine needs before the Worker's loop: a runtime
instantiated. ``load(model)`` (optional, synchronous) receives the model entry the calls run, what
a C engine's load reads of its row: ``model.bytes`` (a ``Uint8Array`` view of the shared memory,
no copy; valid while the loaded model lives), else ``model.path``, or neither for an engine that
declares ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``. It runs on each Worker before the first call on a
loaded model; ``ModelConfig.fromUrl`` attaches the bytes of a JavaScript engine's entry, so the
model file is fetched once, on the main thread, and every Worker reads the same memory.
``process(ctx, prepared)`` runs once per inference (``prepared`` the loaded model's record, which
:js:func:`~registry.engineModel` reads): ``engineTensors(env, ctx)``
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
     models: [{ engine: 'org.example.double' }], // no path: the engine needs no model
     // ...
   }))
   const pipeline = Pipeline.create(anira)
     .addInference(model)
     .registerEngine('org.example.double', { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })

``flags`` states the engine's promises (``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``: it reads no model
file, so its model entry needs no ``path``). Register before ``Handler.create``: a handler copies the pipeline.

ONNX Runtime Web
----------------

``@anira-project/anira/engines/onnxruntime-web`` wraps onnxruntime-web's own WebAssembly build
as such an engine: its ``init`` loads onnxruntime-web's binary (``dist/wasm/
ort-wasm-simd-threaded.wasm``), its ``load`` creates a session from the model entry's bytes
(copied into onnxruntime-web's own heap, a separate module) or, for an entry with a path and no
bytes, from a synchronous fetch of the path, and its ``process`` runs one float32 ORT tensor per
model tensor of the call:

.. code-block:: ts

   // the worker module
   import { defineEngine } from '@anira-project/anira/scope'
   import { ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb } from '@anira-project/anira/engines/onnxruntime-web'

   defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb())

The model is the config's, stated once:

.. code-block:: json

   { "models": [{ "engine": "org.anira.onnxruntime-web", "path": "model.onnx" }], "inputs": [], "outputs": [] }

``await ModelConfig.fromUrl(anira, url)`` fetches ``model.onnx`` once and attaches its bytes;
each inference Worker's engine creates its session from them. ``onnxRuntimeWeb({ model })``
(a URL or bytes) still takes a model of its own, which each Worker then fetches in its ``init``.
The page registers ``ONNXRUNTIME_WEB_ENGINE_ID``, with the id from the root entry (``import { ONNXRUNTIME_WEB_ENGINE_ID } from '@anira-project/anira'``):
the id is a module of its own, so the page does not bundle onnxruntime-web; only the worker
module that defines the engine does. ``npm run check:bundle`` holds the root and scope entries to
that.

``onnxRuntimeWeb({ numThreads: 2 })`` runs each inference on two threads: the inference Worker's
own and one Worker that onnxruntime-web's module starts in the engine's ``init``, before the
inference Worker's loop (its glue loads itself into it). anira's own ONNX Runtime engine cannot do
this: the module starts no pthread. It pays off where anira's inference threads cannot run
inferences side by side: a stateful model (a declared state pair), whose inferences run one
after another, or one large model on one handler. Every inference Worker's runtime starts its own
threads, so a pool of ``n`` Workers uses up to ``n × numThreads`` threads; size the pool and the
count together. The engines one worker module defines share its runtime, so they share one count:
an engine asking for another fails its ``init``. The default is 1.

.. code-block:: ts

   // the worker module: two threads per inference
   defineEngine(ONNXRUNTIME_WEB_ENGINE_ID, onnxRuntimeWeb({ numThreads: 2 }))

``onnxRuntimeWeb({ warmup: 3 })`` runs three inferences over zeros of the model's input shapes
right after the engine creates its session, so the stream's first inferences do not pay
onnxruntime-web's first-run setup. A contract's ``"warmup": { "fixed": n }`` warms anira's
built-in engines only; a JavaScript engine warms itself. Over the model entry the session, and so
the warm-up, comes with the entry at the first call, inside the stream. Given its own ``model``,
the engine creates the session in its ``init``, before the inference Worker's loop, and the
stream starts warm; ``ModelConfig.fromUrl(anira, url, { fetchModel: () => false })`` then keeps
the page from fetching the file a second time.
