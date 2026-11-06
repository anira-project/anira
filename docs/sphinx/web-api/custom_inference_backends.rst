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

.. literalinclude:: ../../../web/examples/processor-and-engine/double-engine.ts
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
gives the call's input and output tensors: ``elements``; ``dtype``, the tensor's ``anira_dtype``
(its spec's: a model of any dtype anira names reaches the engine in that dtype); ``data``, a typed
array of that dtype over the tensor's data (``Float32Array`` for float32, ``Int16Array`` for int16,
``BigInt64Array`` for int64, the 16-bit words as a ``Uint16Array`` for float16 and bfloat16, the
bytes as a ``Uint8Array`` for bool8); and ``f32``, ``data`` when the tensor is float32. It fills
the outputs and returns an ``anira_status``. ``reset`` (with
``hooks: ANIRA_EM_ENGINE_HOOK_RESET`` in ``addEngine``) runs at a new stream of an exclusive
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
     .addEngine('org.example.double', { flags: ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL })

``flags`` states the engine's promises (``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``: it reads no model
file, so its model entry needs no ``path``). Register before ``InferenceHandler.create``: a
handler copies the pipeline.

``edge`` states how the engine passes the slots to its runtime, for the plan report's slot rows
and the contract's edge cost (``anira::Engine::Loaded::edge``, the native descriptor's ``edge``,
:doc:`../custom_engines`): the engine runs in the inference Workers, so the main thread gives the
statements here, ``{ inputs, outputs }`` (``EngineEdges``), one for every input slot and one for
every output slot. That is per direction where C++ states per slot: the main thread adds the
engine before the model is known. A direction left out, and an engine without ``edge``, follows
the provider rule of that direction, zero-copy on
the CPU path; on a device provider ``ANIRA_EDGE_UNKNOWN``, since a browser exposes no device's memory kind
(``PlanInfo.memoryKind`` reads ``ANIRA_MEMORY_UNKNOWN``, the reason "memory kind not probed:
the web exposes no memory kind"), which every limit of the edge cost refuses. An engine that
copies the slots into host memory of its own says so, for both directions or for one:

.. code-block:: ts

   pipeline.addEngine('org.example.copying', {
     edge: {
       inputs: { edgeClass: anira_edge_class.ANIRA_EDGE_COPY, reason: 'copies into its own tensors' },
       outputs: { edgeClass: anira_edge_class.ANIRA_EDGE_COPY, reason: 'copies its results out' },
     },
   })
   // reads its inputs in place, copies its outputs: the inputs keep the provider rule
   pipeline.addEngine('org.example.outputs-copied', {
     edge: { outputs: { edgeClass: anira_edge_class.ANIRA_EDGE_COPY, reason: 'copies its results out' } },
   })

Its rows then read ``ANIRA_EDGE_COPY`` (a copy within one domain) moved by the engine
(``PlanSlot.movedBy``, ``ANIRA_MOVER_ENGINE``), with the reason of their direction.

ONNX Runtime Web
----------------

``@anira-project/anira/engines/onnxruntime-web`` wraps onnxruntime-web's own WebAssembly build
as such an engine: its ``init`` loads onnxruntime-web's binary (``dist/wasm/
ort-wasm-simd-threaded.wasm``), its ``load`` creates a session from the model entry's bytes
(copied into onnxruntime-web's own heap, a separate module) or, for an entry with a path and no
bytes, from a synchronous fetch of the path, and its ``process`` runs one ORT tensor per model
tensor of the call, of the tensor's dtype as ONNX's element type (every dtype anira names; a
tensor whose dtype is not the model's element type is ``ANIRA_ERROR_CONFIG``), its data copied
in and the outputs copied out (anira's module and onnxruntime-web's are two wasm memories):

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
The page registers ``ONNXRUNTIME_WEB_ENGINE_ID``, with the id from the root entry (``import {
ONNXRUNTIME_WEB_ENGINE_ID } from '@anira-project/anira'``; ``pipeline.addEngine(ONNXRUNTIME_WEB_ENGINE_ID)``).
Under that id ``addEngine`` states the engine's edges by default, a copy of every slot within host
memory by the engine (``ANIRA_EDGE_COPY``: the two wasm memories), each input "copied into ORT's
heap" and each output "copied out of ORT's heap", so its plan rows say so; an engine defined
under another id states them with ``addEngine(id, { edge: ONNXRUNTIME_WEB_ENGINE_EDGES })``. The
id and the statements are a module of their own, so the page does not bundle onnxruntime-web; only the worker module
that defines the engine does. ``npm run check:bundle`` holds the root and scope entries to
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
right after the engine creates its session, on every Worker, so the stream's first inferences
do not pay onnxruntime-web's first-run setup: the per-Worker cold start. A contract's warm-up pass
(:doc:`../usage` section 1.3) also reaches a JavaScript engine, as ordinary ``process`` calls on
the Workers that take its jobs, and times them: it measures the engine, which its own ``warmup``
keeps warm on every Worker. Over the model entry the session, and so
the warm-up, comes with the entry at the first call, inside the stream. Given its own ``model``,
the engine creates the session in its ``init``, before the inference Worker's loop, and the
stream starts warm; ``ModelConfig.fromUrl(anira, url, { fetchModel: () => false })`` then keeps
the page from fetching the file a second time.
