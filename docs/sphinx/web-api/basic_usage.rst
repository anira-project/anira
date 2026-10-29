Basic Usage
===========

This guide walks through the smallest end-to-end ``Anira Web`` setup: a bundled model config
run by the module's built-in ONNX Runtime engine on the context's inference Worker, streamed
through the handler's audio node. The code is ``web/examples/model-file/main.ts``, which
``npm run examples`` type-checks and builds and the browser test renders offline in Chromium and
Firefox. The steps are the native host's (:doc:`../usage`), each a
class over the C handle:

1. instantiate the module (:js:class:`Anira`) and route the log,
2. a context (:js:class:`Context`): the pool of inference Workers,
3. the model config and the contract, from the same files a native host loads
   (:js:class:`ModelConfig`, :js:class:`Contract`),
4. the pipeline and the handler (:js:class:`Pipeline`, :js:class:`Handler`),
5. the audio node (:js:meth:`Handler.createAudioNode`),
6. the teardown.

The calls are the C ABI's, typed: every class keeps its handle in ``ptr`` and the module's
bindings in ``abi`` (:js:class:`AniraBindings`, one method per ``anira_*`` entry, generated from
the ABI registry), so anything the classes do not cover is one ``anira.abi.anira_*`` call away.
A failed call throws an :js:class:`AniraError` with the entry's message.

.. note::
   The page must be cross-origin isolated (``Cross-Origin-Opener-Policy: same-origin``,
   ``Cross-Origin-Embedder-Policy: require-corp``): the module runs on a shared
   ``WebAssembly.Memory``. See :doc:`installation_building`.

1. The Module and the Log
-------------------------

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [module]
   :end-before: // [module]
   :dedent:

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [log]
   :end-before: // [log]
   :dedent:

The hook replaces the module's platform sink (its stderr, which the glue would print as
``console.error``): each record reaches the console once, through the console method of its
level (:doc:`../logging`).

2. A Context
------------

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [context]
   :end-before: // [context]
   :dedent:

The context's inference threads are Wasm Workers anira starts from C, ``threads`` of them
(``'auto'``: half the cores). A page with JavaScript engines names the module every Worker
imports first, ``workerModule`` (:doc:`custom_inference_backends`); the built-in engines need
none.

3. Configs Once
---------------

The model and contract files of a native host load unchanged (``modelsUrl``: the directory
that serves ``extras/models/model-pool/``):

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [configs]
   :end-before: // [configs]
   :dedent:

:js:meth:`ModelConfig.fromUrl` parses the file (``anira_model_config_from_json``) and, for every
model entry whose engine the module compiled in (the ONNX Runtime one) or a JavaScript engine's,
fetches the entry's file relative to the config's URL and attaches it as bytes
(``anira_model_config_set_model_bytes``); the other entries (a LibTorch or a TFLite one) keep
their paths unfetched. Serve the model files beside the config the way its paths say. A contract
file may leave the Hard geometry to the host: blocks from the render quantum (128) up to the
context's burst (``renderBurstFrames``) at the context's rate. A live context renders a device
callback's worth of quanta back to back, so ``block_max`` must cover that burst for the
inferences to keep up; the latency the handler reports then covers it too (:doc:`audio_nodes`).
:js:meth:`ModelConfig.fromJson` and :js:meth:`Contract.hard` build the same in code.

4. The Pipeline and the Handler
-------------------------------

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [handler]
   :end-before: // [handler]
   :dedent:

5. The Audio Node
-----------------

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [node]
   :end-before: // [node]
   :dedent:

The node is a Wasm Audio Worklet node whose render callback is C: per quantum it hands the
node's buffers to ``anira_handler_process`` as tensors and applies the miss rule
(:doc:`audio_nodes`). Connect ``audioNode.node`` into the graph. ``params`` serves float32 Static scalars as AudioParams of the node;
``staticInputs`` serves Static inputs of any shape through ``audioNode.setStaticInput``. For an
``OfflineAudioContext`` render that waits for every inference, pass ``waitMs`` and
``await context.ready()`` first (a Worker starts once the page's thread yields).

6. Teardown
-----------

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [teardown]
   :end-before: // [teardown]
   :dedent:

C objects are freed by their ``destroy``, never by the garbage collector.

Complete Examples
-----------------

``web/examples/model-file/`` is the page of this guide: ``main.ts`` builds the chain above and
plays a tone through it, ``check.ts`` renders it offline with the gain automated. Its Vite config
serves the repository's ``extras/models`` at ``/extras/models/``.

``web/examples/stage-and-engine/`` is a complete page that writes its own stage and engine in
TypeScript (:doc:`custom_pre_post_processing`, :doc:`custom_inference_backends`): a gain stage in
the worklet and a stand-in engine on the inference Worker, bundled by Vite. ``npm run examples``
type-checks and builds it, and the browser test renders its check page offline in Chromium and
Firefox, built and under Vite's dev server. Its ``main.ts``:

.. literalinclude:: ../../../web/examples/stage-and-engine/main.ts
   :language: ts

Raw Calls
---------

Every C entry is a method of ``anira.abi`` under its C name, its out parameters returned, its
``anira_error`` made for it:

.. code-block:: ts

   const config = anira.abi.anira_context_config_create()
   anira.abi.anira_context_config_set_threads(config, 2, anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF)
   const raw = anira.abi.anira_context_create(config)
   anira.abi.anira_context_config_destroy(config)

Records are read through ``layout`` (``layout.anira_tensor``, ...), and enums and defines come
from the package by their C names.
