Anira Web (TypeScript / WASM)
=============================

The Anira Web package is anira compiled to WebAssembly with Emscripten and a thin TypeScript
layer over its C ABI: the same ``anira_*`` functions a C host calls (:doc:`../usage`), as typed
methods generated from the ABI registry (:js:class:`AniraBindings`), plus the small Emscripten
companion ``anira/anira_em.h``. The page's main thread instantiates the module; anira starts
every other thread from C on the same shared memory, as natively it starts OS threads: the
context's inference threads are Wasm Workers, and a handler streams audio through a Wasm Audio
Worklet node whose render callback is C.

Three things carry over from a native host unchanged: the model and contract files
(``*.model.json``, ``*.contract.json``; :js:meth:`ModelConfig.fromUrl` fetches a model file the
way a native host opens it), the C ABI and its semantics, and the built-in ONNX Runtime engine.
What a page adds is TypeScript: processors and inference engines written against the package, which
the scopes that run them (the main thread, the inference Workers' module, the worklet's module)
define by name (:js:func:`~registry.defineProcessor`, :js:func:`~registry.defineEngine`), without rebuilding the
WebAssembly module. The context config is the one place that differs: its worker module, the ES
module every inference Worker imports first.

This section is a mix of hand-written guides and an auto-generated API reference.

.. toctree::
   :maxdepth: 2
   :caption: Anira Web

   installation_building
   basic_usage
   architecture
   audio_nodes
   custom_pre_post_processing
   custom_inference_backends
   reference/index
