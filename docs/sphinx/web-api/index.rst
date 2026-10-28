Anira Web (TypeScript / WASM)
=============================

The Anira Web package is anira compiled to WebAssembly with Emscripten and the
TypeScript runtime around it. JavaScript reaches anira through the module's C ABI, the
same ``anira_*`` functions a C host calls (:doc:`../usage`), exported as ``_anira_*``,
plus the small Emscripten companion ``anira/anira_em.h``: its trampolines let a stage,
an inference engine and the log run in JavaScript. The TypeScript layer instantiates the
module on one shared memory in every browser thread that needs it (the main thread, the
inference Workers, the audio worklet), runs the inference threads in Workers and streams
audio through a handler in the worklet.

In this pre-release configs, pipelines, contexts and handlers are built with the raw
``_anira_*`` calls; a typed TypeScript API over them follows. This section is a mix of
hand-written guides and an auto-generated API reference.

.. toctree::
   :maxdepth: 2
   :caption: Anira Web

   installation_building
   basic_usage
   architecture
   custom_audio_worklets
   custom_pre_post_processing
   custom_inference_backends
   reference/index
