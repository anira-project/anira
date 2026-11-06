.. _migration-web:

Anira Web: from the 2.x TypeScript classes
==========================================

The 3.x package (``@anira-project/anira``) no longer wraps the 2.x C++ classes: the module exports
the C ABI (``_anira_*``) and ``anira/anira_em.h`` (``_anira_em_*``), the package is a thin typed layer
over them (:doc:`../web-api/basic_usage` has the whole setup), and anira starts the page's other
threads itself, from C: the context's inference threads are Wasm Workers, the audio path is a Wasm
Audio Worklet node whose render callback is C.

- ``new AniraWeb()`` / ``AniraWeb.create()`` becomes ``await Anira.create()``; its ``abi`` is the C
  ABI, one typed method per entry (``anira.abi.anira_context_create(config)``), which throws an
  ``AniraError`` on a failed status. ``malloc``, ``free``, ``getHeapF32`` and the other heap helpers
  become ``anira.module`` (``_malloc``, ``HEAPF32``, ...); ``getAniraVersion`` is ``anira.version``,
  ``drainAniraLog`` ``anira.drainLog()``.
- ``ModelData``, ``TensorShape``, ``ProcessingSpec`` and ``InferenceConfig`` become one model config:
  ``await ModelConfig.fromUrl(anira, url)`` reads the 3.x model file a native host reads (a 2.x JSON
  document too, :ref:`migration-json`) and fetches its model files.
- ``HostConfig(bufferSize, sampleRate)`` becomes the Hard contract:
  ``ContractHandle.hard(anira, { hostBlockMin: g.quantum, hostBlockMax: g.burst, hostRate:
  context.sampleRate })`` with ``g = await measureRenderGeometry(context)``, or a contract file
  (``ContractHandle.fromUrl``, then ``hardHostBlocks``): the render quantum is the smallest block,
  the context's burst of quanta the largest. The budget is measured unless ``budgetMs`` sets one;
  ``budgetMs`` with ``warmup: ANIRA_WARMUP_NONE`` is an explicit budget without a warm-up pass. A
  running node changes its host blocks through ``audioNode.prepare(contract)``.
- ``InferenceHandler(pp, config)``, ``setInferenceBackend`` and ``prepare`` become
  ``Pipeline.create(anira).addInference(model)``, ``InferenceHandler.create(context, pipeline)`` and
  ``handler.prepare(contract)``; a context (``Context.create(anira, { threads })``) is the core and
  its pool.
- ``inferenceHandler.process(...)`` in a worklet becomes the handler's audio node, ``await
  handler.createAudioNode(audioContext, { params, staticInputs, waitMs })``: the render callback is
  C; ``setInput`` on a Static tensor becomes the node's AudioParam (scalars) or
  ``audioNode.setStaticInput`` (tensors). ``BufferF``, ``RingBuffer`` and the ``Vector*`` classes
  have no successor: tensors are heap memory described by ``anira_tensor`` records. The node streams
  the model's Streamed slots (``inputSlots`` / ``outputSlots``, :doc:`../web-api/audio_nodes`).
- ``JSPrePostProcessor`` and ``registerPrePostProcessor`` become ``defineProcessor(name, factory)`` in
  every scope that runs one of the processor's phases (the page, the worklet's module) and
  ``pipeline.addProcessor(name, { phases, flags })``: the method names stay (``preProcess``,
  ``postProcess``, ``beforeInference``, ``afterInference``), and they receive the
  ``anira_processor_ctx`` pointer instead of buffer vectors.
- ``JSBackendBase`` and ``registerProcessor(backend, className)`` become ``defineEngine(id,
  factory)`` in the context's worker module and ``pipeline.addEngine(id, { flags })``; ``process(ctx,
  prepared)`` reads the ``anira_engine_ctx`` (``engineTensors``), and ``InferenceBackend.CUSTOM``
  becomes the engine id in the model entry. The ONNX Runtime Web backend is ``onnxRuntimeWeb({ model
  })`` from ``@anira-project/anira/engines/onnxruntime-web``; the module's built-in ONNX Runtime
  engine needs no JavaScript.
- ``spinUpInferenceWorker()`` goes: the context's ``threads`` (and ``workerModule``) start the Workers
  at ``Context.create``, and ``await context.ready()`` waits until they run, which a prepare that
  warms up (the default) needs first.
