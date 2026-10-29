Basic Usage
===========

.. note::
   **One config file for native and web.** The model and contract files of a native host load
   unchanged: ``await ModelConfig.fromUrl(anira, 'models/gain.model.json')`` parses the file
   (``anira_model_config_from_json``) and fetches the model file of every entry whose engine the
   module compiled in, relative to the file's URL, attaching it as bytes
   (``anira_model_config_set_model_bytes``); ``await Contract.fromUrl(anira,
   'models/gain.contract.json')`` reads a contract file, whose Hard geometry the page sets
   (``contract.setGeometry(128, 128, audioContext.sampleRate)``). The module's built-in ONNX
   Runtime engine runs the ``"onnxruntime"`` entries as natively; serve the model files beside
   the config the way the paths in it say.

This guide walks through the smallest end-to-end ``Anira Web`` setup: a stereo model run by
``onnxruntime-web`` (:js:class:`ONNXRuntimeWebBackend`) on an inference Worker, streamed through
the audio worklet. The configuration is built with the raw ``_anira_*`` exports, the same C
calls as in :doc:`../usage`; a typed TypeScript API over them follows. The steps:

1. instantiate the module and route the log,
2. a context, a model config and a pipeline with the JavaScript engine,
3. the handler and a Hard contract of the render quantum,
4. an inference Worker,
5. the audio worklet,
6. the stop and the teardown.

1. Bootstrap
------------

.. code-block:: typescript

   import {
     AniraWeb,
     callWithError,
     startLogPump,
     anira_budget_kind,
     anira_log_drain,
     anira_wait_strategy,
     anira_warmup_mode,
     ONNXRUNTIME_WEB_ENGINE_ID,
   } from '@anira-project/anira'

   const anira = await AniraWeb.create()
   const m = anira.module // the raw exports: every anira_* entry as _anira_*
   const heap = anira.heap // views over the shared memory, malloc/free, C strings

   anira.setLogHook() // every instance's records to the console
   const stopLogPump = startLogPump(m) // no drain thread on the web

   // A C out-parameter: a heap slot for one pointer, read after the call.
   const out = heap.malloc(4)
   const created = (call: (err: number) => number): number => {
     callWithError(heap, call) // throws an AniraError with the entry's message
     return heap.readU32(out)
   }

2. Context, Model Config and Pipeline
-------------------------------------

The context brings no pool threads on the web: the inference Workers of step 4 are its
threads. The model config below is the JSON form; ``_anira_model_config_*`` builds the same in
code. The entry names the JavaScript engine by its id, so the pipeline runs it as
``ANIRA_ENGINE_CUSTOM`` with that id; its path is the URL ``ONNXRuntimeWebBackend.init()``
fetches on the Worker.

.. code-block:: typescript

   const contextConfig = created((err) => m._anira_context_config_create(out, err))
   m._anira_context_config_set_threads(contextConfig, 0, anira_wait_strategy.ANIRA_WAIT_SPIN_BACKOFF)
   m._anira_context_config_set_log_drain(contextConfig, anira_log_drain.ANIRA_LOG_DRAIN_MANUAL, 10)
   const context = created((err) => m._anira_context_create(contextConfig, out, err))

   const modelJson = JSON.stringify({
     models: [{ engine: ONNXRUNTIME_WEB_ENGINE_ID, path: new URL('model.onnx', location.href).href }],
     inputs: [
       {
         name: 'audio_in',
         dtype: 'float32',
         role: 'streamed',
         axes: [['batch', 1], ['channel', 2], ['time', 512]],
         window: { min: 512, max: 512 },
       },
     ],
     outputs: [
       {
         name: 'audio_out',
         dtype: 'float32',
         role: 'streamed',
         axes: [['batch', 1], ['channel', 2], ['time', 512]],
         window: { min: 512, max: 512 },
       },
     ],
   })
   const model = heap.withCString(modelJson, (json) =>
     created((err) => m._anira_model_config_from_json(json, modelJson.length, 0, out, err))
   )

   const pipeline = created((err) => m._anira_pipeline_create(out, err))
   const variants = heap.malloc(4)
   heap.writeU32(variants, model)
   // NULL candidates: the default set, which holds every custom engine an entry names.
   callWithError(heap, (err) => m._anira_pipeline_add_inference(pipeline, variants, 1, 0, 0, err))

   // The engine object of the pipeline, run on every inference Worker (step 4).
   await anira.registerJSEngine(pipeline, {
     engineId: ONNXRUNTIME_WEB_ENGINE_ID,
     className: 'ONNXRuntimeWebBackend',
     modelConfigPtr: model,
   })

``registerJSEngine`` calls ``_anira_em_pipeline_register_js_engine`` (create the engine under
its id, add it to the pipeline) and has every Worker construct the class and await its
``init()``. Register before ``_anira_handler_create``: a handler copies the pipeline.

3. Handler and Contract
-----------------------

The worklet calls the handler once per render quantum, so the Hard contract declares that
block: ``block_min = block_max = 128`` at the context's sample rate. A call above the declared
block is refused; when the browser changes the quantum, re-prepare (step 5). A refused call
throws an :js:class:`AniraError` from ``callWithError`` with the entry's message.

.. code-block:: typescript

   const audioContext = new AudioContext({ sampleRate: 48000 })

   const handler = created((err) => m._anira_handler_create(context, pipeline, out, err))
   const contract = created((err) =>
     m._anira_contract_create_hard(128, 128, audioContext.sampleRate, out, err)
   )
   m._anira_contract_hard_set_budget(contract, anira_budget_kind.ANIRA_BUDGET_EXPLICIT, 2.0)
   // The default warm-up (UNTIL_STABLE) is refused by this runtime: none, or a fixed count.
   m._anira_contract_hard_set_warmup(contract, anira_warmup_mode.ANIRA_WARMUP_NONE, 0)
   callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))

4. An Inference Worker
----------------------

.. code-block:: typescript

   const worker = await anira.spinUpInferenceWorker(context)

The Worker gets its own stack and thread-local block, the registered JavaScript engine, and
then runs ``_anira_inference_thread_run_loop`` until stopped; while no job is queued it parks
on the wake word. Spin up more for more inferences in flight.

5. The Audio Worklet
--------------------

.. code-block:: typescript

   await anira.registerAudioWorkletForContext(audioContext)
   const node = await anira.configureAudioWorklet(audioContext, handler, {
     inputChannels: 2,
     outputChannels: 2,
     quantum: 128, // the declared block: the worklet builds its tensors for it now
     onQuantumChange: (quantum) => {
       m._anira_contract_hard_set_geometry(contract, quantum, quantum, audioContext.sampleRate)
       callWithError(heap, (err) => m._anira_handler_prepare(handler, contract, err))
     },
   })
   source.connect(node).connect(audioContext.destination)

The worklet instance builds planar host tensors of ``[2, 128]`` at configure and makes one
``_anira_handler_process`` per quantum, allocating nothing in ``process()``; the output is the
model's, delayed by the handler's latency (``_anira_handler_get_latency(handler, 0)``
samples). A missed quantum carries the miss policy's fill (BYPASS by default). When the
browser renders another quantum the worklet outputs silence and reports it once:
``onQuantumChange`` re-prepares the handler (resolve a promise when that is asynchronous),
then the worklet rebuilds its tensors for the new quantum, outside its ``process()``.

A control of the model, such as a gain, is a Static input: add it to the model config
(``{ name: 'gain', dtype: 'float32', role: 'static', axes: [['any', 1]] }`` in ``inputs``,
here slot 1), have the worklet serve it and write it from the main thread:

.. code-block:: typescript

   const node = await anira.configureAudioWorklet(audioContext, handler, {
     inputChannels: 2,
     outputChannels: 2,
     staticInputs: [1],
     modelConfigPtr: model,
   })
   node.setStaticInput(1, [1.0])
   slider.oninput = () => node.setStaticInput(1, [Number(slider.value)])

The worklet pushes a changed value before its next quantum's call, never torn
(:doc:`custom_audio_worklets`, "Static Inputs").

6. Stop and Teardown
--------------------

.. code-block:: typescript

   node.disconnect()
   await anira.disposeAudioWorklet(node) // the worklet frees its tensors, then its stack goes
   await worker.stop() // _stop, poll _has_exited, _destroy, 'destroyed', free the Worker's stack
   m._anira_handler_destroy(handler)
   m._anira_pipeline_destroy(pipeline)
   m._anira_contract_destroy(contract)
   m._anira_model_config_destroy(model)
   m._anira_context_destroy(context)
   m._anira_context_config_destroy(contextConfig)
   heap.free(variants)
   heap.free(out)
   stopLogPump()

``worker.stop()`` requests the exit (``_anira_inference_thread_stop``), waits until the
Worker has left its loop, which the main thread sees through
``_anira_inference_thread_has_exited`` (destroying the thread before that is refused), and
frees the Worker's stack once the Worker says it destroyed its engines.
``disposeAudioWorklet(node)`` comes before ``_anira_handler_destroy``: the worklet must stop
calling the handler first.

Running on the Built-in ONNX Runtime
------------------------------------

The module also carries ONNX Runtime itself (``ANIRA_ENGINE_ONNXRUNTIME`` on
``ANIRA_PROVIDER_CPU``), run on the inference Worker without a JavaScript engine. Its entry
names the engine and holds the model bytes, since the module cannot open a file: fetch the
model, copy it into the heap and add it with ``_anira_model_config_add_model_bytes``; skip
``registerJSEngine``. Everything else is as above.
