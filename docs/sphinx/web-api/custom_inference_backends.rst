Custom Inference Backends
=========================

``Anira Web`` covers most use cases with the two engines it ships:
the WASM-side ONNX Runtime (the default path in :doc:`basic_usage`)
and ``onnxruntime-web`` on the JS side (see
:ref:`run-inference-in-javascript`). Reach for a custom backend when
neither fits — for example, when you want to run the model through a
different JS runtime, drive a GPU directly, or wire up a stub for
testing.

A custom backend is a JS class that anira's inference worker invokes
in place of the bundled engine. Writing one takes three steps:

1. Subclass :js:class:`JSBackendBase` and override ``process``.
2. Bundle that subclass into a custom inference worker so the worker
   knows how to construct it.
3. Spin up that custom worker and wire the backend into
   :js:class:`InferenceHandler` via ``InferenceBackend.CUSTOM``.

The `js-copying demo <https://anira-project.github.io/anira-web-example>`_ walks through all three
with a passthrough ``JSCopyBackend``; we'll use it as the running
example below.

Step 1: Implement the Backend
-----------------------------

Subclass :js:class:`JSBackendBase` and override
``process(inputVecPtr, outputVecPtr)``. Anira hands you two pointers
into WASM memory — one to a ``VectorBufferF`` of input tensors, one to
outputs — and expects you to populate the outputs in place. Use
``wrapPointer`` to view the WASM structures as TypeScript objects.

.. code-block:: typescript

   // misc/JSCopyBackend.ts
   import { JSBackendBase, BufferF, VectorBufferF } from '@anira-project/anira'

   export class JSCopyBackend extends JSBackendBase {
     override process(inputVecPtr: number, outputVecPtr: number): void {
       const heapF32 = this.wasmInstance.HEAPF32
       const inputVec  = this.wrapPointer(VectorBufferF, inputVecPtr)
       const outputVec = this.wrapPointer(VectorBufferF, outputVecPtr)

       const tensors = Math.min(inputVec.size(), outputVec.size())
       for (let t = 0; t < tensors; t++) {
         const inputBuffer  = this.wrapPointer(BufferF, inputVec.get(t))
         const outputBuffer = this.wrapPointer(BufferF, outputVec.get(t))

         const channels      = inputBuffer.getNumChannels()
         const inputSamples  = inputBuffer.getNumSamples()
         const outputSamples = outputBuffer.getNumSamples()
         const sampleDiff    = inputSamples - outputSamples

         for (let ch = 0; ch < channels; ch++) {
           const readOffset  = inputBuffer.getReadPointer(ch) >> 2
           const writeOffset = outputBuffer.getWritePointer(ch) >> 2
           for (let i = 0; i < outputSamples; i++) {
             heapF32[writeOffset + i] = heapF32[readOffset + i + sampleDiff]
           }
         }
       }
     }
   }

Step 2: Bundle Into a Custom Inference Worker
---------------------------------------------

The default inference worker that ``spinUpInferenceWorker()`` spawns
doesn't know about your backend class — it only knows the built-ins.
You ship a custom worker file that hands your subclass to anira's
worker runtime:

.. code-block:: typescript

   // customInferenceWorker.ts
   import { setupInferenceWorker } from '@anira-project/anira'
   import { JSCopyBackend } from '../misc/JSCopyBackend'

   setupInferenceWorker({ JSCopyBackend })

This is a one-line file: ``setupInferenceWorker`` runs anira's worker
loop and registers the constructors you pass it, so the worker can
instantiate the right class when ``registerProcessor`` is called from
the main thread.

Step 3: Wire It Up
------------------

On the main thread, point ``spinUpInferenceWorker`` at the custom
worker file, then follow the usual setup with three additions: the
model is declared as ``InferenceBackend.CUSTOM``, the backend is
instantiated and registered with ``aniraWeb``, and the same instance
is handed to :js:class:`InferenceHandler` as a third argument.

.. code-block:: typescript

   const customInferenceWorkerUrl = new URL('./customInferenceWorker.ts', import.meta.url)

   const aniraWeb = await AniraWeb.create()
   await aniraWeb.spinUpInferenceWorker(customInferenceWorkerUrl)

   const vectorModelData = aniraWeb.VectorModelData([
     aniraWeb.ModelData(modelBuffer, aniraWeb.InferenceBackend.CUSTOM),
   ])

   // Build inferenceConfig, processingSpec, ppProcessor as in basic_usage...

   const jsCopyBackend = new JSCopyBackend(aniraWeb.getWasmInstance(), inferenceConfig)
   await aniraWeb.registerProcessor(jsCopyBackend, 'JSCopyBackend')

   const inferenceHandler = aniraWeb.InferenceHandler(
     ppProcessor, inferenceConfig, jsCopyBackend
   )
   inferenceHandler.setInferenceBackend(aniraWeb.InferenceBackend.CUSTOM)

``registerProcessor`` ships the backend reference over to the
inference worker so the WASM-side dispatch can call back into it.

.. note::
   ``ModelData`` accepts either an ``ArrayBuffer`` (the binary form,
   shown above) or a URL string. If you pass a URL, anira hands the
   string to your backend as-is — your backend decides how to load
   it. The built-in :js:class:`ONNXRuntimeWebBackend` uses this to
   fetch the model itself; ``modelData.isBinary()`` tells you which
   form is in play:

   .. code-block:: typescript

      if (modelData.isBinary()) {
        const ptr = modelData.getDataPtr()
        const size = modelData.getSize()
        modelBytes = new Uint8Array(wasm.HEAPU32.buffer, ptr, size).slice()
      } else {
        const pathBytes = new Uint8Array(
          wasm.HEAPU32.buffer,
          modelData.getDataPtr(),
          modelData.getSize()
        ).slice()
        const modelUrl = new TextDecoder().decode(pathBytes)
        modelBytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer())
      }

.. warning::
   The custom backend runs on the inference worker thread, not on the
   audio worklet thread, so it doesn't block the real-time callback
   directly. It still has to finish under the
   ``max_inference_time_ms`` set in :js:class:`InferenceConfig`,
   otherwise anira will fall back to the previous block's output and
   you will hear dropouts.

Sanity Check: JSBackendBase as a Passthrough
--------------------------------------------

If you want to verify the JS-bridge plumbing before committing to a
real backend, instantiate :js:class:`JSBackendBase` directly. It
ships with a trivial WASM-side passthrough ``process`` and a JS hook
that fires for every block — and because it's a built-in, it doesn't
need a custom inference worker:

.. code-block:: typescript

   const jsBackendBase = aniraWeb.JSBackendBase(inferenceConfig)
   await aniraWeb.registerProcessor(jsBackendBase, 'JSBackendBase')

   const inferenceHandler = aniraWeb.InferenceHandler(
     ppProcessor, inferenceConfig, jsBackendBase
   )
   inferenceHandler.setInferenceBackend(aniraWeb.InferenceBackend.CUSTOM)

The `js-callback demo <https://anira-project.github.io/anira-web-example>`_ does exactly this.
