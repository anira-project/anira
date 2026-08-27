Custom Pre- and Post-Processing
===============================

When the default :js:class:`PrePostProcessor` doesn't cover what your
model needs — windowing, normalization, parameter smoothing, custom
multi-tensor packing — you subclass :js:class:`JSPrePostProcessor` and
override ``preProcess`` and/or ``postProcess`` in JavaScript. The
`guitar-lstm and steerable-nafx demos <https://anira-project.github.io/anira-web-example>`_
are live examples of this pattern that you can run in the browser.

.. note::
   This page builds on :doc:`custom_audio_worklets`. A custom
   pre/post-processor must be instantiated on the audio worklet thread,
   so you always need a small custom worklet file.

Two-Step Setup
--------------

The processor lives in two places. On the **main thread**, you create
a :js:class:`JSPrePostProcessor` from the factory — this tells anira
that pre/post-processing will be handled in JavaScript:

.. code-block:: typescript

   const ppProcessor = aniraWeb.JSPrePostProcessor(inferenceConfig)
   const inferenceHandler = aniraWeb.InferenceHandler(ppProcessor, inferenceConfig)

On the **audio worklet thread**, you reconstruct the C++ processor as
your subclass and register it. ``createFromPointer`` wraps the existing
C++ instance (``state.prePostProcessorPtr``) so JS overrides hook into
the same object:

.. code-block:: typescript

   // audio-worklet.ts
   import {
     AniraAudioWorkletBase,
     type AniraWorkletState,
   } from '@anira-project/anira/workers/worklet-base'
   import { JSPrePostProcessor } from '@anira-project/anira'

   class MyPrePostProcessor extends JSPrePostProcessor {
     // overrides go here
   }

   class MyWorklet extends AniraAudioWorkletBase {
     protected async onConfigured(state: AniraWorkletState) {
       const { aniraWeb, prePostProcessorPtr } = state

       const ppProcessor = MyPrePostProcessor.createFromPointer(
         aniraWeb.getWasmInstance(),
         prePostProcessorPtr
       )

       this.prePostRegistry.set(prePostProcessorPtr, ppProcessor)
     }
   }

   registerProcessor('my-worklet', MyWorklet)

What You Can Override
---------------------

:js:class:`JSPrePostProcessor` exposes the same hooks as the C++ class:

+-------------------------------------------------+-----------------------------------------------------------------+
| Method                                          | When it runs                                                    |
+=================================================+=================================================================+
| ``preProcess(ringBuffers, buffers, backend)``   | Before each inference call, on the **audio worklet**. Pull      |
|                                                 | samples from the input ring buffers into the model's input      |
|                                                 | tensors.                                                        |
+-------------------------------------------------+-----------------------------------------------------------------+
| ``postProcess(buffers, ringBuffers, backend)``  | After each inference call, on the **audio worklet**. Push the   |
|                                                 | model's output tensors into the output ring buffers.            |
+-------------------------------------------------+-----------------------------------------------------------------+
| ``beforeInference(buffers, backend)``           | On the **inference worker**, immediately before the backend     |
|                                                 | runs. Patch input tensors with data that must reflect the       |
|                                                 | previous inference (e.g. recurrent state).                      |
+-------------------------------------------------+-----------------------------------------------------------------+
| ``afterInference(buffers, backend)``            | On the **inference worker**, immediately after the backend      |
|                                                 | runs. Capture output tensors that must feed the next inference  |
|                                                 | (e.g. recurrent state).                                         |
+-------------------------------------------------+-----------------------------------------------------------------+

``preProcess`` / ``postProcess`` and ``beforeInference`` / ``afterInference``
run on **different threads**, so they are registered in different places (see
:ref:`inference-hooks` below). ``preProcess`` / ``postProcess`` each take two
vectors — the ring buffers and the tensor buffers; ``beforeInference`` /
``afterInference`` take a single ``VectorBufferF`` (the input tensors before the
run, the output tensors after it).

Inside an override you can read and write non-streamable tensor values
with ``getInput`` / ``setInput`` / ``getOutput`` / ``setOutput`` (same
semantics as the C++ class — see :doc:`../usage`), and call into raw
WASM exports through ``this.wasmInstance`` for ring-buffer manipulation.

In Practice: Gain Clamp
-----------------------

This is the smallest possible custom pre-processor: it clamps the gain
parameter to ``[0, 1]`` before passing it to the C++ pre-processing.

.. code-block:: typescript

   // audio-worklet.ts
   import {
     AniraAudioWorkletBase,
     type AniraWorkletState,
   } from '@anira-project/anira/workers/worklet-base'
   import {
     JSPrePostProcessor,
     type PossiblePointer,
     type VectorBufferF,
     type VectorRingBuffer,
   } from '@anira-project/anira'

   class GainClampPrePostProcessor extends JSPrePostProcessor {
     override preProcess(
       ringBuffers: PossiblePointer<VectorRingBuffer>,
       buffers: PossiblePointer<VectorBufferF>,
       backend: number
     ): void {
       const gain = this.getInput(0, 1)
       this.setInput(Math.min(1.0, gain), 0, 1)
       super.preProcess(ringBuffers, buffers, backend)
     }
   }

   class PrePostProcessorWorklet extends AniraAudioWorkletBase {
     protected async onConfigured(state: AniraWorkletState) {
       const { aniraWeb, prePostProcessorPtr } = state
       const ppProcessor = GainClampPrePostProcessor.createFromPointer(
         aniraWeb.getWasmInstance(),
         prePostProcessorPtr
       )
       this.prePostRegistry.set(prePostProcessorPtr, ppProcessor)
     }
   }

   registerProcessor('pre-post-processors', PrePostProcessorWorklet)

The setup is identical to the one on the :doc:`basic_usage` page
except that ``PrePostProcessor`` becomes ``JSPrePostProcessor`` and
``configureAudioWorklet`` is given the processor name:

.. code-block:: typescript

   const ppProcessor = aniraWeb.JSPrePostProcessor(inferenceConfig)
   ppProcessor.setInput(1, 0, 1)

   await aniraWeb.registerAudioWorkletForContext(
     audioContext,
     new URL('./audio-worklet.ts', import.meta.url)
   )
   const inferenceNode = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor,
     'pre-post-processors'
   )

   // The slider sets the raw gain on the main thread; the worklet thread
   // clamps it on every block via the override above.
   gainSlider.oninput = () => {
     ppProcessor.setInput(parseFloat(gainSlider.value), 0, 1)
   }

.. _inference-hooks:

Inference-Thread Hooks (Stateful Models)
----------------------------------------

``beforeInference`` and ``afterInference`` do **not** run on the audio
worklet. They fire on the inference worker, wrapped tightly around the
backend's forward pass — ``beforeInference`` right before it,
``afterInference`` right after and, with ``session_exclusive_processor =
true``, before the next inference is dispatched. That makes them the correct
place to splice cross-inference state such as a recurrent model's hidden
state, which ``preProcess`` cannot do reliably: ``preProcess`` fills input
tensors at submission time, so once several inferences are queued its state is
already stale. See :cpp:func:`anira::PrePostProcessor::before_inference` for
the full rationale.

Because the hooks run on the inference worker, the subclass has to be
reconstructed **there**, not only on the worklet — two steps beyond the
worklet registration above.

**1. Teach the inference worker about the subclass.** Ship a custom inference
worker file and pass the subclass in the second argument of
``setupInferenceWorker`` (the first is the custom-backend map from
:doc:`custom_inference_backends`):

.. code-block:: typescript

   // stateful-inference-worker.ts
   import { setupInferenceWorker } from '@anira-project/anira'
   import { StatefulPrePostProcessor } from './stateful-pre-post-processor'

   setupInferenceWorker({}, { StatefulPrePostProcessor })

where the subclass overrides the inference hooks:

.. code-block:: typescript

   // stateful-pre-post-processor.ts
   import {
     JSPrePostProcessor,
     type PossiblePointer,
     type VectorBufferF,
   } from '@anira-project/anira'

   export class StatefulPrePostProcessor extends JSPrePostProcessor {
     override beforeInference(
       buffers: PossiblePointer<VectorBufferF>,
       backend: number
     ): void {
       // write the state captured last time into the model's state input tensor
     }
     override afterInference(
       buffers: PossiblePointer<VectorBufferF>,
       backend: number
     ): void {
       // read the model's state output tensor and stash it for next time
     }
   }

**2. Register the processor on the worker(s).** On the main thread, register
the processor by class name and spin up the worker with your custom worker
file:

.. code-block:: typescript

   const ppProcessor = aniraWeb.JSPrePostProcessor(inferenceConfig)

   await aniraWeb.registerPrePostProcessor(ppProcessor, 'StatefulPrePostProcessor')
   await aniraWeb.spinUpInferenceWorker(
     new URL('./stateful-inference-worker.ts', import.meta.url)
   )

``registerPrePostProcessor`` forwards the processor to every inference worker
already running and replays it on any spun up later, so the two calls can go
in either order. Registration also *arms* the C++ hooks: until a processor is
registered on a worker, ``beforeInference`` / ``afterInference`` short-circuit
to the base no-op without crossing into JS, so a :js:class:`JSPrePostProcessor`
used only for ``preProcess`` / ``postProcess`` pays nothing on the inference
thread.

The same ``ppProcessor`` still goes to ``configureAudioWorklet`` and, if it
also customizes ``preProcess`` / ``postProcess``, is registered on the worklet
as shown above — the two registrations are independent and both drive the one
shared C++ object.

.. note::
   The class-name string passed to ``registerPrePostProcessor`` must match a
   key in the map given to ``setupInferenceWorker`` — that is how the worker
   knows which subclass to reconstruct around the shared C++ pointer. If it is
   omitted or unknown, the worker falls back to the base
   :js:class:`JSPrePostProcessor`, whose inference hooks are no-ops.

Pointer Arguments
-----------------

``preProcess`` and ``postProcess`` receive ``PossiblePointer<...>``
arguments — either wrapper instances or raw WASM heap addresses. The
:doc:`architecture` page covers the helpers (``resolvePtr``,
``getPointer``, ``wrapPointer``) in full; the short version is to call
``resolvePtr`` to get a numeric pointer, then call the exported WASM
functions on ``this.wasmInstance`` (e.g. ``_vector_ring_buffer_get``,
``_vector_buffer_f_get``,
``_prepostprocessor_pop_samples_from_buffer_window``) to manipulate
buffers in place. The guitar-lstm and steerable-nafx demos show this pattern applied to real windowing
logic.

.. note::
   Calling the underscore-prefixed exports directly looks unusual at
   first, but it's a deliberate performance escape hatch. Wrapping a
   pointer into a TypeScript class (``wrapPointer(BufferF, ptr)``,
   etc.) allocates a JS object — fine on the main thread, but
   unwanted allocation pressure in real-time code that runs every
   audio block. The raw exports skip the wrapper entirely, at the
   cost of dealing in numeric pointers. Reach for them in real-time
   paths; stick with the wrappers everywhere else.

.. warning::
   When a model needs one overlapping window **per batch element**
   (input shape ``[num_batches, ..., window_size]``, as in the
   guitar-lstm/HybridNN demo), do **not** loop in JavaScript calling
   ``_prepostprocessor_pop_samples_from_buffer_window_offset`` once per
   batch. Each call crosses the JS↔WASM boundary, and for large batches
   that per-element overhead runs on the audio render thread and can blow
   the render-quantum budget, underrunning the whole ``AudioContext``.
   Use the batched export instead, which runs the loop in native code in
   a single call::

     // offset stride is (numNewSamples + numOldSamples) per batch
     this.wasmInstance._prepostprocessor_pop_samples_from_buffer_batched(
       this.getPointer(), ringBuffer0, buffer0,
       numNewSamples, numOldSamples, /*offset*/ 0, numBatches)

   or, off the real-time path, the wrapper overload
   ``ppProcessor.popSamplesFromBuffer(ringBuffer, buffer, numNewSamples,
   numOldSamples, offset, numBatches)``.
