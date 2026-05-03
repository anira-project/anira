Custom Pre- and Post-Processing
===============================

When the default :js:class:`PrePostProcessor` doesn't cover what your
model needs — windowing, normalization, parameter smoothing, custom
multi-tensor packing — you subclass :js:class:`JSPrePostProcessor` and
override ``preProcess`` and/or ``postProcess`` in JavaScript.

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
   } from 'anira-web/workers/worklet-base'
   import { JSPrePostProcessor } from 'anira-web'

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

       aniraWeb.registerPrePostProcessor(ppProcessor)
     }
   }

   registerProcessor('my-worklet', MyWorklet)

What You Can Override
---------------------

:js:class:`JSPrePostProcessor` exposes the same hooks as the C++ class:

+-------------------------------------------------+-----------------------------------------------+
| Method                                          | When it runs                                  |
+=================================================+===============================================+
| ``preProcess(ringBuffers, buffers, backend)``   | Before each inference call. Use it to pull    |
|                                                 | samples from the input ring buffers into the  |
|                                                 | model's input tensors.                        |
+-------------------------------------------------+-----------------------------------------------+
| ``postProcess(buffers, ringBuffers, backend)``  | After each inference call. Use it to push     |
|                                                 | the model's output tensors into the output    |
|                                                 | ring buffers.                                 |
+-------------------------------------------------+-----------------------------------------------+

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
   } from 'anira-web/workers/worklet-base'
   import {
     JSPrePostProcessor,
     type PossiblePointer,
     type VectorBufferF,
     type VectorRingBuffer,
   } from 'anira-web'

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
       aniraWeb.registerPrePostProcessor(ppProcessor)
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
buffers in place. The guitar-lstm and steerable-nafx demos in
``anira-web-demo/src/`` show this pattern applied to real windowing
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
