Custom Audio Worklets
=====================

The default worklet that ships with Anira Web handles the common case:
one streamable tensor, single audio I/O, default ``maxBufferSize``. As
soon as your model wants something more — multiple tensors, a different
processing buffer size, ``AudioParam`` integration, or a JS-side
pre/post-processor — you write your own worklet file by subclassing
:js:class:`AniraAudioWorkletBase`.

When You Need a Custom Worklet
------------------------------

* **Multi-tensor I/O.** Models with more than one streamable input or
  output tensor must call ``inferenceHandler.processMulti`` instead of
  ``process`` and marshal a ``float***`` pointer structure into WASM
  memory.
* **Custom ``maxBufferSize``.** When the model expects a larger
  processing block than the default (e.g. 1024 or 2048 samples), pass
  ``maxBufferSize`` through ``configureAudioWorklet``'s options.
* **AudioParam integration.** Web Audio's ``AudioParam`` system gives
  you sample-accurate, k-rate or a-rate parameter automation — but only
  inside a custom worklet that declares ``parameterDescriptors``.
* **Hosting a JSPrePostProcessor subclass.** Custom pre/post-processors
  must be instantiated on the audio thread; that requires a worklet
  file. See :doc:`custom_pre_post_processing`.

Skeleton
--------

A custom worklet file looks like this:

.. code-block:: typescript

   // audio-worklet.ts
   import {
     AniraAudioWorkletBase,
     type AniraWorkletState,
   } from '@anira-project/anira/workers/worklet-base'

   class MyWorklet extends AniraAudioWorkletBase {
     protected async onConfigured(state: AniraWorkletState) {
       // One-time setup: build per-tensor pointer structures with
       // buildMultiTensorPointers, construct JSPrePostProcessor subclasses, etc.
     }

     protected processAudioBlock(
       inputs: Float32Array[][],
       outputs: Float32Array[][],
       state: AniraWorkletState,
       bufferSize: number,
       parameters: Record<string, Float32Array>
     ): void {
       // Real-time path: copy inputs into WASM, call processMulti, copy outputs back.
     }
   }

   registerProcessor('my-worklet', MyWorklet)

And you register the file by URL and pass the matching
processor name to ``configureAudioWorklet``:

.. code-block:: typescript

   await aniraWeb.registerAudioWorkletForContext(
     audioContext,
     new URL('./audio-worklet.ts', import.meta.url)
   )

   const inferenceNode = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor,
     'my-worklet',
     { inputChannels: 2, outputChannels: 2 }
   )

AniraWorkletState
-----------------

``onConfigured`` and ``processAudioBlock`` both receive an
``AniraWorkletState`` object with everything you need to talk to WASM:

+----------------------------+---------------------------------------------------------------+
| Field                      | Description                                                   |
+============================+===============================================================+
| ``aniraWeb``               | The ``AniraWeb`` instance for this thread. Exposes            |
|                            | ``malloc``, ``getHeapU32``, ``getHeapF32``,                   |
|                            | ``getWasmInstance``, factories, etc.                          |
+----------------------------+---------------------------------------------------------------+
| ``inferenceHandler``       | The :js:class:`InferenceHandler` proxy. Call                  |
|                            | ``processMulti`` (or ``process``) on it from                  |
|                            | ``processAudioBlock``.                                        |
+----------------------------+---------------------------------------------------------------+
| ``prePostProcessorPtr``    | Raw pointer to the C++ ``PrePostProcessor`` instance, used    |
|                            | to construct a :js:class:`JSPrePostProcessor` subclass via    |
|                            | ``createFromPointer``.                                        |
+----------------------------+---------------------------------------------------------------+
| ``inputBufferPtr``         | ``float**`` pointer arrays — one entry per channel — already  |
| ``outputBufferPtr``        | laid out contiguously by ``configureAudioWorklet``. Pass to   |
|                            | ``inferenceHandler.process`` for single-tensor models, or     |
|                            | hand to ``buildMultiTensorPointers`` for multi-tensor splits. |
+----------------------------+---------------------------------------------------------------+
| ``inputDataBuffer``        | WASM heap offsets for the per-channel scratch buffers the     |
| ``outputDataBuffer``       | channel pointers above point into. Each channel occupies      |
|                            | ``maxBufferSize * 4`` bytes, laid out contiguously.           |
+----------------------------+---------------------------------------------------------------+
| ``inputChannelViews``      | ``Float32Array`` views over the per-channel slices of the     |
| ``outputChannelViews``     | scratch buffers. Use these to copy audio in and out without   |
|                            | rebuilding views every block.                                 |
+----------------------------+---------------------------------------------------------------+
| ``ioConfig``               | ``{ maxBufferSize, inputChannels, outputChannels,             |
|                            | inputNodeIndex, outputNodeIndex }``.                          |
+----------------------------+---------------------------------------------------------------+
| ``wasmMemory``             | The shared ``WebAssembly.Memory``. Useful when you need a     |
|                            | typed-array view over a custom region of the heap.            |
+----------------------------+---------------------------------------------------------------+

Helpers on the Base Class
-------------------------

:js:class:`AniraAudioWorkletBase` provides a few helpers so subclasses
rarely have to touch raw heap offsets:

+------------------------------------------------+------------------------------------------------+
| Helper                                         | Purpose                                        |
+================================================+================================================+
| ``copyAudioInputsToChannels(inputNode, state,  | Copy a slice of the host's input channels      |
| bufferSize, channelOffset?, channelCount?)``   | into a contiguous range of                     |
|                                                | ``inputChannelViews``. Missing source channels |
|                                                | are zero-filled.                               |
+------------------------------------------------+------------------------------------------------+
| ``copyAudioOutputsFromChannels(outputNode,     | Copy a contiguous range of                     |
| state, samplesProcessed, channelOffset?,       | ``outputChannelViews`` back into the host's    |
| channelCount?)``                               | output channels.                               |
+------------------------------------------------+------------------------------------------------+
| ``buildMultiTensorPointers(direction,          | Slice the existing ``inputBufferPtr`` /        |
| channelsPerTensor)``                           | ``outputBufferPtr`` into the ``float***``      |
|                                                | structure ``processMulti`` expects, plus a     |
|                                                | ``size_t[numTensors]`` array for the           |
|                                                | per-tensor sample counts. Returns              |
|                                                | ``{ tensorPtrs, numSamplesPtr }``.             |
+------------------------------------------------+------------------------------------------------+

In Practice: Multi-Tensor I/O
-----------------------------

The streaming-gain-stereo demo runs a model with two streamable
tensors: a stereo audio tensor (2 channels) and a single-channel gain
tensor driven by an ``AudioParam``. Because there is more than one
tensor, the worklet has to call ``processMulti`` — which expects
``float***`` for each side (tensor → channels → samples) plus matching
``size_t*`` arrays of per-tensor sample counts.

The base class already lays out the channel pointers contiguously
behind ``inputBufferPtr`` / ``outputBufferPtr``;
``buildMultiTensorPointers`` only needs the per-tensor channel split to
produce the ``processMulti`` arguments and the per-tensor sample count
buffer:

.. code-block:: typescript

   const AUDIO_CHANNELS = 2
   const GAIN_CHANNELS = 1
   const NUM_TENSORS = 2

   class StreamingGainStereo extends AniraAudioWorkletBase {
     private inputTensors  = { tensorPtrs: 0, numSamplesPtr: 0 }
     private outputTensors = { tensorPtrs: 0, numSamplesPtr: 0 }

     static get parameterDescriptors() {
       return [{ name: 'gain', defaultValue: 1.0, minValue: 0.0, maxValue: 2.0 }]
     }

     protected async onConfigured(_state: AniraWorkletState) {
       this.inputTensors  = this.buildMultiTensorPointers('input',  [AUDIO_CHANNELS, GAIN_CHANNELS])
       this.outputTensors = this.buildMultiTensorPointers('output', [AUDIO_CHANNELS, GAIN_CHANNELS])
     }
   }

In ``processAudioBlock`` use the input/output copy helpers, fill the
gain channel from the ``AudioParam`` (delivered as the fifth argument),
write the per-tensor sample counts, and call ``processMulti``:

.. code-block:: typescript

   protected processAudioBlock(
     inputs: Float32Array[][],
     outputs: Float32Array[][],
     state: AniraWorkletState,
     bufferSize: number,
     parameters: Record<string, Float32Array>
   ): void {
     const { inferenceHandler, ioConfig, inputChannelViews } = state
     const heapU32 = state.aniraWeb.getHeapU32()

     const inputNode  = inputs[ioConfig.inputNodeIndex]
     const outputNode = outputs[ioConfig.outputNodeIndex]

     // Stereo audio into channels 0–1 (the audio tensor).
     this.copyAudioInputsToChannels(inputNode, state, bufferSize, 0, AUDIO_CHANNELS)

     // Channel 2 (the gain tensor) is driven by the AudioParam.
     const gainParam = parameters.gain
     const gainView  = inputChannelViews[AUDIO_CHANNELS]
     if (gainParam.length === 1) {
       gainView.fill(gainParam[0], 0, bufferSize)
     } else {
       gainView.set(gainParam.subarray(0, bufferSize), 0)
     }

     // Both tensors run with the same per-quantum sample count.
     for (let i = 0; i < NUM_TENSORS; i++) {
       heapU32[this.inputTensors.numSamplesPtr  / 4 + i] = bufferSize
       heapU32[this.outputTensors.numSamplesPtr / 4 + i] = bufferSize
     }

     const resultPtr = inferenceHandler.processMulti(
       this.inputTensors.tensorPtrs,
       this.inputTensors.numSamplesPtr,
       this.outputTensors.tensorPtrs,
       this.outputTensors.numSamplesPtr
     )
     const samplesProcessed = heapU32[resultPtr / 4]

     // Only the audio tensor is routed back to the Web Audio output.
     this.copyAudioOutputsFromChannels(outputNode, state, samplesProcessed, 0, AUDIO_CHANNELS)
   }

The ``AudioParam`` is declared in ``parameterDescriptors`` (as above)
and arrives directly in ``processAudioBlock``'s ``parameters`` argument
— no manual ``process`` override is needed. The ``inferenceNode``
returned by ``configureAudioWorklet`` exposes the parameter on the JS
side via ``inferenceNode.parameters.get('gain')``.

.. note::
   The gain tensor is exposed to users as an ``AudioParam``, not as an
   audio input — but on the WASM side, anira expects every tensor's
   data to live in the contiguous channel layout that the base class
   allocates behind ``inputBufferPtr``. So the worklet allocates a
   third internal channel as scratch space and materialises the
   ``parameters.gain`` values into it on every block. The surrounding
   Web Audio node itself stays stereo by passing
   ``audioWorkletNodeOptions: { channelCount: 2, outputChannelCount: [2] }``
   in the ``configureAudioWorklet`` options:

   .. code-block:: typescript

      const inferenceNode = await aniraWeb.configureAudioWorklet(
        audioContext,
        inferenceHandler,
        ppProcessor,
        'streaming-gain-stereo',
        {
          inputChannels: 3,   // 2 audio + 1 scratch for the gain tensor
          outputChannels: 3,
          audioWorkletNodeOptions: {
            channelCount: 2,
            outputChannelCount: [2],
          },
        }
      )

In Practice: Custom maxBufferSize
---------------------------------

GuitarLSTM expects 2048-sample blocks, far larger than the 128 samples
Web Audio delivers per ``process`` call. ``maxBufferSize`` in the
``configureAudioWorklet`` options propagates into ``ioConfig`` so the
worklet base class allocates buffers of the right size and accumulates
host blocks until the model can run:

.. code-block:: typescript

   await aniraWeb.registerAudioWorkletForContext(
     audioContext,
     new URL('./audio-worklet.ts', import.meta.url)
   )
   const inferenceNode = await aniraWeb.configureAudioWorklet(
     audioContext,
     inferenceHandler,
     ppProcessor,
     'guitar-lstm',
     { inputChannels: 1, outputChannels: 1, maxBufferSize: 2048 }
   )

The ``HostConfig`` passed to ``inferenceHandler.prepare`` should still
match Web Audio's per-callback buffer size (``128``); ``maxBufferSize``
is anira's processing-side block size, not the host-side one.
