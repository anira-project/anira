Class AniraAudioWorkletBase
===========================

.. note::
   This class is exported from the package's audio-worklet subpath:
   ``import { AniraAudioWorkletBase, type AniraWorkletState } from 'anira-web/workers/worklet-base'``.
   The subpath isolates the worklet bundle so the main entry point
   stays free of references to ``AudioWorkletProcessor``, which only
   exists in ``AudioWorkletGlobalScope``.

   See :doc:`../../custom_audio_worklets` for usage
   guidance and worked examples.

.. js:class:: AniraAudioWorkletBase extends AudioWorkletProcessor

   Base class for custom audio worklets that drive an
   :js:class:`InferenceHandler`. Subclasses override
   :js:meth:`onConfigured` for one-time setup and
   :js:meth:`processAudioBlock` for the real-time loop. The class is
   wired to the main thread by ``aniraWeb.configureAudioWorklet`` —
   the configure handshake populates ``this.aniraState`` and then
   invokes ``onConfigured``.

   .. js:attribute:: aniraState
      :type: AniraWorkletState | null

      The configuration handed in by the main thread once the
      configure handshake completes. ``null`` until the first
      ``configure`` message is processed.

   .. js:method:: constructor(options?)

      :param AudioWorkletNodeOptions options: Forwarded to the
         ``AudioWorkletProcessor`` super-constructor. Most subclasses
         do not need to override the constructor.

   .. js:method:: onConfigured(state)

      Override hook for one-time setup, invoked once after the
      configure handshake completes and before the first
      ``processAudioBlock`` call. Default implementation is a no-op.

      Typical uses include allocating the ``float***`` pointer
      structure for ``processMulti`` (via
      :js:meth:`buildMultiTensorPointers`), constructing a
      :js:class:`JSPrePostProcessor` subclass with
      ``createFromPointer``, or caching frequently-accessed views
      onto WASM memory.

      :param AniraWorkletState state: The same state object that will
         later be passed to ``processAudioBlock``.
      :returns: ``Promise<void>``

   .. js:method:: processAudioBlock(inputs, outputs, state, bufferSize, parameters)

      Override hook for the real-time path, invoked from
      ``process()`` once per audio quantum. The default
      implementation copies the host inputs into the WASM scratch
      buffers, calls ``inferenceHandler.process`` with
      ``inputBufferPtr`` / ``outputBufferPtr``, and copies the
      result back to the host outputs — i.e. the single-tensor
      in-place case.

      :param inputs: Host input channels (the worklet ``process``
         argument).
      :type inputs: Float32Array[][]
      :param outputs: Host output channels.
      :type outputs: Float32Array[][]
      :param AniraWorkletState state: The configured state.
      :param number bufferSize: The number of samples in this
         quantum (clamped to ``ioConfig.maxBufferSize``).
      :param parameters: ``AudioParam`` values for this quantum,
         keyed by parameter name.
      :type parameters: Record<string, Float32Array>

   .. js:method:: copyAudioInputsToChannels(inputNode, state, bufferSize, channelOffset?, channelCount?)

      Copy a slice of the host's input channels into a contiguous
      range of ``state.inputChannelViews``. Missing source channels
      are zero-filled.

      :param inputNode: A single host input node (one entry of the
         ``inputs`` argument), or ``undefined``.
      :type inputNode: Float32Array[] | undefined
      :param AniraWorkletState state: The configured state.
      :param number bufferSize: Number of samples to copy / fill per
         channel.
      :param number channelOffset: Index into
         ``inputChannelViews`` where copying starts. Defaults to
         ``0``.
      :param number channelCount: Number of channels to fill.
         Defaults to ``inputChannelViews.length - channelOffset``.

   .. js:method:: copyAudioOutputsFromChannels(outputNode, state, samplesProcessed, channelOffset?, channelCount?)

      Copy a contiguous range of ``state.outputChannelViews`` back
      into the host's output channels. No-op if no output node is
      connected or no samples were produced.

      :param outputNode: A single host output node (one entry of
         the ``outputs`` argument), or ``undefined``.
      :type outputNode: Float32Array[] | undefined
      :param AniraWorkletState state: The configured state.
      :param number samplesProcessed: Number of valid samples per
         channel.
      :param number channelOffset: Index into
         ``outputChannelViews`` where copying starts. Defaults to
         ``0``.
      :param number channelCount: Number of channels to copy.
         Defaults to ``outputChannelViews.length - channelOffset``.

   .. js:method:: buildMultiTensorPointers(direction, channelsPerTensor)

      Build the ``float***`` pointer structure that ``processMulti``
      expects, by slicing the existing ``inputBufferPtr`` /
      ``outputBufferPtr`` (both already ``float**`` channel-pointer
      arrays laid out contiguously by ``configureAudioWorklet``).
      Also allocates a ``size_t[numTensors]`` array for the
      per-tensor sample counts.

      :param direction: ``'input'`` slices ``inputBufferPtr``;
         ``'output'`` slices ``outputBufferPtr``.
      :type direction: 'input' | 'output'
      :param channelsPerTensor: Per-tensor channel split. For
         example, ``[2, 1]`` means tensor 0 owns channels 0–1 and
         tensor 1 owns channel 2.
      :type channelsPerTensor: number[]
      :returns: ``{ tensorPtrs: number, numSamplesPtr: number }`` —
         the pointer to pass as the ``float***`` argument to
         ``processMulti``, and the pointer to fill with per-tensor
         sample counts on each block.

   .. js:method:: process(inputs, outputs, parameters)

      The ``AudioWorkletProcessor.process`` entry point. Clamps the
      requested buffer size to ``ioConfig.maxBufferSize`` and
      delegates to :js:meth:`processAudioBlock`. Returns silence
      until the configure handshake has completed. Subclasses
      rarely need to override this; override
      :js:meth:`processAudioBlock` instead.

      :param inputs: Host input channels.
      :type inputs: Float32Array[][]
      :param outputs: Host output channels.
      :type outputs: Float32Array[][]
      :param parameters: ``AudioParam`` values for this quantum.
      :type parameters: Record<string, Float32Array>
      :returns: ``true`` (keep the processor alive).

Type AniraWorkletState
----------------------

The configuration the main thread hands to the worklet during the
configure handshake. Exposed read-only on
:js:attr:`AniraAudioWorkletBase.aniraState` and re-passed as the
``state`` argument to :js:meth:`AniraAudioWorkletBase.onConfigured`
and :js:meth:`AniraAudioWorkletBase.processAudioBlock`.

``aniraWeb`` (:js:class:`AniraWeb`)
   The :js:class:`AniraWeb` instance for the worklet thread.
   Exposes ``malloc``, ``getHeapU32``, ``getHeapF32``,
   ``getWasmInstance``, factories, and so on.

``inferenceHandler`` (:js:class:`InferenceHandler`)
   The :js:class:`InferenceHandler` proxy. Call ``processMulti``
   (or ``process``) on it from ``processAudioBlock``.

``prePostProcessorPtr`` (number)
   Raw pointer to the C++ ``PrePostProcessor`` instance, used to
   construct a :js:class:`JSPrePostProcessor` subclass via
   ``createFromPointer``.

``inputBufferPtr`` (number)
   ``float**`` channel pointer array, laid out contiguously by
   ``configureAudioWorklet``. Pass directly to
   ``inferenceHandler.process`` for single-tensor models, or hand
   to :js:meth:`AniraAudioWorkletBase.buildMultiTensorPointers`
   for multi-tensor splits.

``outputBufferPtr`` (number)
   Same as ``inputBufferPtr`` but for the output side.

``inputDataBuffer`` (number)
   WASM heap offset for the per-channel input scratch buffers.
   Each channel occupies ``maxBufferSize * 4`` bytes, laid out
   contiguously.

``outputDataBuffer`` (number)
   WASM heap offset for the per-channel output scratch buffers.

``inputChannelViews`` (``Float32Array[]``)
   ``Float32Array`` views over the per-channel slices of the
   input scratch buffers. Use these to copy audio in without
   rebuilding views every block.

``outputChannelViews`` (``Float32Array[]``)
   ``Float32Array`` views over the per-channel slices of the
   output scratch buffers.

``ioConfig`` (``AudioWorkletIOConfig``)
   ``{ maxBufferSize, inputChannels, outputChannels,
   inputNodeIndex, outputNodeIndex }``.

``wasmMemory`` (``WebAssembly.Memory``)
   The shared ``WebAssembly.Memory``. Useful when you need a
   typed-array view over a custom region of the heap.
