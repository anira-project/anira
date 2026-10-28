Class AniraAudioWorkletBase
===========================

.. note::
   This class is exported from the package's audio-worklet subpath:
   ``import { AniraAudioWorkletBase, type AniraWorkletState } from '@anira-project/anira/workers/worklet-base'``.
   The subpath isolates the worklet bundle so the main entry point
   stays free of references to ``AudioWorkletProcessor``, which only
   exists in ``AudioWorkletGlobalScope``.

   See :doc:`../../custom_audio_worklets` for usage guidance.

.. js:class:: AniraAudioWorkletBase extends AudioWorkletProcessor

   The audio worklet of anira: one ``_anira_handler_process`` per render quantum over planar
   host tensors (``_anira_tensor_init_host_planar``, shape ``[channels, quantum]``) that it
   builds at configure for the declared quantum; ``process()`` allocates nothing, a quantum
   other than the declared one is silence and is reported to main (``'quantumChanged'``), and
   the blocks are rebuilt on ``'setQuantum'`` and freed on ``'dispose'``. It is wired to
   the main thread by :js:meth:`AniraWeb.configureAudioWorklet`: the configure handshake
   instantiates the module in the worklet on its own stack and thread-local block, constructs
   the JavaScript stage named in the message from :js:attr:`stageClasses`, fills
   ``this.aniraState`` and then calls :js:meth:`onConfigured`.

   .. js:attribute:: aniraState
      :type: AniraWorkletState | null

      The worklet's state once the configure handshake completed: the module instance
      (``anira``: ``module``, ``heap``, ``hooks``, ``memory``), ``handlerPtr``, ``ioConfig``,
      ``quantum``, the two ``PlanarBlock`` records ``input`` and ``output``, ``deliveredPtr``
      and its view ``delivered``. ``null`` until then, and again after a dispose.

   .. js:attribute:: stageClasses
      :type: WorkletStageClassMap

      Class name to JavaScript stage class: the stages whose ``preProcess``,
      ``postProcess`` and ``reset`` this worklet runs. Empty by default; a subclass sets it.

   .. js:method:: onConfigured(state)

      Override hook for one-time setup, invoked once after the configure handshake and
      before the first ``processAudioBlock`` call. Default: nothing.

      :param AniraWorkletState state: The state later passed to ``processAudioBlock``.
      :returns: ``Promise<void>``

   .. js:method:: processAudioBlock(inputs, outputs, state, quantum, parameters)

      The real-time path, invoked from ``process()`` once per render quantum of the declared
      size. The default copies the host input channels into the input block, calls
      ``_anira_handler_process`` (``_anira_handler_process_wait`` when ``ioConfig.waitMs`` is
      above 0) with the
      configured slots, and copies what the call filled (``samplesToCopy``: the whole quantum on
      ``ANIRA_MISSED``) of the output block to the host outputs; a failed status outputs
      silence. It allocates nothing.

      :param inputs: Host input channels (the worklet ``process`` argument).
      :type inputs: Float32Array[][]
      :param outputs: Host output channels.
      :type outputs: Float32Array[][]
      :param AniraWorkletState state: The configured state.
      :param number quantum: The number of samples in this quantum.
      :param parameters: ``AudioParam`` values for this quantum, keyed by parameter name.
      :type parameters: Record<string, Float32Array>

   .. js:method:: buildPlanarBlock(anira, channels, quantum)

      A planar host block of ``channels`` x ``quantum`` float32 samples in the heap: the
      channel buffers, the array of their pointers, the ``int64_t[2]`` shape and the
      ``anira_tensor`` (``layout.anira_tensor.size`` bytes), with a ``Float32Array`` view per
      channel. For subclasses that stream more than one slot.

      :returns: ``PlanarBlock``

   .. js:method:: freePlanarBlock(anira, block)

      Free what :js:meth:`buildPlanarBlock` allocated.
