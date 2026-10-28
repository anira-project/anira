Custom Audio Worklets
=====================

The default worklet that ships with Anira Web streams one input slot and one output slot of a
prepared handler: per render quantum it copies the node's input channels into a planar host
tensor, makes one ``_anira_handler_process`` call and copies the delivered samples out. A custom
worklet extends :js:class:`AniraAudioWorkletBase` when you need more.

When You Need a Custom Worklet
------------------------------

* a JavaScript stage whose ``preProcess`` / ``postProcess`` / ``reset`` run here: the worklet
  constructs it from its ``stageClasses`` (:doc:`custom_pre_post_processing`);
* more than one streamed slot, or a slot other than the first (``inputSlot`` / ``outputSlot``
  cover one each);
* ``AudioParam`` values written to a Static input (``_anira_handler_set_static_input``) per
  quantum;
* one-time setup in the worklet (views, extra tensors): override ``onConfigured``.

Skeleton
--------

.. code-block:: typescript

   // my-worklet.ts
   import { AniraAudioWorkletBase } from '@anira-project/anira/workers/worklet-base'
   import { InputGainStage } from './gain-stage'

   class MyWorklet extends AniraAudioWorkletBase {
     protected override stageClasses = { InputGainStage }
   }

   registerProcessor('my-processor', MyWorklet)

Bundle the file into one self-contained script (``addModule`` cannot resolve imports), register
it and name its processor:

.. code-block:: typescript

   await anira.registerAudioWorkletForContext(audioContext, new URL('./my-worklet.js', import.meta.url))
   const node = await anira.configureAudioWorklet(audioContext, handler, {
     nodeName: 'my-processor',
     stageKey, // from addJSStage
     inputChannels: 2,
     outputChannels: 2,
   })

The Tensor Process
------------------

The worklet builds a ``PlanarBlock`` per direction with ``buildPlanarBlock(state, channels,
quantum)``: one heap buffer per channel, the array of their pointers, an ``int64_t[2]`` shape
``[channels, quantum]`` and an ``anira_tensor`` filled by ``_anira_tensor_init_host_planar``, with
a ``Float32Array`` view per channel. It rebuilds them only when the quantum or a channel count
changes; per quantum the default ``processAudioBlock`` does

.. code-block:: typescript

   const status = module._anira_handler_process(
     handlerPtr, input.tensor, ioConfig.inputSlot, output.tensor, ioConfig.outputSlot, deliveredPtr
   )

and copies ``heap.readU32(deliveredPtr)`` samples of every output channel to the node (silence
on a failed status). Nothing allocates on the audio thread outside a rebuild. A subclass that
streams more slots builds more blocks in ``onConfigured`` and calls
``_anira_handler_process_multi`` in its own ``processAudioBlock``.

The handler's Hard contract declares ``block_min = block_max =`` the render quantum (128
frames by default). A larger call is refused, so when the browser hands the worklet another
quantum, the worklet rebuilds its blocks and posts ``'quantumChanged'``; the host's
``onQuantumChange`` re-prepares the handler with that block
(``_anira_contract_hard_set_geometry`` and ``_anira_handler_prepare``), and the worklet outputs
silence until then.

The worklet never waits for an inference by default. ``waitMs`` opts in to
``_anira_handler_process_wait``, which waits up to that many milliseconds for each quantum's
inference: legal, but a wait on the audio thread trades a dropout for latency only when the
inference Workers are known to be fast enough.

The Worklet's Instance
----------------------

The configure handshake instantiates the module inside the worklet (the binary arrives by
``postMessage``; the worklet scope has no ``fetch``), sets the instance's stack pointer to the
stack the main thread allocated and installs its thread-local block
(``_anira_em_init_instance``) before any other C call. Records logged in the worklet are
forwarded to the main thread's log hook. The worklet scope has no ``performance``: the base
installs a monotonic ``Date.now()`` clock, which the runtime's clock reads need.
