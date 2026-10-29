Audio Nodes
===========

A handler streams audio through a Wasm Audio Worklet node: ``await handler.createAudioNode(
audioContext, options)`` (:js:meth:`Handler.createAudioNode`, the C entry
``anira_em_audio_node_create``). The node works on an ``AudioContext`` and an
``OfflineAudioContext``; connect ``audioNode.node`` into the graph.

The Render Callback
-------------------

The node's render callback is C, on the browser's audio thread, once per render quantum:

* the node has one input per Streamed input slot it serves and one output per Streamed output
  slot, each with the slot's channels; their buffers are the handler's planar host tensors,
  ``[channels, quantum]``, without a copy: an output tensor is the node's output;
* the call is one ``anira_handler_process_multi`` over every slot (with ``waitMs``,
  ``anira_handler_process_multi_wait``: an offline render that waits for every inference), an
  empty tensor at each position the node does not serve;
* the miss rule: ``ANIRA_MISSED`` keeps the miss policy's fill (bypass, hold the last, the
  callback), a failure is silence, a success silences the samples an output was not delivered;
* a quantum other than the one the handler is prepared for is silence (the creation refuses a
  handler whose Hard block range excludes the context's render quantum);
* it allocates nothing and locks nothing.

The handler is prepared under a Hard contract whose smallest block is the render quantum and
whose largest is the context's burst:

.. code-block:: ts

   Contract.hard(anira, {
     blockMin: 128,
     blockMax: renderBurstFrames(audioContext),
     rate: audioContext.sampleRate,
     budgetMs,
   })

A live ``AudioContext`` does not render one quantum at a time: each device callback renders a
burst of quanta back to back, with no time between them (2048 frames, 16 quanta, in headless
Chromium; the device buffer in a browser on real hardware). An inference started in a burst
cannot complete before the burst's next quantum, so a handler prepared for blocks of 128 alone
misses most quanta live; with ``block_max`` the burst, the latency the handler reports covers it
and every quantum is served once the stream runs. :js:func:`~AudioNode.renderBurstFrames` reads
the burst from the context: Chromium's ``baseLatency`` (its callback buffer), Gecko's
``outputLatency`` where ``baseLatency`` is 0, 128 for an ``OfflineAudioContext``. Call it once the
context runs (Gecko reports its latency then). :js:meth:`Handler.createAudioNode` warns on the
console when the prepared ``block_max`` is below the burst. Destroy the node
(``audioNode.destroy()``) before the handler.

A Model with Several Streams
----------------------------

A model with several Streamed inputs or outputs (a mixture and a control signal, a stereo pair
as two tensors) runs through one node with an input and an output per stream, which is the
default:

.. code-block:: ts

   const audioNode = await handler.createAudioNode(audioContext) // every Streamed slot
   voice.connect(audioNode.node, 0, 0) // node input 0: Streamed input slot 0
   sidechain.connect(audioNode.node, 0, 1) // node input 1: Streamed input slot 1
   audioNode.node.connect(left, 0) // node output 0: Streamed output slot 0
   audioNode.node.connect(right, 1) // node output 1: Streamed output slot 1

One node calls ``anira_handler_process_multi`` once per quantum over every stream, which keeps
the guarantees of the multi form. Two nodes on one handler, one slot each, are not the
supported form: each calls the single-slot entry, so the streams stay aligned (submission waits
for every stream's ring), but

* ``ANIRA_MISS_BYPASS`` passes the input through only on the call that carries the anchored
  slot: the other node gets zeros on a miss;
* the wait credit of ``ANIRA_WAIT_CONTRACT`` (the ``_wait`` forms' budget from the anchored
  tensor) is never taken for the other call;
* an ``anira_miss_fn`` is called once per node, each time with one slot, instead of once per
  block with all of them.

On a miss under ``ANIRA_MISS_BYPASS``, every output carries the anchored input's block (the
contract refuses BYPASS when an output's channels are not the anchored input's).

Options
-------

``inputSlots`` / ``outputSlots``
   The Streamed slots the node's inputs feed and its outputs play, input (output) *k* the *k*-th,
   each with the slot's channels (the node's ``channelCount`` is the largest input slot's,
   explicit and discrete; a slot reads its first channels). Default: every Streamed slot of the
   side, in slot order, so input 0 feeds the first Streamed input. ``inputSlots: []`` is a node
   without an input.
``partialStreams``
   Without it the creation refuses a Streamed slot of the model that no input or output of the
   node serves; with it (``ANIRA_EM_AUDIO_NODE_FLAG_PARTIAL_STREAMS``) such a slot stays an empty
   element of the multi call.
``params``
   Static float32 scalar inputs (rank 1, one element) as AudioParams of the node, by slot:
   ``audioNode.params.get(slot)`` is a k-rate AudioParam (automatable, default 0); the callback
   pushes its value to the handler each quantum it changed.
``staticInputs``
   Static inputs of any shape and dtype the main thread writes whole,
   ``audioNode.setStaticInput(slot, values)`` (a typed array of the spec's bytes, or numbers as
   float32): a seqlock block per slot, so the callback never pushes a torn value and never waits.
``waitMs``
   0 (the default): never wait, a live context. Above 0: wait up to this long per quantum for its
   inference; ``await context.ready()`` first.
``workletModules``
   Modules the context's ``audioWorklet`` loads before the node: the worklet side of the
   handler's JavaScript stage (:doc:`custom_pre_post_processing`), bundled into one file each.

The Worklet Scope
-----------------

The first node of an audio context starts the context's worklet thread, a node of *n*
AudioParams registers the processor ``anira-node-n`` once per context, and the creation
completes on the main thread's event loop (the promise resolves then). Before the first node,
``createAudioNode`` loads ``worklet-scope.js`` into the scope: the scope has no ``performance``,
which the module's clock needs on the audio thread. A JavaScript stage's worklet phases run on
the scope's registry, which anira boots over the worklet's instance at every create; the code
there must not allocate, lock or wait (the stage states ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``).

The module is built with ``-sASSERTIONS=0``: Emscripten's Web Audio library asserts a
``window.AudioContext``, which an ``OfflineAudioContext`` is not.
