Audio Nodes
===========

A handler streams audio through a Wasm Audio Worklet node: ``await handler.createAudioNode(
audioContext, options)`` (:js:meth:`Handler.createAudioNode`, the C entry
``anira_em_audio_node_create``). The node works on an ``AudioContext`` and an
``OfflineAudioContext``; connect ``audioNode.node`` into the graph.

The Render Callback
-------------------

The node's render callback is C, on the browser's audio thread, once per render quantum:

* the node's input and output buffers are the handler's planar host tensors, ``[channels,
  quantum]``, without a copy: the output tensor is the node's output;
* the call is ``anira_handler_process`` (with ``waitMs``, ``anira_handler_process_wait``: an
  offline render that waits for every inference);
* the miss rule: ``ANIRA_MISSED`` keeps the miss policy's fill (bypass, hold the last, the
  callback), a failure is silence, a success silences the samples it did not deliver;
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

Options
-------

``inputChannels`` / ``outputChannels``
   The node's channels (the input's count is explicit and discrete), the tensors' channel axes.
   Default 1 and 1; 0 inputs for a node without an input.
``inputSlot`` / ``outputSlot``
   The Streamed slots of the model config the node feeds and plays. Default 0.
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
