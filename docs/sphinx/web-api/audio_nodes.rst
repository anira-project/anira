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

Destroy the node (``audioNode.destroy()``) before the handler.

The Render Geometry
-------------------

The handler is prepared under a Hard contract whose block range is the context's render
geometry, two numbers :js:func:`~renderGeometry.measureRenderGeometry` gives:

``quantum``, the smallest block
   The frames of one ``process()`` call: the context's ``renderQuantumSize`` where the browser has
   it, else 128. A context's render size is set at construction (``renderSizeHint``: an integer
   "between 1 sample frame and 6 times the context sample rate, inclusive", Web Audio API,
   *Supported Render Quantum Sizes*; ``"hardware"`` lets the browser pick the device's size) and
   only its effect, ``renderQuantumSize``, is observable.
``burst``, the largest block
   The frames the context renders back to back. A live ``AudioContext`` renders each device
   callback as a run of quanta with no time between them (2048 frames, 16 quanta, in headless
   Chromium; the device buffer on real hardware). An inference started in a run cannot complete
   before the run's next quantum, so a handler prepared for one quantum alone misses most quanta
   live; with ``block_max`` the burst, the latency the handler reports covers the run and every
   quantum is served. Chromium reports the burst as ``baseLatency`` (``audio_context.cc``: the
   larger of the destination's frames per buffer and the render quantum, over the rate, fixed
   when the context is constructed), which ``measureRenderGeometry`` returns at once, rounded up
   to whole quanta: a device buffer need not be a multiple of the quantum (441 frames, 9.1875 ms
   at 48 kHz, on a CI runner, which renders 3 or 4 quanta per callback: a burst of 512). Gecko's
   ``baseLatency`` is 0 by its definition (no buffering between the graph and the audio
   subsystem), and its ``outputLatency`` is the output's delay, not the callback's size; there a
   probe ``AudioWorklet`` watches ``process()`` calls for about 300 ms (after a 100 ms lead-in)
   and takes the longest run of back-to-back calls. The probe may measure more than the device
   buffer, never less: on a loaded machine two callbacks can run almost back to back and merge
   into one run (640 frames against the 512 above), which only sizes the handler a little larger.
   The probe needs a running context, so ``await context.resume()`` first. An ``OfflineAudioContext`` renders quantum by
   quantum: its burst is one quantum.

The geometry is explicit: the page measures it and declares it.

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [configs]
   :end-before: // [configs]
   :dedent:

The handler's latency then covers the burst. A dry path beside the model's waits as long:

.. literalinclude:: ../../../web/examples/model-file/main.ts
   :language: ts
   :start-after: // [latency]
   :end-before: // [latency]
   :dedent:

:js:meth:`Handler.createAudioNode` warns on the console when the prepared ``block_max`` is
below the burst the context reports (Chromium); nothing re-prepares on its own.

.. tip::
   Where the browser takes it, ``new AudioContext({ renderSizeHint: 'hardware' })`` renders one
   quantum of the device's size per callback, evenly spaced: the burst is one quantum
   (``block_min == block_max``), the lowest-latency setup. Feature-detect it
   (``context.renderQuantumSize > 0``), never assume it.

Changing the Geometry of a Running Node
---------------------------------------

``audioNode.prepare(contract)`` (``anira_em_audio_node_prepare``) re-prepares the node's handler
while the node streams and its context runs; ``handler.prepare`` refuses then, since the node's
render callback may be inside a process call. The node's callback is held off first (it outputs
silence for those quanta and leaves the handler alone), the main thread waits until the quantum
in flight returned (bounded; the audio thread never waits), the handler re-prepares, the node's
state that depends on the prepare is rebuilt off the audio thread, and the callback resumes.
In headless Chromium the hold lasts about 2 ms, under one burst. Read the handler's latency
again afterwards.

Device changes
   Chromium keeps the ``baseLatency`` of the device the context was created on, so after an
   output-device change the page measures with the probe, declares the new geometry and
   re-prepares through the node:

   .. literalinclude:: ../../../web/examples/model-file/main.ts
      :language: ts
      :start-after: // [device]
      :end-before: // [device]
      :dedent:

   It is the page's call, not anira's: the page switches the device (``setSinkId``), and the
   page decides whether a latency change mid-stream is acceptable. anira does not listen for
   ``sinkchange``; a page that does:

   .. literalinclude:: ../../../web/examples/model-file/main.ts
      :language: ts
      :start-after: // [sinkchange]
      :end-before: // [sinkchange]
      :dedent:

   A change of the operating system's default device may fire no event at all.

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
completes on the main thread's event loop (the promise resolves then). Any number of nodes may
be created at once on one context (``Promise.all``, any interleaving, also while another node of
the context is destroyed): they share the context's first-node setup, and each creation
completes on its own. Before the first node,
``createAudioNode`` loads ``worklet-scope.js`` into the scope: the scope has no ``performance``,
which the module's clock needs on the audio thread. A JavaScript stage's worklet phases run on
the scope's registry, which anira boots over the worklet's instance at every create; the code
there must not allocate, lock or wait (the stage states ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``).

The module is built with ``-sASSERTIONS=0``: Emscripten's Web Audio library asserts a
``window.AudioContext``, which an ``OfflineAudioContext`` is not.
