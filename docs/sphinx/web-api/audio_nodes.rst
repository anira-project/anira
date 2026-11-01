Audio Nodes
===========

A handler streams audio through a Wasm Audio Worklet node: ``await handler.createAudioNode(
audioContext, options)`` (:js:meth:`InferenceHandler.createAudioNode`, the C entry
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
   The frames of every ``process()`` call: the context's ``renderQuantumSize``, else 128 where
   the browser has no such attribute (Firefox and Safari today). It is fixed for the context's
   life, an output-device change (``setSinkId``) included: "the size of each block remaining
   constant for the lifetime of a BaseAudioContext" (Web Audio API, `Rendering an Audio Graph
   <https://webaudio.github.io/web-audio-api/#rendering-loop>`_). A context's size is chosen at
   construction (``renderSizeHint``: an integer "between 1 sample frame and 6 times the context
   sample rate, inclusive", `Supported Render Quantum Sizes
   <https://webaudio.github.io/web-audio-api/#render-quantum-sizes>`_, or ``"hardware"``, the size
   the browser deems best for the current configuration). A ``"hardware"`` context reports 0
   until it runs, and ``measureRenderGeometry`` refuses it then.
``burst``, the largest block
   The frames the context renders back to back. A live ``AudioContext`` renders each callback
   of the audio system as a run of quanta with no time between them (2048 frames, 16 quanta, in
   headless Chromium; the device buffer on real hardware). An inference started in a run cannot
   complete before the run's next quantum, so a handler prepared for one quantum alone misses
   most quanta live; with ``host_block_max`` the burst, the latency the handler reports covers the run
   and every quantum is served. The callback's size is never exposed: the spec calls it "a
   varying number of sample-frames" (`Rendering an Audio Graph
   <https://webaudio.github.io/web-audio-api/#rendering-loop>`_), and no attribute gives it across
   browsers (Chromium's ``baseLatency`` is its callback buffer and keeps the first device's after
   ``setSinkId``, Safari's is one quantum, Firefox's is 0). So the burst is always measured, from
   the timing of ``process()`` calls: "A running AudioContext can be used to infer the user's
   audio hardware buffer size via AudioWorklet timing" (`Privacy Considerations
   <https://webaudio.github.io/web-audio-api/#privacy-considerations>`_, on the ``"hardware"``
   render size). The burst is the worst case, a whole number of quanta: a callback whose size is
   no multiple of the quantum renders a varying number of them, and the largest counts (441
   frames, 9.1875 ms at 48 kHz on a CI runner: 3 or 4 quanta of 128, a burst of 512). The worst
   case can be rarer than any window: a callback of 481 frames at a quantum of 480 renders one
   quantum per callback and two back to back once every 480 callbacks (about 5 s), when the frame
   left over each time adds up to a quantum. So a probe ``AudioWorklet`` measures the frames per
   device callback, the period at which the runs of calls begin timed with a Worker's microsecond
   clock (the worklet scope has only ``Date.now``) times the rate. The burst is the frames per
   callback over the quantum, rounded up (481 / 480: 2 quanta, 960 frames), with half a frame
   taken off first so an exact fit stays one quantum (a ``"hardware"`` quantum of the callback's
   size). A quantum at least the callback's size is rendered one call per callback or fewer: a
   burst of one quantum. The longest run of back-to-back calls counts only while that estimate is
   not settled, or without a Worker (a page whose policy refuses one): a renderer that wakes late
   renders the quanta it owes back to back, a run the device never asks for.
   A context whose sample rate differs from the device's is resampled: each callback renders the
   device's frames times the context's rate over the device's, in context frames (2048 device
   frames at 48 kHz for a context of 44.1 kHz: 1881.6 frames, runs of 14 and 15 quanta of 128, a
   burst of 1920). The probe times the callbacks and counts in the context's rate, so it measures
   that; Chromium's ``baseLatency`` times the context's ``sampleRate`` (2048) does not, one more
   reason no attribute gives the burst.
   An ``OfflineAudioContext`` renders quantum by quantum: its burst is one quantum, nothing to
   measure.

The probe needs a running context, so ``await context.resume()`` first. It costs time once,
after the page's start (its Play click): a 100 ms lead-in, then at least 300 ms, and on until the
frames per callback are known well enough to tell which whole number of quanta covers a
callback (16 run starts or more, within three standard errors and a tenth of a frame), at most
2 s: about 1 s in headless Chromium, whose callbacks of 2048 frames come every 42.7 ms. Its
worklet notifies the clock Worker at every call (a futex wake on the audio thread while it
measures). The geometry is explicit: the page measures it and declares it.

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

Nothing re-prepares on its own, and :js:meth:`InferenceHandler.createAudioNode` cannot tell a
block range too small for the context: measure first.

.. tip::
   Create the context with ``renderSizeHint: 'hardware'`` where the browser has render sizes
   (feature-detect: ``'renderQuantumSize' in AudioContext.prototype``): it renders at the size
   the browser deems best for the device instead of a fixed 128 (Chromium chooses 128; Firefox's
   patches for it, bug 2034508, the audio system's smallest callback). A quantum of the
   callback's size is one ``process()`` call per callback, a burst of one quantum
   (``host_block_min == host_block_max``).

   .. literalinclude:: ../../../web/examples/model-file/main.ts
      :language: ts
      :start-after: // [context-options]
      :end-before: // [context-options]
      :dedent:

Changing the Geometry of a Running Node
---------------------------------------

``audioNode.prepare(contract)`` (``anira_em_audio_node_prepare``) re-prepares the node's handler
while the node streams and its context runs; ``handler.prepare`` refuses then, since the node's
render callback may be inside a process call. The node's callback is held off first (it outputs
silence for those quanta and leaves the handler alone), the main thread waits until the quantum
in flight returned (bounded; the audio thread never waits), the handler re-prepares, the node's
state that depends on the prepare is rebuilt off the audio thread, and the callback resumes.
In headless Chromium the hold lasts about 2 ms, under one burst. It returns the
:js:class:`PlanReport` of that prepare, as :js:meth:`InferenceHandler.prepare` does; read the
handler's latency again afterwards (``handler.getLatency()``).

Device changes
   An output-device change may change the burst (the callback's size), never the quantum. The
   page measures again with its streaming nodes held (``hold``: each outputs silence and leaves
   its handler alone while the probe watches, then streams on, whatever happens), declares the
   new geometry and re-prepares through the node:

   .. code-block:: ts

      const g = await measureRenderGeometry(audioContext, { hold: [audioNode] })
      contract.hardHostBlocks(g.quantum, g.burst, audioContext.sampleRate)
      audioNode.prepare(contract)

   A node's rendering on the audio thread hides the callback's runs from the probe: in headless
   Chromium a node whose JavaScript processor spins 8 ms per inference made the probe measure 512
   frames of the device's 2048; held, it measured 2048. The model-file example does the same:

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
   ``audioNode.setStaticInput(slot, values)`` (in the spec's dtype, as
   :js:meth:`InferenceHandler.setStaticInput` takes it: a typed array of the spec's element type,
   any view of exactly the spec's bytes, or plain numbers converted to it): a seqlock block per
   slot, so the callback never pushes a torn value and never waits.
``waitMs``
   0 (the default): never wait, a live context. Above 0: wait up to this long per quantum for its
   inference; ``await context.ready()`` first.
``workletModules``
   Modules the context's ``audioWorklet`` loads before the node: the worklet side of the
   handler's JavaScript processor (:doc:`custom_pre_post_processing`), bundled into one file each.

The Worklet Scope
-----------------

The first node of an audio context starts the context's worklet thread, a node of *n* AudioParams
registers the processor ``anira-node-n`` once per context, and the creation completes on the main
thread's event loop (the promise resolves then). Any number of nodes may be created at once on one
context (``Promise.all``, any interleaving, also while another node of the context is destroyed):
they share the context's first-node setup, and each creation completes on its own. Before the first
node, ``createAudioNode`` loads ``worklet-scope.js`` into the scope: the scope has no
``performance``, which the module's clock needs on the audio thread. A JavaScript processor's
worklet phases run on the scope's registry, which anira boots over the worklet's instance at every
create; the code there must not allocate, lock or wait (the processor states
``ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST``).

The module is built with ``-sASSERTIONS=0``: Emscripten's Web Audio library asserts a
``window.AudioContext``, which an ``OfflineAudioContext`` is not.
