Custom Audio Worklets
=====================

The default worklet that ships with Anira Web streams one input slot and one output slot of a
prepared handler: per render quantum it copies the node's input channels into a planar host
tensor, makes one ``_anira_handler_process`` call and copies the delivered samples out. It also
serves the model's Static inputs (`Static Inputs`_). A custom worklet extends
:js:class:`AniraAudioWorkletBase` when you need more.

When You Need a Custom Worklet
------------------------------

* a JavaScript stage whose ``preProcess`` / ``postProcess`` / ``reset`` run here: the worklet
  constructs it from its ``stageClasses`` (:doc:`custom_pre_post_processing`);
* more than one streamed slot, or a slot other than the first (``inputSlot`` / ``outputSlot``
  cover one each);
* a Static input driven by an ``AudioParam`` with automation: the worklet declares its own
  ``parameterDescriptors`` and writes the value in ``processAudioBlock`` (a value set from the
  main thread needs no custom worklet: `Static Inputs`_);
* one-time setup in the worklet (views, extra tensors): override ``onConfigured``.

A Complete Example
------------------

``web/examples/custom-worklet/`` in the repository is a complete page: a JavaScript stage that
applies the model's Static input ``gain`` before the model, run in a worklet of its own, a
JavaScript engine on an inference Worker standing in for the model (it doubles its input), and
the gain written from the main thread. CI type-checks it against the built package
(``npm run examples``: ``tsc`` with ``strict``, then the Vite build) and renders its check page
(``check.ts``, an ``OfflineAudioContext`` with the gain changed mid-render) in headless
Chromium. Run it with ``npx vite --config examples/custom-worklet/vite.config.ts`` from ``web/``
after ``npm run build``.

The worklet file names the stage classes the worklet may construct and registers the
processor; ``configureAudioWorklet`` asks for it by ``nodeName``:

.. literalinclude:: ../../../web/examples/custom-worklet/my-worklet.ts
   :language: typescript

The stage, ``InputGainStage``, is the subject of :doc:`custom_pre_post_processing`. Its
``preProcess`` runs here, inside the worklet's ``_anira_handler_process`` call, so it promises
real-time behaviour (``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``); its ``prepare`` runs on the main
thread and allocates the per-handler memory ``preProcess`` and ``reset`` use:

.. literalinclude:: ../../../web/examples/custom-worklet/gain-stage.ts
   :language: typescript

The inference Worker's file lists the engine classes it constructs:

.. literalinclude:: ../../../web/examples/custom-worklet/inference-worker.ts
   :language: typescript

``main.ts`` builds everything in order: the module and its log, the context and the model config
(slot 1 is the Static input ``gain``), the pipeline with the engine and the stage (the stage's
js_key comes back from ``addJSStage``), the handler with a Hard contract of one render quantum
at the context's rate, the inference Worker, then the worklet: its processor, the stage it
constructs (``stageKey``) and the Static input it serves (``staticInputs``). The teardown
disposes the worklet before it stops the Worker and destroys the handler:

.. literalinclude:: ../../../web/examples/custom-worklet/main.ts
   :language: typescript

Bundling and Serving
~~~~~~~~~~~~~~~~~~~~

``addModule`` loads one script, which cannot import another at run time, so the worklet is
bundled into one self-contained ES module: with Vite, ``import url from './my-worklet.ts?worker&url'``
in ``main.ts`` and a single-file worker output. The module's memory is a shared
``WebAssembly.Memory``, which only a cross-origin isolated page (``crossOriginIsolated``) may
create, so the server sends ``Cross-Origin-Opener-Policy: same-origin`` and
``Cross-Origin-Embedder-Policy: require-corp`` with every response; the example's config sets
both for ``vite`` and ``vite preview``:

.. literalinclude:: ../../../web/examples/custom-worklet/vite.config.ts
   :language: typescript

TypeScript's DOM lib does not declare the worklet scope's ``registerProcessor`` and
``AudioWorkletProcessor``; ``@anira-project/anira/workers/worklet-base`` declares both for
the files that import it.

The Tensor Process
------------------

The worklet builds a ``PlanarBlock`` per direction with ``buildPlanarBlock(anira, channels,
quantum)``: one heap buffer per channel, the array of their pointers, an ``int64_t[2]`` shape
``[channels, quantum]`` and an ``anira_tensor`` filled by ``_anira_tensor_init_host_planar``, with
a ``Float32Array`` view per channel, at configure, for the declared quantum (the ``quantum``
option, 128 by default). ``process()`` allocates nothing; per quantum the default
``processAudioBlock`` does

.. code-block:: typescript

   const status = module._anira_handler_process(
     handlerPtr, input.tensor, ioConfig.inputSlot, output.tensor, ioConfig.outputSlot, deliveredPtr
   )

and copies ``heap.readU32(deliveredPtr)`` samples of every output channel to the node (silence
on a failed status, the whole quantum on ``ANIRA_MISSED``, which carries the miss policy's
fill although its delivered count is 0: ``samplesToCopy``). A subclass that
streams more slots builds more blocks in ``onConfigured`` and calls
``_anira_handler_process_multi`` in its own ``processAudioBlock``.

The handler's Hard contract declares ``block_min = block_max =`` the render quantum (128
frames by default). A larger call is refused, so when the browser hands the worklet another
quantum, the worklet outputs silence and posts ``'quantumChanged'`` once; the host's
``onQuantumChange`` re-prepares the handler with that block
(``_anira_contract_hard_set_geometry`` and ``_anira_handler_prepare``), and then main sends
``'setQuantum'``, on which the worklet rebuilds its blocks in its message handler.

``aniraWeb.disposeAudioWorklet(node)`` frees what configure allocated: the worklet frees its
blocks and stops calling into the module (``process()`` outputs silence), then the worklet
instance's stack and thread-local block are freed. Call it before destroying the handler;
``configureAudioWorklet(ctx, handler, { node })`` configures a node again, disposing it first.

A JavaScript stage with pre/post/reset bits must be constructed in the worklet that streams its
handler: pass ``stageKey`` or ``pipelinePtr`` (the pipeline the handler was created from);
``configureAudioWorklet`` refuses to leave a registered stage of that kind unnamed.

The worklet never waits for an inference by default. ``waitMs`` opts in to
``_anira_handler_process_wait``, which waits up to that many milliseconds for each quantum's
inference: legal, but a wait on the audio thread trades a dropout for latency only when the
inference Workers are known to be fast enough.

Static Inputs
-------------

Every worklet, the default one and every subclass, serves the Static inputs ``configureAudioWorklet``
lists, and the main thread writes them through the node:

.. code-block:: typescript

   const node = await anira.configureAudioWorklet(audioContext, handler, {
     staticInputs: [1], // input slot 1 of the model config: a Static input
     modelConfigPtr: model, // its spec sizes the buffer (dtype and shape)
   })
   node.setStaticInput(1, [0.5]) // plain numbers, one per element
   node.setStaticInput(1, new Float32Array([0.25])) // or the slot's bytes, any dtype

Each slot gets a block in the shared heap: a sequence word, the value, the worklet's copy and a
host tensor over that copy in the spec's dtype and shape. ``setStaticInput`` makes the word odd,
writes the value and makes it even again. Before each quantum's process call the worklet reads
the word, copies the value and reads the word again; only when both reads are the same even
number, and the number changed since its last push, does it hand the copy to
``_anira_handler_set_static_input``, which applies it from the next submitted inference on.
Otherwise the handler keeps the last good value and the next quantum tries again: a value is
never pushed torn, and the worklet never waits for the writer. ``process()`` allocates nothing
for it; ``disposeAudioWorklet`` frees the blocks.

``setStaticInput`` takes the whole tensor: a typed array or ``DataView`` of exactly the slot's byte
size (any dtype, the bytes as they are), or plain numbers, one per element, for float32/64,
int8/16/32, uint8/16/32 and bool8. A slot the node does not serve or a size that does not fit
throws and writes nothing. A value for the first inference is pushed before the first quantum's
call, so write it right after ``configureAudioWorklet``; before the worklet streams, the host may
also call ``_anira_handler_set_static_input`` itself (the store survives ``prepare``). Once the
worklet streams, it is the handler's driving thread and the only caller of
``_anira_handler_set_static_input``. The push runs in ``process()`` ahead of
``processAudioBlock``: a subclass that overrides ``process()`` itself calls
``state.staticInputs?.pull()`` first. An ``AudioParam`` per slot is not possible (a processor's
parameters are fixed by its class's static ``parameterDescriptors``), and a port message would
allocate on the audio thread.

The Worklet's Instance
----------------------

The configure handshake instantiates the module inside the worklet (the binary arrives by
``postMessage``; the worklet scope has no ``fetch``), sets the instance's stack pointer to the
stack the main thread allocated and installs its thread-local block
(``_anira_em_init_instance``) before any other C call. Records logged in the worklet are
forwarded to the main thread's log hook. The worklet scope has no ``performance``: the base
installs a monotonic ``Date.now()`` clock, which the runtime's clock reads need.
