Custom Pre- and Post-Processing
===============================

A JavaScript stage is the pipeline's one stage (:doc:`../custom_preprocessing`) with its
phases in JavaScript: an ``anira_stage_desc`` whose slots are trampolines into the hook object
of the instance that runs each phase, added by ``_anira_em_pipeline_add_js_stage``. One hook bit
per phase decides what crosses into JavaScript; an unset bit leaves the slot empty, so anira's
default body runs there and nothing crosses.

===========================  ==================================  ========================
Hook bit                     Method                              Runs on
===========================  ==================================  ========================
``ANIRA_EM_HOOK_PRE``        ``preProcess(ctx, prepared)``       the audio worklet (Hard)
``ANIRA_EM_HOOK_POST``       ``postProcess(ctx, prepared)``      the audio worklet (Hard)
``ANIRA_EM_HOOK_RESET``      ``reset(ctx, prepared)``            the audio worklet (Hard)
``ANIRA_EM_HOOK_BEFORE``     ``beforeInference(ctx, prepared)``  the inference Worker
``ANIRA_EM_HOOK_AFTER``      ``afterInference(ctx, prepared)``   the inference Worker
``ANIRA_EM_HOOK_PREPARE``    ``prepare(info, outPrepared)``      the main thread
``ANIRA_EM_HOOK_UNPREPARE``  ``unprepare(prepared)``             the main thread
``ANIRA_EM_HOOK_INIT``       ``init(info)``                      the main thread
``ANIRA_EM_HOOK_RELEASE``    ``release()``                       the main thread
===========================  ==================================  ========================

``ctx`` points at the call's ``anira_stage_ctx`` (``readStageContext`` reads its fields through
``layout.ts``); the stage reaches its rings and model tensors through the C accessors
(``_anira_stage_input_ring``, ``_anira_stage_input_tensor``, ``_anira_ring_pop_block``, …) and may
compose anira's default bodies (``_anira_stage_default_pre_process(ctx)``,
``_anira_stage_default_post_process(ctx)``); a pre/post bit whose method is missing runs the
default. ``prepared`` is what this handler's ``prepare`` stored into its ``outPrepared`` slot
(a heap pointer the stage owns, or 0). Return an ``anira_status``; a thrown exception fails the
phase (``ANIRA_ERROR_UNKNOWN``, the chunk delivers zeros).

The Real-Time Promise Is Stated
-------------------------------

``stageFlags`` is the stage's real-time promise, never derived from the bits: under a Hard
contract ``preProcess``, ``postProcess`` and ``reset`` run on the driving thread, the audio
thread, and ``_anira_handler_prepare`` refuses such a stage with ``ANIRA_ERROR_CONFIG`` unless it
states ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``. ``ANIRA_STAGE_FLAG_REALTIME_HOOKS`` promises the
same of ``beforeInference`` / ``afterInference``. A stage that states a promise allocates
nothing, takes no lock and blocks on nothing in those methods, in JavaScript too: no object
literals or closures per call, no ``postMessage``, views created once.

An Example: Input Gain Before the Model
---------------------------------------

``InputGainStage`` of the custom-worklet example (``web/examples/custom-worklet/``,
:doc:`custom_audio_worklets`) scales the model's audio input by the model's Static input
``gain``, ramping from the previous block's gain across the block. ``preProcess`` composes
anira's default body (which pops the input ring into the model's audio tensor and leaves every
Static input materialised), then reads both tensors with ``_anira_stage_input_tensor`` into a
record it does not allocate per call: ``prepare`` allocated it, with the ramp's state, for this
handler, and ``prepared`` points at it in every phase. ``reset`` forgets the ramp. The context's
fields are read through ``layout`` offsets, never by an object per call:

.. literalinclude:: ../../../web/examples/custom-worklet/gain-stage.ts
   :language: typescript

The worklet constructs it from its ``stageClasses`` (:doc:`custom_audio_worklets`); a stage with
before/after bits is also listed in the worker file (``setupInferenceWorker(engines, { InputGainStage })``).
With prepare and unprepare bits it passes its main-thread object as ``stage``:

.. code-block:: typescript

   const stageKey = await anira.addJSStage(pipeline, {
     className: 'InputGainStage',
     hookMask: ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_RESET | ANIRA_EM_HOOK_PREPARE | ANIRA_EM_HOOK_UNPREPARE,
     stageFlags: ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
     stage: new InputGainStage({ module: anira.module, heap: anira.heap }),
   })
   // ... _anira_handler_create, the contract, _anira_handler_prepare ...
   const node = await anira.configureAudioWorklet(audioContext, handler, {
     nodeName: 'gain-stage-processor',
     stageKey,
     staticInputs: [GAIN_SLOT],
     modelConfigPtr: model,
   })
   node.setStaticInput(GAIN_SLOT, [0.5])

``addJSStage`` is ``_anira_em_pipeline_add_js_stage(pipeline, jsKey, hookMask, stageFlags, err)``
plus the registrations; call it before ``_anira_handler_create``. A mask with an init, prepare,
unprepare or release bit needs the main-thread object (``stage``): ``addJSStage`` refuses it
otherwise. A stage released on the main thread (its last carrier died) is dropped from every
instance; ``removeJSStage(jsKey)`` drops it explicitly. A pipeline holds at most one
stage: a second is ``ANIRA_ERROR_INVALID_STATE``.
