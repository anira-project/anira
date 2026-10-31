Custom Stages
=============

A JavaScript stage is the pipeline's one stage written in TypeScript: the counterpart of a C
``anira_stage_desc`` (:doc:`../custom_preprocessing`), defined by name in every scope that runs
one of its phases, with no rebuild of the module.

Defining a Stage
----------------

.. code-block:: ts

   import { defineStage } from '@anira-project/anira/scope'

   defineStage('org.example.my-stage', (env) => ({
     preProcess(ctx, prepared) { /* ... */ },
   }))

The factory builds the scope's instance over its ``env``: ``env.abi`` (the bindings over the
scope's module instance) and ``env.module`` (its heap views: ``env.module.HEAPF32``, ...). Each
method receives pointers into the shared memory: ``ctx`` an ``anira_stage_ctx`` (``layout``),
``prepared`` what this handler's ``prepare`` stored. A method returns an ``anira_status``
(nothing is ``ANIRA_OK``); one that throws fails its call. A stage without ``preProcess`` or
``postProcess`` runs anira's default body there.

The phases run in different scopes (:doc:`architecture`): ``preProcess``, ``postProcess`` and
``reset`` in the audio worklet under a Hard contract, ``beforeInference`` and
``afterInference`` on the inference Workers, ``init``, ``prepare``, ``unprepare`` and
``release`` on the main thread. Define the stage in each scope its phases need: import the
module on the main thread, list it in the context's ``workerModule`` for the Workers' phases,
and in ``createAudioNode``'s ``workletModules`` for the worklet's. One name is one stage
(:js:func:`~registry.keyOf`).

Adding It to the Pipeline
-------------------------

.. code-block:: ts

   pipeline.addStage('org.example.my-stage', {
     hooks: ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_RESET | ANIRA_EM_HOOK_PREPARE | ANIRA_EM_HOOK_UNPREPARE,
   })

``hooks`` names the phases that cross into JavaScript; a phase without its bit runs anira's
default and never crosses. ``flags`` is the stage's real-time promise, stated, never derived:
the default ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST`` says ``preProcess``, ``postProcess`` and
``reset`` allocate nothing, lock nothing and wait for nothing, which a Hard contract requires of
them (they run on the audio thread); ``ANIRA_STAGE_FLAG_REALTIME_HOOKS`` says the same of
``beforeInference`` and ``afterInference``.

The Example's Stage
-------------------

``InputGainStage`` of ``web/examples/stage-and-engine/`` scales the model's audio input by the
Static input ``gain`` in ``preProcess``, ramping from the previous block's gain across the
block. Its ``prepare`` and ``unprepare`` run on the main thread and allocate its per-handler
memory in the shared heap; ``preProcess`` and ``reset`` run in the worklet and reach that memory
through ``prepared``; ``preProcess`` reads the tensors through the stage accessors
(``_anira_stage_input_tensor``, ``anira_tensor_data_f32``). The same module is imported by the
page (``main.ts``) and bundled into the worklet:

.. literalinclude:: ../../../web/examples/stage-and-engine/gain-stage.ts
   :language: ts
