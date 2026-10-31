Custom Processors
=================

A JavaScript processor is the pipeline's one processor written in TypeScript: the counterpart of a C
``anira_processor_desc`` (:doc:`../custom_preprocessing`), defined by name in every scope that runs
one of its phases, with no rebuild of the module.

Defining a Processor
--------------------

.. code-block:: ts

   import { defineProcessor } from '@anira-project/anira/scope'

   defineProcessor('org.example.my-processor', (env) => ({
     preProcess(ctx, prepared) { /* ... */ },
   }))

The factory builds the scope's instance over its ``env``: ``env.abi`` (the bindings over the
scope's module instance) and ``env.module`` (its heap views: ``env.module.HEAPF32``, ...). Each
method receives pointers into the shared memory: ``ctx`` an ``anira_processor_ctx`` (``layout``),
``prepared`` what this handler's ``prepare`` stored. A method returns an ``anira_status``
(nothing is ``ANIRA_OK``); one that throws fails its call. A processor without ``preProcess`` or
``postProcess`` runs anira's default body there.

The phases run in different scopes (:doc:`architecture`): ``preProcess``, ``postProcess`` and
``reset`` in the audio worklet under a Hard contract, ``beforeInference`` and
``afterInference`` on the inference Workers, ``init``, ``prepare``, ``unprepare`` and
``release`` on the main thread. Define the processor in each scope its phases need: import the
module on the main thread, list it in the context's ``workerModule`` for the Workers' phases,
and in ``createAudioNode``'s ``workletModules`` for the worklet's. One name is one processor
(:js:func:`~registry.keyOf`).

Adding It to the Pipeline
-------------------------

.. code-block:: ts

   pipeline.addProcessor('org.example.my-processor', {
     phases:
       ANIRA_EM_PROCESSOR_HOOK_PRE | ANIRA_EM_PROCESSOR_HOOK_RESET | ANIRA_EM_PROCESSOR_HOOK_PREPARE | ANIRA_EM_PROCESSOR_HOOK_UNPREPARE,
   })

``phases`` names the phases that cross into JavaScript; a phase without its bit runs anira's default
and never crosses. Each bit is ``1 << ANIRA_PHASE_*``, one index space with
``anira::Processor::k_*`` of ``anira.hpp`` (``ANIRA_EM_PROCESSOR_HOOK_AFTER`` is
``Processor::k_after_inference``, 16), the lifecycle phases (prepare, release, reset, unprepare,
init) above the data phases. ``flags`` is the processor's real-time promise, stated, never derived:
the default ``ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST`` says ``preProcess``, ``postProcess`` and
``reset`` allocate nothing, lock nothing and wait for nothing, which a Hard contract requires of
them (they run on the audio thread); ``ANIRA_PROCESSOR_FLAG_REALTIME_HOOKS`` says the same of
``beforeInference`` and ``afterInference``.

The Example's Processor
-----------------------

``INPUT_GAIN_PROCESSOR`` of ``web/examples/processor-and-engine/`` scales the model's audio input by
the Static input ``gain`` in ``preProcess``, ramping from the previous block's gain across the
block. Its ``prepare`` and ``unprepare`` run on the main thread and allocate its per-handler memory
in the shared heap; ``preProcess`` and ``reset`` run in the worklet and reach that memory through
``prepared``; ``preProcess`` reads the tensors through the processor accessors
(``_anira_processor_input_tensor``, ``anira_tensor_data_f32``). The same module is imported by the
page (``main.ts``) and bundled into the worklet:

.. literalinclude:: ../../../web/examples/processor-and-engine/gain-processor.ts
   :language: ts
