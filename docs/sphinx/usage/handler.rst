.. _usage-handler:

The pipeline and the handler
============================

A **pipeline** says what a handler runs: one inference stage (the model config and its candidate
backends), at most one processor, and the custom engines its entries name. An
:cpp:class:`anira::InferenceHandler` is created from it on a context and prepared with a contract.

.. code-block:: cpp

    anira::Pipeline pipeline;
    pipeline.add_inference(model);                            // the default candidate set
    // pipeline.add_inference(model, {anira::BackendId{...}}) // or named candidates
    // pipeline.add_processor(std::make_shared<MyProcessor>());   (custom_preprocessing)
    // pipeline.add_engine(std::make_shared<MyEngine>());         (custom_engines)
    anira::InferenceHandler handler(context, pipeline);       // copies the pipeline

    contract.hard_host_blocks(block_size, block_size, sample_rate);
    const anira::PlanReport report = handler.prepare(contract);   // the plans that run here

    const auto plans = report.plans();
    for (uint32_t i = 0; i < plans.size(); ++i) {
        std::printf("plan %u: budget %.3f ms, measured %.3f ms over %u inferences, "
                    "real-time factor %.2f of %u: verdict %u\n",
                    i, plans[i].budget_ms, plans[i].measured_ms, plans[i].warmup_iterations,
                    plans[i].rt_factor, plans[i].rt_parallel, plans[i].rt_verdict);
    }
    if (const std::optional<uint32_t> plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}})) {
        handler.set_plan(*plan);
    }
    const uint32_t latency = handler.get_latency(0);              // samples of output 0, for the host

.. _usage-pipeline:

The pipeline
------------

- **The candidates** are ``anira::BackendId`` rows (``anira_backend_id``: an engine on a provider,
  never ``ANIRA_PROVIDER_NONE``). Without them the default set applies: every engine of this build
  on ``ANIRA_PROVIDER_CPU``, every custom engine a model entry names, and every provider a pinned
  entry names. Under it every entry is one plan: a pinned entry on its pin, a neutral one on its
  engine's home provider (the CPU path, or a custom engine's first listed provider when it does not
  serve the CPU path). A plan whose provider its engine cannot use here is dropped, and an entry of
  an engine this build lacks is skipped; a variant left with no plan is ``ANIRA_ERROR_CONFIG``.
- **Named candidates are checked:** an entry is one plan per candidate that names its engine and a
  provider it accepts (a neutral entry any, a pinned entry its pin alone), so ``onnxruntime:coreml``
  and ``onnxruntime`` make two plans of one neutral entry. The plan table is in entry order, then
  candidate order. A candidate's provider is checked for its syntax here; one its engine cannot
  serve here is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``, and candidates that
  match no entry are ``ANIRA_ERROR_CONFIG`` there.
- **At most one processor:** a second ``add_processor`` is ``ANIRA_ERROR_INVALID_STATE``
  (:doc:`../custom_preprocessing`). **Custom engines** are the pipeline's, named by their ids in
  the model entries (:doc:`../custom_engines`).
- The handler copies the pipeline at create, so the pipeline (and your ``Processor`` and ``Engine``
  objects) may go right after; a later addition does not reach that handler. In C:
  ``anira_pipeline_create``, ``anira_pipeline_add_inference``, ``anira_pipeline_add_processor``,
  ``anira_pipeline_add_engine``, ``anira_handler_create(context, pipeline, &h, &err)``.

``anira_handler_create`` (the constructor) checks what the configuration alone decides and throws
``anira::Error`` otherwise: an entry whose engine id no engine of the pipeline serves
(``ANIRA_ERROR_NOT_SUPPORTED``), a default engine without an entry (``ANIRA_ERROR_CONFIG``), a
candidate's provider that its engine does not serve here (``ANIRA_ERROR_NOT_SUPPORTED``). The handler adds a reference to its
context, so the context may be destroyed by its creator first; it is move-only, and its destructor
drains its in-flight inferences.

.. _usage-prepare:

Prepare
-------

``prepare`` is the blocking quiescence point: never on the driver thread, never overlapped by
another entry of the handler, and called again for new host blocks (a new block size or rate),
which drains the in-flight work first. It takes a ``ContractHandle``, an ``anira::Hard``, an
``anira::Async`` or an ``anira::Contract``, and in order it:

1. validates the configuration against the contract (the rules of :ref:`usage-models` and
   :ref:`usage-contracts`, the processor's real-time promise against where its phases run);
2. runs the processor's ``init`` if it has not run yet (once per registration, before any model
   loads);
3. loads the model of every candidate with an entry, once per loaded model the core pools (a
   second handler on an equal model shares it);
4. sizes the rings for the block range and the budget, builds the plan table and the report, and
   selects the default engine's plan (:ref:`usage-default-plan`);
5. prepares every registered engine on its loaded model, then the processor's ``prepare``;
6. runs the contract's warm-up pass and sizes the session again for what it measured
   (:ref:`usage-budget-warmup`).

A status other than ``ANIRA_OK`` from any step fails the prepare with it and leaves the handler
unprepared. A second prepare replaces the session whole. ``anira_handler_num_entries(h)`` then says
how many chunks may be in flight at once, the size of a processor's per-entry scratch. What this
pre-release refuses at prepare: :ref:`usage-pre-release`.

.. _usage-plan-report:

The plan report
---------------

``prepare`` returns the :cpp:class:`anira::PlanReport` (``anira_handler_plan_report(h)``), the
handler's own, valid until its next prepare or destroy. Its rows follow the enumeration convention
(:ref:`usage-c-api`; ``plans()``, ``slots(plan, inputs)``, ``extensions(plan)``):

- **One** ``anira_plan_info`` **per plan:** ``variant``, ``engine`` / ``engine_id``, ``provider`` /
  ``provider_id``, a registered engine's ``engine_flags``, ``budget_ms``, the device's
  ``memory_kind`` (:ref:`usage-edges`), and what the warm-up measured (below).
- **One** ``anira_plan_slot`` **per slot of either side:** ``role``, ``binding``
  (``ANIRA_BINDING_NAME``, ``POSITION``, or ``ENGINE`` for a registered engine, which bound itself),
  ``dtype`` and ``ring_dtype`` (0 without a ring), ``window``, ``hop`` and ``window_derived`` (0 on a
  slot without a window), the edge: ``from_domain`` / ``to_domain``, ``edge_class`` and
  ``moved_by`` with its reason (:ref:`usage-edges`), the core's wait strategy, recipe ``"host"``.
- **One** ``anira_plan_ext`` **per extension** a plan consumes, naming its consumer.

``report.find(query)`` (``anira_plan_report_find``) finds the first plan, in dense-index order,
with every property an :cpp:struct:`anira::PlanQuery` names (``variant``, ``backend.engine`` /
``engine_id``, ``backend.provider`` / ``provider_id``; a field left out matches every plan),
``std::nullopt`` (``ANIRA_ERROR_NOT_FOUND``) when none has them. In C start from
``ANIRA_PLAN_QUERY_INIT``, which matches every plan (``ANIRA_VARIANT_ANY``, ``ANIRA_ENGINE_NONE``,
``ANIRA_PROVIDER_NONE``; a custom engine or provider with its id), never from a zeroed struct:
``variant`` 0 is a real variant and ``struct_size`` 0 is ``ANIRA_ERROR_INVALID_ARGUMENT``. It allocates nothing and is ``[thread-safe]`` and
``ANIRA_NONBLOCKING`` while the report is valid, so a host resolves a backend choice where it
applies it.

**What the warm-up measured, and whether the plan keeps up.** Every plan row carries the pass's
result: ``warmup_iterations``, ``warmup_outcome`` (``anira_warmup_outcome``: ``NONE`` without a
pass, ``FIXED``, ``STABLE``, ``CAPPED`` for a pass that did not settle in 32 inferences,
``REUSED`` for the model's figure from an earlier prepare), ``measured_ms`` (the maximum of the last
four inference times) and ``median_ms``. Beside it the **real-time factor** ``rt_factor``: the
inference time (``measured_ms``, or ``budget_ms`` without a measurement) over the anchor's hop
period, the latency model's :math:`\kappa` (:doc:`../latency`); 1.0 is an inference as long as the
audio it produces. ``rt_parallel`` is the session's parallel inferences (1 for an exclusive one), and
the **load** ``rt_factor / rt_parallel`` the share of the session's capacity one hop takes.
``rt_verdict`` (``anira_rt_verdict``) applies anira's thresholds:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - ``rt_verdict``
     - load
     - meaning
   * - ``ANIRA_RT_COMFORTABLE``
     - below 0.75
     - Misses are unlikely while the machine stays as it was at the prepare.
   * - ``ANIRA_RT_MARGINAL``
     - 0.75 to below 1
     - It keeps up on average, but load the measurement could not see (other handlers, other
       applications, the clock speed) can push it over: misses are likely under load.
   * - ``ANIRA_RT_NOT_REALTIME``
     - 1 or more
     - The pool cannot finish what the stream submits: misses are certain, and the miss policy
       fills them. The session is sized for one host block's burst, and the prepare succeeds.
   * - ``ANIRA_RT_UNMEASURED``
     -
     - No pass (an explicit budget with ``ANIRA_WARMUP_NONE`` or ``FIXED`` 0): the factor is the
       declared budget's,
       a statement about the contract, not the machine.

The verdict is an estimate at prepare: below the limit a likelihood, above it a certainty (a
quarter of headroom absorbs a third more inference time). The rows can be read at any time after the
prepare. **A host that plays live** acts on the row of the plan it
runs:

- it holds a ``NOT_REALTIME`` plan (bypassed or silent) and asks its user, "Try nonetheless", before
  it runs it, plays a ``MARGINAL`` one with a note ("marginal on this machine: may drop out under
  load"), and plays a ``COMFORTABLE`` one;
- it never gates an offline render (a DAW bounce, JUCE's ``isNonRealtime()``, CLAP's offline render
  mode, an ``OfflineAudioContext``), which does not run against the clock;
- after a re-prepare (a new device, block size or sample rate) it asks again only when the verdict
  got worse than the one its user accepted.

The JUCE, CLAP and web examples hold a ``NOT_REALTIME`` plan behind a "Try nonetheless" control
(:doc:`../examples`).

**The log.** Every successful prepare logs the measurement, then the report, in the group
``anira.capi``: one record per plan's measurement (Info; Warning for a marginal or not real-time
plan, a pass that did not settle and an explicit budget below its measurement), one for the session
(its budget, the worst verdict, the latency and the entries; Warning when the pool cannot keep up),
then the report at Info (set the context's level to ``ANIRA_LOG_INFO`` to see it): a head line, one
record per plan, per slot and per consumed extension.

.. code-block:: text

    anira_handler_prepare: plan 0 (onnxruntime, cpu): warm-up until_stable, 6 inferences, stable; measured 1.842 ms (median 1.701 ms) on the inference threads; real-time factor 0.17 of 4 parallel inferences (load 0.04): comfortable (96 % headroom)
    anira_handler_prepare: plan 1 (libtorch, cpu): warm-up until_stable, 5 inferences, stable; measured 48.120 ms (median 47.990 ms) on the inference threads; real-time factor 4.51 of 4 parallel inferences (load 1.13): NOT real-time on this machine; the stream will miss blocks, which on_miss fills
    anira_handler_prepare: budget 48.120 ms, measured (the slowest of 2 plans): not real-time; the pool cannot keep up (every hop brings 4.51 hop periods of inference for 4 parallel inferences): latency 3072 samples and 12 entries are sized for one host block's burst, and the stream will miss blocks
    anira_handler_prepare: plan report: plans 2, input slots 1, output slots 1, selected plan 0
    anira_handler_prepare: plan 0: variant 0, engine onnxruntime, provider cpu, budget 1.842 ms (measured), memory unified
    anira_handler_prepare: plan 0: input 0 'audio_in': host -> host, edge zero_copy (allocate zero_copy), moved by none, wait spin_backoff, recipe host, dtype float32, ring float32, window 13844, hop 512, bound by position
    ...

A slot line names how a window was chosen where the spec did not fix it: ``derived from the Time
extent``, ``derived, flexible, pinned to the anchor hop`` or ``flexible, pinned to the anchor hop``.

.. _usage-select-plan:

Selecting a plan
----------------

A plan is a dense index ``0 .. num_plans - 1``. ``set_plan(plan)`` / ``get_plan()``
(``anira_handler_set_plan`` / ``_get_plan``) are the whole runtime selection: one relaxed store of
the index, ``ANIRA_NONBLOCKING`` and callable from any thread (a UI may switch while audio runs),
but not while ``prepare`` runs, which rebuilds the report. Two plans on one engine (two variants,
two providers) stay distinct, and a switch keeps the model's state. The switch applies from the next
submitted chunk: a chunk keeps the plan it was stamped with from its pre- to its post-processing,
so exactly one engine runs for every chunk. An index out of range is a no-op recorded as
``ANIRA_ERROR_CONFIG`` in ``rt_error()``. Every prepare starts on the default plan again.

The engines a build may carry are ``ANIRA_ENGINE_ONNXRUNTIME`` (``"onnxruntime"``),
``ANIRA_ENGINE_LIBTORCH`` (``"libtorch"``), ``ANIRA_ENGINE_LITERT`` (``"litert"``, the default of
the TensorFlow Lite family), ``ANIRA_ENGINE_TFLITE`` (``"tflite"``, the legacy one, exclusive with
LiteRT), ``ANIRA_ENGINE_EXECUTORCH`` (``"executorch"``), and ``ANIRA_ENGINE_CUSTOM`` with an id;
``anira::enabled_engines()`` lists them.

.. _usage-latency:

The latency
-----------

``get_latency(slot)`` (slot 0 by default) and ``get_latencies()`` (one per output, ``0`` for a
Static or a State output; ``anira_handler_get_latency`` / ``_get_latencies``) are valid from prepare
on and constant until the next one: what a plugin reports to its host for compensation. anira
computes the figure from the host blocks, the budget and the model (:doc:`../latency`); a contract
that declares one per output replaces it (:ref:`usage-contract-options`). The first ``latency``
samples of the stream are zeros.
