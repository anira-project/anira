.. _usage-handler:

The pipeline and the handler
============================

A **pipeline** says what a handler runs: one inference stage (the model config, or several
variants of one form, and its candidate backends), at most one processor, and the custom engines
its entries name. An
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
  and ``onnxruntime`` make two plans of one neutral entry. The plan table is in variant order
  (:ref:`usage-variants`), then entry order, then candidate order. A candidate's provider is
  checked for its syntax here; one its engine cannot
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
handler's own, valid until its next prepare or destroy. A variant added or removed at run time
makes a new report, and ``anira_handler_plan_report`` returns the newest
(:ref:`usage-variant-set`). Its rows follow the enumeration convention
(:ref:`usage-c-api`; ``plans()``, ``slots(plan, inputs)``, ``extensions(plan)``):

- **One** ``anira_plan_info`` **per plan:** ``variant``, ``engine`` / ``engine_id``, ``provider`` /
  ``provider_id``, a registered engine's ``engine_flags``, ``budget_ms``, the device's
  ``memory_kind`` (:ref:`usage-edges`), what the warm-up measured (below), and ``state``
  (``ANIRA_PLAN_READY``, or ``ANIRA_PLAN_REMOVED`` for a plan of a removed variant).
- **One** ``anira_plan_slot`` **per slot of either side:** ``role``, ``binding``
  (``ANIRA_BINDING_NAME``, ``POSITION``, or ``ENGINE`` for a registered engine, which bound itself),
  ``dtype`` and ``ring_dtype`` (0 without a ring), ``window``, ``hop`` and ``window_derived`` (0 on a
  slot without a window), the edge: ``from_domain`` / ``to_domain``, ``edge_class`` and
  ``moved_by`` with its reason (:ref:`usage-edges`), the core's wait strategy, recipe ``"host"``.
- **One** ``anira_plan_ext`` **per extension** a plan consumes, naming its consumer.

``report.find(query)`` (``anira_plan_report_find``) finds the first plan, in dense-index order,
with every property an :cpp:struct:`anira::PlanQuery` names (``variant``, ``backend.engine`` /
``engine_id``, ``backend.provider`` / ``provider_id``; a field left out matches every plan),
``std::nullopt`` (``ANIRA_ERROR_NOT_FOUND``) when none has them; a row marked
``ANIRA_PLAN_REMOVED`` is never found. In C start from
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
audio it produces. ``rt_parallel`` is the plan's parallel inferences (1 for an exclusive handler,
else the instances of the plan's own variant), and
the **load** ``rt_factor / rt_parallel`` the share of that capacity one hop takes.
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

A plan is a dense index ``0 .. num_plans - 1``. ``set_plan(plan)`` selects one and ``get_plan()``
reads the selection (``anira_handler_set_plan`` / ``_get_plan``): one relaxed store of
the index, ``ANIRA_NONBLOCKING`` and callable from any thread (a UI may switch while audio runs),
but not while ``prepare`` runs, which rebuilds the report. Two plans on one engine (two variants,
two providers) stay distinct. The switch applies from the next
submitted chunk: a chunk keeps the plan it was stamped with from its pre- to its post-processing,
so exactly one engine runs for every chunk. A switch inside one variant keeps the model's state;
what a switch to another variant does to it is the contract's (:ref:`usage-variant-state`).
``set_model(variant)`` and ``set_backend(backend)`` move one axis of the same selection and keep
the other (:ref:`usage-variant-switch`). An index out of range is a no-op recorded as
``ANIRA_ERROR_CONFIG`` in ``rt_error()``, the index of a removed plan one recorded as
``ANIRA_ERROR_NOT_FOUND``. The three selection entries read the plan table through four reader
slots: when more than four calls of them overlap on one handler, a call returns
``ANIRA_ERROR_CAPACITY`` with the selection unchanged and nothing recorded, and the caller calls
again. Every prepare starts on the default plan again.

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
samples of the stream are zeros. A plan switch does not change it, and neither does a variant
added or removed at run time.

.. _usage-variants:

Model variants
--------------

A **variant** is one more model of the inference stage's form. The variants of a stage differ in
what they compute (other weights, another training of one architecture) and in nothing the stream
can see. One handler loads and measures them together, runs them in one stream and switches
between them while audio runs, as it switches backends. A model of another form (another shape,
another hop, another tensor) is no variant: it is another handler, which the host creates in
place of this one or beside it.

.. code-block:: cpp

    // half, twice, inverted: three anira::ModelConfig of one architecture (a gain of 0.5, 2 and -1)
    anira::Pipeline pipeline;
    pipeline.add_inference({half, twice, inverted}, {});      // variants 0, 1, 2; {}: the default candidates
    anira::InferenceHandler handler(context, pipeline);
    handler.prepare(contract);                                // every plan of every variant, loaded and measured

    handler.set_model(2);                                     // the inverted gain, on the backend that runs now
    handler.set_backend({.engine = ANIRA_ENGINE_LIBTORCH});   // the same model on LibTorch

The bundled VariantGainNetwork is such a set:
``extras/models/model-pool/variant_gain_half.model.json``, ``variant_gain_double.model.json`` and
``variant_gain_invert.model.json`` under the one contract file ``variant_gain.contract.json``
(:doc:`../examples`).

In C the variants are the list ``anira_pipeline_add_inference`` takes, and the two selection
entries return their status:

.. code-block:: c

    const anira_model_config* variants[] = { half, twice, inverted };
    anira_pipeline_add_inference(pipe, variants, 3, NULL, 0, &err);   /* NULL, 0: the default set */
    /* ... anira_handler_create, anira_handler_prepare ... */
    const anira_backend_id libtorch = { sizeof(anira_backend_id), ANIRA_ENGINE_LIBTORCH,
                                        ANIRA_PROVIDER_NONE, NULL, NULL };
    anira_status st = anira_handler_set_model(h, 2);
    if (st == ANIRA_OK) { st = anira_handler_set_backend(h, &libtorch); }
    if (st == ANIRA_ERROR_NOT_FOUND) { /* no such plan: the selection is as it was */ }

In TypeScript ``addInference`` takes the list, and the two selection methods throw an
:js:class:`AniraError` where C returns a failure:

.. code-block:: ts

    const pipeline = Pipeline.create(anira).addInference([half, twice, inverted])
    const handler = InferenceHandler.create(context, pipeline)
    handler.prepare(contract)
    handler.setModel(2)
    handler.setBackend({ engine: anira_engine.ANIRA_ENGINE_ONNXRUNTIME })

.. _usage-variant-form:

The form rule
~~~~~~~~~~~~~

Every variant has the form of the first one. The comparison is on the resolved quantities, so two
model files that spell one window differently are variants.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - A property
     - What it covers
   * - **Identical**
     - The number of tensors per side, and at every position the role, the canonical name, the
       dtype, the resolved extents, the channel count, the window in effect (minimum, maximum,
       overlap, the value a flexible one is pinned to), the hop and the Time axis; the anchor; the
       State pairs (which input is fed from which output); whether the handler is exclusive (a
       stateful model or a declared State pair).
   * - **Free per variant**
     - Its model entries (engines, providers, files or bytes), their layouts and tensor binding
       names, its extensions, ``default_engine`` and ``default_provider``, ``max_instances``, the
       internal latency (``latency`` on an output spec).
   * - **Reported, not refused**
     - Differing internal latencies, an added variant slower than the stream is sized for, and
       differing ``max_instances`` (:ref:`usage-variant-cost`).

The rule is checked three times: at ``anira_handler_create`` on the configurations alone (a
flexible window is compared as the smallest anchor hop pins it), at ``anira_handler_prepare`` under
the contract, and at ``anira_handler_add_variant``. A difference is ``ANIRA_ERROR_CONFIG``, and the
message names the variant, the slot, the property and both values:

.. code-block:: text

    variants[1]: input slot 0 'in': the hop is 512, but 256 in variants[0]; every variant of an inference stage has the form of the first one (a model of another form is another handler)

Every message about one variant of a stage with several carries the prefix ``variants[i]:``; the
messages of a stage with one variant are what they were.

The plan table over variants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **One plan per variant, entry and candidate.** A variant is more rows in the plan table: one
  plan per variant, model entry of that variant and candidate that names the entry's engine and a
  provider it accepts. The table is in variant order, then entry order, then candidate order, so
  the plans of one variant are consecutive rows. Three variants with an ONNX Runtime and a LibTorch
  entry each are six plans under the default candidate set of a build with both engines: plans 0
  and 1 run variant 0, plans 2 and 3 variant 1, plans 4 and 5 variant 2.
- **The variant's index** is its position in the list ``add_inference`` took, or the index
  ``add_variant`` returned. It is in every plan row (``anira_plan_info.variant``), in a plan query
  (``report.find({.variant = 1})``, with or without a backend beside it) and in the processor
  context (``ctx.variant()``, :ref:`processor-accessors`).
- **The default candidate set** is the union over the entries of every variant. A variant may
  have fewer plans than another (an entry for one engine only); a variant left with no plan is
  ``ANIRA_ERROR_CONFIG`` at ``anira_handler_create``, naming the variant.
- **Prepare** loads the model of every plan of every variant, and the contract's warm-up pass
  covers every plan (:ref:`usage-budget-warmup`). Every row carries its own figure and verdict,
  and under a measured budget the session runs the slowest plan's, so a switch to any plan stays
  within the latency the handler reports. The handler starts on the first variant's default plan
  (:ref:`usage-default-plan`); the defaults of the other variants are not read.

.. _usage-variant-switch:

Switching
~~~~~~~~~

Three entries move the one selection ``get_plan()`` reads. All three are ``[thread-safe]`` and
``ANIRA_NONBLOCKING``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry
     - The plan it selects
   * - ``set_plan(plan)``
     - The plan of that dense index: both axes at once.
   * - ``set_model(variant)``
     - The plan of that variant on the selected plan's backend (its engine and its provider,
       exactly).
   * - ``set_backend(backend)``
     - The first plan, in dense-index order, of the selected plan's variant with every property
       the backend names: an engine and a provider, each exact or left out, as ``find`` reads
       them. ``set_backend({})`` is the variant's first plan.

- **The switch is seamless.** The next chunk the driver thread forms runs on the selected plan. A
  chunk that is queued or in flight finishes on the plan it was formed on and is delivered. Nothing
  restarts, the switch drops and fills nothing, and the latency and the entry count stay as they
  are.
- **One axis moves, the other stays.** ``set_model`` and ``set_backend`` are each a
  compare-and-swap on the selection: the plan is resolved against the plan selected at that
  moment and stored only while that plan is still selected. A ``set_model`` and a ``set_backend``
  from two threads both land, in either order.
- ``ANIRA_ERROR_NOT_FOUND`` **changes nothing.** When the variant has no plan on that backend (its
  entries run on other engines or providers, the index names no variant, or the variant was
  removed), the selection stays as it is, and the status is recorded in ``rt_error()`` and logged
  once per prepare or reset. In C and C++ the entry returns the status; in TypeScript the method
  throws.
- ``set_backend`` refuses a backend against the pair rule (:ref:`usage-providers`) with
  ``ANIRA_ERROR_INVALID_ARGUMENT``. All three return ``ANIRA_ERROR_NOT_PREPARED`` before a
  successful prepare and ``ANIRA_ERROR_CAPACITY`` under more than four overlapping calls
  (:ref:`usage-select-plan`).

.. _usage-variant-state:

The state of a switch
~~~~~~~~~~~~~~~~~~~~~

What the model's state is behind a switch to another variant is an option of the Hard contract,
read at prepare: ``Hard::variant_state`` (``anira_contract_hard_set_variant_state``, the contract
file's ``"variant_state"``, ``variantState`` of :js:meth:`ContractHandle.hard`), an
``anira_variant_state``.

.. code-block:: cpp

    anira::Hard hard{.host_block_min = 512, .host_block_max = 512, .host_rate = 48000,
                     .variant_state = ANIRA_VARIANT_STATE_KEEP};   // the default: ANIRA_VARIANT_STATE_FRESH

- ``ANIRA_VARIANT_STATE_FRESH`` (the default, ``"fresh"``): the new variant starts from its
  initial state. The driver thread notes the switch when it forms the first chunk on a plan of
  another variant than the chunk before it. Ahead of that chunk's inference, on the inference
  thread, every declared State pair is re-initialised to zeros (:ref:`usage-state`) and, for an
  exclusive handler, the engine of the chunk's plan is reset through its ``reset``. Every plan is
  reset at its own first inference behind the switch. A switch back is a switch: A, B, A starts A
  fresh again.
- ``ANIRA_VARIANT_STATE_KEEP`` (``"keep"``): nothing happens at the switch. The new variant's
  first inference reads the declared State pairs as the previous variant's last one left them,
  which the one form makes the same tensors.
- **A backend switch inside one variant always keeps the declared state,** under either value:
  every plan binds the same State pairs.
- **State an engine keeps inside itself** (a hidden state the model does not declare) is never
  carried from one plan's engine to another's. Under ``KEEP`` each engine continues from where it
  stopped; under ``FRESH`` it starts over.
- The stream continues under both values: chunks formed ahead of the switch finish on the state
  they ran with, the processor's ``reset`` is not called, and the latency does not change.

.. _usage-variant-set:

Adding and removing at run time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A handler takes one more variant, and gives one back, while its stream plays.

.. code-block:: cpp

    const uint32_t added = handler.add_variant(another);      // blocks: the load, the warm-up; the stream plays on
    handler.set_model(added);                                 // from the next chunk
    handler.remove_variant(0);                                // blocks until variant 0's models are released
    const anira::PlanReport report = handler.plan_report();   // the newest: variant 0's rows read ANIRA_PLAN_REMOVED

In C ``anira_handler_add_variant(h, model, &variant, &err)`` and
``anira_handler_remove_variant(h, variant, &err)``; in TypeScript ``handler.addVariant(model)``
and ``handler.removeVariant(variant)``
(:cpp:func:`anira::InferenceHandler::add_variant`,
:cpp:func:`anira::InferenceHandler::remove_variant`, :js:meth:`InferenceHandler.addVariant`,
:js:meth:`InferenceHandler.removeVariant`). Both are **control entries**, like ``prepare`` and
``destroy``: they block their caller, none is real-time, and one control entry runs on a handler
at a time (a second one on another thread waits for the first). Unlike ``prepare`` they are legal
while the driver thread and the inference threads run. A failed call throws ``anira::Error`` (an
:js:class:`AniraError`) and changes nothing.

**Adding.** On an unprepared handler the model is validated as ``anira_handler_create`` validates
it and joins the handler's copy of the pipeline; the next prepare builds it. On a prepared handler
the call, in order:

1. validates the model under the contract of the last prepare and against the handler's form
   (:ref:`usage-variant-form`);
2. derives its plans from the handler's candidates: under the default candidate set the set grows
   by the new variant's entries, under named candidates it stays, and a variant left without a
   plan is ``ANIRA_ERROR_CONFIG``;
3. builds and measures the plans beside the stream, in a session of its own with its own chunks
   and its own State buffers: the models load, the engines' handles are prepared for the entry
   count of the playing stream, a plan above the contract's edge cost is dropped as at prepare,
   and the contract's warm-up pass runs for every new plan whose model has no stored figure;
4. appends the plans to the plan table behind the existing ones and publishes a new plan report.

- **It waits for** the model loads, the engines' prepares and the warm-up pass: as long as a
  prepare takes for those plans. A warm-up needs an inference thread in its loop, as at prepare.
- **It never touches** the playing stream: not its chunks, not its rings, not its declared state.
  The selection, the latency and the entry count do not change, and the build's session is dropped
  when the call returns.
- **It never calls the processor.** The processor is prepared once per handler, for shapes every
  variant has, and no phase of it runs for the build. The figure of an added plan is therefore the
  engine's alone, without the processor's ``before_inference`` and ``after_inference``; a figure
  an earlier prepare measured with them is reused, and the next prepare measures the whole.
- **A failed add publishes nothing** and releases what it built: the report, the plan count, the
  latency, the entry count and ``rt_error()`` are as before the call.
- **Refused** with ``ANIRA_ERROR_CONFIG``: a rule the model breaks, another form, a variant
  without a plan, and a plan whose engine binds a chunk's buffers on a coarser boundary than the
  chunks of the playing stream start on (only a prepare builds the chunks). What fails a prepare
  for a plan fails the add with the same status: a model that does not load, the status an
  engine's ``prepare`` returned, ``ANIRA_ERROR_EDGE_UNREACHABLE`` when no plan of the variant is
  within the edge cost, ``ANIRA_ERROR_ENGINE`` for a failed inference of the warm-up, and
  ``ANIRA_ERROR_INVALID_STATE`` for a warm-up without an inference thread in its loop or with an
  inference not complete within 10 s.
- The load and the warm-up share the inference threads with the playing stream, so misses during
  the call are possible.

**Removing.** ``remove_variant`` refuses an index that names no variant, or one removed before
(``ANIRA_ERROR_NOT_FOUND``), the last variant the handler has (``ANIRA_ERROR_INVALID_STATE``: a
handler keeps one), and the variant of the selected plan (``ANIRA_ERROR_INVALID_STATE``: select a
plan of another variant first). On an unprepared handler the variant leaves the handler's copy of
the pipeline. On a prepared handler the call, in order:

1. publishes a new plan report in which the variant's rows are marked ``ANIRA_PLAN_REMOVED``
   (``anira_plan_info.state``). From here on ``find``, ``set_model`` and ``set_backend`` skip those
   rows, and ``set_plan`` on one is ``ANIRA_ERROR_NOT_FOUND``. A selection of such a plan that
   raced the call is moved to the first ready plan, with one Warning record;
2. waits until no entry of the driver thread that could still form a chunk on one of the plans is
   running. A ``_wait`` twin inside its wait is waited for, for its timeout;
3. waits until every chunk formed on one of the plans has left the inference threads. A running
   inference of the variant is waited for, without a timeout. Those chunks are delivered as every
   chunk is;
4. gives the engines' prepared handles back (a registered engine's ``unprepare``) and releases
   the loaded models, on the calling thread. A model that no other plan of any handler runs is
   unloaded (the engine's ``unload``); one that another handler, or another variant of this one,
   still runs stays loaded.

The call returns when the memory is released. A driver thread that is not running holds nothing
back. The selection (unless it raced), the entry count and the processor's prepared object do not
change, and neither does the latency: the handler reports what its last prepare computed, also
when the removed variant had the largest internal latency.

**Indices are stable between two prepares.**

- Plans are only appended. A removed plan keeps its row and its index, and the row says what the
  plan was.
- A variant's index is never given to another variant: ``add_variant`` continues behind the
  highest index the stage has had.
- Every add and every removal makes a new report and leaves the earlier ones as they were. A
  ``PlanReport`` taken earlier stays valid, with the rows it had, until the next prepare; call
  ``plan_report()`` again for the newest.
- **A prepare numbers the plans anew.** It builds the variants the handler has then: an added one
  is among them, a removed one is not. The remaining variants keep their indices, the plan table
  has no removed row, and the first variant the handler still has takes the role of the first (the
  form the others are compared with, the defaults that select the starting plan). A host that
  holds plan indices looks them up again in the report of every prepare.

**The callbacks of a control entry** (an engine's or a processor's ``init``, ``load``,
``prepare``, ``unprepare`` or ``unload``) run on the thread that called the entry.

- From inside one, every control entry of the same handler (``prepare``, ``add_variant``,
  ``remove_variant``, ``destroy``) is refused with ``ANIRA_ERROR_INVALID_STATE``; ``destroy`` does
  nothing and logs one Error record. The handler's getters answer.
- The callbacks of an add or a removal run beside the playing stream. An engine's ``prepare`` of
  an add carries ``ANIRA_PREPARE_LIVE`` in its record's ``flags``
  (:ref:`a prepare beside running phases <engines-live>`). From inside a callback of either call
  the two Static entries are refused with ``ANIRA_ERROR_INVALID_STATE`` too (returned, not recorded
  in ``rt_error()``): the driver thread is the one writer of a Static store while the stream plays.

.. _usage-variant-cost:

Warnings and cost
~~~~~~~~~~~~~~~~~

What differs between variants without breaking the form rule is reported, never refused. The
first three are one Warning record each, in the group ``anira.capi`` (:doc:`../logging`); the
fourth is in the plan rows alone.

- **Internal latencies at prepare.** The handler reports the largest internal latency per output
  over its variants. A variant with less plays that many samples early against the reported
  latency.
- **The internal latency of an added variant.** The latency does not change in an add, so an added
  variant whose internal latency is not the reported one plays early or late by the difference,
  until a prepare computes the latency again.
- **An added variant slower than the stream.** A playing stream is sized for the budget of its
  last prepare: under a measured budget, the figure of the slowest plan it was prepared with. An
  added plan whose measured inference time is above that budget can miss blocks while it is
  selected, until a prepare sizes the stream again; its row carries its own figure and verdict.
- ``max_instances``. The stream is sized for the fewest instances any variant allows, and each
  plan row reports its own (``rt_parallel``).

.. code-block:: text

    anira_handler_prepare: the variants differ in their internal latency (output slot 0 'out': variants[0] 0, variants[1] 64 samples, reported 64 samples); the handler reports the largest, and a variant with less plays that many samples early
    anira_handler_add_variant: variants[1] differs from the handler in its internal latency (output slot 0 'out': 16 samples, the handler reports 0, so the variant plays 16 samples late against the reported latency); the latency does not change in this call, and a prepare computes it again
    anira_handler_add_variant: variants[2]: plan 2 (models[0] on onnxruntime) needs 4.210 ms per inference, the stream is sized for 1.842 ms; expect misses on it until the next prepare

**Every variant stays loaded** for as long as the handler is prepared, until it is removed:

- the memory is every variant's model on every plan it has (a model the core already holds for
  another handler is shared, :ref:`engines-sharing`);
- a prepare loads every plan and warms up every plan without a stored figure;
- under a measured budget the latency covers the slowest plan of any variant.

Remove a variant, or name fewer candidates, to pay less. The plan reports of the adds and removals
since the last prepare stay allocated until the next prepare.
