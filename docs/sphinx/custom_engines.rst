Custom Engines
==============

An **engine** runs the model: it takes the model tensors of a chunk in, runs one inference and
writes the model tensors out. anira ships five (ONNX Runtime, ExecuTorch, LiteRT, LibTorch and
TensorFlow Lite, ``ANIRA_ENGINE_*``), and a host adds its own: a Core ML or a WebGPU runtime, an
engine the build does not carry, a synthetic model, a bypass that measures the pipeline's overhead.
A custom engine is one more implementation the inference stage resolves to, never a processor (the
pre- and post-processing around it is, :doc:`custom_preprocessing`): a model entry names it by its
id, and the handler loads it, prepares it, runs it and resets it exactly as it does a built-in
engine, with the same plan report, the same real-time rules and the same failure path.

.. toctree::
   :maxdepth: 1

   custom_engines/sharing
   custom_engines/examples

.. _engines-create:

Creating and adding an engine
-----------------------------

In C++ an engine is a subclass of :cpp:class:`anira::Engine`, constructed with its id; the pipeline
holds it:

.. code-block:: cpp

    class Gain : public anira::Engine {
    public:
        explicit Gain(float gain) : anira::Engine("com.example.gain"), m_gain(gain) {}
        std::unique_ptr<Loaded> load(const anira::EngineLoadInfo& info) override;   // the model
    private:
        float m_gain;
    };

    cfg.add_model_path("com.example.gain", "gain.bin");   // the entry the engine serves
    anira::Pipeline pipe;
    pipe.add_engine(std::make_shared<Gain>(0.5F));
    pipe.add_inference(cfg);

In C it is a descriptor, ``anira_engine_desc`` of ``anira/abi/engine.h``, made into an object:

.. code-block:: c

    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>   /* includes anira/abi/engine.h */

    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;   /* every slot NULL, no promise */
    desc.user_data = &my_engine;                       /* handed to every callback as it is */
    desc.process = my_process;                         /* the one required slot */
    desc.load = my_load;                               /* once per loaded model */
    desc.unload = my_unload;
    desc.prepare = my_prepare;                         /* once per handler on it */
    desc.unprepare = my_unprepare;

    anira_custom_engine* engine = NULL;
    anira_custom_engine_create("com.example.myengine", &desc, &engine, &err); /* the object */
    anira_pipeline_add_engine(pipe, engine, &err);
    anira_custom_engine_destroy(engine);          /* the pipeline keeps its own reference */

    uint32_t row = 0;                                  /* the entry the engine serves */
    anira_model_config_add_model_path(cfg, ANIRA_ENGINE_CUSTOM, "com.example.myengine", "model.bin",
                                      &row, &err);

:doc:`custom_engines/examples` has the whole engine in both languages. Two calls, two jobs:

- ``anira_custom_engine_create(id, &desc, &engine, &err)`` copies the descriptor (within
  ``struct_size``, with its strings) into a refcounted **engine object** under an **id**. The object
  is what the engine *is*: anira recognises the same engine by it and shares loaded models between
  the handlers that run it (:ref:`engines-sharing`). The id is how model entries name the engine,
  and every pipeline, message and plan report row about it uses that one id.
- ``anira_pipeline_add_engine(pipe, engine, &err)`` adds it to a pipeline. One engine may be added
  to any number of pipelines; the engines of one pipeline have distinct ids. The id is unique per
  pipeline, not per process: two plugin instances that each create their own engine under one id
  share nothing.

The handle, every pipeline the engine was added to, every handler created from them and every loaded
model of the engine hold a reference. ``anira_custom_engine_destroy`` drops the handle's, so it may
run right after the last addition; ``release`` fires exactly once, when the last reference is gone.
A **detached** handle (``anira_custom_engine_detach``) still names the engine for pipelines yet to
come without keeping it alive: the addition is ``ANIRA_ERROR_INVALID_STATE`` once the engine was
released, and ``anira_custom_engine_destroy`` still frees the handle.

- **The id** is a reverse-URI name: it contains a ``.``, and the prefix ``anira.`` is anira's own,
  except ``anira.v2.custom``, the 2.x ``CUSTOM`` backend's id (:ref:`migration-runtime`).
- **Adding** is legal before or after ``anira_pipeline_add_inference``; a handler copies the
  pipeline at ``anira_handler_create``, so an engine added afterwards does not reach it.
- **A model entry** names the engine as ``ANIRA_ENGINE_CUSTOM`` with the id (the pair rule;
  ``add_model_path`` / ``add_model_bytes``, ``"engine": "com.example.myengine"`` in a model file).
  Several entries are several plans, which ``anira_handler_set_plan`` switches between, resolved by
  entry. A candidate names the engine on a provider (``anira_backend_id``: ``engine_id`` with
  ``provider``, or ``provider_id`` for a provider the enum does not name); a plan is one entry on one
  provider: a neutral entry is a plan per candidate of its engine, a pinned one a plan on its pin's
  candidate alone (:ref:`usage-providers`).
- An added engine no entry names is not a plan and not an error; an entry whose id no engine of the
  pipeline serves is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``, naming the id. anira
  never opens a custom entry's path or reads its bytes: the engine's ``load`` does.

``anira_custom_engine_create`` refuses with ``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL`` id,
descriptor or ``out``, an id without a ``.`` or with the prefix ``anira.``, a ``struct_size`` below
the three leading slots (``struct_size``, ``abi_version``, ``user_data``), a ``flags`` bit the header
does not define, a ``NULL`` ``process``, a ``NULL`` ``consumed_kinds`` (or entry) with a count above
0 and a ``NULL`` ``providers`` (or a ``NULL`` or empty entry) with a count above 0, and with
``ANIRA_ERROR_ABI_VERSION`` an ``abi_version`` ``anira_check_abi`` refuses; a refused create hands
out nothing and never calls ``release``. ``anira_pipeline_add_engine`` refuses with
``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL`` pipeline or engine, and with ``ANIRA_ERROR_INVALID_STATE``
an engine whose id the pipeline already has; a refused addition takes no reference.

.. _engines-descriptor:

The descriptor
--------------

``anira_engine_desc`` is Tier 2 (``struct_size`` first, ``user_data`` third, growth at the tail);
``ANIRA_ENGINE_DESC_INIT`` is an engine without a callback and without a promise.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Slot
     - What it is
   * - ``user_data``
     - Handed to every callback as it is, never read by anira. One per engine object, shared by
       every pipeline, loaded model and handler of it.
   * - ``consumed_kinds``
     - The extensions the engine reads at load, ``"<host>:<kind>"`` strings (hosts ``tensor_spec``,
       ``model``, ``model_config``, ``context``, ``contract``), copied. They join the consumed-or-fail
       walk for the engine's entries, keyed by its id: its ``model:`` kinds read its own entries
       alone, its other kinds any host, and each consumed one is an ``anira_plan_ext`` row naming the
       engine's id. ``"context:provider_options"`` delivers the context's provider options for its
       backend at ``load``; a set for an engine that does not list it is refused at
       ``anira_handler_create``.
   * - ``flags``
     - The engine's promises (:ref:`engines-flags`); 0 promises nothing.
   * - ``providers``
     - The providers the engine serves, as strings, copied: exactly the listed ones, the CPU path
       only if the list has ``"cpu"``. The JSON spellings of ``anira_provider`` (``"cpu"``,
       ``"cuda"``, ``"webgpu"``, ``"vulkan"``, ``"directml"``, ``"coreml"``, ``"xnnpack"``) name a
       provider of the enum; any other string is a custom provider in the engine's own vocabulary
       (a reverse-URI name by convention), which a candidate names as ``ANIRA_PROVIDER_CUSTOM`` with
       ``provider_id`` and an entry's pin under ``"provider"``; ``"default"`` and ``"none"`` are
       refused at create. ``NULL`` with a count of 0 serves ``ANIRA_PROVIDER_CPU`` alone.
   * - ``process``
     - The engine call (``anira_engine_process_fn``). Required.
   * - ``reset``, ``prepare``, ``unprepare``, ``load``, ``unload``, ``init``, ``release``
     - The lifecycle (below). ``NULL``: nothing to do; without ``load`` or ``prepare`` the
       ``loaded`` or ``prepared`` pointer is ``NULL`` in every later call.
   * - ``query``
     - Which of ``providers`` are usable here, now (:ref:`engines-capabilities`). ``NULL``: every
       listed provider is.
   * - ``edge``
     - Where the engine takes each slot's bytes, when it is not the provider rule
       (:ref:`engines-edges`).

Under the default candidate set a neutral entry of the engine runs on the CPU path when the engine
serves it, else on the first listed provider, and a plan on a provider the query clears is dropped
there (a named candidate on it is refused). A candidate naming a provider the list lacks, or one the
query reports unavailable, is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``, naming the
engine, the provider and what is served; ``load`` may still refuse one it cannot serve at run time (a
device lost since).

Three levels, three pointers, three lifetimes: ``user_data`` lives with the engine object,
``loaded`` with one loaded model of it, ``prepared`` with one handler on that loaded model. The
processor's descriptor has the same slots minus the model level (:ref:`processor-lifecycle`), and its
``init`` and ``prepare`` receive the same two records of ``anira/abi/lifecycle.h``. ``anira_phase``
names the phase of every slot, ``ANIRA_PHASE_INIT``, ``_LOAD`` and ``_UNLOAD`` among them.

.. _engines-lifecycle:

The lifecycle
-------------

``init`` runs once per engine object, ``load`` once per loaded model anira pools, ``prepare`` once
per handler on a loaded model (twice when a measured warm-up moves the entry count, below); every
per-chunk call receives ``(ctx, prepared, user_data)`` with the
loaded pointer in ``ctx->loaded``; ``unprepare`` and ``unload`` give the two pointers back;
``release`` fires once per engine object. A refusal fails that ``anira_handler_prepare``, the message
naming the engine (``the engine 'com.example.myengine' refused load: it returned N``).

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Slot
     - When, and what for
   * - ``init(info, user_data)``
     - Once per engine object, at the first ``anira_handler_prepare`` that reaches it (a row of the
       engine survives validation), on that caller's thread under the core's lifecycle lock, after the
       processor's ``init`` and right before the engine's first ``load``. Build what the engine keeps
       for its whole life: a device context, a thread pool, a weights cache. It may allocate, log and
       query the context's capabilities, never call an entry that takes that lock (context create and
       destroy, handler create, destroy and prepare, ``anira_inference_thread_create``,
       ``anira_shutdown``, ``anira_release_core_if_idle``). Its ``anira_init_info``: ``log_level``
       (the core's, the most verbose of its contexts'), ``num_threads`` (the pool's size; 0 when the
       context brought its own threads or on a target without threads), ``context`` (that handler's,
       its capabilities legal to query, never kept). A refusal leaves the object uninitialised and
       the next prepare calls ``init`` again.
   * - ``load(info, user_data, &loaded)``
     - Once per loaded model (:ref:`engines-sharing`), on the caller of ``anira_handler_prepare``,
       under the lock, after ``init``: load the entry's model, bind the slots, size the shared call
       slots. It may allocate, read files, log, call the config getters and the tensor accessors,
       never an entry that takes that lock (the list under ``init``). Hands back the ``loaded``
       pointer (``NULL`` is legal) that every ``prepare`` of this loaded
       model, every ``process`` and ``reset`` (``ctx->loaded``) and ``unload`` receive. A refusal
       owes no ``unload``.
   * - ``prepare(info, loaded, user_data, &prepared)``
     - Once per handler and loaded model, on the caller of ``anira_handler_prepare``, after the plan
       report is built and while the handler already counts as prepared (its getters answer), before
       the processor's ``prepare``. Build
       what the engine keeps per handler (an exclusive handler's own executor). The prepares and
       unprepares of one loaded model never overlap (two handlers prepared on two threads reach it one
       after the other; different loaded models may run at once). It may call the
       ``[callback-safe]`` entries, the handler's getters and the two Static entries, never
       ``prepare``, ``destroy`` or a Hard entry. Hands back the ``prepared`` pointer (``NULL`` legal)
       every ``process``, ``reset`` and ``unprepare`` of this handler on this loaded model receives. A
       refusal leaves the handler unprepared, owes no ``unprepare``, and the loaded model goes back
       (an ``unload`` when the handler was its last holder).
   * - ``process(ctx, prepared, user_data)``
     - One inference, on an inference thread, in ``ANIRA_PHASE_INFERENCE`` (between the processor's
       ``before_inference`` and ``after_inference``). A **shared call** (a handler that is not
       exclusive) claims one of the loaded model's shared slots and runs on it; an **exclusive call**
       claims nothing and runs on what the handler's ``prepare`` built. It may block, never allocate
       per call.
   * - ``reset(ctx, prepared, user_data)``
     - Re-initialises what an **exclusive** handler keeps per stream (a hidden state the model does
       not declare, a variable tensor, a filter's history): at the first inference of a new stream
       (after ``prepare``, after ``anira_handler_reset``), on the inference thread, right before its
       ``process``, with that inference's context. The boundary is the plan's: every plan is reset at
       its own first inference of the new stream, and a plan switch alone is none. Never for a
       stateless handler. Of the built-in engines only TensorFlow Lite has a body
       (``TfLiteInterpreterResetVariableTensors``).
   * - ``unprepare(prepared, user_data)``
     - Once per successful prepare, on the thread of the handler's next ``anira_handler_prepare``
       (once the previous session is released, before the new prepare; a next prepare that fails
       earlier unprepares the previous one all the same), inside a prepare whose warm-up pass moved
       the entry count (before the prepare with the final record), or of its
       ``anira_handler_destroy``: after the handler's in-flight inferences drained and before the
       loaded model is released.
   * - ``unload(loaded, user_data)``
     - Once per successful load, on the thread of the prepare or destroy that drops the last handler
       holding the loaded model, after every ``unprepare`` of it. It may run under the lock (a later
       plan of the same prepare failed), so like ``load`` it never calls an entry that takes it.
   * - ``release(user_data)``
     - Once, when the last reference to the engine object dies (``anira_custom_engine_destroy``,
       ``anira_custom_engine_detach``, ``anira_pipeline_destroy`` or ``anira_handler_destroy``,
       whichever comes last), after every
       ``unload``, whether or not ``init`` ran. No callback runs afterwards.

**A built-in engine has the same level.** The core keeps one engine object per compiled-in engine
while a loaded model of it holds it, so a failed load or the last unload frees what init built right
there (never at process exit), and the next load makes a new object. Its init: ONNX Runtime and
LiteRT create their environment (at anira's log level; LiteRT with the accelerators its registration
finds, which a load is checked against), ExecuTorch initialises its runtime, LibTorch sets its
intra-op threads to one (OpenBLAS's too on Linux aarch64, where ``torch::set_num_threads`` does not
reach it) and c10's log level; TensorFlow Lite has nothing per process. A load before the init is
``ANIRA_ERROR_INVALID_STATE``, for a built-in and a registered engine alike.

**The load record**, ``anira_engine_load_info`` (Tier 2, valid for the call; the engine copies what
it keeps):

- ``row``: the engine's entry in the variant's ``models[]``, whose path or bytes it loads;
  ``model``: anira's copy of the model configuration, read through the getters
  (``anira_model_config_model_path(model, row)``, ``_model_bytes(model, row, &bytes, &size)``,
  ``_tensor_name`` / ``_tensor_layout``, ``_model_ext(model, row, "entry")``, ``_input`` /
  ``_output`` and the ``anira_tensor_spec`` getters).
- ``inputs`` / ``outputs`` (with their counts): one template per slot, State tensors included, in
  slot order: the **engine side** of every tensor, the spec's dtype and the extents in the engine's
  order (the entry's ``tensors`` layout applied) at the pinned window, all-zero strides,
  ``ANIRA_DOMAIN_HOST``, no memory. What ``process`` will be handed, minus the data.
- ``input_names`` / ``output_names``: the name each slot binds to, the entry's ``tensors`` record's,
  else the canonical name (:ref:`usage-tensor-records`). An engine whose side has names binds each
  slot by it and refuses load for a name its side lacks; one without names binds by position. The
  plan report says ``ANIRA_BINDING_ENGINE`` for every slot: the engine bound itself.
- ``provider`` / ``provider_id``: the provider this load is for, in the words of the descriptor's
  list. A provider is part of the loaded model (two providers, two loads); a load that cannot serve
  it returns ``ANIRA_ERROR_NOT_SUPPORTED`` (a device missing at run time: create checked the list).
- ``option_keys`` / ``option_values`` / ``num_options``: the context's provider options for this
  backend (:ref:`usage-provider-options`), only for an engine that lists
  ``"context:provider_options"``; ``NULL`` and 0 otherwise and when none are set. Part of the loaded model: two contexts with
  different options load twice.
- ``instances``: the **shared call slots**, the ``process`` calls that may run at once, each on its
  ``ctx->instance`` below this count: a stateless model's ``max_instances`` (``ANIRA_INSTANCES_AUTO``
  resolved to the pool's size, at least 1), clamped to the pool's size; 0 for a model declared
  ``ANIRA_MODEL_STATEFUL`` or with a State pair, whose handlers are all **exclusive**.

**The warm-up** is no part of the record: the contract's pass runs at ``anira_handler_prepare``
through the handler's session and reaches the engine as ordinary ``process`` calls, which it times
(:ref:`usage-budget-warmup`). Under a measured budget the engine's ``prepare`` may run twice in one
``anira_handler_prepare``: ``prepare``, the pass's ``process`` calls, then ``unprepare`` and
``prepare`` again with the final record when the pass moved the entry count. The figure is the
model's on this registration: a later handler on it with the same model key reuses it
(:ref:`usage-budget-warmup`), an engine registered anew (one object per plugin instance) measures its
own. An engine that wants its first-run cost gone before the pass
runs a warm inference itself, over the shared slots at load and over an exclusive handler's executor
at prepare, as the built-in engines do over every executor they make.

**The prepare record**, ``anira_prepare_info`` (Tier 2; the handler pointer and the report stay valid
while the handler stays prepared, the arrays for the call): ``handler``, ``report``,
``num_entries`` (the chunks in flight at once; ``ctx->entry`` stays below it), one template per slot
of the **model end** as the processor sees it (the spec's dtype and shape at the pinned window, the
slot's host domain, no memory), the canonical names, and ``flags``: ``ANIRA_PREPARE_EXCLUSIVE`` when
the handler's inferences run one at a time and in order (a stateful model or a State pair), its calls
claim no shared slot, and the engine builds what it keeps per stream here, its executor included; 0
otherwise: the handler's calls run on the shared slots, and an engine that keeps nothing per handler
hands back nothing.

**The call context**, ``anira_engine_ctx`` (Tier 1, 64 bytes on anira's stack for the call):
``instance`` (the shared slot, ``0 .. instances - 1``, never two calls at once on one; 0 under
``ANIRA_ENGINE_CALL_EXCLUSIVE``), ``entry`` (the processor context's ``entry`` of the same chunk),
``inputs`` / ``outputs`` (one descriptor per slot, State included, the spec's dtype and the
engine-side extents, with the data), ``ticket`` (``ANIRA_TICKET_INVALID`` under Hard), ``flags``
(``ANIRA_ENGINE_CALL_EXCLUSIVE`` for an exclusive call) and ``loaded``; two reserved words and a
reserved pointer read 0 and ``NULL``.

**The adapter rule**, which every engine follows, built in or registered: read every extent and every
memory handle (the data pointer, ``byte_offset``, the strides, the domain) from the tensors of
**this** call, never from a value kept at load or prepare, and never assume a tensor is anira's own
buffer. The halves of a State pair alternate between two buffers, unless the engine sets
``ANIRA_ENGINE_FLAG_STATE_ALIAS`` (:ref:`engines-state`), and a later
pre-release hands a caller's Buffer tensor over in place. Every tensor is in its spec's dtype
(``anira_tensor_data(t, dtype)`` reads it), and an engine may write its inputs (an in-place model):
only an ``after_inference`` hook reads them afterwards and sees what the engine left. A Time extent
below the template's is legal only under ``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``. The other half is the
executor's: what one call runs on owns everything with run-time state, so an engine keeps a shared
slot's objects behind ``loaded``, indexed by ``ctx->instance``, and an exclusive handler's behind
``prepared``, and never lets two calls meet on one (what the built-in engines share:
:doc:`architecture`).

A ``process`` that returns any status but ``ANIRA_OK`` fails the chunk as a built-in engine's failure
does: ``rt_error`` reads ``ANIRA_ERROR_ENGINE`` with one latched record naming the status, anira
zeroes the outputs, the chunk delivers zeros at its position, the processor's ``after_inference``
does not run for it, and a State pair keeps its last good value. A throw across this boundary is
undefined. The typedef carries no real-time attribute: ``ANIRA_ENGINE_FLAG_REALTIME_SAFE`` is the
engine's promise, and an author who wants clang's check declares the function ``ANIRA_NONBLOCKING``;
the tensor accessors are ``ANIRA_NONBLOCKING``, so a body made of them keeps it.

.. _engines-flags:

Flags and the engine in the report
----------------------------------

The flags are reported in ``anira_plan_info.engine_flags`` of every plan of the engine (0 for a
built-in one); a bit the header does not define is ``ANIRA_ERROR_INVALID_ARGUMENT`` at create.

- ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``: an entry may have no source, neither path nor bytes
  (``ModelConfig::add_model(id)``, a NULL path in C, a JSON entry without ``"path"``); the load reads
  none. ``anira_handler_create`` refuses a source-less entry of any other engine
  (``ANIRA_ERROR_CONFIG``).
- ``ANIRA_ENGINE_FLAG_REALTIME_SAFE``: ``process`` allocates nothing, takes no lock and makes no
  system call after prepare. Reported; a later contract option consumes it.
- ``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``: ``process`` accepts a Time extent that varies per call at or
  below the template's. Reported.
- ``ANIRA_ENGINE_FLAG_STATE_ALIAS``: the engine keeps its own aliasing of a State pair: one stable
  buffer as both halves, never flipped (:ref:`engines-state`). Honoured per plan.

No built-in engine sets any of them. ``ANIRA_PREPARE_EXCLUSIVE`` and ``ANIRA_ENGINE_CALL_EXCLUSIVE``
are anira's flags, not the engine's: they say what the handler is. Wherever the engine appears, the
pair is ``ANIRA_ENGINE_CUSTOM`` with the id and the plan's provider beside it (``ANIRA_PROVIDER_CUSTOM``
with ``provider_id`` for a custom one): ``anira_plan_info.engine`` / ``engine_id`` / ``provider`` /
``provider_id``, ``anira_processor_ctx`` (whole through ``anira_processor_engine`` / ``_provider``),
the pipeline's edges (``to_engine_id`` from the host, ``from_engine_id`` back) and the candidates'
``anira_backend_id``. Its slot rows read ``ANIRA_BINDING_ENGINE``, and its extension rows name it by
its id.

.. _engines-cpp:

In C++
------

:cpp:class:`anira::Engine` is the descriptor with virtual functions in place of the function
pointers, split as the C levels are:

- **The engine object** is constructed with its id (``id()`` reads it back) and states ``flags()``,
  ``consumed_kinds()`` and ``providers()`` (empty: the CPU path alone; all read once, when its C
  engine is created at its first registration); it may override ``query(const InitInfo&)`` (the base
  answers every bit) and ``init(const InitInfo&)`` (the base does nothing). Its ``load(const
  EngineLoadInfo&)`` returns a ``std::unique_ptr<Engine::Loaded>``, the loaded model, whose
  ``prepare(const PrepareInfo&)`` returns a ``std::unique_ptr<Engine::Prepared>``, the handler's,
  whose ``process(EngineContext&)`` and ``reset(EngineContext&)`` are ``noexcept`` (the base
  ``reset`` does nothing).
- **The records are views:** :cpp:class:`anira::InitInfo` (``log_level()``, ``num_threads()``,
  ``context()``), :cpp:class:`anira::EngineLoadInfo` (``row()``, ``model()``, ``model_path(i)``,
  ``model_bytes(i)``, ``model_engine(i)`` (an :cpp:struct:`anira::EngineRef`), ``inputs()`` /
  ``outputs()`` as ``std::span<const Tensor>``, ``input_names()`` / ``output_names()``,
  ``instances()``, ``provider()`` (an :cpp:struct:`anira::ProviderRef`),
  ``option_keys()`` / ``option_values()``), :cpp:class:`anira::PrepareInfo` (``handler()``,
  ``report()``, ``num_entries()``, ``inputs()`` / ``outputs()``, the names, ``flags()``,
  ``exclusive()``) and :cpp:class:`anira::EngineContext` (``instance()``, ``entry()``, ``ticket()``,
  ``flags()``, ``exclusive()``, ``loaded()``, ``inputs()`` and the writable ``outputs()``; every
  method a ``noexcept`` field read).
- ``init``, ``load`` and ``prepare`` may throw: an ``anira::Error`` fails the prepare with its status,
  ``std::bad_alloc`` with ``ANIRA_ERROR_OUT_OF_MEMORY``, anything else (and a null return from
  ``load`` or ``prepare``) with ``ANIRA_ERROR_INTERNAL``, ``what()`` going to the log; a throw never
  crosses the C boundary.
- **Ownership:** the ``Loaded`` is anira's from ``load`` on (deleted at the C ``unload``), the
  ``Prepared`` from ``prepare`` on (deleted at ``unprepare``), the engine object shared by every
  pipeline and handler (a ``std::shared_ptr``). What every handler shares is a member of the
  ``Loaded``, what an exclusive handler keeps per stream a member of the ``Prepared``, nothing an
  inference needs a member of the ``Engine``.
- **Identity:** ``Pipeline::add_engine(impl)`` is the one way to add it. The first registration of an
  object creates its C engine under the object's id; every later registration of **the same object**,
  on another ``Pipeline``, reuses that C engine while anything holds it (the object keeps a detached
  handle), so the handlers of all of them share loaded models. The C engine holds a copy of the
  ``shared_ptr``, so the object lives until its ``release()``, which runs once per C engine,
  when the last pipeline, handler and loaded model carrying it are gone; a registration after that
  creates a fresh C engine with a ``release()`` of its own. An id the create refuses creates nothing;
  a registration refused for a duplicate id takes no reference (a C engine that call created is
  dropped again, and ``release()`` answers it).

.. note::
    A 2.x ``BackendBase`` subclass becomes an :cpp:class:`anira::Engine` constructed with
    ``anira::v2::k_custom_engine_id`` (``"anira.v2.custom"``, the id a 2.x ``CUSTOM`` row names);
    ``anira::v2::PassthroughEngine`` of ``anira/compat/v2.hpp`` is ``BackendBase::process`` as such
    an engine (:ref:`migration-runtime`). The built-in engines are adapters of the same descriptor
    shape (``src/engines/``, internal): where to look for how a real engine binds by name, checks
    the shapes it was handed and splits what one load shares from what one executor owns.
