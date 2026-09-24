Custom Engines
==============

An **engine** runs the model: it takes the model tensors of a chunk in, runs one inference,
and writes the model tensors out. anira ships five (ONNX Runtime, LibTorch, ExecuTorch, LiteRT
and TensorFlow Lite, ``ANIRA_ENGINE_*``), and a host adds its own: it **creates** the engine
once as an object under a reverse-URI id and **adds** it to pipelines. Examples: a Core ML or
a WebGPU runtime, an engine the build does not carry, a synthetic model, a bypass for
measuring the pipeline's overhead. A custom engine is one more implementation the inference
stage resolves to, never a stage of its own (the stage is the pre- and post-processing around
it, :doc:`custom_preprocessing`): a model entry names it by its id, and the handler loads it,
prepares it, runs it and resets it exactly as it does a built-in engine, with the same plan
report, the same real-time rules and the same failure path. In C it is a descriptor,
``anira_engine_desc`` of ``anira/abi/engine.h``; in C++ the same descriptor is a subclass of
:cpp:class:`anira::Engine` (below). Its lifecycle is the stage's with one level more, the same
words on both sides, from the outermost level in: ``init`` runs once per engine object,
``load`` once per loaded model anira pools and hands back a ``loaded`` pointer, ``prepare``
once per handler on a loaded model and hands back a ``prepared`` pointer, every per-chunk call
receives them as ``(ctx, prepared, user_data)`` with the loaded pointer in ``ctx->loaded``,
``reset`` runs at a stream boundary of an exclusive handler, ``unprepare`` and ``unload`` give
the two pointers back, ``release`` fires once per engine object.

Creating and adding an engine
-----------------------------

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
    anira_model_config_add_model_path_custom(cfg, "com.example.myengine", "model.bin", &row, &err);

Two calls, two jobs:

- ``anira_custom_engine_create(id, &desc, &engine, &err)`` copies the descriptor (within
  ``struct_size``, with its strings) into a refcounted **engine object** under an **id**. The
  object is what the engine *is*: anira recognises the same engine by it, and shares loaded
  models between handlers that run it (below). The id is how model entries name the engine,
  and it stays the object's for its whole life: every pipeline the engine is added to, every
  message and every plan report row about it use that one id.
- ``anira_pipeline_add_engine(pipe, engine, &err)`` adds the engine to a pipeline. One engine
  may be added to any number of pipelines, and the engines of one pipeline have distinct ids.
  The id is unique per pipeline, not per process: two plugin instances that each create their
  own engine under one id add them to their own pipelines, and the two engines share nothing.

The handle, every pipeline the engine was added to, every handler created from those pipelines
and every loaded model of the engine hold a reference. ``anira_custom_engine_destroy`` drops
the handle's, so it may run right after the last ``anira_pipeline_add_engine``; ``release``
fires exactly once, when the last reference is gone. A handle that should add the engine to
pipelines yet to come without keeping it alive is **detached** (``anira_custom_engine_detach``):
from then on it names the engine while a pipeline, a handler or a loaded model holds it, the
addition is ``ANIRA_ERROR_INVALID_STATE`` once the engine was released, and
``anira_custom_engine_destroy`` still frees it. The id is a reverse-URI name: it must
contain a ``.``, and the prefix ``anira.`` is anira's own (``anira.v2.custom`` names the 2.x
``CUSTOM`` backend of :doc:`migration`). Adding is legal before or after
``anira_pipeline_add_inference``; a handler copies the pipeline at ``anira_handler_create``, so
an engine added afterwards does not reach that handler. A model entry that names the id
(``anira_model_config_add_model_path_custom``, ``anira_model_config_add_model_bytes_custom``;
``"engine": "com.example.myengine"`` in a model file) is a plan of the handler like a built-in
engine's entry; several such entries on one model configuration are several plans, which
``anira_handler_set_plan`` switches between, resolved by entry, never by the 2.x backend they
map to. A candidate of ``anira_pipeline_add_inference`` names the engine on a **provider**
(``anira_backend_id``: ``engine_id`` with ``provider``, or ``provider_id`` for a provider the
enum does not name), and a plan is one entry on one provider: an entry without a pin is a
plan per candidate of its engine, an entry pinned to a provider (:doc:`usage` section 1.1) a
plan on that candidate alone. An added engine no entry names is not a plan and not an error; an entry whose id no
engine of the pipeline serves is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``,
naming the id. anira never opens a custom entry's path or reads its bytes: the engine's
``load`` does, through the record it receives.

``anira_custom_engine_create`` refuses with ``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL`` id,
descriptor or ``out``, an id without a ``.`` or with the prefix ``anira.``, a ``struct_size``
below the three leading slots (``struct_size``, ``abi_version``, ``user_data``), a ``flags``
bit the header does not define, a ``NULL`` ``process``, a ``NULL`` ``consumed_kinds`` (or
entry) with a count above 0 and a ``NULL`` ``providers`` (or a ``NULL`` or empty entry) with a
count above 0, and with ``ANIRA_ERROR_ABI_VERSION`` an ``abi_version`` ``anira_check_abi``
refuses; a refused create hands out nothing and never calls ``release``.
``anira_pipeline_add_engine`` refuses with ``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL`` pipeline
or engine, and with ``ANIRA_ERROR_INVALID_STATE`` an engine whose id the pipeline already has,
the same engine or another; a refused addition takes no reference.

The descriptor
--------------

``anira_engine_desc`` (Tier 2: ``struct_size`` first, ``user_data`` third, growth at the tail;
``ANIRA_ENGINE_DESC_INIT`` is an engine without a callback and without a promise) names, as
the stage's descriptor does, the slots from the innermost level of the lifecycle out:

- ``user_data``: handed to every callback as it is, never read by anira. One per engine
  object, shared by every pipeline it was added to, every loaded model of the engine and every
  handler prepared on one; what one loaded model keeps lives behind the ``loaded`` pointer its
  ``load`` hands back, what one handler keeps behind the ``prepared`` pointer its ``prepare``
  hands back.
- ``consumed_kinds`` / ``num_consumed_kinds``: the extensions the engine reads at load, as
  ``"<host>:<kind>"`` strings (hosts ``tensor_spec``, ``model``, ``model_config``,
  ``context``, ``contract``), copied. They join the consumed-or-fail walk for the entries of
  this engine, keyed by its id: its ``model:`` kinds read its own entries alone, its other
  kinds any host, and each consumed slot is an ``anira_plan_ext`` row of the plan whose
  consumer reads the engine's id. An engine that lists ``"context:provider_options"``
  receives, at ``load``, the provider options the context config carries for its backend
  (the load record, below); a set for an engine that does not list it is refused at
  ``anira_handler_create``.
- ``flags``: the engine's promises, an OR of the four ``ANIRA_ENGINE_FLAG_*`` bits below;
  ``0`` promises nothing.
- ``providers`` / ``num_providers``: the providers the engine serves beyond
  ``ANIRA_PROVIDER_DEFAULT``, as strings, copied. The JSON spellings of ``anira_provider``
  (``"cuda"``, ``"webgpu"``, ``"directml"``, ``"coreml"``, ``"xnnpack"``, ``"vulkan"``) name a
  provider of the enum; any other string is a custom provider in the engine's own vocabulary
  (a reverse-URI name is the convention, not a rule), which a candidate names in
  ``provider_id`` beside ``ANIRA_PROVIDER_DEFAULT`` and a model entry's pin spells under its
  ``"provider"`` key. ``NULL`` with a count of 0 serves the default provider alone. The list is
  what the engine can ever serve; which of it is usable here, now, is the ``query`` slot's
  answer (below). A candidate naming a provider the list lacks, or one the query reports
  unavailable, is ``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``, naming the
  engine, the provider and what is served (a declared provider the query cleared: "declares
  provider 'x' but its query reports it unavailable here"); ``load`` may still refuse one it
  cannot serve at run time (a device lost since).
- ``process``: the engine call, ``anira_engine_process_fn``. Required.
- ``reset``: ``anira_engine_reset_fn``. ``NULL``: nothing to reset.
- ``prepare``: ``anira_engine_prepare_fn``. ``NULL``: nothing to prepare, and ``prepared`` is
  ``NULL`` in every later call.
- ``unprepare``: ``anira_engine_unprepare_fn``. ``NULL``: none.
- ``load``: ``anira_engine_load_fn``. ``NULL``: nothing to load, and ``loaded`` is ``NULL`` in
  every later call.
- ``unload``: ``anira_engine_unload_fn``. ``NULL``: none.
- ``init``: ``anira_engine_init_fn``. ``NULL``: nothing to initialise.
- ``release``: ``anira_engine_release_fn``. ``NULL``: none.
- ``query``: ``anira_engine_query_fn``, a tail field beyond the providers list, the engine's
  own ``GetAvailableProviders``: which of ``providers`` are usable here, now, as a bitmask over
  the list (bit *i* set: ``providers[i]``; the default provider is always served and has no
  bit). It runs before ``init`` and any number of times, on the main thread, at
  ``anira_handler_create`` and at the pipeline's capabilities entries (below), with an
  ``anira_init_info``; it may log and must not call an entry that takes the core's lifecycle
  lock. A status other than ``ANIRA_OK`` fails the calling entry with it, naming the engine.
  ``NULL``: every listed provider is usable.

Three levels, three pointers, three lifetimes: ``user_data`` lives with the engine object,
``loaded`` with one loaded model of it, ``prepared`` with one handler on that loaded model.
The stage's descriptor (``anira_stage_desc``, :doc:`custom_preprocessing`) has the same slots
minus the model level, and its ``init`` and ``prepare`` receive the same two records,
``anira_init_info`` and ``anira_prepare_info`` of ``anira/abi/lifecycle.h``.
``anira_phase`` names the phase of every slot, ``ANIRA_PHASE_INIT``,
``ANIRA_PHASE_LOAD`` and ``ANIRA_PHASE_UNLOAD`` among them for the outer levels.

The lifecycle
-------------

**init(info, user_data)** runs once per engine object, from the first ``anira_handler_prepare``
that reaches it (a row of the engine survives validation), on that caller's thread under the
core's lifecycle lock, before the engine's first ``load``: it may allocate, log and query the
context's capabilities, and must not call an entry that takes that lock (context create and
destroy, handler create, destroy and prepare, ``anira_inference_thread_create``,
``anira_shutdown``, ``anira_release_core_if_idle``). This is where an engine builds what it
keeps for its whole life beside ``user_data``: a device context, a thread pool, a weights
cache. Its record, ``anira_init_info`` (Tier 2, valid for the duration of the call; the
stage's ``init`` receives the same record):

- ``log_level``: the ``anira_log_level`` in effect in this copy of anira (the core reconciles
  one per process, the most verbose of its contexts').
- ``num_threads``: the size of the inference-thread pool this copy of anira runs, what
  ``anira_num_inference_threads`` answers; ``0`` when the context brought its own threads
  (``anira_inference_thread_create``) or on a target without threads.
- ``context``: the context of the handler whose prepare reached the engine first; its
  capabilities (``anira_context_capabilities``) are legal to query; valid until ``init``
  returns, never kept.

A built-in engine has the same level. The core keeps one engine object per compiled-in engine
for its own life and runs the engine's init once, with the same record, right before its
first ``load``: ONNX Runtime creates its environment there, with the level in effect as its
logger's severity, and every session of every loaded model is created over it; LiteRT its
environment likewise, with the accelerators its registration finds, which a load's record is
checked against; ExecuTorch initialises its runtime; LibTorch sets its intra-op thread count
to one and c10's log level; TensorFlow Lite has nothing per process. What an engine builds at
init outlives every loaded model of the engine on both sides, and a load before the init is
``ANIRA_ERROR_INVALID_STATE`` on both.

A status other than ``ANIRA_OK`` fails that ``anira_handler_prepare`` with it, the message
naming the engine (``the engine 'com.example.myengine' refused init: it returned N``), and
leaves the object uninitialised, so the next prepare calls ``init`` again. The engine's
``init`` runs before the stage's: the session is created, loading the models, before the
stage is prepared.

**load(info, user_data, &loaded)** runs on the caller of ``anira_handler_prepare``, once per
loaded model anira pools (below), under the core's lifecycle lock, after ``init``: it may
allocate, read files, log, call the config getters and the tensor accessors, and must not call
an entry that takes that lock. This is where the engine loads the entry's model, binds the
slots and sizes the shared call slots the record counts. Its record,
``anira_engine_load_info`` (Tier 2, valid for the duration of the call; the engine copies
what it keeps and never keeps the pointers):

- ``row``: this engine's entry in the variant's ``models[]`` list, the entry whose path or
  bytes the engine loads; ``model``: the variant, anira's own copy of the model
  configuration, read through the config getters (``anira_model_config_model_path(model,
  row)``, ``anira_model_config_model_bytes(model, row, &bytes, &size)``, the tensor records,
  the extensions).
- ``inputs`` / ``num_inputs`` and ``outputs`` / ``num_outputs``: one template per slot of
  the model's input list and output list, State tensors included, in slot order: the
  **engine side** of every tensor, the spec's dtype and the extents in the engine's order
  (the entry's ``tensors`` layout applied) at the pinned window, all-zero strides,
  ``ANIRA_DOMAIN_HOST``, no memory. What ``process`` will be handed, minus the data.
- ``input_names`` / ``output_names``: the name each slot binds to, the entry's ``tensors``
  record where it names the slot, else the canonical name. This is the binding rule of
  :doc:`usage` section 1.2 as a registered engine sees it: an engine whose side has names
  binds each slot to the tensor of that name and refuses load for a name its side does not
  have; an engine without names binds by position. Either way the plan report says
  ``ANIRA_BINDING_ENGINE`` for every slot of the plan: the engine received the names and bound
  itself.
- ``provider`` / ``provider_id``: the provider this load is for, a value of the enum or
  ``ANIRA_PROVIDER_DEFAULT`` beside the custom name, in the words of the descriptor's list. A
  provider is part of the loaded model: two providers of one model are two loads, each its own
  ``loaded``; a load that cannot serve the one it is asked for returns
  ``ANIRA_ERROR_NOT_SUPPORTED`` (the handler checked the plan's provider against the list at
  create, so this is a device missing at run time).
- ``option_keys`` / ``option_values`` / ``num_options``: the provider options of this
  backend, the set of the context config's ``provider_options`` extension for the engine on
  this provider (:doc:`usage` section 3.1), as string pairs in the engine's own vocabulary;
  on the record only for an engine whose ``consumed_kinds`` lists
  ``"context:provider_options"``, ``NULL`` with a count of ``0`` otherwise and for none. The
  options are part of the loaded model: two contexts with different options for one backend
  load twice.
- ``instances``: the **shared call slots** of this loaded model, the ``process`` calls that
  may run at once on them, each on its own instance below this count (``ctx->instance``): the
  model's ``max_instances`` for a stateless model, clamped to the size of the inference-thread
  pool when anira runs one; ``0`` for a model declared ``ANIRA_MODEL_STATEFUL`` or with a
  declared State pair, whose handlers are **exclusive** (``ANIRA_PREPARE_EXCLUSIVE`` at their
  ``prepare``, below) and run on what their ``prepare`` builds, never on a shared slot. Which
  of the two a handler is follows from its model, so every handler of one loaded model is of
  one kind.

The record carries no warm-up count: the contract's fixed warm-up is what the built-in
adapters run over every executor they make, and an engine that wants a warm inference runs it
itself, over the shared slots here and over an exclusive handler's own executor at its
``prepare``. A status other than ``ANIRA_OK`` fails ``anira_handler_prepare`` with it, the
message naming the engine (``the engine 'com.example.myengine' refused load: it returned N``),
and no ``unload`` follows for that load. What the call hands back through ``out_loaded``
(which starts ``NULL``; ``NULL`` is a legal value) is the ``loaded`` pointer every
``prepare`` of **this** loaded model receives, every ``process`` and ``reset`` call sees in
``ctx->loaded``, and ``unload`` gets back.

**What one load shares.** The shared call slots are what the three levels are for. An
**executor** is one call slot, what one ``process`` call runs on and never two at once (a
session, an interpreter, a method), and it owns everything with run-time state; what one load
shares between its executors is only what is immutable during a run. The built-in engines
draw the line so (``src/backends/``, internal):

- ONNX Runtime: one session-options object per loaded model, shared, over the engine object's
  environment (one per process, created at the engine's init with anira's log level; the web
  build creates it with a global thread pool of one thread through ``Ort::ThreadingOptions``,
  since a WebAssembly session cannot spawn its own); one ``Ort::Session`` per executor, kept
  per executor so that no two executors share a session's allocator.
- LibTorch: one TorchScript module per executor, each with its own copy of the weights, since
  a module is not shareable between threads; what one load shares is the record and the
  binding tables.
- TensorFlow Lite: one ``TfLiteModel`` (the flatbuffer parsed once) and one options object per
  loaded model, shared; one interpreter, with its signature runner, per executor.
- LiteRT: one model and one compilation-options object per loaded model, shared, over the
  engine object's environment (one per process, created at the engine's init with anira's log
  level, the accelerators it registered checked at load); one compiled model with its managed
  buffers per executor.
- ExecuTorch: one data loader and one ``Program`` (the ``.pte`` parsed once) per loaded model,
  shared; one ``Module`` over that program, with its own loaded method, per executor. Every
  method is loaded with the XNNPACK delegate's runtime options set to a workspace per delegate
  instance and no process-wide weight cache, so two executors never meet on the process-wide
  mutexes the prebuilt runtime otherwise takes around every call, at the price of a workspace
  and an unpacked copy of the weights per executor (memory grows with the model per executor,
  an exclusive handler's own included); and it is loaded and run under a
  no-thread-pool guard, so the delegate captures no thread pool and every call runs inline on
  the inference thread that made it.

Each of them makes the executors of the shared slots at load and one more for an exclusive
handler at its ``prepare`` (a model without a shared slot keeps the executor its load probed
the binding with, for the first exclusive handler), and runs the fixed warm-up on every one of
them where it is made.

**prepare(info, loaded, user_data, &prepared)** runs on the caller of
``anira_handler_prepare``, once per handler and loaded model the handler runs on, after the
plan report is built and while the handler already counts as prepared (its getters answer),
with the loaded pointer and the record the stage's ``prepare`` receives. It may allocate: this
is where an engine builds what it keeps per handler. Of anira it may call the
``[callback-safe]`` entries, the handler's getters and the two Static entries, never
``anira_handler_prepare``, ``anira_handler_destroy`` or a Hard entry. Its record,
``anira_prepare_info`` (Tier 2, valid for the duration of the call; the handler pointer and
the report stay valid while the handler stays prepared, the arrays do not):

- ``handler``: the handler being prepared; ``report``: this prepare's plan report;
  ``num_entries``: ``anira_handler_num_entries`` of this prepare, the chunks that can be in
  flight at once (``ctx->entry`` stays below it).
- ``inputs`` / ``num_inputs`` and ``outputs`` / ``num_outputs``: one template per slot, State
  tensors included, in slot order: the **model end** of every tensor as the stage sees it (the
  spec's dtype and shape at the pinned window, the slot's host domain, no memory); the engine
  side of the same slots is in the templates of the load record.
- ``input_names`` / ``output_names``: the canonical names, in slot order.
- ``flags``: ``ANIRA_PREPARE_EXCLUSIVE`` when the handler's inferences run one at a time and
  in order, under the dispatch gate of a model declared ``ANIRA_MODEL_STATEFUL`` or with a
  declared State pair, with a ``reset`` at every stream start: the handler's calls claim no
  shared slot, so the engine builds what it keeps per stream here, its own executor included.
  ``0`` otherwise: the handler's calls run on the shared slots, and an engine that keeps
  nothing per handler hands back nothing. No other bit is defined in this pre-release.

A status other than ``ANIRA_OK`` fails ``anira_handler_prepare`` with it, the message naming
the engine (``the engine 'com.example.myengine' refused prepare: it returned N``), and leaves
the handler unprepared; no ``unprepare`` follows for that prepare, and the loaded model the
failed session had acquired goes back (an ``unload``, when the handler was its last holder).
What the call hands back through ``out_prepared`` (which starts ``NULL``; ``NULL`` is a legal
value) is the ``prepared`` pointer every ``process``, ``reset`` and ``unprepare`` of **this**
handler on **this** loaded model receives beside the engine's ``user_data``.

**process(ctx, prepared, user_data)** runs on an inference thread, in
``ANIRA_PHASE_INFERENCE`` (between the stage's ``before_inference`` and ``after_inference``),
one inference over the context's tensors: a **shared call**, the call of a handler that is
not exclusive, claims one of the loaded model's shared slots and runs on it; an **exclusive
call** claims nothing and runs on what the handler's ``prepare`` built. It may block; it must
not allocate per call. Its context, ``anira_engine_ctx`` (Tier 1, 64 bytes, on anira's stack
for the duration of the call, like the stage's ``anira_stage_ctx``):

- ``instance``: the shared slot of the loaded model this call runs on, ``0 .. instances - 1``;
  never two calls at once on one instance. ``0`` under ``ANIRA_ENGINE_CALL_EXCLUSIVE``, where
  no slot was claimed.
- ``entry``: the chunk's position in the handler's inference queue, the stage context's
  ``entry`` of the same chunk (:doc:`usage` section 2).
- ``inputs`` / ``num_inputs`` and ``outputs`` / ``num_outputs``: one descriptor per slot of
  either side in slot order, State tensors included, over the memory the engine reads and
  writes: the spec's dtype and the engine-side extents of the load record's templates, with
  the data.
- ``ticket``: ``ANIRA_TICKET_INVALID`` under a Hard contract, the job's ticket under an Async
  one.
- ``flags``: ``ANIRA_ENGINE_CALL_EXCLUSIVE`` for a call of an exclusive handler: it runs on
  what that handler's ``prepare`` built, no shared slot was claimed, ``instance`` is ``0``. No
  other bit is defined in this pre-release.
- ``loaded``: what this loaded model's ``load`` handed back; ``NULL`` for an engine without a
  ``load`` slot. Two reserved words and one reserved pointer slot read ``0`` and ``NULL``; the
  per-call facts grow through them.

**The adapter rule**, the one rule every engine follows, built in or registered: read every
extent and every memory handle (the data pointer, ``byte_offset``, the strides, the domain)
from the tensors of **this** call, never from a value kept at load or prepare, and never
assume a tensor is anira's own buffer. The halves of a declared State pair alternate between
two buffers from one inference to the next (below), unless the engine's flags carry
``ANIRA_ENGINE_FLAG_STATE_ALIAS``, and a later pre-release hands a caller's Buffer tensor
over in place. The templates of the load record say what the shapes will be;
the memory is the context's. A Time extent below the template's is legal only under
``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``. The rule's other half is the executor's: what one call
runs on owns everything with run-time state (above), so an engine keeps a shared slot's
objects behind ``loaded``, indexed by ``ctx->instance``, and an exclusive handler's behind
``prepared``, and never lets two calls meet on one. The typedef carries no real-time
attribute: whether the body is real-time is the engine's own promise,
``ANIRA_ENGINE_FLAG_REALTIME_SAFE``, and an author who wants clang's compile-time check
declares the function ``ANIRA_NONBLOCKING`` themselves; the tensor accessors
(``anira_tensor_data_f32``, ``anira_tensor_num_elements``, ``anira_tensor_plane``) are
``ANIRA_NONBLOCKING``, so a body composed of them keeps the promise.

A ``process`` that returns any status but ``ANIRA_OK`` fails the chunk, as a built-in engine's
failure does: ``anira_handler_rt_error`` reads ``ANIRA_ERROR_ENGINE`` with one latched record
naming the status, anira zeroes the outputs, the chunk delivers zeros at its stream position,
the stage's ``after_inference`` does not run for it, and a State pair keeps its last good
value (an aliasing engine writes in place: what a failed call left in the buffer is its own).
A throw across this boundary is undefined.

**reset(ctx, prepared, user_data)** re-initialises the state an engine keeps per handler (a
recurrent hidden state the model does not declare, a variable tensor, a filter's history): for
an **exclusive** handler, at the first inference of a new stream (after ``prepare``, after
``anira_handler_reset``), on the inference thread, with that inference's context
(``ANIRA_ENGINE_CALL_EXCLUSIVE`` set) and the handler's ``prepared``, right before its
``process``. The boundary is the plan's: after ``prepare`` or a reset every plan of the
handler is reset at its own first inference of the new stream, whichever plan ran first, and a
plan switch alone is none. Never for a handler whose model is stateless, whose calls run on the
shared slots
and keep nothing between them. ``NULL`` for an engine without such state. (Of the built-in
engines only TensorFlow Lite has a body, ``TfLiteInterpreterResetVariableTensors`` on the
handler's own interpreter.)

**unprepare(prepared, user_data)** gives back what one prepare built: once per successful
prepare, on the thread of the ``anira_handler_prepare`` or ``anira_handler_destroy`` of that
handler (at the next prepare once the previous session is released, before the new prepare;
a later prepare that fails earlier unprepares the previous one all the same), after the
handler's in-flight inferences have drained and before the loaded model it ran on is
released, so no ``process`` or ``reset`` with this prepared pointer runs afterwards.

**unload(loaded, user_data)** frees what one load loaded: once per successful load, on the
thread of the ``anira_handler_prepare`` or ``anira_handler_destroy`` that drops the last
handler holding the loaded model, after every ``unprepare`` of its prepared handles, so no
``process``, ``reset`` or ``prepare`` of the loaded model runs afterwards. It may run under
the core's lifecycle lock (it does when a later plan of the same prepare fails and the plans
loaded before it are unloaded on the way out), so like ``load`` it must not call an entry
that takes that lock.

**release(user_data)** runs exactly once, when the last reference to the engine object dies
(``anira_custom_engine_destroy``, ``anira_pipeline_destroy`` or ``anira_handler_destroy``,
whichever comes last), after every ``unload``, whether or not ``init`` ever ran; no callback
of the engine runs afterwards.

Statuses and flags
------------------

The flags are the engine's promises, declared once and reported in
``anira_plan_info.engine_flags`` of every plan of the engine (``0`` for a built-in engine):

- ``ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL``: a model entry of this engine may have no source.
  Declared and reported now; consumed when a source-less entry can be built, in a later
  pre-release (today every entry carries a path or bytes, which the engine is free to ignore).
- ``ANIRA_ENGINE_FLAG_REALTIME_SAFE``: ``process`` allocates nothing, takes no lock and makes
  no syscall after prepare. Reported; consumed by a later contract option. No built-in engine
  sets it.
- ``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``: ``process`` accepts a Time extent varying per call at or
  below the template's. Reported; no built-in engine sets it.
- ``ANIRA_ENGINE_FLAG_STATE_ALIAS``: the engine keeps its own aliasing of a declared State
  pair. anira binds one stable buffer per pair as both the State input and the State output of
  every call of a plan of this engine and never flips it: the engine reads the state before it
  writes it, in place, and every address stays put across the calls of aliasing plans, which
  a captured graph (CUDA graphs, WebGPU replay) needs (a chunk of a flipping plan in between
  moves the pair: re-capture after such a switch). Honoured per plan (below); no built-in
  engine sets it.

A bit the header does not define is ``ANIRA_ERROR_INVALID_ARGUMENT`` at
``anira_custom_engine_create``. The flags of the two records are anira's, not the
engine's, and say what the handler is: ``ANIRA_PREPARE_EXCLUSIVE`` in
``anira_prepare_info.flags`` and ``ANIRA_ENGINE_CALL_EXCLUSIVE`` in ``anira_engine_ctx.flags``
(above). Where a custom engine appears, the engine-provider pair is ``ANIRA_ENGINE_NONE`` with
the id and the plan's provider beside it (``provider``, or ``provider_id`` for a custom one):
``anira_plan_info.engine``, ``engine_id``, ``provider`` and ``provider_id``,
``anira_stage_ctx.engine`` and ``provider`` in the stage's phases, ``anira_backend_id`` among
the candidates of ``anira_pipeline_add_inference`` (``engine_id`` set, ``engine``
``ANIRA_ENGINE_NONE``). The plan report's slot rows read ``ANIRA_BINDING_ENGINE``, and its
extension rows name the engine by its id.

Declared state and the two buffers
----------------------------------

A model with a declared State pair (:doc:`usage` section 3.4) is fed its state by anira,
without a copy: the handler owns two buffers per pair, the model's State input is bound to
the one the last inference wrote and the model's State output to the other, and the pair
flips behind every successful inference. For the engine this is the adapter rule and nothing
more: the State input's descriptor and the State output's descriptor name the pair's two
buffers in turn from one call to the next, so an engine that read the data pointer at load or
at prepare would read the wrong half every other inference. An engine that binds caller
memory (an ``Ort::Value`` over the descriptor, a ``torch::from_blob`` view) runs the pair
without a copy; one that stages its inputs copies as it copies every other slot.

An engine whose runtime wants the addresses to stay put, a captured graph above all, sets
``ANIRA_ENGINE_FLAG_STATE_ALIAS``: for every call of a plan of that engine anira binds **one**
buffer as both halves of the pair and never flips, so the State input's memory is the State
output's, and the engine reads the state before it writes it (in place) or copies for itself.
A failed call leaves the buffer as the engine left it (the last-good rule of a flipping pair,
which a failed inference does not flip, is the engine's to keep here), and a chunk of a plan
without the bit in between flips the pair, so the addresses stay put across the chunks of
aliasing plans alone.
The pair is the model's, not a plan's: every plan of the variant runs the same pair, in the
engine domain of the plans that run it (host memory for every engine of this pre-release; the
slot's declared host domain overrides it, :doc:`usage` section 1.3), so a plan switch keeps
the state whether the plans flip or alias (the one buffer an aliasing plan updates is the read
buffer a flipping plan reads next), and the first inference of a new stream reads zeros
either way.

A model with a declared State pair is stateful: it is loaded once, with no shared slot, and
every handler of it is exclusive (``ANIRA_PREPARE_EXCLUSIVE``), so an engine that keeps state
of its own beside the declared pair builds it at ``prepare`` and re-initialises it at
``reset``. The two buffers take the spec's dtype, and a State pair of any dtype the engine
accepts is legal on the handler (its buffers never travel through the inference queue): an
``int32`` pair runs on a registered engine that binds it. Every other model tensor is
``float32`` in this pre-release, the queue's storage; the built-in adapters refuse a model
with a tensor of another dtype at load with ``ANIRA_ERROR_CONFIG`` naming the engine and the
tensor, a registered engine binds what its ``load`` accepts.

Sharing and lifetime
--------------------

anira loads once per loaded model it **pools**. Two handlers share one loaded model and its
shared slots, and ``load`` runs once for both, when

- they run **the same engine object**, whichever pipelines they were created from (two
  objects never share, even under one id over the same callbacks and ``user_data``),
- their **model configurations are equal**, the whole variant: every entry, spec and extension,
  since ``load`` may read any of it through its record, and
- their plans run on **the same provider**, with the same provider options of their contexts
  (:doc:`usage` section 3.1): a provider is part of the loaded model, so two providers of one
  model are two loads, each its own ``loaded``, and
- their tensors **resolve to the same shapes** (two handlers with different resolved windows
  never share), with the same shared-slot count and the same fixed warm-up.

Whether a handler is exclusive is the handler's property, not the loaded model's, and no part
of the pool's key: a model declared ``ANIRA_MODEL_STATEFUL`` or with a declared State pair is
loaded once too, with ``instances`` ``0``, and its handlers are prepared on that one loaded
model one by one, each with ``ANIRA_PREPARE_EXCLUSIVE`` and each building its own executor;
they are the only handlers ``reset`` runs on. The order for one engine serving two handlers
of one model: ``init`` (once), ``load`` (once), the ``prepare`` of each handler, the
``process`` calls of both, the ``unprepare`` of each handler as it is re-prepared or
destroyed, ``unload`` with the last of them, ``release`` when the handle, the pipelines and
the handlers are all gone. An ``unprepare`` always precedes the ``unload`` of the model it ran
on; for one handler alone a re-prepare unloads and loads again, since the old session is
released before the new one is created.

What a refusal owes, at every level: a refused ``init`` fails that ``anira_handler_prepare``
and leaves the object uninitialised, so the next prepare calls ``init`` again; a refused
``load`` fails it and owes no ``unload``; a refused ``prepare`` fails it, owes no
``unprepare``, and the loaded model the session had acquired goes back, through ``unload``
when the handler was its last holder. Whatever an earlier plan of the same prepare loaded and
prepared goes back on the way out, through its ``unprepare`` and ``unload``, and the handler
is left unprepared; a refused ``anira_custom_engine_create`` owes no ``release``.

What a custom engine can serve here
-----------------------------------

A built-in engine's capabilities are the context's: its runtime is asked at
``anira_context_create`` and ``anira_context_probe``, and ``anira_capabilities_backends``
lists one row per provider it reports usable (:doc:`usage` section 3.1). A custom engine has
the same three parts, on the pipeline it belongs to: its **declared** vocabulary
(``providers``), a **measurement** (the ``query`` slot, run before ``init`` and any number of
times) and a **visible report**, the pipeline's capabilities:

.. code-block:: c

    /* The rows a handler of this pipeline sees on this context: the context's, then one per
       added engine and provider usable here (engine ANIRA_ENGINE_NONE, engine_id the
       engine's id), the default provider first. Every call runs the engines' queries. */
    uint32_t count = 0;
    anira_pipeline_capabilities_backends(pipe, context, sizeof(anira_backend_id), &count, NULL);
    anira_backend_id* rows = malloc(count * sizeof *rows);
    anira_pipeline_capabilities_backends(pipe, context, sizeof(anira_backend_id), &count, rows);

    /* One edge: host memory to the engine on a provider (zero-copy to the default provider and
       XNNPACK, a host copy the engine makes itself to every other). */
    anira_backend_id to = ANIRA_BACKEND_ID_INIT;
    to.engine = ANIRA_ENGINE_NONE;
    to.engine_id = "com.example.myengine";
    to.provider = ANIRA_PROVIDER_COREML;
    anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
    anira_pipeline_capabilities_edge(pipe, context, ANIRA_DOMAIN_HOST, &to, &edge);

The rows' strings point into the pipeline's engines and the context's store: valid until the
pipeline is destroyed and until the context's next probe. ``anira_handler_create`` asks the
same question for every candidate's provider of the engine, so what the report lists is what a
handler runs on. In C++ the query is ``Engine::available(const InitInfo&)`` (the base answers
every bit) and the report ``pipe.capabilities(context).backends()`` / ``.edge(from, to)``
(:cpp:class:`anira::PipelineCapabilities`).

Sharing an engine across plugin instances
-----------------------------------------

The built-in engines share models between plugin instances without any help: two instances of
one plugin with equal settings load a model once. A custom engine does the same when the
instances hand anira **the same engine object**. The plugin keeps one object for its whole
binary and adds it to every instance's pipeline:

.. code-block:: c

    /* One engine object per plugin binary, created by the first instance. */
    static anira_custom_engine* shared_engine(void) {
        static anira_custom_engine* engine = NULL;   /* guard with a lock if instances are
                                                        created on several threads */
        if (engine == NULL) {
            anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
            desc.process = my_process;
            desc.load = my_load;
            desc.unload = my_unload;
            desc.prepare = my_prepare;
            desc.unprepare = my_unprepare;
            anira_custom_engine_create("com.example.myengine", &desc, &engine, NULL);
        }
        return engine;
    }

    /* Every plugin instance: */
    anira_pipeline_add_engine(pipe, shared_engine(), &err);

The static handle keeps the engine alive for the life of the binary; destroy it at the plugin's
unload, after the last instance. In C++ the idiom is a function-local ``std::weak_ptr``, which
lets the engine go with the last instance (below). Two different plugin binaries never share a
custom engine: each has its own object, and a plugin's engine code never outlives the plugin.

An example: a gain engine in C
------------------------------

An engine that multiplies its first input into its first output, every slot filled. ``load``
keeps the element count of the templates and the shared-slot count, nothing of the memory;
``prepare`` keeps the loaded model it runs on and whether the handler is exclusive;
``process`` finds both through ``ctx->loaded`` and ``prepared`` and reads the pointers of its
own call; the entry's path is read but not opened, since this engine has no file to load:

.. code-block:: c

    #include <stdlib.h>
    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>   /* includes anira/abi/engine.h, lifecycle.h and tensor.h */

    typedef struct gain_engine { float gain; } gain_engine;        /* the user_data */
    typedef struct gain_loaded {                                    /* one loaded model */
        size_t elements;                                            /* of the first slot */
        uint32_t instances;                                         /* its shared slots */
        uint32_t provider;                                          /* what it runs on */
    } gain_loaded;
    typedef struct gain_prepared {                                  /* one handler on it */
        const gain_loaded* loaded;
        int exclusive;                                              /* ANIRA_PREPARE_EXCLUSIVE */
    } gain_prepared;

    /* Once per engine object, before its first load: the facts of the core; nothing to build. */
    static anira_status ANIRA_CALL gain_init(const anira_init_info* info, void* user_data) {
        (void)info; (void)user_data;                     /* info->log_level, info->num_threads */
        return ANIRA_OK;
    }

    static anira_status ANIRA_CALL gain_load(const anira_engine_load_info* info,
                                             void* user_data,
                                             void** out_loaded) {
        gain_loaded* l;
        (void)user_data;
        if (info->num_inputs < 1 || info->num_outputs < 1) { return ANIRA_ERROR_CONFIG; }
        /* The entry's file, through the getters on the variant; anira never opened it. */
        (void)anira_model_config_model_path(info->model, info->row);
        l = (gain_loaded*)calloc(1, sizeof(gain_loaded));
        if (l == NULL) { return ANIRA_ERROR_OUT_OF_MEMORY; }
        l->elements = anira_tensor_num_elements(&info->inputs[0]);   /* the template's shape */
        l->instances = info->instances;          /* 0 when every handler of it is exclusive */
        l->provider = info->provider;            /* the plan's; info->provider_id names a
                                                    custom one; this engine does the same work
                                                    on every provider it lists */
        *out_loaded = l;                         /* every prepare, process and reset gets it */
        return ANIRA_OK;
    }

    static void ANIRA_CALL gain_unload(void* loaded, void* user_data) {
        (void)user_data;
        free(loaded);                  /* once per successful load, after every unprepare */
    }

    static anira_status ANIRA_CALL gain_prepare(const anira_prepare_info* info,
                                                void* loaded,
                                                void* user_data,
                                                void** out_prepared) {
        gain_prepared* p;
        (void)user_data;
        p = (gain_prepared*)calloc(1, sizeof(gain_prepared));
        if (p == NULL) { return ANIRA_ERROR_OUT_OF_MEMORY; }
        p->loaded = (const gain_loaded*)loaded;
        p->exclusive = (info->flags & ANIRA_PREPARE_EXCLUSIVE) != 0;
        /* An engine with state per stream builds it here when exclusive, its own executor too. */
        *out_prepared = p;                       /* every process, reset and unprepare gets it */
        return ANIRA_OK;
    }

    /* Real-time: the accessors are ANIRA_NONBLOCKING, and so is this body (REALTIME_SAFE). */
    static anira_status ANIRA_CALL gain_process(const anira_engine_ctx* ctx,
                                                void* prepared,
                                                void* user_data) ANIRA_NONBLOCKING {
        const gain_loaded* l = (const gain_loaded*)ctx->loaded;      /* what load handed back */
        const gain_prepared* p = (const gain_prepared*)prepared;     /* what prepare handed back */
        const float gain = ((const gain_engine*)user_data)->gain;
        const float* in = anira_tensor_data_f32(&ctx->inputs[0]);   /* THIS call's memory */
        float* out = anira_tensor_data_f32(&ctx->outputs[0]);
        size_t n;
        if (p == NULL || p->loaded != l) { return ANIRA_ERROR_INTERNAL; }
        if (in == NULL || out == NULL) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        if (anira_tensor_num_elements(&ctx->outputs[0]) < l->elements) {
            return ANIRA_ERROR_CONFIG;
        }
        /* A shared call runs on slot ctx->instance (below l->instances) of the loaded model, an
           exclusive call (ANIRA_ENGINE_CALL_EXCLUSIVE, instance 0) on what p holds: this engine
           keeps nothing per slot, so both do the same work. */
        for (n = 0; n < l->elements; ++n) { out[n] = in[n] * gain; }
        return ANIRA_OK;
    }

    /* The first inference of an exclusive handler's new stream: nothing kept per stream here. */
    static void ANIRA_CALL gain_reset(const anira_engine_ctx* ctx,
                                      void* prepared,
                                      void* user_data) {
        (void)ctx; (void)prepared; (void)user_data;
    }

    static void ANIRA_CALL gain_unprepare(void* prepared, void* user_data) {
        (void)user_data;
        free(prepared);                          /* once per successful prepare, before unload */
    }

    static void ANIRA_CALL gain_release(void* user_data) {
        (void)user_data;                         /* once, after every unload */
    }

At setup:

.. code-block:: c

    static gain_engine engine = { 0.5f };
    static const char* const providers[] = { "coreml" };  /* beyond the default provider */
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE;       /* the promise the body keeps */
    desc.providers = providers;                         /* the enum's spellings, or */
    desc.num_providers = 1;                             /* any name of the engine's own */
    desc.process = gain_process;
    desc.reset = gain_reset;                  /* init, reset and release could stay NULL */
    desc.prepare = gain_prepare;
    desc.unprepare = gain_unprepare;
    desc.load = gain_load;
    desc.unload = gain_unload;
    desc.init = gain_init;
    desc.release = gain_release;

    anira_custom_engine* gain = NULL;
    anira_custom_engine_create("com.example.gain", &desc, &gain, &err);
    anira_pipeline_add_engine(pipe, gain, &err);
    anira_custom_engine_destroy(gain);

The same in C++
---------------

:cpp:class:`anira::Engine` of ``anira/anira.hpp`` is the descriptor with virtual functions in
place of the function pointers, split as the C lifecycle is, exactly like
:cpp:class:`anira::Stage`: the engine object, ``anira::Engine``, is constructed with its id
(``anira::Engine(std::string id)``, read back by ``id()``), states its promise in
``flags()``, its extensions in ``consumed_kinds()`` and the providers it serves beyond the
default one in ``providers()`` (all read once, when its C engine is
created at its first registration), may override ``available(const InitInfo&)`` (the query,
the base answers every bit) and ``init(const InitInfo&)`` (the base does
nothing), and its ``load(const EngineLoadInfo&)`` returns a
``std::unique_ptr<Engine::Loaded>``, the loaded model, whose ``prepare(const PrepareInfo&)``
returns a ``std::unique_ptr<Engine::Prepared>``, the handler's handle, on which
``process(EngineContext&)`` and ``reset(EngineContext&)`` are the virtuals (``noexcept``; the
base ``reset`` does nothing). The records are views: :cpp:class:`anira::InitInfo`
(``log_level()``, ``num_threads()``, ``context()``), :cpp:class:`anira::EngineLoadInfo`
(``row()``, ``model()``, the getters ``model_path(i)``, ``model_bytes(i)``,
``model_engine_id(i)``, ``inputs()`` / ``outputs()`` as ``std::span<const Tensor>``,
``input_names()`` / ``output_names()``, ``instances()``, ``provider()``,
``provider_id()`` and the options ``option_keys()`` / ``option_values()``),
:cpp:class:`anira::PrepareInfo`
(``handler()``, ``report()``, ``num_entries()``, ``inputs()`` / ``outputs()``,
``input_names()`` / ``output_names()``, ``flags()`` and ``exclusive()``), and
:cpp:class:`anira::EngineContext` the context (``instance()``, ``entry()``, ``ticket()``,
``flags()`` and ``exclusive()``, ``loaded()``, ``inputs()`` and the writable ``outputs()`` as
spans of :cpp:struct:`anira::Tensor`, every method a ``noexcept`` field read). ``init``,
``load`` and ``prepare`` may throw: an ``anira::Error`` fails the handler's prepare with its
status, ``std::bad_alloc`` with ``ANIRA_ERROR_OUT_OF_MEMORY``, anything else with
``ANIRA_ERROR_INTERNAL``, and ``what()`` goes to the log (a null return from ``load`` or
``prepare`` is ``ANIRA_ERROR_INTERNAL`` too); a throw never crosses the C boundary. Why the
ownership spellings: the loaded model is anira's from ``load`` on (deleted at the C
``unload``) and the prepared handle from ``prepare`` on (deleted at the C ``unprepare``), so a
``std::unique_ptr`` says so each time, while the engine object is shared by every pipeline it
is registered on and every handler created from them (a ``std::shared_ptr``). What every
handler shares is a member of the ``Loaded``, what an exclusive handler keeps per stream (its
own executor) a member of the ``Prepared``; nothing an inference needs is a member of the
``Engine``. The engine is registered with ``Pipeline::register_engine(impl)``, or brought
along by the inference stage, ``stage::Inference(cfg).engine(impl)``, which
``Pipeline::add`` registers before it adds the stage.

The ``anira::Engine`` object is the engine's identity, as ``anira_custom_engine`` is in C. The
first registration of an object creates its C engine under the object's id; every later
registration of **the same object**, on another ``Pipeline``, reuses that C engine for as long
as anything holds it (a ``Pipeline``, a handler created from one, a loaded model in the pool:
the object keeps a detached handle of it), so handlers of all of them share loaded models. The
C engine holds a copy of the ``shared_ptr``; ``release()`` runs once per C engine, when the
last pipeline, handler and loaded model carrying it are gone, and a registration after that
creates a fresh C engine with a ``release()`` of its own. An id the C engine's create refuses
creates nothing. A
registration refused because the pipeline already has an engine with the object's id takes no
reference; when that call created the object's C engine, the C engine is dropped again and
``release()`` answers it. The gain engine, its ``process`` the passthrough of the first slot
times the gain:

.. code-block:: cpp

    #include <algorithm>
    #include <array>
    #include <memory>
    #include <anira/anira.hpp>

    class Gain : public anira::Engine {
    public:
        explicit Gain(float gain) : anira::Engine("com.example.gain"), m_gain(gain) {}
        uint32_t flags() const noexcept override { return ANIRA_ENGINE_FLAG_REALTIME_SAFE; }
        // The providers served beyond the default one: the enum's spellings or any name of
        // the engine's own; the gain does the same work on every one of them.
        std::span<const char* const> providers() const noexcept override { return k_providers; }
        // init(const anira::InitInfo&) is not overridden: nothing to build once per object.
        std::unique_ptr<Loaded> load(const anira::EngineLoadInfo& info) override;

    private:
        static constexpr std::array<const char*, 1> k_providers{"coreml"};
        class Shared;
        class Run;
        float m_gain;
    };

    // One loaded model: what every handler of it shares, immutable after load.
    class Gain::Shared : public anira::Engine::Loaded {
    public:
        Shared(float gain, std::size_t elements) : m_gain(gain), m_elements(elements) {}
        std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override;

    private:
        float m_gain;
        std::size_t m_elements;
    };

    // One handler on it: process runs here, a shared call on the slot the context names, an
    // exclusive call on what this object built at prepare (nothing: the gain keeps no state).
    class Gain::Run : public anira::Engine::Prepared {
    public:
        Run(float gain, std::size_t elements) : m_gain(gain), m_elements(elements) {}

        anira_status process(anira::EngineContext& ctx) noexcept override {
            const float* in = ctx.inputs()[0].data_f32();        // THIS call's memory
            float* out = ctx.outputs()[0].data_f32();
            if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
            const std::size_t count = std::min(m_elements, ctx.outputs()[0].num_elements());
            for (std::size_t n = 0; n < count; ++n) { out[n] = in[n] * m_gain; }
            return ANIRA_OK;
        }
        // reset(anira::EngineContext&) is not overridden: nothing kept per stream.

    private:
        float m_gain;
        std::size_t m_elements;
    };

    std::unique_ptr<anira::Engine::Loaded> Gain::load(const anira::EngineLoadInfo& info) {
        if (info.inputs().empty() || info.outputs().empty()) {
            throw anira::Error(ANIRA_ERROR_CONFIG,
                               "the gain engine needs one input and one output");
        }
        return std::make_unique<Shared>(m_gain, info.inputs()[0].num_elements());   // anira owns it
    }

    std::unique_ptr<anira::Engine::Prepared>
    Gain::Shared::prepare(const anira::PrepareInfo& info) {
        // An engine with state per stream builds its own executor here when info.exclusive().
        static_cast<void>(info);
        return std::make_unique<Run>(m_gain, m_elements);   // anira owns it
    }

    anira::ModelConfig cfg = anira::ModelConfig::from_file("gain.model.json");
    cfg.add_model_path("com.example.gain", "gain.bin");             // the entry the engine serves
    anira::Pipeline pipe{anira::stage::Inference(cfg).engine(std::make_shared<Gain>(0.5F))};
    // or: pipe.register_engine(std::make_shared<Gain>(0.5F));

To share the engine across the instances of a plugin, every instance registers the same object,
kept once per binary; the ``std::weak_ptr`` lets it go with the last instance:

.. code-block:: cpp

    std::shared_ptr<anira::Engine> shared_gain() {
        static std::mutex mutex;
        static std::weak_ptr<anira::Engine> cache;       // one per plugin binary
        const std::lock_guard<std::mutex> lock(mutex);
        if (std::shared_ptr<anira::Engine> engine = cache.lock()) { return engine; }
        auto engine = std::make_shared<Gain>(0.5F);
        cache = engine;
        return engine;
    }

    // every plugin instance, each with its own Pipeline:
    pipe.register_engine(shared_gain());

Every instance's ``Pipeline`` holds the object's C engine, so two instances with equal
configurations load the model once. Keep the ``Pipeline`` for as long as the instance runs:
a registration made after every ``Pipeline`` holding the object is gone creates a new C engine,
which does not share with handlers of the old one.

Using an engine directly in a custom engine
-------------------------------------------

A custom engine may drive one of the bundled inference engines itself, for example to use an
ONNX Runtime execution provider or session option anira's own ONNX Runtime adapter
(``src/backends/OnnxRuntimeAdapter.cpp``, internal) does not expose. Two rules keep that
safe, and they are the same rules anira's own adapters follow.

**Link the engine target.** ``anira::anira`` carries anira's headers and the ``USE_<ENGINE>``
definitions, but no engine header: anira links its engines privately. The engine's include
directory and library live on the matching engine target, which exists both in anira's build
tree and in the installed package:

.. code-block:: cmake

    target_link_libraries(my_engine PRIVATE anira::anira anira::onnxruntime)

``anira::onnxruntime``, ``anira::tflite``, ``anira::litert``, ``anira::libtorch`` and
``anira::executorch`` are the very files anira links, a shared library in a shared anira build,
a static archive (linked on demand and hidden) in a static one, so the process holds exactly
one copy of the engine. Never link an engine by any other path next to anira.

**Keep the engine out of your header.** The loaded model and the prepared handle are the
natural places for the engine's objects, split as the built-in adapters split them: what one
load shares (the environment, the options) and the executors of the shared slots in the
``Engine::Loaded``, an exclusive handler's own executor in its ``Engine::Prepared``. Declare
both subclasses in the ``.cpp`` alone, so the header that a plugin includes names no engine
type, exactly like anira's adapters, which are file-local classes over the engine:

.. code-block:: cpp

    // MyOnnxEngine.h: no engine include here
    #include <anira/anira.hpp>

    class MyOnnxEngine final : public anira::Engine {
    public:
        std::unique_ptr<Loaded> load(const anira::EngineLoadInfo& info) override;   // in the .cpp
    };

.. code-block:: cpp

    // MyOnnxEngine.cpp: the only place the engine header is included
    #include "MyOnnxEngine.h"
    #include <onnxruntime_cxx_api.h>

    namespace {

    class Model final : public anira::Engine::Loaded {       // owns the shared Ort:: objects
    public:
        explicit Model(const anira::EngineLoadInfo& info) {
            // the model from info.model_path(info.row()) or info.model_bytes(info.row()), the
            // graph's names matched against info.input_names() / output_names(); one
            // Ort::Session per shared slot, info.instances() of them (none for a stateful model)
        }
        std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override;
        std::unique_ptr<Ort::Session> make_session();       // one more over m_env and m_options
        Ort::Session& session(uint32_t instance) { return *m_sessions[instance]; }

    private:
        Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "my-engine"};
        Ort::SessionOptions m_options;
        std::vector<std::unique_ptr<Ort::Session>> m_sessions;   // one per shared slot
    };

    class Run final : public anira::Engine::Prepared {        // one handler on the model
    public:
        Run(Model& model, std::unique_ptr<Ort::Session> own)
            : m_model(model), m_own(std::move(own)) {}
        anira_status process(anira::EngineContext& ctx) noexcept override {
            // Ort::Value over ctx.inputs()[i] and ctx.outputs()[i] of THIS call, run on m_own
            // when ctx.exclusive(), else on m_model.session(ctx.instance())
            return ANIRA_OK;
        }
    private:
        Model& m_model;
        std::unique_ptr<Ort::Session> m_own;   // an exclusive handler's own; null for a shared one
    };

    std::unique_ptr<anira::Engine::Prepared> Model::prepare(const anira::PrepareInfo& info) {
        std::unique_ptr<Ort::Session> own;
        if (info.exclusive()) { own = make_session(); }   // its calls claim no shared slot
        return std::make_unique<Run>(*this, std::move(own));
    }

    }  // namespace

    std::unique_ptr<anira::Engine::Loaded> MyOnnxEngine::load(const anira::EngineLoadInfo& info) {
        return std::make_unique<Model>(info);
    }

Whatever includes your header then compiles with anira's include directories alone, and only
your ``.cpp`` needs the engine target's, which is what lets a plugin keep every engine symbol
private: compile that ``.cpp`` with hidden visibility (see :doc:`troubleshooting`, "Host
application ships its own backend runtime").

.. note::
    Subclassing :cpp:class:`anira::BackendBase` is the 2.x runtime's way of adding an engine,
    and stays with the 2.x :cpp:class:`anira::InferenceHandler` until the runtime cut-over;
    :doc:`migration` maps each of its virtuals onto the descriptor above. The engines anira
    ships are implemented as adapters of the same descriptor shape (``src/backends/``, internal),
    which is where to look for how a real engine binds by name, checks the shapes it was
    handed, splits what one load shares from what one executor owns, and stages its buffers.
