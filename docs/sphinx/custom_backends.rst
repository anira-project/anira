Custom Engines
==============

An **engine** runs the model: it takes the model tensors of a chunk in, runs one inference,
and writes the model tensors out. anira ships five (ONNX Runtime, LibTorch, ExecuTorch, LiteRT
and TensorFlow Lite, ``ANIRA_ENGINE_*``), and a host adds its own: it **creates** the engine
once as an object and **adds** it to a pipeline under a reverse-URI id. Examples: a Core ML or
a WebGPU runtime, an engine the build does not carry, a synthetic model, a bypass for
measuring the pipeline's overhead. A custom engine is one more implementation the inference stage resolves to, never a stage of its own (the
stage is the pre- and post-processing around it, :doc:`custom_preprocessing`): a model entry
names it by its id, and the handler prepares it, runs it and resets it exactly as it does a
built-in engine, with the same plan report, the same real-time rules and the same failure
path. In C it is a descriptor, ``anira_engine_desc`` of ``anira/abi/engine.h``; in C++ the
same descriptor is a subclass of :cpp:class:`anira::Engine` (below). Its lifecycle is the
stage's, with the same words on both sides: ``prepare`` hands back a ``prepared`` pointer,
every per-chunk call receives it as ``(ctx, prepared, user_data)``, ``reset`` runs at a
stream boundary, ``unprepare`` frees what one prepare loaded, ``release`` fires once per
engine object.

Creating and adding an engine
-----------------------------

.. code-block:: c

    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>   /* includes anira/abi/engine.h */

    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;   /* every slot NULL, no promise */
    desc.user_data = &my_engine;                       /* handed to every callback as it is */
    desc.process = my_process;                         /* the one required slot */
    desc.prepare = my_prepare;
    desc.unprepare = my_unprepare;

    anira_custom_engine* engine = NULL;
    anira_custom_engine_create(&desc, &engine, &err);                /* the engine object */
    anira_pipeline_add_engine(pipe, "com.example.myengine", engine, &err);
    anira_custom_engine_destroy(engine);          /* the pipeline keeps its own reference */

    uint32_t row = 0;                                  /* the entry the engine serves */
    anira_model_config_add_model_path_custom(cfg, "com.example.myengine", "model.bin", &row, &err);

Two calls, two jobs:

- ``anira_custom_engine_create(&desc, &engine, &err)`` copies the descriptor (within
  ``struct_size``, with its strings) into a refcounted **engine object**. The object is what
  the engine *is*: anira recognises the same engine by it, and shares prepared models between
  handlers that run it (below).
- ``anira_pipeline_add_engine(pipe, id, engine, &err)`` adds the engine to a pipeline under an
  **id**, which is how the pipeline's model entries name it. The id belongs to the pipeline, not
  to the engine: one engine may be added to any number of pipelines, under the same id or
  different ones, and to one pipeline under two ids.

The handle, every pipeline the engine was added to, every handler created from those pipelines
and every prepared model of the engine hold a reference. ``anira_custom_engine_destroy`` drops
the handle's, so it may run right after the last ``anira_pipeline_add_engine``; ``release``
fires exactly once, when the last reference is gone. The id is a reverse-URI name: it must
contain a ``.``, and the prefix ``anira.`` is anira's own (``anira.v2.custom`` names the 2.x
``CUSTOM`` backend of :doc:`migration`). Adding is legal before or after
``anira_pipeline_add_inference``; a handler copies the pipeline at ``anira_handler_create``, so
an engine added afterwards does not reach that handler. A model entry that names the id (``anira_model_config_add_model_path_custom``,
``anira_model_config_add_model_bytes_custom``; ``"engine": "com.example.myengine"`` in a model
file) is a plan of the handler like a built-in engine's entry; several such entries on one
model configuration are several plans, which ``anira_handler_set_plan`` switches between,
resolved by entry, never by the 2.x backend they map to. An added engine no entry names is not
a plan and not an error; an entry whose id no engine of the pipeline serves is
``ANIRA_ERROR_NOT_SUPPORTED`` at ``anira_handler_create``, naming the id. anira never opens
a custom entry's path or reads its bytes: the engine's ``prepare`` does, through the record it
receives.

``anira_custom_engine_create`` refuses with ``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL``
descriptor or ``out``, a ``struct_size`` below the three leading slots (``struct_size``,
``abi_version``, ``user_data``), a ``flags`` bit the header does not define, a ``NULL``
``process`` and a ``NULL`` ``consumed_kinds`` (or entry) with a count above 0, and with
``ANIRA_ERROR_ABI_VERSION`` an ``abi_version`` ``anira_check_abi`` refuses; a refused create
hands out nothing and never calls ``release``. ``anira_pipeline_add_engine`` refuses with
``ANIRA_ERROR_INVALID_ARGUMENT`` a ``NULL`` pipeline, id or engine and an id without a ``.`` or
with the prefix ``anira.``, and with ``ANIRA_ERROR_INVALID_STATE`` an id the pipeline already
has; a refused addition takes no reference.

The descriptor
--------------

``anira_engine_desc`` (Tier 2: ``struct_size`` first, ``user_data`` third, growth at the tail;
``ANIRA_ENGINE_DESC_INIT`` is an engine without a callback and without a promise) names, in
the order of the lifecycle:

- ``user_data``: handed to every callback as it is, never read by anira. One per engine
  object, shared by every pipeline it was added to and every prepared model of the engine; what one prepared model keeps
  lives behind the ``prepared`` pointer its ``prepare`` hands back.
- ``consumed_kinds`` / ``num_consumed_kinds``: the extensions the engine reads at prepare, as
  ``"<host>:<kind>"`` strings (hosts ``tensor_spec``, ``model``, ``model_config``,
  ``context``, ``contract``), copied. They join the consumed-or-fail walk for the entries of
  this engine, keyed by its id: its ``model:`` kinds read its own entries alone, its other
  kinds any host, and each consumed slot is an ``anira_plan_ext`` row of the plan whose
  consumer reads the engine's id.
- ``flags``: the engine's promises, an OR of the three ``ANIRA_ENGINE_FLAG_*`` bits below;
  ``0`` promises nothing.
- ``process``: the engine call, ``anira_engine_process_fn``. Required.
- ``reset``: ``anira_engine_reset_fn``. ``NULL``: nothing to reset.
- ``prepare``: ``anira_engine_prepare_fn``. ``NULL``: nothing to prepare, and ``prepared`` is
  ``NULL`` in every later call.
- ``unprepare``: ``anira_engine_unprepare_fn``. ``NULL``: none.
- ``release``: ``anira_engine_release_fn``. ``NULL``: none.

The lifecycle
-------------

**prepare(info, user_data, &prepared)** runs on the caller of ``anira_handler_prepare``, once
per prepared model anira pools (below), under the core's lifecycle lock: it may allocate, read
files, log, call the config getters and the tensor accessors, and must not call an entry that
takes that lock (context create and destroy, handler create, destroy and prepare,
``anira_inference_thread_create``, ``anira_shutdown``, ``anira_release_core_if_idle``). This
is where the engine loads the entry's model, binds the slots and sizes what its instances
need. Its record, ``anira_engine_prepare_info`` (Tier 2, valid for the duration of the call;
the engine copies what it keeps and never keeps the pointers):

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
  binds each slot to the tensor of that name and refuses prepare for a name its side does not
  have; an engine without names binds by position. Either way the plan report says
  ``ANIRA_BINDING_ENGINE`` for every slot of the plan: the engine received the names and bound
  itself.
- ``instances``: the ``process`` calls that may run at once on this prepared model, each on
  its own instance below this count (``ctx->instance``): ``1`` for a session-exclusive model,
  the model's ``max_instances`` for a shared one, at most the core's thread count.

The record carries no warm-up count: the contract's fixed warm-up is what the built-in
adapters run inside their own prepare, and an engine that wants a warm inference runs it here
itself. A status other than ``ANIRA_OK`` fails ``anira_handler_prepare`` with it, the message
naming the engine (``the engine 'com.example.myengine' refused prepare: it returned N``), and
no ``unprepare`` follows for that prepare. What the call hands back through ``out_prepared``
(which starts ``NULL``; ``NULL`` is a legal value) is the ``prepared`` pointer every
``process``, ``reset`` and ``unprepare`` of **this** prepared model receives beside the
engine's ``user_data``: two pointers, two lifetimes.

**process(ctx, prepared, user_data)** runs on an inference thread, in
``ANIRA_PHASE_INFERENCE`` (between the stage's ``before_inference`` and ``after_inference``),
one inference over the context's tensors on the instance the context names. It may block; it
must not allocate per call. Its context, ``anira_engine_ctx`` (Tier 1, 64 bytes, on anira's
stack for the duration of the call, like the stage's ``anira_stage_ctx``):

- ``instance``: the instance of the prepared model this call runs on, ``0 .. instances - 1``;
  never two calls at once on one instance.
- ``entry``: the chunk's position in the handler's inference queue, the stage context's
  ``entry`` of the same chunk (:doc:`usage` section 2).
- ``inputs`` / ``num_inputs`` and ``outputs`` / ``num_outputs``: one descriptor per slot of
  either side in slot order, State tensors included, over the memory the engine reads and
  writes: the spec's dtype and the engine-side extents of the prepare record's templates, with
  the data.
- ``ticket``: ``ANIRA_TICKET_INVALID`` under a Hard contract, the job's ticket under an Async
  one; ``flags``: per-call flags, none defined in this pre-release; two reserved words and two
  reserved pointer slots that read 0 and ``NULL``, through which the per-call facts grow.

**The adapter rule**, the one rule every engine follows, built in or registered: read every
extent and every memory handle (the data pointer, ``byte_offset``, the strides, the domain)
from the tensors of **this** call, never from a value kept at prepare, and never assume a
tensor is anira's own buffer. The halves of a declared State pair alternate between two
buffers from one inference to the next (below), and a later pre-release hands a caller's
Buffer tensor over in place. The templates of the prepare record say what the shapes will be;
the memory is the context's. A Time extent below the template's is legal only under
``ANIRA_ENGINE_FLAG_DYNAMIC_TIME``. The typedef carries no real-time attribute: whether the
body is real-time is the engine's own promise, ``ANIRA_ENGINE_FLAG_REALTIME_SAFE``, and an
author who wants clang's compile-time check declares the function ``ANIRA_NONBLOCKING``
themselves; the tensor accessors (``anira_tensor_data_f32``, ``anira_tensor_num_elements``,
``anira_tensor_plane``) are ``ANIRA_NONBLOCKING``, so a body composed of them keeps the promise.

A ``process`` that returns any status but ``ANIRA_OK`` fails the chunk, as a built-in engine's
failure does: ``anira_handler_rt_error`` reads ``ANIRA_ERROR_ENGINE`` with one latched record
naming the status, anira zeroes the outputs, the chunk delivers zeros at its stream position,
the stage's ``after_inference`` does not run for it, and a State pair keeps its last good
value. A throw across this boundary is undefined.

**reset(ctx, prepared, user_data)** re-initialises the state an engine keeps inside itself (a
recurrent hidden state the model does not declare, a filter's history): for a
**session-exclusive** prepared model, at the first inference of a new stream (after prepare,
after ``anira_handler_reset``), on the claimed instance, with that inference's context, right
before its ``process``; never for a shared prepared model, which several handlers may run and
which anira never resets. ``NULL`` for an engine without such state. (Of the built-in engines
only TensorFlow Lite has a body, ``TfLiteInterpreterResetVariableTensors``.)

**unprepare(prepared, user_data)** frees what one prepare loaded: once per successful
prepare, on the thread of the ``anira_handler_prepare`` or ``anira_handler_destroy`` that
drops the last handler sharing the prepared model, after that handler's in-flight inferences
have drained, so no ``process`` or ``reset`` of the prepared model runs afterwards. It may run
under the core's lifecycle lock (it does when a later plan of the same prepare fails and the
plans prepared before it are unprepared on the way out), so like ``prepare`` it must not call
an entry that takes that lock.

**release(user_data)** runs exactly once, when the last reference to the engine object dies
(``anira_custom_engine_destroy``, ``anira_pipeline_destroy`` or ``anira_handler_destroy``,
whichever comes last), after every ``unprepare``; no callback of the engine runs afterwards.

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

A bit the header does not define is ``ANIRA_ERROR_INVALID_ARGUMENT`` at
``anira_custom_engine_create``; the
name ``ANIRA_ENGINE_FLAG_STATE_ALIAS`` is reserved for the aliased State pair of a later
pre-release. Where a custom engine appears, the engine-provider pair is
``ANIRA_ENGINE_NONE`` with the id: ``anira_plan_info.engine`` and ``engine_id``,
``anira_stage_ctx.engine`` in the stage's phases, ``anira_backend_id`` among the candidates of
``anira_pipeline_add_inference`` (``engine_id`` set, ``engine`` ``ANIRA_ENGINE_NONE``). The
plan report's slot rows read ``ANIRA_BINDING_ENGINE``, and its extension rows name the engine
by its id.

Declared state and the two buffers
----------------------------------

A model with a declared State pair (:doc:`usage` section 3.4) is fed its state by anira,
without a copy: the handler owns two buffers per pair, the model's State input is bound to
the one the last inference wrote and the model's State output to the other, and the pair
flips behind every successful inference. For the engine this is the adapter rule and nothing
more: the State input's descriptor and the State output's descriptor name the pair's two
buffers in turn from one call to the next, so an engine that read the data pointer at prepare
would read the wrong half every other inference. An engine that binds caller memory (an
``Ort::Value`` over the descriptor, a ``torch::from_blob`` view) runs the pair without a
copy; one that stages its inputs copies as it copies every other slot.

The two buffers take the spec's dtype, and a State pair of any dtype the engine accepts is
legal on the handler (its buffers never travel through the inference queue): an ``int32``
pair runs on a registered engine that binds it. Every other model tensor is ``float32`` in
this pre-release, the queue's storage; the built-in adapters refuse a model with a tensor of
another dtype at prepare with ``ANIRA_ERROR_CONFIG`` naming the engine and the tensor, a
registered engine binds what its ``prepare`` accepts.

Sharing and lifetime
--------------------

anira prepares once per prepared model it **pools**. Two handlers share one prepared model and
its instances, and ``prepare`` runs once for both, when

- they run **the same engine object**, whichever pipelines they were created from and under
  whichever ids those pipelines added it (two objects never share, even over the same
  callbacks and ``user_data``),
- their **model configurations are equal**, the whole variant: every entry, spec and extension,
  since ``prepare`` may read any of it through its record (so the id the entry names is part of
  it, and two pipelines that add one engine under different ids prepare twice), and
- their tensors **resolve to the same shapes** (two handlers with different resolved windows
  never share).

A **session-exclusive** model, one declared ``ANIRA_MODEL_STATEFUL`` or one with a declared
State pair, is prepared for its handler alone, with ``instances`` ``1``, and is the only one
``reset`` runs on. ``unprepare`` fires when the last handler sharing the prepared model is
re-prepared or destroyed; ``release`` once, when the last reference to the engine object dies.
The order for one engine serving two handlers: ``prepare`` (once, or once per handler), the
``process`` calls of both handlers, the ``unprepare`` of each prepared model as its last handler
goes, ``release`` when the handle, the pipelines and the handlers are all gone.

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
            desc.prepare = my_prepare;
            desc.unprepare = my_unprepare;
            anira_custom_engine_create(&desc, &engine, NULL);
        }
        return engine;
    }

    /* Every plugin instance: */
    anira_pipeline_add_engine(pipe, "com.example.myengine", shared_engine(), &err);

The static handle keeps the engine alive for the life of the binary; destroy it at the plugin's
unload, after the last instance. In C++ the idiom is a function-local ``std::weak_ptr``, which
lets the engine go with the last instance (below). Two different plugin binaries never share a
custom engine: each has its own object, and a plugin's engine code never outlives the plugin.

An example: a gain engine in C
------------------------------

An engine that multiplies its first input into its first output. ``prepare`` keeps the
element count of the templates and nothing of the memory; ``process`` reads the pointers of
its own call; the entry's path is read but not opened, since this engine has no file to load:

.. code-block:: c

    #include <stdlib.h>
    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>   /* includes anira/abi/engine.h and anira/abi/tensor.h */

    typedef struct gain_engine { float gain; } gain_engine;               /* the user_data */
    typedef struct gain_prepared { size_t elements; } gain_prepared;    /* one prepared model */

    static anira_status ANIRA_CALL gain_prepare(const anira_engine_prepare_info* info,
                                                void* user_data,
                                                void** out_prepared) {
        gain_prepared* p;
        (void)user_data;
        if (info->num_inputs < 1 || info->num_outputs < 1) { return ANIRA_ERROR_CONFIG; }
        /* The entry's file, through the getters on the variant; anira never opened it. */
        (void)anira_model_config_model_path(info->model, info->row);
        p = (gain_prepared*)calloc(1, sizeof(gain_prepared));
        if (p == NULL) { return ANIRA_ERROR_OUT_OF_MEMORY; }
        p->elements = anira_tensor_num_elements(&info->inputs[0]);    /* the template's shape */
        *out_prepared = p;                                            /* every process gets it back */
        return ANIRA_OK;
    }

    /* Real-time: the accessors are ANIRA_NONBLOCKING, and so is this body (REALTIME_SAFE). */
    static anira_status ANIRA_CALL gain_process(const anira_engine_ctx* ctx,
                                                void* prepared,
                                                void* user_data) ANIRA_NONBLOCKING {
        const gain_prepared* p = (const gain_prepared*)prepared;
        const float gain = ((const gain_engine*)user_data)->gain;
        const float* in = anira_tensor_data_f32(&ctx->inputs[0]);   /* THIS call's memory */
        float* out = anira_tensor_data_f32(&ctx->outputs[0]);
        size_t n;
        if (in == NULL || out == NULL) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        if (anira_tensor_num_elements(&ctx->outputs[0]) < p->elements) { return ANIRA_ERROR_CONFIG; }
        for (n = 0; n < p->elements; ++n) { out[n] = in[n] * gain; }
        return ANIRA_OK;
    }

    static void ANIRA_CALL gain_unprepare(void* prepared, void* user_data) {
        (void)user_data;
        free(prepared);                                   /* once per successful prepare */
    }

At setup:

.. code-block:: c

    static gain_engine engine = { 0.5f };
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE;       /* the promise the body keeps */
    desc.process = gain_process;
    desc.prepare = gain_prepare;
    desc.unprepare = gain_unprepare;                    /* reset and release stay NULL */

    anira_custom_engine* gain = NULL;
    anira_custom_engine_create(&desc, &gain, &err);
    anira_pipeline_add_engine(pipe, "com.example.gain", gain, &err);
    anira_custom_engine_destroy(gain);

The same in C++
---------------

:cpp:class:`anira::Engine` of ``anira/anira.hpp`` is the descriptor with virtual functions in
place of the function pointers, split as the C lifecycle is, exactly like
:cpp:class:`anira::Stage`: the engine object, ``anira::Engine``, states its promise in
``flags()`` and its extensions in ``consumed_kinds()`` (both read once, when its C engine is
created at its first registration),
and its ``prepare(const EnginePrepareInfo&)`` returns a ``std::unique_ptr<Engine::Prepared>``,
the prepared model, on which ``process(EngineContext&)`` and ``reset(EngineContext&)`` are the
virtuals (``noexcept``; the base ``reset`` does nothing). :cpp:class:`anira::EnginePrepareInfo`
is the record as views (``row()``, ``model()``, the getters ``model_path(i)``,
``model_bytes(i)``, ``model_engine_id(i)``, ``inputs()`` / ``outputs()`` as
``std::span<const Tensor>``, ``input_names()`` / ``output_names()``, ``instances()``), and
:cpp:class:`anira::EngineContext` the context (``instance()``, ``entry()``, ``ticket()``,
``flags()``, ``inputs()`` and the writable ``outputs()`` as spans of :cpp:struct:`anira::Tensor`,
every method a ``noexcept`` field read). ``prepare`` may throw: an ``anira::Error`` fails the
handler's prepare with its status, ``std::bad_alloc`` with ``ANIRA_ERROR_OUT_OF_MEMORY``,
anything else with ``ANIRA_ERROR_INTERNAL``, and ``what()`` goes to the log (a null return is
``ANIRA_ERROR_INTERNAL`` too); a throw never crosses the C boundary. Why the two ownership
spellings: the prepared model is anira's from ``prepare`` on (deleted at the C ``unprepare``,
so a ``std::unique_ptr`` says so), while the engine object is shared by every pipeline it is
registered on and every handler created from them (a ``std::shared_ptr``). The engine is
registered with ``Pipeline::register_engine(id, impl)``, or brought along by the inference
stage, ``stage::Inference(cfg).engine(id, impl)``, which ``Pipeline::add`` registers before it
adds the stage.

The ``anira::Engine`` object is the engine's identity, as ``anira_custom_engine`` is in C. The
first registration of an object creates its C engine; every later registration of **the same
object**, on the same ``Pipeline`` or another, reuses that C engine for as long as a
``Pipeline`` holding it lives, so handlers of all of them share prepared models. The C engine
holds a copy of the ``shared_ptr``; ``release()`` runs once per C engine, when the last
pipeline and handler carrying it are gone, and a registration after that creates a fresh C
engine with a ``release()`` of its own. A registration refused for its id after it created the
object's C engine drops that C engine again, and ``release()`` answers it. The gain engine, its
``process`` the passthrough of the first slot times the gain:

.. code-block:: cpp

    #include <algorithm>
    #include <memory>
    #include <anira/anira.hpp>

    class Gain : public anira::Engine {
    public:
        explicit Gain(float gain) : m_gain(gain) {}
        uint32_t flags() const noexcept override { return ANIRA_ENGINE_FLAG_REALTIME_SAFE; }
        std::unique_ptr<Prepared> prepare(const anira::EnginePrepareInfo& info) override;

    private:
        class Run;
        float m_gain;
    };

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

    private:
        float m_gain;
        std::size_t m_elements;
    };

    std::unique_ptr<anira::Engine::Prepared> Gain::prepare(const anira::EnginePrepareInfo& info) {
        if (info.inputs().empty() || info.outputs().empty()) {
            throw anira::Error(ANIRA_ERROR_CONFIG, "the gain engine needs one input and one output");
        }
        return std::make_unique<Run>(m_gain, info.inputs()[0].num_elements());   // anira owns it
    }

    anira::ModelConfig cfg = anira::ModelConfig::from_file("gain.model.json");
    cfg.add_model_path("com.example.gain", "gain.bin");             // the entry the engine serves
    anira::Pipeline pipe{anira::stage::Inference(cfg).engine("com.example.gain", std::make_shared<Gain>(0.5F))};
    // or: pipe.register_engine("com.example.gain", std::make_shared<Gain>(0.5F));

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
    pipe.register_engine("com.example.gain", shared_gain());

Every instance's ``Pipeline`` holds the object's C engine, so two instances with equal
configurations prepare the model once. Keep the ``Pipeline`` for as long as the instance runs:
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

**Keep the engine out of your header.** The prepared model is the natural place for the
engine's objects: declare the ``Engine::Prepared`` subclass in the ``.cpp`` alone, so the
header that a plugin includes names no engine type, exactly like anira's adapters, which are
file-local classes over the engine:

.. code-block:: cpp

    // MyOnnxEngine.h: no engine include here
    #include <anira/anira.hpp>

    class MyOnnxEngine final : public anira::Engine {
    public:
        std::unique_ptr<Prepared> prepare(const anira::EnginePrepareInfo& info) override;   // in the .cpp
    };

.. code-block:: cpp

    // MyOnnxEngine.cpp: the only place the engine header is included
    #include "MyOnnxEngine.h"
    #include <onnxruntime_cxx_api.h>

    namespace {

    class Session final : public anira::Engine::Prepared {   // owns the Ort:: objects
    public:
        explicit Session(const anira::EnginePrepareInfo& info) {
            // one Ort::Session per instance below info.instances(), the model from
            // info.model_path(info.row()) or info.model_bytes(info.row()), the graph's names
            // matched against info.input_names() / output_names()
        }
        anira_status process(anira::EngineContext& ctx) noexcept override {
            // Ort::Value over ctx.inputs()[i] and ctx.outputs()[i] of THIS call, on the
            // session of ctx.instance()
            return ANIRA_OK;
        }
    private:
        Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "my-engine"};
        Ort::SessionOptions m_options;
        std::vector<std::unique_ptr<Ort::Session>> m_sessions;   // one per instance
    };

    }  // namespace

    std::unique_ptr<anira::Engine::Prepared> MyOnnxEngine::prepare(const anira::EnginePrepareInfo& info) {
        return std::make_unique<Session>(info);
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
    handed and stages its buffers.
