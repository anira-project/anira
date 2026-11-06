.. _processors:

Custom Pre/Post Processing
==========================

A **processor** is a pipeline's pre- and post-processing: the code that takes the host's data to
the model's tensors ahead of an inference and back behind it. Without one anira runs its default
bodies, the windowing of a Streamed model, which is all most models need. In C++ a processor is a
subclass of :cpp:class:`anira::Processor`; in C an ``anira_processor_desc`` of
``anira/abi/processor.h``.

.. code-block:: cpp

    class Normalise : public anira::Processor {
    public:
        explicit Normalise(float gain) : m_gain(gain) {}
        uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
        uint32_t flags() const noexcept override { return ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST; }
        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo&) override {
            return std::make_unique<Scaler>(m_gain);              // one per handler
        }

    private:
        class Scaler : public anira::Processor::Prepared {
        public:
            explicit Scaler(float gain) : m_gain(gain) {}
            anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
                const anira_status st = Prepared::pre_process(ctx);   // the default fill
                return st != ANIRA_OK ? st : scale(ctx, true, m_gain);
            }
            anira_status post_process(anira::ProcessorContext& ctx) noexcept override {
                const anira_status st = scale(ctx, false, 1.0F / m_gain);
                return st != ANIRA_OK ? st : Prepared::post_process(ctx);   // the default push
            }

        private:
            static anira_status scale(anira::ProcessorContext& ctx, bool input, float factor) noexcept {
                anira::Tensor tensor{};
                const anira_status st = input ? ctx.input_tensor(0, tensor) : ctx.output_tensor(0, tensor);
                if (st != ANIRA_OK) { return st; }
                float* samples = tensor.data_f32();
                if (samples == nullptr) { return ANIRA_ERROR_CONFIG; }   // not a float32 host tensor
                for (size_t n = 0; n < tensor.num_elements(); ++n) { samples[n] *= factor; }
                return ANIRA_OK;
            }
            float m_gain;
        };
        float m_gain;
    };

    pipeline.add_inference(model).add_processor(std::make_shared<Normalise>(2.0F));

.. toctree::
   :maxdepth: 1

   custom_preprocessing/examples

When you need one
-----------------

The default bodies pop one hop per channel of every Streamed input from its ring into the model
tensor (with the ring's history ahead of it where the window is longer than the hop) and push the
last hop of every Streamed output's window back; anira materialises the Static inputs, captures the
Static outputs, and feeds and captures declared state itself. A processor is for what is left:

- a conversion between the host's element type and the model's (:ref:`usage-dtypes`);
- a transform between the host end and the model end: a spectrogram model's FFT and its inverse;
- a layout the default fill does not make, such as overlapping windows as one batch
  (``anira_ring_pop_windows``);
- a value normalised on the way in and denormalised on the way out, one object for both;
- work beside the engine call on the inference thread (``before_inference`` / ``after_inference``):
  a constraint on a fed state, a statistic of the outputs, the heavy half of a transform.

Not a processor's job: chunking a stream (the default does it), feeding a recurrent state back
(declare the pair, :ref:`usage-state`), and an engine (:doc:`custom_engines`).

.. _processor-phases:

The phases of a chunk
---------------------

A chunk is one inference's worth of data, one hop of every Streamed tensor. In order:

1. materialise the Static inputs into their model tensors;
2. ``pre_process``, on the thread that produces the host end (the one that drives the Hard entries;
   under Async an inference thread): every input slot's host end to its model tensor. One call forms
   exactly one chunk; a host block of several hops forms several;
3. on an inference thread: bind the State pairs, ``before_inference``, the edge into the engine's
   domain, the engine, the edge back, ``after_inference``, flip the State pairs;
4. ``post_process``, on the thread that consumes the host end: every output slot's model tensor to
   its ring; then capture the Static outputs.

Whoever writes last wins: a processor that overwrites a Static input's model tensor changes that
chunk and nothing else (only the host and the capture write the store). The edge is anira's: a
processor never crosses a domain, and every phase works at the host end in the domain the contract
declares for that tensor (host memory in this pre-release). Every model tensor carries its spec's
dtype, a ring its ring dtype.

**Filled and NULL phases.** A phase you leave out of ``phases()`` (a ``NULL`` slot in C) runs the
default: ``anira_processor_default_pre_process`` / ``_post_process``, in C++ the base class's
``Processor::Prepared::pre_process`` / ``post_process``. A filled phase **owns that phase for every
slot with a host end** and calls the default for what it does not handle ("call super"). The two
hooks have no default. In C the reset slot is filled or ``NULL`` on its own; in C++ ``reset`` comes
with ``pre_process`` or ``post_process`` unnamed, and a processor of the hooks alone whose
``Prepared`` overrides ``reset`` names ``Processor::k_reset`` in ``phases()``, else its reset is
never called. The defaults are real-time, never touch a tensor of another role than Streamed, and
return ``ANIRA_ERROR_CONFIG`` for a slot whose ring dtype is not its tensor's.

.. _processor-accessors:

What a callback sees
--------------------

Every phase receives an ``anira_processor_ctx`` (C++ :cpp:class:`anira::ProcessorContext`), 64 bytes
on anira's stack for the call: ``phase``, the plan's ``engine`` and ``provider`` (whole through
``anira_processor_engine`` / ``_provider``, ``ctx.engine()`` / ``ctx.provider()``), ``variant``, the
two lists' lengths, ``ticket`` (``ANIRA_TICKET_INVALID`` under Hard), ``entry`` and an opaque
``frame``. A callback asks **per slot** (:ref:`usage-slots`) through six accessors, ``[callback-safe]``
and ``ANIRA_NONBLOCKING``:

- ``input_role`` / ``output_role``: the slot's ``anira_role``, the same in every phase;
- ``input_ring`` / ``output_ring``: a Streamed slot's ring, in ``pre_process`` (inputs) and
  ``post_process`` (outputs);
- ``input_tensor`` / ``output_tensor``: a descriptor of the slot's *model end*, built over the
  memory it has right now (an engine may swap it between inferences, so ask in every callback and
  keep nothing). Inputs answer in ``pre_process`` (Streamed and Static) and ``before_inference``
  (every role, the State inputs fed); outputs in ``after_inference`` (every role, the State outputs
  not yet captured) and ``post_process`` (Streamed and Static). The tensor is the whole one in the
  spec's shape, packed row-major: channel ``c`` of ``n`` elements per channel starts at element
  ``c * n``, and the default fill puts the ring's history at the head of each channel and the hop
  behind it, so a scaling written over the model input after the default fill covers the history
  too.

``anira_processor_engine`` / ``_provider`` hand back the id beside the value, ``NULL`` beside every
value but CUSTOM (the id out-parameter may be ``NULL``). Each accessor returns a status and fills its
out-parameter, which it resets on anything but ``ANIRA_OK`` (``ANIRA_ROLE_FORCE32``, ``NULL``, an
all-zero record); a slot always has a role.
Asking for what a slot does not have in this phase (a ring of a Static or State tensor, an input
ring outside ``pre_process``, a State tensor's host end) is a **bug in the processor**:
``ANIRA_ERROR_INVALID_STATE``, latched into ``rt_error()`` with one record per kind naming the entry,
the slot and the phase; a slot out of range or a NULL out-parameter is
``ANIRA_ERROR_INVALID_ARGUMENT``, latched the same way.

**The entry.** ``ctx->entry`` (``ctx.entry()``) is the chunk's position in the handler's chunk pool,
``0 .. anira_handler_num_entries(h) - 1``: the same in all four phases of a chunk, distinct across
the chunks in flight, reused once the chunk completed. It indexes a **per-chunk scratch** sized in
``prepare`` (``num_entries`` is on the record), so that what ``pre_process`` pops is found again by
``before_inference`` on another thread, without an allocation.

**The rings** stay inside anira; a processor moves elements through the ``anira_ring`` accessors
(``RingView`` in C++, typed: the element type names the dtype), ``[callback-safe]`` and
``ANIRA_NONBLOCKING``: ``dtype``, ``num_channels``, ``available``, ``available_past`` (the consumed
elements kept as history), ``pop_block``, ``peek_past_block`` (the history, oldest first),
``push_block``, ``push_fill``, ``discard`` and ``pop_windows`` (``num_batches`` windows of
``num_old`` history and ``num_new`` fresh elements each; a destination span too short for them is
refused, 0 and nothing popped). Every data accessor states its dtype: another one than the ring's
moves nothing, returns 0 and records ``ANIRA_ERROR_CONFIG`` (a channel out of range:
``ANIRA_ERROR_INVALID_ARGUMENT``; a NULL ring: 0). A ring's dtype may differ from its spec's exactly
when the processor fills the phase that moves that ring; its channel count and domain never do.

**The hop rule.** After ``pre_process`` every Streamed input ring must have given up exactly its hop
per channel, after ``post_process`` every Streamed output ring gained exactly its hop. A shortfall
is repaired (missing elements discarded from an input ring, an output topped up with zeros), an
excess cannot be undone; either way ``rt_error()`` reads ``ANIRA_ERROR_CONFIG`` and one record names
the phase, the tensor and both counts.

**Failure.** A phase returns ``ANIRA_OK``, or any other status to fail its chunk (a status, never a
throw: the C++ phases are ``noexcept``, so a throw terminates): the status is
latched into ``rt_error()`` with a record naming the phase, and the chunk delivers zeros at its
stream position. A failed ``pre_process`` is not submitted; a failed hook skips the rest and zeroes
the model outputs (the State outputs are not captured, so the state keeps its last good value); after
a failed ``post_process`` anira tops every Streamed output up to the hop with zeros.

.. _processor-promise:

The real-time promise
---------------------

Whether a body is real-time is the processor's own promise, its ``flags``:
``ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST`` says ``pre_process``, ``post_process`` and ``reset``
allocate nothing, lock nothing and block on nothing; ``ANIRA_PROCESSOR_FLAG_REALTIME_HOOKS`` says
the same of the two hooks; 0 promises nothing. A real-time body calls only the ``[callback-safe]``
entries of anira. Under a Hard contract those three phases run on the driving thread, so a filled
one without ``REALTIME_PRE_POST`` is ``ANIRA_ERROR_CONFIG`` at prepare, naming the flag; a processor
of ``before_inference`` and ``after_inference`` alone (in C++ without ``k_reset``) fills none of the
three and needs no promise there. Under Async no bit is required; ``REALTIME_HOOKS`` is required by
no contract of this pre-release. A bit the header does not define is
``ANIRA_ERROR_INVALID_ARGUMENT`` at ``anira_pipeline_add_processor``. anira's own side (the
accessors, the ring calls, the defaults) is ``ANIRA_NONBLOCKING``, so a body made of them can be
checked by clang: declare the callback ``ANIRA_NONBLOCKING`` yourself (``anira_processor_fn``
carries no attribute).

.. _processor-lifecycle:

The lifecycle
-------------

One **registration** (``add_processor``) serves every handler created from the pipeline, which share
it; each handler is prepared on its own. The engine descriptor has the same lifecycle with one level
more (:doc:`custom_engines`).

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Slot
     - When, and what for
   * - ``init(info)``
     - Once per registration, at the first prepare of a handler of the pipeline, on the caller of
       that prepare and before it, with an ``anira_init_info`` (``anira/abi/lifecycle.h``,
       :cpp:class:`anira::InitInfo`: the log level, the pool's size, 0 when the context brought its
       own threads, and the handler's context, whose capabilities it may query). Build what the
       processor keeps for its whole registration (behind ``user_data`` in C, which the caller may
       also fill before ``anira_pipeline_add_processor``; as members in C++). It may allocate and
       log, never call an entry that takes the lifecycle lock. A refusal
       fails that prepare (``the processor refused init``) and the next prepare calls ``init`` again.
   * - ``prepare(info)``
     - Once per prepare of each handler, on its caller, after the plan report is built, with an
       ``anira_prepare_info`` (:cpp:class:`anira::PrepareInfo`: ``handler()``, ``report()``,
       ``num_entries()``, a template of every slot's model end (the spec's dtype and shape at the
       pinned window, the host domain, all-zero strides, no memory), the canonical names,
       ``flags()`` / ``exclusive()``: ``ANIRA_PREPARE_EXCLUSIVE`` when the inferences run one at a
       time and in order, informational for a processor). Size the per-entry scratch here and hand
       back this handler's *prepared* object (C: ``prepared``, ``NULL`` legal; C++: the
       ``Processor::Prepared``), which every phase, the reset and the unprepare of this handler
       receive. It may call the ``[callback-safe]`` entries, the handler's
       getters and the Static entries, never ``prepare``, ``destroy`` or a Hard entry. A refusal fails
       the prepare (``the processor refused prepare``), and no unprepare follows. Under any warm-up
       but ``ANIRA_WARMUP_NONE``, unless the model's figure is reused, the pass then runs the hooks on this prepared object (never
       ``pre_process`` or ``post_process``), so a hook must accept an entry ``pre_process`` never
       filled. Under a measured budget ``prepare`` may run twice in one ``anira_handler_prepare``:
       ``prepare``, the pass, then ``unprepare`` and ``prepare`` again with the final record when the
       pass moved the entry count.
   * - ``reset(ctx)``
     - For the first chunk of a new stream (after prepare, after ``anira_handler_reset``), before its
       ``pre_process`` and on its thread, in ``ANIRA_PHASE_RESET`` (the accessors answer the role only):
       a resampler, an overlap buffer, a filter's history starts over. C++ fills the slot with
       ``k_pre_process``, ``k_post_process`` or ``k_reset`` in ``phases()``.
   * - ``unprepare(prepared)``
     - Once per successful prepare, at the next prepare of the handler (after the old session is
       released; a next prepare that fails earlier undoes the previous one too) and at its destroy,
       after the last phase call. C++ deletes the ``Prepared`` there.
   * - ``release()``
     - Once, when the last carrier dies (the pipeline or the last handler, whichever goes last), on
       the thread of that destroy, after every unprepare, whether or not ``init`` ran.

What one handler keeps lives behind its prepared object, never in the registration, which every
handler of the pipeline shares. In C++, ``init`` and ``prepare`` may throw: an ``anira::Error`` fails
the prepare with its status, ``std::bad_alloc`` with ``ANIRA_ERROR_OUT_OF_MEMORY``, anything else
(and a null ``Prepared``) with ``ANIRA_ERROR_INTERNAL``. The registration is held by a
``std::shared_ptr`` (shared by the pipeline and every handler), the prepared object by a
``std::unique_ptr`` (anira's from ``prepare`` on).

**An id** (``anira_processor_desc.id``, ``anira::Processor(std::string id)``), a reverse-URI name,
says what a processor's ``before_inference`` and ``after_inference`` are, for the model's measured
figure: the warm-up pass times the hooks with the engine call, so the handlers of processors with
the same id share one figure and identical plugins report identical latency. Without an id, a
processor that fills a hook measures its own; one without hooks needs none
(:ref:`usage-budget-warmup`). A JavaScript processor's id is ``org.anira.web.processor.<key>``,
``<key>`` being the hash of the name it is defined under, so processors defined under one name share
the figure.

.. code-block:: cpp

    class WindowProcessor : public anira::Processor {
    public:
        WindowProcessor() : anira::Processor("com.example.myplugin.window") {}
        uint32_t phases() const noexcept override { return k_before_inference; }   // no flag needed
        // ...
    };

Static and State slots
----------------------

A processor sees every tensor at its slot, whatever its role; ask the role first when it handles
every slot of a model it does not know. A **Static** input is a per-chunk snapshot materialised
ahead of ``pre_process``, so ``pre_process`` and ``before_inference`` may read or alter it for that
chunk; what ``after_inference`` or ``post_process`` leaves in a Static output's model tensor is what
``get_static_output`` returns. A **State** pair is fed after ``pre_process`` and captured before
``post_process``: the two hooks see the fed and the produced state and may alter either (a
constraint, a decay), while ``pre_process`` and ``post_process`` get ``ANIRA_ERROR_INVALID_STATE``
for it. Neither role has a ring.

In C
----

The same descriptor: up to four phase callbacks, a reset, a prepare, an unprepare, an init and a
release function, one ``user_data``, the ``flags``, the ``id`` and, for the extensions it reads,
``consumed_kinds`` (``"<host>:<kind>"`` strings; the plan report's ``anira_plan_ext`` rows of those
read ``processor``). ``anira_pipeline_add_processor(pipe, &desc, &err)`` copies it (within
``struct_size``, with its strings) into a carrier the pipeline and its handlers share; a second call
is ``ANIRA_ERROR_INVALID_STATE``, and the order relative to ``anira_pipeline_add_inference`` means
nothing. The processor has no name: every record about it says "the processor". The four phases
and the reset run as ``(ctx, prepared, user_data)``, ``prepare`` as ``(info, user_data,
&prepared)``, ``unprepare`` as ``(prepared, user_data)``, ``init`` as ``(info, user_data)``,
``release`` as ``(user_data)``.

The int16 processor of :doc:`custom_preprocessing/examples` in C; it asks the role first, as a
processor that does not know its model does:

.. code-block:: c

    #include <stdlib.h>
    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>   /* includes anira/abi/processor.h */

    #define HOP ((size_t)512)

    typedef struct convert_scratch { int16_t* samples; uint32_t entries; } convert_scratch;

    static anira_status ANIRA_CALL convert_prepare(const anira_prepare_info* info,
                                                   void* user_data,
                                                   void** out_prepared) {
        convert_scratch* scratch = (convert_scratch*)calloc(1, sizeof(convert_scratch));
        (void)user_data;                                  /* the registration: nothing shared here */
        if (scratch == NULL) { return ANIRA_ERROR_OUT_OF_MEMORY; }
        scratch->entries = info->num_entries;             /* one slot per chunk in flight */
        scratch->samples = (int16_t*)calloc((size_t)scratch->entries * HOP, sizeof(int16_t));
        if (scratch->samples == NULL) { free(scratch); return ANIRA_ERROR_OUT_OF_MEMORY; }
        *out_prepared = scratch;                          /* this handler's: every phase gets it back */
        return ANIRA_OK;
    }

    /* Real-time: every anira call below is ANIRA_NONBLOCKING, and so is this body. */
    static anira_status ANIRA_CALL convert_pre_process(const anira_processor_ctx* ctx,
                                                       void* prepared,
                                                       void* user_data) ANIRA_NONBLOCKING {
        int16_t* scratch = ((convert_scratch*)prepared)->samples + (size_t)ctx->entry * HOP;
        (void)user_data;
        anira_tensor model;
        float* samples;
        anira_ring* ring = NULL;
        anira_role role = ANIRA_ROLE_FORCE32;
        anira_status st;
        st = anira_processor_input_role(ctx, 0, &role);          /* what slot 0 is */
        if (st != ANIRA_OK) { return st; }                   /* latched already: the processor's bug */
        if (role != ANIRA_ROLE_STREAMED) { return ANIRA_ERROR_CONFIG; }
        st = anira_processor_input_ring(ctx, 0, &ring);          /* the host's int16 ring */
        if (st != ANIRA_OK) { return st; }
        st = anira_processor_input_tensor(ctx, 0, &model);       /* the float32 model end */
        if (st != ANIRA_OK) { return st; }
        samples = anira_tensor_data_f32(&model);
        if (samples == NULL || anira_ring_dtype(ring) != ANIRA_DTYPE_I16) {
            return ANIRA_ERROR_CONFIG;
        }
        for (uint32_t c = 0; c < anira_ring_num_channels(ring); ++c) {
            if (anira_ring_pop_block(ring, c, scratch, ANIRA_DTYPE_I16, HOP) != HOP) {
                return ANIRA_ERROR_INTERNAL;                /* latched; the chunk delivers zeros */
            }
            for (size_t n = 0; n < HOP; ++n) {              /* channel c starts at c * HOP */
                samples[c * HOP + n] = (float)scratch[n] / 32768.0f;
            }
        }
        return ANIRA_OK;                                    /* the ring gave up exactly one hop */
    }

    /* Once per successful prepare: the next prepare of the handler, or its destroy. */
    static void ANIRA_CALL convert_unprepare(void* prepared, void* user_data) {
        convert_scratch* scratch = (convert_scratch*)prepared;
        (void)user_data;
        free(scratch->samples);
        free(scratch);
    }

At setup, beside ``anira_pipeline_add_inference`` (``user_data``, ``init`` and ``release`` stay
``NULL``: nothing is shared by the handlers, and the scratch dies with each prepare):

.. code-block:: c

    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.flags = ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST;   /* required: pre_process runs on the driving thread */
    desc.pre_process = convert_pre_process;        /* post_process stays NULL: the default push */
    desc.prepare = convert_prepare;                /* per handler: hands the scratch back */
    desc.unprepare = convert_unprepare;            /* once per prepare: frees it */
    anira_pipeline_add_processor(pipe, &desc, &err);  /* once per pipeline */

:doc:`custom_preprocessing/examples` has a C processor split across the hooks.

.. note::
    A 2.x ``PrePostProcessor`` runs as a 3.x pipeline's processor through
    ``anira::v2::LegacyProcessor`` of ``anira/compat/v2.hpp``; :ref:`migration-processor` maps its
    virtuals and helpers.
