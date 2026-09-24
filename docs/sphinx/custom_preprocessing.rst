Custom Pre/Post Processing
===========================

A model whose host data is not its model tensor as it stands gets a **stage**: the one pre-
and post-processing object of a pipeline, four phase callbacks around the engine call. Section
2 of the :doc:`usage` guide is the reference this page builds on: the phases and their
threads, the context accessors, the rings, the hop rule, the failure rules and the real-time
promise. In C a stage is an ``anira_stage_desc`` of ``anira/abi/stage.h`` handed to
``anira_pipeline_add_stage``; in C++ it is a subclass of :cpp:class:`anira::Stage`, the
registration, whose ``prepare`` returns the :cpp:class:`anira::Stage::Prepared` that runs the
phases for one handler, handed to an :cpp:class:`anira::Pipeline` as
:cpp:class:`anira::stage::Custom`. Both run on the 3.x
handler of ``anira/abi/handler.h`` (:doc:`usage` section 3.2). This page walks through the
cases a stage is for, in the order they come up.

.. note::
    Coming from anira 2.x? The 2.x :cpp:class:`anira::PrePostProcessor` stays with the 2.x
    :cpp:class:`anira::InferenceHandler` in this pre-release, unchanged, and its API reference
    still describes it; :ref:`migration-runtime` maps each of its virtuals and helpers onto the
    stage.

When you need a stage
---------------------

Without a stage anira runs its default bodies: ahead of every inference it pops one hop per
channel of every Streamed input from its ring into the model tensor (with the ring's history
ahead of it where the window is longer than the hop), behind it it pushes one hop per channel
of every Streamed output from the model tensor into its ring; a Static input is the handler's
stored value (``anira_handler_set_static_input``), a Static output is captured into the store
(``anira_handler_get_static_output``), and declared state (``ANIRA_ROLE_STATE``, :doc:`usage`
section 3.4) is fed and captured by anira itself. That covers a time-domain model whose
tensors are ``float32`` and whose channels are the model's channels. A stage is for what is
left:

- a conversion between the host's element type and the model's: an ``int16`` stream into a
  ``float32`` model, the ring dtype of the contract plus a ``pre_process`` that converts
  (:doc:`usage` section 2 has this one in full, in C and in C++);
- a transform between the host end and the model end: a spectrogram model, the FFT on the way
  in and its inverse on the way out (below);
- a layout the default fill does not produce: several overlapping windows per inference as
  one batch (``anira_ring_pop_windows``);
- a value normalised on the way in and denormalised on the way out, one factor shared by
  ``pre_process`` and ``post_process``, which is why the two live in one object (below);
- work between the model's tensors and the engine call, on the inference thread
  (``before_inference`` / ``after_inference``): a constraint on a fed state, a statistic of
  the outputs, or the heavy half of a transform (below).

What is no longer a stage's job: chunking a stream (the default body does it, and a stage that
fills ``pre_process`` calls it for the slots it does not handle), feeding a recurrent state
back (declare the pair), and a custom engine (part of the inference stage,
:doc:`custom_backends`).

The rules in short
------------------

- One stage per pipeline. A filled phase slot owns that phase for every slot with a host end
  and calls ``anira_stage_default_pre_process`` / ``anira_stage_default_post_process``
  (``Stage::pre_process`` / ``Stage::post_process`` in C++, the base class) for what it does
  not handle itself; a ``NULL`` slot (a phase left out of ``phases()``) means the default runs
  for ``pre_process`` and ``post_process``, and that the stage takes no part in a hook, which
  has no default.
- ``pre_process``, ``post_process`` and ``reset`` run on the thread that drives the Hard entries;
  under a Hard contract they need ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST`` in the flags, and the body must
  keep the promise: no allocation, no lock, no system call, only the ``[callback-safe]``
  entries of anira. ``before_inference`` and ``after_inference`` run on an inference thread;
  ``ANIRA_STAGE_FLAG_REALTIME_HOOKS`` is a promise, not a requirement, in this pre-release.
- After ``pre_process`` every Streamed input ring has given up exactly one hop per channel,
  after ``post_process`` every Streamed output ring has gained one: anira checks and records
  a deviation as ``ANIRA_ERROR_CONFIG``.
- Nothing converts: a ring accessor called with a dtype other than the ring's moves nothing
  and returns ``0``; a ring dtype that differs from the spec's is legal exactly when the stage
  fills the phase that moves that ring.
- Ask per slot in every callback and keep nothing: the tensor an accessor fills describes the
  memory the tensor has right now, which an engine may swap between inferences.
- One registration, initialised once, one prepared object per handler. The descriptor's
  slots, from the innermost level out: ``pre_process``, ``post_process``,
  ``before_inference``, ``after_inference``, ``reset``, ``prepare``, ``unprepare``, ``init``,
  ``release``. ``init`` runs once per ``anira_pipeline_add_stage``, at the first
  ``anira_handler_prepare`` of a handler of that pipeline, before that handler's prepare, with
  the facts of the core (``anira_init_info`` of ``anira/abi/lifecycle.h``,
  :cpp:class:`anira::InitInfo`: the log level, the size of the inference thread pool, the
  handler's context); what a stage keeps for its whole registration (a lookup table, a thread
  of its own) is built there, behind ``user_data`` in C, as a member of the object in C++. A
  refused init fails that prepare (``the stage refused init``) and leaves the registration
  uninitialised, so the next prepare calls init again. ``prepare`` receives the record
  (``anira_prepare_info`` of the same header, shared with the engine descriptor,
  :cpp:class:`anira::PrepareInfo`: the handler, the plan report, ``num_entries``, a template
  of the model end of every slot, the canonical names and the ``flags`` of the prepare,
  ``ANIRA_PREPARE_EXCLUSIVE`` when the handler's inferences run one at a time and in order,
  informational for a stage) and hands back what the phases of that handler run on: the
  ``prepared`` pointer in C, the :cpp:class:`anira::Stage::Prepared` in C++. What a stage
  keeps per handler, its per-entry scratch above all, lives there, never in ``user_data`` or
  in the registration, which every handler of the pipeline shares. anira calls ``unprepare``
  (deletes the ``Prepared``) once per successful prepare, at the next prepare of that handler
  and at its destroy, and ``release`` once, with the last carrier, whether init ever ran or
  not.
- ``reset`` runs for the first chunk of a new stream (after prepare, after
  ``anira_handler_reset``), before that chunk's ``pre_process`` and on its thread, with the
  chunk's context in ``ANIRA_PHASE_RESET``, where the accessors answer the role only: what the
  prepared object keeps between chunks (a resampler, an overlap buffer, a filter's history)
  starts over there.
- Per-chunk data lives in a scratch of ``num_entries`` entries (the record's count, what
  ``anira_handler_num_entries(handler)`` answers), sized in ``prepare``, indexed by
  ``ctx->entry`` (``ctx.entry()``).
- A failure is a returned status, never a throw: the status is latched in
  ``anira_handler_rt_error`` and the chunk delivers zeros.

A normalising stage
-------------------

The smallest useful stage: ``pre_process`` lets the default pop the hop and then scales the
model input, ``post_process`` scales the model output back and lets the default push it. One
factor, two phases, one prepared object per handler. The registration names both phases in
``phases()``, so the descriptor's ``pre_process`` and ``post_process`` slots are filled and
the two hooks stay ``NULL``, and its ``prepare`` returns the object the phases run on; the
factor is copied into it, since nothing here is per entry:

.. code-block:: cpp

    #include <memory>
    #include <anira/anira.hpp>

    class Normalise : public anira::Stage {
    public:
        explicit Normalise(float gain) : m_gain(gain) {}

        uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
        uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }

        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& /*info*/) override {
            return std::make_unique<Scaler>(m_gain);   // one per handler; anira deletes it at unprepare
        }

    private:
        class Scaler : public anira::Stage::Prepared {
        public:
            explicit Scaler(float gain) : m_gain(gain) {}

            anira_status pre_process(anira::StageContext& ctx) noexcept override {
                const anira_status st = Prepared::pre_process(ctx);   // the default fill, every Streamed input
                if (st != ANIRA_OK) { return st; }
                return scale(ctx, 0, m_gain, true);
            }
            anira_status post_process(anira::StageContext& ctx) noexcept override {
                const anira_status st = scale(ctx, 0, 1.0F / m_gain, false);
                if (st != ANIRA_OK) { return st; }
                return Prepared::post_process(ctx);                    // the default push, every Streamed output
            }

        private:
            static anira_status scale(anira::StageContext& ctx, uint32_t slot, float factor,
                                      bool input) noexcept {
                anira::Tensor tensor{};
                const anira_status st = input ? ctx.input_tensor(slot, tensor) : ctx.output_tensor(slot, tensor);
                if (st != ANIRA_OK) { return st; }
                float* samples = tensor.data_f32();
                if (samples == nullptr) { return ANIRA_ERROR_CONFIG; }    // not a float32 host tensor
                for (size_t n = 0; n < tensor.num_elements(); ++n) { samples[n] *= factor; }
                return ANIRA_OK;
            }

            float m_gain;
        };

        float m_gain;
    };

The model tensor an accessor fills is the whole tensor in the spec's shape, channel ``c`` at
``c`` times the elements per channel (packed row-major); ``num_elements()`` is the product of
the extents. Written over the model input after the default fill, the scaling covers the
history the default put ahead of the hop as well.

A spectrogram model
-------------------

A model that reads a spectrum and writes one. The window of the input spec is longer than the
hop (``window(2048, 2048, 1536)``: 512 fresh samples per inference and 1536 of history), the
model tensor is the spectrum, so the default fill cannot be used: ``pre_process`` assembles the
window itself, the history through ``peek_past_block`` and the hop through ``pop_block``, and
transforms it into the model tensor; ``post_process`` transforms the model output back and
pushes the hop. The transform runs on the driving thread, as it did in a 2.x processor, so it
has to be real-time itself: preallocated plans, no allocation in the call. The prepared object
of each handler keeps its windows in a per-entry scratch sized by the record's ``num_entries``
in ``prepare``, and clears them at ``reset``, the first chunk of a new stream:

.. code-block:: cpp

    #include <algorithm>
    #include <memory>
    #include <span>
    #include <vector>
    #include <anira/anira.hpp>

    // Your transform of choice, over preallocated plans: no allocation in the call.
    void forward_transform(const float* window, size_t window_size, float* spectrum) noexcept;
    void inverse_transform(const float* spectrum, size_t window_size, float* samples) noexcept;

    class Spectral : public anira::Stage {
    public:
        uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
        uint32_t flags() const noexcept override { return ANIRA_STAGE_FLAG_REALTIME_PRE_POST; }
        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& info) override;

    private:
        class Frames;
    };

    class Spectral::Frames : public anira::Stage::Prepared {
    public:
        explicit Frames(uint32_t entries) : m_windows(static_cast<size_t>(entries) * k_window) {}

        void reset(anira::StageContext& /*ctx*/) noexcept override {
            std::ranges::fill(m_windows, 0.0F);                    // a new stream: no history
        }

        anira_status pre_process(anira::StageContext& ctx) noexcept override {
            anira::RingView ring;                                    // the audio input: Streamed, float32
            if (const anira_status st = ctx.input_ring(0, ring); st != ANIRA_OK) { return st; }
            anira::Tensor spectrum{};
            if (const anira_status st = ctx.input_tensor(0, spectrum); st != ANIRA_OK) { return st; }
            float* bins = spectrum.data_f32();
            if (bins == nullptr) { return ANIRA_ERROR_CONFIG; }
            const std::span<float> window = scratch(ctx.entry());
            // the window: the history first, then the hop, popped exactly once
            ring.peek_past_block(0, window.subspan(0, k_window - k_hop));
            if (ring.pop_block(0, window.subspan(k_window - k_hop, k_hop)) != k_hop) {
                return ANIRA_ERROR_INTERNAL;
            }
            forward_transform(window.data(), k_window, bins);
            return ANIRA_OK;
        }

        anira_status post_process(anira::StageContext& ctx) noexcept override {
            anira::RingView ring;
            if (const anira_status st = ctx.output_ring(0, ring); st != ANIRA_OK) { return st; }
            anira::Tensor spectrum{};
            if (const anira_status st = ctx.output_tensor(0, spectrum); st != ANIRA_OK) { return st; }
            const float* bins = spectrum.data_f32();
            if (bins == nullptr) { return ANIRA_ERROR_CONFIG; }
            const std::span<float> window = scratch(ctx.entry());
            inverse_transform(bins, k_window, window.data());
            // the stream advances by one hop: push what this frame contributes to it
            if (ring.push_block(0, window.subspan(k_window - k_hop, k_hop)) != k_hop) {
                return ANIRA_ERROR_INTERNAL;
            }
            return ANIRA_OK;
        }

    private:
        std::span<float> scratch(uint32_t entry) noexcept {
            return {m_windows.data() + static_cast<size_t>(entry) * k_window, k_window};
        }

        static constexpr size_t k_window = 2048;
        static constexpr size_t k_hop = 512;
        std::vector<float> m_windows;
    };

    std::unique_ptr<anira::Stage::Prepared> Spectral::prepare(const anira::PrepareInfo& info) {
        return std::make_unique<Frames>(info.num_entries());   // this handler's windows
    }

The example is mono (channel ``0``); a multi-channel stage loops over
``ring.num_channels()`` and lays the channels out as the model tensor expects. What an
overlap-add keeps from one frame to the next is the prepared object's own business, in a
scratch of its own that ``reset`` clears; the shape of the stage is the point here: pop the
hop once, push the hop once, transform in between.

When the transform is heavy
---------------------------

An FFT of a few thousand points fits the audio callback; a transform that does not (a large
network's front end, a resampler with a long filter) moves to the inference thread. The split
uses the entry: ``pre_process`` only pops the window into the per-entry scratch, which is
cheap and real-time; ``before_inference`` transforms the scratch into the model tensor on the
inference thread; ``after_inference`` transforms the model output back into the scratch; and
``post_process`` pushes the hop. The two hooks may then use a library that allocates, since
the descriptor promises nothing for them (``flags`` carries ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST``
alone). The same stage in C, over the descriptor: ``prepare`` allocates this handler's windows
and hands them back through ``out_prepared``, every phase finds them in ``prepared``, and
``unprepare`` frees them, once per successful prepare; ``user_data`` stays ``NULL``, since
nothing is shared by the handlers of the pipeline:

.. code-block:: c

    #include <stdlib.h>
    #include <string.h>
    #include <anira/abi/config.h>
    #include <anira/abi/handler.h>

    #define WINDOW ((size_t)2048)
    #define HOP ((size_t)512)

    /* Your transform of choice. */
    void forward_transform(const float* window, size_t window_size, float* spectrum);
    void inverse_transform(const float* spectrum, size_t window_size, float* samples);

    typedef struct spectral { float* windows; uint32_t entries; } spectral;   /* one per handler */

    static anira_status ANIRA_CALL spectral_prepare(const anira_prepare_info* info,
                                                    void* user_data,
                                                    void** out_prepared) {
        spectral* s = (spectral*)calloc(1, sizeof(spectral));
        (void)user_data;                                  /* the registration: nothing shared */
        if (s == NULL) { return ANIRA_ERROR_OUT_OF_MEMORY; }
        s->entries = info->num_entries;
        s->windows = (float*)calloc((size_t)s->entries * WINDOW, sizeof(float));
        if (s->windows == NULL) { free(s); return ANIRA_ERROR_OUT_OF_MEMORY; }
        *out_prepared = s;                                /* what every phase of this handler gets */
        return ANIRA_OK;
    }

    /* the first chunk of a new stream, on the thread of pre_process: no history */
    static void ANIRA_CALL spectral_reset(const anira_stage_ctx* ctx,
                                          void* prepared,
                                          void* user_data) ANIRA_NONBLOCKING {
        spectral* s = (spectral*)prepared;
        (void)ctx;
        (void)user_data;
        memset(s->windows, 0, (size_t)s->entries * WINDOW * sizeof(float));
    }

    /* driving thread, real-time: the pop alone */
    static anira_status ANIRA_CALL spectral_pre_process(const anira_stage_ctx* ctx,
                                                        void* prepared,
                                                        void* user_data) ANIRA_NONBLOCKING {
        float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_ring* ring = NULL;
        anira_status st = anira_stage_input_ring(ctx, 0, &ring);   /* slot 0 is Streamed: the model config says so */
        if (st != ANIRA_OK) { return st; }
        if (anira_ring_peek_past_block(ring, 0, window, ANIRA_DTYPE_F32, WINDOW - HOP) != WINDOW - HOP) {
            return ANIRA_ERROR_CONFIG;                      /* not float32 */
        }
        if (anira_ring_pop_block(ring, 0, window + (WINDOW - HOP), ANIRA_DTYPE_F32, HOP) != HOP) {
            return ANIRA_ERROR_INTERNAL;
        }
        return ANIRA_OK;
    }

    /* inference thread: the transform into the model tensor */
    static anira_status ANIRA_CALL spectral_before_inference(const anira_stage_ctx* ctx,
                                                             void* prepared,
                                                             void* user_data) {
        const float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_tensor spectrum;
        float* bins;
        anira_status st = anira_stage_input_tensor(ctx, 0, &spectrum);
        if (st != ANIRA_OK) { return st; }
        bins = anira_tensor_data_f32(&spectrum);
        if (bins == NULL) { return ANIRA_ERROR_CONFIG; }
        forward_transform(window, WINDOW, bins);
        return ANIRA_OK;
    }

    /* inference thread: the inverse transform back into the scratch of the same entry */
    static anira_status ANIRA_CALL spectral_after_inference(const anira_stage_ctx* ctx,
                                                            void* prepared,
                                                            void* user_data) {
        float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_tensor spectrum;
        const float* bins;
        anira_status st = anira_stage_output_tensor(ctx, 0, &spectrum);
        if (st != ANIRA_OK) { return st; }
        bins = anira_tensor_data_f32(&spectrum);
        if (bins == NULL) { return ANIRA_ERROR_CONFIG; }
        inverse_transform(bins, WINDOW, window);
        return ANIRA_OK;
    }

    /* driving thread, real-time: the push alone */
    static anira_status ANIRA_CALL spectral_post_process(const anira_stage_ctx* ctx,
                                                         void* prepared,
                                                         void* user_data) ANIRA_NONBLOCKING {
        const float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_ring* ring = NULL;
        anira_status st = anira_stage_output_ring(ctx, 0, &ring);
        if (st != ANIRA_OK) { return st; }
        if (anira_ring_push_block(ring, 0, window + (WINDOW - HOP), ANIRA_DTYPE_F32, HOP) != HOP) {
            return ANIRA_ERROR_INTERNAL;
        }
        return ANIRA_OK;
    }

    /* once per successful prepare: the next prepare of the handler, or its destroy */
    static void ANIRA_CALL spectral_unprepare(void* prepared, void* user_data) {
        spectral* s = (spectral*)prepared;
        (void)user_data;
        free(s->windows);
        free(s);
    }

and its descriptor, filled once at setup (``init`` and ``release`` stay ``NULL``: nothing is
shared by the handlers of the pipeline):

.. code-block:: c

    anira_stage_desc desc = ANIRA_STAGE_DESC_INIT;
    desc.flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST;   /* the hooks promise nothing */
    desc.pre_process = spectral_pre_process;
    desc.before_inference = spectral_before_inference;
    desc.after_inference = spectral_after_inference;
    desc.post_process = spectral_post_process;
    desc.reset = spectral_reset;
    desc.prepare = spectral_prepare;
    desc.unprepare = spectral_unprepare;
    anira_pipeline_add_stage(pipe, &desc, &err);

The entry is what makes the split correct: chunks of one handler may be in flight on several
inference threads at once, one entry holds one chunk at a time, and the four phases of one
chunk see the same ``entry``. The entry count is fixed by ``anira_handler_prepare`` for the
life of the session and may change with the next prepare, which is why the scratch is sized in
``prepare``, per handler, from the record's ``num_entries``, and handed back as the prepared
pointer rather than kept in the registration. A ``pre_process`` that pops nothing but the hop,
as here, moves exactly what the hop rule expects.

Static and State tensors in a stage
-----------------------------------

A stage sees every tensor of the model's two lists at its slot, whatever its role, and
``anira_stage_input_role`` / ``anira_stage_output_role`` (``ctx.input_role(slot, role)``) say
which it is. A **Static** input is materialised from the handler's store into the model tensor when
the chunk is formed, so ``pre_process`` and ``before_inference`` find the value there and may
read or alter it for that chunk (the store is untouched: only the host and the capture write
it); a Static output is captured behind ``post_process``, so what ``after_inference`` or
``post_process`` leaves in the model tensor is what ``anira_handler_get_static_output`` will
return. A **State** input is fed after ``pre_process`` and before ``before_inference``, and a
State output is captured after ``after_inference`` and before ``post_process``: the two hooks
see the fed and the produced state and may alter either (a constraint, a decay, a splice),
while ``pre_process`` and ``post_process`` have no host end of a State slot to hand out.
Neither role has a ring. Asking for what a slot does not have is the stage's bug, not an
answer: the accessor answers ``ANIRA_ERROR_INVALID_STATE`` and latches it into
``anira_handler_rt_error``, naming the entry, the slot and the phase. A stage written for one
model knows its slots; a stage that handles every slot of a model it does not know asks the
role first and takes the ring of a Streamed slot only.

.. note::
    Feeding a recurrent state back is no longer a stage's job: declare the pair with the role
    ``state`` and ``state_source`` (:doc:`usage` section 3.4), and anira runs the feedback,
    re-initialises the state on reset and keeps it across a plan switch. A model with such a
    pair runs Stateful, one inference at a time and in order, whatever its ``state`` says.

Integration
-----------

In C, the descriptor goes into the pipeline beside the inference stage, before or after it,
and the handler is created from the pipeline as in section 3.2 of the :doc:`usage` guide:

.. code-block:: c

    anira_pipeline* pipe = NULL;
    anira_pipeline_create(&pipe, &err);
    anira_pipeline_add_inference(pipe, variants, 1, NULL, 0, &err);   /* NULL: the default candidate set */
    anira_pipeline_add_stage(pipe, &desc, &err);                       /* at most one */
    anira_handler* h = NULL;
    anira_handler_create(context, pipe, &h, &err);                     /* copies the pipeline */
    anira_pipeline_destroy(pipe);                                      /* the handler keeps its copy */

In C++, :cpp:class:`anira::Pipeline` takes the stage as :cpp:class:`anira::stage::Custom` over
a ``std::shared_ptr``; the pipeline's carrier shares the pointer, so the registration lives at
least until the last pipeline or handler that carries it is destroyed, whatever you do with
your own pointer, the first ``anira_handler_prepare`` of a handler of the pipeline calls its
``Stage::init()`` (the base class does nothing; an override may throw, which fails that prepare
and leaves init to the next one), each handler's ``anira_handler_prepare`` asks it for a
``Prepared`` of its own (returned as a ``std::unique_ptr``, since anira is its one owner from
then on and deletes it at unprepare), and ``Stage::release()`` runs once, when the last carrier
dies, whether init ever ran or not:

.. code-block:: cpp

    anira::Pipeline pipe{anira::stage::Inference(cfg),   // the default candidate set
                         anira::stage::Custom(std::make_shared<Spectral>())};
    anira_handler* h = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (ANIRA_FAILED(anira_handler_create(context.native(), pipe.native(), &h, &err))) {
        return fail(&err);
    }

``anira_handler_prepare`` then validates the stage with the rest: under a Hard contract a
filled ``pre_process``, ``post_process`` or ``reset`` without ``ANIRA_STAGE_FLAG_REALTIME_PRE_POST`` is
``ANIRA_ERROR_CONFIG`` naming the flag, a ring dtype that differs from its spec's is accepted
only where the stage fills the phase that moves that ring, and the stage runs last: its
``init`` when this is the first prepare of a handler of the pipeline (a status it returns, or
an exception a C++ ``init`` throws, fails the prepare with ``the stage refused init``, and the
next prepare calls init again), then its own ``prepare``, with the record; a status a C
``prepare`` returns, an exception a C++ ``prepare`` throws or a null ``Prepared`` it returns,
fails the prepare (``the stage refused prepare``), and no unprepare follows. The
:cpp:class:`anira::InferenceHandler` class of ``anira/anira.hpp`` arrives with the runtime
cut-over; until then a stage is driven through the C entries of the handler.
