Processor examples
==================

Each example is a whole processor; :doc:`../custom_preprocessing` is the reference for what they
call. The normaliser is on that page.

An int16 stream into a float32 model
------------------------------------

The host pushes ``int16`` samples (``contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16)``), the
model reads ``float32``. Without a processor prepare refuses the contract; with this one
``pre_process`` pops the hop as ``int16`` and converts, and ``post_process``, left out of
``phases()``, stays the default push. The scratch is sized per entry in ``prepare``, one per handler.

.. code-block:: cpp

    #include <memory>
    #include <span>
    #include <vector>
    #include <anira/anira.hpp>

    class Int16ToFloat : public anira::Processor {
    public:
        uint32_t phases() const noexcept override { return k_pre_process; }   // post_process: the default
        uint32_t flags() const noexcept override { return ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST; }
        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& info) override;

    private:
        class Converter;
    };

    class Int16ToFloat::Converter : public anira::Processor::Prepared {
    public:
        explicit Converter(uint32_t entries) : m_scratch(static_cast<size_t>(entries) * k_hop) {}

        anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
            anira::Role role = ANIRA_ROLE_FORCE32;                       // what slot 0 is
            if (const anira_status st = ctx.input_role(0, role); st != ANIRA_OK) { return st; }
            if (role != ANIRA_ROLE_STREAMED) { return ANIRA_ERROR_CONFIG; }
            anira::RingView ring;                                        // the host's int16 ring
            if (const anira_status st = ctx.input_ring(0, ring); st != ANIRA_OK) { return st; }
            anira::Tensor model{};                                       // the float32 model end
            if (const anira_status st = ctx.input_tensor(0, model); st != ANIRA_OK) { return st; }
            float* samples = model.data_f32();
            if (samples == nullptr || ring.dtype() != ANIRA_DTYPE_I16) { return ANIRA_ERROR_CONFIG; }
            const std::span<int16_t> scratch(m_scratch.data() + static_cast<size_t>(ctx.entry()) * k_hop, k_hop);
            for (uint32_t c = 0; c < ring.num_channels(); ++c) {
                if (ring.pop_block(c, scratch) != k_hop) { return ANIRA_ERROR_INTERNAL; }
                for (size_t n = 0; n < k_hop; ++n) {
                    samples[c * k_hop + n] = static_cast<float>(scratch[n]) / 32768.0F;
                }
            }
            return ANIRA_OK;
        }

    private:
        static constexpr size_t k_hop = 512;
        std::vector<int16_t> m_scratch;
    };

    std::unique_ptr<anira::Processor::Prepared> Int16ToFloat::prepare(const anira::PrepareInfo& info) {
        return std::make_unique<Converter>(info.num_entries());   // one per handler, its own scratch
    }

A float32 host on a float16 model
---------------------------------

Both rings float32, the model tensors float16 (``to_half`` and ``from_half`` are the host's helpers;
``HOP`` is the model's hop). ``test/abi/test_Dtypes.cpp`` runs this processor and its int8 twin.

.. code-block:: cpp

    class HalfPrepared final : public anira::Processor::Prepared {
    public:
        anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
            anira::Tensor tensor{};
            anira::RingView ring;
            if (ctx.input_tensor(0, tensor) != ANIRA_OK || ctx.input_ring(0, ring) != ANIRA_OK) {
                return ANIRA_ERROR_CONFIG;
            }
            auto* model = static_cast<uint16_t*>(tensor.data(ANIRA_DTYPE_F16));
            if (model == nullptr || ring.pop_block(0, std::span<float>(m_scratch)) != HOP) {
                return ANIRA_ERROR_CONFIG;
            }
            for (size_t n = 0; n < HOP; ++n) { model[n] = to_half(m_scratch[n]); }
            return ANIRA_OK;
        }
        anira_status post_process(anira::ProcessorContext& ctx) noexcept override {
            anira::Tensor tensor{};
            anira::RingView ring;
            if (ctx.output_tensor(0, tensor) != ANIRA_OK || ctx.output_ring(0, ring) != ANIRA_OK) {
                return ANIRA_ERROR_CONFIG;
            }
            const auto* model = static_cast<const uint16_t*>(tensor.data(ANIRA_DTYPE_F16));
            if (model == nullptr) { return ANIRA_ERROR_CONFIG; }
            for (size_t n = 0; n < HOP; ++n) { m_scratch[n] = from_half(model[n]); }
            return ring.push_block(0, std::span<const float>(m_scratch)) == HOP
                       ? ANIRA_OK
                       : ANIRA_ERROR_CONFIG;
        }

    private:
        std::array<float, HOP> m_scratch{};
    };

    class HalfProcessor final : public anira::Processor {
    public:
        uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
        uint32_t flags() const noexcept override { return ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST; }
        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo&) override {
            return std::make_unique<HalfPrepared>();
        }
    };

    contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_F32).hard_ring_dtype("audio_out", ANIRA_DTYPE_F32);
    pipeline.add_processor(std::make_shared<HalfProcessor>());

A spectrogram model
-------------------

The input spec's window is longer than the hop (``window(2048, 2048, 1536)``: 512 fresh samples and
1536 of history), and the model tensor is the spectrum, so the default fill cannot be used:
``pre_process`` assembles the window (the history through ``peek_past_block``, the hop through
``pop_block``) and transforms it; ``post_process`` transforms back and pushes the hop. The transform
runs on the driving thread, so it is real-time itself (preallocated plans). The windows live in a
per-entry scratch that ``reset`` clears.

.. code-block:: cpp

    #include <algorithm>
    #include <memory>
    #include <span>
    #include <vector>
    #include <anira/anira.hpp>

    // Your transform of choice, over preallocated plans: no allocation in the call.
    void forward_transform(const float* window, size_t window_size, float* spectrum) noexcept;
    void inverse_transform(const float* spectrum, size_t window_size, float* samples) noexcept;

    class Spectral : public anira::Processor {
    public:
        uint32_t phases() const noexcept override { return k_pre_process | k_post_process; }
        uint32_t flags() const noexcept override { return ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST; }
        std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& info) override;

    private:
        class Frames;
    };

    class Spectral::Frames : public anira::Processor::Prepared {
    public:
        explicit Frames(uint32_t entries) : m_windows(static_cast<size_t>(entries) * k_window) {}

        void reset(anira::ProcessorContext& /*ctx*/) noexcept override {
            std::ranges::fill(m_windows, 0.0F);                    // a new stream: no history
        }

        anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
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

        anira_status post_process(anira::ProcessorContext& ctx) noexcept override {
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

    std::unique_ptr<anira::Processor::Prepared> Spectral::prepare(const anira::PrepareInfo& info) {
        return std::make_unique<Frames>(info.num_entries());   // this handler's windows
    }

The example is mono (channel 0); a multi-channel processor loops over ``ring.num_channels()`` and
lays the channels out as the model tensor expects. What an overlap-add keeps between frames lives in
a scratch of the prepared object's own that ``reset`` clears; the shape is the point: pop the hop
once, push the hop once, transform in between.

When the transform is heavy: the same in C, split across the hooks
------------------------------------------------------------------

A transform too heavy for the audio callback moves to the inference thread, keyed by the entry:
``pre_process`` only pops the window into the per-entry scratch, ``before_inference`` transforms it
into the model tensor, ``after_inference`` transforms the output back into the scratch, and
``post_process`` pushes the hop. The hooks may then use a library that allocates, since the flags
promise ``REALTIME_PRE_POST`` alone. ``prepare`` allocates this handler's windows and hands them back
through ``out_prepared``; ``unprepare`` frees them; ``user_data`` stays NULL, since nothing is shared.

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
    static void ANIRA_CALL spectral_reset(const anira_processor_ctx* ctx,
                                          void* prepared,
                                          void* user_data) ANIRA_NONBLOCKING {
        spectral* s = (spectral*)prepared;
        (void)ctx;
        (void)user_data;
        memset(s->windows, 0, (size_t)s->entries * WINDOW * sizeof(float));
    }

    /* driving thread, real-time: the pop alone */
    static anira_status ANIRA_CALL spectral_pre_process(const anira_processor_ctx* ctx,
                                                        void* prepared,
                                                        void* user_data) ANIRA_NONBLOCKING {
        float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_ring* ring = NULL;
        anira_status st = anira_processor_input_ring(ctx, 0, &ring);   /* slot 0 is Streamed: the model config says so */
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
    static anira_status ANIRA_CALL spectral_before_inference(const anira_processor_ctx* ctx,
                                                             void* prepared,
                                                             void* user_data) {
        const float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_tensor spectrum;
        float* bins;
        anira_status st = anira_processor_input_tensor(ctx, 0, &spectrum);
        if (st != ANIRA_OK) { return st; }
        bins = anira_tensor_data_f32(&spectrum);
        if (bins == NULL) { return ANIRA_ERROR_CONFIG; }
        forward_transform(window, WINDOW, bins);
        return ANIRA_OK;
    }

    /* inference thread: the inverse transform back into the scratch of the same entry */
    static anira_status ANIRA_CALL spectral_after_inference(const anira_processor_ctx* ctx,
                                                            void* prepared,
                                                            void* user_data) {
        float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_tensor spectrum;
        const float* bins;
        anira_status st = anira_processor_output_tensor(ctx, 0, &spectrum);
        if (st != ANIRA_OK) { return st; }
        bins = anira_tensor_data_f32(&spectrum);
        if (bins == NULL) { return ANIRA_ERROR_CONFIG; }
        inverse_transform(bins, WINDOW, window);
        return ANIRA_OK;
    }

    /* driving thread, real-time: the push alone */
    static anira_status ANIRA_CALL spectral_post_process(const anira_processor_ctx* ctx,
                                                         void* prepared,
                                                         void* user_data) ANIRA_NONBLOCKING {
        const float* window = ((spectral*)prepared)->windows + (size_t)ctx->entry * WINDOW;
        (void)user_data;
        anira_ring* ring = NULL;
        anira_status st = anira_processor_output_ring(ctx, 0, &ring);
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

    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.flags = ANIRA_PROCESSOR_FLAG_REALTIME_PRE_POST;   /* the hooks promise nothing */
    desc.pre_process = spectral_pre_process;
    desc.before_inference = spectral_before_inference;
    desc.after_inference = spectral_after_inference;
    desc.post_process = spectral_post_process;
    desc.reset = spectral_reset;
    desc.prepare = spectral_prepare;
    desc.unprepare = spectral_unprepare;
    anira_pipeline_add_processor(pipe, &desc, &err);

The entry makes the split correct: one handler's chunks may be in flight on several inference
threads, one entry holds one chunk at a time, and the four phases of a chunk see the same entry. The
entry count is fixed by each prepare, which is why the scratch is sized in ``prepare``, per handler.

Integration
-----------

.. code-block:: cpp

    anira::Pipeline pipeline;
    pipeline.add_inference(model)                                    // the default candidate set
        .add_processor(std::make_shared<Spectral>());
    anira::InferenceHandler handler(context, pipeline);              // the pipeline is copied
    handler.prepare(contract);   // validates the processor, then runs init (once) and prepare

.. code-block:: c

    anira_pipeline* pipe = NULL;
    anira_pipeline_create(&pipe, &err);
    anira_pipeline_add_inference(pipe, variants, 1, NULL, 0, &err);   /* NULL: the default candidate set */
    anira_pipeline_add_processor(pipe, &desc, &err);                  /* at most one */
    anira_handler* h = NULL;
    anira_handler_create(context, pipe, &h, &err);                    /* copies the pipeline */
    anira_pipeline_destroy(pipe);                                     /* the handler keeps its copy */
