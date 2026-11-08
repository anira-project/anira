Engine examples
===============

Whole engines; :doc:`../custom_engines` is the reference for what they call.

.. _engines-example-c:

A gain engine in C
------------------

An engine that multiplies its first input into its first output, every slot filled. ``load`` keeps
the element count of the templates and the shared-slot count, nothing of the memory; ``prepare``
keeps the loaded model it runs on and whether the handler is exclusive; ``process`` finds both
through ``ctx->loaded`` and ``prepared`` and reads the pointers of its own call. The entry's path is
read but not opened, since this engine has no file to load:

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

At setup (``init``, ``reset`` and ``release`` could stay ``NULL``):

.. code-block:: c

    static gain_engine engine = { 0.5f };
    static const char* const providers[] = { "cpu", "coreml" };  /* exactly what it serves */
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE;       /* the promise the body keeps */
    desc.providers = providers;                         /* the enum's spellings, or */
    desc.num_providers = 2;                             /* any name of the engine's own */
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

.. _engines-example-cpp:

The same in C++
---------------

The ``Loaded`` holds what every handler of the model shares, immutable after load; the ``Prepared``
is one handler on it, where ``process`` runs (a shared call on the slot the context names, an
exclusive call on what it built at prepare: nothing, since the gain keeps no state).

.. code-block:: cpp

    #include <algorithm>
    #include <array>
    #include <memory>
    #include <anira/anira.hpp>

    class Gain : public anira::Engine {
    public:
        explicit Gain(float gain) : anira::Engine("com.example.gain"), m_gain(gain) {}
        uint32_t flags() const noexcept override { return ANIRA_ENGINE_FLAG_REALTIME_SAFE; }
        // The providers served, exactly these: the enum's spellings ("cpu" for the CPU path)
        // or any name of the engine's own; the gain does the same work on every one of them.
        std::span<const char* const> providers() const noexcept override { return k_providers; }
        // init(const anira::InitInfo&) is not overridden: nothing to build once per object.
        std::unique_ptr<Loaded> load(const anira::EngineLoadInfo& info) override;

    private:
        static constexpr std::array<const char*, 2> k_providers{"cpu", "coreml"};
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
    anira::Pipeline pipe;
    pipe.add_engine(std::make_shared<Gain>(0.5F));
    pipe.add_inference(cfg);

To share it across the instances of a plugin, register one object per binary
(:ref:`engines-plugin-instances`).

.. _engines-example-onnx:

An engine that drives ONNX Runtime itself
-----------------------------------------

A custom engine may drive one of the bundled inference engines itself, for an ONNX Runtime execution
provider or session option anira's own adapter does not expose. Two rules keep that safe, the rules
anira's own adapters follow.

**Link the engine target.** ``anira::anira`` carries anira's headers but no engine header: anira links
its engines privately, and the ``USE_<ENGINE>`` definitions reach only the consumers inside anira's
build tree, not the installed ``anira::anira`` (which engines a build compiled in is
``anira_enabled_engines``, ``anira::enabled_engines()``). The engine's include directory and library live on the matching engine
target, in anira's build tree and in the installed package alike:

.. code-block:: cmake

    target_link_libraries(my_engine PRIVATE anira::anira anira::onnxruntime)

``anira::onnxruntime``, ``anira::tflite``, ``anira::litert``, ``anira::libtorch`` and
``anira::executorch`` are the very files anira links, a shared library in a shared anira build, a
static archive (linked on demand and hidden) in a static one, so the process holds exactly one copy
of the engine. Never link an engine by any other path next to anira.

**Keep the engine out of your header.** What one load shares (the environment, the options) and the
executors of the shared slots belong in the ``Engine::Loaded``, an exclusive handler's own executor
in its ``Engine::Prepared`` (:doc:`../architecture` shows how the built-in engines draw the line).
Declare both subclasses in the ``.cpp`` alone, so the header a plugin includes names no engine type:

.. code-block:: cpp

    // MyOnnxEngine.h: no engine include here
    #include <anira/anira.hpp>

    class MyOnnxEngine final : public anira::Engine {
    public:
        MyOnnxEngine() : anira::Engine("com.example.myonnx") {}
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

Whatever includes your header compiles with anira's include directories alone, and only your
``.cpp`` needs the engine target's, which is what lets a plugin keep every engine symbol private:
compile that ``.cpp`` with hidden visibility (:ref:`plugins-engine-symbols`).
