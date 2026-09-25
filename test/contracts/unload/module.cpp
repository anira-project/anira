// The plugin-shaped test module: anira embedded in a loadable library, driven through the
// C API (anira/abi/). See CMakeLists.txt in this directory for what the test proves.

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "../../../extras/models/model_files.h"
#include "module_api.h"

namespace {

constexpr uint32_t k_block_size = 512;
constexpr double k_sample_rate = 48000.0;
// The id of the 2.x CUSTOM backend, under which the module adds its pass-through engine; the
// row names no model file.
constexpr const char* k_custom_engine = "anira.v2.custom";
constexpr const char* k_custom_path = "custom-processor";
constexpr const char* k_missing_model = "/nonexistent/anira-unload-test.model";
constexpr double k_wait_ms = 10000.0;

struct Instance {
    anira_context_config* m_config = nullptr;
    anira_context* m_context = nullptr;
    anira_handler* m_handler = nullptr;
    std::array<float, k_block_size> m_buffer{};
};

// The pass-through engine of the custom row: the one stream in, copied out. Its code lives in
// this module. The cases that destroy their instance before the unload run it
// (DefaultPolicyLeavesNoThreadBehind, ReloadAfterUnloadWorks, through unloadtest_create). The
// case that leaks a live session into the unload (UnloadWithLiveSessionIsJoinedByHook) runs a
// built-in engine where the build has one that runs the gain model (unloadtest_create_builtin):
// the loader may unmap the module before libanira's unload hook joins the pool (dyld does),
// and this function must never run after that. On a build without one the leaked case falls
// back to this engine, which is safe only because nothing reaches the module once the leaked
// session's queue is drained: anira_custom_engine_create copies the descriptor into the
// engine's carrier and owns the id (src/capi/engine.cpp), the teardown slots (unload,
// unprepare, release) are called only when set and are NULL here, process runs only for an
// inference, and the leaked case completes every block before the unload.
anira_status ANIRA_CALL passthrough_process(const anira_engine_ctx* ctx,
                                            void* /*prepared*/,
                                            void* /*user_data*/) {
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    std::memcpy(out, in, sizeof(float) * k_block_size);
    return ANIRA_OK;
}

// The inferences of the engines below that reached their process, and how long the slow one
// takes per call.
std::atomic<int> s_entered{0};
std::atomic<int> s_slow_ms{0};

// An inference that never ends on its own: it sleeps until the process dies.
anira_status ANIRA_CALL stalling_process(const anira_engine_ctx* /*ctx*/,
                                         void* /*prepared*/,
                                         void* /*user_data*/) {
    s_entered.fetch_add(1);
    while (true) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
}

// An inference that needs the loader lock again and again (what LibTorch's first inference on a
// thread does once, through __cxa_thread_atexit): on glibc dlopen takes dl_load_lock, which a
// dlclose holds while the unload hook runs, so this call stops inside dlopen for good once the
// unload has begun. RTLD_NOLOAD on libc: nothing is loaded and no reference of the module is
// taken. Elsewhere it stalls like stalling_process.
anira_status ANIRA_CALL loader_lock_process([[maybe_unused]] const anira_engine_ctx* ctx,
                                            [[maybe_unused]] void* prepared,
                                            [[maybe_unused]] void* user_data) {
#if defined(__linux__)
    s_entered.fetch_add(1);
    while (true) {
        if (void* libc = dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD)) { dlclose(libc); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#else
    return stalling_process(ctx, prepared, user_data);
#endif
}

// The pass-through after s_slow_ms of sleep.
anira_status ANIRA_CALL slow_process(const anira_engine_ctx* ctx, void* prepared, void* user_data) {
    s_entered.fetch_add(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(s_slow_ms.load()));
    return passthrough_process(ctx, prepared, user_data);
}

// The process of an engine of unloadtest_create_engine; nullptr for an unknown behaviour.
anira_engine_process_fn engine_process_of(int behaviour) {
    switch (behaviour) {
        case UNLOADTEST_ENGINE_STALLING: return stalling_process;
        case UNLOADTEST_ENGINE_LOADER_LOCK: return loader_lock_process;
        case UNLOADTEST_ENGINE_SLOW: return slow_process;
        default: return nullptr;
    }
}

// A new engine object over `process` under k_custom_engine; nullptr when the create fails.
anira_custom_engine* make_engine(anira_engine_process_fn process) {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.process = process;
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_custom_engine_create(k_custom_engine, &desc, &engine, &err) != ANIRA_OK) {
        return nullptr;
    }
    return engine;
}

// The first engine this build carries; ONNX Runtime when there is none (an engine-less leg
// refuses the entry at create, which is the failure the throwing case wants there).
anira_engine first_enabled_engine() {
    anira_backend_id id = ANIRA_BACKEND_ID_INIT;
    uint32_t count = 1;
    const anira_status status = anira_enabled_engines(sizeof(anira_backend_id), &count, &id);
    if ((status != ANIRA_OK && status != ANIRA_INCOMPLETE) || count == 0) {
        return ANIRA_ENGINE_ONNXRUNTIME;
    }
    return static_cast<anira_engine>(id.engine);
}

// A mono streamed tensor of 512 samples: batch 1, channel 1, time 512, window 512/512.
anira_tensor_spec* make_spec(const char* name) {
    anira_tensor_spec* spec = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_tensor_spec_create(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED, &spec, &err) !=
        ANIRA_OK) {
        return nullptr;
    }
    const bool ok =
        anira_tensor_spec_set_axis(spec, 0, ANIRA_AXIS_BATCH, 1) == ANIRA_OK &&
        anira_tensor_spec_set_axis(spec, 1, ANIRA_AXIS_CHANNEL, 1) == ANIRA_OK &&
        anira_tensor_spec_set_axis(spec, 2, ANIRA_AXIS_TIME, k_block_size) == ANIRA_OK &&
        anira_tensor_spec_set_window(spec, k_block_size, k_block_size, 0) == ANIRA_OK;
    if (!ok) {
        anira_tensor_spec_destroy(spec);
        return nullptr;
    }
    return spec;
}

// What an instance runs: an engine of this module on the custom row, the first enabled engine
// at a path that does not exist (a model that never loads), or the bundled gain model on a
// built-in engine of this build.
enum class Kind { Custom, MissingModel, BuiltIn };

// The bundled gain model (extras/models/model-pool/gain.model.json: one entry per built-in
// engine, a 512-sample stream and a Static gain, the paths relative to the file); nullptr when
// the file does not load.
anira_model_config* make_builtin_model() {
    anira_model_config* config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_model_config_from_json_file(k_gain_model_json, &config, &err) != ANIRA_OK) {
        return nullptr;
    }
    return config;
}

// A model config with one entry (the custom row, or the first enabled engine at a path that
// does not exist) and one streamed input and output; nullptr when a step fails.
anira_model_config* make_model(bool custom) {
    anira_model_config* config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_model_config_create(&config, &err) != ANIRA_OK) { return nullptr; }
    uint32_t index = 0;
    anira_status status = custom ? anira_model_config_add_model_path(config,
                                                                     ANIRA_ENGINE_CUSTOM,
                                                                     k_custom_engine,
                                                                     k_custom_path,
                                                                     &index,
                                                                     &err)
                                 : anira_model_config_add_model_path(config,
                                                                     first_enabled_engine(),
                                                                     nullptr,
                                                                     k_missing_model,
                                                                     &index,
                                                                     &err);
    anira_tensor_spec* in = make_spec("in");
    anira_tensor_spec* out = make_spec("out");
    if (status == ANIRA_OK && in != nullptr && out != nullptr) {
        status = anira_model_config_add_input(config, in);
        if (status == ANIRA_OK) { status = anira_model_config_add_output(config, out); }
    }
    const bool ok = status == ANIRA_OK && in != nullptr && out != nullptr;
    anira_tensor_spec_destroy(in);
    anira_tensor_spec_destroy(out);
    if (!ok) {
        anira_model_config_destroy(config);
        return nullptr;
    }
    return config;
}

void destroy_instance(Instance* instance) {
    anira_handler_destroy(instance->m_handler);
    anira_context_destroy(instance->m_context);
    anira_context_config_destroy(instance->m_config);
    delete instance;
}

// A context (2 pool threads, SpinBackoff, Warning) and a handler over the model of the kind,
// the custom row on an engine over `process`; nullptr, with nothing left behind, when a step
// fails.
Instance* make_instance(Kind kind, anira_engine_process_fn process = passthrough_process) {
    const bool custom = kind == Kind::Custom;
    auto* instance = new Instance();
    anira_error err = ANIRA_ERROR_INIT;
    bool ok =
        anira_context_config_create(&instance->m_config, &err) == ANIRA_OK &&
        anira_context_config_set_threads(instance->m_config, 2, ANIRA_WAIT_SPIN_BACKOFF) ==
            ANIRA_OK &&
        anira_context_config_set_log_level(instance->m_config, ANIRA_LOG_WARNING) == ANIRA_OK &&
        anira_context_create(instance->m_config, &instance->m_context, &err) == ANIRA_OK;
    if (ok) {
        anira_model_config* model =
            kind == Kind::BuiltIn ? make_builtin_model() : make_model(custom);
        anira_pipeline* pipeline = nullptr;
        ok = model != nullptr && anira_pipeline_create(&pipeline, &err) == ANIRA_OK;
        if (ok && custom) {
            // The custom row's engine, added before the handler copies the pipeline; the
            // pipeline holds the object, so the handle goes right away.
            anira_custom_engine* engine = make_engine(process);
            ok = engine != nullptr && anira_pipeline_add_engine(pipeline, engine, &err) == ANIRA_OK;
            anira_custom_engine_destroy(engine);
        }
        // NULL candidates: the default set (every engine this build carries plus the custom
        // engines the entries name), under which an entry for an absent engine is skipped; an
        // engine-less build has no plan for the gain model, and create refuses it.
        ok = ok &&
             anira_pipeline_add_inference(pipeline, &model, 1, nullptr, 0, &err) == ANIRA_OK &&
             anira_handler_create(instance->m_context, pipeline, &instance->m_handler, &err) ==
                 ANIRA_OK;
        // The handler copied everything.
        anira_pipeline_destroy(pipeline);
        anira_model_config_destroy(model);
    }
    if (!ok) {
        destroy_instance(instance);
        return nullptr;
    }
    return instance;
}

// A 512/512/48 kHz Hard contract with an explicit 1 ms budget and no warm-up.
anira_status prepare_instance(Instance& instance) {
    anira_contract* contract = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    anira_status status =
        anira_contract_create_hard(k_block_size, k_block_size, k_sample_rate, &contract, &err);
    if (status == ANIRA_OK) {
        status = anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1.0);
    }
    if (status == ANIRA_OK) {
        status = anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_FIXED, 0);
    }
    if (status == ANIRA_OK) { status = anira_handler_prepare(instance.m_handler, contract, &err); }
    anira_contract_destroy(contract);
    return status;
}

// num_blocks blocks through the handler, in place on slot 0. With wait, each block waits for
// its inference (anira_handler_process_wait, 10 s at most), so the queue is drained when the
// call returns and no pool thread is inside an engine.
void process_blocks(Instance& instance, int num_blocks, bool wait) {
    // In place: one planar tensor over the channel pointers, handed over as input and output.
    const std::array<float*, 1> channels{instance.m_buffer.data()};
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block_size)};
    anira_tensor io{};
    anira_tensor_init_host_planar(&io,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    for (int block = 0; block < num_blocks; ++block) {
        if (wait) {
            static_cast<void>(
                anira_handler_process_wait(instance.m_handler, &io, 0, &io, 0, nullptr, k_wait_ms));
        } else {
            static_cast<void>(anira_handler_process(instance.m_handler, &io, 0, &io, 0, nullptr));
        }
    }
}

}  // namespace

extern "C" {

void* unloadtest_create(void) {
    try {
        return make_instance(Kind::Custom);
    } catch (...) { return nullptr; }
}

void* unloadtest_create_builtin(void) {
    // Created and prepared here: a build whose built-in engines cannot run the gain model
    // (none carried, or the model file of the chosen one missing) refuses one of the two.
    Instance* instance = nullptr;
    try {
        instance = make_instance(Kind::BuiltIn);
    } catch (...) { return nullptr; }
    if (instance == nullptr) { return nullptr; }
    if (prepare_instance(*instance) != ANIRA_OK) {
        destroy_instance(instance);
        return nullptr;
    }
    return instance;
}

void* unloadtest_create_engine(int behaviour, int slow_ms) {
    s_slow_ms.store(slow_ms);
    const anira_engine_process_fn process = engine_process_of(behaviour);
    if (process == nullptr) { return nullptr; }
    Instance* instance = nullptr;
    try {
        instance = make_instance(Kind::Custom, process);
    } catch (...) { return nullptr; }
    if (instance == nullptr) { return nullptr; }
    if (prepare_instance(*instance) != ANIRA_OK) {
        destroy_instance(instance);
        return nullptr;
    }
    return instance;
}

int unloadtest_engine_entered(void) {
    return s_entered.load();
}

void unloadtest_prepare(void* instance) {
    static_cast<void>(prepare_instance(*static_cast<Instance*>(instance)));
}

void unloadtest_process(void* instance, int num_blocks) {
    process_blocks(*static_cast<Instance*>(instance), num_blocks, /*wait=*/false);
}

void unloadtest_process_wait(void* instance, int num_blocks) {
    process_blocks(*static_cast<Instance*>(instance), num_blocks, /*wait=*/true);
}

void unloadtest_destroy(void* instance) {
    destroy_instance(static_cast<Instance*>(instance));
}

int unloadtest_create_throwing(void) {
    // An engine at a path that does not exist: on an engine-less leg create fails
    // (ANIRA_ERROR_CONFIG: the only entry is skipped under the default set, no candidate
    // matches), elsewhere create succeeds and prepare fails (ANIRA_ERROR_NO_SUCH_FILE).
    // Either way nothing may be left behind.
    Instance* instance = nullptr;
    try {
        instance = make_instance(Kind::MissingModel);
    } catch (...) { return 1; }
    if (instance == nullptr) { return 1; }
    const anira_status status = prepare_instance(*instance);
    destroy_instance(instance);
    return status == ANIRA_OK ? 0 : 1;
}

unsigned int unloadtest_num_inference_threads(void) {
    return anira_num_inference_threads();
}

int unloadtest_has_inference_threads(void) {
    return anira::Core::has_inference_threads() ? 1 : 0;
}

int unloadtest_num_sessions(void) {
    return anira::Core::get_num_sessions();
}

int unloadtest_has_core(void) {
    return anira_has_core() ? 1 : 0;
}

void unloadtest_shutdown(void) {
    // The forcing one: anira_shutdown refuses while a context or a handler lives.
    anira::Core::shutdown();
}

void unloadtest_leak_thread(void) {
    // A context that is never destroyed and a user-managed inference thread that is never
    // stopped. SpinBackoff (the configuration's default): the thread wakes every <= 100 us,
    // so it runs into the unmapped code within a millisecond of the unload.
    anira_context_config* config = nullptr;
    anira_context* context = nullptr;
    anira_inference_thread* thread = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_context_config_create(&config, &err) != ANIRA_OK) { return; }
    if (anira_context_create(config, &context, &err) != ANIRA_OK) { return; }
    if (anira_inference_thread_create(context, &thread, &err) != ANIRA_OK) { return; }
    static_cast<void>(anira_inference_thread_start(thread, &err));
}
}
