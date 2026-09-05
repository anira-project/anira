// The plugin-shaped test module: anira embedded in a loadable library, driven through the
// C API (anira/abi/). See CMakeLists.txt in this directory for what the test proves.

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>

#include <array>
#include <cstdint>

#include "module_api.h"

namespace {

constexpr uint32_t k_block_size = 512;
constexpr double k_sample_rate = 48000.0;
// The custom engine the 2.x CUSTOM backend maps to: no model file, the default processor.
constexpr const char* k_custom_engine = "anira.v2.custom";
constexpr const char* k_custom_path = "custom-processor";
constexpr const char* k_missing_model = "/nonexistent/anira-unload-test.model";

struct Instance {
    anira_context_config* m_config = nullptr;
    anira_context* m_context = nullptr;
    anira_handler* m_handler = nullptr;
    std::array<float, k_block_size> m_buffer{};
};

// The first engine this build carries; ONNX Runtime when there is none (an engine-less leg
// refuses the entry at create, which is the failure the throwing case wants there).
anira_engine first_enabled_engine() {
    anira_backend_id id = ANIRA_BACKEND_ID_INIT;
    uint32_t count = 1;
    const anira_status status = anira_enabled_backends(sizeof(anira_backend_id), &count, &id);
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

// A model config with one entry (the custom row, or the first enabled engine at a path that
// does not exist) and one streamed input and output; nullptr when a step fails.
anira_model_config* make_model(bool custom) {
    anira_model_config* config = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    if (anira_model_config_create(&config, &err) != ANIRA_OK) { return nullptr; }
    uint32_t index = 0;
    anira_status status = custom ? anira_model_config_add_model_path_custom(config,
                                                                            k_custom_engine,
                                                                            k_custom_path,
                                                                            &index,
                                                                            &err)
                                 : anira_model_config_add_model_path(config,
                                                                     first_enabled_engine(),
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

// A context (2 pool threads, SpinBackoff, Warning) and a handler over the model; nullptr,
// with nothing left behind, when a step fails.
Instance* make_instance(bool custom) {
    auto* instance = new Instance();
    anira_error err = ANIRA_ERROR_INIT;
    bool ok =
        anira_context_config_create(&instance->m_config, &err) == ANIRA_OK &&
        anira_context_config_set_threads(instance->m_config, 2, ANIRA_WAIT_SPIN_BACKOFF) ==
            ANIRA_OK &&
        anira_context_config_set_log_level(instance->m_config, ANIRA_LOG_WARNING) == ANIRA_OK &&
        anira_context_create(instance->m_config, &instance->m_context, &err) == ANIRA_OK;
    if (ok) {
        anira_model_config* model = make_model(custom);
        anira_pipeline* pipeline = nullptr;
        // NULL candidates: the default set (every engine this build carries plus the custom
        // rows), under which an entry for an absent engine is skipped.
        ok = model != nullptr && anira_pipeline_create(&pipeline, &err) == ANIRA_OK &&
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

}  // namespace

extern "C" {

void* unloadtest_create(void) {
    try {
        return make_instance(/*custom=*/true);
    } catch (...) { return nullptr; }
}

void unloadtest_prepare(void* instance) {
    static_cast<void>(prepare_instance(*static_cast<Instance*>(instance)));
}

void unloadtest_process(void* instance, int num_blocks) {
    auto* i = static_cast<Instance*>(instance);
    float* channel = i->m_buffer.data();
    for (int block = 0; block < num_blocks; ++block) {
        static_cast<void>(anira_handler_process(i->m_handler, &channel, k_block_size, 0));
    }
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
        instance = make_instance(/*custom=*/false);
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
