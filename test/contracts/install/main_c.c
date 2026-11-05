/*
 * Gate 5: a pure-C consumer of the installed anira. The raw-C sequence of the architecture
 * document's section 9 against the installed package, with a registered engine that needs no
 * model file in place of the built-in candidates: the ABI check, a context, a model config, a
 * pipeline over the engine, a handler prepared under a Hard contract (an explicit budget and a
 * fixed warm-up), one block through the Hard entry, the teardown and anira_shutdown. It also
 * proves, from the installed include directories, that nothing private is reachable (gate 4's
 * probes): the package installs anira's public headers and nothing of tanh-lib, the queue or the
 * JSON library, and an engine's headers (include/anira-engines/<engine>/) are reachable through
 * anira::<engine> alone, never through anira::anira. Exit code 0 on success; every failed check
 * prints one line.
 */
#include <anira/anira.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__has_include)
#if __has_include(<tanh/core/Logger.h>) || __has_include(<nlohmann/json.hpp>) || \
    __has_include(<concurrentqueue.h>) || __has_include(<concurrentqueue/moodycamel/concurrentqueue.h>)
#error "a private dependency of anira is reachable through the installed anira::anira"
#endif
#if __has_include(<onnxruntime_cxx_api.h>) || __has_include(<scheduler/Core.h>)
#error "an engine or private header of anira is reachable through the installed anira::anira"
#endif
#endif

#define CONSUMER_ENGINE_ID "org.anira.install.c-copy"
#define CONSUMER_BLOCK 64

static int s_failures = 0;

static void fail(const char* what, anira_status status, const anira_error* err) {
    printf("FAILED: %s: %s (%s)\n",
           what,
           anira_status_string(status),
           err != NULL ? err->message : "");
    ++s_failures;
}

/* A pass-through engine: the first input into the first output. */
static anira_status ANIRA_CALL copy_engine(const anira_engine_ctx* ctx,
                                           void* prepared,
                                           void* user_data) {
    (void)prepared;
    (void)user_data;
    if (ctx->num_inputs == 0u || ctx->num_outputs == 0u) { return ANIRA_OK; }
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    const size_t count = anira_tensor_num_elements(&ctx->outputs[0]);
    if (in == NULL || out == NULL) { return ANIRA_ERROR_ENGINE; }
    for (size_t i = 0; i < count; ++i) { out[i] = in[i]; }
    return ANIRA_OK;
}

#define REQUIRE(call, what)                  \
    do {                                     \
        const anira_status status_ = (call); \
        if (ANIRA_FAILED(status_)) {         \
            fail((what), status_, &err);     \
            goto done;                       \
        }                                    \
    } while (0)

int main(void) {
    static const char k_model[] =
        "{\"models\": [{\"engine\": \"" CONSUMER_ENGINE_ID
        "\", \"path\": \"never-opened.bin\"}],"
        " \"inputs\": [{\"name\": \"audio_in\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 64]],"
        " \"window\": {\"min\": 64, \"max\": 64}}],"
        " \"outputs\": [{\"name\": \"audio_out\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 64]],"
        " \"window\": {\"min\": 64, \"max\": 64}}]}";
    anira_error err = ANIRA_ERROR_INIT;
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    anira_model_config* model = NULL;
    anira_pipeline* pipeline = NULL;
    anira_custom_engine* engine = NULL;
    anira_handler* handler = NULL;
    anira_contract* contract = NULL;
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.flags = ANIRA_ENGINE_FLAG_REALTIME_SAFE | ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    desc.process = copy_engine;

    /* The header this consumer compiled against must be served by this library. */
    REQUIRE(anira_check_abi(ANIRA_ABI_VERSION), "anira_check_abi");
    REQUIRE(anira_context_config_create(&config, &err), "anira_context_config_create");
    REQUIRE(anira_context_create(config, &context, &err), "anira_context_create");
    REQUIRE(anira_model_config_from_json(k_model, sizeof k_model - 1, NULL, &model, &err),
            "anira_model_config_from_json");
    REQUIRE(anira_custom_engine_create(CONSUMER_ENGINE_ID, &desc, &engine, &err),
            "anira_custom_engine_create");
    REQUIRE(anira_pipeline_create(&pipeline, &err), "anira_pipeline_create");
    REQUIRE(anira_pipeline_add_engine(pipeline, engine, &err), "anira_pipeline_add_engine");
    const anira_model_config* variants[1] = {model};
    REQUIRE(anira_pipeline_add_inference(pipeline, variants, 1, NULL, 0, &err),
            "anira_pipeline_add_inference");
    /* The handler copies everything: the configuration handles may go now. */
    REQUIRE(anira_handler_create(context, pipeline, &handler, &err), "anira_handler_create");
    REQUIRE(anira_contract_create_hard(CONSUMER_BLOCK, CONSUMER_BLOCK, 48000.0, &contract, &err),
            "anira_contract_create_hard");
    REQUIRE(anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 5.0),
            "anira_contract_hard_set_budget");
    REQUIRE(anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_FIXED, 2),
            "anira_contract_hard_set_warmup");
    REQUIRE(anira_handler_prepare(handler, contract, &err), "anira_handler_prepare");

    {
        float samples[CONSUMER_BLOCK];
        for (int i = 0; i < CONSUMER_BLOCK; ++i) { samples[i] = 0.25F; }
        float* planes[1] = {samples};
        const int64_t shape[2] = {1, CONSUMER_BLOCK};
        anira_tensor block;
        anira_tensor_init_host_planar(&block, planes, 1u, ANIRA_DTYPE_F32, 2u, shape);
        size_t delivered = 0;
        const anira_status status =
            anira_handler_process(handler, &block, 0, &block, 0, &delivered);
        if (status != ANIRA_OK) { fail("anira_handler_process", status, NULL); }
        if (anira_handler_rt_error(handler) != ANIRA_OK) {
            fail("anira_handler_rt_error", anira_handler_rt_error(handler), NULL);
        }
        printf("anira %s (ABI 0x%08x), process %s, latency %u\n",
               anira_version_string(),
               (unsigned)ANIRA_ABI_VERSION,
               anira_status_string(status),
               anira_handler_get_latency(handler, 0));
    }

done:
    anira_handler_destroy(handler);
    anira_contract_destroy(contract);
    anira_custom_engine_destroy(engine);
    anira_pipeline_destroy(pipeline);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    anira_context_config_destroy(config);
    /* The library's last call: nothing of this copy of anira lives any more. */
    {
        const anira_status shutdown = anira_shutdown();
        if (shutdown != ANIRA_OK) { fail("anira_shutdown", shutdown, NULL); }
    }
    printf("%s: consumer_c, %d failed check(s)\n",
           s_failures == 0 ? "PASSED" : "FAILED",
           s_failures);
    return s_failures == 0 ? 0 : 1;
}
