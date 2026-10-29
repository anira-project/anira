/*
 * The node test of issue #212 (WebAssembly only): the core's mutexes exclude across the threads
 * of one module. Two threads on the module's one shared memory, the main thread and a second
 * one (a pthread of the -pthread link: the same threaded libc a Wasm Worker and the audio
 * worklet of AniraWeb run on), each build and tear down a context, a pipeline over a C engine
 * and a prepared Hard handler, many times over and at the same time: every prepare and every
 * destroy enters Core's lifecycle mutex and edits the session registry under it, the contexts
 * the context registry under capi's (after one round alone, which creates the core's state).
 * With stub mutexes (the no-pthread libraries of emsdk 4,
 * which lock nothing) the two edit those vectors at once. Every call must succeed, and the core
 * must be empty again afterwards. Exit code 0 on success; every failed check prints one line.
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

#define MUTEX_ROUNDS 1000
#define MUTEX_BLOCK 128u
#define MUTEX_ENGINE_ID "org.anira-test.c-copy"

static atomic_int s_failures = 0;
/* Both threads start their rounds together. */
static atomic_int s_ready = 0;

static void fail(const char* what, anira_status status, const anira_error* err) {
    printf("FAILED: %s: %s (%s)\n", what, anira_status_string(status), err ? err->message : "");
    atomic_fetch_add(&s_failures, 1);
}

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
        if (status_ != ANIRA_OK) {           \
            fail((what), status_, &err);     \
            goto done;                       \
        }                                    \
    } while (0)

/* One round: everything a host builds for one handler, prepared, then destroyed. */
static void round_trip(void) {
    static const char k_json[] =
        "{\"models\": [{\"engine\": \"" MUTEX_ENGINE_ID
        "\", \"path\": \"never-opened.bin\"}],"
        " \"inputs\": [{\"name\": \"in\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}],"
        " \"outputs\": [{\"name\": \"out\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}]}";
    anira_error err = ANIRA_ERROR_INIT;
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    anira_model_config* model = NULL;
    anira_pipeline* pipeline = NULL;
    anira_custom_engine* engine = NULL;
    anira_handler* handler = NULL;
    anira_contract* contract = NULL;
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.flags = ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    desc.process = copy_engine;

    REQUIRE(anira_context_config_create(&config, &err), "anira_context_config_create");
    REQUIRE(anira_context_config_set_threads(config, 0, ANIRA_WAIT_SPIN_BACKOFF),
            "anira_context_config_set_threads");
    REQUIRE(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10),
            "anira_context_config_set_log_drain");
    REQUIRE(anira_context_create(config, &context, &err), "anira_context_create");
    REQUIRE(anira_model_config_from_json(k_json, sizeof k_json - 1, NULL, &model, &err),
            "anira_model_config_from_json");
    REQUIRE(anira_pipeline_create(&pipeline, &err), "anira_pipeline_create");
    const anira_model_config* variants[1] = {model};
    REQUIRE(anira_pipeline_add_inference(pipeline, variants, 1, NULL, 0, &err),
            "anira_pipeline_add_inference");
    REQUIRE(anira_custom_engine_create(MUTEX_ENGINE_ID, &desc, &engine, &err),
            "anira_custom_engine_create");
    REQUIRE(anira_pipeline_add_engine(pipeline, engine, &err), "anira_pipeline_add_engine");
    REQUIRE(anira_handler_create(context, pipeline, &handler, &err), "anira_handler_create");
    REQUIRE(anira_contract_create_hard(MUTEX_BLOCK, MUTEX_BLOCK, 48000.0, &contract, &err),
            "anira_contract_create_hard");
    REQUIRE(anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1.0),
            "anira_contract_hard_set_budget");
    REQUIRE(anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_NONE, 0),
            "anira_contract_hard_set_warmup");
    REQUIRE(anira_handler_prepare(handler, contract, &err), "anira_handler_prepare");
done:
    anira_handler_destroy(handler);
    anira_contract_destroy(contract);
    anira_custom_engine_destroy(engine);
    anira_pipeline_destroy(pipeline);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    anira_context_config_destroy(config);
}

static void run_rounds(void) {
    atomic_fetch_add(&s_ready, 1);
    while (atomic_load(&s_ready) < 2) {}
    for (int i = 0; i < MUTEX_ROUNDS; ++i) { round_trip(); }
}

static void* second_thread(void* arg) {
    (void)arg;
    run_rounds();
    return NULL;
}

int main(void) {
    /* The core's state first, alone: it is created once per process, at the first context, and
       two threads that race to create it each build one (the loser's is dropped), which this
       module's fixed memory cannot hold twice. */
    round_trip();
    pthread_t thread;
    if (pthread_create(&thread, NULL, second_thread, NULL) != 0) {
        printf("FAILED: pthread_create\n");
        return 1;
    }
    run_rounds();
    pthread_join(thread, NULL);
    if (anira_num_inference_threads() != 0u) {
        printf("FAILED: the core keeps a thread pool after every context is gone\n");
        atomic_fetch_add(&s_failures, 1);
    }
    /* A round alone afterwards: the registries the two edited are intact. */
    round_trip();
    const int failures = atomic_load(&s_failures);
    printf("%s: %d rounds on each of two threads, %d failed check(s)\n",
           failures == 0 ? "PASSED" : "FAILED",
           MUTEX_ROUNDS,
           failures);
    return failures == 0 ? 0 : 1;
}
