/*
 * The node test of the context's pool on WebAssembly: a context of two inference threads, which
 * anira starts as Wasm Workers (InferenceThread::start), each importing the worker module named
 * on the command line (em_pool_module.mjs: a JavaScript engine that doubles its input, defined on
 * the registry of the Worker's scope) before it enters its loop. The main thread pushes blocks
 * through a Hard handler over that engine and waits for each (anira_handler_process_wait); the
 * blocks come back doubled at the handler's latency. Destroying the handler and the context
 * stops the pool: its Workers leave their loops and are terminated, and the core reports no
 * thread afterwards. Exit code 0 on success; every failed check prints one line.
 * usage: node anira_em_pool.js <path of em_pool_module.mjs>
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/anira_em.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define POOL_THREADS 2u
#define POOL_BLOCK 128u
#define POOL_BLOCKS 32u
#define POOL_ENGINE_ID "org.anira-test.pool-double"
#define POOL_WAIT_MS 5000.0

static int s_failures = 0;

static void check(int ok, const char* what) {
    if (!ok) {
        printf("FAILED: %s\n", what);
        ++s_failures;
    }
}

static void check_status(anira_status status, const anira_error* err, const char* what) {
    if (status != ANIRA_OK) {
        printf("FAILED: %s: %s (%s)\n", what, anira_status_string(status), err ? err->message : "");
        ++s_failures;
    }
}

/* The engine's key on the registry: FNV-1a of the id, as the package's keyOf computes it. */
static uint32_t key_of(const char* name) {
    uint32_t hash = 0x811c9dc5u;
    for (const char* c = name; *c != '\0'; ++c) {
        hash ^= (uint32_t)(unsigned char)*c;
        hash *= 0x01000193u;
    }
    return hash;
}

int main(int argc, char** argv) {
    static const char k_json[] =
        "{\"models\": [{\"engine\": \"" POOL_ENGINE_ID
        "\", \"path\": \"never-opened.bin\"}],"
        " \"inputs\": [{\"name\": \"in\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}],"
        " \"outputs\": [{\"name\": \"out\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}]}";
    if (argc < 2) {
        printf("FAILED: usage: anira_em_pool <path of em_pool_module.mjs>\n");
        return 1;
    }
    char url[1024];
    snprintf(url, sizeof url, "file://%s", argv[1]);

    anira_error err = ANIRA_ERROR_INIT;
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    anira_model_config* model = NULL;
    anira_pipeline* pipeline = NULL;
    anira_handler* handler = NULL;
    anira_contract* contract = NULL;
    check_status(anira_context_config_create(&config, &err), &err, "anira_context_config_create");
    check_status(anira_context_config_set_threads(config, POOL_THREADS, ANIRA_WAIT_SPIN_BACKOFF),
                 NULL,
                 "anira_context_config_set_threads");
    check_status(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10),
                 NULL,
                 "anira_context_config_set_log_drain");
    check_status(anira_em_context_config_set_worker_module(config, url, &err),
                 &err,
                 "anira_em_context_config_set_worker_module");
    check(
        anira_em_context_config_set_worker_module(NULL, url, NULL) == ANIRA_ERROR_INVALID_ARGUMENT,
        "the worker module setter refuses a NULL config");
    check_status(anira_context_create(config, &context, &err), &err, "anira_context_create");
    check_status(anira_model_config_from_json(k_json, sizeof k_json - 1, NULL, &model, &err),
                 &err,
                 "anira_model_config_from_json");
    check_status(anira_pipeline_create(&pipeline, &err), &err, "anira_pipeline_create");
    const anira_model_config* variants[1] = {model};
    check_status(anira_pipeline_add_inference(pipeline, variants, 1, NULL, 0, &err),
                 &err,
                 "anira_pipeline_add_inference");
    check_status(anira_em_pipeline_add_js_engine(pipeline,
                                                 POOL_ENGINE_ID,
                                                 key_of(POOL_ENGINE_ID),
                                                 0,
                                                 ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
                                                 &err),
                 &err,
                 "anira_em_pipeline_add_js_engine");
    check_status(anira_handler_create(context, pipeline, &handler, &err),
                 &err,
                 "anira_handler_create");
    check_status(anira_contract_create_hard(POOL_BLOCK, POOL_BLOCK, 48000.0, &contract, &err),
                 &err,
                 "anira_contract_create_hard");
    check_status(anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1.0),
                 NULL,
                 "anira_contract_hard_set_budget");
    check_status(anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_NONE, 0),
                 NULL,
                 "anira_contract_hard_set_warmup");
    check_status(anira_handler_prepare(handler, contract, &err), &err, "anira_handler_prepare");
    check(anira_num_inference_threads() == POOL_THREADS,
          "the context's pool runs two inference threads");

    float in[POOL_BLOCK];
    float out[POOL_BLOCK];
    int64_t shape[2] = {1, POOL_BLOCK};
    anira_tensor in_tensor;
    anira_tensor out_tensor;
    anira_tensor_init_host(&in_tensor, in, ANIRA_DTYPE_F32, 2, shape);
    anira_tensor_init_host(&out_tensor, out, ANIRA_DTYPE_F32, 2, shape);
    const uint32_t lag = anira_handler_get_latency(handler, 0) / POOL_BLOCK;
    uint32_t matched = 0;
    for (uint32_t block = 0; block < POOL_BLOCKS; ++block) {
        for (uint32_t i = 0; i < POOL_BLOCK; ++i) { in[i] = (float)(block + 1u); }
        size_t delivered = 0;
        /* The first blocks wait for the Workers to import their module. */
        const anira_status status = anira_handler_process_wait(handler,
                                                               &in_tensor,
                                                               0,
                                                               &out_tensor,
                                                               0,
                                                               &delivered,
                                                               POOL_WAIT_MS);
        if (status != ANIRA_OK && status != ANIRA_MISSED) {
            check_status(status, NULL, "anira_handler_process_wait");
            continue;
        }
        if (block >= lag && delivered == POOL_BLOCK && out[0] == 2.0f * (float)(block + 1u - lag) &&
            out[POOL_BLOCK - 1u] == 2.0f * (float)(block + 1u - lag)) {
            ++matched;
        }
    }
    printf("blocks doubled by the pool's Wasm Workers: %u of %u\n", matched, POOL_BLOCKS);
    check(matched >= POOL_BLOCKS - lag - 1u, "the pool's Workers ran the blocks");

    anira_handler_destroy(handler);
    anira_pipeline_destroy(pipeline);
    anira_contract_destroy(contract);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    anira_context_config_destroy(config);
    check(anira_num_inference_threads() == 0u, "the pool stopped with the last context");
    printf("%s: %d failed check(s)\n", s_failures == 0 ? "PASSED" : "FAILED", s_failures);
    return s_failures == 0 ? 0 : 1;
}
