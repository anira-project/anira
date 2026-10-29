/*
 * The node smoke test of a user-driven inference thread on WebAssembly (decision 9.12): the
 * main instance builds a Hard handler over a JavaScript engine and creates the thread;
 * em_smoke_thread_worker.mjs runs a second instance of this module in a worker_threads Worker
 * on the same memory, with its own stack and thread-local block (anira_em_init_instance), and
 * calls anira_inference_thread_run_loop there for real: the loop parks on the wake word while
 * the queue is empty and is woken by the enqueues and the stop. em_smoke_thread.mjs drives
 * the exports below from the main thread. Every failed check prints one line.
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
#include <emscripten/emscripten.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SMOKE_BLOCK 128u
#define SMOKE_ENGINE_KEY 7u
#define SMOKE_ENGINE_ID "org.anira-test.js-double"
#define SMOKE_STACK_BYTES (4u * 1024u * 1024u)

static int s_failures = 0;
static anira_context_config* s_config = NULL;
static anira_context* s_context = NULL;
static anira_model_config* s_model = NULL;
static anira_pipeline* s_pipeline = NULL;
static anira_handler* s_handler = NULL;
static anira_contract* s_contract = NULL;
static anira_inference_thread* s_thread = NULL;
static void* s_stack = NULL;

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

/* The offsets the Worker's engine reads the engine context through (the compiler's). */
EMSCRIPTEN_KEEPALIVE uint32_t smoke_ctx_inputs_offset(void) {
    return (uint32_t)offsetof(anira_engine_ctx, inputs);
}
EMSCRIPTEN_KEEPALIVE uint32_t smoke_ctx_outputs_offset(void) {
    return (uint32_t)offsetof(anira_engine_ctx, outputs);
}

/* The handler, the thread (started: the Worker enters its loop next) and the Worker's stack. */
EMSCRIPTEN_KEEPALIVE int smoke_thread_setup(void) {
    static const char k_json[] =
        "{\"models\": [{\"engine\": \"" SMOKE_ENGINE_ID
        "\", \"path\": \"never-opened.bin\"}],"
        " \"inputs\": [{\"name\": \"in\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}],"
        " \"outputs\": [{\"name\": \"out\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}]}";
    anira_error err = ANIRA_ERROR_INIT;
    check_status(anira_context_config_create(&s_config, &err), &err, "anira_context_config_create");
    check_status(anira_context_config_set_log_drain(s_config, ANIRA_LOG_DRAIN_MANUAL, 10),
                 NULL,
                 "anira_context_config_set_log_drain");
    check_status(anira_context_create(s_config, &s_context, &err), &err, "anira_context_create");
    check_status(anira_model_config_from_json(k_json, sizeof k_json - 1, NULL, &s_model, &err),
                 &err,
                 "anira_model_config_from_json");
    check_status(anira_pipeline_create(&s_pipeline, &err), &err, "anira_pipeline_create");
    const anira_model_config* variants[1] = {s_model};
    anira_backend_id candidate;
    memset(&candidate, 0, sizeof candidate);
    candidate.struct_size = sizeof candidate;
    candidate.engine = ANIRA_ENGINE_CUSTOM;
    candidate.provider = ANIRA_PROVIDER_CPU;
    candidate.engine_id = SMOKE_ENGINE_ID;
    check_status(anira_pipeline_add_inference(s_pipeline, variants, 1, &candidate, 1, &err),
                 &err,
                 "anira_pipeline_add_inference");
    check_status(anira_em_pipeline_register_js_engine(s_pipeline,
                                                      SMOKE_ENGINE_ID,
                                                      SMOKE_ENGINE_KEY,
                                                      0,
                                                      ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
                                                      &err),
                 &err,
                 "anira_em_pipeline_register_js_engine");
    check_status(anira_handler_create(s_context, s_pipeline, &s_handler, &err),
                 &err,
                 "anira_handler_create");
    check_status(anira_contract_create_hard(SMOKE_BLOCK, SMOKE_BLOCK, 48000.0, &s_contract, &err),
                 &err,
                 "anira_contract_create_hard");
    check_status(anira_contract_hard_set_budget(s_contract, ANIRA_BUDGET_EXPLICIT, 1.0),
                 NULL,
                 "anira_contract_hard_set_budget");
    check_status(anira_contract_hard_set_warmup(s_contract, ANIRA_WARMUP_FIXED, 0),
                 NULL,
                 "anira_contract_hard_set_warmup");
    check_status(anira_handler_prepare(s_handler, s_contract, &err), &err, "anira_handler_prepare");
    check_status(anira_inference_thread_create(s_context, &s_thread, &err),
                 &err,
                 "anira_inference_thread_create");
    check_status(anira_inference_thread_start(s_thread, &err),
                 &err,
                 "anira_inference_thread_start");
    s_stack = anira_em_alloc_thread_stack(SMOKE_STACK_BYTES);
    check(s_stack != NULL, "anira_em_alloc_thread_stack");
    return s_failures;
}

EMSCRIPTEN_KEEPALIVE uintptr_t smoke_thread(void) {
    return (uintptr_t)s_thread;
}
EMSCRIPTEN_KEEPALIVE uintptr_t smoke_stack_base(void) {
    return (uintptr_t)s_stack;
}
EMSCRIPTEN_KEEPALIVE uint32_t smoke_stack_bytes(void) {
    return SMOKE_STACK_BYTES;
}
/* Pushes `blocks` blocks through the handler, each waited for (the Worker's loop runs them);
   returns how many came back doubled at the handler's latency. */
EMSCRIPTEN_KEEPALIVE int smoke_thread_run_blocks(uint32_t blocks) {
    float in[SMOKE_BLOCK];
    float out[SMOKE_BLOCK];
    int64_t shape[2] = {1, SMOKE_BLOCK};
    anira_tensor in_tensor;
    anira_tensor out_tensor;
    anira_tensor_init_host(&in_tensor, in, ANIRA_DTYPE_F32, 2, shape);
    anira_tensor_init_host(&out_tensor, out, ANIRA_DTYPE_F32, 2, shape);
    const uint32_t lag = anira_handler_get_latency(s_handler, 0) / SMOKE_BLOCK;
    int matched = 0;
    for (uint32_t block = 0; block < blocks; ++block) {
        for (uint32_t i = 0; i < SMOKE_BLOCK; ++i) { in[i] = (float)(block + 1u); }
        size_t delivered = 0;
        const anira_status status = anira_handler_process_wait(s_handler,
                                                               &in_tensor,
                                                               0,
                                                               &out_tensor,
                                                               0,
                                                               &delivered,
                                                               2000.0);
        if (status != ANIRA_OK && status != ANIRA_MISSED) {
            check_status(status, NULL, "anira_handler_process_wait");
            continue;
        }
        if (block >= lag && delivered == SMOKE_BLOCK &&
            out[0] == 2.0f * (float)(block + 1u - lag) &&
            out[SMOKE_BLOCK - 1u] == 2.0f * (float)(block + 1u - lag)) {
            ++matched;
        }
    }
    return matched;
}

/* Stops the thread and waits for the Worker to leave its loop; the milliseconds that took, or
   -1 after `timeout_ms`. */
EMSCRIPTEN_KEEPALIVE double smoke_thread_stop(double timeout_ms) {
    const double start = emscripten_get_now();
    anira_inference_thread_stop(s_thread);
    while (anira_inference_thread_has_exited(s_thread) == 0) {
        if (emscripten_get_now() - start > timeout_ms) { return -1.0; }
    }
    return emscripten_get_now() - start;
}

EMSCRIPTEN_KEEPALIVE int smoke_thread_teardown(void) {
    anira_inference_thread_destroy(s_thread);
    anira_handler_destroy(s_handler);
    anira_pipeline_destroy(s_pipeline);
    anira_contract_destroy(s_contract);
    anira_model_config_destroy(s_model);
    anira_context_destroy(s_context);
    anira_context_config_destroy(s_config);
    anira_em_free_thread_stack(s_stack);
    return s_failures;
}
