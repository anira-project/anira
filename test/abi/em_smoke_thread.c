/*
 * The node test of a loop the host drives on a thread of its own (WebAssembly only): a context
 * without a pool, a user-driven inference thread (anira_inference_thread_create, no start), and
 * a Wasm Worker this test creates that runs anira_inference_thread_run_loop on it. The loop parks
 * on the wake word while the queue is empty and is woken by the enqueues and the stop. The main
 * thread pushes blocks through a Hard handler over a C engine that doubles its input and waits
 * for each (anira_handler_process_wait); then the stop: has_exited within a bound, and the
 * teardown. Every failed check prints one line; exit code 0 on success.
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
#include <emscripten/wasm_worker.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SMOKE_BLOCK 128u
#define SMOKE_BLOCKS 16u
#define SMOKE_ENGINE_ID "org.anira-test.c-double"
#define SMOKE_STACK_BYTES (1024u * 1024u)
#define SMOKE_IDLE_MS 50.0
#define SMOKE_ENTER_BOUND_MS 10000.0
#define SMOKE_STOP_BOUND_MS 100.0

static int s_failures = 0;
static anira_inference_thread* s_thread = NULL;

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

static anira_status ANIRA_CALL double_engine(const anira_engine_ctx* ctx,
                                             void* prepared,
                                             void* user_data) {
    (void)prepared;
    (void)user_data;
    const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
    float* out = anira_tensor_data_f32(&ctx->outputs[0]);
    const size_t count = anira_tensor_num_elements(&ctx->outputs[0]);
    if (in == NULL || out == NULL) { return ANIRA_ERROR_ENGINE; }
    for (size_t i = 0; i < count; ++i) { out[i] = 2.0f * in[i]; }
    return ANIRA_OK;
}

/* The Worker's one function: the loop, until the main thread stops the thread. */
static void run_loop(void) {
    anira_inference_thread_run_loop(s_thread);
}

static void idle(double ms) {
    const double until = emscripten_get_now() + ms;
    while (emscripten_get_now() < until) {}
}

int main(void) {
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
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    anira_model_config* model = NULL;
    anira_pipeline* pipeline = NULL;
    anira_custom_engine* engine = NULL;
    anira_handler* handler = NULL;
    anira_contract* contract = NULL;
    check_status(anira_context_config_create(&config, &err), &err, "anira_context_config_create");
    /* No pool: the Worker of this test is the context's one thread. */
    check_status(anira_context_config_set_threads(config, 0, ANIRA_WAIT_SPIN_BACKOFF),
                 NULL,
                 "anira_context_config_set_threads");
    check_status(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10),
                 NULL,
                 "anira_context_config_set_log_drain");
    check_status(anira_context_create(config, &context, &err), &err, "anira_context_create");
    check_status(anira_model_config_from_json(k_json, sizeof k_json - 1, NULL, &model, &err),
                 &err,
                 "anira_model_config_from_json");
    check_status(anira_pipeline_create(&pipeline, &err), &err, "anira_pipeline_create");
    const anira_model_config* variants[1] = {model};
    check_status(anira_pipeline_add_inference(pipeline, variants, 1, NULL, 0, &err),
                 &err,
                 "anira_pipeline_add_inference");
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.flags = ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL;
    desc.process = double_engine;
    check_status(anira_custom_engine_create(SMOKE_ENGINE_ID, &desc, &engine, &err),
                 &err,
                 "anira_custom_engine_create");
    check_status(anira_pipeline_add_engine(pipeline, engine, &err),
                 &err,
                 "anira_pipeline_add_engine");
    anira_custom_engine_destroy(engine);
    check_status(anira_handler_create(context, pipeline, &handler, &err),
                 &err,
                 "anira_handler_create");
    check_status(anira_contract_create_hard(SMOKE_BLOCK, SMOKE_BLOCK, 48000.0, &contract, &err),
                 &err,
                 "anira_contract_create_hard");
    check_status(anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1.0),
                 NULL,
                 "anira_contract_hard_set_budget");
    check_status(anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_NONE, 0),
                 NULL,
                 "anira_contract_hard_set_warmup");
    check_status(anira_handler_prepare(handler, contract, &err), &err, "anira_handler_prepare");
    check_status(anira_inference_thread_create(context, &s_thread, &err),
                 &err,
                 "anira_inference_thread_create");

    const emscripten_wasm_worker_t worker = emscripten_malloc_wasm_worker(SMOKE_STACK_BYTES);
    check(worker != 0, "the test's Wasm Worker");
    emscripten_wasm_worker_post_function_v(worker, run_loop);
    /* The Worker starts and enters the loop on its own, then parks on the wake word. A wait
     * refuses (ANIRA_ERROR_INVALID_STATE) while no thread is in its loop, and a Worker loads the
     * module's glue before it runs anything: how long that takes is the machine's, so wait for
     * the loop itself, not for a fixed time. */
    const double posted = emscripten_get_now();
    while (anira_em_inference_threads_in_loop() == 0 &&
           emscripten_get_now() - posted < SMOKE_ENTER_BOUND_MS) {}
    printf("the host's Worker entered its loop %.1f ms after the post\n",
           emscripten_get_now() - posted);
    check(anira_em_inference_threads_in_loop() == 1,
          "the host's Worker entered its loop within the bound");

    float in[SMOKE_BLOCK];
    float out[SMOKE_BLOCK];
    int64_t shape[2] = {1, SMOKE_BLOCK};
    anira_tensor in_tensor;
    anira_tensor out_tensor;
    anira_tensor_init_host(&in_tensor, in, ANIRA_DTYPE_F32, 2, shape);
    anira_tensor_init_host(&out_tensor, out, ANIRA_DTYPE_F32, 2, shape);
    const uint32_t lag = anira_handler_get_latency(handler, 0) / SMOKE_BLOCK;
    uint32_t matched = 0;
    for (uint32_t block = 0; block < SMOKE_BLOCKS; ++block) {
        for (uint32_t i = 0; i < SMOKE_BLOCK; ++i) { in[i] = (float)(block + 1u); }
        size_t delivered = 0;
        const anira_status status =
            anira_handler_process_wait(handler, &in_tensor, 0, &out_tensor, 0, &delivered, 2000.0);
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
    printf("blocks doubled on the host's Wasm Worker: %u of %u\n", matched, SMOKE_BLOCKS);
    check(matched + lag + 1u >= SMOKE_BLOCKS, "the host's Worker ran the blocks in its loop");

    /* Parked again; the stop's notify wakes it, and has_exited follows within the bound. */
    idle(SMOKE_IDLE_MS);
    const double stop = emscripten_get_now();
    anira_inference_thread_stop(s_thread);
    while (anira_inference_thread_has_exited(s_thread) == 0 &&
           emscripten_get_now() - stop < 1000.0) {}
    const double stop_ms = emscripten_get_now() - stop;
    printf("the Worker left its loop %.2f ms after the stop\n", stop_ms);
    check(anira_inference_thread_has_exited(s_thread) != 0 && stop_ms < SMOKE_STOP_BOUND_MS,
          "has_exited within the bound of the stop");
    emscripten_terminate_wasm_worker(worker);

    anira_inference_thread_destroy(s_thread);
    anira_handler_destroy(handler);
    anira_pipeline_destroy(pipeline);
    anira_contract_destroy(contract);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    anira_context_config_destroy(config);
    printf("%s: %d failed check(s)\n", s_failures == 0 ? "PASSED" : "FAILED", s_failures);
    return s_failures == 0 ? 0 : 1;
}
