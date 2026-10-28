/*
 * The node smoke test of anira/anira_em.h (WebAssembly only): one module instance, no Worker.
 * em_smoke.pre.js installs Module.anira before main runs; this file checks the hook version,
 * gives the instance a stack and thread-local block of its own (anira_em_init_instance), runs a
 * JavaScript stage (pre_process, post_process, reset, init, prepare, unprepare, release) and a
 * JavaScript engine (process, query) on one Hard handler, driving the inference queue itself
 * with anira_inference_thread_execute, catches a throwing hook in the trampoline, and receives
 * a log record through the JavaScript log hook. It also holds the thread-local budget of the
 * link: the block anira_em_alloc_thread_stack carves stays within 1024 bytes. Exit code 0 on
 * success; every failed check prints one line.
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/anira_em.h>
#include <emscripten/em_js.h>
#include <emscripten/em_macros.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SMOKE_BLOCK 128u
#define SMOKE_BLOCKS 16u
#define SMOKE_TLS_BUDGET 1024u
#define SMOKE_STAGE_KEY 3u
#define SMOKE_THROWING_KEY 99u
#define SMOKE_ENGINE_KEY 7u
#define SMOKE_ENGINE_ID "org.anira-test.js-double"
#define SMOKE_QUERY_KEY 8u
#define SMOKE_QUERY_ID "org.anira-test.js-query"

/* The EM_JS macros end their own declarations: no semicolon follows them (-pedantic). */
EM_JS_DEPS(anira_em_smoke, "$UTF8ToString")

/* The pre-js reads the records through the offsets the compiler gives, never by hand. */
EM_JS(void,
      smoke_set_layout,
      (uint32_t ctx_inputs, uint32_t ctx_outputs, uint32_t record_level, uint32_t record_message),
      {
          Module.smoke.layout = {
              ctxInputs: ctx_inputs,
              ctxOutputs: ctx_outputs,
              recordLevel: record_level,
              recordMessage: record_message
          };
      })
EM_JS(uint32_t, smoke_hook_version, (void), { return Module.anira.version >>> 0; })
EM_JS(uint32_t, smoke_count, (const char* name), {
    return Module.smoke.counts[UTF8ToString(name)] >>> 0;
})

static int s_failures = 0;
static _Thread_local int t_marker = 0;

/* Out of line: within one function the compiler may keep a thread_local's address, which is
   invariant on a thread but moves here, at anira_em_init_instance. */
__attribute__((noinline)) static void set_marker(int value) {
    *(volatile int*)&t_marker = value;
}
__attribute__((noinline)) static int get_marker(void) {
    return *(volatile int*)&t_marker;
}

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

/* The instance gets the stack region's thread-local block; its thread_local variables start
   over from their initial values there. */
static void check_instance_tls(void) {
    const size_t tls_size = __builtin_wasm_tls_size();
    printf("thread-local block of the link: %zu bytes (align %zu)\n",
           tls_size,
           (size_t)__builtin_wasm_tls_align());
    check(tls_size <= SMOKE_TLS_BUDGET, "the thread-local block stays within its 1024-byte budget");
    set_marker(42);
    void* stack = anira_em_alloc_thread_stack(0);
    check(stack != NULL, "anira_em_alloc_thread_stack(0) allocates");
    if (stack == NULL) { return; }
    check(((uintptr_t)stack % 16u) == 0, "the stack region is 16-byte aligned");
    check(anira_em_init_instance(NULL, 0) == ANIRA_ERROR_INVALID_ARGUMENT,
          "anira_em_init_instance refuses a NULL base");
    check(anira_em_init_instance(stack, 0) == ANIRA_OK, "anira_em_init_instance");
    const uintptr_t base = (uintptr_t)__builtin_wasm_tls_base();
    const uintptr_t top = (uintptr_t)stack + (size_t)4 * 1024 * 1024;
    check(base >= top && base < top + 64u + tls_size,
          "the thread-local block sits above the stack's top");
    check(get_marker() == 0, "a thread_local starts from its initial value in the new block");
    /* The region stays: it is this instance's thread-local block until the process exits. */
}

/* A throwing hook is caught in the trampoline and fails the call; nothing unwinds into C. */
static void check_throwing_hook(void) {
    anira_stage_desc desc = ANIRA_STAGE_DESC_INIT;
    check(anira_em_stage_desc_init(NULL, 0, ANIRA_EM_HOOK_PRE, 0) == ANIRA_ERROR_INVALID_ARGUMENT,
          "anira_em_stage_desc_init refuses a NULL desc");
    check(anira_em_stage_desc_init(&desc, 0, 512u, 0) == ANIRA_ERROR_INVALID_ARGUMENT,
          "anira_em_stage_desc_init refuses an undefined bit");
    check_status(anira_em_stage_desc_init(&desc, SMOKE_THROWING_KEY, ANIRA_EM_HOOK_PRE, 0),
                 NULL,
                 "anira_em_stage_desc_init");
    check(desc.pre_process != NULL && desc.post_process == NULL && desc.release != NULL,
          "one slot per bit, the release slot always");
    check(desc.flags == 0u, "no promise unless one is stated");
    desc.release(desc.user_data);
    check_status(anira_em_stage_desc_init(&desc,
                                          SMOKE_THROWING_KEY,
                                          ANIRA_EM_HOOK_PRE,
                                          ANIRA_STAGE_FLAG_REALTIME_PRE_POST),
                 NULL,
                 "anira_em_stage_desc_init with a promise");
    check(desc.flags == ANIRA_STAGE_FLAG_REALTIME_PRE_POST, "the stated promise, as given");
    anira_stage_ctx ctx;
    memset(&ctx, 0, sizeof ctx);
    check(desc.pre_process(&ctx, NULL, desc.user_data) == ANIRA_ERROR_UNKNOWN,
          "a throwing stage hook fails its call with ANIRA_ERROR_UNKNOWN");
    desc.release(desc.user_data);
    check(smoke_count("thrown") == 1u, "the throwing hook ran once");
}

static anira_model_config* smoke_model(anira_error* err) {
    static const char k_json[] =
        "{\"models\": [{\"engine\": \"" SMOKE_ENGINE_ID
        "\", \"path\": \"never-opened.bin\"}],"
        " \"inputs\": [{\"name\": \"in\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}],"
        " \"outputs\": [{\"name\": \"out\", \"dtype\": \"float32\", \"role\": \"streamed\","
        " \"axes\": [[\"batch\", 1], [\"channel\", 1], [\"time\", 128]],"
        " \"window\": {\"min\": 128, \"max\": 128}}]}";
    anira_model_config* model = NULL;
    check_status(anira_model_config_from_json(k_json, sizeof k_json - 1, NULL, &model, err),
                 err,
                 "anira_model_config_from_json");
    return model;
}

/* A JavaScript engine with a providers list, built with anira_em_engine_desc_init: the
   capabilities ask its query, whose mask keeps the CPU path and drops the other provider. */
static void check_query_engine(anira_pipeline* pipeline, const anira_context* context) {
    static const char* const k_providers[2] = {"cpu", "org.anira-test.never-usable"};
    anira_error err = ANIRA_ERROR_INIT;
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    check(anira_em_engine_desc_init(&desc, SMOKE_QUERY_KEY, 4u, 0) == ANIRA_ERROR_INVALID_ARGUMENT,
          "anira_em_engine_desc_init refuses an undefined bit");
    check_status(anira_em_engine_desc_init(&desc,
                                           SMOKE_QUERY_KEY,
                                           ANIRA_EM_ENGINE_HOOK_QUERY,
                                           ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL),
                 NULL,
                 "anira_em_engine_desc_init");
    check(desc.flags == ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL, "the engine's flags, as given");
    check(desc.process != NULL && desc.query != NULL && desc.reset == NULL && desc.init == NULL &&
              desc.release != NULL,
          "process always, query per bit, the release slot always, init never");
    desc.providers = k_providers;
    desc.num_providers = 2;
    anira_custom_engine* engine = NULL;
    check_status(anira_custom_engine_create(SMOKE_QUERY_ID, &desc, &engine, &err),
                 &err,
                 "anira_custom_engine_create");
    check_status(anira_pipeline_add_engine(pipeline, engine, &err),
                 &err,
                 "anira_pipeline_add_engine");
    anira_custom_engine_destroy(engine);
    uint32_t count = 0;
    check_status(anira_pipeline_capabilities_backends(pipeline,
                                                      context,
                                                      sizeof(anira_backend_id),
                                                      &count,
                                                      NULL),
                 NULL,
                 "anira_pipeline_capabilities_backends");
    anira_backend_id rows[32];
    count = count < 32u ? count : 32u;
    check_status(anira_pipeline_capabilities_backends(pipeline,
                                                      context,
                                                      sizeof(anira_backend_id),
                                                      &count,
                                                      rows),
                 NULL,
                 "anira_pipeline_capabilities_backends");
    uint32_t query_rows = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (rows[i].engine_id != NULL && strcmp(rows[i].engine_id, SMOKE_QUERY_ID) == 0) {
            ++query_rows;
            check(rows[i].provider == ANIRA_PROVIDER_CPU, "the query's mask keeps the CPU path");
        }
    }
    check(query_rows == 1u, "one row of the query engine: the provider whose bit is set");
    check(smoke_count("query") >= 1u, "the engine's query crossed at the capabilities call");
}

/* A pre_process hook without ANIRA_STAGE_FLAG_REALTIME_PRE_POST: under a Hard contract it runs
   on the driving thread, so prepare refuses the stage instead of trusting a promise nobody made. */
static void check_unpromised_stage(anira_context* context, const anira_contract* contract) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* model = smoke_model(&err);
    anira_pipeline* pipeline = NULL;
    check_status(anira_pipeline_create(&pipeline, &err), &err, "anira_pipeline_create");
    const anira_model_config* variants[1] = {model};
    anira_backend_id candidate;
    memset(&candidate, 0, sizeof candidate);
    candidate.struct_size = sizeof candidate;
    candidate.engine = ANIRA_ENGINE_CUSTOM;
    candidate.provider = ANIRA_PROVIDER_CPU;
    candidate.engine_id = SMOKE_ENGINE_ID;
    check_status(anira_pipeline_add_inference(pipeline, variants, 1, &candidate, 1, &err),
                 &err,
                 "anira_pipeline_add_inference");
    check_status(anira_em_pipeline_register_js_engine(pipeline,
                                                      SMOKE_ENGINE_ID,
                                                      SMOKE_ENGINE_KEY,
                                                      0,
                                                      ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL,
                                                      &err),
                 &err,
                 "anira_em_pipeline_register_js_engine");
    check_status(
        anira_em_pipeline_add_js_stage(pipeline, SMOKE_STAGE_KEY, ANIRA_EM_HOOK_PRE, 0, &err),
        &err,
        "anira_em_pipeline_add_js_stage without a promise");
    anira_handler* handler = NULL;
    check_status(anira_handler_create(context, pipeline, &handler, &err),
                 &err,
                 "anira_handler_create");
    err = ANIRA_ERROR_INIT;
    check(anira_handler_prepare(handler, contract, &err) == ANIRA_ERROR_CONFIG,
          "a Hard prepare refuses a pre_process hook without ANIRA_STAGE_FLAG_REALTIME_PRE_POST");
    anira_handler_destroy(handler);
    anira_pipeline_destroy(pipeline);
    anira_model_config_destroy(model);
}

/* A JavaScript stage and a JavaScript engine (output = 2 x input) on one Hard handler. */
static void check_pipeline(void) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    check_status(anira_context_config_create(&config, &err), &err, "anira_context_config_create");
    check_status(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10),
                 NULL,
                 "anira_context_config_set_log_drain");
    check_status(anira_context_create(config, &context, &err), &err, "anira_context_create");

    anira_model_config* model = smoke_model(&err);
    anira_pipeline* pipeline = NULL;
    check_status(anira_pipeline_create(&pipeline, &err), &err, "anira_pipeline_create");
    const anira_model_config* variants[1] = {model};
    anira_backend_id candidate;
    memset(&candidate, 0, sizeof candidate);
    candidate.struct_size = sizeof candidate;
    candidate.engine = ANIRA_ENGINE_CUSTOM;
    candidate.provider = ANIRA_PROVIDER_CPU;
    candidate.engine_id = SMOKE_ENGINE_ID;
    check_status(anira_pipeline_add_inference(pipeline, variants, 1, &candidate, 1, &err),
                 &err,
                 "anira_pipeline_add_inference");
    check(anira_em_pipeline_register_js_engine(pipeline, "no-dot", SMOKE_ENGINE_KEY, 0, 0, &err) ==
              ANIRA_ERROR_INVALID_ARGUMENT,
          "anira_em_pipeline_register_js_engine refuses an id without a '.'");
    check_status(anira_em_pipeline_register_js_engine(
                     pipeline,
                     SMOKE_ENGINE_ID,
                     SMOKE_ENGINE_KEY,
                     ANIRA_EM_ENGINE_HOOK_QUERY,
                     ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL | ANIRA_ENGINE_FLAG_REALTIME_SAFE,
                     &err),
                 &err,
                 "anira_em_pipeline_register_js_engine");
    check_status(anira_em_pipeline_add_js_stage(pipeline,
                                                SMOKE_STAGE_KEY,
                                                ANIRA_EM_HOOK_PRE | ANIRA_EM_HOOK_POST |
                                                    ANIRA_EM_HOOK_RESET | ANIRA_EM_HOOK_INIT |
                                                    ANIRA_EM_HOOK_PREPARE |
                                                    ANIRA_EM_HOOK_UNPREPARE | ANIRA_EM_HOOK_RELEASE,
                                                ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
                                                &err),
                 &err,
                 "anira_em_pipeline_add_js_stage");
    check(anira_em_pipeline_add_js_stage(pipeline,
                                         SMOKE_STAGE_KEY,
                                         ANIRA_EM_HOOK_PRE,
                                         ANIRA_STAGE_FLAG_REALTIME_PRE_POST,
                                         &err) == ANIRA_ERROR_INVALID_STATE,
          "a second JavaScript stage is refused");
    check_query_engine(pipeline, context);

    anira_handler* handler = NULL;
    check_status(anira_handler_create(context, pipeline, &handler, &err),
                 &err,
                 "anira_handler_create");
    anira_contract* contract = NULL;
    check_status(anira_contract_create_hard(SMOKE_BLOCK, SMOKE_BLOCK, 48000.0, &contract, &err),
                 &err,
                 "anira_contract_create_hard");
    check_status(anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1.0),
                 NULL,
                 "anira_contract_hard_set_budget");
    check_status(anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_FIXED, 0),
                 NULL,
                 "anira_contract_hard_set_warmup");
    check_status(anira_handler_prepare(handler, contract, &err), &err, "anira_handler_prepare");
    check(smoke_count("init") == 1u, "the stage's init crossed once");
    check(smoke_count("prepare") == 1u, "the stage's prepare crossed once");

    anira_inference_thread* thread = NULL;
    check_status(anira_inference_thread_create(context, &thread, &err),
                 &err,
                 "anira_inference_thread_create");
    const uint32_t latency = anira_handler_get_latency(handler, 0);
    printf("handler latency: %u samples\n", latency);
    float in[SMOKE_BLOCK];
    float out[SMOKE_BLOCK];
    int64_t shape[2] = {1, SMOKE_BLOCK};
    anira_tensor in_tensor;
    anira_tensor out_tensor;
    anira_tensor_init_host(&in_tensor, in, ANIRA_DTYPE_F32, 2, shape);
    anira_tensor_init_host(&out_tensor, out, ANIRA_DTYPE_F32, 2, shape);
    int matched = 0;
    for (uint32_t block = 0; block < SMOKE_BLOCKS; ++block) {
        for (uint32_t i = 0; i < SMOKE_BLOCK; ++i) { in[i] = (float)(block + 1u); }
        size_t delivered = 0;
        check_status(anira_handler_process(handler, &in_tensor, 0, &out_tensor, 0, &delivered),
                     NULL,
                     "anira_handler_process");
        /* The output of this block is the input of `latency` samples ago, doubled. */
        const uint32_t lag = latency / SMOKE_BLOCK;
        if (block >= lag + 1u && delivered == SMOKE_BLOCK &&
            out[0] == 2.0f * (float)(block + 1u - lag) &&
            out[SMOKE_BLOCK - 1u] == 2.0f * (float)(block + 1u - lag)) {
            ++matched;
        }
        while (anira_inference_thread_execute(thread) != 0u) {}
    }
    printf("blocks doubled by the JavaScript engine: %d of %u\n", matched, SMOKE_BLOCKS);
    check(matched > 0, "the JavaScript engine's output reached the host");
    check(smoke_count("process") >= SMOKE_BLOCKS - 1u, "the engine's process crossed per block");
    check(smoke_count("pre") >= SMOKE_BLOCKS - 1u, "the stage's pre_process crossed per block");
    check(smoke_count("post") >= SMOKE_BLOCKS - 2u, "the stage's post_process crossed per block");
    check(smoke_count("reset") == 1u, "the stage's reset crossed once, for the first chunk");

    anira_inference_thread_destroy(thread);
    anira_handler_destroy(handler);
    check(smoke_count("unprepare") == 1u, "the stage's unprepare crossed at anira_handler_destroy");
    anira_pipeline_destroy(pipeline);
    check(smoke_count("release") == 1u, "the stage's release crossed with the last carrier");
    check_unpromised_stage(context, contract);
    anira_contract_destroy(contract);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    anira_context_config_destroy(config);
}

/* A record of the core reaches Module.anira.log while the hook is registered, none after. */
static void check_log_hook(void) {
    anira_em_set_log_hook(1u);
    anira_em_set_log_hook(1u); /* a repeated call changes nothing */
    anira_log(ANIRA_LOG_ERROR, "anira.smoke", "smoke: a record for the JavaScript log hook");
    check(smoke_count("log") == 1u, "the record reached the hook once");
    anira_em_set_log_hook(0u);
    anira_log(ANIRA_LOG_ERROR, "anira.smoke", "smoke: a record after the hook was removed");
    check(smoke_count("log") == 1u, "no record reaches a removed hook");
}

int main(void) {
    smoke_set_layout((uint32_t)offsetof(anira_engine_ctx, inputs),
                     (uint32_t)offsetof(anira_engine_ctx, outputs),
                     (uint32_t)offsetof(anira_log_record, level),
                     (uint32_t)offsetof(anira_log_record, message));
    check(anira_em_hooks_version() == ANIRA_EM_HOOKS_VERSION, "anira_em_hooks_version");
    check(smoke_hook_version() == anira_em_hooks_version(),
          "Module.anira.version is the module's hook version");
    check_instance_tls();
    check_throwing_hook();
    check_pipeline();
    check_log_hook();
    printf("%s: %d failed check(s)\n", s_failures == 0 ? "PASSED" : "FAILED", s_failures);
    return s_failures == 0 ? 0 : 1;
}
