/*
 * The node test of out of memory as a status (WebAssembly only): a consumer of the archive with
 * a fixed memory (512 MiB, ALLOW_MEMORY_GROWTH=0, as AniraWeb) and nothing of its own about
 * malloc: the library's INTERFACE link options give it ABORTING_MALLOC=0, so malloc returns NULL
 * and operator new throws std::bad_alloc, which the C entries' firewall turns into
 * ANIRA_ERROR_OUT_OF_MEMORY. With the heap filled, a handler on the model config of argv[1] (the
 * built-in ONNX Runtime engine; NODERAWFS reads it and its model file) must be refused with a
 * status, not abort the module; with the heap freed it prepares, and the heap is as free as
 * before. On the way it reports what the context and one instance of the model take.
 * Exit code 0 on success.
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define OOM_MIB (1024.0 * 1024.0)
#define OOM_MAX_BLOCKS 65536

static int s_failures = 0;
static void* s_blocks[OOM_MAX_BLOCKS];
static size_t s_num_blocks = 0;

static void check(int ok, const char* what) {
    if (!ok) {
        printf("FAILED: %s\n", what);
        ++s_failures;
    }
}

/* Allocates blocks of halving sizes until not even 64 KiB is left; returns their sum. */
static size_t fill(void) {
    size_t total = 0;
    for (size_t size = (size_t)64 << 20; size >= ((size_t)64 << 10); size /= 2) {
        while (s_num_blocks < OOM_MAX_BLOCKS) {
            void* block = malloc(size);
            if (block == NULL) { break; }
            s_blocks[s_num_blocks++] = block;
            total += size;
        }
    }
    return total;
}

static void release(void) {
    while (s_num_blocks > 0) { free(s_blocks[--s_num_blocks]); }
}

/* What the heap has free now: fill, count, release. */
static size_t free_heap(void) {
    const size_t total = fill();
    release();
    return total;
}

/* A context with `threads` inference threads (anira's Wasm Workers). */
static anira_context* context_of(uint32_t threads, anira_error* err) {
    anira_context_config* config = NULL;
    anira_context* context = NULL;
    check(anira_context_config_create(&config, err) == ANIRA_OK, "anira_context_config_create");
    check(anira_context_config_set_threads(config, threads, ANIRA_WAIT_SPIN_BACKOFF) == ANIRA_OK,
          "anira_context_config_set_threads");
    check(anira_context_create(config, &context, err) == ANIRA_OK, "anira_context_create");
    anira_context_config_destroy(config);
    return context;
}

static anira_status handler_on(anira_context* context,
                               anira_model_config* model,
                               const anira_contract* contract,
                               anira_handler** out,
                               anira_error* err) {
    *out = NULL;
    anira_pipeline* pipeline = NULL;
    anira_status status = anira_pipeline_create(&pipeline, err);
    const anira_model_config* variants[1] = {model};
    if (status == ANIRA_OK) {
        status = anira_pipeline_add_inference(pipeline, variants, 1, NULL, 0, err);
    }
    if (status == ANIRA_OK) { status = anira_handler_create(context, pipeline, out, err); }
    if (status == ANIRA_OK) { status = anira_handler_prepare(*out, contract, err); }
    if (status != ANIRA_OK && *out != NULL) {
        anira_handler_destroy(*out);
        *out = NULL;
    }
    anira_pipeline_destroy(pipeline);
    return status;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: anira_em_oom <model.json>\n");
        return 2;
    }
    anira_error err = ANIRA_ERROR_INIT;
    const size_t before_context = free_heap();
    /* A pool of two: a model's shared instances are as many as the pool has threads at most. */
    anira_context* context = context_of(2, &err);
    anira_model_config* model = NULL;
    check(anira_model_config_from_json_file(argv[1], &model, &err) == ANIRA_OK, err.message);
    anira_contract* contract = NULL;
    check(anira_contract_create_hard(128.0, 128.0, 48000.0, &contract, &err) == ANIRA_OK,
          "anira_contract_create_hard");
    anira_contract_hard_set_budget(contract, ANIRA_BUDGET_EXPLICIT, 1000.0);
    anira_contract_hard_set_warmup(contract, ANIRA_WARMUP_NONE, 0);
    if (s_failures != 0) { return 1; }

    /* What the core and a handler of one, then two, instances of the model take, each handler
     * measured from just before it: its pool's Wasm Workers stopped while still booting are
     * freed by a main-thread timer (InferenceThread::stop), which this synchronous main never
     * yields to, so their 4 MiB stacks stay counted until the process ends. */
    anira_handler* handler = NULL;
    const size_t with_context = free_heap();
    /* A first handler sets up what stays for the process (ONNX Runtime's environment). */
    check(handler_on(context, model, contract, &handler, &err) == ANIRA_OK, err.message);
    anira_handler_destroy(handler);
    double handler_takes[3] = {0, 0, 0};
    for (uint32_t instances = 1; instances <= 2; ++instances) {
        anira_model_config_set_max_instances(model, instances);
        const size_t base = free_heap();
        const anira_status status = handler_on(context, model, contract, &handler, &err);
        check(status == ANIRA_OK, err.message);
        handler_takes[instances] = (double)base - (double)free_heap();
        anira_handler_destroy(handler);
    }
    printf(
        "free heap: %.1f MiB before the context, %.1f MiB with it (the core and the config "
        "take %.1f MiB)\n",
        (double)before_context / OOM_MIB,
        (double)with_context / OOM_MIB,
        (double)(before_context - with_context) / OOM_MIB);
    printf(
        "a handler of %s takes %.1f MiB with one instance, %.1f MiB with two (both with the "
        "pool's two Worker stacks): %.1f MiB per instance\n",
        argv[1],
        handler_takes[1] / OOM_MIB,
        handler_takes[2] / OOM_MIB,
        (handler_takes[2] - handler_takes[1]) / OOM_MIB);

    /* The heap full on the pool's context: refused with the status whichever allocation runs
     * out first (its pool stopped with its last handler, so a prepare that gets that far starts
     * it again: a Worker's 4 MiB region, as oom.html sees; here a smaller one fails first). */
    {
        fill();
        const anira_status refused = handler_on(context, model, contract, &handler, &err);
        release();
        printf("with a pool and the heap full the handler is refused: %s (%s)\n",
               anira_status_string(refused),
               err.message);
        check(refused == ANIRA_ERROR_OUT_OF_MEMORY,
              "with a pool the refusal is ANIRA_ERROR_OUT_OF_MEMORY");
        check(handler_on(context, model, contract, &handler, &err) == ANIRA_OK,
              "with a pool and the memory freed a handler prepares");
        anira_handler_destroy(handler);
    }
    anira_context_destroy(context);

    /* The refusals on a context without a pool, whose handlers leave nothing to a timer. */
    context = context_of(0, &err);
    anira_model_config_set_max_instances(model, 1);
    check(handler_on(context, model, contract, &handler, &err) == ANIRA_OK, err.message);
    anira_handler_destroy(handler);
    const size_t settled = free_heap();

    /* The heap full: refused with a status, three times; with the memory back a handler
     * prepares, and the heap is as free as before each time. */
    size_t after[3] = {0, 0, 0};
    for (int round = 0; round < 3; ++round) {
        const size_t filled = fill();
        const anira_status refused = handler_on(context, model, contract, &handler, &err);
        release();
        printf("round %d: with %.1f MiB filled the handler is refused: %s (%s)\n",
               round,
               (double)filled / OOM_MIB,
               anira_status_string(refused),
               err.message);
        check(refused == ANIRA_ERROR_OUT_OF_MEMORY, "the refusal is ANIRA_ERROR_OUT_OF_MEMORY");
        check(handler_on(context, model, contract, &handler, &err) == ANIRA_OK,
              "with the memory freed a handler prepares");
        anira_handler_destroy(handler);
        after[round] = free_heap();
        printf("round %d: free heap after %.1f MiB (%.1f MiB before the first refusal)\n",
               round,
               (double)after[round] / OOM_MIB,
               (double)settled / OOM_MIB);
    }
    check(after[2] + ((size_t)1 << 20) >= after[0], "a refused handler leaves nothing behind");

    anira_contract_destroy(contract);
    anira_model_config_destroy(model);
    anira_context_destroy(context);
    printf("%s: %d failed check(s)\n", s_failures == 0 ? "PASSED" : "FAILED", s_failures);
    return s_failures == 0 ? 0 : 1;
}
