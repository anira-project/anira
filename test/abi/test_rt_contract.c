/*
 * The real-time contract as a consumer's compiler sees it (gate 6 of
 * docs/anira-v3-architecture.md, section 6a; the seed of the translation unit that gate
 * compiles). One ANIRA_NONBLOCKING function calls every [callback-safe] entry point of
 * anira/abi/tensor.h and anira/abi/draft/tensor_platform.h. Under clang the target adds
 * -Werror=function-effects, so an entry whose generated declaration lost ANIRA_NONBLOCKING no
 * longer compiles here: clang cannot infer the effect of a function defined in another
 * translation unit, the declaration is all it has. Every other compiler sees plain C11 (the
 * macro is empty there) under the strict flags of the header gates. Nothing runs and nothing
 * links: this is an OBJECT, the bodies are test_Tensor.cpp's business.
 *
 * Deliberately absent: anira_tensor_init_dlpack ([main-thread], it can fail with a message)
 * and anira_sync_token_reset / anira_sync_token_dup ([thread-safe, !audio-thread]: close and
 * dup are system calls). Calling one of them from the function below is a compile error under
 * clang, which is the point. A second ANIRA_NONBLOCKING function calls the six [driver-thread]
 * Hard entries over host tensors of anira/abi/handler.h, which is what a host's audio callback
 * does; their four _wait twins are [any-thread, blocking] and absent for the same reason. A
 * third one is a backup function of ANIRA_MISS_CALLBACK, which reads and fills its tensors
 * through the [callback-safe] accessors and is handed to anira_contract_hard_set_miss_fn: the
 * conversion to anira_miss_fn is clean only because the function is declared
 * ANIRA_NONBLOCKING itself (clang refuses to add the attribute through a conversion). A
 * fourth one is a phase callback of anira/abi/stage.h: it calls the ten ring accessors and the
 * two default bodies, all [callback-safe], and lands in the four anira_stage_fn slots of a
 * descriptor, which needs the attribute for the same reason. The file grows with the
 * handler's [callback-safe] entries.
 */
#include <anira/abi/config.h>
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <stddef.h>
#include <stdint.h>

/* Exported on purpose, so that the object holds a symbol a linker would see. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) */
size_t anira_rt_contract_tensor(anira_tensor* tensor,
                                void* memory,
                                const anira_sync_token* fence,
                                const int64_t* shape) ANIRA_NONBLOCKING;
size_t anira_rt_contract_tensor(anira_tensor* tensor,
                                void* memory,
                                const anira_sync_token* fence,
                                const int64_t* shape) ANIRA_NONBLOCKING {
    size_t total = 0;
    /* anira/abi/tensor.h: the nine nonblocking factories. */
    anira_tensor_init_host(tensor, memory, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_pinned(tensor, memory, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_host_planar(tensor, NULL, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_cuda(tensor, memory, 0, NULL, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_gl_buffer(tensor, 0u, 0u, NULL, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_vulkan(tensor, 0u, 0u, 0u, 0u, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_opaque_fd(tensor, -1, 0u, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_wgpu_buffer(tensor, memory, 0u, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_dmabuf(tensor, -1, 0u, 0u, -1, ANIRA_DTYPE_F32, 1u, shape);
    /* anira/abi/draft/tensor_platform.h: the four draft factories. */
    anira_tensor_init_metal(tensor, memory, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_iosurface(tensor, memory, 0u, fence, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_ahardwarebuffer(tensor, memory, -1, ANIRA_DTYPE_F32, 1u, shape);
    anira_tensor_init_d3d12(tensor, memory, NULL, fence, ANIRA_DTYPE_F32, 1u, shape);
    /* The accessors and anira_sizeof. */
    total += anira_tensor_data_f32(tensor) != NULL ? 1u : 0u;
    total += anira_tensor_data(tensor, ANIRA_DTYPE_F32) != NULL ? 1u : 0u;
    total += anira_tensor_plane(tensor, 0u, ANIRA_DTYPE_F32) != NULL ? 1u : 0u;
    total += anira_tensor_num_elements(tensor);
    total += anira_tensor_extent(tensor, 0u);
    total += anira_sizeof(ANIRA_STRUCT_TENSOR);
    return total;
}

/* anira/abi/handler.h: the six nonblocking Hard entries over host tensors, as a host's audio
   callback calls them. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) */
size_t anira_rt_contract_hard(anira_handler* handler,
                              const anira_tensor* inputs,
                              const anira_tensor* outputs,
                              size_t* delivered) ANIRA_NONBLOCKING;
size_t anira_rt_contract_hard(anira_handler* handler,
                              const anira_tensor* inputs,
                              const anira_tensor* outputs,
                              size_t* delivered) ANIRA_NONBLOCKING {
    size_t total = 0;
    total += anira_handler_process(handler, inputs, outputs, 0u, delivered) == ANIRA_OK ? 1u : 0u;
    total += anira_handler_process_multi(handler, inputs, 1u, outputs, 1u, delivered) == ANIRA_OK
                 ? 1u
                 : 0u;
    total += anira_handler_push_data(handler, inputs, 0u) == ANIRA_OK ? 1u : 0u;
    total += anira_handler_push_data_multi(handler, inputs, 1u) == ANIRA_OK ? 1u : 0u;
    total += anira_handler_pop_data(handler, outputs, 0u, delivered) == ANIRA_OK ? 1u : 0u;
    total += anira_handler_pop_data_multi(handler, outputs, 1u, delivered) == ANIRA_OK ? 1u : 0u;
    /* The Static entries are [driver-thread] and nonblocking like the Hard entries. */
    total += anira_handler_set_static_input(handler, 1u, inputs) == ANIRA_OK ? 1u : 0u;
    total += anira_handler_get_static_output(handler, 1u, outputs) == ANIRA_OK ? 1u : 0u;
    return total;
}

/* anira/abi/config.h: a backup function of ANIRA_MISS_CALLBACK as a host writes one. It fills
   channel 0 of every requested float32 output with the first sample of input 0 and declines a
   block it cannot fill. */
static anira_status ANIRA_CALL anira_rt_contract_miss(anira_handler* handler,
                                                      const anira_tensor* inputs,
                                                      uint32_t num_inputs,
                                                      const anira_tensor* outputs,
                                                      uint32_t num_outputs,
                                                      void* user_data) ANIRA_NONBLOCKING {
    float first = 0.0f;
    uint32_t slot = 0;
    (void)user_data;
    if (anira_handler_rt_error(handler) != ANIRA_OK) { return ANIRA_ERROR_INVALID_STATE; }
    if (num_inputs > 0u && anira_tensor_extent(&inputs[0], 1u) > 0u) {
        const float* in = (inputs[0].flags & (uint32_t)ANIRA_TENSOR_PLANAR) != 0u
                              ? (const float*)anira_tensor_plane(&inputs[0], 0u, ANIRA_DTYPE_F32)
                              : anira_tensor_data_f32(&inputs[0]);
        if (in != NULL) { first = in[0]; }
    }
    for (slot = 0; slot < num_outputs; ++slot) {
        const size_t samples = anira_tensor_extent(&outputs[slot], 1u);
        const int64_t step = outputs[slot].strides[1] != 0 ? outputs[slot].strides[1] : 1;
        float* out = (outputs[slot].flags & (uint32_t)ANIRA_TENSOR_PLANAR) != 0u
                         ? (float*)anira_tensor_plane(&outputs[slot], 0u, ANIRA_DTYPE_F32)
                         : anira_tensor_data_f32(&outputs[slot]);
        size_t i = 0;
        if (samples == 0u) { continue; }
        if (out == NULL) { return ANIRA_ERROR_NOT_SUPPORTED; }
        for (i = 0; i < samples; ++i) { out[(int64_t)i * step] = first; }
    }
    return ANIRA_OK;
}

/* anira/abi/stage.h: a phase callback as a host writes one. In pre_process it moves one hop of
   every Streamed input by hand, through every accessor a stage has; in post_process it pushes
   the outputs; anything else is left to the default bodies. */
static anira_status ANIRA_CALL anira_rt_contract_stage(const anira_stage_ctx* ctx,
                                                       void* user_data) ANIRA_NONBLOCKING {
    uint32_t slot = 0;
    (void)user_data;
    if (ctx->phase == (uint32_t)ANIRA_PHASE_PRE_PROCESS) {
        for (slot = 0; slot < ctx->num_inputs; ++slot) {
            anira_ring* ring = ctx->input_rings[slot];
            const anira_dtype dtype = anira_ring_dtype(ring);
            void* data = anira_tensor_data(&ctx->model_inputs[slot], dtype);
            const size_t elements = anira_tensor_num_elements(&ctx->model_inputs[slot]);
            uint32_t channel = 0;
            if (ring == NULL || data == NULL) { continue; }
            for (channel = 0; channel < anira_ring_num_channels(ring); ++channel) {
                const size_t fresh = anira_ring_available(ring, channel);
                const size_t past = anira_ring_available_past(ring, channel);
                if (fresh + past < elements) {
                    (void)anira_ring_discard(ring, channel, fresh);
                } else if (past > 0u) {
                    (void)anira_ring_pop_windows(ring, channel, data, dtype, fresh, past, 0u, 1u);
                } else {
                    (void)anira_ring_peek_past_block(ring, channel, data, dtype, 0u);
                    (void)anira_ring_pop_block(ring, channel, data, dtype, fresh);
                }
            }
        }
        return anira_stage_default_pre_process(ctx);
    }
    if (ctx->phase == (uint32_t)ANIRA_PHASE_POST_PROCESS) {
        for (slot = 0; slot < ctx->num_outputs; ++slot) {
            anira_ring* ring = ctx->output_rings[slot];
            const anira_dtype dtype = anira_ring_dtype(ring);
            const void* data = anira_tensor_data(&ctx->model_outputs[slot], dtype);
            if (ring == NULL || data == NULL) { continue; }
            (void)anira_ring_push_block(ring, 0u, data, dtype, 0u);
            (void)anira_ring_push_fill(ring, 0u, data, dtype, 0u);
        }
        return anira_stage_default_post_process(ctx);
    }
    return ANIRA_OK;
}

/* anira_pipeline_add_stage is [main-thread]: a plain function fills the descriptor. The four
   phase slots take the nonblocking function as it is. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) */
anira_status anira_rt_contract_add_stage(anira_pipeline* pipeline, void* user_data);
anira_status anira_rt_contract_add_stage(anira_pipeline* pipeline, void* user_data) {
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.user_data = user_data;
    stage.name = "rt-contract";
    stage.pre_process = anira_rt_contract_stage;
    stage.post_process = anira_rt_contract_stage;
    stage.before_inference = anira_rt_contract_stage;
    stage.after_inference = anira_rt_contract_stage;
    return anira_pipeline_add_stage(pipeline, &stage, NULL);
}

/* The setter is [main-thread]: a plain function hands the pair over. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) */
anira_status anira_rt_contract_set_miss(anira_contract* contract, void* user_data);
anira_status anira_rt_contract_set_miss(anira_contract* contract, void* user_data) {
    return anira_contract_hard_set_miss_fn(contract, anira_rt_contract_miss, user_data);
}
