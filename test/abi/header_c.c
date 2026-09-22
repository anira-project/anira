/*
 * Gate 4 (docs/anira-v3-architecture.md, section 6a): every C header in one C11
 * translation unit under -std=c11 -Wall -Wextra -Werror -pedantic (or /std:c11 /W4 /WX),
 * with no anira define at all, exercising each _INIT initializer and the macros a C host
 * uses. The per-file wrappers test/abi/CMakeLists.txt generates cover self-containment.
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <anira/abi/version.h>
#include <stddef.h>
#include <string.h>

static void on_record(const anira_log_record* record, void* user_data) {
    (void)record;
    (void)user_data;
}

/* anira/abi/stage.h from C11: a phase callback as a C host writes one (it asks the context
   what the first input is, pops its ring into its model end through the accessors and falls
   back on the default body), a prepare and a release function. The three are assigned to the
   descriptor's slots below, so their types are the typedefs' to the letter. */
static anira_status ANIRA_CALL on_pre_process(const anira_stage_ctx* ctx,
                                              void* user_data) ANIRA_NONBLOCKING {
    anira_role role = ANIRA_ROLE_FORCE32;
    (void)user_data;
    if (ctx->num_inputs > 0u && anira_stage_input_role(ctx, 0u, &role) == ANIRA_OK &&
        role == ANIRA_ROLE_STREAMED) {
        anira_ring* ring = NULL;
        anira_tensor tensor;
        if (anira_stage_input_ring(ctx, 0u, &ring) == ANIRA_OK &&
            anira_stage_input_tensor(ctx, 0u, &tensor) == ANIRA_OK) {
            const anira_dtype dtype = anira_ring_dtype(ring);
            void* data = anira_tensor_data(&tensor, dtype);
            const size_t count = anira_ring_available(ring, 0u);
            if (data != NULL && anira_ring_pop_block(ring, 0u, data, dtype, count) == count) {
                return ANIRA_OK;
            }
        }
    }
    return anira_stage_default_pre_process(ctx);
}

static anira_status ANIRA_CALL on_stage_prepare(anira_handler* handler,
                                                const anira_plan_report* report,
                                                void* user_data) {
    (void)handler;
    (void)user_data;
    return anira_plan_report_num_plans(report) > 0u ? ANIRA_OK : ANIRA_ERROR_INVALID_STATE;
}

static void ANIRA_CALL on_stage_release(void* user_data) {
    (void)user_data;
}

int anira_header_c_probe(void);
int anira_header_c_probe(void) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    anira_log_record record;
    const anira_dtype f32 = ANIRA_DTYPE_F32;
    const uint32_t abi = ANIRA_ABI_VERSION;
    const uint32_t version =
        ANIRA_MAKE_VERSION(ANIRA_VERSION_MAJOR, ANIRA_VERSION_MINOR, ANIRA_VERSION_PATCH);
    int checks = 0;

    memset(&record, 0, sizeof(record));
    record.group = "anira.test";
    desc.callback = on_record;
    desc.level = ANIRA_LOG_ERROR;

    checks += ANIRA_SUCCEEDED(err.status) ? 1 : 0;
    checks += ANIRA_FAILED(ANIRA_ERROR_JSON) ? 1 : 0;
    checks += desc.struct_size == sizeof(anira_log_desc) ? 1 : 0;
    checks += desc.abi_version == abi ? 1 : 0;
    checks += ANIRA_DTYPE_CODE(f32) == ANIRA_DTYPE_FLOAT ? 1 : 0;
    checks += ANIRA_DTYPE_BITS(f32) == 32u ? 1 : 0;
    checks += ANIRA_ABI_VERSION_MAJOR(abi) == ANIRA_ABI_MAJOR ? 1 : 0;
    checks += version != 0u || ANIRA_VERSION_MAJOR == 0 ? 1 : 0;
    checks += record.group_bits != 0u ? 1 : 0;
    checks += err.message[0] == '\0' ? 1 : 0;
    checks += ANIRA_MAX_RANK == 8 ? 1 : 0;
    checks += ANIRA_DYNAMIC == ANIRA_UNBOUNDED ? 1 : 0;
    checks += (int)ANIRA_AXIS_INSERT == -1 ? 1 : 0;

    {
        anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
        anira_cuda_desc cuda = ANIRA_CUDA_DESC_INIT;
        anira_gl_desc gl = ANIRA_GL_DESC_INIT;
        anira_vulkan_desc vulkan = ANIRA_VULKAN_DESC_INIT;
        anira_metal_desc metal = ANIRA_METAL_DESC_INIT;
        anira_d3d12_desc d3d12 = ANIRA_D3D12_DESC_INIT;
        anira_webgpu_desc webgpu = ANIRA_WEBGPU_DESC_INIT;
        entry.name = "forward";
        checks += entry.header.struct_size == sizeof(anira_ext_entry) ? 1 : 0;
        checks += entry.header.version == 1u ? 1 : 0;
        checks +=
            cuda.struct_size == sizeof(anira_cuda_desc) && cuda.ownership == ANIRA_OWNERSHIP_OWNED
                ? 1
                : 0;
        checks +=
            gl.struct_size == sizeof(anira_gl_desc) && gl.threads == ANIRA_GL_CALLER_THREAD ? 1 : 0;
        checks += vulkan.struct_size == sizeof(anira_vulkan_desc) && vulkan.device == NULL ? 1 : 0;
        checks += vulkan.device_index == 0 && vulkan.reserved == 0u ? 1 : 0;
        checks += metal.struct_size == sizeof(anira_metal_desc) ? 1 : 0;
        checks += d3d12.struct_size == sizeof(anira_d3d12_desc) ? 1 : 0;
        checks +=
            webgpu.struct_size == sizeof(anira_webgpu_desc) && webgpu.exec == ANIRA_EXEC_WORKER ? 1
                                                                                                : 0;
    }
    {
        /* The context header's records and the entries a C host calls without a context:
           referenced so that the declarations compile; the object is never linked. */
        anira_backend_id backend = ANIRA_BACKEND_ID_INIT;
        anira_edge_info edge = ANIRA_EDGE_INFO_INIT;
        anira_plan_slot slot = ANIRA_PLAN_SLOT_INIT;
        anira_plan_ext ext = ANIRA_PLAN_EXT_INIT;
        anira_plan_info info = ANIRA_PLAN_INFO_INIT;
        uint32_t count = 0;
        checks +=
            backend.struct_size == sizeof(anira_backend_id) && backend.engine_id == NULL ? 1 : 0;
        checks += edge.struct_size == sizeof(anira_edge_info) && edge.available == 0u ? 1 : 0;
        checks += slot.struct_size == sizeof(anira_plan_slot) && slot.recipe == NULL ? 1 : 0;
        checks += ext.struct_size == sizeof(anira_plan_ext) && ext.host == NULL ? 1 : 0;
        checks += info.struct_size == sizeof(anira_plan_info) && info.budget_ms == 0.0 ? 1 : 0;
        if (checks < 0) { /* never true: keeps the calls out of the probe's own result */
            const double now = anira_now_ms();
            const anira_status status =
                anira_enabled_backends((uint32_t)sizeof(anira_backend_id), &count, NULL);
            checks += now > 0.0 && ANIRA_SUCCEEDED(status) && count > 0u ? 1 : 0;
            checks += anira_num_inference_threads() == 0u ? 1 : 0;
            checks += anira_handler_rt_error(NULL) == ANIRA_OK ? 1 : 0;
            /* The Static entries: a slot and a whole tensor; a NULL handler is refused. */
            checks += anira_handler_set_static_input(NULL, 0u, NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks +=
                anira_handler_get_static_output(NULL, 0u, NULL) == ANIRA_ERROR_INVALID_ARGUMENT ? 1
                                                                                                : 0;
            checks += anira_plan_report_num_plans(NULL) == 0u ? 1 : 0;
            /* Declared state: the fourth role, and the setter that pairs the two halves; a
               NULL spec is refused. */
            checks += ANIRA_ROLE_STATE == 3 ? 1 : 0;
            checks += anira_tensor_spec_set_state_source(NULL, "state_out") ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
        }
    }
    {
        /* anira/abi/tensor.h from C11: the three frozen sizes, a record zeroed and filled by
           hand (both names of an ANIRA_PTR slot), and behind the never-true branch the host
           factory, the planar factory over a float*[2] WITHOUT a cast (the channel pointers of
           an audio host), the accessors, anira_sizeof, a draft factory and the token reset. */
        anira_tensor tensor;
        anira_sync_token token;
        float samples[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float* channels[2];
        const float* const* read_channels;
        const int64_t shape[1] = {4};
        const int64_t block[2] = {2, 2};
        channels[0] = samples;
        channels[1] = samples + 2;
        read_channels = (const float* const*)channels;
        memset(&tensor, 0, sizeof(tensor));
        memset(&token, 0, sizeof(token));
        tensor.handle.host.ptr = samples;
        checks += sizeof(anira_tensor) == 216u && sizeof(anira_sync_token) == 24u ? 1 : 0;
        checks += sizeof(anira_memory_handle) == 24u && sizeof(tensor.handle.raw) == 24u ? 1 : 0;
        checks += tensor.handle.host.ptr_bits != 0u && tensor.release == NULL ? 1 : 0;
        checks += token.kind == (uint32_t)ANIRA_SYNC_NONE && token.u.raw[1] == 0u ? 1 : 0;
        if (checks < 0) { /* never true: the object is never linked */
            anira_tensor_init_host(&tensor, samples, ANIRA_DTYPE_F32, 1u, shape);
            checks += anira_tensor_data_f32(&tensor) == samples ? 1 : 0;
            checks += anira_tensor_data(&tensor, ANIRA_DTYPE_F32) != NULL ? 1 : 0;
            checks += anira_tensor_num_elements(&tensor) == 4u ? 1 : 0;
            checks += anira_tensor_extent(&tensor, 0u) == 4u ? 1 : 0;
            checks += anira_sizeof(ANIRA_STRUCT_TENSOR) == sizeof(anira_tensor) ? 1 : 0;
            /* No cast, on purpose: the gate proves a float** converts as it is. */
            /* NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion) */
            anira_tensor_init_host_planar(&tensor, channels, 2u, ANIRA_DTYPE_F32, 2u, block);
            checks += anira_tensor_plane(&tensor, 1u, ANIRA_DTYPE_F32) == samples + 2 ? 1 : 0;
            checks += (tensor.flags & (uint32_t)ANIRA_TENSOR_PLANAR) != 0u ? 1 : 0;
            /* NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion) */
            anira_tensor_init_host_planar(&tensor, read_channels, 2u, ANIRA_DTYPE_F32, 2u, block);
            anira_tensor_init_metal(&tensor, NULL, NULL, ANIRA_DTYPE_F32, 1u, shape);
            anira_sync_token_reset(&token);
            /* A Hard entry over host tensors from C11: the planar block of above as the input
               and the output of one call (in place), the descriptor const on both sides. */
            /* NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion) */
            anira_tensor_init_host_planar(&tensor, channels, 2u, ANIRA_DTYPE_F32, 2u, block);
            checks +=
                anira_contract_hard_set_miss_fn(NULL, NULL, NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                    ? 1
                    : 0;
            checks += anira_handler_process(NULL, &tensor, 0u, &tensor, 0u, NULL) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
        }
    }
    {
        /* anira/abi/stage.h from C11: the descriptor's initializer with the three callback
           typedefs in its slots, the frozen size of the stage context, a context zeroed and
           filled by hand (both names of an ANIRA_PTR slot, the frame's pointee const), and
           behind the never-true branch the entries a C host calls. */
        static const char* const kinds[1] = {"model:entry"};
        anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
        anira_stage_ctx ctx;
        anira_tensor model_end;
        anira_role role = ANIRA_ROLE_FORCE32;
        anira_ring* ring = NULL;
        memset(&ctx, 0, sizeof(ctx));
        ctx.phase = (uint32_t)ANIRA_PHASE_PRE_PROCESS;
        ctx.ticket = ANIRA_TICKET_INVALID;
        ctx.entry = 0u;
        ctx.frame = kinds; /* any address: the frame is anira's, this one is never read */
        checks += stage.struct_size == sizeof(anira_stage_desc) && stage.user_data == NULL ? 1 : 0;
        checks += stage.flags == 0u && stage.pre_process == NULL ? 1 : 0;
        stage.consumed_kinds = kinds;
        stage.num_consumed_kinds = 1u;
        /* The real-time promise of the two host-end phases, as a Hard contract requires it. */
        stage.flags = ANIRA_STAGE_REALTIME_PRE_POST;
        stage.pre_process = on_pre_process;
        stage.prepare = on_stage_prepare;
        stage.release = on_stage_release;
        checks += sizeof(anira_stage_ctx) == 64u && offsetof(anira_stage_ctx, frame) == 32u ? 1 : 0;
        checks += ctx.frame_bits != 0u && ctx.reserved_ptr0 == NULL ? 1 : 0;
        if (checks < 0) { /* never true: the object is never linked */
            checks += anira_sizeof(ANIRA_STRUCT_STAGE_CTX) == 64u ? 1 : 0;
            checks += anira_pipeline_add_stage(NULL, &stage, NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_handler_num_entries(NULL) == 0u ? 1 : 0;
            checks += anira_contract_set_host_domain(NULL, "in", ANIRA_DOMAIN_HOST) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += stage.pre_process(&ctx, NULL) == ANIRA_OK ? 1 : 0;
            checks +=
                anira_stage_default_post_process(&ctx) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
            checks += anira_ring_num_channels(NULL) == 0u ? 1 : 0;
            /* The six context accessors, as a C host calls them: a status, and the answer in
               the out-parameter. */
            checks +=
                anira_stage_input_role(NULL, 0u, &role) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
            checks +=
                anira_stage_output_role(NULL, 0u, &role) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
            checks +=
                anira_stage_input_ring(NULL, 0u, &ring) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
            checks +=
                anira_stage_output_ring(NULL, 0u, &ring) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
            checks += role == ANIRA_ROLE_FORCE32 && ring == NULL ? 1 : 0;
            checks += anira_stage_input_tensor(NULL, 0u, &model_end) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks +=
                anira_stage_output_tensor(NULL, 0u, &model_end) == ANIRA_ERROR_INVALID_ARGUMENT ? 1
                                                                                                : 0;
        }
    }
    return checks;
}
