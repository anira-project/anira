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
#include <anira/abi/engine.h>
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
   back on the default body), a reset, a prepare, an unprepare, an init and a release
   function. The six are assigned to the descriptor's slots below, so their types are the
   typedefs' to the letter; every parameter a body does not read is voided (MSVC /W4 /WX). */
static anira_status ANIRA_CALL on_pre_process(const anira_stage_ctx* ctx,
                                              void* prepared,
                                              void* user_data) ANIRA_NONBLOCKING {
    anira_role role = ANIRA_ROLE_FORCE32;
    (void)prepared;
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

/* The reset slot: the first chunk of a new stream, on the thread of pre_process, where the
   role is the one answer the context gives. */
static void ANIRA_CALL on_stage_reset(const anira_stage_ctx* ctx,
                                      void* prepared,
                                      void* user_data) ANIRA_NONBLOCKING {
    anira_role role = ANIRA_ROLE_FORCE32;
    (void)prepared;
    (void)user_data;
    (void)anira_stage_input_role(ctx, 0u, &role);
}

static anira_status ANIRA_CALL on_stage_prepare(const anira_prepare_info* info,
                                                void* user_data,
                                                void** out_prepared) {
    (void)user_data;
    *out_prepared = NULL; /* nothing per handler: the phases receive NULL */
    return info->num_entries > 0u && anira_plan_report_num_plans(info->report) > 0u &&
                   (info->flags & ANIRA_PREPARE_EXCLUSIVE) == 0u
               ? ANIRA_OK
               : ANIRA_ERROR_INVALID_STATE;
}

/* The init slot: once per registration, with the shared init record of lifecycle.h. */
static anira_status ANIRA_CALL on_stage_init(const anira_init_info* info, void* user_data) {
    (void)user_data;
    return info->struct_size == sizeof(anira_init_info) && info->context == NULL
               ? ANIRA_OK
               : ANIRA_ERROR_INVALID_STATE;
}

static void ANIRA_CALL on_stage_unprepare(void* prepared, void* user_data) {
    (void)prepared;
    (void)user_data;
}

static void ANIRA_CALL on_stage_release(void* user_data) {
    (void)user_data;
}

/* anira/abi/engine.h from C11: the eight functions of an engine descriptor, assigned to
   its slots below. process copies the first input into the first output, reading every
   extent and every pointer from the tensors of the call (the adapter rule). */
static anira_status ANIRA_CALL on_engine_process(const anira_engine_ctx* ctx,
                                                 void* prepared,
                                                 void* user_data) {
    (void)prepared;
    (void)user_data;
    if (ctx->num_inputs > 0u && ctx->num_outputs > 0u) {
        const float* in = anira_tensor_data_f32(&ctx->inputs[0]);
        float* out = anira_tensor_data_f32(&ctx->outputs[0]);
        const size_t count = anira_tensor_num_elements(&ctx->outputs[0]);
        size_t i;
        if (in == NULL || out == NULL || anira_tensor_num_elements(&ctx->inputs[0]) < count) {
            return ANIRA_ERROR_ENGINE;
        }
        for (i = 0; i < count; ++i) { out[i] = in[i]; }
    }
    return ANIRA_OK;
}

static void ANIRA_CALL on_engine_reset(const anira_engine_ctx* ctx,
                                       void* prepared,
                                       void* user_data) {
    (void)ctx;
    (void)prepared;
    (void)user_data;
}

static anira_status ANIRA_CALL on_engine_init(const anira_init_info* info, void* user_data) {
    (void)user_data;
    return info->struct_size == sizeof(anira_init_info) ? ANIRA_OK : ANIRA_ERROR_INVALID_STATE;
}

static anira_status ANIRA_CALL on_engine_load(const anira_engine_load_info* info,
                                              void* user_data,
                                              void** out_loaded) {
    (void)user_data;
    *out_loaded = NULL;
    return info->num_inputs > 0u ? ANIRA_OK : ANIRA_ERROR_CONFIG;
}

static void ANIRA_CALL on_engine_unload(void* loaded, void* user_data) {
    (void)loaded;
    (void)user_data;
}

static anira_status ANIRA_CALL on_engine_prepare(const anira_prepare_info* info,
                                                 void* loaded,
                                                 void* user_data,
                                                 void** out_prepared) {
    (void)loaded;
    (void)user_data;
    *out_prepared = NULL; /* a stateless handler keeps nothing per handler */
    return info->num_entries > 0u || (info->flags & ANIRA_PREPARE_EXCLUSIVE) == 0u
               ? ANIRA_OK
               : ANIRA_ERROR_INVALID_STATE;
}

static void ANIRA_CALL on_engine_unprepare(void* prepared, void* user_data) {
    (void)prepared;
    (void)user_data;
}

static anira_status ANIRA_CALL on_engine_query(const anira_init_info* info,
                                               void* user_data,
                                               uint64_t* out_available) {
    (void)info;
    (void)user_data;
    *out_available = 0u;
    return ANIRA_OK;
}

static void ANIRA_CALL on_engine_release(void* user_data) {
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
        anira_provider_option_set option_set = ANIRA_PROVIDER_OPTION_SET_INIT;
        anira_ext_provider_options provider_options = ANIRA_EXT_PROVIDER_OPTIONS_INIT;
        entry.name = "forward";
        checks += entry.header.struct_size == sizeof(anira_ext_entry) ? 1 : 0;
        checks += entry.header.version == 1u ? 1 : 0;
        checks += option_set.struct_size == sizeof(anira_provider_option_set) &&
                          option_set.num_options == 0u && option_set.keys == NULL
                      ? 1
                      : 0;
        checks += provider_options.header.struct_size == sizeof(anira_ext_provider_options) &&
                          provider_options.sets == NULL && provider_options.num_sets == 0u
                      ? 1
                      : 0;
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
        checks += backend.provider == ANIRA_PROVIDER_DEFAULT && backend.provider_id == NULL ? 1 : 0;
        checks += edge.struct_size == sizeof(anira_edge_info) && edge.available == 0u ? 1 : 0;
        checks += edge.reason == NULL && edge.to_provider_id == NULL ? 1 : 0;
        checks += edge.to_engine_id == NULL ? 1 : 0;
        checks += slot.struct_size == sizeof(anira_plan_slot) && slot.recipe == NULL ? 1 : 0;
        checks += ext.struct_size == sizeof(anira_plan_ext) && ext.host == NULL ? 1 : 0;
        checks += info.struct_size == sizeof(anira_plan_info) && info.budget_ms == 0.0 ? 1 : 0;
        checks += info.provider_id == NULL && info.engine_flags == 0u ? 1 : 0;
        if (checks < 0) { /* never true: keeps the calls out of the probe's own result */
            const double now = anira_now_ms();
            const anira_status status =
                anira_enabled_engines((uint32_t)sizeof(anira_backend_id), &count, NULL);
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
        /* The read-back of anira/abi/config.h from C11: one call per shape behind the
           never-true branch. A value getter answers a pointer, a count or an enum; a status
           getter fills enum, int64_t, double, string, function-pointer and void* out-parameters
           and a Tier-2 record within its struct_size. */
        const anira_tensor_spec* spec = NULL;
        const anira_ext_header* ext = NULL;
        const char* canonical = NULL;
        anira_axis_tag tag = ANIRA_AXIS_ANY;
        int64_t extent = 0;
        uint32_t count = 0u;
        uint32_t axes[ANIRA_MAX_RANK];
        double ms = 0.0;
        anira_budget_kind budget = ANIRA_BUDGET_MEASURED;
        anira_dtype dtype = 0u;
        anira_miss_fn miss = NULL;
        void* miss_user_data = NULL;
        anira_wait_strategy wait = ANIRA_WAIT_SPIN_BACKOFF;
        anira_log_desc log = ANIRA_LOG_DESC_INIT;
        anira_engine engine = ANIRA_ENGINE_FORCE32;
        anira_provider provider = ANIRA_PROVIDER_FORCE32;
        const char* pair_id = NULL;
        axes[0] = 0u;
        if (checks < 0) { /* never true: the object is never linked */
            spec = anira_model_config_input(NULL, 0u);
            checks += anira_tensor_spec_name(spec) == NULL ? 1 : 0;
            checks += anira_tensor_spec_role(spec) == ANIRA_ROLE_FORCE32 ? 1 : 0;
            checks += anira_tensor_spec_ndim(spec) == 0u ? 1 : 0;
            checks +=
                anira_tensor_spec_axis(spec, 0u, &tag, &extent) == ANIRA_ERROR_INVALID_ARGUMENT ? 1
                                                                                                : 0;
            checks += anira_tensor_spec_latency(spec) == 0 ? 1 : 0;
            checks += anira_model_config_num_outputs(NULL) == 0u ? 1 : 0;
            checks += anira_model_config_state(NULL) == ANIRA_MODEL_STATE_FORCE32 ? 1 : 0;
            checks += anira_model_config_set_default_provider(NULL, ANIRA_PROVIDER_CUDA, NULL) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            /* The pair getters: a status and two out-parameters, the id one nullable. */
            checks += anira_model_config_default_provider(NULL, &provider, &pair_id) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_model_config_default_engine(NULL, &engine, NULL) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_model_config_model_engine(NULL, 0u, &engine, &pair_id) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_model_config_model_provider(NULL, 0u, &provider, NULL) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_model_config_tensor_layout(NULL, 0u, "audio_in", &count, axes) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            ext = anira_model_config_model_ext(NULL, 0u, "entry");
            checks += ext == NULL ? 1 : 0;
            checks += anira_contract_hard_budget(NULL, &budget, &ms) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_contract_hard_miss_fn(NULL, &miss, &miss_user_data) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_contract_hard_ring_dtype(NULL, 0u, &canonical, &dtype) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_contract_edge_cost(NULL) == ANIRA_EDGE_COST_PERMISSIVE ? 1 : 0;
            /* The declared stream latency: the setter, the count and the enumeration. */
            checks += anira_contract_hard_set_latency(NULL, "audio_out", 1024u) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_contract_hard_num_latencies(NULL) == 0u ? 1 : 0;
            checks += anira_contract_hard_latency(NULL, 0u, &canonical, &count) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks +=
                anira_context_config_threads(NULL, &count, &wait) == ANIRA_ERROR_INVALID_ARGUMENT
                    ? 1
                    : 0;
            checks += anira_context_config_log(NULL, &log) == ANIRA_ERROR_INVALID_ARGUMENT ? 1 : 0;
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
        /* anira/abi/stage.h from C11: the descriptor's initializer with the six callback
           typedefs in its slots, the initializers of the two shared records of lifecycle.h,
           the frozen size of the stage context, a context zeroed and filled by hand (both
           names of an ANIRA_PTR slot, the frame's pointee const), the phases of the shared
           lifecycle, and behind the never-true branch the entries a C host calls and the
           slots as anira calls them. */
        static const char* const kinds[1] = {"model:entry"};
        anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
        anira_prepare_info prepare_info = ANIRA_PREPARE_INFO_INIT;
        anira_init_info init_info = ANIRA_INIT_INFO_INIT;
        anira_stage_ctx ctx;
        anira_tensor model_end;
        anira_role role = ANIRA_ROLE_FORCE32;
        anira_ring* ring = NULL;
        const char* name = NULL;
        anira_engine plan_engine = ANIRA_ENGINE_FORCE32;
        anira_provider plan_provider = ANIRA_PROVIDER_FORCE32;
        void* prepared = NULL;
        memset(&ctx, 0, sizeof(ctx));
        ctx.phase = (uint32_t)ANIRA_PHASE_PRE_PROCESS;
        ctx.ticket = ANIRA_TICKET_INVALID;
        ctx.entry = 0u;
        ctx.frame = kinds; /* any address: the frame is anira's, this one is never read */
        checks += stage.struct_size == sizeof(anira_stage_desc) && stage.user_data == NULL ? 1 : 0;
        checks += stage.flags == 0u && stage.pre_process == NULL ? 1 : 0;
        checks += stage.reset == NULL && stage.unprepare == NULL && stage.init == NULL ? 1 : 0;
        checks += prepare_info.struct_size == sizeof(anira_prepare_info) &&
                          prepare_info.handler == NULL && prepare_info.num_entries == 0u &&
                          prepare_info.flags == 0u
                      ? 1
                      : 0;
        checks += init_info.struct_size == sizeof(anira_init_info) && init_info.context == NULL &&
                          init_info.num_threads == 0u
                      ? 1
                      : 0;
        checks += ANIRA_PHASE_RESET == 7 && ANIRA_PHASE_UNPREPARE == 8 ? 1 : 0;
        checks +=
            ANIRA_PHASE_INIT == 9 && ANIRA_PHASE_LOAD == 10 && ANIRA_PHASE_UNLOAD == 11 ? 1 : 0;
        stage.consumed_kinds = kinds;
        stage.num_consumed_kinds = 1u;
        /* The real-time promise of the two host-end phases (the reset runs on their thread), as
           a Hard contract requires it, and the hooks' promise beside it. */
        stage.flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS;
        stage.pre_process = on_pre_process;
        stage.reset = on_stage_reset;
        stage.prepare = on_stage_prepare;
        stage.unprepare = on_stage_unprepare;
        stage.init = on_stage_init;
        stage.release = on_stage_release;
        checks += stage.flags == 3u ? 1 : 0;
        checks += sizeof(anira_stage_ctx) == 64u && offsetof(anira_stage_ctx, frame) == 32u ? 1 : 0;
        checks += ctx.frame_bits != 0u && ctx.engine_id == NULL && ctx.provider_id == NULL ? 1 : 0;
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
            /* The slots, called as anira calls them: (init_info, user_data) once per
               registration, (ctx, prepared, user_data) per chunk, (info, user_data, &prepared)
               at prepare, (prepared, user_data) at unprepare. */
            checks += stage.init(&init_info, NULL) == ANIRA_OK ? 1 : 0;
            checks += stage.pre_process(&ctx, prepared, NULL) == ANIRA_OK ? 1 : 0;
            stage.reset(&ctx, prepared, NULL);
            checks +=
                stage.prepare(&prepare_info, NULL, &prepared) == ANIRA_ERROR_INVALID_STATE ? 1 : 0;
            stage.unprepare(prepared, NULL);
            stage.release(NULL);
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
            checks += anira_stage_engine(NULL, &plan_engine, &name) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks +=
                anira_stage_provider(NULL, &plan_provider, NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                    ? 1
                    : 0;
        }
    }
    {
        /* anira/abi/engine.h from C11: the descriptor's initializer with the eight callback
           typedefs in its slots, the load record's initializer and the two shared records',
           the frozen size of the engine context, a context zeroed and filled by hand (both
           names of an ANIRA_PTR slot, the loaded pointer among them), the three promises
           OR-ed, the binding enum and the two tail fields of the plan records, and behind the
           never-true branch the engine entries and the slots as anira calls them, from the
           outermost level in. */
        anira_engine_desc engine = ANIRA_ENGINE_DESC_INIT;
        anira_engine_load_info load_info = ANIRA_ENGINE_LOAD_INFO_INIT;
        anira_prepare_info prepare_info = ANIRA_PREPARE_INFO_INIT;
        anira_init_info init_info = ANIRA_INIT_INFO_INIT;
        anira_engine_ctx engine_ctx;
        anira_tensor model_end;
        anira_plan_slot slot = ANIRA_PLAN_SLOT_INIT;
        anira_plan_info plan = ANIRA_PLAN_INFO_INIT;
        void* loaded = NULL;
        void* prepared = NULL;
        memset(&engine_ctx, 0, sizeof(engine_ctx));
        memset(&model_end, 0, sizeof(model_end));
        engine_ctx.ticket = ANIRA_TICKET_INVALID;
        engine_ctx.num_inputs = 1u;
        engine_ctx.inputs = &model_end; /* the pointer name of the ANIRA_PTR slot */
        checks +=
            engine.struct_size == sizeof(anira_engine_desc) && engine.user_data == NULL ? 1 : 0;
        checks += engine.flags == 0u && engine.process == NULL && engine.release == NULL ? 1 : 0;
        checks += engine.load == NULL && engine.unload == NULL && engine.init == NULL ? 1 : 0;
        checks +=
            engine.providers == NULL && engine.num_providers == 0u && engine.query == NULL ? 1 : 0;
        checks += load_info.struct_size == sizeof(anira_engine_load_info) &&
                          load_info.model == NULL && load_info.instances == 0u
                      ? 1
                      : 0;
        checks +=
            load_info.provider == ANIRA_PROVIDER_DEFAULT && load_info.provider_id == NULL ? 1 : 0;
        checks += load_info.option_keys == NULL && load_info.option_values == NULL &&
                          load_info.num_options == 0u
                      ? 1
                      : 0;
        checks += sizeof(anira_engine_ctx) == 64u && offsetof(anira_engine_ctx, inputs) == 32u &&
                          offsetof(anira_engine_ctx, loaded) == 48u &&
                          offsetof(anira_engine_ctx, reserved_ptr1) == 56u
                      ? 1
                      : 0;
        checks += engine_ctx.inputs_bits != 0u && engine_ctx.outputs == NULL &&
                          engine_ctx.loaded_bits == 0u && engine_ctx.loaded == NULL &&
                          (engine_ctx.flags & ANIRA_ENGINE_CALL_EXCLUSIVE) == 0u
                      ? 1
                      : 0;
        /* The two tail fields of the plan records and the binding enum. */
        checks += slot.binding == (uint32_t)ANIRA_BINDING_POSITION && plan.engine_flags == 0u &&
                          plan.reserved == 0u
                      ? 1
                      : 0;
        checks += ANIRA_BINDING_NAME == 1 && ANIRA_BINDING_ENGINE == 2 ? 1 : 0;
        /* The three promises OR-ed, and every slot filled with a C function of its typedef. */
        engine.flags = ANIRA_ENGINE_FLAG_NEEDS_NO_MODEL | ANIRA_ENGINE_FLAG_REALTIME_SAFE |
                       ANIRA_ENGINE_FLAG_DYNAMIC_TIME;
        engine.process = on_engine_process;
        engine.reset = on_engine_reset;
        engine.prepare = on_engine_prepare;
        engine.unprepare = on_engine_unprepare;
        engine.load = on_engine_load;
        engine.unload = on_engine_unload;
        engine.init = on_engine_init;
        engine.release = on_engine_release;
        engine.query = on_engine_query;
        checks += engine.flags == 7u ? 1 : 0;
        if (checks < 0) { /* never true: the object is never linked */
            checks += anira_sizeof(ANIRA_STRUCT_ENGINE_CTX) == 64u ? 1 : 0;
            anira_custom_engine* custom = NULL;
            checks +=
                anira_custom_engine_create("org.example.c", &engine, &custom, NULL) == ANIRA_OK ? 1
                                                                                                : 0;
            checks += anira_pipeline_add_engine(NULL, custom, NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            anira_custom_engine_detach(custom);
            anira_custom_engine_destroy(custom);
            /* The pipeline's capabilities: the context's rows and the custom engines'. */
            checks += anira_pipeline_capabilities_backends(NULL,
                                                           NULL,
                                                           sizeof(anira_backend_id),
                                                           NULL,
                                                           NULL) == ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            checks += anira_pipeline_capabilities_edge(NULL, NULL, ANIRA_DOMAIN_HOST, NULL, NULL) ==
                              ANIRA_ERROR_INVALID_ARGUMENT
                          ? 1
                          : 0;
            {
                uint64_t available = 0u;
                checks += engine.query(&init_info, NULL, &available) == ANIRA_OK ? 1 : 0;
            }
            /* The slots, from the outermost level in: init once per object, load once per
               loaded model, prepare once per handler on it, process and reset per call,
               then the way back out. */
            checks += engine.init(&init_info, NULL) == ANIRA_OK ? 1 : 0;
            checks += engine.load(&load_info, NULL, &loaded) == ANIRA_ERROR_CONFIG ? 1 : 0;
            checks += engine.prepare(&prepare_info, loaded, NULL, &prepared) == ANIRA_OK ? 1 : 0;
            checks += engine.process(&engine_ctx, prepared, NULL) == ANIRA_OK ? 1 : 0;
            engine.reset(&engine_ctx, prepared, NULL);
            engine.unprepare(prepared, NULL);
            engine.unload(loaded, NULL);
            engine.release(NULL);
        }
    }
    return checks;
}
