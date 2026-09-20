/*
 * Gate 4 (docs/anira-v3-architecture.md, section 6a): every C header in one C11
 * translation unit under -std=c11 -Wall -Wextra -Werror -pedantic (or /std:c11 /W4 /WX),
 * with no anira define at all, exercising each _INIT initializer and the macros a C host
 * uses. The per-file wrappers test/abi/CMakeLists.txt generates cover self-containment.
 */
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/abi/version.h>
#include <stddef.h>
#include <string.h>

static void on_record(const anira_log_record* record, void* user_data) {
    (void)record;
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
            checks += anira_plan_report_num_plans(NULL) == 0u ? 1 : 0;
        }
    }
    return checks;
}
