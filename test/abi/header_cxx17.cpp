// Gate 4 (docs/anira-v3-architecture.md, section 6a): the M1 headers as C++17 under the
// strict flags with no anira define at all; the C headers never need C++20 (anira.hpp may).
// Every header is included on purpose, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>

#include <cstring>

namespace {

void on_record(const anira_log_record* /*record*/, void* /*user_data*/) {}

static_assert(sizeof(anira_error) == 520, "anira_error is frozen at 520 bytes");
static_assert(sizeof(anira_log_record) == 56, "anira_log_record is frozen at 56 bytes");
static_assert(sizeof(anira_status) == 4, "enums are 32-bit");
static_assert(ANIRA_DTYPE_F32 == 0x00012002u, "DLPack float32");

[[maybe_unused]] int anira_header_cxx17_probe() {
    const anira_error err = ANIRA_ERROR_INIT;
    anira_log_desc desc = ANIRA_LOG_DESC_INIT;
    anira_log_record record{};
    desc.callback = on_record;
    record.message = "message";
    int checks = 0;
    checks += ANIRA_SUCCEEDED(err.status) ? 1 : 0;
    checks += desc.abi_version == ANIRA_ABI_VERSION ? 1 : 0;
    checks += std::strlen(record.message) == 7 ? 1 : 0;
    checks += ANIRA_ABI_VERSION_MINOR(ANIRA_ABI_VERSION) == ANIRA_ABI_MINOR ? 1 : 0;
    const anira_ext_entry entry = ANIRA_EXT_ENTRY_INIT;
    const anira_cuda_desc cuda = ANIRA_CUDA_DESC_INIT;
    const anira_gl_desc gl = ANIRA_GL_DESC_INIT;
    const anira_vulkan_desc vulkan = ANIRA_VULKAN_DESC_INIT;
    const anira_metal_desc metal = ANIRA_METAL_DESC_INIT;
    const anira_d3d12_desc d3d12 = ANIRA_D3D12_DESC_INIT;
    const anira_webgpu_desc webgpu = ANIRA_WEBGPU_DESC_INIT;
    checks += entry.header.version == 1u && entry.name == nullptr ? 1 : 0;
    checks += cuda.ownership == ANIRA_OWNERSHIP_OWNED && gl.gbm == nullptr ? 1 : 0;
    checks += vulkan.queue_family == 0u && metal.device == nullptr ? 1 : 0;
    checks += d3d12.device == nullptr && webgpu.exec == ANIRA_EXEC_WORKER ? 1 : 0;
    return checks;
}

// Every C entry is noexcept in C++ (ANIRA_NOEXCEPT): a control-path entry, a void entry and
// a nonblocking one.
static_assert(noexcept(anira_model_config_create(nullptr, nullptr)));
static_assert(noexcept(anira_model_config_destroy(nullptr)));
static_assert(noexcept(anira_abi_version()));
static_assert(noexcept(anira_log_rt(ANIRA_LOG_ERROR, "g", "m", 0, 0)));

}  // namespace
// NOLINTEND(misc-include-cleaner)
