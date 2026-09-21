// Gate 4 (docs/anira-v3-architecture.md, section 6a): the M1 headers as C++17 under the
// strict flags with no anira define at all; the C headers never need C++20 (anira.hpp may).
// Every header is included on purpose, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/abi/config.h>
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

void on_record(const anira_log_record* /*record*/, void* /*user_data*/) {}

static_assert(sizeof(anira_error) == 520, "anira_error is frozen at 520 bytes");
static_assert(sizeof(anira_log_record) == 56, "anira_log_record is frozen at 56 bytes");
static_assert(sizeof(anira_status) == 4, "enums are 32-bit");
static_assert(sizeof(anira_tensor) == 216 && alignof(anira_tensor) == 8, "anira_tensor is frozen");
static_assert(sizeof(anira_sync_token) == 24 && sizeof(anira_memory_handle) == 24,
              "the token and the handle are frozen at 24 bytes");
// What lets the record travel through a lock-free FIFO and lets anira::Tensor derive from it.
static_assert(std::is_trivially_copyable_v<anira_tensor> && std::is_standard_layout_v<anira_tensor>,
              "anira_tensor is a POD");
// The release proc is a function type, so its ANIRA_PTR slot holds a function pointer.
static_assert(std::is_function_v<anira_tensor_release_proc>,
              "anira_tensor_release_proc is a function type");
static_assert(std::is_same_v<decltype(anira_tensor::release), anira_tensor_release_proc*>,
              "release is a pointer to it");
static_assert(ANIRA_DTYPE_F32 == 0x00012002u, "DLPack float32");
// The three inference-thread phases are numbered in the order they run: the engine call sits
// between the two hooks named after it.
static_assert(ANIRA_PHASE_BEFORE_INFERENCE == 2 && ANIRA_PHASE_INFERENCE == 3 &&
                  ANIRA_PHASE_AFTER_INFERENCE == 4,
              "before < inference < after");
static_assert(ANIRA_PHASE_PREPARE == 5 && ANIRA_PHASE_RELEASE == 6, "the lifecycle phases");

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
    checks += vulkan.device_index == 0 && vulkan.reserved == 0u ? 1 : 0;
    anira_tensor tensor{};  // the record and a union arm as C++17 spells them
    tensor.handle.host.ptr = &checks;
    checks += tensor.release == nullptr && tensor.ndim == 0u ? 1 : 0;
    checks += d3d12.device == nullptr && webgpu.exec == ANIRA_EXEC_WORKER ? 1 : 0;
    return checks;
}

// Every C entry is noexcept in C++ (ANIRA_NOEXCEPT): a control-path entry, a void entry and
// a nonblocking one.
static_assert(noexcept(anira_model_config_create(nullptr, nullptr)));
static_assert(noexcept(anira_model_config_destroy(nullptr)));
static_assert(noexcept(anira_abi_version()));
static_assert(noexcept(anira_log_rt(ANIRA_LOG_ERROR, "g", "m", 0, 0)));
// The tensor entries: a nonblocking fill, an accessor, the fallible bridge, a token call and a
// draft factory.
static_assert(noexcept(anira_tensor_init_host(nullptr, nullptr, ANIRA_DTYPE_F32, 0, nullptr)));
static_assert(noexcept(anira_tensor_data_f32(nullptr)));
static_assert(noexcept(anira_sizeof(ANIRA_STRUCT_TENSOR)));
static_assert(noexcept(anira_tensor_init_dlpack(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_sync_token_reset(nullptr)));
static_assert(
    noexcept(anira_tensor_init_metal(nullptr, nullptr, nullptr, ANIRA_DTYPE_F32, 0, nullptr)));
// The Hard entries over host tensors: a nonblocking stem, a multi form and a _wait twin. The
// host block is a const anira_tensor*, an output's included.
static_assert(noexcept(anira_handler_process(nullptr, nullptr, nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_pop_data_multi(nullptr, nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_process_wait(nullptr, nullptr, nullptr, 0.0, 0, nullptr)));
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_handler_push_data),
                                    anira_handler*,
                                    const anira_tensor*,
                                    uint32_t>);

}  // namespace
// NOLINTEND(misc-include-cleaner)
