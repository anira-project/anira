// Gate 4 (docs/anira-v3-architecture.md, section 6a): the M1 headers as C++17 under the
// strict flags with no anira define at all; the C headers never need C++20 (anira.hpp may).
// Every header is included on purpose, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/abi/config.h>
#include <anira/abi/draft/tensor_platform.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>

#include <cstddef>
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
// The phases of the shared lifecycle, appended: the stage's reset slot and its unprepare,
// then init (both descriptors), then the engine's model level, load and unload.
static_assert(ANIRA_PHASE_RESET == 7 && ANIRA_PHASE_UNPREPARE == 8,
              "the phases of the shared lifecycle");
static_assert(ANIRA_PHASE_INIT == 9 && ANIRA_PHASE_LOAD == 10 && ANIRA_PHASE_UNLOAD == 11,
              "init, load and unload appended");
// anira/abi/lifecycle.h: the two records both descriptors share, and the bits of a prepare and
// of an engine call, bit 0 of their words (one assertion each: the operands of a joined one
// would read as the same expression).
constexpr uint32_t k_bit0 = 1u;
static_assert(std::is_same_v<decltype(anira_prepare_info::flags), uint32_t>,
              "the shared prepare record's flags");
static_assert(std::is_same_v<decltype(anira_init_info::num_threads), uint32_t>,
              "the shared init record's thread count");
static_assert(ANIRA_PREPARE_EXCLUSIVE == k_bit0, "the exclusive bit of a prepare");
static_assert(ANIRA_ENGINE_CALL_EXCLUSIVE == k_bit0, "the exclusive bit of an engine call");
static_assert(ANIRA_ENGINE_FLAG_STATE_ALIAS == (k_bit0 << 3), "the fourth engine promise");
// anira/abi/stage.h: the stage context is frozen at 64 bytes, eight scalars and four pointer
// slots (the frame, the pair's two ids and a reserved one), and travels by value like the
// tensor.
static_assert(sizeof(anira_stage_ctx) == 64 && alignof(anira_stage_ctx) == 8,
              "anira_stage_ctx is frozen");
static_assert(offsetof(anira_stage_ctx, frame) == 32 &&
                  offsetof(anira_stage_ctx, engine_id) == 40 &&
                  offsetof(anira_stage_ctx, provider_id) == 48 &&
                  offsetof(anira_stage_ctx, reserved_ptr2) == 56,
              "the four pointer slots follow the eight scalars");
static_assert(std::is_trivially_copyable_v<anira_stage_ctx> &&
                  std::is_standard_layout_v<anira_stage_ctx>,
              "anira_stage_ctx is a POD");
// The frame of a context is anira's: opaque, and not to be written through.
static_assert(std::is_same_v<decltype(anira_stage_ctx::frame), const void*>,
              "frame is an opaque pointer to const");
static_assert(std::is_same_v<decltype(anira_stage_desc::consumed_kinds), const char* const*>,
              "consumed_kinds is an array of const strings");
// The descriptor's callback slots are the named typedefs, the shared lifecycle's included.
static_assert(std::is_same_v<decltype(anira_stage_desc::pre_process), anira_stage_fn> &&
                  std::is_same_v<decltype(anira_stage_desc::reset), anira_stage_reset_fn> &&
                  std::is_same_v<decltype(anira_stage_desc::prepare), anira_stage_prepare_fn> &&
                  std::is_same_v<decltype(anira_stage_desc::unprepare), anira_stage_unprepare_fn> &&
                  std::is_same_v<decltype(anira_stage_desc::release), anira_stage_release_fn>,
              "the slots of anira_stage_desc");
// anira/abi/engine.h: the engine context is frozen like the stage context, eight scalars and
// four pointer slots (the two tensor arrays and two reserved ones); the inputs are read and
// the outputs written; the descriptor's slots are the named typedefs, listed as the stage's.
static_assert(sizeof(anira_engine_ctx) == 64 && alignof(anira_engine_ctx) == 8,
              "anira_engine_ctx is frozen");
static_assert(offsetof(anira_engine_ctx, inputs) == 32 &&
                  offsetof(anira_engine_ctx, outputs) == 40 &&
                  offsetof(anira_engine_ctx, reserved_ptr1) == 56,
              "the four pointer slots follow the eight scalars");
static_assert(std::is_trivially_copyable_v<anira_engine_ctx> &&
                  std::is_standard_layout_v<anira_engine_ctx>,
              "anira_engine_ctx is a POD");
static_assert(std::is_same_v<decltype(anira_engine_ctx::inputs), const anira_tensor*> &&
                  std::is_same_v<decltype(anira_engine_ctx::outputs), anira_tensor*>,
              "the inputs are read, the outputs written");
static_assert(
    std::is_same_v<decltype(anira_engine_desc::process), anira_engine_process_fn> &&
        std::is_same_v<decltype(anira_engine_desc::reset), anira_engine_reset_fn> &&
        std::is_same_v<decltype(anira_engine_desc::prepare), anira_engine_prepare_fn> &&
        std::is_same_v<decltype(anira_engine_desc::unprepare), anira_engine_unprepare_fn> &&
        std::is_same_v<decltype(anira_engine_desc::release), anira_engine_release_fn>,
    "the slots of anira_engine_desc");
// The two tail fields of the plan records.
static_assert(std::is_same_v<decltype(anira_plan_slot::binding), uint32_t>,
              "the tail field of anira_plan_slot is anira_binding, as uint32_t");
static_assert(std::is_same_v<decltype(anira_plan_info::engine_flags), uint32_t>,
              "the tail field of anira_plan_info is the engine flags, as uint32_t");

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
    const anira_provider_option_set option_set = ANIRA_PROVIDER_OPTION_SET_INIT;
    const anira_ext_provider_options provider_options = ANIRA_EXT_PROVIDER_OPTIONS_INIT;
    checks += option_set.struct_size == sizeof(anira_provider_option_set) ? 1 : 0;
    checks += provider_options.header.version == 1u && provider_options.num_sets == 0u ? 1 : 0;
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
    const anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;  // the initializer as C++17 braces it
    // anira/abi/lifecycle.h: the two shared records as C++17 braces their initializers.
    const anira_prepare_info info = ANIRA_PREPARE_INFO_INIT;
    const anira_init_info init_info = ANIRA_INIT_INFO_INIT;
    anira_stage_ctx ctx{};
    ctx.ticket = ANIRA_TICKET_INVALID;
    checks += stage.struct_size == sizeof(anira_stage_desc) && stage.user_data == nullptr ? 1 : 0;
    checks += stage.flags == 0U && stage.release == nullptr && stage.init == nullptr ? 1 : 0;
    checks += stage.reset == nullptr && stage.unprepare == nullptr ? 1 : 0;
    checks += info.struct_size == sizeof(anira_prepare_info) && info.handler == nullptr &&
                      info.num_entries == 0U && info.flags == 0U
                  ? 1
                  : 0;
    checks +=
        init_info.struct_size == sizeof(anira_init_info) && init_info.context == nullptr ? 1 : 0;
    checks += ctx.frame == nullptr && ctx.entry == 0U && ctx.reserved_ptr2_bits == 0u ? 1 : 0;
    // anira/abi/engine.h: the two initializers as C++17 braces them, and the context with its
    // loaded pointer.
    const anira_engine_desc engine = ANIRA_ENGINE_DESC_INIT;
    const anira_engine_load_info load_info = ANIRA_ENGINE_LOAD_INFO_INIT;
    anira_engine_ctx engine_ctx{};
    engine_ctx.ticket = ANIRA_TICKET_INVALID;
    checks += engine.struct_size == sizeof(anira_engine_desc) && engine.process == nullptr &&
                      engine.flags == 0U && engine.load == nullptr && engine.init == nullptr
                  ? 1
                  : 0;
    checks += load_info.struct_size == sizeof(anira_engine_load_info) && load_info.instances == 0U
                  ? 1
                  : 0;
    checks += load_info.provider == ANIRA_PROVIDER_CPU && load_info.provider_id == nullptr &&
                      engine.providers == nullptr && engine.num_providers == 0U &&
                      engine.query == nullptr
                  ? 1
                  : 0;
    checks += load_info.option_keys == nullptr && load_info.num_options == 0U ? 1 : 0;
    checks += engine_ctx.inputs == nullptr && engine_ctx.outputs_bits == 0u &&
                      engine_ctx.loaded == nullptr
                  ? 1
                  : 0;
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
// The two-slot forms: each side names its own slot, the tensor's position in its list.
static_assert(noexcept(anira_handler_process(nullptr, nullptr, 0, nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_pop_data_multi(nullptr, nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_process_wait(nullptr, nullptr, 0, nullptr, 0, nullptr, 0.0)));
// The Static entries: a slot and a whole tensor, const on both (anira writes the memory an
// output names, never its descriptor); the handler of the getter is not const.
static_assert(noexcept(anira_handler_set_static_input(nullptr, 0, nullptr)));
static_assert(noexcept(anira_handler_get_static_output(nullptr, 0, nullptr)));
// Declared state: the fourth role, appended, and the setter that pairs the two halves, stated
// once, on the input.
static_assert(ANIRA_ROLE_STATE == 3);
static_assert(noexcept(anira_tensor_spec_set_state_source(nullptr, nullptr)));
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_tensor_spec_set_state_source),
                                    anira_tensor_spec*,
                                    const char*>);
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_handler_get_static_output),
                                    anira_handler*,
                                    uint32_t,
                                    const anira_tensor*>);
// The stage entries: a ring accessor, a default body and the control-path add.
static_assert(noexcept(anira_ring_pop_block(nullptr, 0, nullptr, ANIRA_DTYPE_F32, 0)));
static_assert(noexcept(anira_stage_default_pre_process(nullptr)));
static_assert(noexcept(anira_stage_input_role(nullptr, 0, nullptr)));
static_assert(noexcept(anira_stage_output_ring(nullptr, 0, nullptr)));
static_assert(noexcept(anira_stage_input_tensor(nullptr, 0, nullptr)));
static_assert(noexcept(anira_pipeline_add_stage(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_custom_engine_create(nullptr, nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_custom_engine_destroy(nullptr)));
static_assert(noexcept(anira_custom_engine_detach(nullptr)));
static_assert(noexcept(anira_pipeline_add_engine(nullptr, nullptr, nullptr)));
static_assert(
    noexcept(anira_pipeline_capabilities_backends(nullptr, nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(
    anira_pipeline_capabilities_edge(nullptr, nullptr, ANIRA_DOMAIN_HOST, nullptr, nullptr)));
static_assert(noexcept(anira_handler_num_entries(nullptr)));
static_assert(noexcept(anira_contract_set_host_domain(nullptr, nullptr, ANIRA_DOMAIN_HOST)));
// The read-back of anira/abi/config.h: one pin per shape (a string, a count, an enum, a spec
// pointer and an extension record out of a value getter; a status getter over an enum and an
// int64_t, a string, a function pointer and a void*, and a Tier-2 record), and the axis
// getter's signature.
static_assert(noexcept(anira_tensor_spec_name(nullptr)));
static_assert(noexcept(anira_tensor_spec_ndim(nullptr)));
static_assert(noexcept(anira_tensor_spec_role(nullptr)));
static_assert(noexcept(anira_model_config_input(nullptr, 0)));
static_assert(noexcept(anira_model_config_model_ext(nullptr, 0, nullptr)));
static_assert(noexcept(anira_tensor_spec_axis(nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(anira_model_config_tensor_layout(nullptr, 0, nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_contract_hard_ring_dtype(nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(anira_contract_hard_miss_fn(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_contract_edge_cost(nullptr)));
static_assert(noexcept(anira_contract_hard_set_latency(nullptr, nullptr, 0)));
static_assert(noexcept(anira_contract_hard_num_latencies(nullptr)));
static_assert(noexcept(anira_contract_hard_latency(nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(anira_context_config_log(nullptr, nullptr)));
// The engine word on the public surface: the default provider and its read-back, the pair
// getters and the stage's two pair accessors (a status, the value and the nullable id), the
// renamed enumeration and the pair setters.
static_assert(
    noexcept(anira_model_config_set_default_provider(nullptr, ANIRA_PROVIDER_NONE, nullptr)));
static_assert(noexcept(anira_model_config_default_provider(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_model_config_default_engine(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_model_config_model_engine(nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(anira_model_config_model_provider(nullptr, 0, nullptr, nullptr)));
static_assert(noexcept(anira_stage_engine(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_stage_provider(nullptr, nullptr, nullptr)));
static_assert(noexcept(anira_enabled_engines(0, nullptr, nullptr)));
static_assert(noexcept(
    anira_model_config_set_default_engine(nullptr, ANIRA_ENGINE_CUSTOM, nullptr, nullptr)));
static_assert(noexcept(anira_model_config_add_model_path(nullptr,
                                                         ANIRA_ENGINE_CUSTOM,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr)));
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_model_config_model_engine),
                                    const anira_model_config*,
                                    uint32_t,
                                    anira_engine*,
                                    const char**>);
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_stage_provider),
                                    const anira_stage_ctx*,
                                    anira_provider*,
                                    const char**>);
static_assert(
    std::is_same_v<decltype(anira_model_config_input(nullptr, 0)), const anira_tensor_spec*>);
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_tensor_spec_axis),
                                    const anira_tensor_spec*,
                                    uint32_t,
                                    anira_axis_tag*,
                                    int64_t*>);
// The callback typedef carries no real-time attribute: a plain function converts to it.
static_assert(std::is_same_v<decltype(anira_stage_ctx::entry), uint32_t>);
static_assert(std::is_same_v<decltype(anira_stage_desc::flags), uint32_t>);
static_assert(std::is_invocable_r_v<anira_status,
                                    decltype(&anira_handler_push_data),
                                    anira_handler*,
                                    const anira_tensor*,
                                    uint32_t>);

}  // namespace
// NOLINTEND(misc-include-cleaner)
