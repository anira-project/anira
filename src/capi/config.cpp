#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "ext_registry.h"
#include "handles.h"
#include "layout.h"

using anira::capi::report_void_failure;
using anira::capi::translate_exception;

namespace {

// ---- value checks -----------------------------------------------------------------------

bool valid_role(anira_role role) {
    return role == ANIRA_ROLE_STREAMED || role == ANIRA_ROLE_BUFFER || role == ANIRA_ROLE_STATIC;
}
bool valid_axis_tag(anira_axis_tag tag) {
    return tag >= ANIRA_AXIS_BATCH && tag <= ANIRA_AXIS_ANY;
}
bool builtin_engine(anira_engine engine) {
    return engine >= ANIRA_ENGINE_ONNXRUNTIME && engine <= ANIRA_ENGINE_EXECUTORCH;
}
bool valid_bytes_ownership(anira_bytes_ownership ownership) {
    return ownership == ANIRA_BYTES_COPY || ownership == ANIRA_BYTES_BORROW;
}
bool valid_budget_kind(anira_budget_kind kind) {
    return kind == ANIRA_BUDGET_MEASURED || kind == ANIRA_BUDGET_EXPLICIT;
}
bool valid_warmup_mode(anira_warmup_mode mode) {
    return mode >= ANIRA_WARMUP_NONE && mode <= ANIRA_WARMUP_UNTIL_STABLE;
}
bool valid_miss_policy(anira_miss_policy policy) {
    return policy >= ANIRA_MISS_BYPASS && policy <= ANIRA_MISS_ZEROS;
}
bool valid_late_policy(anira_late_policy policy) {
    return policy == ANIRA_LATE_FINISH || policy == ANIRA_LATE_DROP;
}
bool valid_priority(anira_priority priority) {
    return priority >= ANIRA_PRIORITY_AUTO && priority <= ANIRA_PRIORITY_BATCH;
}
bool valid_delivery(anira_delivery delivery) {
    return delivery == ANIRA_DELIVERY_POLLED || delivery == ANIRA_DELIVERY_IMMEDIATE;
}
bool valid_edge_cost(anira_edge_cost cost) {
    return cost == ANIRA_EDGE_COST_PERMISSIVE || cost == ANIRA_EDGE_COST_STRICT;
}
bool valid_wait_strategy(anira_wait_strategy wait) {
    return wait == ANIRA_WAIT_SPIN_BACKOFF || wait == ANIRA_WAIT_BLOCKING;
}
bool valid_log_level(anira_log_level level) {
    return level >= ANIRA_LOG_DEBUG && level <= ANIRA_LOG_ERROR;
}
bool valid_log_drain(anira_log_drain drain) {
    return drain == ANIRA_LOG_DRAIN_THREAD || drain == ANIRA_LOG_DRAIN_MANUAL;
}
bool valid_model_state(anira_model_state state) {
    return state == ANIRA_MODEL_STATELESS || state == ANIRA_MODEL_STATEFUL;
}
bool valid_pad_policy(anira_pad_policy policy) {
    return policy == ANIRA_PAD_REJECT || policy == ANIRA_PAD_ZEROS;
}
bool custom_engine_id(const char* engine_id) {
    return engine_id != nullptr && std::strchr(engine_id, '.') != nullptr;
}
bool non_empty(const char* text) {
    return text != nullptr && text[0] != '\0';
}

// Canonical names are unique across both sides: the per-entry tensor records and the anchor
// refer to a tensor by bare name.
bool name_taken(const anira_model_config& config, const std::string& name) {
    for (const anira_tensor_spec& spec : config.m_inputs) {
        if (spec.m_name == name) { return true; }
    }
    for (const anira_tensor_spec& spec : config.m_outputs) {
        if (spec.m_name == name) { return true; }
    }
    return false;
}

// A Tier-2 descriptor handed once to a setter: NULL clears the block, a short one is
// refused, a longer one (a newer header) is read within this build's size.
template <class T>
anira_status copy_desc(std::optional<T>& slot, const T* desc, const T& defaults) {
    if (desc == nullptr) {
        slot.reset();
        return ANIRA_OK;
    }
    if (desc->struct_size < sizeof(uint32_t)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    T value = defaults;
    const size_t readable = std::min<size_t>(desc->struct_size, sizeof(T));
    std::memcpy(&value, desc, readable);
    value.struct_size = sizeof(T);
    slot = value;
    return ANIRA_OK;
}

anira_status set_ext_json(anira::capi::ExtBag& bag,
                          const char* kind,
                          const char* utf8,
                          size_t len,
                          anira_error* err) {
    ANIRA_CAPI_REQUIRE(utf8 != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "extension: NULL JSON text");
    return bag.set_json(kind, std::string_view(utf8, len), err);
}

std::shared_ptr<anira::capi::BytesCarrier> make_carrier(const void* bytes,
                                                        size_t size,
                                                        anira_bytes_ownership ownership,
                                                        anira_bytes_release_fn release,
                                                        void* ctx) {
    return std::make_shared<anira::capi::BytesCarrier>(bytes, size, ownership, release, ctx);
}

}  // namespace

// ==== registry ==============================================================================

anira_status ANIRA_CALL anira_registered_ext_kinds(uint32_t* count, const char** out) ANIRA_NOEXCEPT
    try {
    if (count == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::vector<const char*>& kinds = anira::capi::ext_kinds();
    const auto total = static_cast<uint32_t>(kinds.size());
    if (out == nullptr) {
        *count = total;
        return ANIRA_OK;
    }
    const uint32_t capacity = *count;
    const uint32_t written = std::min(capacity, total);
    for (uint32_t i = 0; i < written; ++i) { out[i] = kinds[i]; }
    *count = total;
    return capacity < total ? ANIRA_INCOMPLETE : ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

// ==== tensor spec ===========================================================================

anira_status ANIRA_CALL anira_tensor_spec_create(const char* name,
                                                 anira_dtype dtype,
                                                 anira_role role,
                                                 anira_tensor_spec** out,
                                                 anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "tensor spec: NULL out");
    ANIRA_CAPI_REQUIRE(non_empty(name),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "tensor spec: NULL or empty name");
    ANIRA_CAPI_REQUIRE(valid_role(role),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "tensor '%s': unknown role %d",
                       name,
                       static_cast<int>(role));
    auto spec = std::make_unique<anira_tensor_spec>();
    spec->m_name = name;
    spec->m_dtype = dtype;
    spec->m_role = role;
    *out = spec.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_axis(anira_tensor_spec* spec,
                                                   uint32_t i,
                                                   anira_axis_tag tag,
                                                   int64_t extent) ANIRA_NOEXCEPT try {
    if (spec == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (i >= ANIRA_MAX_RANK || !valid_axis_tag(tag)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (extent <= 0 && extent != ANIRA_DYNAMIC) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    spec->m_axes[i] = anira::capi::Axis{.m_tag = tag, .m_extent = extent, .m_written = true};
    spec->m_ndim = std::max(spec->m_ndim, i + 1);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_window(anira_tensor_spec* spec,
                                                     int64_t window_min,
                                                     int64_t window_max,
                                                     int64_t context) ANIRA_NOEXCEPT try {
    if (spec == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (window_min < 0 || context < 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (window_max < 0 && window_max != ANIRA_UNBOUNDED) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    spec->m_window_min = window_min;
    spec->m_window_max = window_max;
    spec->m_context = context;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_time_ratio(anira_tensor_spec* spec,
                                                         int64_t num,
                                                         int64_t den) ANIRA_NOEXCEPT try {
    if (spec == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (num < 0 || den < 0 || (den == 0 && num != 0)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    spec->m_ratio_num = num;
    spec->m_ratio_den = den;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_latency(anira_tensor_spec* spec,
                                                      int64_t latency) ANIRA_NOEXCEPT try {
    if (spec == nullptr || latency < 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    spec->m_latency = latency;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_ext(anira_tensor_spec* spec,
                                                  const anira_ext_header* ext,
                                                  anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(spec != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "tensor spec: NULL handle");
    return spec->m_ext.set(ext, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_tensor_spec_set_ext_json(anira_tensor_spec* spec,
                                                       const char* kind,
                                                       const char* utf8,
                                                       size_t len,
                                                       anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(spec != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "tensor spec: NULL handle");
    return set_ext_json(spec->m_ext, kind, utf8, len, err);
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_tensor_spec_destroy(anira_tensor_spec* spec) ANIRA_NOEXCEPT try {
    delete spec;
} catch (...) { report_void_failure(__func__); }

// ==== contract ==============================================================================

anira_status ANIRA_CALL anira_contract_create_hard(uint32_t block_min,
                                                   uint32_t block_max,
                                                   double rate,
                                                   anira_contract** out,
                                                   anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "contract: NULL out");
    ANIRA_CAPI_REQUIRE(block_min <= block_max,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "contract: block_min %u > block_max %u",
                       static_cast<unsigned>(block_min),
                       static_cast<unsigned>(block_max));
    ANIRA_CAPI_REQUIRE(rate >= 0.0, err, ANIRA_ERROR_INVALID_ARGUMENT, "contract: negative rate");
    auto contract = std::make_unique<anira_contract>();
    anira::capi::HardContract hard;
    hard.m_block_min = block_min;
    hard.m_block_max = block_max;
    hard.m_rate = rate;
    contract->m_kind = hard;
    *out = contract.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_contract_create_async(anira_contract** out,
                                                    anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "contract: NULL out");
    auto contract = std::make_unique<anira_contract>();
    contract->m_kind = anira::capi::AsyncContract{};
    *out = contract.release();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_geometry(anira_contract* contract,
                                                         uint32_t block_min,
                                                         uint32_t block_max,
                                                         double rate) ANIRA_NOEXCEPT try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (block_min > block_max || rate < 0.0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    hard->m_block_min = block_min;
    hard->m_block_max = block_max;
    hard->m_rate = rate;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_budget(anira_contract* contract,
                                                       anira_budget_kind kind,
                                                       double explicit_ms) ANIRA_NOEXCEPT try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (!valid_budget_kind(kind)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (kind == ANIRA_BUDGET_EXPLICIT && !(explicit_ms > 0.0)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    hard->m_budget = kind;
    hard->m_budget_ms = kind == ANIRA_BUDGET_EXPLICIT ? explicit_ms : 0.0;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_warmup(anira_contract* contract,
                                                       anira_warmup_mode mode,
                                                       uint32_t iterations) ANIRA_NOEXCEPT try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (!valid_warmup_mode(mode)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    hard->m_warmup = mode;
    hard->m_warmup_iterations = mode == ANIRA_WARMUP_FIXED ? iterations : 0;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_on_miss(anira_contract* contract,
                                                        anira_miss_policy policy) ANIRA_NOEXCEPT
    try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (!valid_miss_policy(policy)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    hard->m_on_miss = policy;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_wait_ratio(anira_contract* contract,
                                                           double ratio) ANIRA_NOEXCEPT try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (!(ratio >= 0.0)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    hard->m_wait_ratio = ratio;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_hard_set_ring_dtype(anira_contract* contract,
                                                           const char* canonical,
                                                           anira_dtype dtype) ANIRA_NOEXCEPT try {
    if (contract == nullptr || !non_empty(canonical) || dtype == 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira::capi::HardContract* hard = contract->hard();
    if (hard == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    hard->m_ring_dtypes[canonical] = dtype;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_async_set_deadline(anira_contract* contract,
                                                          double deadline_ms) ANIRA_NOEXCEPT try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::AsyncContract* async_part = contract->asynchronous();
    if (async_part == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    async_part->m_deadline_ms = deadline_ms;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_async_set_policy(anira_contract* contract,
                                                        anira_late_policy on_late,
                                                        anira_priority priority,
                                                        uint32_t lanes,
                                                        uint32_t max_in_flight,
                                                        anira_delivery delivery) ANIRA_NOEXCEPT
    try {
    if (contract == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::AsyncContract* async_part = contract->asynchronous();
    if (async_part == nullptr) { return ANIRA_ERROR_WRONG_CONTRACT; }
    if (!valid_late_policy(on_late) || !valid_priority(priority) || !valid_delivery(delivery)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    async_part->m_on_late = on_late;
    async_part->m_priority = priority;
    async_part->m_lanes = lanes;
    async_part->m_max_in_flight = max_in_flight;
    async_part->m_delivery = delivery;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_set_edge_cost(anira_contract* contract,
                                                     anira_edge_cost cost) ANIRA_NOEXCEPT try {
    if (contract == nullptr || !valid_edge_cost(cost)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    contract->m_edge_cost = cost;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_contract_set_ext(anira_contract* contract,
                                               const anira_ext_header* ext,
                                               anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(contract != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "contract: NULL handle");
    return contract->m_ext.set(ext, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_contract_set_ext_json(anira_contract* contract,
                                                    const char* kind,
                                                    const char* utf8,
                                                    size_t len,
                                                    anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(contract != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "contract: NULL handle");
    return set_ext_json(contract->m_ext, kind, utf8, len, err);
} catch (...) { return translate_exception(err, __func__); }

anira_contract_kind ANIRA_CALL anira_contract_get_kind(const anira_contract* contract)
    ANIRA_NOEXCEPT {
    if (contract == nullptr) { return ANIRA_CONTRACT_HARD; }
    return contract->is_hard() ? ANIRA_CONTRACT_HARD : ANIRA_CONTRACT_ASYNC;
}

void ANIRA_CALL anira_contract_destroy(anira_contract* contract) ANIRA_NOEXCEPT try {
    delete contract;
} catch (...) { report_void_failure(__func__); }

// ==== machine config ========================================================================

namespace {
/// The ANIRA_LOG_FLAG_* bits a machine config accepts; the machine applies both while it
/// lives (the platform sink off, the boundary trace on).
constexpr uint32_t k_known_log_flags =
    ANIRA_LOG_FLAG_DISABLE_PLATFORM_SINK | ANIRA_LOG_FLAG_TRACE_FAILURES;
}  // namespace

anira_status ANIRA_CALL anira_machine_config_create(anira_machine_config** out,
                                                    anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "machine config: NULL out");
    *out = new anira_machine_config();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_threads(anira_machine_config* config,
                                                         uint32_t num_threads,
                                                         anira_wait_strategy wait) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || !valid_wait_strategy(wait)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_num_threads = num_threads;
    config->m_wait = wait;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log_level(anira_machine_config* config,
                                                           anira_log_level level) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || !valid_log_level(level)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_log_level = level;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log_drain(anira_machine_config* config,
                                                           anira_log_drain drain,
                                                           uint32_t interval_ms) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || !valid_log_drain(drain)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_log_drain = drain;
    config->m_drain_interval_ms = interval_ms == 0 ? 10 : interval_ms;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log_queue_capacity(anira_machine_config* config,
                                                                    uint32_t capacity)
    ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_queue_capacity = std::clamp<uint32_t>(capacity, 64, 65536);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log_flags(anira_machine_config* config,
                                                           uint32_t flags) ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if ((flags & ~k_known_log_flags) != 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_log_flags = flags;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log_sink(anira_machine_config* config,
                                                          anira_log_fn callback,
                                                          void* user_data) ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_sink = callback;
    config->m_sink_user_data = callback != nullptr ? user_data : nullptr;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_log(anira_machine_config* config,
                                                     const anira_log_desc* desc) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || desc == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    // The three leading slots {struct_size, abi_version, user_data} are the minimum.
    if (desc->struct_size < offsetof(anira_log_desc, callback)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_log_desc value = ANIRA_LOG_DESC_INIT;
    std::memcpy(&value, desc, std::min<size_t>(desc->struct_size, sizeof(anira_log_desc)));
    if (ANIRA_FAILED(anira_check_abi(value.abi_version))) { return ANIRA_ERROR_ABI_VERSION; }
    if (!valid_log_level(static_cast<anira_log_level>(value.level)) ||
        !valid_log_drain(static_cast<anira_log_drain>(value.drain)) ||
        (value.flags & ~k_known_log_flags) != 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    config->m_log_level = static_cast<anira_log_level>(value.level);
    config->m_log_drain = static_cast<anira_log_drain>(value.drain);
    config->m_queue_capacity = std::clamp<uint32_t>(value.queue_capacity, 64, 65536);
    config->m_drain_interval_ms = value.drain_interval_ms == 0 ? 10 : value.drain_interval_ms;
    config->m_log_flags = value.flags;
    config->m_sink = value.callback;
    config->m_sink_user_data = value.callback != nullptr ? value.user_data : nullptr;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_cuda(anira_machine_config* config,
                                                      const anira_cuda_desc* desc) ANIRA_NOEXCEPT
    try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    static const anira_cuda_desc k_defaults = ANIRA_CUDA_DESC_INIT;
    return copy_desc(config->m_cuda, desc, k_defaults);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_gl(anira_machine_config* config,
                                                    const anira_gl_desc* desc) ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    static const anira_gl_desc k_defaults = ANIRA_GL_DESC_INIT;
    return copy_desc(config->m_gl, desc, k_defaults);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_vulkan(anira_machine_config* config,
                                                        const anira_vulkan_desc* desc)
    ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    static const anira_vulkan_desc k_defaults = ANIRA_VULKAN_DESC_INIT;
    return copy_desc(config->m_vulkan, desc, k_defaults);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_metal(anira_machine_config* config,
                                                       const anira_metal_desc* desc) ANIRA_NOEXCEPT
    try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    static const anira_metal_desc k_defaults = ANIRA_METAL_DESC_INIT;
    return copy_desc(config->m_metal, desc, k_defaults);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_d3d12(anira_machine_config* config,
                                                       const anira_d3d12_desc* desc) ANIRA_NOEXCEPT
    try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    static const anira_d3d12_desc k_defaults = ANIRA_D3D12_DESC_INIT;
    return copy_desc(config->m_d3d12, desc, k_defaults);
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_webgpu(anira_machine_config* config,
                                                        const anira_webgpu_desc* desc)
    ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
#if defined(__EMSCRIPTEN__)
    static_cast<void>(desc);
    return ANIRA_ERROR_NOT_SUPPORTED;
#else
    static const anira_webgpu_desc k_defaults = ANIRA_WEBGPU_DESC_INIT;
    return copy_desc(config->m_webgpu, desc, k_defaults);
#endif
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_ext(anira_machine_config* config,
                                                     const anira_ext_header* ext,
                                                     anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "machine config: NULL handle");
    return config->m_ext.set(ext, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_machine_config_set_ext_json(anira_machine_config* config,
                                                          const char* kind,
                                                          const char* utf8,
                                                          size_t len,
                                                          anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "machine config: NULL handle");
    return set_ext_json(config->m_ext, kind, utf8, len, err);
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_machine_config_destroy(anira_machine_config* config) ANIRA_NOEXCEPT try {
    delete config;
} catch (...) { report_void_failure(__func__); }

// ==== model config ==========================================================================

anira_status ANIRA_CALL anira_model_config_create(anira_model_config** out,
                                                  anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model config: NULL out");
    *out = new anira_model_config();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

namespace {

anira_status add_entry(anira_model_config* config,
                       anira::capi::ModelEntry entry,
                       uint32_t* out_index,
                       anira_error* err) {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    config->m_models.push_back(std::move(entry));
    if (out_index != nullptr) { *out_index = static_cast<uint32_t>(config->m_models.size() - 1); }
    return ANIRA_OK;
}

}  // namespace

anira_status ANIRA_CALL anira_model_config_add_model_path(anira_model_config* config,
                                                          anira_engine engine,
                                                          const char* utf8_path,
                                                          uint32_t* out_index,
                                                          anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(builtin_engine(engine),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: engine %d is not a built-in engine",
                       static_cast<int>(engine));
    ANIRA_CAPI_REQUIRE(non_empty(utf8_path),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: NULL or empty path");
    anira::capi::ModelEntry entry;
    entry.m_engine = engine;
    entry.m_path = utf8_path;
    return add_entry(config, std::move(entry), out_index, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_add_model_bytes(anira_model_config* config,
                                                           anira_engine engine,
                                                           const void* bytes,
                                                           size_t size,
                                                           anira_bytes_ownership ownership,
                                                           anira_bytes_release_fn release,
                                                           void* ctx,
                                                           uint32_t* out_index,
                                                           anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(builtin_engine(engine),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: engine %d is not a built-in engine",
                       static_cast<int>(engine));
    ANIRA_CAPI_REQUIRE(bytes != nullptr && size > 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: NULL bytes or zero size");
    ANIRA_CAPI_REQUIRE(valid_bytes_ownership(ownership),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: unknown ownership %d",
                       static_cast<int>(ownership));
    anira::capi::ModelEntry entry;
    entry.m_engine = engine;
    entry.m_bytes = make_carrier(bytes, size, ownership, release, ctx);
    return add_entry(config, std::move(entry), out_index, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_add_model_path_custom(anira_model_config* config,
                                                                 const char* engine_id,
                                                                 const char* utf8_path,
                                                                 uint32_t* out_index,
                                                                 anira_error* err) ANIRA_NOEXCEPT
    try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(custom_engine_id(engine_id),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: a custom engine id is reverse-URI (contains a '.')");
    ANIRA_CAPI_REQUIRE(non_empty(utf8_path),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: NULL or empty path");
    anira::capi::ModelEntry entry;
    entry.m_engine_id = engine_id;
    entry.m_path = utf8_path;
    return add_entry(config, std::move(entry), out_index, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_add_model_bytes_custom(anira_model_config* config,
                                                                  const char* engine_id,
                                                                  const void* bytes,
                                                                  size_t size,
                                                                  anira_bytes_ownership ownership,
                                                                  anira_bytes_release_fn release,
                                                                  void* ctx,
                                                                  uint32_t* out_index,
                                                                  anira_error* err) ANIRA_NOEXCEPT
    try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(custom_engine_id(engine_id),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: a custom engine id is reverse-URI (contains a '.')");
    ANIRA_CAPI_REQUIRE(bytes != nullptr && size > 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: NULL bytes or zero size");
    ANIRA_CAPI_REQUIRE(valid_bytes_ownership(ownership),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: unknown ownership %d",
                       static_cast<int>(ownership));
    anira::capi::ModelEntry entry;
    entry.m_engine_id = engine_id;
    entry.m_bytes = make_carrier(bytes, size, ownership, release, ctx);
    return add_entry(config, std::move(entry), out_index, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_set_model_bytes(anira_model_config* config,
                                                           uint32_t model_index,
                                                           const void* bytes,
                                                           size_t size,
                                                           anira_bytes_ownership ownership,
                                                           anira_bytes_release_fn release,
                                                           void* ctx,
                                                           anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(model_index < config->m_models.size(),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model index %u out of range (%u entries)",
                       static_cast<unsigned>(model_index),
                       static_cast<unsigned>(config->m_models.size()));
    ANIRA_CAPI_REQUIRE(bytes != nullptr && size > 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: NULL bytes or zero size");
    ANIRA_CAPI_REQUIRE(valid_bytes_ownership(ownership),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model entry: unknown ownership %d",
                       static_cast<int>(ownership));
    config->m_models[model_index].m_bytes = make_carrier(bytes, size, ownership, release, ctx);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

uint32_t ANIRA_CALL anira_model_config_model_count(const anira_model_config* config)
    ANIRA_NOEXCEPT {
    return config == nullptr ? 0u : static_cast<uint32_t>(config->m_models.size());
}

anira_engine ANIRA_CALL anira_model_config_model_engine(const anira_model_config* config,
                                                        uint32_t model_index) ANIRA_NOEXCEPT {
    if (config == nullptr || model_index >= config->m_models.size()) { return ANIRA_ENGINE_NONE; }
    return config->m_models[model_index].m_engine;
}

const char* ANIRA_CALL anira_model_config_model_engine_id(const anira_model_config* config,
                                                          uint32_t model_index) ANIRA_NOEXCEPT {
    if (config == nullptr || model_index >= config->m_models.size()) { return nullptr; }
    const anira::capi::ModelEntry& entry = config->m_models[model_index];
    return entry.is_custom() ? entry.m_engine_id.c_str() : nullptr;
}

const char* ANIRA_CALL anira_model_config_model_path(const anira_model_config* config,
                                                     uint32_t model_index) ANIRA_NOEXCEPT {
    if (config == nullptr || model_index >= config->m_models.size()) { return nullptr; }
    const anira::capi::ModelEntry& entry = config->m_models[model_index];
    return entry.has_bytes() || entry.m_path.empty() ? nullptr : entry.m_path.c_str();
}

anira_status ANIRA_CALL anira_model_config_model_bytes(const anira_model_config* config,
                                                       uint32_t model_index,
                                                       const void** bytes,
                                                       size_t* size) ANIRA_NOEXCEPT try {
    if (config == nullptr || bytes == nullptr || size == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (model_index >= config->m_models.size()) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const anira::capi::ModelEntry& entry = config->m_models[model_index];
    if (!entry.has_bytes()) { return ANIRA_ERROR_INVALID_STATE; }
    *bytes = entry.m_bytes->data();
    *size = entry.m_bytes->size();
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_tensor_name(anira_model_config* config,
                                                           uint32_t model_index,
                                                           const char* canonical,
                                                           const char* engine_name) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || model_index >= config->m_models.size()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!non_empty(canonical) || !non_empty(engine_name)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_models[model_index].m_tensors[canonical].m_name = engine_name;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_tensor_layout(anira_model_config* config,
                                                             uint32_t model_index,
                                                             const char* canonical,
                                                             const uint32_t* axes,
                                                             uint32_t ndim) ANIRA_NOEXCEPT try {
    if (config == nullptr || model_index >= config->m_models.size()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!non_empty(canonical)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira::capi::ModelEntry& entry = config->m_models[model_index];
    if (ndim == 0) {
        if (axes != nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const auto record = entry.m_tensors.find(canonical);
        if (record != entry.m_tensors.end()) {
            record->second.m_layout.clear();
            if (record->second.m_name.empty()) { entry.m_tensors.erase(record); }
        }
        return ANIRA_OK;
    }
    if (axes == nullptr || ndim > ANIRA_MAX_RANK) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::vector<uint32_t> layout(axes, axes + ndim);
    if (!anira::capi::valid_layout_shape(layout, nullptr)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    entry.m_tensors[canonical].m_layout = layout;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_model_ext(anira_model_config* config,
                                                         uint32_t model_index,
                                                         const anira_ext_header* ext,
                                                         anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(model_index < config->m_models.size(),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model index %u out of range (%u entries)",
                       static_cast<unsigned>(model_index),
                       static_cast<unsigned>(config->m_models.size()));
    return config->m_models[model_index].m_ext.set(ext, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_set_model_ext_json(anira_model_config* config,
                                                              uint32_t model_index,
                                                              const char* kind,
                                                              const char* utf8,
                                                              size_t len,
                                                              anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    ANIRA_CAPI_REQUIRE(model_index < config->m_models.size(),
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model index %u out of range (%u entries)",
                       static_cast<unsigned>(model_index),
                       static_cast<unsigned>(config->m_models.size()));
    return set_ext_json(config->m_models[model_index].m_ext, kind, utf8, len, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_add_input(anira_model_config* config,
                                                     const anira_tensor_spec* spec) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || spec == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (name_taken(*config, spec->m_name)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_inputs.push_back(*spec);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_add_output(anira_model_config* config,
                                                      const anira_tensor_spec* spec) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || spec == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (name_taken(*config, spec->m_name)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_outputs.push_back(*spec);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_default_engine(anira_model_config* config,
                                                              anira_engine engine) ANIRA_NOEXCEPT
    try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (engine != ANIRA_ENGINE_NONE && !builtin_engine(engine)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    config->m_default_engine = engine;
    config->m_default_engine_id.clear();
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_default_engine_custom(anira_model_config* config,
                                                                     const char* engine_id)
    ANIRA_NOEXCEPT try {
    if (config == nullptr || !custom_engine_id(engine_id)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_default_engine = ANIRA_ENGINE_NONE;
    config->m_default_engine_id = engine_id;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_state(anira_model_config* config,
                                                     anira_model_state state) ANIRA_NOEXCEPT try {
    if (config == nullptr || !valid_model_state(state)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_state = state;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_max_instances(anira_model_config* config,
                                                             uint32_t max_instances) ANIRA_NOEXCEPT
    try {
    if (config == nullptr || max_instances == 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_max_instances = max_instances;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_anchor(anira_model_config* config,
                                                      const char* canonical) ANIRA_NOEXCEPT try {
    if (config == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    config->m_anchor = non_empty(canonical) ? canonical : "";
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_model_config_set_ext(anira_model_config* config,
                                                   const anira_ext_header* ext,
                                                   anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    return config->m_ext.set(ext, err);
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_model_config_set_ext_json(anira_model_config* config,
                                                        const char* kind,
                                                        const char* utf8,
                                                        size_t len,
                                                        anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(config != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "model config: NULL handle");
    return set_ext_json(config->m_ext, kind, utf8, len, err);
} catch (...) { return translate_exception(err, __func__); }

void ANIRA_CALL anira_model_config_destroy(anira_model_config* config) ANIRA_NOEXCEPT try {
    delete config;
} catch (...) { report_void_failure(__func__); }

// ==== job options ===========================================================================

anira_status ANIRA_CALL anira_job_options_create(anira_job_options** out,
                                                 anira_error* err) ANIRA_NOEXCEPT try {
    ANIRA_CAPI_REQUIRE(out != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "job options: NULL out");
    *out = new anira_job_options();
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status ANIRA_CALL anira_job_options_set_head_trim(anira_job_options* options,
                                                        uint32_t count,
                                                        const int64_t* trims) ANIRA_NOEXCEPT try {
    if (options == nullptr || (count > 0 && trims == nullptr)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (trims[i] < -1) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    }
    options->m_head_trim.assign(trims, trims + count);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_job_options_set_tail_flush(anira_job_options* options,
                                                         anira_bool tail_flush) ANIRA_NOEXCEPT try {
    if (options == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    options->m_tail_flush = tail_flush != 0;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_job_options_set_below_min(anira_job_options* options,
                                                        anira_pad_policy policy) ANIRA_NOEXCEPT
    try {
    if (options == nullptr || !valid_pad_policy(policy)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    options->m_below_min = policy;
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_job_options_set_ext(anira_job_options* options,
                                                  const anira_ext_header* ext) ANIRA_NOEXCEPT try {
    if (options == nullptr || ext == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (ext->struct_size < sizeof(anira_ext_header) || !non_empty(ext->kind)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    // Borrowed: one slot per kind, a second set of the same kind replaces the pointer.
    for (const anira_ext_header*& borrowed : options->m_borrowed_ext) {
        if (std::strcmp(borrowed->kind, ext->kind) == 0) {
            borrowed = ext;
            return ANIRA_OK;
        }
    }
    options->m_borrowed_ext.push_back(ext);
    return ANIRA_OK;
} catch (...) { return translate_exception(nullptr, __func__); }

anira_status ANIRA_CALL anira_job_options_set_ext_json(anira_job_options* options,
                                                       const char* kind,
                                                       const char* utf8,
                                                       size_t len) ANIRA_NOEXCEPT try {
    if (options == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    return set_ext_json(options->m_json_ext, kind, utf8, len, nullptr);
} catch (...) { return translate_exception(nullptr, __func__); }

void ANIRA_CALL anira_job_options_destroy(anira_job_options* options) ANIRA_NOEXCEPT try {
    delete options;
} catch (...) { report_void_failure(__func__); }
