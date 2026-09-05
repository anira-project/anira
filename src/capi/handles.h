#ifndef ANIRA_CAPI_HANDLES_H
#define ANIRA_CAPI_HANDLES_H

/*
 * The bodies of the opaque configuration handles of anira/abi/config.h. Private to
 * src/capi (and the tests through the src/ include directory): the layouts never enter
 * the ABI, which is what lets them change in any release. Every C setter is a thin,
 * firewalled wrapper over a member here, so the JSON loaders and the translator share the
 * same code paths.
 */

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "ext_registry.h"

namespace anira::capi {

/// One axis of a tensor spec; written says whether set_axis touched the slot.
struct Axis {
    anira_axis_tag m_tag = ANIRA_AXIS_ANY;
    int64_t m_extent = 0;
    bool m_written = false;
};

/// Model bytes with their ownership: COPY holds a private copy, BORROW points at the
/// caller's memory and fires the release callback exactly once, when the last carrier
/// (config, and later handler and pooled processor) dies.
class BytesCarrier {
public:
    BytesCarrier(const void* bytes,
                 size_t size,
                 anira_bytes_ownership ownership,
                 anira_bytes_release_fn release,
                 void* ctx);
    ~BytesCarrier();
    BytesCarrier(const BytesCarrier&) = delete;
    BytesCarrier& operator=(const BytesCarrier&) = delete;

    const void* data() const noexcept { return m_bytes; }
    size_t size() const noexcept { return m_size; }
    anira_bytes_ownership ownership() const noexcept { return m_ownership; }

private:
    std::vector<unsigned char> m_copy;
    const void* m_bytes = nullptr;
    size_t m_size = 0;
    anira_bytes_ownership m_ownership = ANIRA_BYTES_COPY;
    anira_bytes_release_fn m_release = nullptr;
    void* m_ctx = nullptr;
};

/// One models[] entry: a built-in engine or a custom engine id, a path or bytes, the
/// canonical -> engine tensor names, and its extensions (host "model").
/// What one entry's file calls a tensor and how it holds its axes (section 5): the
/// JSON file's models[].tensors record, keyed by the spec's canonical name.
struct TensorBinding {
    std::string m_name;              ///< the export's name; empty = bind positionally
    std::vector<uint32_t> m_layout;  ///< engine axis k = spec axis m_layout[k]; empty = identity
};

struct ModelEntry {
    anira_engine m_engine = ANIRA_ENGINE_NONE;
    std::string m_engine_id;  ///< non-empty for a custom engine
    std::string m_path;       ///< kept for to_json even after set_model_bytes
    std::shared_ptr<BytesCarrier> m_bytes;
    std::map<std::string, TensorBinding> m_tensors;
    ExtBag m_ext;

    bool is_custom() const noexcept { return !m_engine_id.empty(); }
    bool has_bytes() const noexcept { return m_bytes != nullptr; }
};

/// The Hard (real-time) half of a contract: v2's HostConfig geometry, budget and warmup.
/// Namespace-level (not nested in anira_contract): clang parses a nested class's default
/// member initializers only once the enclosing class is complete, which would leave
/// std::variant without a default constructor.
struct HardContract {
    uint32_t m_block_min = 0;
    uint32_t m_block_max = 0;
    double m_rate = 0.0;
    anira_budget_kind m_budget = ANIRA_BUDGET_MEASURED;
    double m_budget_ms = 0.0;
    anira_warmup_mode m_warmup = ANIRA_WARMUP_UNTIL_STABLE;
    uint32_t m_warmup_iterations = 0;
    anira_miss_policy m_on_miss = ANIRA_MISS_BYPASS;
    double m_wait_ratio = 0.0;
    /// Per-tensor ring dtype by canonical name (the host's element type, held as is by the
    /// ring; pre/post convert to the spec's dtype); absent = ANIRA_DTYPE_F32. Data only at
    /// M1: the bridge to the 2.x runtime refuses anything but F32.
    std::map<std::string, anira_dtype> m_ring_dtypes;
};

/// The Async half of a contract.
struct AsyncContract {
    double m_deadline_ms = -1.0;
    anira_late_policy m_on_late = ANIRA_LATE_FINISH;
    anira_priority m_priority = ANIRA_PRIORITY_AUTO;
    uint32_t m_lanes = 0;
    uint32_t m_max_in_flight = 0;
    anira_delivery m_delivery = ANIRA_DELIVERY_POLLED;
};

}  // namespace anira::capi

// The handle bodies carry the C tag names the header forward-declares.
// NOLINTBEGIN(readability-identifier-naming)

struct anira_tensor_spec {
    std::string m_name;
    anira_dtype m_dtype = 0;
    anira_role m_role = ANIRA_ROLE_STREAMED;
    std::array<anira::capi::Axis, ANIRA_MAX_RANK> m_axes{};
    uint32_t m_ndim = 0;
    int64_t m_window_min = 0;
    int64_t m_window_max = 0;
    int64_t m_overlap = 0;
    int64_t m_ratio_num = 0;
    int64_t m_ratio_den = 0;
    int64_t m_latency = 0;
    anira::capi::ExtBag m_ext;
};

struct anira_contract {
    std::variant<anira::capi::HardContract, anira::capi::AsyncContract> m_kind;
    anira_edge_cost m_edge_cost = ANIRA_EDGE_COST_PERMISSIVE;
    anira::capi::ExtBag m_ext;
    bool m_legacy = false;  ///< produced by the version 2 JSON upgrade

    bool is_hard() const noexcept {
        return std::holds_alternative<anira::capi::HardContract>(m_kind);
    }
    anira::capi::HardContract* hard() noexcept {
        return std::get_if<anira::capi::HardContract>(&m_kind);
    }
    const anira::capi::HardContract* hard() const noexcept {
        return std::get_if<anira::capi::HardContract>(&m_kind);
    }
    anira::capi::AsyncContract* asynchronous() noexcept {
        return std::get_if<anira::capi::AsyncContract>(&m_kind);
    }
    const anira::capi::AsyncContract* asynchronous() const noexcept {
        return std::get_if<anira::capi::AsyncContract>(&m_kind);
    }
};

struct anira_model_config {
    std::vector<anira::capi::ModelEntry> m_models;
    std::vector<anira_tensor_spec> m_inputs;
    std::vector<anira_tensor_spec> m_outputs;
    anira_engine m_default_engine = ANIRA_ENGINE_NONE;
    std::string m_default_engine_id;
    anira_model_state m_state = ANIRA_MODEL_STATELESS;
    uint32_t m_max_instances = 1;
    std::string m_anchor;  ///< canonical name of the clock tensor; empty = the first streamed
    anira::capi::ExtBag m_ext;
    std::unique_ptr<anira_contract> m_legacy_contract;  ///< after a version 2 JSON upgrade
    bool m_upgraded = false;
};

struct anira_context_config {
    uint32_t m_num_threads = ANIRA_THREADS_AUTO;
    anira_wait_strategy m_wait = ANIRA_WAIT_SPIN_BACKOFF;
    anira_log_level m_log_level = ANIRA_LOG_WARNING;
    anira_log_drain m_log_drain = ANIRA_LOG_DRAIN_THREAD;
    uint32_t m_drain_interval_ms = 10;
    uint32_t m_queue_capacity = 512;
    uint32_t m_log_flags = 0;
    anira_log_fn m_sink = nullptr;
    void* m_sink_user_data = nullptr;
    std::optional<anira_cuda_desc> m_cuda;
    std::optional<anira_gl_desc> m_gl;
    std::optional<anira_vulkan_desc> m_vulkan;
    int32_t m_vulkan_device = 0;  ///< the JSON file's "vulkan": {"device"} index; the descriptor
                                  ///< has no slot for it
    std::optional<anira_metal_desc> m_metal;
    std::optional<anira_d3d12_desc> m_d3d12;
    std::optional<anira_webgpu_desc> m_webgpu;
    anira::capi::ExtBag m_ext;
    bool m_upgraded = false;
};

struct anira_job_options {
    std::vector<int64_t> m_head_trim;
    bool m_tail_flush = true;
    anira_pad_policy m_below_min = ANIRA_PAD_REJECT;
    std::vector<const anira_ext_header*> m_borrowed_ext;  ///< set_ext: borrowed until submit
                                                          ///< returns
    anira::capi::ExtBag m_json_ext;                       ///< set_ext_json: owned
};

// NOLINTEND(readability-identifier-naming)

#endif  // ANIRA_CAPI_HANDLES_H
