/**
 * @file v3_to_v2.h
 * @brief The transitional bridge from the 3.x configuration handles to the 2.x runtime.
 *
 * In this pre-release the runtime (anira::InferenceHandler, prepare, process) still takes the
 * 2.x configuration classes. This header turns a 3.x configuration (a model config, a Hard
 * contract, a machine config) into them: anira::InferenceConfig, anira::ContextConfig and
 * anira::HostConfig. It validates the section-2 rules the 2.x runtime can honour and refuses,
 * with ANIRA_ERROR_NOT_SUPPORTED, what the 2.x runtime cannot do (an Async contract, a
 * MEASURED budget, UNTIL_STABLE warmup, a miss policy other than BYPASS, a dtype other than
 * float32, a layout that transposes a material axis, an engine this build does not carry).
 *
 * @warning Transitional: v3.0.0-alpha.1 only. The 3.x handler of the next pre-release takes the
 * handles directly, and this header is removed with it. Nothing here is part of the C ABI.
 *
 * Lifetime rules. A model entry added by path is copied into the InferenceConfig; a model entry
 * added by bytes is borrowed (a 2.x binary ModelData never copies): the config handle must
 * outlive the InferenceConfig and every handler built on it. Destroy in this order: the
 * handler, the InferenceConfig, the handle. The candidate list narrows which entries the
 * InferenceConfig receives; with none, every entry is one, and an entry naming an engine this
 * build does not carry is refused (enabled_engines() is the list that admits every build).
 *
 * The window a flexible spec (window_min < window_max) uses is pinned from the contract's
 * geometry when to_inference_config runs; set the geometry (anira_contract_hard_set_geometry,
 * anira::ContractHandle::hard_geometry) before the call, or the smallest window is used.
 *
 * Every entry is [main-thread], may allocate, returns the status and writes `err` on failure
 * (never throws; the exception firewall of the C entries applies). The C++ overloads over the
 * anira.hpp handles return the 2.x object and throw anira::Error instead.
 */
#ifndef ANIRA_COMPAT_V3_TO_V2_H
#define ANIRA_COMPAT_V3_TO_V2_H

#include <anira/ContextConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/system/Exports.h>
#include <anira/utils/HostConfig.h>

#include <cstdint>
#include <vector>

namespace anira::v3compat {

/**
 * @brief The 2.x InferenceConfig of a model config under a Hard contract.
 *
 * Model entries become ModelData (path copied, bytes borrowed, the `entry` extension the
 * model function); the specs become one universal TensorShape (a dynamic Time extent
 * resolved to the pinned window) plus one backend-qualified TensorShape per entry whose
 * tensor record holds a layout; the Channel extents, the hops (window minus context; 0 for a
 * Static or Buffer tensor) and the output latencies become the ProcessingSpec; the contract's
 * explicit budget, FIXED warmup count and wait ratio become max_inference_time, warm_up and
 * blocking_ratio; a STATEFUL model config becomes session_exclusive_processor;
 * max_instances becomes num_parallel_processors.
 *
 * @param model The model config.
 * @param contract A Hard contract with an explicit budget and a FIXED or NONE warmup.
 * @param candidates The engines to keep, or NULL for every entry (see the file comment).
 * @param num_candidates Entries in `candidates`.
 * @param out Receives the configuration on success; untouched on failure.
 * @param err Nullable; receives the status and the reason on failure.
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL model or contract;
 * ANIRA_ERROR_CONFIG for a rule the configuration breaks (the message names the tensor or
 * the entry); ANIRA_ERROR_NOT_SUPPORTED for what the 2.x runtime cannot do;
 * ANIRA_ERROR_EXTENSION_UNKNOWN / _UNCONSUMED from the consumed-or-fail walk.
 */
ANIRA_API anira_status to_inference_config(const anira_model_config* model,
                                           const anira_contract* contract,
                                           const anira_engine* candidates,
                                           uint32_t num_candidates,
                                           anira::InferenceConfig& out,
                                           anira_error* err) noexcept;

/**
 * @brief The 2.x ContextConfig of a machine config: the thread count (ANIRA_THREADS_AUTO is
 * the 2.x default), the wait strategy and the log level, drain, interval and queue capacity.
 *
 * The log sink, the log flags and the device descriptors have no 2.x counterpart and are not
 * carried; a machine extension the walk cannot consume fails the call by name.
 *
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL machine config;
 * ANIRA_ERROR_EXTENSION_UNKNOWN / _UNCONSUMED.
 */
ANIRA_API anira_status to_context_config(const anira_machine_config* machine,
                                         anira::ContextConfig& out,
                                         anira_error* err) noexcept;

/**
 * @brief The 2.x HostConfig of a Hard contract's geometry and a model config's anchor.
 *
 * block_max and rate become the host buffer size and sample rate, block_min < block_max
 * allows smaller buffers, and the anchor names the reference tensor (the first Streamed input,
 * else output, when none is set).
 *
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL argument; ANIRA_ERROR_CONFIG
 * when the geometry is missing (block_max 0 or rate 0) or a spec breaks a rule;
 * ANIRA_ERROR_NOT_SUPPORTED for an Async contract.
 */
ANIRA_API anira_status to_host_config(const anira_contract* contract,
                                      const anira_model_config* model,
                                      anira::HostConfig& out,
                                      anira_error* err) noexcept;

/**
 * @brief The 2.x HostConfig with the host's own geometry, which may be fractional (a plugin
 * that prepares a 2048-sample decoder with `samplesPerBlock / 2048.f`).
 *
 * @return ANIRA_OK; ANIRA_ERROR_INVALID_ARGUMENT for a NULL model config or a non-positive
 * buffer size or sample rate; ANIRA_ERROR_CONFIG when a spec breaks a rule.
 */
ANIRA_API anira_status to_host_config(const anira_model_config* model,
                                      float buffer_size,
                                      float sample_rate,
                                      bool allow_smaller_buffers,
                                      anira::HostConfig& out,
                                      anira_error* err) noexcept;

/// The engines this build carries an adapter for, in anira_engine order: the candidate list
/// that lets one model config serve every build.
ANIRA_API std::vector<anira_engine> enabled_engines();

/// The 2.x default number of parallel processors (half the hardware threads, at least 1):
/// what a model config's max_instances must be for the 2.x default.
ANIRA_API unsigned int v2_default_instances() noexcept;

}  // namespace anira::v3compat

// ---- The C++ face over the anira.hpp handles (C++20) ----------------------------------------

#if defined(__cplusplus) && __cplusplus >= 202002L

#include <anira/anira.hpp>
#include <span>
#include <utility>

namespace anira::v3compat {

/// to_inference_config over the handles; throws anira::Error with the reason. The model
/// config is taken by lvalue reference only: a bytes entry is borrowed, so the object must
/// outlive the result (see the file comment).
inline anira::InferenceConfig to_inference_config(const anira::ModelConfig& model,
                                                  const anira::ContractHandle& contract,
                                                  std::span<const anira::Engine> candidates = {}) {
    anira::InferenceConfig out;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status =
        to_inference_config(model.native(),
                            contract.native(),
                            candidates.empty() ? nullptr : candidates.data(),
                            static_cast<uint32_t>(candidates.size()),
                            out,
                            &err);
    if (ANIRA_FAILED(status)) { throw anira::Error(err); }
    return out;
}
inline anira::InferenceConfig to_inference_config(const anira::ModelConfig& model,
                                                  const anira::Hard& hard,
                                                  std::span<const anira::Engine> candidates = {}) {
    return to_inference_config(model, anira::ContractHandle(hard), candidates);
}
anira::InferenceConfig to_inference_config(anira::ModelConfig&& model,
                                           const anira::ContractHandle& contract,
                                           std::span<const anira::Engine> candidates = {}) = delete;
anira::InferenceConfig to_inference_config(anira::ModelConfig&& model,
                                           const anira::Hard& hard,
                                           std::span<const anira::Engine> candidates = {}) = delete;

/// to_context_config over the handle; throws anira::Error with the reason.
inline anira::ContextConfig to_context_config(const anira::MachineConfig& machine) {
    anira::ContextConfig out;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status = to_context_config(machine.native(), out, &err);
    if (ANIRA_FAILED(status)) { throw anira::Error(err); }
    return out;
}

/// to_host_config over the handles; throws anira::Error with the reason.
inline anira::HostConfig to_host_config(const anira::ContractHandle& contract,
                                        const anira::ModelConfig& model) {
    anira::HostConfig out;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status = to_host_config(contract.native(), model.native(), out, &err);
    if (ANIRA_FAILED(status)) { throw anira::Error(err); }
    return out;
}
inline anira::HostConfig to_host_config(const anira::Hard& hard, const anira::ModelConfig& model) {
    return to_host_config(anira::ContractHandle(hard), model);
}
inline anira::HostConfig to_host_config(const anira::ModelConfig& model,
                                        float buffer_size,
                                        float sample_rate,
                                        bool allow_smaller_buffers = false) {
    anira::HostConfig out;
    anira_error err = ANIRA_ERROR_INIT;
    const anira_status status =
        to_host_config(model.native(), buffer_size, sample_rate, allow_smaller_buffers, out, &err);
    if (ANIRA_FAILED(status)) { throw anira::Error(err); }
    return out;
}

}  // namespace anira::v3compat

#endif  // C++20

#endif  // ANIRA_COMPAT_V3_TO_V2_H
