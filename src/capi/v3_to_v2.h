/*
 * The bridge from the 3.x configuration to the 2.x runtime: the configuration handles as the
 * 2.x InferenceConfig / CoreConfig / HostConfig the scheduler still takes (InferenceManager,
 * SessionElement, Core and the inference threads run on them), made from what the validator
 * derived (validate.h). The C handler prepares its sessions through make_inference_config and
 * make_host_config, and the exported face (anira/compat/v3_to_v2.h, anira::v3compat) hands the
 * same objects to a 2.x host. Everything here goes with the 2.x configuration objects at the
 * runtime cut-over. Private to src/capi (the scheduler and the tests reach it through the src/
 * include directory).
 *
 * Every function here throws anira::StatusError (the status the C boundary returns, with the
 * message the caller reads) or std::invalid_argument (the 2.x constructors' own cross-checks,
 * which the firewall classifies as ANIRA_ERROR_CONFIG); the exported entries catch at the
 * boundary and say it once. Nothing here logs.
 */
#ifndef ANIRA_CAPI_V3_TO_V2_H
#define ANIRA_CAPI_V3_TO_V2_H

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/system/Exports.h>
#include <anira/utils/HostConfig.h>

#include <cstdint>

#include "handles.h"
#include "validate.h"

namespace anira::capi {

/// The 2.x InferenceConfig of a model config under a Hard contract (validate, then map). A
/// registered engine's row becomes a ModelData row on the 2.x CUSTOM backend, like the
/// anira.v2.custom row: the plan table resolves plans by row, never by the backend, so several
/// such rows are legal.
ANIRA_API anira::InferenceConfig make_inference_config(const anira_model_config& model,
                                                       const anira_contract& contract,
                                                       const anira_backend_id* candidates,
                                                       uint32_t num_candidates,
                                                       const StageFacts* stages = nullptr,
                                                       const EngineFacts* engines = nullptr,
                                                       bool default_set = false);

/// The 2.x HostConfig of a Hard contract's geometry and the model config's anchor.
ANIRA_API anira::HostConfig make_host_config(const anira_contract& contract,
                                             const anira_model_config& model);

/// The same with the host's own (possibly fractional) geometry.
ANIRA_API anira::HostConfig make_host_config(const anira_model_config& model,
                                             float buffer_size,
                                             float sample_rate,
                                             bool allow_smaller);

/// The 2.x CoreConfig of a context config: threads, wait strategy and the log scalars, after
/// check_context_extensions. Kept for the bridge (anira::v3compat::to_core_config); the core
/// itself reads the context config.
ANIRA_API anira::CoreConfig make_core_config(const anira_context_config& config);

/// The context config of a 2.x CoreConfig, field by field: threads, wait strategy, log
/// level, drain, interval and queue capacity; no sink, no flags, no device block, no
/// extensions. The log level is copied explicitly (CoreConfig defaults to Info/Error,
/// anira_context_config to WARNING). The 2.x InferenceManager constructor is the only
/// caller; it leaves with the 2.x classes at the cut-over.
ANIRA_API anira_context_config make_context_config(const anira::CoreConfig& core_config);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_V3_TO_V2_H
