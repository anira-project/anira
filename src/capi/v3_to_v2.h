/*
 * The translators from the 3.x configuration into the 2.x configuration objects the scheduler
 * still takes (InferenceManager, SessionElement, Core and the inference threads run on the 2.x
 * InferenceConfig and HostConfig), made from what the validator derived (validate.h). The C
 * handler prepares its sessions through make_inference_config and make_host_config.
 * Everything here goes when the core takes the 3.x types (PR 10b). Private to src/capi (the
 * scheduler and the white-box tests reach it through the src/ include directory).
 *
 * Every function here throws anira::StatusError (the status the C boundary returns, with the
 * message the caller reads) or std::invalid_argument (the 2.x constructors' own cross-checks,
 * which the firewall classifies as ANIRA_ERROR_CONFIG); the exported entries catch at the
 * boundary and say it once. Nothing here logs.
 */
#ifndef ANIRA_CAPI_V3_TO_V2_H
#define ANIRA_CAPI_V3_TO_V2_H

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
                                                       const ProcessorFacts* processors = nullptr,
                                                       const EngineFacts* engines = nullptr,
                                                       bool default_set = false);

/// The same over a model validated already: `derived` is what validate filled for it under
/// `contract` (the C handler's prepare validates once and maps here).
ANIRA_API anira::InferenceConfig make_inference_config(const anira_model_config& model,
                                                       const anira_contract& contract,
                                                       const Derived& derived);

/// The 2.x HostConfig of a Hard contract's geometry and the model config's anchor.
ANIRA_API anira::HostConfig make_host_config(const anira_contract& contract,
                                             const anira_model_config& model);

/// The same with the host's own (possibly fractional) geometry (the white-box tests).
ANIRA_API anira::HostConfig make_host_config(const anira_model_config& model,
                                             float buffer_size,
                                             float sample_rate,
                                             bool allow_smaller);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_V3_TO_V2_H
