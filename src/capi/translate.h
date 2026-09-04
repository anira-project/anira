/*
 * The section-2 validator and the translation of the 3.x configuration handles into the 2.x
 * runtime's InferenceConfig / CoreConfig / HostConfig. Private to src/capi (the tests reach
 * it through the src/ include directory); the exported face is anira/compat/v3_to_v2.h.
 *
 * Every function here throws anira::StatusError (the status the C boundary returns, with the
 * message the caller reads) or std::invalid_argument (the 2.x constructors' own cross-checks,
 * which the firewall classifies as ANIRA_ERROR_CONFIG); the exported entries catch at the
 * boundary and say it once. Nothing here logs.
 */
#ifndef ANIRA_CAPI_TRANSLATE_H
#define ANIRA_CAPI_TRANSLATE_H

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/system/Exports.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "handles.h"

namespace anira::capi {

/// What the validator derives for one tensor spec.
struct DerivedSpec {
    std::vector<int64_t> m_dims;        ///< the spec's extents, a dynamic Time extent resolved
    int64_t m_channels = 1;             ///< the Channel axis extent, 1 without one
    int64_t m_window_used = 0;          ///< Streamed only: the window pinned for this contract
    int64_t m_hop = 0;                  ///< Streamed: window_used - context; Static and Buffer: 0
    std::optional<size_t> m_time_axis;  ///< the Time axis, when the spec has one
};

/// What the validator derives for one model config under one (optional) Hard contract.
struct Derived {
    std::vector<DerivedSpec> m_inputs;
    std::vector<DerivedSpec> m_outputs;
    std::vector<size_t> m_rows;   ///< the models[] indices that survived the candidate filter
    bool m_anchor_named = false;  ///< false: the first Streamed input, else the first
                                  ///< Streamed output (2.x k_first_streamable)
    bool m_anchor_is_input = true;
    size_t m_anchor_index = 0;
};

/// The 2.x backend a model row maps to, or nullopt when this build has no adapter for it
/// (an engine that is not compiled in, a custom engine other than anira.v2.custom).
ANIRA_API std::optional<anira::InferenceBackend> backend_of(const ModelEntry& row) noexcept;

/// The lower-case engine name of the JSON vocabulary, or the custom id.
ANIRA_API std::string engine_label(const ModelEntry& row);

/// The engines this build carries an adapter for, in anira_engine order.
ANIRA_API std::vector<anira_engine> enabled_engines();

/// Runs every section-2 rule the 2.x runtime can honour, in order, and derives the
/// per-tensor quantities; contract may be NULL (no contract rule runs, flexible windows
/// pin to window_min). Throws StatusError with ANIRA_ERROR_CONFIG for a rule the
/// configuration breaks and ANIRA_ERROR_NOT_SUPPORTED for what the 2.x runtime cannot do.
ANIRA_API void validate(const anira_model_config& model,
                        const anira_contract* contract,
                        const anira_engine* candidates,
                        uint32_t num_candidates,
                        Derived& out);

/// The 2.x InferenceConfig of a model config under a Hard contract (validate, then map).
ANIRA_API anira::InferenceConfig make_inference_config(const anira_model_config& model,
                                                       const anira_contract& contract,
                                                       const anira_engine* candidates,
                                                       uint32_t num_candidates);

/// The 2.x CoreConfig of a context config: threads, wait strategy and the log scalars.
ANIRA_API anira::CoreConfig make_core_config(const anira_context_config& context);

/// The 2.x HostConfig of a Hard contract's geometry and the model config's anchor.
ANIRA_API anira::HostConfig make_host_config(const anira_contract& contract,
                                             const anira_model_config& model);

/// The same with the host's own (possibly fractional) geometry.
ANIRA_API anira::HostConfig make_host_config(const anira_model_config& model,
                                             float buffer_size,
                                             float sample_rate,
                                             bool allow_smaller);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_TRANSLATE_H
