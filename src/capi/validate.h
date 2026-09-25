/*
 * The validator of the 3.x configuration (section 2 of the v3 design): every rule a model
 * config, its contract and the pipeline's candidates must meet, in order (axes, roles,
 * windows, layouts, extensions, declared state, entries against candidates), and what it
 * derives from a configuration that passed: the resolved extents, channels, hops and anchor
 * of every tensor spec (DerivedSpec), the plan table (PlanKey, matching_plans: a model entry
 * on a provider per plan, and the default-set rule), and the per-slot facts it resolves from
 * the contract and the model (ring dtypes, host domains, declared state pairs). What the
 * pipeline's stage and engines mean to it arrives as StageFacts and EngineFacts. It is what
 * anira_handler_create and anira_handler_prepare run before anything is loaded; the 2.x
 * configuration objects the scheduler still takes are made from its results by the bridge
 * (v3_to_v2.h). Private to src/capi (the tests reach it through the src/ include directory).
 *
 * Every function here throws anira::StatusError (the status the C boundary returns, with the
 * message the caller reads); the exported entries catch at the boundary and say it once.
 * Nothing here logs.
 */
#ifndef ANIRA_CAPI_VALIDATE_H
#define ANIRA_CAPI_VALIDATE_H

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/system/Exports.h>
#include <anira/utils/InferenceBackend.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "handles.h"

namespace anira {
/// The ring dtype of every slot (include/anira/scheduler/SessionElement.h).
struct RingDtypes;
/// A latency per output tensor replacing the computed one (the same header).
struct CustomLatencies;
}  // namespace anira

namespace anira::capi {

/// What the validator derives for one tensor spec.
struct DerivedSpec {
    std::vector<int64_t> m_dims;        ///< the spec's extents, a dynamic Time extent resolved
    int64_t m_channels = 1;             ///< Streamed: the Channel axis extent, 1 without one;
                                        ///< Static and Buffer: 1 (the tag names an axis there)
    int64_t m_window_used = 0;          ///< Streamed only: the window pinned for this contract
    int64_t m_hop = 0;                  ///< Streamed: window_used - context; Static and Buffer: 0
    std::optional<size_t> m_time_axis;  ///< the Time axis, when the spec has one
};

/// One plan of the table the validator derives: a model entry on a provider. A row is a plan
/// once per candidate that names its engine (a built-in one, or the custom id) and a provider
/// the entry accepts: any for a neutral entry, its pin alone for a pinned one (models[].engine
/// with a suffix, anira_model_config_set_model_provider); with no candidate list a row is one
/// plan, on its pin or on ANIRA_PROVIDER_CPU. The provider is the plan's, in the engine's
/// vocabulary: a value of the enum, or ANIRA_PROVIDER_CUSTOM with a custom name.
struct PlanKey {
    size_t m_row = 0;  ///< the models[] index
    anira_provider m_provider = ANIRA_PROVIDER_CPU;
    std::string m_provider_id;  ///< a custom provider's name; empty for one the enum names

    bool operator==(const PlanKey& other) const = default;
};

/// What the validator derives for one model config under one (optional) Hard contract.
struct Derived {
    std::vector<DerivedSpec> m_inputs;
    std::vector<DerivedSpec> m_outputs;
    std::vector<size_t> m_rows;    ///< the models[] indices that survived the candidate filter
                                   ///< (a plan of at least one candidate), in entry order
    std::vector<PlanKey> m_plans;  ///< the plan table: the plans of every surviving row, in
                                   ///< entry order and, within a row, candidate order; the
                                   ///< dense plan index of the handler is the position here
    bool m_anchor_named = false;   ///< false: the first Streamed input, else the first
                                   ///< Streamed output (2.x k_first_streamable)
    bool m_anchor_is_input = true;
    size_t m_anchor_index = 0;
};

/// What a pipeline's stage means to the validator (stage_facts() of stage.h builds it from the
/// carrier; no stage is the default everywhere).
struct StageFacts {
    /// The stage fills pre_process: it pops the input rings itself, so the ring dtype of an
    /// input may differ from its spec's dtype (the stage takes the difference on itself).
    bool m_fills_pre = false;
    /// The stage fills post_process: the same for the ring dtype of an output.
    bool m_fills_post = false;
    /// The stage as a consumer when it declares consumed kinds, named after it (m_name points
    /// into the carrier, which outlives the facts); it joins the consumed-or-fail walk behind
    /// anira's own adapters. Empty otherwise.
    std::vector<ExtConsumer> m_consumers;
};

/// What a pipeline's custom engines mean to the validator (engine_facts() of engine.h
/// builds it from the pipeline's engines; NULL is the bridge's case, which has no pipeline:
/// anira.v2.custom is the one custom id then, served by the 2.x pass-through).
struct EngineFacts {
    /// The ids of the pipeline's engines: a custom row naming one is a plan on the 2.x CUSTOM
    /// backend, resolved by row like every plan; a custom row naming none of them,
    /// anira.v2.custom included, is refused at create.
    std::vector<std::string> m_ids;
    /// One consumer per registered engine that declares consumed kinds, keyed by its id
    /// (m_engine_id; m_name is the id too, pointing into the carrier, which outlives the
    /// facts): it joins the consumed-or-fail walk beside the stage's consumer, reading its
    /// "model:" kinds from its own entries and its other kinds from any host, while it is a
    /// candidate. Empty otherwise.
    std::vector<ExtConsumer> m_consumers;
    /// The providers list of each engine, beside m_ids (the descriptor's words, empty for an
    /// engine without a list): where a neutral entry of the engine runs under the default set
    /// (home_provider).
    std::vector<std::vector<std::string>> m_providers;
};

/// The consumers a pipeline declares, in one vector: the stage's, then one per registered
/// engine that declares kinds (either may be NULL). What the extension walk of validate and
/// the plan report's consumer column read.
ANIRA_API std::vector<ExtConsumer> pipeline_consumers(const StageFacts* stages,
                                                      const EngineFacts* engines);

/// Whether a candidate names an engine: a built-in engine by its value, a custom engine by its
/// id (ANIRA_ENGINE_CUSTOM with an engine_id keeps the custom rows of that name); NULL keeps
/// everything (the bridge's rule). The engine rule alone, what the extension walk keys a
/// consumer by; the provider is not read.
ANIRA_API bool engine_is_candidate(anira_engine engine,
                                   const std::string& engine_id,
                                   const anira_backend_id* candidates,
                                   uint32_t num_candidates) noexcept;

/// The plans of one model entry under the candidates (PlanKey): one per candidate naming its
/// engine whose provider the entry accepts, in candidate order, an equal provider once; with
/// a NULL list one plan, on the entry's pin or on ANIRA_PROVIDER_CPU. Under the default
/// set (default_set: the list anira_pipeline_add_inference built for a NULL one, every engine
/// on the CPU path and every pin) an entry whose engine is named is one plan too, on
/// its pin or on its engine's home provider (home_provider: the CPU path for a built-in
/// engine and for a custom engine that serves it, else the custom engine's first listed
/// provider), whatever else the list names. Empty for an entry no candidate runs.
ANIRA_API std::vector<PlanKey> matching_plans(size_t row_index,
                                              const ModelEntry& row,
                                              const anira_backend_id* candidates,
                                              uint32_t num_candidates,
                                              bool default_set = false,
                                              const EngineFacts* engines = nullptr);

/// Where a neutral entry runs under the default set: ANIRA_PROVIDER_CPU for a built-in engine
/// and for a custom engine that serves it (no providers list, or "cpu" in it), else the
/// custom engine's first listed provider. `engines` NULL, or an id it lacks: the CPU path.
ANIRA_API PlanKey home_provider(size_t row_index,
                                const ModelEntry& row,
                                const EngineFacts* engines);

/// Whether a model entry is a plan of the candidates (matching_plans is not empty): what the
/// extension walk keys an entry by.
ANIRA_API bool row_is_candidate(const ModelEntry& row,
                                const anira_backend_id* candidates,
                                uint32_t num_candidates);

/// The 2.x backend a model row maps to: CUSTOM for every custom row (whether its id is served
/// is check_rows' question: an engine registered on the pipeline, or anira.v2.custom on the
/// bridge), the
/// engine's own backend for a built-in engine of this build, nullopt for an engine this build
/// does not carry.
ANIRA_API std::optional<anira::InferenceBackend> backend_of(const ModelEntry& row) noexcept;

/// The engine's extents of one slot under one row: the spec's derived extents (its dynamic
/// Time extent resolved) reordered by the row's layout for the slot (layout.h engine_dims), or
/// those extents themselves for a row without a layout for it. What the row's engine is handed
/// for the slot.
ANIRA_API std::vector<int64_t> engine_dims_of(const anira_tensor_spec& spec,
                                              const DerivedSpec& derived,
                                              const std::vector<uint32_t>& layout);

/// The lower-case engine name of the JSON vocabulary, or the custom id.
/// The row's engine in a message (words.h engine_label over its two fields).
ANIRA_API std::string engine_label(const ModelEntry& row);

/// The engines this build carries an adapter for, in anira_engine order.
ANIRA_API std::vector<anira_engine> enabled_engines();

/// Runs every section-2 rule the 2.x runtime can honour, in order, and derives the
/// per-tensor quantities; contract may be NULL (no contract rule runs, flexible windows
/// pin to window_min). The candidates narrow the model entries and make the plan table
/// (Derived::m_plans): NULL keeps every row as one plan (the bridge's rule; the handler always
/// names its set), a built-in engine keeps its rows, ANIRA_ENGINE_CUSTOM with an engine_id
/// keeps the custom rows of that name, and a row is a
/// plan once per candidate whose provider it accepts (matching_plans; under the default set,
/// default_set, one plan per row on its pin or on the CPU path). Throws StatusError with
/// ANIRA_ERROR_CONFIG for a rule the configuration breaks (no surviving row among them)
/// and ANIRA_ERROR_NOT_SUPPORTED for what the 2.x runtime cannot do. `stages` is what the
/// pipeline's stage adds (NULL: no stage, the bridge's case): the ring dtype rule
/// relaxes for a side whose phase the stage fills, and the stage's consumed kinds join the
/// extension walk. `engines` are the pipeline's registered engines (NULL: none, the bridge's
/// case): a custom row naming one of their ids is a plan, and their consumed kinds join the
/// walk; a custom row naming no registered id is ANIRA_ERROR_NOT_SUPPORTED, anira.v2.custom
/// included (without engines, the bridge, that id alone is served). A registered row's path is
/// never opened here: the engine's prepare decides.
ANIRA_API void validate(const anira_model_config& model,
                        const anira_contract* contract,
                        const anira_backend_id* candidates,
                        uint32_t num_candidates,
                        Derived& out,
                        const StageFacts* stages = nullptr,
                        const EngineFacts* engines = nullptr,
                        bool default_set = false);

/// The ring dtype of every slot: two vectors sized to the model's input and output lists,
/// ANIRA_DTYPE_F32 everywhere, then each entry of the Hard contract's ring dtypes resolved
/// by tensor name into its slot. Run validate first: it refuses a name that matches no
/// tensor, a non-Streamed tensor, and a dtype other than the spec's (nothing converts) unless
/// a stage fills the phase that moves that ring.
ANIRA_API anira::RingDtypes ring_dtypes_of(const anira_contract& contract,
                                           const anira_model_config& model);

/// The declared stream latency of every output slot (anira_contract_hard_set_latency), what the
/// C prepare hands the scheduler: one entry per tensor of the model's output list, -1 (the
/// computed figure) everywhere, then each entry of the Hard contract's latencies resolved by
/// tensor name into its slot; exact on every platform, since a figure is at most INT32_MAX.
/// Run validate first: it refuses a name that matches no tensor, an input, a non-Streamed
/// output and a figure below the spec's latency.
ANIRA_API anira::CustomLatencies latencies_of(const anira_contract& contract,
                                              const anira_model_config& model);

/// The declared host-end domain of every slot (anira_contract_set_host_domain): two vectors
/// sized to the model's input and output lists, ANIRA_DOMAIN_HOST everywhere, then each entry
/// of the contract's host domains resolved by tensor name into its slot. What the plan report's
/// slot rows carry as domain_in of an input and domain_out of an output, against the engine's
/// domain on the other side. Run validate first: it refuses a name that matches no tensor and,
/// in this pre-release, any domain but ANIRA_DOMAIN_HOST.
struct HostDomains {
    std::vector<anira_domain> m_inputs;
    std::vector<anira_domain> m_outputs;
};
ANIRA_API HostDomains host_domains_of(const anira_contract& contract,
                                      const anira_model_config& model);

/// One declared state pair (ANIRA_ROLE_STATE) as slots: the State input and the State output
/// it is fed from, each the tensor's position in the model config's list of its side.
struct StateLink {
    size_t m_input = 0;
    size_t m_output = 0;
};

/// The declared state pairs of a model, in the order of the State inputs: what the handler
/// pairs its two port vectors by at anira_handler_create (the two halves name each other).
/// Empty for a model without State specs. Run validate first: it refuses a State input without
/// a source, a source that names no State output, a State output named by no input or by two,
/// and two halves of unequal dtype or shape.
ANIRA_API std::vector<StateLink> state_links(const anira_model_config& model);

/// The consumed-or-fail walk over a context config's extension bag alone (the model,
/// contract and spec bags are walked at anira_handler_create and prepare). Throws
/// StatusError with ANIRA_ERROR_EXTENSION_UNKNOWN / _UNCONSUMED naming the kind.
ANIRA_API void check_context_extensions(const anira_context_config& config);

/// The stream a model's host block counts in (anira_model_config_set_anchor): the named
/// Streamed tensor or, unnamed, the first Streamed input, else the first Streamed output.
struct Anchor {
    bool m_named = false;  ///< false: the default choice (2.x k_first_streamable)
    bool m_is_input = true;
    size_t m_index = 0;  ///< the tensor's position in the list of its side
};

/// The anchor of a model without a contract: the tensor spec rules that need none (the
/// extents, the roles, the windows pinned to window_min), then the anchor rule. Throws
/// StatusError with ANIRA_ERROR_CONFIG for a model without an input or an output, a spec that
/// breaks a rule, no Streamed tensor, or an anchor that names no Streamed tensor. What the
/// bridge's HostConfig of the host's own geometry reads (v3_to_v2.h make_host_config).
ANIRA_API Anchor anchor_of(const anira_model_config& model);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_VALIDATE_H
