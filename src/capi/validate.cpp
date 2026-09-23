#include "validate.h"

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../utils/StatusError.h"
#include "ext_registry.h"
#include "handles.h"
#include "layout.h"
#include "words.h"

namespace anira::capi {
namespace {

constexpr const char* k_v2_custom_engine = "anira.v2.custom";

[[noreturn]] void refuse(anira_status status, const std::string& message) {
    throw StatusError(status, message);
}
[[noreturn]] void config_error(const std::string& message) {
    refuse(ANIRA_ERROR_CONFIG, message);
}
[[noreturn]] void not_supported(const std::string& message) {
    refuse(ANIRA_ERROR_NOT_SUPPORTED, message);
}

std::string at(const anira_tensor_spec& spec) {
    return "tensor '" + spec.m_name + "': ";
}
std::string at_row(size_t index) {
    return "models[" + std::to_string(index) + "]: ";
}

const char* role_word(anira_role role) {
    switch (role) {
        case ANIRA_ROLE_STREAMED: return "Streamed";
        case ANIRA_ROLE_BUFFER: return "Buffer";
        case ANIRA_ROLE_STATIC: return "Static";
        case ANIRA_ROLE_STATE: return "State";
        default: return "unknown";
    }
}

std::string hex_dtype(anira_dtype dtype) {
    static constexpr const char* k_digits = "0123456789abcdef";
    std::string text = "0x";
    for (int shift = 28; shift >= 0; shift -= 4) {
        text += k_digits[(dtype >> static_cast<unsigned>(shift)) & 0xfU];
    }
    return text;
}

std::string engines_list(const std::vector<anira_engine>& engines) {
    std::string text;
    for (const anira_engine engine : engines) {
        if (!text.empty()) { text += ", "; }
        text += engine_word(engine);
    }
    return text.empty() ? "none" : text;
}

// The provider a candidate names, as its own words: the id where the record carries one, the
// enum's value else.
std::string_view candidate_provider_id(const anira_backend_id& id) noexcept {
    return id.provider_id != nullptr ? std::string_view(id.provider_id) : std::string_view();
}

// The candidate list as a message names it: the built-in engines by word, a custom engine
// by its id, a provider as the "engine" word's suffix.
std::string candidates_list(const anira_backend_id* candidates, uint32_t num_candidates) {
    std::string text;
    if (candidates == nullptr) { return "every engine"; }
    for (uint32_t i = 0; i < num_candidates; ++i) {
        if (!text.empty()) { text += ", "; }
        const anira_backend_id& id = candidates[i];
        text += id.engine_id != nullptr ? id.engine_id
                                        : engine_word(static_cast<anira_engine>(id.engine));
        if (id.provider != ANIRA_PROVIDER_DEFAULT || !candidate_provider_id(id).empty()) {
            text += ":" + provider_label(static_cast<anira_provider>(id.provider),
                                         candidate_provider_id(id));
        }
    }
    return text.empty() ? "none" : text;
}

// The provider a pinned entry names, as a message names it.
std::string pin_label(const ModelEntry& row) {
    return provider_label(row.m_provider, row.m_provider_id);
}

const anira_tensor_spec* find_spec(const anira_model_config& model,
                                   const std::string& name,
                                   bool* is_input,
                                   size_t* index) {
    for (size_t i = 0; i < model.m_inputs.size(); ++i) {
        if (model.m_inputs[i].m_name == name) {
            if (is_input != nullptr) { *is_input = true; }
            if (index != nullptr) { *index = i; }
            return &model.m_inputs[i];
        }
    }
    for (size_t i = 0; i < model.m_outputs.size(); ++i) {
        if (model.m_outputs[i].m_name == name) {
            if (is_input != nullptr) { *is_input = false; }
            if (index != nullptr) { *index = i; }
            return &model.m_outputs[i];
        }
    }
    return nullptr;
}

// ---- the rules, one tensor at a time ------------------------------------------------------

void check_spec(const anira_tensor_spec& spec,
                bool is_input,
                const HardContract* hard,
                DerivedSpec& out) {
    const std::string where = at(spec);
    if (spec.m_ndim == 0) { config_error(where + "no axis was set"); }
    const bool streamed = spec.m_role == ANIRA_ROLE_STREAMED;
    const char* role = role_word(spec.m_role);

    std::optional<size_t> time_axis;
    std::optional<size_t> channel_axis;
    size_t time_count = 0;
    size_t channel_count = 0;
    out.m_dims.assign(spec.m_ndim, 0);
    for (size_t i = 0; i < spec.m_ndim; ++i) {
        const Axis& axis = spec.m_axes[i];
        if (!axis.m_written) {
            config_error(where + "axis " + std::to_string(i) + " was never set (ndim is " +
                         std::to_string(spec.m_ndim) + ")");
        }
        if (axis.m_tag == ANIRA_AXIS_TIME) {
            ++time_count;
            time_axis = i;
        } else if (axis.m_tag == ANIRA_AXIS_CHANNEL) {
            ++channel_count;
            channel_axis = i;
        }
        out.m_dims[i] = axis.m_extent;
    }
    if (time_count > 1) {
        config_error(where + "has " + std::to_string(time_count) + " Time axes; at most one");
    }
    if (channel_count > 1) {
        config_error(where + "has " + std::to_string(channel_count) + " Channel axes; at most one");
    }
    if (streamed && !time_axis.has_value()) {
        config_error(where + "a Streamed tensor needs a Time axis");
    }
    if (spec.m_role == ANIRA_ROLE_STATIC && time_axis.has_value()) {
        config_error(where +
                     "a Static tensor has no Time axis (a whole-buffer tensor with "
                     "time semantics is the Buffer role)");
    }
    if (spec.m_role == ANIRA_ROLE_STATE && time_axis.has_value()) {
        config_error(where +
                     "a State tensor has no Time axis: it is fed back whole, once per "
                     "inference");
    }
    for (size_t i = 0; i < spec.m_ndim; ++i) {
        const int64_t extent = out.m_dims[i];
        if (extent == ANIRA_DYNAMIC) {
            if (time_axis.has_value() && i == *time_axis) {
                if (streamed) { continue; }  // resolved from the pinned window below
                not_supported(where +
                              "a dynamic Time extent on a Buffer tensor: the 2.x runtime "
                              "binds a fixed shape (give the extent)");
            }
            config_error(where + "axis " + std::to_string(i) +
                         " is dynamic; only the Time axis of a Streamed tensor may be");
        }
        if (extent <= 0) {
            config_error(where + "axis " + std::to_string(i) + " has extent " +
                         std::to_string(extent) + "; extents are positive");
        }
    }

    if (streamed) {
        if (spec.m_window_min <= 0) {
            config_error(where + "window_min must be positive (got " +
                         std::to_string(spec.m_window_min) + ")");
        }
        if (spec.m_window_max != ANIRA_UNBOUNDED && spec.m_window_max < spec.m_window_min) {
            config_error(where + "window_max " + std::to_string(spec.m_window_max) +
                         " is below window_min " + std::to_string(spec.m_window_min));
        }
        if (spec.m_overlap >= spec.m_window_min) {
            config_error(where + "overlap " + std::to_string(spec.m_overlap) +
                         " must be below window_min " + std::to_string(spec.m_window_min));
        }
    } else {
        if (spec.m_window_min != 0 || spec.m_window_max != 0 || spec.m_overlap != 0) {
            config_error(where + std::string("a ") + role + " tensor has no window");
        }
        if (spec.m_ratio_num != 0 || spec.m_ratio_den != 0) {
            config_error(where + std::string("a ") + role + " tensor has no time ratio");
        }
    }

    // The Channel tag maps onto ring channels on a Streamed tensor only. A non-Streamed tensor
    // has the spec's shape, a Channel axis of any extent being one of its axes, and moves as the
    // product of its extents: the 2.x runtime carries it as one channel.
    out.m_channels = streamed && channel_axis.has_value() ? out.m_dims[*channel_axis] : 1;
    if (spec.m_role == ANIRA_ROLE_STATE && spec.m_latency != 0) {
        config_error(where + "a State tensor has no latency: it never reaches a stream");
    }
    if (is_input && spec.m_latency != 0) { config_error(where + "latency is an output property"); }
    if (spec.m_latency < 0) {
        config_error(where + "latency must not be negative (got " + std::to_string(spec.m_latency) +
                     ")");
    }
    // The queue stores float32 in this pre-release: every tensor that travels through it (a
    // Streamed one through its ring and the chunk's buffer, a Static one materialised into the
    // chunk's buffer) is float32 until the typed storage arrives. A State tensor travels
    // through neither: its pair lives in the handler's two buffers in the spec's dtype (port.h),
    // which the engine is bound to as they are, so the rule does not stand on it (a built-in
    // engine refuses a model with a non-float32 tensor at load; a registered engine binds
    // what its load accepts).
    if (spec.m_dtype != ANIRA_DTYPE_F32 && spec.m_role != ANIRA_ROLE_STATE) {
        not_supported(where + "dtype " + hex_dtype(spec.m_dtype) +
                      ": the queue stores float32 in this pre-release");
    }
    out.m_time_axis = time_axis;

    // Window pinning: a fixed window is used as is; a flexible one covers one host block per
    // inference (block_max scaled by the tensor's time ratio, plus the context), clamped to
    // [window_min, window_max]; without a geometry the smallest window.
    if (streamed) {
        int64_t used = spec.m_window_min;
        if (spec.m_window_max != spec.m_window_min && hard != nullptr && hard->m_block_max > 0) {
            const int64_t num = spec.m_ratio_den == 0 ? 1 : spec.m_ratio_num;
            const int64_t den = spec.m_ratio_den == 0 ? 1 : spec.m_ratio_den;
            const auto block = static_cast<int64_t>(hard->m_block_max);
            if ((block * num) % den != 0) {
                config_error(where + "time ratio " + std::to_string(num) + "/" +
                             std::to_string(den) + " gives a fractional hop for block_max " +
                             std::to_string(block));
            }
            used = block * num / den + spec.m_overlap;
            used = std::max(used, spec.m_window_min);
            if (spec.m_window_max != ANIRA_UNBOUNDED) { used = std::min(used, spec.m_window_max); }
        }
        out.m_window_used = used;
        out.m_hop = used - spec.m_overlap;
        if (out.m_dims[*time_axis] == ANIRA_DYNAMIC) { out.m_dims[*time_axis] = used; }
    }
}

void check_contract(const anira_contract& contract,
                    const anira_model_config& model,
                    const StageFacts* stages) {
    const HardContract* hard = contract.hard();
    if (hard == nullptr) {
        not_supported(
            "contract: an Async contract has no 2.x counterpart; it arrives with "
            "the 3.x runtime");
    }
    if (hard->m_block_min > hard->m_block_max) {
        config_error("contract: block_min " + std::to_string(hard->m_block_min) +
                     " exceeds block_max " + std::to_string(hard->m_block_max));
    }
    if (hard->m_budget == ANIRA_BUDGET_EXPLICIT && !(hard->m_budget_ms > 0.0)) {
        config_error("contract: an explicit budget must be positive (got " +
                     std::to_string(hard->m_budget_ms) + " ms)");
    }
    if (hard->m_budget != ANIRA_BUDGET_EXPLICIT) {
        not_supported(
            "contract: a MEASURED budget needs the 3.x runtime's warmup "
            "measurement; set an explicit budget (ANIRA_BUDGET_EXPLICIT, the 2.x "
            "max_inference_time)");
    }
    if (hard->m_warmup == ANIRA_WARMUP_UNTIL_STABLE) {
        not_supported(
            "contract: UNTIL_STABLE warmup needs the 3.x runtime; set "
            "ANIRA_WARMUP_FIXED (the 2.x warm_up) or ANIRA_WARMUP_NONE");
    }
    if (hard->m_wait_ratio < 0.0) {
        config_error("contract: wait_ratio must not be negative (got " +
                     std::to_string(hard->m_wait_ratio) + ")");
    }
    // The ring dtype rules: a ring exists on a Streamed tensor only, and it holds the
    // spec's dtype as is (nothing in anira converts), unless a stage of the pipeline fills the
    // phase that moves that ring (pre_process pops the input rings, post_process pushes the
    // output rings) and so takes the difference on itself. The default bodies still refuse
    // such a slot at run time.
    for (const auto& [name, dtype] : hard->m_ring_dtypes) {
        bool is_input = true;
        const anira_tensor_spec* spec = find_spec(model, name, &is_input, nullptr);
        if (spec == nullptr) {
            config_error("contract: the ring dtype of '" + name + "' names no tensor");
        }
        if (spec->m_role != ANIRA_ROLE_STREAMED) {
            config_error("contract: the ring dtype of '" + name + "' is set on a " +
                         role_word(spec->m_role) + " tensor; only a Streamed tensor has a ring");
        }
        const bool stage_moves_ring =
            stages != nullptr && (is_input ? stages->m_fills_pre : stages->m_fills_post);
        if (dtype != spec->m_dtype && !stage_moves_ring) {
            config_error("contract: the ring dtype of '" + name + "' is " + hex_dtype(dtype) +
                         " but its spec's dtype is " + hex_dtype(spec->m_dtype) +
                         "; nothing converts (add a stage that fills " +
                         phase_word(is_input ? ANIRA_PHASE_PRE_PROCESS : ANIRA_PHASE_POST_PROCESS) +
                         " and converts, anira_pipeline_add_stage)");
        }
    }
    // The host-end domain rules: every tensor of either side, whatever its role, has one
    // declared host-end domain (anira_contract_set_host_domain, absent = host memory), the
    // domain anira allocates the ring, the model tensor, the Static store and the state buffers
    // of the slot in and every stage phase works in. The name must be a tensor's. The 2.x
    // runtime allocates in host memory only, so in this pre-release any other domain is refused
    // here, where the declaration meets the runtime; the declaration itself is data.
    for (const auto& [name, domain] : contract.m_host_domains) {
        if (find_spec(model, name, nullptr, nullptr) == nullptr) {
            config_error("contract: the host domain of '" + name + "' names no tensor");
        }
        if (domain != ANIRA_DOMAIN_HOST) {
            not_supported("contract: the host domain of '" + name + "' is domain " +
                          std::to_string(static_cast<unsigned int>(domain)) +
                          ", but every host end is host memory (ANIRA_DOMAIN_HOST) in this "
                          "pre-release");
        }
    }
}

// ---- declared state ------------------------------------------------------------------------

// The State output a State input names, with its index in the output list. CONFIG naming the
// input: no source, a source that names no output, or an output of another role.
const anira_tensor_spec& state_output_of(const anira_model_config& model,
                                         const anira_tensor_spec& input,
                                         size_t& index) {
    if (input.m_state_source.empty()) {
        config_error(at(input) +
                     "a State input needs a state_source: the canonical name of the State "
                     "output it is fed from (anira_tensor_spec_set_state_source, the JSON key "
                     "\"state_source\")");
    }
    bool is_input = true;
    const anira_tensor_spec* output = find_spec(model, input.m_state_source, &is_input, &index);
    if (output == nullptr || is_input) {
        config_error(at(input) + "state_source '" + input.m_state_source +
                     "' names no output tensor");
    }
    if (output->m_role != ANIRA_ROLE_STATE) {
        config_error(at(input) + "state_source '" + input.m_state_source + "' is a " +
                     role_word(output->m_role) +
                     " output; a State input is fed from a State output");
    }
    return *output;
}

// The pairing of declared state (ANIRA_ROLE_STATE): every State input names exactly one State
// output (state_source, stated once, on the input), every State output is named by exactly one
// State input, and the two halves have the same dtype, rank and extents, because the handler
// keeps one value per pair, on the input half's port, and copies it both ways. It runs on the
// raw specs, ahead of check_spec, so that an unequal dtype is this rule's CONFIG and not the
// float32 rule's NOT_SUPPORTED; what a State spec may not carry (a Time axis, a window, a time
// ratio, a latency) is check_spec's, a ring dtype or the anchor naming one is the rule of those
// two.
void check_state(const anira_model_config& model) {
    for (const anira_tensor_spec& spec : model.m_outputs) {
        if (!spec.m_state_source.empty()) {
            config_error(at(spec) +
                         "state_source is set on an output; the pairing is stated on the State "
                         "input, which names the State output it is fed from");
        }
    }
    std::vector<uint32_t> named(model.m_outputs.size(), 0);
    for (const anira_tensor_spec& input : model.m_inputs) {
        if (input.m_role != ANIRA_ROLE_STATE) {
            if (!input.m_state_source.empty()) {
                config_error(at(input) + "state_source is set on a " + role_word(input.m_role) +
                             " tensor; only a State input has one");
            }
            continue;
        }
        size_t index = 0;
        const anira_tensor_spec& output = state_output_of(model, input, index);
        if (++named[index] > 1) {
            config_error(at(output) +
                         "is the state_source of more than one State input (the second is '" +
                         input.m_name + "'); a State output feeds exactly one State input");
        }
        if (output.m_dtype != input.m_dtype) {
            config_error(at(input) + "dtype " + hex_dtype(input.m_dtype) +
                         " differs from the dtype " + hex_dtype(output.m_dtype) +
                         " of its state_source '" + output.m_name +
                         "'; the two halves of a state pair have one dtype");
        }
        bool same_shape = output.m_ndim == input.m_ndim;
        for (uint32_t axis = 0; same_shape && axis < input.m_ndim; ++axis) {
            same_shape = output.m_axes[axis].m_extent == input.m_axes[axis].m_extent;
        }
        if (!same_shape) {
            config_error(at(input) + "its shape differs from the shape of its state_source '" +
                         output.m_name + "'; the two halves of a state pair have one shape");
        }
    }
    for (size_t i = 0; i < model.m_outputs.size(); ++i) {
        if (model.m_outputs[i].m_role == ANIRA_ROLE_STATE && named[i] == 0) {
            config_error(at(model.m_outputs[i]) +
                         "a State output that no State input names as its state_source");
        }
    }
}

void resolve_anchor(const anira_model_config& model, Derived& out) {
    bool any_streamed = false;
    for (const anira_tensor_spec& spec : model.m_inputs) {
        any_streamed = any_streamed || spec.m_role == ANIRA_ROLE_STREAMED;
    }
    for (const anira_tensor_spec& spec : model.m_outputs) {
        any_streamed = any_streamed || spec.m_role == ANIRA_ROLE_STREAMED;
    }
    if (!any_streamed) {
        config_error("no Streamed tensor: the reference stream needs one on either side");
    }
    if (model.m_anchor.empty()) {
        out.m_anchor_named = false;
        for (size_t i = 0; i < model.m_inputs.size(); ++i) {
            if (model.m_inputs[i].m_role == ANIRA_ROLE_STREAMED) {
                out.m_anchor_is_input = true;
                out.m_anchor_index = i;
                return;
            }
        }
        for (size_t i = 0; i < model.m_outputs.size(); ++i) {
            if (model.m_outputs[i].m_role == ANIRA_ROLE_STREAMED) {
                out.m_anchor_is_input = false;
                out.m_anchor_index = i;
                return;
            }
        }
        return;  // unreachable: any_streamed
    }
    bool is_input = true;
    size_t index = 0;
    const anira_tensor_spec* spec = find_spec(model, model.m_anchor, &is_input, &index);
    if (spec == nullptr) { config_error("anchor '" + model.m_anchor + "' names no tensor"); }
    if (spec->m_role != ANIRA_ROLE_STREAMED) {
        config_error("anchor '" + model.m_anchor + "' is " + role_word(spec->m_role) +
                     "; the anchor is a Streamed tensor");
    }
    if (spec->m_ratio_den != 0 && spec->m_ratio_num != spec->m_ratio_den) {
        config_error(at(*spec) + "the anchor's time ratio is 1:1 by definition (got " +
                     std::to_string(spec->m_ratio_num) + "/" + std::to_string(spec->m_ratio_den) +
                     ")");
    }
    out.m_anchor_named = true;
    out.m_anchor_is_input = is_input;
    out.m_anchor_index = index;
}

// A declared time ratio must agree with the hops the windows fix: hop * den == anchor hop * num.
void check_ratios(const anira_model_config& model, const Derived& derived) {
    const DerivedSpec& anchor = derived.m_anchor_is_input
                                    ? derived.m_inputs[derived.m_anchor_index]
                                    : derived.m_outputs[derived.m_anchor_index];
    const auto check = [&](const std::vector<anira_tensor_spec>& specs,
                           const std::vector<DerivedSpec>& rows) {
        for (size_t i = 0; i < specs.size(); ++i) {
            const anira_tensor_spec& spec = specs[i];
            if (spec.m_role != ANIRA_ROLE_STREAMED || spec.m_ratio_den == 0) { continue; }
            if (rows[i].m_hop * spec.m_ratio_den != anchor.m_hop * spec.m_ratio_num) {
                config_error(at(spec) + "time ratio " + std::to_string(spec.m_ratio_num) + "/" +
                             std::to_string(spec.m_ratio_den) + " does not match its hop " +
                             std::to_string(rows[i].m_hop) + " against the anchor's " +
                             std::to_string(anchor.m_hop));
            }
        }
    };
    check(model.m_inputs, derived.m_inputs);
    check(model.m_outputs, derived.m_outputs);
}

// Whether a custom row's id is served: anira.v2.custom (the 2.x pass-through, kept through
// this pre-release), or an engine registered on the pipeline.
bool serves_custom_id(const std::string& id, const EngineFacts* engines) {
    if (id == k_v2_custom_engine) { return true; }
    return engines != nullptr && std::ranges::find(engines->m_ids, id) != engines->m_ids.end();
}

void check_rows(const anira_model_config& model,
                const anira_backend_id* candidates,
                uint32_t num_candidates,
                const EngineFacts* engines,
                Derived& out,
                bool default_set) {
    if (model.m_models.empty()) {
        config_error("no model entry: add a model by path or bytes at least once");
    }
    // The rows kept so far, keyed by (engine, id, pin): every custom row maps to the 2.x CUSTOM
    // backend, and the plans are resolved by row, so two registered engines on two rows are
    // two plans, and so are two rows of one engine pinned to two providers (an export per
    // backend); two rows naming one engine (or one id) with the same pin, or both without one,
    // are one plan too many.
    struct Taken {
        anira_engine m_engine;
        std::string m_engine_id;
        anira_provider m_provider;
        std::string m_provider_id;
        size_t m_row;
    };
    std::vector<Taken> taken;
    for (size_t i = 0; i < model.m_models.size(); ++i) {
        const ModelEntry& row = model.m_models[i];
        std::vector<PlanKey> plans =
            matching_plans(i, row, candidates, num_candidates, default_set);
        if (plans.empty()) { continue; }
        if (row.is_custom()) {
            if (!serves_custom_id(row.m_engine_id, engines)) {
                not_supported(at_row(i) + "custom engine '" + row.m_engine_id +
                              "' is not added to this pipeline "
                              "(anira_pipeline_add_engine)");
            }
        } else if (!backend_of(row).has_value()) {
            not_supported(
                at_row(i) + "engine '" + engine_word(row.m_engine) +
                "' is not in this build (its engines: " + engines_list(enabled_engines()) +
                "); name the candidates to skip the entries this build cannot load");
        }
        if (row.m_path.empty() && !row.has_bytes()) {
            config_error(at_row(i) + "neither a path nor bytes");
        }
        for (const Taken& other : taken) {
            if (other.m_engine == row.m_engine && other.m_engine_id == row.m_engine_id &&
                other.m_provider == row.m_provider && other.m_provider_id == row.m_provider_id) {
                config_error(at_row(other.m_row) + "and models[" + std::to_string(i) +
                             "] both name engine '" + engine_label(row) + "'" +
                             (row.is_pinned() ? " pinned to provider '" + pin_label(row) + "'"
                                              : " without a provider pin") +
                             "; one entry per engine and provider");
            }
        }
        taken.push_back(Taken{.m_engine = row.m_engine,
                              .m_engine_id = row.m_engine_id,
                              .m_provider = row.m_provider,
                              .m_provider_id = row.m_provider_id,
                              .m_row = i});
        out.m_rows.push_back(i);
        out.m_plans.insert(out.m_plans.end(),
                           std::make_move_iterator(plans.begin()),
                           std::make_move_iterator(plans.end()));
    }
    if (out.m_rows.empty()) {
        // Zero plans is a configuration the caller can fix, not a limit of this build.
        config_error("none of the " + std::to_string(model.m_models.size()) +
                     " model entries names a candidate engine (candidates: " +
                     candidates_list(candidates, num_candidates) + ")");
    }
    if (model.m_default_engine != ANIRA_ENGINE_NONE || !model.m_default_engine_id.empty()) {
        bool found = false;
        for (const ModelEntry& row : model.m_models) {
            found = found || (model.m_default_engine_id.empty()
                                  ? (row.m_engine == model.m_default_engine && !row.is_custom())
                                  : row.m_engine_id == model.m_default_engine_id);
        }
        if (!found) {
            config_error("default_engine '" +
                         (model.m_default_engine_id.empty() ? engine_word(model.m_default_engine)
                                                            : model.m_default_engine_id) +
                         "' names no model entry");
        }
    }
}

// A spec with its dynamic Time extent resolved, for the layout helpers.
anira_tensor_spec resolved_spec(const anira_tensor_spec& spec, const DerivedSpec& derived) {
    anira_tensor_spec copy = spec;
    for (size_t i = 0; i < copy.m_ndim; ++i) { copy.m_axes[i].m_extent = derived.m_dims[i]; }
    return copy;
}

// Whether an engine binds one side of its tensors by position alone, with no name to bind
// to: ExecuTorch on both sides (its method meta carries no tensor names), LibTorch on the
// output side (a method's return is a tuple of unnamed tensors); the other engines and the
// other sides have names (the graph's, the signature's, the method's arguments).
bool binds_by_position(anira_engine engine, bool is_input) noexcept {
    switch (engine) {
        case ANIRA_ENGINE_EXECUTORCH: return true;
        case ANIRA_ENGINE_LIBTORCH: return !is_input;
        default: return false;
    }
}

// The tensor records of every surviving row: a record must name a tensor of the spec; a name
// on a side its engine binds by position is refused (the name would bind nothing, and a
// silently ignored name is what the rule replaces); a layout must fit the spec and move
// axes of extent 1 alone.
void check_layouts(const anira_model_config& model, const Derived& derived) {
    for (const size_t index : derived.m_rows) {
        const ModelEntry& row = model.m_models[index];
        for (const auto& [canonical, binding] : row.m_tensors) {
            bool is_input = true;
            size_t spec_index = 0;
            const anira_tensor_spec* spec = find_spec(model, canonical, &is_input, &spec_index);
            if (spec == nullptr) {
                config_error(at_row(index) + "the tensor record names no tensor '" + canonical +
                             "'");
            }
            if (!binding.m_name.empty() && !row.is_custom() &&
                binds_by_position(row.m_engine, is_input)) {
                not_supported(at_row(index) + at(*spec) + std::string(engine_word(row.m_engine)) +
                              " binds its " + (is_input ? "inputs" : "outputs") +
                              " by position; drop the name, keep the layout");
            }
            if (binding.m_layout.empty()) { continue; }  // a name alone binds by that name
            const DerivedSpec& d =
                is_input ? derived.m_inputs[spec_index] : derived.m_outputs[spec_index];
            std::string why;
            switch (classify_layout(resolved_spec(*spec, d), binding.m_layout, &why)) {
                case LayoutKind::Identity:
                case LayoutKind::View: break;
                case LayoutKind::Transpose:
                    not_supported(at_row(index) + at(*spec) +
                                  "the layout moves an axis of extent above 1 (a transpose); "
                                  "the 2.x runtime binds the spec's axis order and can only "
                                  "re-view unit axes");
                case LayoutKind::Invalid:
                default: config_error(at_row(index) + at(*spec) + "layout " + why);
            }
        }
    }
}

void check_extensions(const anira_model_config& model,
                      const anira_context_config* config,
                      const anira_contract* contract,
                      const anira_backend_id* candidates,
                      uint32_t num_candidates,
                      const std::vector<ExtConsumer>* pipeline = nullptr) {
    anira_error local = ANIRA_ERROR_INIT;
    const anira_status status =
        ext_check_consumed(model, config, contract, candidates, num_candidates, &local, pipeline);
    if (ANIRA_FAILED(status)) { refuse(status, local.message); }
}

}  // namespace

// ---- the public pieces ----------------------------------------------------------------------

bool engine_is_candidate(anira_engine engine,
                         const std::string& engine_id,
                         const anira_backend_id* candidates,
                         uint32_t num_candidates) noexcept {
    if (candidates == nullptr) { return true; }
    for (uint32_t i = 0; i < num_candidates; ++i) {
        const anira_backend_id& id = candidates[i];
        if (id.engine_id != nullptr) {
            if (!engine_id.empty() && engine_id == id.engine_id) { return true; }
            continue;
        }
        if (id.engine == static_cast<uint32_t>(engine)) { return true; }
    }
    return false;
}

std::vector<PlanKey> matching_plans(size_t row_index,
                                    const ModelEntry& row,
                                    const anira_backend_id* candidates,
                                    uint32_t num_candidates,
                                    bool default_set) {
    std::vector<PlanKey> plans;
    if (candidates == nullptr ||
        (default_set &&
         engine_is_candidate(row.m_engine, row.m_engine_id, candidates, num_candidates))) {
        // The bridge's rule, and the default set's for an engine it names: one plan per row,
        // on its pin or on the default provider.
        plans.push_back(PlanKey{.m_row = row_index,
                                .m_provider = row.m_provider,
                                .m_provider_id = row.m_provider_id});
        return plans;
    }
    if (default_set) { return plans; }
    for (uint32_t i = 0; i < num_candidates; ++i) {
        const anira_backend_id& id = candidates[i];
        if (!engine_is_candidate(row.m_engine, row.m_engine_id, &id, 1)) { continue; }
        PlanKey plan{.m_row = row_index,
                     .m_provider = static_cast<anira_provider>(id.provider),
                     .m_provider_id = std::string(candidate_provider_id(id))};
        // A pinned entry accepts its pin alone; a neutral one whatever the candidate names.
        if (row.is_pinned() &&
            (plan.m_provider != row.m_provider || plan.m_provider_id != row.m_provider_id)) {
            continue;
        }
        if (std::ranges::find(plans, plan) == plans.end()) { plans.push_back(std::move(plan)); }
    }
    return plans;
}

bool row_is_candidate(const ModelEntry& row,
                      const anira_backend_id* candidates,
                      uint32_t num_candidates) {
    return !matching_plans(0, row, candidates, num_candidates).empty();
}

std::vector<ExtConsumer> pipeline_consumers(const StageFacts* stages, const EngineFacts* engines) {
    std::vector<ExtConsumer> consumers;
    if (stages != nullptr) {
        consumers.insert(consumers.end(), stages->m_consumers.begin(), stages->m_consumers.end());
    }
    if (engines != nullptr) {
        consumers.insert(consumers.end(), engines->m_consumers.begin(), engines->m_consumers.end());
    }
    return consumers;
}

std::optional<anira::InferenceBackend> backend_of(const ModelEntry& row) noexcept {
    // Every custom row is a plan on the 2.x CUSTOM backend, resolved by row in the plan
    // table: the pass-through and a registered engine alike (check_rows refuses an id that
    // neither is).
    if (row.is_custom()) { return anira::InferenceBackend::CUSTOM; }
    switch (row.m_engine) {
#ifdef USE_ONNXRUNTIME
        case ANIRA_ENGINE_ONNXRUNTIME: return anira::InferenceBackend::ONNX;
#endif
#ifdef USE_LIBTORCH
        case ANIRA_ENGINE_LIBTORCH: return anira::InferenceBackend::LIBTORCH;
#endif
#ifdef USE_TFLITE
        case ANIRA_ENGINE_TFLITE: return anira::InferenceBackend::TFLITE;
#endif
#ifdef USE_LITERT
        case ANIRA_ENGINE_LITERT: return anira::InferenceBackend::LITERT;
#endif
#ifdef USE_EXECUTORCH
        case ANIRA_ENGINE_EXECUTORCH: return anira::InferenceBackend::EXECUTORCH;
#endif
        default: return std::nullopt;
    }
}

std::vector<int64_t> engine_dims_of(const anira_tensor_spec& spec,
                                    const DerivedSpec& derived,
                                    const std::vector<uint32_t>& layout) {
    if (layout.empty()) { return derived.m_dims; }
    return engine_dims(resolved_spec(spec, derived), layout);
}

std::string engine_label(const ModelEntry& row) {
    return row.is_custom() ? row.m_engine_id : engine_word(row.m_engine);
}

std::vector<anira_engine> enabled_engines() {
    std::vector<anira_engine> engines;
#ifdef USE_ONNXRUNTIME
    engines.push_back(ANIRA_ENGINE_ONNXRUNTIME);
#endif
#ifdef USE_LIBTORCH
    engines.push_back(ANIRA_ENGINE_LIBTORCH);
#endif
#ifdef USE_TFLITE
    engines.push_back(ANIRA_ENGINE_TFLITE);
#endif
#ifdef USE_LITERT
    engines.push_back(ANIRA_ENGINE_LITERT);
#endif
#ifdef USE_EXECUTORCH
    engines.push_back(ANIRA_ENGINE_EXECUTORCH);
#endif
    return engines;
}

void validate(const anira_model_config& model,
              const anira_contract* contract,
              const anira_backend_id* candidates,
              uint32_t num_candidates,
              Derived& out,
              const StageFacts* stages,
              const EngineFacts* engines,
              bool default_set) {
    out = Derived{};
    if (model.m_inputs.empty()) { config_error("no input tensor"); }
    if (model.m_outputs.empty()) { config_error("no output tensor"); }
    if (contract != nullptr) { check_contract(*contract, model, stages); }
    const HardContract* hard = contract != nullptr ? contract->hard() : nullptr;
    check_state(model);
    out.m_inputs.resize(model.m_inputs.size());
    out.m_outputs.resize(model.m_outputs.size());
    for (size_t i = 0; i < model.m_inputs.size(); ++i) {
        check_spec(model.m_inputs[i], true, hard, out.m_inputs[i]);
    }
    for (size_t i = 0; i < model.m_outputs.size(); ++i) {
        check_spec(model.m_outputs[i], false, hard, out.m_outputs[i]);
    }
    resolve_anchor(model, out);
    check_ratios(model, out);
    check_rows(model, candidates, num_candidates, engines, out, default_set);
    check_layouts(model, out);
    const std::vector<ExtConsumer> consumers = pipeline_consumers(stages, engines);
    check_extensions(model,
                     nullptr,
                     contract,
                     candidates,
                     num_candidates,
                     consumers.empty() ? nullptr : &consumers);
}

anira::RingDtypes ring_dtypes_of(const anira_contract& contract, const anira_model_config& model) {
    anira::RingDtypes dtypes;
    dtypes.m_inputs.assign(model.m_inputs.size(), ANIRA_DTYPE_F32);
    dtypes.m_outputs.assign(model.m_outputs.size(), ANIRA_DTYPE_F32);
    const HardContract* hard = contract.hard();
    if (hard == nullptr) { return dtypes; }
    for (const auto& [name, dtype] : hard->m_ring_dtypes) {
        bool is_input = true;
        size_t index = 0;
        if (find_spec(model, name, &is_input, &index) == nullptr) { continue; }  // validate ran
        if (is_input) {
            dtypes.m_inputs[index] = dtype;
        } else {
            dtypes.m_outputs[index] = dtype;
        }
    }
    return dtypes;
}

HostDomains host_domains_of(const anira_contract& contract, const anira_model_config& model) {
    HostDomains domains;
    domains.m_inputs.assign(model.m_inputs.size(), ANIRA_DOMAIN_HOST);
    domains.m_outputs.assign(model.m_outputs.size(), ANIRA_DOMAIN_HOST);
    for (const auto& [name, domain] : contract.m_host_domains) {
        bool is_input = true;
        size_t index = 0;
        if (find_spec(model, name, &is_input, &index) == nullptr) { continue; }  // validate ran
        if (is_input) {
            domains.m_inputs[index] = domain;
        } else {
            domains.m_outputs[index] = domain;
        }
    }
    return domains;
}

std::vector<StateLink> state_links(const anira_model_config& model) {
    std::vector<StateLink> links;
    for (size_t i = 0; i < model.m_inputs.size(); ++i) {
        if (model.m_inputs[i].m_role != ANIRA_ROLE_STATE) { continue; }
        bool is_input = true;
        size_t output = 0;
        // validate ran: the source names a State output.
        if (find_spec(model, model.m_inputs[i].m_state_source, &is_input, &output) == nullptr) {
            continue;
        }
        links.push_back({.m_input = i, .m_output = output});
    }
    return links;
}

void check_context_extensions(const anira_context_config& config) {
    const anira_model_config no_model;
    check_extensions(no_model, &config, nullptr, nullptr, 0);
}

Anchor anchor_of(const anira_model_config& model) {
    if (model.m_inputs.empty()) { config_error("no input tensor"); }
    if (model.m_outputs.empty()) { config_error("no output tensor"); }
    Derived derived;
    derived.m_inputs.resize(model.m_inputs.size());
    derived.m_outputs.resize(model.m_outputs.size());
    for (size_t i = 0; i < model.m_inputs.size(); ++i) {
        check_spec(model.m_inputs[i], true, nullptr, derived.m_inputs[i]);
    }
    for (size_t i = 0; i < model.m_outputs.size(); ++i) {
        check_spec(model.m_outputs[i], false, nullptr, derived.m_outputs[i]);
    }
    resolve_anchor(model, derived);
    return {.m_named = derived.m_anchor_named,
            .m_is_input = derived.m_anchor_is_input,
            .m_index = derived.m_anchor_index};
}

}  // namespace anira::capi
