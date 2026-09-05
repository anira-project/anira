#include "translate.h"

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../utils/StatusError.h"
#include "ext_registry.h"
#include "handles.h"
#include "layout.h"

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
        default: return "unknown";
    }
}

const char* engine_word(anira_engine engine) {
    switch (engine) {
        case ANIRA_ENGINE_ONNXRUNTIME: return "onnxruntime";
        case ANIRA_ENGINE_LIBTORCH: return "libtorch";
        case ANIRA_ENGINE_TFLITE: return "tflite";
        case ANIRA_ENGINE_LITERT: return "litert";
        case ANIRA_ENGINE_EXECUTORCH: return "executorch";
        default: return "none";
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

// The same rule as the extension walk (ext_registry.cpp): NULL = every row is a candidate;
// a custom row carries ANIRA_ENGINE_NONE, so ANIRA_ENGINE_NONE in the list keeps the
// custom-engine rows (the anira_backend_id list of a later pre-release names them by id).
bool is_candidate(const ModelEntry& row, const anira_engine* candidates, uint32_t num_candidates) {
    if (candidates == nullptr) { return true; }
    for (uint32_t i = 0; i < num_candidates; ++i) {
        if (candidates[i] == row.m_engine) { return true; }
    }
    return false;
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
        if (spec.m_context >= spec.m_window_min) {
            config_error(where + "context " + std::to_string(spec.m_context) +
                         " must be below window_min " + std::to_string(spec.m_window_min));
        }
    } else {
        if (spec.m_window_min != 0 || spec.m_window_max != 0 || spec.m_context != 0) {
            config_error(where + std::string("a ") + role + " tensor has no window");
        }
        if (spec.m_ratio_num != 0 || spec.m_ratio_den != 0) {
            config_error(where + std::string("a ") + role + " tensor has no time ratio");
        }
    }

    out.m_channels = channel_axis.has_value() ? out.m_dims[*channel_axis] : 1;
    if (!streamed && out.m_channels != 1) {
        config_error(where + std::string("a ") + role +
                     " tensor's Channel axis must have extent 1 (got " +
                     std::to_string(out.m_channels) +
                     "): the 2.x runtime carries one channel for a non-streamed tensor");
    }
    if (is_input && spec.m_latency != 0) { config_error(where + "latency is an output property"); }
    if (spec.m_latency < 0) {
        config_error(where + "latency must not be negative (got " + std::to_string(spec.m_latency) +
                     ")");
    }
    if (spec.m_dtype != ANIRA_DTYPE_F32) {
        not_supported(where + "dtype " + hex_dtype(spec.m_dtype) +
                      ": the 2.x runtime streams float32 only");
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
            used = block * num / den + spec.m_context;
            used = std::max(used, spec.m_window_min);
            if (spec.m_window_max != ANIRA_UNBOUNDED) { used = std::min(used, spec.m_window_max); }
        }
        out.m_window_used = used;
        out.m_hop = used - spec.m_context;
        if (out.m_dims[*time_axis] == ANIRA_DYNAMIC) { out.m_dims[*time_axis] = used; }
    }
}

void check_contract(const anira_contract& contract, const anira_model_config& model) {
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
    if (hard->m_on_miss != ANIRA_MISS_BYPASS) {
        not_supported(
            "contract: on_miss HOLD_LAST and ZEROS need the 3.x runtime; the 2.x "
            "runtime bypasses on a miss");
    }
    if (hard->m_wait_ratio < 0.0) {
        config_error("contract: wait_ratio must not be negative (got " +
                     std::to_string(hard->m_wait_ratio) + ")");
    }
    for (const auto& [name, dtype] : hard->m_ring_dtypes) {
        if (find_spec(model, name, nullptr, nullptr) == nullptr) {
            config_error("contract: the ring dtype of '" + name + "' names no tensor");
        }
        if (dtype != ANIRA_DTYPE_F32) {
            not_supported("contract: the ring dtype of '" + name + "' is " + hex_dtype(dtype) +
                          "; the 2.x runtime streams float32 only");
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

void check_rows(const anira_model_config& model,
                const anira_engine* candidates,
                uint32_t num_candidates,
                Derived& out) {
    if (model.m_models.empty()) {
        config_error("no model entry: add a model by path or bytes at least once");
    }
    std::vector<std::pair<anira::InferenceBackend, size_t>> taken;
    for (size_t i = 0; i < model.m_models.size(); ++i) {
        const ModelEntry& row = model.m_models[i];
        if (!is_candidate(row, candidates, num_candidates)) { continue; }
        const std::optional<anira::InferenceBackend> backend = backend_of(row);
        if (!backend.has_value()) {
            if (row.is_custom()) {
                not_supported(at_row(i) + "custom engine '" + row.m_engine_id +
                              "' has no 2.x adapter (only '" + k_v2_custom_engine +
                              "' maps to the 2.x CUSTOM backend)");
            }
            not_supported(
                at_row(i) + "engine '" + engine_word(row.m_engine) +
                "' is not in this build (its engines: " + engines_list(enabled_engines()) +
                "); name the candidates to skip the entries this build cannot load");
        }
        if (row.m_path.empty() && !row.has_bytes()) {
            config_error(at_row(i) + "neither a path nor bytes");
        }
        for (const auto& [other, index] : taken) {
            if (other == *backend) {
                config_error(at_row(index) + "and models[" + std::to_string(i) +
                             "] both name engine '" + engine_label(row) +
                             "'; the 2.x runtime holds one model per engine");
            }
        }
        taken.emplace_back(*backend, i);
        out.m_rows.push_back(i);
    }
    if (out.m_rows.empty()) {
        const std::vector<anira_engine> named(candidates, candidates + num_candidates);
        not_supported(
            "none of the " + std::to_string(model.m_models.size()) +
            " model entries names a candidate engine (candidates: " + engines_list(named) + ")");
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
            if (binding.m_layout.empty()) { continue; }  // a name alone: bound positionally at M1
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
                      const anira_engine* candidates,
                      uint32_t num_candidates) {
    anira_error local = ANIRA_ERROR_INIT;
    const anira_status status =
        ext_check_consumed(model, config, contract, candidates, num_candidates, &local);
    if (ANIRA_FAILED(status)) { refuse(status, local.message); }
}

// The backend of a row the validator accepted (check_rows refused every row without one).
anira::InferenceBackend validated_backend(const ModelEntry& row) {
    const std::optional<anira::InferenceBackend> backend = backend_of(row);
    if (!backend.has_value()) {
        refuse(ANIRA_ERROR_INTERNAL, "a validated row has no 2.x backend");
    }
    return *backend;
}

anira::LogLevel log_level_of(anira_log_level level) {
    switch (level) {
        case ANIRA_LOG_DEBUG: return anira::LogLevel::Debug;
        case ANIRA_LOG_INFO: return anira::LogLevel::Info;
        case ANIRA_LOG_WARNING: return anira::LogLevel::Warning;
        case ANIRA_LOG_ERROR:
        default: return anira::LogLevel::Error;
    }
}

}  // namespace

// ---- the public pieces ----------------------------------------------------------------------

std::optional<anira::InferenceBackend> backend_of(const ModelEntry& row) noexcept {
    if (row.is_custom()) {
        if (row.m_engine_id == k_v2_custom_engine) { return anira::InferenceBackend::CUSTOM; }
        return std::nullopt;
    }
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
              const anira_engine* candidates,
              uint32_t num_candidates,
              Derived& out) {
    out = Derived{};
    if (model.m_inputs.empty()) { config_error("no input tensor"); }
    if (model.m_outputs.empty()) { config_error("no output tensor"); }
    if (contract != nullptr) { check_contract(*contract, model); }
    const HardContract* hard = contract != nullptr ? contract->hard() : nullptr;
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
    check_rows(model, candidates, num_candidates, out);
    check_layouts(model, out);
    check_extensions(model, nullptr, contract, candidates, num_candidates);
}

anira::InferenceConfig make_inference_config(const anira_model_config& model,
                                             const anira_contract& contract,
                                             const anira_engine* candidates,
                                             uint32_t num_candidates) {
    Derived derived;
    validate(model, &contract, candidates, num_candidates, derived);
    const HardContract& hard = *contract.hard();

    std::vector<anira::ModelData> model_data;
    for (const size_t index : derived.m_rows) {
        const ModelEntry& row = model.m_models[index];
        const anira::InferenceBackend backend = validated_backend(row);
        const auto* entry = row.m_ext.payload<EntryPayload>("entry");
        const std::string model_function = entry != nullptr ? entry->m_name : std::string();
        if (row.has_bytes()) {
            // Borrowed (decision 12): a 2.x binary ModelData never copies, and the engines
            // read the bytes only; the config handle must outlive the InferenceConfig.
            model_data.emplace_back(
                const_cast<void*>(
                    row.m_bytes->data()),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                row.m_bytes->size(),
                backend,
                model_function,
                true);
        } else {
            model_data.emplace_back(row.m_path, backend, model_function);
        }
    }

    anira::TensorShapeList input_dims;
    anira::TensorShapeList output_dims;
    for (const DerivedSpec& d : derived.m_inputs) { input_dims.push_back(d.m_dims); }
    for (const DerivedSpec& d : derived.m_outputs) { output_dims.push_back(d.m_dims); }
    // One backend-qualified shape per row whose file holds a tensor's axes in another order,
    // listed before the universal one: InferenceConfig::get_tensor_shape(backend) returns the
    // first row whose backend matches, and the universal row's backend field defaults to
    // CUSTOM, so listed first it would shadow a qualified CUSTOM row.
    std::vector<anira::TensorShape> shapes;
    for (const size_t index : derived.m_rows) {
        const ModelEntry& row = model.m_models[index];
        bool any_layout = false;
        for (const auto& [canonical, binding] : row.m_tensors) {
            any_layout = any_layout || !binding.m_layout.empty();
        }
        if (!any_layout) { continue; }
        const auto dims_of = [&](const std::vector<anira_tensor_spec>& specs,
                                 const std::vector<DerivedSpec>& rows) {
            anira::TensorShapeList list;
            for (size_t i = 0; i < specs.size(); ++i) {
                const auto binding = row.m_tensors.find(specs[i].m_name);
                if (binding == row.m_tensors.end() || binding->second.m_layout.empty()) {
                    list.push_back(rows[i].m_dims);
                } else {
                    list.push_back(
                        engine_dims(resolved_spec(specs[i], rows[i]), binding->second.m_layout));
                }
            }
            return list;
        };
        shapes.emplace_back(dims_of(model.m_inputs, derived.m_inputs),
                            dims_of(model.m_outputs, derived.m_outputs),
                            validated_backend(row));
    }
    shapes.emplace_back(input_dims, output_dims);

    std::vector<size_t> input_channels;
    std::vector<size_t> output_channels;
    std::vector<size_t> input_sizes;
    std::vector<size_t> output_sizes;
    std::vector<size_t> latencies;
    for (const DerivedSpec& d : derived.m_inputs) {
        input_channels.push_back(static_cast<size_t>(d.m_channels));
        input_sizes.push_back(static_cast<size_t>(d.m_hop));
    }
    for (size_t i = 0; i < derived.m_outputs.size(); ++i) {
        const DerivedSpec& d = derived.m_outputs[i];
        output_channels.push_back(static_cast<size_t>(d.m_channels));
        output_sizes.push_back(static_cast<size_t>(d.m_hop));
        latencies.push_back(static_cast<size_t>(model.m_outputs[i].m_latency));
    }
    anira::ProcessingSpec processing_spec(std::move(input_channels),
                                          std::move(output_channels),
                                          std::move(input_sizes),
                                          std::move(output_sizes),
                                          std::move(latencies));

    const unsigned int warm_up =
        hard.m_warmup == ANIRA_WARMUP_FIXED ? hard.m_warmup_iterations : 0U;
    return {std::move(model_data),
            std::move(shapes),
            std::move(processing_spec),
            static_cast<float>(hard.m_budget_ms),
            warm_up,
            model.m_state == ANIRA_MODEL_STATEFUL,
            static_cast<float>(hard.m_wait_ratio),
            model.m_max_instances};
}

anira::CoreConfig make_core_config(const anira_context_config& config) {
    const anira_model_config no_model;
    check_extensions(no_model, &config, nullptr, nullptr, 0);
    const unsigned int threads = config.m_num_threads == ANIRA_THREADS_AUTO
                                     ? anira::default_num_threads()
                                     : config.m_num_threads;
    const anira::WaitStrategy wait = config.m_wait == ANIRA_WAIT_BLOCKING
                                         ? anira::WaitStrategy::Blocking
                                         : anira::WaitStrategy::SpinBackoff;
    anira::CoreConfig core(threads, wait, log_level_of(config.m_log_level));
    core.m_log.m_drain = config.m_log_drain == ANIRA_LOG_DRAIN_MANUAL ? anira::LogDrain::Manual
                                                                      : anira::LogDrain::Thread;
    core.m_log.m_drain_interval_ms = config.m_drain_interval_ms;
    core.m_log.m_queue_capacity = config.m_queue_capacity;
    return core;
}

anira::HostConfig make_host_config(const anira_contract& contract,
                                   const anira_model_config& model) {
    const HardContract* hard = contract.hard();
    if (hard == nullptr) {
        not_supported(
            "contract: an Async contract has no 2.x counterpart; it arrives with "
            "the 3.x runtime");
    }
    if (hard->m_block_max == 0 || !(hard->m_rate > 0.0)) {
        config_error("contract: Hard geometry missing (block_max " +
                     std::to_string(hard->m_block_max) + ", rate " + std::to_string(hard->m_rate) +
                     "); set it with anira_contract_hard_set_geometry before preparing");
    }
    return make_host_config(model,
                            static_cast<float>(hard->m_block_max),
                            static_cast<float>(hard->m_rate),
                            hard->m_block_min < hard->m_block_max);
}

anira::HostConfig make_host_config(const anira_model_config& model,
                                   float buffer_size,
                                   float sample_rate,
                                   bool allow_smaller) {
    if (!(buffer_size > 0.0F) || !(sample_rate > 0.0F)) {
        refuse(ANIRA_ERROR_INVALID_ARGUMENT,
               "buffer_size and sample_rate must be positive (got " + std::to_string(buffer_size) +
                   ", " + std::to_string(sample_rate) + ")");
    }
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
    return {buffer_size,
            sample_rate,
            allow_smaller,
            derived.m_anchor_named ? derived.m_anchor_index : anira::HostConfig::k_first_streamable,
            derived.m_anchor_is_input};
}

}  // namespace anira::capi
