// The bridge from the 3.x configuration to the 2.x runtime (v3_to_v2.h): the translators
// into the 2.x configuration objects, made from what the validator derived (validate.h), for
// the C handler's sessions until the runtime cut-over, and the exported face over them
// (anira/compat/v3_to_v2.h), whose every entry is the boundary where a failure is said once
// (capi_internal.h), like the C entries of config.cpp.
#include "v3_to_v2.h"

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "capi_internal.h"
#include "ext_registry.h"
#include "handles.h"
#include "validate.h"

// ---- the translators ------------------------------------------------------------------------

namespace anira::capi {
namespace {

[[noreturn]] void refuse(anira_status status, const std::string& message) {
    throw StatusError(status, message);
}
[[noreturn]] void config_error(const std::string& message) {
    refuse(ANIRA_ERROR_CONFIG, message);
}
[[noreturn]] void not_supported(const std::string& message) {
    refuse(ANIRA_ERROR_NOT_SUPPORTED, message);
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

anira::InferenceConfig make_inference_config(const anira_model_config& model,
                                             const anira_contract& contract,
                                             const anira_backend_id* candidates,
                                             uint32_t num_candidates,
                                             const StageFacts* stages,
                                             const EngineFacts* engines,
                                             bool default_set) {
    Derived derived;
    validate(model, &contract, candidates, num_candidates, derived, stages, engines, default_set);
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
                list.push_back(engine_dims_of(specs[i],
                                              rows[i],
                                              binding == row.m_tensors.end()
                                                  ? std::vector<uint32_t>{}
                                                  : binding->second.m_layout));
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
    // A model with a declared state pair is Stateful whatever it says: the state of inference
    // k is the input of inference k + 1, so the inferences of a session run one at a time and
    // in order (the session-exclusive processor and its dispatch gate), which is also what
    // serialises the stage processor's feed, capture and re-initialisation of the state.
    const auto is_state = [](const anira_tensor_spec& spec) {
        return spec.m_role == ANIRA_ROLE_STATE;
    };
    const bool has_state_pair = std::ranges::any_of(model.m_inputs, is_state);
    return {std::move(model_data),
            std::move(shapes),
            std::move(processing_spec),
            static_cast<float>(hard.m_budget_ms),
            warm_up,
            model.m_state == ANIRA_MODEL_STATEFUL || has_state_pair,
            static_cast<float>(hard.m_wait_ratio),
            model.m_max_instances};
}

anira::CoreConfig make_core_config(const anira_context_config& config) {
    check_context_extensions(config);
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

anira_context_config make_context_config(const anira::CoreConfig& core_config) {
    anira_context_config config;
    // 0 stays "bring your own threads"; a CoreConfig never says AUTO (its default is
    // default_num_threads(), resolved at construction).
    config.m_num_threads = static_cast<uint32_t>(core_config.m_num_threads);
    config.m_wait = core_config.m_wait_strategy == anira::WaitStrategy::Blocking
                        ? ANIRA_WAIT_BLOCKING
                        : ANIRA_WAIT_SPIN_BACKOFF;
    // Explicit, not a cast: the two enums agree numerically today, the switch survives a
    // reorder.
    switch (core_config.m_log.m_level) {
        case anira::LogLevel::Debug: config.m_log_level = ANIRA_LOG_DEBUG; break;
        case anira::LogLevel::Info: config.m_log_level = ANIRA_LOG_INFO; break;
        case anira::LogLevel::Warning: config.m_log_level = ANIRA_LOG_WARNING; break;
        case anira::LogLevel::Error: config.m_log_level = ANIRA_LOG_ERROR; break;
    }
    config.m_log_drain = core_config.m_log.m_drain == anira::LogDrain::Manual
                             ? ANIRA_LOG_DRAIN_MANUAL
                             : ANIRA_LOG_DRAIN_THREAD;
    config.m_drain_interval_ms = core_config.m_log.m_drain_interval_ms;
    // Core::ensure_log_queue_locked clamps to [64, 65536] and warns, as today.
    config.m_queue_capacity =
        static_cast<uint32_t>(std::min<size_t>(core_config.m_log.m_queue_capacity, UINT32_MAX));
    // m_log_flags 0, m_sink nullptr, m_sink_user_data nullptr, no device block, m_ext
    // empty, m_upgraded false: the struct's defaults.
    return config;
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
    const Anchor anchor = anchor_of(model);
    return {buffer_size,
            sample_rate,
            allow_smaller,
            anchor.m_named ? anchor.m_index : anira::HostConfig::k_first_streamable,
            anchor.m_is_input};
}

}  // namespace anira::capi

// ---- the exported face ----------------------------------------------------------------------

namespace anira::v3compat {

using anira::capi::translate_exception;

namespace {

// The first State spec of a model, inputs first, or NULL.
const anira_tensor_spec* first_state_spec(const anira_model_config& model) noexcept {
    for (const anira_tensor_spec& spec : model.m_inputs) {
        if (spec.m_role == ANIRA_ROLE_STATE) { return &spec; }
    }
    for (const anira_tensor_spec& spec : model.m_outputs) {
        if (spec.m_role == ANIRA_ROLE_STATE) { return &spec; }
    }
    return nullptr;
}

}  // namespace

anira_status to_inference_config(const anira_model_config* model,
                                 const anira_contract* contract,
                                 const anira_engine* candidates,
                                 uint32_t num_candidates,
                                 anira::InferenceConfig& out,
                                 anira_error* err) noexcept try {
    ANIRA_CAPI_REQUIRE(model != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model is NULL");
    ANIRA_CAPI_REQUIRE(contract != nullptr,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "contract is NULL: the 2.x InferenceConfig carries the Hard contract's "
                       "budget, warmup and wait ratio");
    ANIRA_CAPI_REQUIRE(candidates != nullptr || num_candidates == 0,
                       err,
                       ANIRA_ERROR_INVALID_ARGUMENT,
                       "candidates is NULL with num_candidates %u",
                       static_cast<unsigned>(num_candidates));
    // Declared state is fed and captured by a 3.x handler's session. A bare 2.x
    // InferenceConfig reaches a 2.x InferenceHandler, which runs no feedback: the model would
    // run on a zero state without a word. The shared translator maps the role (the handler
    // needs it); this entry refuses it.
    const anira_tensor_spec* state = first_state_spec(*model);
    ANIRA_CAPI_REQUIRE(state == nullptr,
                       err,
                       ANIRA_ERROR_NOT_SUPPORTED,
                       "tensor '%s': a State tensor (declared state passing) has no 2.x "
                       "counterpart: a 2.x InferenceHandler feeds no state back; run the model "
                       "through a 3.x handler (anira_handler_create)",
                       state->m_name.c_str());
    // The bridge keeps its engine list: every engine maps to the default provider (a
    // custom engine, ANIRA_ENGINE_NONE, keeps the custom rows as before); NULL stays NULL.
    std::vector<anira_backend_id> ids;
    if (candidates != nullptr) {
        ids.reserve(num_candidates);
        for (uint32_t i = 0; i < num_candidates; ++i) {
            ids.push_back(anira_backend_id{.struct_size = sizeof(anira_backend_id),
                                           .engine = static_cast<uint32_t>(candidates[i]),
                                           .provider = ANIRA_PROVIDER_DEFAULT,
                                           .engine_id = nullptr});
        }
    }
    out = anira::capi::make_inference_config(*model,
                                             *contract,
                                             candidates != nullptr ? ids.data() : nullptr,
                                             num_candidates);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status to_core_config(const anira_context_config* config,
                            anira::CoreConfig& out,
                            anira_error* err) noexcept try {
    ANIRA_CAPI_REQUIRE(config != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "config is NULL");
    out = anira::capi::make_core_config(*config);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status to_host_config(const anira_contract* contract,
                            const anira_model_config* model,
                            anira::HostConfig& out,
                            anira_error* err) noexcept try {
    ANIRA_CAPI_REQUIRE(contract != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "contract is NULL");
    ANIRA_CAPI_REQUIRE(model != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model is NULL");
    out = anira::capi::make_host_config(*contract, *model);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

anira_status to_host_config(const anira_model_config* model,
                            float buffer_size,
                            float sample_rate,
                            bool allow_smaller_buffers,
                            anira::HostConfig& out,
                            anira_error* err) noexcept try {
    ANIRA_CAPI_REQUIRE(model != nullptr, err, ANIRA_ERROR_INVALID_ARGUMENT, "model is NULL");
    out = anira::capi::make_host_config(*model, buffer_size, sample_rate, allow_smaller_buffers);
    return ANIRA_OK;
} catch (...) { return translate_exception(err, __func__); }

std::vector<anira_engine> enabled_engines() {
    return anira::capi::enabled_engines();
}

}  // namespace anira::v3compat
