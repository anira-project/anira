// The exported face of the translator (anira/compat/v3_to_v2.h): every entry is the boundary
// where a failure is said once (capi_internal.h), like the C entries of config.cpp.

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/utils/HostConfig.h>

#include <cstdint>
#include <vector>

#include "capi_internal.h"
#include "translate.h"

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
