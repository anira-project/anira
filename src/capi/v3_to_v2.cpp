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
