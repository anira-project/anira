// The 2.x configuration objects of the white-box tests (the suites that build a 2.x session, the
// 2.x arms of test_abi, the fixture and loader tests), made from 3.x configurations through the
// private translator the C handler prepares its sessions with (src/capi/v3_to_v2.h, reached
// through the src include directory). Header only; goes with the translator in PR 10b.
#ifndef ANIRA_TEST_SUPPORT_V2_OBJECTS_H
#define ANIRA_TEST_SUPPORT_V2_OBJECTS_H

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>

#include <anira/anira.hpp>
#include <cstdint>
#include <span>
#include <vector>

#include "capi/v3_to_v2.h"
#include "capi/validate.h"
#include "capi/words.h"

namespace anira_test {

/// The 2.x InferenceConfig of a model config under a contract: one candidate per engine named,
/// on ANIRA_PROVIDER_CPU (ANIRA_ENGINE_CUSTOM keeps the 2.x custom rows, those of
/// anira.v2.custom), or every entry when none is named. The model config is taken by lvalue: a
/// bytes entry is borrowed, so the handle must outlive the result; path entries are copied.
inline anira::InferenceConfig inference_config_of(const anira::ModelConfig& model,
                                                  const anira::ContractHandle& contract,
                                                  std::span<const anira_engine> engines = {}) {
    std::vector<anira_backend_id> candidates;
    for (const anira_engine engine : engines) {
        const bool custom = engine == ANIRA_ENGINE_CUSTOM;
        candidates.push_back({.struct_size = sizeof(anira_backend_id),
                              .engine = static_cast<uint32_t>(engine),
                              .provider = ANIRA_PROVIDER_CPU,
                              .engine_id = custom ? anira::capi::k_v2_custom_engine : nullptr});
    }
    return anira::capi::make_inference_config(*model.native(),
                                              *contract.native(),
                                              candidates.empty() ? nullptr : candidates.data(),
                                              static_cast<uint32_t>(candidates.size()));
}

/// The 2.x InferenceConfig of a bundled model and contract file over this build's engines, one
/// candidate each on ANIRA_PROVIDER_CPU, plus the custom row anira.v2.custom and the candidate
/// that keeps it (ANIRA_ENGINE_CUSTOM with its id) when with_custom: the row a 2.x session runs on
/// the 2.x pass-through (the roundtrip it builds for a CUSTOM row without a backend). Path entries
/// are copied into the InferenceConfig, so the handles may die afterwards.
inline anira::InferenceConfig inference_config_of(const char* model_json,
                                                  const char* contract_json,
                                                  bool with_custom = false) {
    anira::ModelConfig model = anira::ModelConfig::from_file(model_json);
    if (with_custom) { model.add_model_path(anira::capi::k_v2_custom_engine, "custom-processor"); }
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    std::vector<anira_engine> engines = anira::capi::enabled_engines();
    if (with_custom) { engines.push_back(ANIRA_ENGINE_CUSTOM); }
    return inference_config_of(model, contract, engines);
}

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_V2_OBJECTS_H
