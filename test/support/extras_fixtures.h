// The bundled model configurations (extras/models/**/*.model.json and *.contract.json) as the
// 2.x runtime of this pre-release takes them: loaded with the 3.x loaders and bridged with
// anira::v3compat. Shared by the runtime tests and the fixture tests.
#ifndef ANIRA_TEST_SUPPORT_EXTRAS_FIXTURES_H
#define ANIRA_TEST_SUPPORT_EXTRAS_FIXTURES_H

#include <anira/InferenceConfig.h>
#include <anira/abi/enums.h>
#include <anira/compat/v3_to_v2.h>

#include <anira/anira.hpp>
#include <vector>

namespace anira_test {

/// The 2.x InferenceConfig of a bundled model and contract file over this build's engines.
/// Path entries are copied into the InferenceConfig, so the handle may die afterwards.
inline anira::InferenceConfig bridged(const char* model_json, const char* contract_json) {
    const anira::ModelConfig cfg = anira::ModelConfig::from_file(model_json);
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    return anira::v3compat::to_inference_config(cfg, contract, anira::v3compat::enabled_engines());
}

/// The same with the 2.x CUSTOM backend as one more entry, for a test that drives a custom
/// processor (on a build without any engine that is the only entry the bridge keeps).
inline anira::InferenceConfig bridged_with_custom(const char* model_json,
                                                  const char* contract_json) {
    anira::ModelConfig cfg = anira::ModelConfig::from_file(model_json);
    cfg.add_model_path("anira.v2.custom", "custom-processor");
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    std::vector<anira_engine> candidates = anira::v3compat::enabled_engines();
    candidates.push_back(ANIRA_ENGINE_NONE);  // keeps the custom entry
    return anira::v3compat::to_inference_config(cfg, contract, candidates);
}

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_EXTRAS_FIXTURES_H
