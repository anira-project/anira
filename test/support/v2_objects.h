// The 2.x configuration objects of the white-box tests (the test_scheduler cases that build a
// 2.x session, test_abi's thread case), made from the bundled 3.x files through the private
// translator the C handler prepares its sessions with (src/capi/v3_to_v2.h, reached through the
// src include directory), not through the exported bridge, which goes at the cut-over. Header
// only; the tests of the extras processors keep test/support/extras_fixtures.h until they go.
#ifndef ANIRA_TEST_SUPPORT_V2_OBJECTS_H
#define ANIRA_TEST_SUPPORT_V2_OBJECTS_H

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>

#include <anira/anira.hpp>
#include <cstdint>
#include <vector>

#include "capi/v3_to_v2.h"
#include "capi/validate.h"

namespace anira_test {

/// The 2.x InferenceConfig of a bundled model and contract file over this build's engines, one
/// candidate each on the default provider, plus the custom row anira.v2.custom and the NONE
/// candidate that keeps it when with_custom: the row a 2.x session runs on the 2.x pass-through
/// (the roundtrip it builds for a CUSTOM row without a backend). Path entries are copied into
/// the InferenceConfig, so the handles may die afterwards.
inline anira::InferenceConfig inference_config_of(const char* model_json,
                                                  const char* contract_json,
                                                  bool with_custom = false) {
    anira::ModelConfig model = anira::ModelConfig::from_file(model_json);
    if (with_custom) { model.add_model_path("anira.v2.custom", "custom-processor"); }
    const anira::ContractHandle contract = anira::ContractHandle::from_file(contract_json);
    std::vector<anira_backend_id> candidates;
    for (const anira_engine engine : anira::capi::enabled_engines()) {
        candidates.push_back({.struct_size = sizeof(anira_backend_id),
                              .engine = static_cast<uint32_t>(engine),
                              .provider = ANIRA_PROVIDER_DEFAULT,
                              .engine_id = nullptr});
    }
    if (with_custom) {
        candidates.push_back({.struct_size = sizeof(anira_backend_id),
                              .engine = ANIRA_ENGINE_NONE,
                              .provider = ANIRA_PROVIDER_DEFAULT,
                              .engine_id = nullptr});
    }
    return anira::capi::make_inference_config(*model.native(),
                                              *contract.native(),
                                              candidates.data(),
                                              static_cast<uint32_t>(candidates.size()));
}

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_V2_OBJECTS_H
