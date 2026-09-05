#ifndef ANIRA_CAPI_HANDLER_H
#define ANIRA_CAPI_HANDLER_H
/*
 * The bodies of the opaque handles of anira/abi/handler.h: the pipeline, the plan report and
 * the handler. Private to src/capi (and the tests through the src/ include directory): the
 * layouts never enter the ABI.
 */
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/context.h>
#include <anira/abi/handler.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RtLatch.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "handles.h"

namespace anira::capi {

/// A candidate backend with its engine_id string owned (anira_backend_id::engine_id is a
/// pointer). The pointer inside m_id is not trusted after a copy: anira_pipeline::candidate_ids
/// re-points it at m_engine_id.
struct Candidate {
    anira_backend_id m_id = ANIRA_BACKEND_ID_INIT;
    std::string m_engine_id;  ///< m_id.engine_id points here when set
};

/// A field-wise copy of a model config (the handle is move-only: its legacy contract is a
/// unique_ptr, which the copy does not carry; the bytes carriers are shared).
anira_model_config clone_model_config(const anira_model_config& model);

/// One plan of the table: the dense index is the position.
struct Plan {
    size_t m_row = 0;                                                     ///< the models[] index
    anira::InferenceBackend m_backend = anira::InferenceBackend::CUSTOM;  ///< set_backend's arg
    anira_plan_info m_info = ANIRA_PLAN_INFO_INIT;  ///< engine_id points into the report's
                                                    ///< string store
};

}  // namespace anira::capi

// The handle bodies carry the C tag names the header forward-declares.
// NOLINTBEGIN(readability-identifier-naming)

struct anira_pipeline {
    std::vector<anira_model_config> m_variants;        ///< copied at add_inference; exactly one in
                                                       ///< this pre-release
    std::vector<anira::capi::Candidate> m_candidates;  ///< never empty after add_inference: the
                                                       ///< caller's list, or the default set
    bool m_has_inference = false;

    /// The candidate view a translate/ext call takes (pointers into the strings); control
    /// thread only. Never empty after add_inference.
    std::vector<anira_backend_id> candidate_ids() const;

    anira_pipeline() = default;
    /// Clones the variants (anira_model_config is move-only); anira_handler_create's copy.
    anira_pipeline(const anira_pipeline& other);
    anira_pipeline& operator=(const anira_pipeline&) = delete;
    /// Moves; anira_handler_create assigns its cloning copy through these.
    anira_pipeline(anira_pipeline&&) = default;
    anira_pipeline& operator=(anira_pipeline&&) = default;
    ~anira_pipeline() = default;
};

struct anira_plan_report {
    std::vector<anira_plan_info> m_plans;
    std::vector<std::vector<anira_plan_slot>> m_inputs;   ///< per plan
    std::vector<std::vector<anira_plan_slot>> m_outputs;  ///< per plan
    std::vector<std::vector<anira_plan_ext>> m_exts;      ///< per plan
    std::deque<std::string> m_strings;  ///< backing storage of every const char* the rows
                                        ///< carry (a deque keeps pointers stable)
};

struct anira_handler {
    anira_context* m_context = nullptr;  ///< add-ref'd at create, released at destroy
    anira_pipeline m_pipeline;           ///< the copy
    anira_contract m_contract;           ///< the snapshot of the last successful prepare (Hard)
    anira::InferenceConfig m_inference_config;      ///< built at prepare; must outlive m_manager
                                                    ///< and m_pp
    std::unique_ptr<anira::PrePostProcessor> m_pp;  ///< the default 2.x processor until the
                                                    ///< stages arrive (needs m_inference_config)
    std::unique_ptr<anira::InferenceManager> m_manager;  ///< the session; null while
                                                         ///< unprepared (declared after m_pp:
                                                         ///< destroyed first)
    anira::HostConfig m_host_config;
    std::chrono::steady_clock::duration m_contract_wait{0};  ///< ANIRA_WAIT_CONTRACT of the pop
                                                             ///< twins: wait_ratio x block_max
                                                             ///< / rate
    std::vector<anira::capi::Plan> m_plans;  ///< dense index -> row/backend; rebuilt at
                                             ///< prepare only
    std::atomic<uint32_t> m_plan{0};         ///< the selected dense index (set_plan / get_plan)
    anira_plan_report m_report;
    std::atomic<bool> m_prepared{false};  ///< release at the end of a successful prepare;
                                          ///< acquire in every nonblocking entry
    anira::RtLatch m_rt;                  ///< rt_error, the kind bits, the suppressed count
    // The scratch arrays of the single-tensor forms, sized at prepare, indexed on the driver
    // thread.
    std::vector<const float* const*> m_input_ptrs;
    std::vector<size_t> m_input_num;
    std::vector<float* const*> m_output_ptrs;
    std::vector<size_t> m_output_num;
    std::vector<anira_dtype> m_input_ring_dtypes;  ///< resolved at prepare, F32 default
    std::vector<anira_dtype> m_output_ring_dtypes;
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
};

// NOLINTEND(readability-identifier-naming)

#endif  // ANIRA_CAPI_HANDLER_H
