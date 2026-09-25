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
#include <anira/abi/tensor.h>
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

#include "engine.h"
#include "handles.h"
#include "port.h"
#include "stage.h"

namespace anira::capi {

/// A candidate backend with its engine_id and provider_id strings owned (the two pointers of
/// anira_backend_id). The pointers inside m_id are not trusted after a copy:
/// anira_pipeline::candidate_ids re-points them at the strings.
struct Candidate {
    anira_backend_id m_id = ANIRA_BACKEND_ID_INIT;
    std::string m_engine_id;    ///< m_id.engine_id points here when set
    std::string m_provider_id;  ///< m_id.provider_id points here when set
};

/// A field-wise copy of a model config (the handle is move-only: its legacy contract is a
/// unique_ptr, which the copy does not carry; the bytes carriers are shared).
anira_model_config clone_model_config(const anira_model_config& model);

/// One plan of the table: the dense index is the position.
struct Plan {
    size_t m_row = 0;                                                     ///< the models[] index
    anira::InferenceBackend m_backend = anira::InferenceBackend::CUSTOM;  ///< set_backend's arg
    anira_plan_info m_info = ANIRA_PLAN_INFO_INIT;  ///< engine_id and provider_id point into
                                                    ///< the report's string store
};

}  // namespace anira::capi

// The handle bodies carry the C tag names the header forward-declares.
// NOLINTBEGIN(readability-identifier-naming)

struct anira_pipeline {
    std::vector<anira_model_config> m_variants;        ///< copied at add_inference; exactly one in
                                                       ///< this pre-release
    std::vector<anira::capi::Candidate> m_candidates;  ///< never empty after add_inference: the
                                                       ///< caller's list, or the default set
    /// Whether m_candidates is the default set (a NULL list at add_inference: every engine of
    /// the build on ANIRA_PROVIDER_CPU, the custom entries, every pin): under it an entry is
    /// one plan, on its pin or on the CPU path (validate.h matching_plans).
    bool m_default_set = false;
    bool m_has_inference = false;
    /// The one stage of the pipeline (anira_pipeline_add_stage; a second call is refused), or
    /// null. The carrier is shared with every copy of the pipeline (anira_handler_create's), so
    /// the stage's release fires once, when the last of them dies.
    std::shared_ptr<anira::capi::StageCarrier> m_stage;
    /// The custom engines added to the pipeline (anira_pipeline_add_engine), in the order they
    /// were added, their ids distinct (an engine whose id the pipeline has is refused). The
    /// carriers are shared with the engine's handle, with every other pipeline the engine was
    /// added to and with every copy of the pipeline, so an engine's release fires once, when
    /// the last of them dies.
    std::vector<std::shared_ptr<const anira::capi::EngineCarrier>> m_engines;

    /// The candidate view a validate/ext call takes (pointers into the strings); control
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
    /// What the queries of the pipeline's custom engines answered at create (the main
    /// thread); every prepare checks the plans' providers against these answers and calls no
    /// query. Never changed after create.
    anira::capi::QueryAnswers m_query_answers;
    anira_contract m_contract;  ///< the snapshot of the last successful prepare (Hard)
    anira::InferenceConfig m_inference_config;  ///< built at prepare; must outlive m_manager
                                                ///< and m_pp
    /// The ports: per side one entry per tensor of the model config's list, indexed by slot
    /// like everything else of the handler and the stage processor, and a variant of what the
    /// tensor is in here (port.h): a stream port (what a host block must carry, the session's
    /// ring), a static port (the stored whole-tensor value, in the spec's shape and dtype), a
    /// state port (the input half: the state's two buffers in the spec's shape and dtype, one
    /// read and one written per inference, and the generation they were last bound under; the
    /// output half: the input slot it feeds), a buffer port (nothing: refused under a Hard
    /// contract). Built by anira_handler_create from the pipeline copy, the Static and State
    /// values zeroed, and never resized; an arm never changes. prepare fills the stream ports'
    /// fields, zeroes the State buffers (a new session starts on a fresh state) and leaves the
    /// Static values alone, and so does reset (m_pp and m_manager are rebuilt by every prepare,
    /// the Static values set before one survive it). Declared before m_pp, which reads and
    /// writes them: destroyed after.
    std::vector<anira::capi::Port> m_input_ports;
    std::vector<anira::capi::Port> m_output_ports;
    std::unique_ptr<anira::PrePostProcessor> m_pp;       ///< the StageProcessor over
                                                         ///< m_pipeline.m_stage, rebuilt by every
                                                         ///< prepare (needs m_inference_config)
    std::unique_ptr<anira::InferenceManager> m_manager;  ///< the session; null while
                                                         ///< unprepared (declared after m_pp:
                                                         ///< destroyed first)
    anira::HostConfig m_host_config;
    std::chrono::steady_clock::duration m_contract_wait{0};  ///< ANIRA_WAIT_CONTRACT of the pop
                                                             ///< twins: wait_ratio x block_max
                                                             ///< / rate
    /// dense index -> row/backend; rebuilt at prepare only, and handed to the session as its
    /// plan table (the InferenceManager's constructor). The selected plan has no storage
    /// here: it is the session's one atomic, the dense index itself (SessionElement::
    /// m_current_plan), so set_plan has no pair to tear and two plans on one backend stay
    /// distinct.
    std::vector<anira::capi::Plan> m_plans;
    anira_plan_report m_report;
    std::atomic<bool> m_prepared{false};  ///< release before the engines' and the stage's
                                          ///< prepares of a prepare, so the getters answer
                                          ///< inside them; acquire in every getter
    std::atomic<bool> m_runnable{false};  ///< release at the end of a successful prepare,
                                          ///< after its last callback returned: the Hard
                                          ///< entries and reset run; cleared first by
                                          ///< unprepare; acquire in those entries
    anira::RtLatch m_rt;                  ///< rt_error, the kind bits, the suppressed count
    // One array of empty tensors per side, built at prepare from the ports (rank 2, shape
    // {channels, 0}, the slot's dtype, host memory, no pointer; the channel count and the dtype
    // are the stream port's, 1 and the spec's dtype for a Static slot, which has no host
    // block). A single-tensor form on a side with several slots copies the caller's descriptor
    // into its slot of the array, hands the array to the manager, which takes one tensor per
    // slot, and sets that entry back to the empty tensor before it returns. The entry of a
    // Static or a State slot stays empty: a single form never names one. Indexed by slot, like
    // the ports: the tensor's position in the model config's list of its side.
    std::vector<anira_tensor> m_input_tensors;
    std::vector<anira_tensor> m_output_tensors;
    /// The lengths of the model config's two tensor lists, set by anira_handler_create (the
    /// two Static entries are legal on an unprepared handler and need their bound).
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    /// The entries of the prepared session (the structs of its pool, what a stage context
    /// reports as the chunk's entry): set by prepare from the processor's table, 0 while
    /// unprepared. anira_handler_num_entries reads it.
    uint32_t m_num_entries = 0;
    /// What the stage's prepare handed back for this handler (the out_prepared of
    /// anira_stage_prepare_fn; NULL is a legal value): the `prepared` pointer every phase call,
    /// the reset and the unprepare of this handler receive beside the registration's user_data.
    /// Held here for the unprepare and on the stage processor, which passes it per call.
    void* m_stage_prepared = nullptr;
    /// A successful stage prepare is outstanding: its unprepare is owed, once, when the session
    /// is released (the next prepare, a failed prepare, destroy). A refused prepare never sets it.
    bool m_stage_unprepare_owed = false;
    /// ANIRA_MISS_CALLBACK: the contract's pair, cached at prepare so that the driver thread
    /// reads two plain members and not the contract's variant.
    anira_miss_fn m_miss_fn = nullptr;
    void* m_miss_user_data = nullptr;
    /// Set by the trampoline of ANIRA_MISS_CALLBACK when it ran for the block of the running
    /// call: it filled the Static output elements ahead of the host's function, which may have
    /// overwritten them, so the entry skips its own get step for that block. Cleared by every
    /// multi form before its stem. Driving thread only.
    bool m_miss_fn_ran = false;
};

// NOLINTEND(readability-identifier-naming)

#endif  // ANIRA_CAPI_HANDLER_H
