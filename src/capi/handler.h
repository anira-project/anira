#ifndef ANIRA_CAPI_HANDLER_H
#define ANIRA_CAPI_HANDLER_H
/*
 * The bodies of the opaque handles of anira/abi/handler.h: the pipeline, the plan report and
 * the handler. Private to src/capi (and the tests through the src/ include directory): the
 * layouts never enter the ABI.
 */
#include <anira/abi/context.h>
#include <anira/abi/handler.h>
#include <anira/abi/tensor.h>
#include <tanh/core/threading/RCU.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../scheduler/InferenceManager.h"
#include "../utils/RcuReaders.h"
#include "../utils/RtLatch.h"
#include "engine.h"
#include "handles.h"
#include "port.h"
#include "processor.h"

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

/// One plan of the table: the dense index is the position. The table spans the variants of the
/// inference stage, in variant order.
struct Plan {
    size_t m_variant = 0;                           ///< the variant: its index in the pipeline's
                                                    ///< list (m_info.variant)
    size_t m_row = 0;                               ///< the index in that variant's models[]
    anira_plan_info m_info = ANIRA_PLAN_INFO_INIT;  ///< engine_id and provider_id point into
                                                    ///< the handler's own store
                                                    ///< (anira_handler::m_plan_strings)
};

/**
 * @brief The plan rows as the selection entries read them: one published version
 *
 * anira_handler_set_plan, _set_model and _set_backend run on any thread while the stream
 * runs, and a control thread replaces the handler's plan table (anira_handler::m_plans, the
 * control side). So the rows they resolve a selection in are published through a thl::RCU and
 * read through a lease (anira_handler::m_plan_rows, m_plan_row_readers): a version is
 * immutable, and it owns everything it names. The rows are the report's plan rows by value,
 * and the two id strings a row points at are the version's own (m_strings), not the report's:
 * a lease keeps a version alive, not the report it was made from. Moved, never copied (a copy's
 * rows would point into the original's strings).
 */
struct PlanRows {
    std::vector<anira_plan_info> m_rows;  ///< dense index -> the plan's row
    std::deque<std::string> m_strings;    ///< the engine and provider ids the rows point at

    PlanRows() = default;
    ~PlanRows() = default;
    PlanRows(const PlanRows&) = delete;
    PlanRows& operator=(const PlanRows&) = delete;
    PlanRows(PlanRows&&) = default;
    PlanRows& operator=(PlanRows&&) = default;
};

/**
 * @brief The slot signature of a handler: what a slot number names on either side, for the
 * handler's life
 *
 * A slot is the tensor's position in the model config's list of its side, and every variant of
 * the inference stage has the same tensors there (anira::capi::check_same_form). The signature
 * is the handler's own record of them, so that no entry a host calls while the stream runs
 * reads the pipeline copy or a variant's derivation: a variant may be added to a playing
 * handler and the first one removed from it, and neither changes what a slot is.
 *
 * Fixed at anira_handler_create: the canonical name of every tensor (m_names). Fixed at every
 * anira_handler_prepare: one empty tensor per slot (m_empty). The rest of what a slot is lives
 * in the handler's ports and nowhere else, one fact in one place: the role is the port's arm, a
 * Static tensor's shape and dtype are its static port's value's (both anira_handler_create's),
 * the channel count, the ring dtype and the largest block of a Streamed tensor are its stream
 * port's fields (prepare's; port.h). Signature and ports are written by create and by prepare
 * alone, which no real-time entry overlaps, and only read between two prepares.
 */
struct SlotSignature {
    /// One side of the handler, indexed by slot.
    struct Side {
        /// The canonical names, in slot order: the model file names a tensor by it, a record of
        /// a refusal names it beside the slot number. Empty strings on a handler that
        /// anira_handler_create did not build (a test's white-box handler).
        std::vector<std::string> m_names;
        /// One empty tensor per slot, built at prepare from the ports: rank 2, shape
        /// {channels, 0}, the slot's dtype, host memory, no pointer (the channel count and the
        /// dtype are the stream port's; 1 and the spec's dtype for a Static slot, which has no
        /// host block). What a position of a multi form holds for a slot the call leaves out,
        /// and what a single form's staged slot goes back to (anira_handler::m_staged_inputs).
        /// Empty before the first prepare.
        std::vector<anira_tensor> m_empty;
    };
    Side m_inputs;
    Side m_outputs;

    const Side& side(bool input) const noexcept { return input ? m_inputs : m_outputs; }
};

}  // namespace anira::capi

// The handle bodies carry the C tag names the header forward-declares.
// NOLINTBEGIN(readability-identifier-naming)

struct anira_pipeline {
    /// The variants of the inference stage, copied at add_inference, in the caller's order
    /// (the index is the variant's, anira_plan_info.variant): models of one form
    /// (anira::capi::check_same_form, checked at anira_handler_create and at prepare). A
    /// handler's copy grows by anira_handler_add_variant, at the end, and keeps the entry of a
    /// variant anira_handler_remove_variant took out, emptied (m_removed), so that an index
    /// names one variant for the handler's life.
    std::vector<anira_model_config> m_variants;
    /// The variants a handler's copy no longer has (anira_handler_remove_variant), in
    /// ascending order: indices into m_variants, whose entries there are empty configs. Always
    /// empty in a pipeline a host builds.
    std::vector<size_t> m_removed;
    std::vector<anira::capi::Candidate> m_candidates;  ///< never empty after add_inference: the
                                                       ///< caller's list, or the default set
    /// Whether m_candidates is the default set (a NULL list at add_inference: every engine of
    /// the build on ANIRA_PROVIDER_CPU, the custom entries and every pin of every variant):
    /// under it an entry is one plan, on its pin or on the CPU path (validate.h
    /// matching_plans).
    bool m_default_set = false;
    bool m_has_inference = false;
    /// The one processor of the pipeline (anira_pipeline_add_processor; a second call is refused),
    /// or null. The carrier is shared with every copy of the pipeline (anira_handler_create's), so
    /// the processor's release fires once, when the last of them dies.
    std::shared_ptr<anira::capi::ProcessorCarrier> m_processor;
    /// The custom engines added to the pipeline (anira_pipeline_add_engine), in the order they
    /// were added, their ids distinct (an engine whose id the pipeline has is refused). The
    /// carriers are shared with the engine's handle, with every other pipeline the engine was
    /// added to and with every copy of the pipeline, so an engine's release fires once, when
    /// the last of them dies.
    std::vector<std::shared_ptr<const anira::capi::EngineCarrier>> m_engines;

    /// The candidate view a validate/ext call takes (pointers into the strings); control
    /// thread only. Never empty after add_inference.
    std::vector<anira_backend_id> candidate_ids() const;

    /// Whether the stage has the variant: an index of m_variants that was not removed.
    bool has_variant(size_t variant) const noexcept;
    /// The first variant the stage has, the one whose form the others have and whose defaults
    /// select a prepare's initial plan: 0 unless the first ones were removed. m_variants.size()
    /// for a stage without any (no handler has one: the last variant is never removed).
    size_t first_variant() const noexcept;
    /// How many variants the stage has: those of m_variants that were not removed.
    size_t num_variants() const noexcept { return m_variants.size() - m_removed.size(); }

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
    /// A handler over `num_inputs` and `num_outputs` tensors: its ports, one per tensor, made
    /// here once (anira::capi::Ports never resizes), and with them what is sized by the slot
    /// counts and never resized either: the names of the slot signature (empty until
    /// anira_handler_create fills them) and one latency per output.
    anira_handler(size_t num_inputs, size_t num_outputs)
        : m_input_ports(num_inputs), m_output_ports(num_outputs), m_latencies(num_outputs) {
        m_signature.m_inputs.m_names.resize(num_inputs);
        m_signature.m_outputs.m_names.resize(num_outputs);
    }

    anira_context* m_context = nullptr;  ///< add-ref'd at create, released at destroy
    anira_pipeline m_pipeline;           ///< the copy
    /// What the queries of the pipeline's custom engines answered at create (the main
    /// thread); every prepare checks the plans' providers against these answers and calls no
    /// query. Never changed after create.
    anira::capi::QueryAnswers m_query_answers;
    /// The snapshot of the last prepare (Hard), installed before the session is built: the
    /// session reads its Hard half by reference. Read while the handler is runnable, and while
    /// prepared by the session; a failed prepare unprepares the handler.
    anira_contract m_contract;
    /// What the validator derived for each variant under m_contract, in variant order,
    /// installed with it: the variant's own rows and plans (its part of the plan table) and its
    /// specs as it spells them (a window declared or derived). Every one has the form of the
    /// first (anira::capi::check_same_form).
    std::vector<anira::capi::Derived> m_variant_derived;
    /// The form of the stream, installed with m_contract: every tensor's extents, hop and
    /// channel count, the anchor and the exclusivity, which every variant derives alike
    /// (anira::capi::form_of the first variant's derivation: no rows, no plans). The session
    /// reads it by reference, and so do the handler's own getters.
    anira::capi::Derived m_derived;
    /// The ports: per side one entry per tensor of the model config's list, indexed by slot
    /// like everything else of the handler and the processor runner, and a variant of what the
    /// tensor is in here (port.h): a stream port (what a host block must carry, the session's
    /// ring), a static port (the stored whole-tensor value, in the spec's shape and dtype), a
    /// state port (the input half: the state's two buffers in the spec's shape and dtype, one
    /// read and one written per inference, and the generation they were last bound under; the
    /// output half: the input slot it feeds), a buffer port (nothing: refused under a Hard
    /// contract). Built by anira_handler_create from the pipeline copy, the Static and State
    /// values zeroed; never resized (the type has nothing that resizes) and an arm never
    /// changes. Every tensor with a stream (a hop) has a stream port: create and prepare refuse
    /// a derivation that breaks it (anira::capi::stream_port_refusal). prepare fills the stream
    /// ports' fields, zeroes the State buffers (a new session starts on a fresh state) and leaves
    /// the Static values alone, and so does reset (m_pp and m_manager are rebuilt by every prepare,
    /// the Static values set before one survive it). Declared before m_pp, which reads and
    /// writes them: destroyed after.
    anira::capi::Ports m_input_ports;
    anira::capi::Ports m_output_ports;
    std::unique_ptr<anira::capi::ProcessorRunner> m_pp;      ///< the processor runner over
                                                             ///< m_pipeline.m_processor, rebuilt by
                                                             ///< every prepare
    std::unique_ptr<anira::InferenceManager> m_manager;      ///< the session; null while
                                                             ///< unprepared (declared after m_pp:
                                                             ///< destroyed first)
    std::chrono::steady_clock::duration m_contract_wait{0};  ///< ANIRA_WAIT_CONTRACT of the pop
                                                             ///< twins: wait_ratio x host_block_max
                                                             ///< / host_rate
    /// dense index -> variant/row/backend, over every variant in variant order; rebuilt at
    /// prepare, in the order the session's plan table is built in from the same derivations
    /// (plan_requests), and extended by anira_handler_add_variant, which appends the plans of
    /// its variant behind the others as it appends their slots to the session's table: the
    /// dense index is the position here, in the session's list and in the published rows. A
    /// plan anira_handler_remove_variant took out keeps its row here, marked
    /// ANIRA_PLAN_REMOVED, as its entry stays, empty, in the session's list: no index moves
    /// before the next prepare. The control side of the table: read and written by the thread
    /// inside a control entry alone. What an entry of another thread reads is m_plan_rows.
    std::vector<anira::capi::Plan> m_plans;
    /// The engine and provider ids the rows of m_plans point at, the handler's own copies: a
    /// deque, so that a string stays where it is when plans are appended. Cleared with m_plans.
    std::deque<std::string> m_plan_strings;
    /// The plan rows the selection entries read (anira::capi::PlanRows), published from
    /// m_plans by the control thread: at the end of a prepare, before the handler is runnable,
    /// behind every change of the table by a later control entry, and empty by unprepare. Read
    /// through a lease of m_plan_row_readers, never directly.
    thl::RCU<anira::capi::PlanRows> m_plan_rows{anira::capi::PlanRows{}};
    /// The reader slots of m_plan_rows, added once, with the handler: a selection entry claims
    /// one for its call (all busy: ANIRA_ERROR_CAPACITY). Declared behind the rows it reads.
    anira::RcuReaders<anira::capi::PlanRows> m_plan_row_readers{m_plan_rows};
    /**
     * @brief The selected plan: the dense index the next submitted chunk is stamped with.
     *
     * The whole runtime selection is this one atomic, so set_plan has no pair to tear and two
     * plans on one backend stay distinct. The handler's, not the session's: the session is bound
     * to it (SessionElement::m_current_plan refers to it, as its latch pointer does to m_rt), so
     * anira_handler_get_plan answers with one load and never reaches into the session. A new
     * session starts it at 0; prepare then selects its initial plan.
     */
    std::atomic<uint32_t> m_selected_plan{0};
    /// The plan reports of the prepared handler, oldest first: the one its prepare built, then
    /// one per control entry that changed the plan table since (anira_handler_add_variant,
    /// anira_handler_remove_variant), each a whole report: its slot and extension rows name
    /// strings of its own store, its plan rows the ids of m_plan_strings, which lives as long
    /// as the reports do. A host holds raw pointers to them
    /// (anira_handler_plan_report, a prepare record's report), so each is an object of its own
    /// at one address, none is changed once a newer one exists and none is freed before the
    /// next prepare or destroy. unprepare frees the later ones and empties the first, which
    /// the next prepare fills again: the prepare's report is one object for the handler's
    /// life, as the handler's one report always was. Control thread.
    std::vector<std::unique_ptr<anira_plan_report>> m_reports;
    /// The report anira_handler_plan_report hands out: the newest of m_reports while the
    /// handler is prepared, NULL else. Stored by the control thread (prepare, before its
    /// callbacks run; a later control entry, once its report is complete; unprepare), loaded
    /// by the getter, which never reaches into the session or the report itself.
    std::atomic<const anira_plan_report*> m_published_report{nullptr};
    std::atomic<bool> m_prepared{false};  ///< release before the engines' and the processor's
                                          ///< prepares of a prepare, so the getters answer
                                          ///< inside them; acquire in every getter
    std::atomic<bool> m_runnable{false};  ///< release at the end of a successful prepare,
                                          ///< after its last callback returned: the Hard
                                          ///< entries and reset run; cleared first by
                                          ///< unprepare; acquire in those entries
    anira::RtLatch m_rt;                  ///< rt_error, the kind bits, the suppressed count
    /// The audio nodes that stream through this handler (WebAssembly, anira_em_audio_node_create
    /// counts up, anira_em_audio_node_destroy down). While any exists its render callback may be
    /// inside a process call, so anira_handler_prepare refuses; the node re-prepares the handler
    /// under its own hold (anira_em_audio_node_prepare). Always 0 on other targets.
    std::atomic<uint32_t> m_audio_nodes{0};
    /// What a slot number names on either side (SlotSignature): the canonical names, set by
    /// anira_handler_create, and the empty tensors, set by every prepare. With the ports it is
    /// everything an entry reads about a slot while the stream runs; none of them reads
    /// m_pipeline or m_variant_derived.
    anira::capi::SlotSignature m_signature;
    // The per-slot arrays a single-tensor form stages its one tensor in, one per side: copies
    // of the signature's empty tensors, made at prepare. A single form on a side with several
    // slots copies the caller's descriptor into its slot of the array, hands the array to the
    // manager, which takes one tensor per slot, and sets that entry back to the signature's
    // empty tensor before it returns. The entry of a Static or a State slot stays empty: a
    // single form never names one. Indexed by slot, like the ports. Driving thread only.
    std::vector<anira_tensor> m_staged_inputs;
    std::vector<anira_tensor> m_staged_outputs;
    /// The lengths of the model config's two tensor lists, set by anira_handler_create (the
    /// two Static entries are legal on an unprepared handler and need their bound).
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    /// The entries of the prepared session (the structs of its pool, what a processor context
    /// reports as the chunk's entry): stored by prepare from the processor's table at every
    /// sizing of the session, 0 while unprepared. anira_handler_num_entries loads it.
    std::atomic<uint32_t> m_num_entries{0};
    /// The latency of every output slot in samples, 0 for a tensor without a ring (Static,
    /// State): stored by prepare from the session's vector at every sizing of the session, so
    /// anira_handler_get_latency and anira_handler_get_latencies answer with plain loads and
    /// never reach into the session. One atomic per output slot, sized by the constructor and
    /// never resized.
    std::vector<std::atomic<uint32_t>> m_latencies;
    /**
     * @brief The control lock: one control entry of the handler at a time.
     *
     * Serialises anira_handler_prepare (anira_em_audio_node_prepare's included),
     * anira_handler_add_variant, anira_handler_remove_variant and anira_handler_destroy. No
     * real-time entry takes it, and
     * none waits for it: the getters answer from the atomics above, the Hard entries are the
     * driver thread's. m_control_holder is the thread inside the locked region
     * (std::thread::id() while nobody is): a callback of that region (an engine's or a
     * processor's prepare, a load) runs on the holding thread, so a control entry called from
     * it finds itself there and is refused (ANIRA_ERROR_INVALID_STATE; a destroy does
     * nothing): it neither locks again nor runs inside the entry that called it.
     *
     * m_live_build is set while the lock is held by an entry that runs beside the playing
     * stream (anira_handler_add_variant, anira_handler_remove_variant): the two Static
     * entries, which a callback of a plain
     * prepare may call, are refused on the holding thread then, since the driver thread is the
     * one writer of a Static store while the stream plays. They read the flag first and the
     * holder only when it is set, so a Static entry of a stream that no control entry runs
     * beside costs one load.
     */
    std::mutex m_control;
    std::atomic<std::thread::id> m_control_holder{};
    static_assert(std::atomic<std::thread::id>::is_always_lock_free,
                  "the Static entries read the holder of the control lock on the driver thread");
    std::atomic<bool> m_live_build{false};
    /// What the processor's prepare handed back for this handler (the out_prepared of
    /// anira_processor_prepare_fn; NULL is a legal value): the `prepared` pointer every phase call,
    /// the reset and the unprepare of this handler receive beside the registration's user_data.
    /// Held here for the unprepare and on the processor runner, which passes it per call.
    void* m_processor_prepared = nullptr;
    /// A successful processor prepare is outstanding: its unprepare is owed, once, when the session
    /// is released (the next prepare, a failed prepare, destroy). A refused prepare never sets it.
    bool m_processor_unprepare_owed = false;
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

namespace anira::capi {

/// What breaks the invariant every reader of a ring relies on, for a derivation and the ports of
/// a handler: one port per tensor on either side, and a stream port for every tensor with a
/// stream (a hop above 0; SessionElement::send_ring / receive_ring reach the ring through it).
/// Empty when both hold, else the side and the slot. anira_handler_create and
/// anira_handler_prepare refuse a derivation it names (ANIRA_ERROR_INTERNAL; an assertion in a
/// debug build): validate gives a hop to a Streamed tensor alone and create gives every one a
/// stream port, so neither ever does.
std::string stream_port_refusal(const Derived& derived, const Ports& inputs, const Ports& outputs);

}  // namespace anira::capi

#endif  // ANIRA_CAPI_HANDLER_H
