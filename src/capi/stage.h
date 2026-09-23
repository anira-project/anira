#ifndef ANIRA_CAPI_STAGE_H
#define ANIRA_CAPI_STAGE_H
/*
 * The stage behind anira/abi/stage.h: the refcounted carrier of the one anira_stage_desc a
 * pipeline holds and the one PrePostProcessor a C-created handler's session sees. Private to
 * src/capi (and the tests through the src/ include directory): nothing here enters the ABI.
 *
 * The phases of one chunk, in the order they run, with the library's own steps between them:
 *
 *   host end -> [reset]            the first chunk of a new stream only (after prepare, after
 *                                   anira_handler_reset), on the thread of pre_process: the
 *                                   stage's own state starts over; the accessors answer the
 *                                   role alone
 *            -> [pre_process]      driving thread under Hard: the ring of a Streamed slot to
 *                                   the model tensor, chunking and conversion
 *            -> bind State pairs    inference thread, the library's: the model's State input
 *                                   over the pair's read buffer, its State output over the
 *                                   write buffer (zeroed first at a new stream)
 *            -> [before_inference]  inference thread
 *            -> EDGE up             the library's: host-end domain -> engine domain (nothing
 *                                   in this pre-release, both are host memory)
 *            -> engine
 *            -> EDGE down           the library's
 *            -> [after_inference]   inference thread
 *            -> promote State pairs, the library's: the two buffers flip, unless the chunk failed
 *            -> [post_process]      driving thread under Hard: the model tensor back to the
 *                                   host end
 *
 * A bracketed phase is the stage's when its descriptor slot is filled and anira's default body
 * when the slot is NULL (pre_process, post_process) or nothing (the two hooks). A Static input
 * is materialised ahead of pre_process and a Static output captured behind post_process, by
 * the library, whoever fills the phase.
 */
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/export.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/Exports.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "port.h"
#include "translate.h"

namespace anira::capi {

/// The one stage of a pipeline: the descriptor copied by anira_pipeline_add_stage, with the
/// strings it names owned here. The pipeline and every handler created from it share the
/// carrier (anira_handler_create copies the pipeline); release fires once, when the last of
/// them dies. The stage has no name: a pipeline holds one, so every record about it says "the
/// stage" and its consumer rows read k_stage_consumer.
class ANIRA_API StageCarrier {
public:
    /// What the plan report's anira_plan_ext rows name as the consumer of the kinds the stage
    /// declares.
    static constexpr const char* k_stage_consumer = "stage";

    /// `desc` is already the library's own record (copied within the caller's struct_size over
    /// ANIRA_STAGE_DESC_INIT and checked).
    explicit StageCarrier(const anira_stage_desc& desc);
    ~StageCarrier();
    StageCarrier(const StageCarrier&) = delete;
    StageCarrier& operator=(const StageCarrier&) = delete;
    StageCarrier(StageCarrier&&) = delete;
    StageCarrier& operator=(StageCarrier&&) = delete;

    /// The descriptor; consumed_kinds points into this carrier.
    const anira_stage_desc& desc() const noexcept { return m_desc; }
    const std::vector<std::string>& consumed_kinds() const noexcept { return m_kinds; }

    /// The stage's init slot, once per registration (per carrier), with the facts of the core
    /// in effect: the first call runs init and remembers a success, every later call answers
    /// ANIRA_OK at once; a refused init is not remembered, so the next call runs it again.
    /// ANIRA_OK without an init slot. Serialised by a mutex of the carrier: two handlers of one
    /// pipeline may prepare from two threads.
    anira_status ensure_init(const anira_init_info& info) const;

private:
    anira_stage_desc m_desc;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
    mutable std::mutex m_init_mutex;
    mutable bool m_initialised = false;
};

/// What a pipeline's stage means to the validator: which ring-moving phases it fills, and the
/// extension kinds it declares, as the consumer StageCarrier::k_stage_consumer. NULL is a
/// pipeline without a stage: the defaults everywhere.
ANIRA_API StageFacts stage_facts(const StageCarrier* stage);

/// What anira_stage_ctx::frame names: what the six context accessors of anira/abi/stage.h
/// (anira_stage_input_role to anira_stage_output_tensor) read to answer for a slot, and nothing
/// else; the two default bodies go through the accessors. One lives on the stack of each phase
/// entry point of the processor for the duration of that call. Nothing in it is built per
/// slot: the accessors build what a stage asks for. A test of the default bodies fills one by
/// hand.
struct StageFrame {
    /// The handler's ports per side, indexed by slot: the role of a slot is its port's arm, the
    /// ring of a Streamed slot its stream port's.
    const std::vector<Port>* m_input_ports = nullptr;
    const std::vector<Port>* m_output_ports = nullptr;
    /// The chunk's descriptors on the side the phase exposes (one anira_tensor per slot over
    /// the struct's buffer, ThreadSafeStruct::m_input_tensors / m_output_tensors): the inputs
    /// in pre_process and before_inference, the outputs in after_inference and post_process.
    /// The other side stays NULL, and its tensor accessor answers ANIRA_ERROR_INVALID_STATE.
    const std::vector<anira_tensor>* m_input_tensors = nullptr;
    const std::vector<anira_tensor>* m_output_tensors = nullptr;
    /// The latch a refused accessor records into (the session's); NULL records nothing.
    anira::RtLatch* m_rt = nullptr;
};

/// The one PrePostProcessor the session of a C-created handler sees: it fills an
/// anira_stage_ctx and the StageFrame it names on the stack per call and runs the stage's
/// filled slot of the phase, or the default body when the slot is NULL (pre_process,
/// post_process) or the pipeline has no stage.
///
/// The 2.x virtuals receive the buffers of a ThreadSafeStruct and never the struct, so bind()
/// builds a table of the session's structs, looked up by the address of a struct's input or
/// output vector (the structs are rebuilt by every session prepare). The position in that
/// table is the chunk's entry (anira_stage_ctx::entry, anira_handler_num_entries). The struct
/// carries what the ctx needs: the plan the chunk was stamped with (the session's plan table
/// names its engine and provider), the descriptors of its tensors (what a tensor accessor
/// hands out, a copy of the struct's), and the status slot the scheduler reads after a
/// failed phase.
///
/// The processor reads the handler's two port vectors (port.h), indexed by slot: the role of
/// a slot is its port's arm and the ring of a slot its stream port's, both plain reads. The
/// Static tensors travel through their static ports, whole and typed: every one is
/// materialised into the model's input ahead of pre_process and captured from the model's
/// output behind post_process, each under its slot's latch. A chunk that completed as zeros
/// (dropped, or failed in a stage or in the engine) captures nothing: the port holds what the
/// model produced. The declared state travels by pointer, on the inference thread: as the
/// first step of before_inference, ahead of the stage's hook, the chunk's descriptor of every
/// State input is bound to the read buffer of its port's StateSlot and the descriptor of every
/// State output to the write buffer of the input it feeds (the port's partner), the pair's read
/// buffer zeroed first when the chunk's generation stamp is not the one the pair was last bound
/// under (a reset, a prepare); the engine reads and writes the two through the descriptors,
/// zero-copy where it binds caller memory; and as the last step of after_inference, behind the
/// stage's hook, every pair flips, so the produced state is the next inference's input, unless
/// the chunk failed: the read buffer keeps the last good state. The struct's own buffer of a
/// State slot stays unused. The float atomics this class inherits from the 2.x processor are
/// not used.
///
/// The registration is shared by every handler of the pipeline, the prepared pointer is this
/// handler's: what the stage's prepare handed back (set_prepared, after it returned) reaches
/// every phase call and the reset beside the registration's user_data. The stage's reset
/// boundary is here, on the thread that runs pre_process: a chunk whose dispatch stamp is not
/// the one the last pre_process ran under is the first of a new stream (after prepare, after
/// anira_handler_reset), and the stage's reset slot runs for it, with the chunk's context in
/// ANIRA_PHASE_RESET, before its pre_process. The chunk's stamp, never the session's atomic,
/// as on the inference thread.
class ANIRA_API StageProcessor final : public anira::PrePostProcessor {
public:
    /// `config`, `stage` (NULL for a pipeline without one) and the two port vectors must
    /// outlive the processor (the handler owns all four). The ports and the model's tensors
    /// are indexed by slot, the tensor's position in the model config's list of its side: the
    /// one numbering of the handler's entries too. The vectors are never resized and no arm
    /// changes while the processor lives.
    StageProcessor(anira::InferenceConfig& config,
                   const StageCarrier* stage,
                   std::vector<Port>& input_ports,
                   std::vector<Port>& output_ports);

    /// Control thread, after the session's prepare and before its first chunk: the struct
    /// table, the ring of every stream port (the session's, at the port's slot) and the count
    /// check's scratch. The session's plan table is what a ctx reports the engine and the
    /// provider of a chunk's plan from.
    void bind(anira::SessionElement& session);

    /// Control thread, after the stage's prepare returned: what it handed back for this
    /// handler, passed to every phase call and the reset from here on. NULL until then, and
    /// for a stage without a prepare.
    void set_prepared(void* prepared) noexcept { m_prepared = prepared; }

    /// The entries of the bound session: the size of the struct table, what
    /// anira_handler_num_entries answers. 0 before bind().
    uint32_t num_entries() const noexcept { return static_cast<uint32_t>(m_chunks.size()); }

    void pre_process(std::vector<anira::RingBuffer>& input,
                     std::vector<anira::BufferF>& output,
                     anira::InferenceBackend current_inference_backend) override;
    void post_process(std::vector<anira::BufferF>& input,
                      std::vector<anira::RingBuffer>& output,
                      anira::InferenceBackend current_inference_backend) override;
    void before_inference(std::vector<anira::BufferF>& input,
                          anira::InferenceBackend current_inference_backend) override;
    void after_inference(std::vector<anira::BufferF>& output,
                         anira::InferenceBackend current_inference_backend) override;

private:
    using Chunk = anira::SessionElement::ThreadSafeStruct;

    /// "No entry": what entry_of_inputs / entry_of_outputs answer for a vector that is no
    /// struct's of the bound session.
    static constexpr size_t k_no_entry = static_cast<size_t>(-1);
    /// No chunk carries this dispatch stamp: the first pre_process after construction (every
    /// prepare rebuilds the processor) finds a new stream.
    static constexpr uint64_t k_no_generation = UINT64_MAX;

    /// The entry of the bound session's struct that owns the vector, or k_no_entry.
    size_t entry_of_inputs(const std::vector<anira::BufferF>& inputs) const noexcept
        ANIRA_NONBLOCKING;
    size_t entry_of_outputs(const std::vector<anira::BufferF>& outputs) const noexcept
        ANIRA_NONBLOCKING;
    /// The frame of one phase call: the ports and the latch, and the chunk's descriptors of
    /// the side the phase exposes (the two _with_ forms).
    StageFrame make_frame() const noexcept ANIRA_NONBLOCKING;
    StageFrame make_frame_with_inputs(const Chunk& chunk) const noexcept ANIRA_NONBLOCKING;
    StageFrame make_frame_with_outputs(const Chunk& chunk) const noexcept ANIRA_NONBLOCKING;
    /// The eight scalars of a ctx for the chunk at `entry`, and `frame` as its frame; the
    /// reserved slots stay NULL.
    anira_stage_ctx make_ctx(anira_phase phase,
                             size_t entry,
                             const StageFrame& frame) const noexcept ANIRA_NONBLOCKING;
    /// Runs the stage's slot of one phase (the caller checked that it is filled) with the
    /// prepared pointer; a status that is not ANIRA_OK is recorded and returned.
    anira_status run_phase(const anira_stage_ctx& ctx) noexcept ANIRA_NONBLOCKING;
    /// The stage's reset boundary: when the dispatch stamp of `chunk` is not the one the last
    /// pre_process ran under, it is the first chunk of a new stream (after prepare, after
    /// anira_handler_reset), so the stage's reset slot runs for it, when filled, with the
    /// chunk's context in ANIRA_PHASE_RESET over a frame without buffers (the accessors answer
    /// the role alone). The stamp is adopted either way.
    void reset_if_new_stream(const Chunk& chunk, size_t entry) noexcept ANIRA_NONBLOCKING;
    /// Records a failed phase: last-wins into rt_error, one record per kind naming the phase
    /// and `who` ran it (k_the_stage, k_the_default).
    void fail(const char* who, uint32_t phase, anira_status status) noexcept ANIRA_NONBLOCKING;
    /// The declared state around the engine call, on the inference thread (the class comment).
    /// bind_state is the one bind step: the chunk's descriptor of every State input over the
    /// read buffer of its port's StateSlot, in the spec's dtype, shape and domain, and the
    /// descriptor of every State output over the write buffer of the input it feeds, or over
    /// the same read buffer when the chunk's plan keeps its own aliasing
    /// (PlanSlot::m_state_alias); a field fill. promote_state flips every pair, what an
    /// inference produced being what the next one reads, unless the chunk's plan aliases, where
    /// the one buffer already holds the produced state.
    void bind_state(Chunk& chunk) noexcept ANIRA_NONBLOCKING;
    void promote_state(const Chunk& chunk) noexcept ANIRA_NONBLOCKING;
    /// Whether the chunk's plan keeps its own aliasing of the State pairs (the session's
    /// table; a stamp is always in range).
    bool aliases_state(const Chunk& chunk) const noexcept ANIRA_NONBLOCKING;
    /// The count check around the two ring-moving phases, over the stream ports of one side:
    /// snapshot() reads available() of every channel ahead of the phase; the check behind it
    /// repairs a shortfall (an input ring discards to its hop, an output ring is topped up with
    /// zeros) and, with `report`, records ANIRA_ERROR_CONFIG with one latched record for the first
    /// channel that moved another count than its hop.
    void snapshot(const std::vector<Port>& ports) noexcept ANIRA_NONBLOCKING;
    void check_input_hops(bool report) noexcept ANIRA_NONBLOCKING;
    void check_output_hops(bool report) noexcept ANIRA_NONBLOCKING;
    void report_hop(uint32_t phase,
                    size_t tensor,
                    size_t channel,
                    size_t expected,
                    size_t moved) noexcept ANIRA_NONBLOCKING;
    /// Who the count check reports for a ring-moving phase: k_the_stage when the stage fills
    /// the phase, k_the_default when the default body ran.
    const char* mover(uint32_t phase) const noexcept ANIRA_NONBLOCKING;

    /// The words a record about a phase opens with: the stage's callback ran, or the default
    /// body did.
    static constexpr const char* k_the_stage = "the stage";
    static constexpr const char* k_the_default = "the default body";

    const StageCarrier* m_stage;  ///< NULL: the defaults everywhere
    /// The handler's ports. bind() sets the stream ports' rings; pre_process reads the static
    /// input ports, post_process writes the static output ports.
    std::vector<Port>& m_input_ports;
    std::vector<Port>& m_output_ports;
    bool m_fills_pre = false;     ///< the stage fills pre_process; else the default body runs
    bool m_fills_post = false;    ///< the same for post_process
    bool m_fills_before = false;  ///< the stage fills before_inference; else the hook returns
    bool m_fills_after = false;   ///< the same for after_inference
    bool m_has_state = false;     ///< a State input exists: the two hooks bind and promote
    anira::SessionElement* m_session = nullptr;  ///< bound by bind(); outlived by the handler's
                                                 ///< manager, which goes first
    /// What the stage's prepare handed back for this handler (set_prepared); NULL without one.
    void* m_prepared = nullptr;
    /// The dispatch stamp the last pre_process ran under; k_no_generation after construction.
    /// Written and read on the thread that runs pre_process alone.
    uint64_t m_pre_generation = k_no_generation;
    std::vector<std::vector<int64_t>> m_input_shapes;   ///< the spec's extents per tensor
    std::vector<std::vector<int64_t>> m_output_shapes;  ///< the spec's extents per tensor
    std::vector<Chunk*> m_chunks;  ///< the structs of the bound session, by entry
    /// The count check's scratch: available() of every channel of every ring of one side,
    /// read ahead of the phase (one entry per channel, in slot order). Driving thread only.
    std::vector<size_t> m_before;
};

}  // namespace anira::capi

#endif  // ANIRA_CAPI_STAGE_H
