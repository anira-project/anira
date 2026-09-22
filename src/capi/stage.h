#ifndef ANIRA_CAPI_STAGE_H
#define ANIRA_CAPI_STAGE_H
/*
 * The stage behind anira/abi/stage.h: the refcounted carrier of the one anira_stage_desc a
 * pipeline holds and the one PrePostProcessor a C-created handler's session sees. Private to
 * src/capi (and the tests through the src/ include directory): nothing here enters the ABI.
 *
 * The phases of one chunk, in the order they run, with the library's own steps between them:
 *
 *   host end -> [pre_process]      driving thread under Hard: the ring of a Streamed slot to
 *                                   the model tensor, chunking and conversion
 *            -> feed State inputs   inference thread, the library's
 *            -> [before_inference]  inference thread
 *            -> EDGE up             the library's: host-end domain -> engine domain (nothing
 *                                   in this pre-release, both are host memory)
 *            -> engine
 *            -> EDGE down           the library's
 *            -> [after_inference]   inference thread
 *            -> capture State outputs, the library's
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
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/Exports.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
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

private:
    anira_stage_desc m_desc;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
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
    /// The packed buffers of the chunk's struct on the side the phase exposes: the inputs in
    /// pre_process and before_inference, the outputs in after_inference and post_process. The
    /// other side stays NULL, and its tensor accessor answers ANIRA_ERROR_INVALID_STATE.
    std::vector<anira::BufferF>* m_model_inputs = nullptr;
    std::vector<anira::BufferF>* m_model_outputs = nullptr;
    /// The spec's extents per slot.
    const std::vector<std::vector<int64_t>>* m_input_shapes = nullptr;
    const std::vector<std::vector<int64_t>>* m_output_shapes = nullptr;
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
/// carries what the ctx needs: the plan the chunk was stamped with, and the status slot the
/// scheduler reads after a failed phase.
///
/// The processor reads the handler's two port vectors (port.h), indexed by slot: the role of
/// a slot is its port's arm and the ring of a slot its stream port's, both plain reads. The
/// Static tensors travel through their static ports, whole and typed: every one is
/// materialised into the model's input ahead of pre_process and captured from the model's
/// output behind post_process, each under its slot's latch. A chunk that completed as zeros
/// (dropped, or failed in a stage or in the engine) captures nothing: the port holds what the
/// model produced. The declared state travels through its state ports the same way, on the
/// inference thread: the value of every State input is fed into the model's input as the first
/// step of before_inference, ahead of the stage's hook (zeroed first when the chunk's generation
/// stamp is not the one the value was last fed under: a reset, a prepare), and the model's
/// output of every State output is captured into the value of the input it feeds (the port's
/// partner) as the last step of after_inference, behind the stage's hook, unless the chunk
/// failed: the state keeps its last good value. The float atomics this class inherits from the
/// 2.x processor are not used.
class ANIRA_API StageProcessor final : public anira::PrePostProcessor {
public:
    /// What the ctx reports for a chunk stamped with one plan of the handler's table.
    struct PlanPair {
        uint32_t m_engine = ANIRA_ENGINE_NONE;
        uint32_t m_provider = ANIRA_PROVIDER_DEFAULT;
    };

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
    /// check's scratch.
    void bind(anira::SessionElement& session, std::vector<PlanPair> plans);

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

    /// The entry of the bound session's struct that owns the vector, or k_no_entry.
    size_t entry_of_inputs(const std::vector<anira::BufferF>& inputs) const noexcept
        ANIRA_NONBLOCKING;
    size_t entry_of_outputs(const std::vector<anira::BufferF>& outputs) const noexcept
        ANIRA_NONBLOCKING;
    /// The frame of one phase call: the ports, the shapes and the latch, and the buffers of
    /// the side the phase exposes (the two _with_ forms).
    StageFrame make_frame() const noexcept ANIRA_NONBLOCKING;
    StageFrame make_frame_with_inputs(std::vector<anira::BufferF>& inputs) const noexcept
        ANIRA_NONBLOCKING;
    StageFrame make_frame_with_outputs(std::vector<anira::BufferF>& outputs) const noexcept
        ANIRA_NONBLOCKING;
    /// The eight scalars of a ctx for the chunk at `entry`, and `frame` as its frame; the
    /// reserved slots stay NULL.
    anira_stage_ctx make_ctx(anira_stage_phase phase,
                             size_t entry,
                             const StageFrame& frame) const noexcept ANIRA_NONBLOCKING;
    /// Runs the stage's slot of one phase (the caller checked that it is filled); a status
    /// that is not ANIRA_OK is recorded and returned.
    anira_status run_phase(const anira_stage_ctx& ctx) noexcept ANIRA_NONBLOCKING;
    /// Records a failed phase: last-wins into rt_error, one record per kind naming the phase
    /// and `who` ran it (k_the_stage, k_the_default).
    void fail(const char* who, uint32_t phase, anira_status status) noexcept ANIRA_NONBLOCKING;
    /// The declared state around the engine call, on the inference thread (the class comment):
    /// the feed of every State input of `chunk` out of its port's value into `inputs`, and the
    /// capture of every State output out of `outputs` into the value of the input it feeds.
    void feed_state(const Chunk& chunk,
                    std::vector<anira::BufferF>& inputs) noexcept ANIRA_NONBLOCKING;
    void capture_state(std::vector<anira::BufferF>& outputs) noexcept ANIRA_NONBLOCKING;
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
    bool m_has_state = false;     ///< a State input exists: the two hooks feed and capture
    anira::SessionElement* m_session = nullptr;  ///< bound by bind(); outlived by the handler's
                                                 ///< manager, which goes first
    std::vector<PlanPair> m_plans;
    std::vector<std::vector<int64_t>> m_input_shapes;   ///< the spec's extents per tensor
    std::vector<std::vector<int64_t>> m_output_shapes;  ///< the spec's extents per tensor
    std::vector<Chunk*> m_chunks;  ///< the structs of the bound session, by entry
    /// The count check's scratch: available() of every channel of every ring of one side,
    /// read ahead of the phase (one entry per channel, in slot order). Driving thread only.
    std::vector<size_t> m_before;
};

}  // namespace anira::capi

#endif  // ANIRA_CAPI_STAGE_H
