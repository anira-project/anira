#ifndef ANIRA_CAPI_STAGE_H
#define ANIRA_CAPI_STAGE_H
/*
 * The stage chain behind anira/abi/stage.h: the refcounted carrier of one anira_stage_desc and
 * the one PrePostProcessor a C-created handler's session sees. Private to src/capi (and the
 * tests through the src/ include directory): nothing here enters the ABI.
 */
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/export.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/Exports.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "port.h"
#include "translate.h"

namespace anira::capi {

/// One stage of a pipeline: the descriptor copied by anira_pipeline_add_stage, with the strings
/// it names owned here. The pipeline and every handler created from it share the carrier
/// (anira_handler_create copies the pipeline); release fires once, when the last of them dies.
class ANIRA_API StageCarrier {
public:
    /// `desc` is already the library's own record (copied within the caller's struct_size over
    /// ANIRA_STAGE_DESC_INIT and checked); `index` is the stage's position in the chain, the
    /// name of a stage without one ("stage#<index>").
    StageCarrier(const anira_stage_desc& desc, size_t index);
    ~StageCarrier();
    StageCarrier(const StageCarrier&) = delete;
    StageCarrier& operator=(const StageCarrier&) = delete;
    StageCarrier(StageCarrier&&) = delete;
    StageCarrier& operator=(StageCarrier&&) = delete;

    /// The descriptor; name and consumed_kinds point into this carrier.
    const anira_stage_desc& desc() const noexcept { return m_desc; }
    /// Never NULL, never empty.
    const char* name() const noexcept { return m_name.c_str(); }
    const std::vector<std::string>& consumed_kinds() const noexcept { return m_kinds; }

private:
    anira_stage_desc m_desc;
    std::string m_name;
    std::vector<std::string> m_kinds;
    std::vector<const char*> m_kind_pointers;
};

/// The chain of a pipeline, in the order of the anira_pipeline_add_stage calls.
using StageChain = std::vector<std::shared_ptr<StageCarrier>>;

/// What a chain means to the validator: which ring-moving phases a stage fills, and the
/// extension kinds the stages declare, as consumers named after them.
ANIRA_API StageFacts stage_facts(const StageChain& chain);

/// The one PrePostProcessor the session of a C-created handler sees: it fills an
/// anira_stage_ctx on the stack per call and runs the chain's filled slots in chain order, or
/// the default body of a phase, once, when no stage fills it.
///
/// The 2.x virtuals receive the buffers of a ThreadSafeStruct and never the struct, so bind()
/// builds a table from the address of each struct's input and output vector to the struct
/// (the structs are rebuilt by every session prepare). The struct carries what the ctx needs:
/// the plan the chunk was stamped with, and the status slot the scheduler reads after a failed
/// phase.
///
/// The chain reads the handler's two port vectors (port.h), indexed by slot: the role of a
/// slot is its port's arm and the ring of a slot its stream port's, both plain reads. The
/// Static tensors travel through their static ports, whole and typed: every one is
/// materialised into model_inputs ahead of any stage's pre_process and captured from
/// model_outputs behind the last post_process, each under its slot's latch. A chunk that
/// completed as zeros (dropped, or failed in a stage or in the engine) captures nothing: the
/// port holds what the model produced. The float atomics this class inherits from the 2.x
/// processor are not used.
class ANIRA_API StageChainProcessor final : public anira::PrePostProcessor {
public:
    /// What the ctx reports for a chunk stamped with one plan of the handler's table.
    struct PlanPair {
        uint32_t m_engine = ANIRA_ENGINE_NONE;
        uint32_t m_provider = ANIRA_PROVIDER_DEFAULT;
    };

    /// `config`, `chain` and the two port vectors must outlive the processor (the handler
    /// owns all five). The ports and the model's tensors are indexed by slot, the tensor's
    /// position in the model config's list of its side: the one numbering of the handler's
    /// entries too. The vectors are never resized and no arm changes while the processor lives.
    StageChainProcessor(anira::InferenceConfig& config,
                        const StageChain& chain,
                        std::vector<Port>& input_ports,
                        std::vector<Port>& output_ports);

    /// Control thread, after the session's prepare and before its first chunk: the struct
    /// table, the ring of every stream port (the session's, at the port's slot), the ring
    /// arrays a ctx carries and one preallocated tensor array per struct and side.
    void bind(anira::SessionElement& session, std::vector<PlanPair> plans);

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
    /// One ThreadSafeStruct of the bound session and the tensor arrays its ctx points at.
    struct Chunk {
        anira::SessionElement::ThreadSafeStruct* m_struct = nullptr;
        std::vector<anira_tensor> m_inputs;   ///< re-pointed on every call
        std::vector<anira_tensor> m_outputs;  ///< re-pointed on every call
    };

    Chunk* chunk_of_inputs(const std::vector<anira::BufferF>& inputs) noexcept ANIRA_NONBLOCKING;
    Chunk* chunk_of_outputs(const std::vector<anira::BufferF>& outputs) noexcept ANIRA_NONBLOCKING;
    anira_stage_ctx make_ctx(anira_stage_phase phase,
                             const Chunk& chunk) const noexcept ANIRA_NONBLOCKING;
    /// Runs the filled slots of one phase in chain order; the first status that is not
    /// ANIRA_OK ends the phase, is recorded with the stage's name and returned.
    anira_status run_phase(const anira_stage_ctx& ctx, bool on_driver) noexcept ANIRA_NONBLOCKING;
    /// Records a failed phase: last-wins into rt_error, one record per kind naming the stage.
    void fail(const char* stage, uint32_t phase, anira_status status) noexcept ANIRA_NONBLOCKING;
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

    const StageChain& m_chain;
    /// The handler's ports. bind() sets the stream ports' rings; pre_process reads the static
    /// input ports, post_process writes the static output ports.
    std::vector<Port>& m_input_ports;
    std::vector<Port>& m_output_ports;
    /// The first and the last stage that fill each ring-moving phase (NULL: none does, the
    /// default body runs): whom the count check names.
    const StageCarrier* m_first_pre = nullptr;
    const StageCarrier* m_last_pre = nullptr;
    const StageCarrier* m_first_post = nullptr;
    const StageCarrier* m_last_post = nullptr;
    bool m_fills_before = false;  ///< a stage fills before_inference; else the hook returns
    bool m_fills_after = false;   ///< the same for after_inference
    anira::SessionElement* m_session = nullptr;  ///< bound by bind(); outlived by the handler's
                                                 ///< manager, which goes first
    std::vector<PlanPair> m_plans;
    std::vector<std::vector<int64_t>> m_input_shapes;   ///< the spec's extents per tensor
    std::vector<std::vector<int64_t>> m_output_shapes;  ///< the spec's extents per tensor
    std::vector<Chunk> m_chunks;
    /// What anira_stage_ctx::input_rings and output_rings point at, the one array the struct
    /// asks for: the stream ports' rings by slot, NULL for a slot of another role. Filled by
    /// bind() from the ports and read by nothing in here but the ctx fill.
    std::vector<anira_ring*> m_ctx_input_rings;
    std::vector<anira_ring*> m_ctx_output_rings;
    /// The count check's scratch: available() of every channel of every ring of one side,
    /// read ahead of the phase (one entry per channel, in slot order). Driving thread only.
    std::vector<size_t> m_before;
};

}  // namespace anira::capi

#endif  // ANIRA_CAPI_STAGE_H
