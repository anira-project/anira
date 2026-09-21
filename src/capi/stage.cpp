// anira/abi/stage.h: the ring accessors, the context accessors, the two default bodies, and the
// stage chain behind them (the carrier of a descriptor and the one PrePostProcessor a C-created
// handler's session sees).
//
// Everything a phase callback can reach is ANIRA_NONBLOCKING: the ring accessors are thin
// dtype-checked calls onto anira_ring (anira/utils/RingBuffer.h), a refusal records into the
// latch of the session that owns the ring (anira_ring::owner) and logs one record per kind
// naming the running stage; the context accessors answer per slot out of the StageFrame the
// chain fills on the stack next to its anira_stage_ctx, and record into the frame's latch. No
// handler, no lock, no allocation on any per-call path.
#include "stage.h"

#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/Logger.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ext_registry.h"
#include "port.h"
#include "translate.h"

// The members of anira_ring dispatch through std::visit, which is declared to throw
// std::bad_variant_access for a valueless variant. A ring never is one: its arm is emplaced by
// initialize_with_positions() on the control thread, where a failed construction propagates,
// and nothing on a per-call path replaces the arm. The check sees the inline visit from every
// noexcept function below, so it is off for this file; nothing here throws on its own.
// NOLINTBEGIN(bugprone-exception-escape)

namespace {

// What a record says when an accessor is refused outside a pre_process or post_process
// callback of the chain (a ring pointer kept across calls, which the contract forbids).
constexpr const char* k_no_stage = "(none running)";

[[maybe_unused]] const char* running_stage(const anira_ring& ring) noexcept ANIRA_NONBLOCKING {
    const anira::RingOwner* owner = ring.owner();
    return owner != nullptr && owner->m_stage != nullptr ? owner->m_stage : k_no_stage;
}

// Records a refusal into the latch of the session that owns the ring. True on the kind's first
// occurrence since the latch was re-armed: log it. A ring outside a session records nothing.
bool ring_record(const anira_ring& ring, anira_status status) noexcept ANIRA_NONBLOCKING {
    const anira::RingOwner* owner = ring.owner();
    return owner != nullptr && owner->m_rt != nullptr && owner->m_rt->record(status);
}

// The channel check of every accessor that names one. A NULL ring is not recorded: it has no
// session to record into.
bool ring_channel(const anira_ring* ring,
                  uint32_t channel,
                  [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    if (ring == nullptr) { return false; }
    const size_t channels = ring->num_channels();
    if (channel < channels) { return true; }
    if (ring_record(*ring, ANIRA_ERROR_INVALID_ARGUMENT)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: stage '%s': channel %u is out of range, the ring has %zu",
                               entry,
                               running_stage(*ring),
                               static_cast<unsigned int>(channel),
                               channels);
    }
    return false;
}

// The checks of an accessor that moves elements: the channel, the memory, then the dtype, which
// must be the ring's own (nothing converts, in either direction).
bool ring_transfer(const anira_ring* ring,
                   uint32_t channel,
                   const void* memory,
                   anira_dtype dtype,
                   size_t count,
                   [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    if (!ring_channel(ring, channel, entry)) { return false; }
    if (memory == nullptr && count > 0) {
        if (ring_record(*ring, ANIRA_ERROR_INVALID_ARGUMENT)) {
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: stage '%s': NULL memory for %zu elements",
                                   entry,
                                   running_stage(*ring),
                                   count);
        }
        return false;
    }
    if (dtype == ring->dtype()) { return true; }
    if (ring_record(*ring, ANIRA_ERROR_CONFIG)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: stage '%s': dtype 0x%x is not the ring's 0x%x; nothing "
                               "converts",
                               entry,
                               running_stage(*ring),
                               static_cast<unsigned int>(dtype),
                               static_cast<unsigned int>(ring->dtype()));
    }
    return false;
}

[[maybe_unused]] const char* phase_word(uint32_t phase) noexcept ANIRA_NONBLOCKING {
    switch (phase) {
        case ANIRA_PHASE_PRE_PROCESS: return "pre_process";
        case ANIRA_PHASE_POST_PROCESS: return "post_process";
        case ANIRA_PHASE_BEFORE_INFERENCE: return "before_inference";
        case ANIRA_PHASE_AFTER_INFERENCE: return "after_inference";
        default: return "unknown phase";
    }
}

anira_stage_fn phase_slot(const anira_stage_desc& desc, uint32_t phase) noexcept ANIRA_NONBLOCKING {
    switch (phase) {
        case ANIRA_PHASE_PRE_PROCESS: return desc.pre_process;
        case ANIRA_PHASE_POST_PROCESS: return desc.post_process;
        case ANIRA_PHASE_BEFORE_INFERENCE: return desc.before_inference;
        case ANIRA_PHASE_AFTER_INFERENCE: return desc.after_inference;
        default: return nullptr;
    }
}

// Element `offset` of a packed run of `element_size`-byte elements.
void* element_at(void* base, size_t offset, size_t element_size) noexcept ANIRA_NONBLOCKING {
    return static_cast<unsigned char*>(base) + (offset * element_size);
}

using anira::capi::Port;
using anira::capi::StageFrame;

// The frame a ctx names; NULL for a NULL ctx and for a ctx without one. A forged frame is not
// defended against.
const StageFrame* frame_of(const anira_stage_ctx* ctx) noexcept ANIRA_NONBLOCKING {
    return ctx != nullptr ? static_cast<const StageFrame*>(ctx->frame) : nullptr;
}

// Records a refused context accessor into the frame's latch. True on the kind's first
// occurrence since the latch was re-armed: log it. A frame without a latch records nothing.
bool frame_record(const StageFrame& frame, anira_status status) noexcept ANIRA_NONBLOCKING {
    return frame.m_rt != nullptr && frame.m_rt->record(status);
}

[[maybe_unused]] const char* running_stage(const StageFrame& frame) noexcept ANIRA_NONBLOCKING {
    return frame.m_stage != nullptr ? frame.m_stage : k_no_stage;
}

// The slot check of every context accessor: the port of the slot on one side, or NULL. A slot
// at or beyond the side's count is recorded; a NULL ctx and a ctx without a frame name no latch
// to record into.
const Port* ctx_port(const anira_stage_ctx* ctx,
                     bool input,
                     uint32_t slot,
                     [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr) { return nullptr; }
    const std::vector<Port>* ports = input ? frame->m_input_ports : frame->m_output_ports;
    if (ports == nullptr) { return nullptr; }
    if (slot < ports->size()) { return &(*ports)[slot]; }
    if (frame_record(*frame, ANIRA_ERROR_INVALID_ARGUMENT)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: stage '%s': slot %u is out of range, the model has %zu %s "
                               "tensors",
                               entry,
                               running_stage(*frame),
                               static_cast<unsigned int>(slot),
                               ports->size(),
                               input ? "input" : "output");
    }
    return nullptr;
}

// The ring of a slot's port in the one phase that hands out the rings of its side; NULL, which
// is an answer and not a refusal, for a port of another role and in every other phase.
anira_ring* port_ring(const Port* port, bool ring_phase) noexcept ANIRA_NONBLOCKING {
    if (port == nullptr || !ring_phase) { return nullptr; }
    const auto* stream = std::get_if<anira::capi::StreamPort>(port);
    return stream != nullptr ? stream->m_ring : nullptr;
}

// The model end of a slot: one descriptor over the memory the struct's buffer holds right now.
// An engine may swap that memory between two inferences (LibTorchProcessor, TFLiteProcessor),
// so the descriptor is built per question and nothing of it survives a call. A field fill, no
// allocation. The frame names the buffers of a side only in the phases that expose it.
anira_status ctx_tensor(const anira_stage_ctx* ctx,
                        bool input,
                        uint32_t slot,
                        anira_tensor* out,
                        [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    // Every refusal leaves the all-zero record of a refused anira_tensor_init_* factory.
    if (out != nullptr) { std::memset(out, 0, sizeof(*out)); }
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (out == nullptr) {
        if (frame_record(*frame, ANIRA_ERROR_INVALID_ARGUMENT)) {
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: stage '%s': NULL out for slot %u",
                                   entry,
                                   running_stage(*frame),
                                   static_cast<unsigned int>(slot));
        }
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (ctx_port(ctx, input, slot, entry) == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    std::vector<anira::BufferF>* buffers = input ? frame->m_model_inputs : frame->m_model_outputs;
    const std::vector<std::vector<int64_t>>* shapes =
        input ? frame->m_input_shapes : frame->m_output_shapes;
    // Not recorded: the status is the report.
    if (buffers == nullptr || shapes == nullptr || slot >= buffers->size() ||
        slot >= shapes->size()) {
        return ANIRA_ERROR_INVALID_STATE;
    }
    const std::vector<int64_t>& shape = (*shapes)[slot];
    anira_tensor_init_host(out,
                           (*buffers)[slot].get_write_pointer(0),
                           ANIRA_DTYPE_F32,
                           static_cast<uint32_t>(shape.size()),
                           shape.data());
    return ANIRA_OK;
}

}  // namespace

// ==== the ring accessors ======================================================================

anira_dtype ANIRA_CALL anira_ring_dtype(const anira_ring* ring) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ring != nullptr ? ring->dtype() : 0U;
}

uint32_t ANIRA_CALL anira_ring_num_channels(const anira_ring* ring)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ring != nullptr ? static_cast<uint32_t>(ring->num_channels()) : 0U;
}

size_t ANIRA_CALL anira_ring_available(const anira_ring* ring,
                                       uint32_t channel) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ring_channel(ring, channel, __func__) ? ring->available(channel) : 0;
}

size_t ANIRA_CALL anira_ring_available_past(const anira_ring* ring,
                                            uint32_t channel) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ring_channel(ring, channel, __func__) ? ring->available_past(channel) : 0;
}

size_t ANIRA_CALL anira_ring_pop_block(anira_ring* ring,
                                       uint32_t channel,
                                       void* out,
                                       anira_dtype dtype,
                                       size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!ring_transfer(ring, channel, out, dtype, count, __func__)) { return 0; }
    return ring->pop_block(channel, out, dtype, count);
}

size_t ANIRA_CALL anira_ring_peek_past_block(const anira_ring* ring,
                                             uint32_t channel,
                                             void* out,
                                             anira_dtype dtype,
                                             size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!ring_transfer(ring, channel, out, dtype, count, __func__)) { return 0; }
    return ring->peek_past_block(channel, out, dtype, count);
}

size_t ANIRA_CALL anira_ring_push_block(anira_ring* ring,
                                        uint32_t channel,
                                        const void* in,
                                        anira_dtype dtype,
                                        size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (!ring_transfer(ring, channel, in, dtype, count, __func__)) { return 0; }
    return ring->push_block(channel, in, dtype, count);
}

size_t ANIRA_CALL anira_ring_push_fill(anira_ring* ring,
                                       uint32_t channel,
                                       const void* value,
                                       anira_dtype dtype,
                                       size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    // One element is read whatever the count, so a NULL value is refused at a count of 0 too.
    if (!ring_transfer(ring, channel, value, dtype, std::max<size_t>(count, 1), __func__)) {
        return 0;
    }
    return ring->push_fill(channel, value, dtype, count);
}

size_t ANIRA_CALL anira_ring_discard(anira_ring* ring,
                                     uint32_t channel,
                                     size_t count) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ring_channel(ring, channel, __func__) ? ring->discard(channel, count) : 0;
}

size_t ANIRA_CALL anira_ring_pop_windows(anira_ring* ring,
                                         uint32_t channel,
                                         void* out,
                                         anira_dtype dtype,
                                         size_t num_new,
                                         size_t num_old,
                                         size_t offset,
                                         uint32_t num_batches) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const size_t written = static_cast<size_t>(num_batches) * (num_new + num_old);
    if (!ring_transfer(ring, channel, out, dtype, written, __func__)) { return 0; }
    return ring->pop_windows(channel, out, dtype, num_new, num_old, offset, num_batches);
}

// ==== the context accessors ===================================================================
// A stage asks per slot; nothing is laid out for it up front. A refusal is recorded into the
// frame's latch, the session's, with one record per kind naming the running stage.

anira_role ANIRA_CALL anira_stage_input_role(const anira_stage_ctx* ctx,
                                             uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const Port* port = ctx_port(ctx, /*input=*/true, slot, __func__);
    return port != nullptr ? anira::capi::port_role(*port) : ANIRA_ROLE_FORCE32;
}

anira_role ANIRA_CALL anira_stage_output_role(const anira_stage_ctx* ctx,
                                              uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const Port* port = ctx_port(ctx, /*input=*/false, slot, __func__);
    return port != nullptr ? anira::capi::port_role(*port) : ANIRA_ROLE_FORCE32;
}

anira_ring* ANIRA_CALL anira_stage_input_ring(const anira_stage_ctx* ctx,
                                              uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const Port* port = ctx_port(ctx, /*input=*/true, slot, __func__);
    return port_ring(port, ctx != nullptr && ctx->phase == ANIRA_PHASE_PRE_PROCESS);
}

anira_ring* ANIRA_CALL anira_stage_output_ring(const anira_stage_ctx* ctx,
                                               uint32_t slot) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    const Port* port = ctx_port(ctx, /*input=*/false, slot, __func__);
    return port_ring(port, ctx != nullptr && ctx->phase == ANIRA_PHASE_POST_PROCESS);
}

anira_status ANIRA_CALL anira_stage_input_tensor(const anira_stage_ctx* ctx,
                                                 uint32_t slot,
                                                 anira_tensor* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_tensor(ctx, /*input=*/true, slot, out, __func__);
}

anira_status ANIRA_CALL anira_stage_output_tensor(const anira_stage_ctx* ctx,
                                                  uint32_t slot,
                                                  anira_tensor* out)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_tensor(ctx, /*input=*/false, slot, out, __func__);
}

// ==== the default bodies ======================================================================
// The 2.x default processor's streamed branches (PrePostProcessor.cpp) over anira_ring* and
// anira_tensor, written with the context accessors a stage has: the ring of a slot, then its
// model end. The tensors are read as packed row-major memory, which is how the chain fills
// them. The status is returned and never recorded here: a stage that calls the default for the
// slots it does not handle itself may drop it, and the chain records what a stage returns.

anira_status ANIRA_CALL anira_stage_default_pre_process(const anira_stage_ctx* ctx)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (ctx == nullptr || ctx->phase != ANIRA_PHASE_PRE_PROCESS) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->num_inputs == 0) { return ANIRA_OK; }
    if (ctx->frame == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira_status status = ANIRA_OK;
    for (uint32_t slot = 0; slot < ctx->num_inputs; ++slot) {
        anira_ring* ring = anira_stage_input_ring(ctx, slot);
        if (ring == nullptr) { continue; }  // not a Streamed tensor
        anira_tensor tensor;
        const anira_status exposed = anira_stage_input_tensor(ctx, slot, &tensor);
        if (exposed != ANIRA_OK) {
            status = exposed;
            continue;
        }
        const anira_dtype dtype = ring->dtype();
        void* data = anira_tensor_data(&tensor, dtype);
        const size_t channels = ring->num_channels();
        const size_t hop = ring->hop();
        // Elements per channel: a multichannel tensor ([1, 4, 1]) holds channels * hop elements
        // without being a window.
        const size_t per_channel = channels > 0 ? anira_tensor_num_elements(&tensor) / channels : 0;
        if (data == nullptr || per_channel < hop) {
            // Another dtype than the ring's (nothing converts), or a tensor one hop does not
            // fit into: this tensor and its ring stay untouched.
            status = ANIRA_ERROR_CONFIG;
            continue;
        }
        // A receptive-field model holds a window longer than its hop: the head of the window
        // is the ring's history, read before the pop advances the read position.
        const size_t history = per_channel - hop;
        const size_t element_size = ring->element_size();
        for (size_t channel = 0; channel < channels; ++channel) {
            void* window = element_at(data, channel * per_channel, element_size);
            if (history > 0) { ring->peek_past_block(channel, window, dtype, history); }
            ring->pop_block(channel, element_at(window, history, element_size), dtype, hop);
        }
    }
    return status;
}

anira_status ANIRA_CALL anira_stage_default_post_process(const anira_stage_ctx* ctx)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (ctx == nullptr || ctx->phase != ANIRA_PHASE_POST_PROCESS) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->num_outputs == 0) { return ANIRA_OK; }
    if (ctx->frame == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    anira_status status = ANIRA_OK;
    for (uint32_t slot = 0; slot < ctx->num_outputs; ++slot) {
        anira_ring* ring = anira_stage_output_ring(ctx, slot);
        if (ring == nullptr) { continue; }  // not a Streamed tensor
        anira_tensor tensor;
        const anira_status exposed = anira_stage_output_tensor(ctx, slot, &tensor);
        if (exposed != ANIRA_OK) {
            status = exposed;
            continue;
        }
        const anira_dtype dtype = ring->dtype();
        void* data = anira_tensor_data(&tensor, dtype);
        const size_t channels = ring->num_channels();
        const size_t hop = ring->hop();
        if (data == nullptr || anira_tensor_num_elements(&tensor) < channels * hop) {
            status = ANIRA_ERROR_CONFIG;
            continue;
        }
        const size_t element_size = ring->element_size();
        for (size_t channel = 0; channel < channels; ++channel) {
            ring->push_block(channel, element_at(data, channel * hop, element_size), dtype, hop);
        }
    }
    return status;
}

// ==== the carrier and the chain ===============================================================

namespace anira::capi {

StageCarrier::StageCarrier(const anira_stage_desc& desc, size_t index) : m_desc(desc) {
    if (desc.name != nullptr && desc.name[0] != '\0') {
        m_name = desc.name;
    } else {
        m_name = "stage#" + std::to_string(index);
    }
    m_kinds.reserve(desc.num_consumed_kinds);
    for (uint32_t i = 0; i < desc.num_consumed_kinds; ++i) {
        m_kinds.emplace_back(desc.consumed_kinds[i]);
    }
    m_kind_pointers.reserve(m_kinds.size());
    for (const std::string& kind : m_kinds) { m_kind_pointers.push_back(kind.c_str()); }
    // The descriptor names this carrier's strings from here on: the caller's may die.
    m_desc.name = m_name.c_str();
    m_desc.consumed_kinds = m_kind_pointers.empty() ? nullptr : m_kind_pointers.data();
}

StageCarrier::~StageCarrier() {
    // Exactly once: the carrier dies with the last pipeline or handler that shares it.
    if (m_desc.release != nullptr) { m_desc.release(m_desc.user_data); }
}

StageFacts stage_facts(const StageChain& chain) {
    StageFacts facts;
    for (const std::shared_ptr<StageCarrier>& stage : chain) {
        facts.m_fills_pre = facts.m_fills_pre || stage->desc().pre_process != nullptr;
        facts.m_fills_post = facts.m_fills_post || stage->desc().post_process != nullptr;
        if (!stage->consumed_kinds().empty()) {
            facts.m_consumers.push_back(ExtConsumer{.m_name = stage->name(),
                                                    .m_engine = ANIRA_ENGINE_NONE,
                                                    .m_consumed = stage->consumed_kinds()});
        }
    }
    return facts;
}

StageChainProcessor::StageChainProcessor(anira::InferenceConfig& config,
                                         const StageChain& chain,
                                         std::vector<Port>& input_ports,
                                         std::vector<Port>& output_ports)
    : anira::PrePostProcessor(config)
    , m_chain(chain)
    , m_input_ports(input_ports)
    , m_output_ports(output_ports)
    , m_input_shapes(config.get_tensor_input_shape())
    , m_output_shapes(config.get_tensor_output_shape()) {
    for (const std::shared_ptr<StageCarrier>& stage : m_chain) {
        if (stage->desc().pre_process != nullptr) {
            if (m_first_pre == nullptr) { m_first_pre = stage.get(); }
            m_last_pre = stage.get();
        }
        if (stage->desc().post_process != nullptr) {
            if (m_first_post == nullptr) { m_first_post = stage.get(); }
            m_last_post = stage.get();
        }
        m_fills_before = m_fills_before || stage->desc().before_inference != nullptr;
        m_fills_after = m_fills_after || stage->desc().after_inference != nullptr;
    }
}

void StageChainProcessor::bind(anira::SessionElement& session, std::vector<PlanPair> plans) {
    m_session = &session;
    m_plans = std::move(plans);
    m_chunks.clear();
    m_chunks.reserve(session.m_inference_queue.size());
    for (const auto& thread_safe_struct : session.m_inference_queue) {
        m_chunks.push_back(thread_safe_struct.get());
    }
    // A ring exists for a Streamed tensor only: its stream port names the session's ring from
    // here on, which is what the ring accessors of a ctx hand out.
    size_t input_channels = 0;
    for (size_t i = 0; i < m_input_ports.size() && i < session.m_send_buffer.size(); ++i) {
        auto* stream = std::get_if<StreamPort>(&m_input_ports[i]);
        if (stream == nullptr || m_inference_config.get_preprocess_input_size()[i] == 0) {
            continue;
        }
        stream->m_ring = &session.m_send_buffer[i];
        input_channels += stream->m_ring->num_channels();
    }
    size_t output_channels = 0;
    for (size_t i = 0; i < m_output_ports.size() && i < session.m_receive_buffer.size(); ++i) {
        auto* stream = std::get_if<StreamPort>(&m_output_ports[i]);
        if (stream == nullptr || m_inference_config.get_postprocess_output_size()[i] == 0) {
            continue;
        }
        stream->m_ring = &session.m_receive_buffer[i];
        output_channels += stream->m_ring->num_channels();
    }
    m_before.assign(std::max(input_channels, output_channels), 0);
}

StageChainProcessor::Chunk* StageChainProcessor::chunk_of_inputs(
    const std::vector<anira::BufferF>& inputs) noexcept ANIRA_NONBLOCKING {
    for (Chunk* chunk : m_chunks) {
        if (&chunk->m_tensor_input_data == &inputs) { return chunk; }
    }
    return nullptr;
}

StageChainProcessor::Chunk* StageChainProcessor::chunk_of_outputs(
    const std::vector<anira::BufferF>& outputs) noexcept ANIRA_NONBLOCKING {
    for (Chunk* chunk : m_chunks) {
        if (&chunk->m_tensor_output_data == &outputs) { return chunk; }
    }
    return nullptr;
}

StageFrame StageChainProcessor::make_frame() const noexcept ANIRA_NONBLOCKING {
    StageFrame frame;
    frame.m_input_ports = &m_input_ports;
    frame.m_output_ports = &m_output_ports;
    frame.m_input_shapes = &m_input_shapes;
    frame.m_output_shapes = &m_output_shapes;
    frame.m_rt = m_session->m_rt;
    return frame;
}

anira_stage_ctx StageChainProcessor::make_ctx(anira_stage_phase phase,
                                              const Chunk& chunk,
                                              const StageFrame& frame) const noexcept
    ANIRA_NONBLOCKING {
    // The plan the chunk was stamped with (Core::pre_process), never the session's atomic: all
    // four phases of a chunk report one plan.
    const uint32_t plan = chunk.m_plan;
    const PlanPair pair = plan < m_plans.size() ? m_plans[plan] : PlanPair{};
    anira_stage_ctx ctx{};  // every slot NULL, the high halves of the pointer slots zero
    ctx.phase = static_cast<uint32_t>(phase);
    ctx.engine = pair.m_engine;
    ctx.provider = pair.m_provider;
    ctx.variant = 0;
    ctx.num_inputs = static_cast<uint32_t>(m_input_shapes.size());
    ctx.num_outputs = static_cast<uint32_t>(m_output_shapes.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.reserved = 0;
    ctx.frame = &frame;
    return ctx;
}

void StageChainProcessor::fail([[maybe_unused]] const char* stage,
                               [[maybe_unused]] uint32_t phase,
                               anira_status status) noexcept ANIRA_NONBLOCKING {
    if (!m_session->m_rt->record_any(status)) { return; }
    ANIRA_LOG_RT_ERROR(anira::log_group::k_capi,
                       "stage '%s': %s returned %d (%s); the chunk delivers zeros",
                       stage,
                       phase_word(phase),
                       static_cast<int>(status),
                       anira_status_string(status));
}

anira_status StageChainProcessor::run_phase(const anira_stage_ctx& ctx,
                                            StageFrame& frame,
                                            bool on_driver) noexcept ANIRA_NONBLOCKING {
    for (const std::shared_ptr<StageCarrier>& stage : m_chain) {
        const anira_stage_fn callback = phase_slot(stage->desc(), ctx.phase);
        if (callback == nullptr) { continue; }
        // A refused accessor names the stage that made the call: a context accessor through
        // the frame of this call, a ring accessor through the session's ring owner. The rings
        // are reachable in the two driver-thread phases only, and the owner is one per session,
        // so the inference threads never write it.
        frame.m_stage = stage->name();
        if (on_driver) { m_session->m_ring_owner.m_stage = stage->name(); }
        const anira_status status = callback(&ctx, stage->desc().user_data);
        if (on_driver) { m_session->m_ring_owner.m_stage = nullptr; }
        frame.m_stage = nullptr;
        if (status != ANIRA_OK) {
            fail(stage->name(), ctx.phase, status);
            return status;  // later stages of this phase do not run
        }
    }
    return ANIRA_OK;
}

void StageChainProcessor::snapshot(const std::vector<Port>& ports) noexcept ANIRA_NONBLOCKING {
    size_t entry = 0;
    for (const Port& port : ports) {
        const auto* stream = std::get_if<StreamPort>(&port);
        if (stream == nullptr || stream->m_ring == nullptr) { continue; }
        const anira_ring* ring = stream->m_ring;
        const size_t channels = ring->num_channels();
        for (size_t channel = 0; channel < channels; ++channel) {
            m_before[entry++] = ring->available(channel);
        }
    }
}

void StageChainProcessor::report_hop([[maybe_unused]] uint32_t phase,
                                     [[maybe_unused]] size_t tensor,
                                     [[maybe_unused]] size_t channel,
                                     [[maybe_unused]] size_t expected,
                                     [[maybe_unused]] size_t moved) noexcept ANIRA_NONBLOCKING {
    if (!m_session->m_rt->record(ANIRA_ERROR_CONFIG)) { return; }
    const bool pre = phase == ANIRA_PHASE_PRE_PROCESS;
    [[maybe_unused]] const StageCarrier* first = pre ? m_first_pre : m_first_post;
    [[maybe_unused]] const StageCarrier* last = pre ? m_last_pre : m_last_post;
    // More than the hop cannot be undone; less was repaired (discarded from an input ring,
    // topped up with zeros on an output ring).
    [[maybe_unused]] const char* outcome =
        moved > expected ? "the stream has shifted"
                         : (pre ? "the rest was discarded" : "the rest is zeros");
    ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                           "stage chain: %s of stage%s '%s'%s%s%s: %s tensor %zu, channel %zu "
                           "moved %zu elements, the hop is %zu; %s",
                           phase_word(phase),
                           first != last ? "s" : "",
                           first != nullptr ? first->name() : "(default)",
                           first != last ? " to '" : "",
                           first != last ? last->name() : "",
                           first != last ? "'" : "",
                           pre ? "input" : "output",
                           tensor,
                           channel,
                           moved,
                           expected,
                           outcome);
}

void StageChainProcessor::check_input_hops(bool report) noexcept ANIRA_NONBLOCKING {
    size_t entry = 0;
    for (size_t tensor = 0; tensor < m_input_ports.size(); ++tensor) {
        const auto* stream = std::get_if<StreamPort>(&m_input_ports[tensor]);
        if (stream == nullptr || stream->m_ring == nullptr) { continue; }
        anira_ring* ring = stream->m_ring;
        const size_t channels = ring->num_channels();
        for (size_t channel = 0; channel < channels; ++channel) {
            const size_t before = m_before[entry++];
            const size_t after = ring->available(channel);
            const size_t moved = before > after ? before - after : 0;
            // A ring that held less than one hop gives up what it held (the pop pads).
            const size_t expected = std::min(ring->hop(), before);
            if (moved == expected) { continue; }
            if (moved < expected) { ring->discard(channel, expected - moved); }
            if (report) {
                report_hop(ANIRA_PHASE_PRE_PROCESS, tensor, channel, expected, moved);
                report = false;  // one record per chunk
            }
        }
    }
}

void StageChainProcessor::check_output_hops(bool report) noexcept ANIRA_NONBLOCKING {
    size_t entry = 0;
    for (size_t tensor = 0; tensor < m_output_ports.size(); ++tensor) {
        const auto* stream = std::get_if<StreamPort>(&m_output_ports[tensor]);
        if (stream == nullptr || stream->m_ring == nullptr) { continue; }
        anira_ring* ring = stream->m_ring;
        const size_t channels = ring->num_channels();
        for (size_t channel = 0; channel < channels; ++channel) {
            const size_t before = m_before[entry++];
            const size_t after = ring->available(channel);
            const size_t moved = after > before ? after - before : 0;
            // A full ring overwrites its oldest elements: it cannot gain more than its room.
            const size_t expected = std::min(ring->hop(), ring->capacity() - before);
            if (moved == expected) { continue; }
            if (moved < expected) { ring->push_zeros(channel, expected - moved); }
            if (report) {
                report_hop(ANIRA_PHASE_POST_PROCESS, tensor, channel, expected, moved);
                report = false;  // one record per chunk
            }
        }
    }
}

void StageChainProcessor::pre_process(std::vector<anira::RingBuffer>& input,
                                      std::vector<anira::BufferF>& output,
                                      anira::InferenceBackend current_inference_backend) {
    Chunk* chunk = chunk_of_inputs(output);
    if (chunk == nullptr) {
        // Not a struct of the bound session: the 2.x default, which needs no ctx.
        anira::PrePostProcessor::pre_process(input, output, current_inference_backend);
        return;
    }
    // Every Static tensor is materialised ahead of any stage: the whole tensor out
    // of its static port, under the slot's latch, into the struct's packed buffer. A
    // State tensor has no ring and no value either: the session feeds it on the inference
    // thread, and nothing on this thread touches it.
    for (size_t tensor = 0; tensor < output.size() && tensor < m_input_ports.size(); ++tensor) {
        const auto* fixed = std::get_if<StaticPort>(&m_input_ports[tensor]);
        if (fixed == nullptr) { continue; }
        fixed->m_value.read_packed(
            output[tensor].get_write_pointer(0),
            m_inference_config.get_tensor_input_size()[tensor] * sizeof(float));
    }
    StageFrame frame = make_frame();
    frame.m_model_inputs = &output;
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_PRE_PROCESS, *chunk, frame);

    snapshot(m_input_ports);
    anira_status status = ANIRA_OK;
    if (m_first_pre != nullptr) {
        status = run_phase(ctx, frame, /*on_driver=*/true);
    } else {
        // No stage fills the phase: the default body, once per chunk.
        status = anira_stage_default_pre_process(&ctx);
        if (status != ANIRA_OK) { fail("(default)", ctx.phase, status); }
    }
    // Every Streamed input ring has given up exactly its hop, whatever the stages did: a
    // shortfall is discarded so that the stream stays aligned. A failed phase is already
    // recorded, so its repair is silent.
    check_input_hops(/*report=*/status == ANIRA_OK);
    // Core::pre_process reads it: a failed chunk completes as zeros and is not enqueued.
    chunk->m_stage_status = status;
}

void StageChainProcessor::post_process(std::vector<anira::BufferF>& input,
                                       std::vector<anira::RingBuffer>& output,
                                       anira::InferenceBackend current_inference_backend) {
    const Chunk* chunk = chunk_of_outputs(input);
    if (chunk == nullptr) {
        anira::PrePostProcessor::post_process(input, output, current_inference_backend);
        return;
    }
    StageFrame frame = make_frame();
    frame.m_model_outputs = &input;
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_POST_PROCESS, *chunk, frame);

    snapshot(m_output_ports);
    anira_status status = ANIRA_OK;
    if (m_first_post != nullptr) {
        status = run_phase(ctx, frame, /*on_driver=*/true);
    } else {
        status = anira_stage_default_post_process(&ctx);
        if (status != ANIRA_OK) { fail("(default)", ctx.phase, status); }
    }
    // Every Streamed output ring has gained exactly its hop: a ring cannot take a push back,
    // so a failed or short phase is topped up with zeros and the stream stays aligned.
    check_output_hops(/*report=*/status == ANIRA_OK);

    // Every Static tensor is captured behind the last stage, the whole tensor into
    // its static port under the slot's latch (a State tensor has no port value: the session
    // captured it on the inference thread). Not from a chunk that completed as zeros (dropped,
    // failed in a stage or in the engine) and not behind a failed post_process: the port holds
    // what the model produced.
    if (status != ANIRA_OK || chunk->m_completed_as_zeros) { return; }
    for (size_t tensor = 0; tensor < input.size() && tensor < m_output_ports.size(); ++tensor) {
        auto* fixed = std::get_if<StaticPort>(&m_output_ports[tensor]);
        if (fixed == nullptr) { continue; }
        fixed->m_value.write_packed(
            input[tensor].get_read_pointer(0),
            m_inference_config.get_tensor_output_size()[tensor] * sizeof(float));
    }
}

void StageChainProcessor::before_inference(
    std::vector<anira::BufferF>& input,
    [[maybe_unused]] anira::InferenceBackend current_inference_backend) {
    if (!m_fills_before) { return; }
    Chunk* chunk = chunk_of_inputs(input);
    if (chunk == nullptr) { return; }
    StageFrame frame = make_frame();
    frame.m_model_inputs = &input;
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_BEFORE_INFERENCE, *chunk, frame);
    // InferenceThread::do_inference reads it: a failure skips the engine call and
    // after_inference, and the chunk delivers zeros.
    chunk->m_stage_status = run_phase(ctx, frame, /*on_driver=*/false);
}

void StageChainProcessor::after_inference(
    std::vector<anira::BufferF>& output,
    [[maybe_unused]] anira::InferenceBackend current_inference_backend) {
    if (!m_fills_after) { return; }
    Chunk* chunk = chunk_of_outputs(output);
    if (chunk == nullptr) { return; }
    StageFrame frame = make_frame();
    frame.m_model_outputs = &output;
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_AFTER_INFERENCE, *chunk, frame);
    // do_inference zeroes the outputs of a chunk whose after_inference failed.
    chunk->m_stage_status = run_phase(ctx, frame, /*on_driver=*/false);
}

}  // namespace anira::capi

// NOLINTEND(bugprone-exception-escape)
