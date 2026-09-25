// anira/abi/stage.h: the ring accessors, the context accessors, the two default bodies, and the
// stage processor behind them (the carrier of the one descriptor a pipeline holds and the one
// PrePostProcessor a C-created handler's session sees; the phase order is in stage.h).
//
// Everything a phase callback can reach is ANIRA_NONBLOCKING: the ring accessors are thin
// dtype-checked calls onto anira_ring (anira/utils/RingBuffer.h), a refusal records into the
// latch of the session that owns the ring (anira_ring::owner) and logs one record per kind;
// the context accessors answer per slot out of the StageFrame the
// processor fills on the stack next to its anira_stage_ctx, and record every status but
// ANIRA_OK into the frame's latch (asking for a ring or a model tensor the slot does not have
// in this phase is the stage's bug, not an answer). The two default bodies read the frame's
// ports directly and ask the context nothing they could be refused, so a stage-less pipeline
// records nothing. No handler, no lock, no allocation on any per-call path, whatever the
// stage's own flags promise: anira's side is real-time regardless.
#include "stage.h"

#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/lifecycle.h>
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
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#include "ext_registry.h"
#include "port.h"
#include "validate.h"
#include "words.h"

// The members of anira_ring dispatch through std::visit, which is declared to throw
// std::bad_variant_access for a valueless variant. A ring never is one: its arm is emplaced by
// initialize_with_positions() on the control thread, where a failed construction propagates,
// and nothing on a per-call path replaces the arm. The check sees the inline visit from every
// noexcept function below, so it is off for this file; nothing here throws on its own.
// NOLINTBEGIN(bugprone-exception-escape)

namespace {

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
                               "%s: the stage: channel %u is out of range, the ring has %zu",
                               entry,
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
                                   "%s: the stage: NULL memory for %zu elements",
                                   entry,
                                   count);
        }
        return false;
    }
    if (dtype == ring->dtype()) { return true; }
    if (ring_record(*ring, ANIRA_ERROR_CONFIG)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: the stage: dtype 0x%x is not the ring's 0x%x; nothing "
                               "converts",
                               entry,
                               static_cast<unsigned int>(dtype),
                               static_cast<unsigned int>(ring->dtype()));
    }
    return false;
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

// The ring of a slot as the default bodies read it: its stream port's, or NULL for a slot at or
// beyond the side's count, a port of another role and a stream port without a ring. A plain
// read, never a refusal.
anira_ring* slot_ring(const std::vector<Port>& ports, size_t slot) noexcept ANIRA_NONBLOCKING {
    const anira::capi::StreamPort* stream = anira::capi::stream_port(ports, slot);
    return stream != nullptr ? stream->m_ring : nullptr;
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
                               "%s: the stage: slot %u is out of range, the model has %zu %s "
                               "tensors",
                               entry,
                               static_cast<unsigned int>(slot),
                               ports->size(),
                               input ? "input" : "output");
    }
    return nullptr;
}

// The prologue of every context accessor, in this order: a NULL ctx and a ctx without a frame
// are refused with no latch to record into; a NULL out is recorded; then the slot, through
// ctx_port. The port of the slot, or NULL when the call is refused with
// ANIRA_ERROR_INVALID_ARGUMENT (the caller has reset its out-parameter already).
const Port* ctx_slot(const anira_stage_ctx* ctx,
                     bool input,
                     uint32_t slot,
                     bool null_out,
                     const char* entry) noexcept ANIRA_NONBLOCKING {
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr) { return nullptr; }
    if (null_out) {
        if (frame_record(*frame, ANIRA_ERROR_INVALID_ARGUMENT)) {
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                                   "%s: the stage: NULL out for slot %u",
                                   entry,
                                   static_cast<unsigned int>(slot));
        }
        return nullptr;
    }
    return ctx_port(ctx, input, slot, entry);
}

// A representation the slot does not have in this phase: asking for it is the stage's bug and
// not an answer, so the refusal is recorded into the frame's latch like a slot out of range,
// one record per kind naming the entry, the slot and the phase. The ctx passed the prologue,
// so it has a frame.
anira_status no_such(const anira_stage_ctx* ctx,
                     [[maybe_unused]] uint32_t slot,
                     [[maybe_unused]] const char* what,
                     [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const StageFrame& frame = *frame_of(ctx);
    if (frame_record(frame, ANIRA_ERROR_INVALID_STATE)) {
        ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                               "%s: the stage: slot %u has no %s in %s",
                               entry,
                               static_cast<unsigned int>(slot),
                               what,
                               anira::capi::phase_word(static_cast<anira_phase>(ctx->phase)));
    }
    return ANIRA_ERROR_INVALID_STATE;
}

// The role of a slot: its port's arm. A slot always has one, so the only refusals are the
// prologue's.
anira_status ctx_role(const anira_stage_ctx* ctx,
                      bool input,
                      uint32_t slot,
                      anira_role* out,
                      const char* entry) noexcept ANIRA_NONBLOCKING {
    if (out != nullptr) { *out = ANIRA_ROLE_FORCE32; }
    const Port* port = ctx_slot(ctx, input, slot, out == nullptr, entry);
    if (port == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    *out = anira::capi::port_role(*port);
    return ANIRA_OK;
}

// The ring of a slot: its stream port's, in the one phase that moves the rings of its side.
// A port of another role, and every other phase, has no ring to hand out.
anira_status ctx_ring(const anira_stage_ctx* ctx,
                      bool input,
                      uint32_t slot,
                      anira_ring** out,
                      const char* entry) noexcept ANIRA_NONBLOCKING {
    if (out != nullptr) { *out = nullptr; }
    const Port* port = ctx_slot(ctx, input, slot, out == nullptr, entry);
    if (port == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto ring_phase =
        static_cast<uint32_t>(input ? ANIRA_PHASE_PRE_PROCESS : ANIRA_PHASE_POST_PROCESS);
    const auto* stream = std::get_if<anira::capi::StreamPort>(port);
    anira_ring* ring = ctx->phase == ring_phase && stream != nullptr ? stream->m_ring : nullptr;
    if (ring == nullptr) { return no_such(ctx, slot, "ring", entry); }
    *out = ring;
    return ANIRA_OK;
}

// The model end of a slot: a copy of the chunk's descriptor of it (216 bytes, the record the
// engine call hands its adapter: the spec's shape, the storage's dtype, the struct's memory of
// the moment, which the scheduler re-points at every claim). Nothing of it survives a call.
// The frame names the descriptors of a side only in the phases that expose it, and a State slot
// has no host end (it is fed behind pre_process and captured ahead of post_process), so the two
// host-end phases have no model end of it to hand out.
anira_status ctx_tensor(const anira_stage_ctx* ctx,
                        bool input,
                        uint32_t slot,
                        anira_tensor* out,
                        const char* entry) noexcept ANIRA_NONBLOCKING {
    // Every refusal leaves the all-zero record of a refused anira_tensor_init_* factory.
    if (out != nullptr) { std::memset(out, 0, sizeof(*out)); }
    const Port* port = ctx_slot(ctx, input, slot, out == nullptr, entry);
    if (port == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const StageFrame& frame = *frame_of(ctx);
    const bool host_end_phase =
        ctx->phase == ANIRA_PHASE_PRE_PROCESS || ctx->phase == ANIRA_PHASE_POST_PROCESS;
    const std::vector<anira_tensor>* tensors =
        input ? frame.m_input_tensors : frame.m_output_tensors;
    if ((host_end_phase && std::get_if<anira::capi::StatePort>(port) != nullptr) ||
        tensors == nullptr || slot >= tensors->size()) {
        return no_such(ctx, slot, "model tensor", entry);
    }
    *out = (*tensors)[slot];
    return ANIRA_OK;
}

// The prologue of the two pair accessors: whether a ctx may be answered. A NULL ctx and a ctx
// without a frame are refused with no latch to record into; a NULL value out-parameter is
// recorded. The caller has reset its out-parameters already.
bool pair_answerable(const anira_stage_ctx* ctx,
                     bool null_value,
                     [[maybe_unused]] const char* entry) noexcept ANIRA_NONBLOCKING {
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr) { return false; }
    if (null_value) {
        if (frame_record(*frame, ANIRA_ERROR_INVALID_ARGUMENT)) {
            ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi, "%s: the stage: NULL out", entry);
        }
        return false;
    }
    return true;
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
// A stage asks per slot; nothing is laid out for it up front. Every status but ANIRA_OK is
// recorded into the frame's latch, the session's, with one record per kind naming the entry,
// the slot and the phase: a slot out of range and a NULL out
// (INVALID_ARGUMENT), a ring or a model tensor the slot does not have in this phase
// (INVALID_STATE). A NULL ctx and a ctx without a frame name no latch.

anira_status ANIRA_CALL anira_stage_input_role(const anira_stage_ctx* ctx,
                                               uint32_t slot,
                                               anira_role* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_role(ctx, /*input=*/true, slot, out, __func__);
}

anira_status ANIRA_CALL anira_stage_output_role(const anira_stage_ctx* ctx,
                                                uint32_t slot,
                                                anira_role* out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_role(ctx, /*input=*/false, slot, out, __func__);
}

anira_status ANIRA_CALL anira_stage_input_ring(const anira_stage_ctx* ctx,
                                               uint32_t slot,
                                               anira_ring** out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_ring(ctx, /*input=*/true, slot, out, __func__);
}

anira_status ANIRA_CALL anira_stage_output_ring(const anira_stage_ctx* ctx,
                                                uint32_t slot,
                                                anira_ring** out) ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    return ctx_ring(ctx, /*input=*/false, slot, out, __func__);
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

// The pairs of the chunk's plan: the record's value field and its id slot, which make_ctx
// filled from the session's table (the pair rule: the id is set beside CUSTOM alone). The id
// out-parameter may be NULL; both are reset on a refusal.
anira_status ANIRA_CALL anira_stage_engine(const anira_stage_ctx* ctx,
                                           anira_engine* engine,
                                           const char** engine_id)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (engine != nullptr) { *engine = ANIRA_ENGINE_FORCE32; }
    if (engine_id != nullptr) { *engine_id = nullptr; }
    if (!pair_answerable(ctx, engine == nullptr, __func__)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    *engine = static_cast<anira_engine>(ctx->engine);
    if (engine_id != nullptr) { *engine_id = ctx->engine_id; }
    return ANIRA_OK;
}

anira_status ANIRA_CALL anira_stage_provider(const anira_stage_ctx* ctx,
                                             anira_provider* provider,
                                             const char** provider_id)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (provider != nullptr) { *provider = ANIRA_PROVIDER_FORCE32; }
    if (provider_id != nullptr) { *provider_id = nullptr; }
    if (!pair_answerable(ctx, provider == nullptr, __func__)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    *provider = static_cast<anira_provider>(ctx->provider);
    if (provider_id != nullptr) { *provider_id = ctx->provider_id; }
    return ANIRA_OK;
}

// ==== the default bodies ======================================================================
// The 2.x default processor's streamed branches (PrePostProcessor.cpp) over anira_ring* and
// anira_tensor: the ring of a slot, then its model end through the tensor accessor. The ring is
// read off the frame's stream port directly, never asked of the context: a slot without a ring
// here (another role, a stream port the session gave no ring) is skipped, so a default body is
// never refused and records nothing, whatever the pipeline's slots are. The tensors are read as
// packed row-major memory, which is how the chain fills them. The status is returned and never
// recorded here: a stage that calls the default for the slots it does not handle itself may
// drop it, and the chain records what a stage returns.

anira_status ANIRA_CALL anira_stage_default_pre_process(const anira_stage_ctx* ctx)
    ANIRA_NOEXCEPT ANIRA_NONBLOCKING {
    if (ctx == nullptr || ctx->phase != ANIRA_PHASE_PRE_PROCESS) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->num_inputs == 0) { return ANIRA_OK; }
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr || frame->m_input_ports == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = ANIRA_OK;
    for (uint32_t slot = 0; slot < ctx->num_inputs; ++slot) {
        anira_ring* ring = slot_ring(*frame->m_input_ports, slot);
        if (ring == nullptr) { continue; }  // not a Streamed tensor, or no ring here
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
    const StageFrame* frame = frame_of(ctx);
    if (frame == nullptr || frame->m_output_ports == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    anira_status status = ANIRA_OK;
    for (uint32_t slot = 0; slot < ctx->num_outputs; ++slot) {
        anira_ring* ring = slot_ring(*frame->m_output_ports, slot);
        if (ring == nullptr) { continue; }  // not a Streamed tensor, or no ring here
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

// ==== the carrier and the processor ===========================================================

namespace anira::capi {

StageCarrier::StageCarrier(const anira_stage_desc& desc) : m_desc(desc) {
    m_kinds.reserve(desc.num_consumed_kinds);
    for (uint32_t i = 0; i < desc.num_consumed_kinds; ++i) {
        m_kinds.emplace_back(desc.consumed_kinds[i]);
    }
    m_kind_pointers.reserve(m_kinds.size());
    for (const std::string& kind : m_kinds) { m_kind_pointers.push_back(kind.c_str()); }
    // The descriptor names this carrier's strings from here on: the caller's may die.
    m_desc.consumed_kinds = m_kind_pointers.empty() ? nullptr : m_kind_pointers.data();
}

StageCarrier::~StageCarrier() {
    // Exactly once: the carrier dies with the last pipeline or handler that shares it, whether
    // or not init ever ran.
    if (m_desc.release != nullptr) { m_desc.release(m_desc.user_data); }
}

anira_status StageCarrier::ensure_init(const anira_init_info& info) const {
    const std::scoped_lock<std::mutex> lock(m_init_mutex);
    if (m_initialised) { return ANIRA_OK; }
    if (m_desc.init != nullptr) {
        const anira_status status = m_desc.init(&info, m_desc.user_data);
        if (status != ANIRA_OK) { return status; }
    }
    m_initialised = true;
    return ANIRA_OK;
}

StageFacts stage_facts(const StageCarrier* stage) {
    StageFacts facts;
    if (stage == nullptr) { return facts; }
    facts.m_fills_pre = stage->desc().pre_process != nullptr;
    facts.m_fills_post = stage->desc().post_process != nullptr;
    if (!stage->consumed_kinds().empty()) {
        facts.m_consumers.push_back(ExtConsumer{.m_name = StageCarrier::k_stage_consumer,
                                                .m_engine = ANIRA_ENGINE_NONE,
                                                .m_consumed = stage->consumed_kinds()});
    }
    return facts;
}

StageProcessor::StageProcessor(anira::InferenceConfig& config,
                               const StageCarrier* stage,
                               std::vector<Port>& input_ports,
                               std::vector<Port>& output_ports)
    : anira::PrePostProcessor(config)
    , m_stage(stage)
    , m_input_ports(input_ports)
    , m_output_ports(output_ports)
    , m_input_shapes(config.get_tensor_input_shape())
    , m_output_shapes(config.get_tensor_output_shape()) {
    m_has_state = std::ranges::any_of(m_input_ports, [](const Port& port) {
        return std::holds_alternative<StatePort>(port);
    });
    if (m_stage == nullptr) { return; }
    const anira_stage_desc& desc = m_stage->desc();
    m_fills_pre = desc.pre_process != nullptr;
    m_fills_post = desc.post_process != nullptr;
    m_fills_before = desc.before_inference != nullptr;
    m_fills_after = desc.after_inference != nullptr;
}

void StageProcessor::bind(anira::SessionElement& session) {
    m_session = &session;
    // The struct table, by entry: the position of a struct in the session's pool is what a
    // ctx reports as the chunk's entry.
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

size_t StageProcessor::entry_of_inputs(const std::vector<anira::BufferF>& inputs) const noexcept
    ANIRA_NONBLOCKING {
    for (size_t entry = 0; entry < m_chunks.size(); ++entry) {
        if (&m_chunks[entry]->m_tensor_input_data == &inputs) { return entry; }
    }
    return k_no_entry;
}

size_t StageProcessor::entry_of_outputs(const std::vector<anira::BufferF>& outputs) const noexcept
    ANIRA_NONBLOCKING {
    for (size_t entry = 0; entry < m_chunks.size(); ++entry) {
        if (&m_chunks[entry]->m_tensor_output_data == &outputs) { return entry; }
    }
    return k_no_entry;
}

StageFrame StageProcessor::make_frame() const noexcept ANIRA_NONBLOCKING {
    StageFrame frame;
    frame.m_input_ports = &m_input_ports;
    frame.m_output_ports = &m_output_ports;
    frame.m_rt = m_session->m_rt;
    return frame;
}

StageFrame StageProcessor::make_frame_with_inputs(const Chunk& chunk) const noexcept
    ANIRA_NONBLOCKING {
    StageFrame frame = make_frame();
    frame.m_input_tensors = &chunk.m_input_tensors;
    return frame;
}

StageFrame StageProcessor::make_frame_with_outputs(const Chunk& chunk) const noexcept
    ANIRA_NONBLOCKING {
    StageFrame frame = make_frame();
    frame.m_output_tensors = &chunk.m_output_tensors;
    return frame;
}

anira_stage_ctx StageProcessor::make_ctx(anira_phase phase,
                                         size_t entry,
                                         const StageFrame& frame) const noexcept ANIRA_NONBLOCKING {
    // The plan the chunk was stamped with (Core::pre_process), never the session's atomic: all
    // four phases of a chunk report one plan, and one entry. The session's table names the
    // plan's engine and provider; a stamp is always in range.
    const uint32_t plan = m_chunks[entry]->m_plan;
    const std::vector<anira::SessionElement::PlanSlot>& plans = m_session->m_plans;
    anira_stage_ctx ctx{};  // every slot NULL, the high halves of the pointer slots zero
    ctx.phase = static_cast<uint32_t>(phase);
    ctx.engine = plan < plans.size() ? static_cast<uint32_t>(plans[plan].m_engine)
                                     : static_cast<uint32_t>(ANIRA_ENGINE_NONE);
    ctx.provider = plan < plans.size() ? static_cast<uint32_t>(plans[plan].m_provider)
                                       : static_cast<uint32_t>(ANIRA_PROVIDER_DEFAULT);
    // The pairs' names for anira_stage_engine / _provider: the session's own strings,
    // which live as long as its table; NULL for a built-in engine and a provider of the enum.
    if (plan < plans.size()) {
        const std::string& engine_id = plans[plan].m_engine_id;
        const std::string& provider_id = plans[plan].m_provider_id;
        ctx.engine_id = engine_id.empty() ? nullptr : engine_id.c_str();
        ctx.provider_id = provider_id.empty() ? nullptr : provider_id.c_str();
    }
    ctx.variant = 0;
    ctx.num_inputs = static_cast<uint32_t>(m_input_shapes.size());
    ctx.num_outputs = static_cast<uint32_t>(m_output_shapes.size());
    ctx.ticket = ANIRA_TICKET_INVALID;
    ctx.entry = static_cast<uint32_t>(entry);
    ctx.frame = &frame;
    return ctx;
}

void StageProcessor::fail([[maybe_unused]] const char* who,
                          [[maybe_unused]] uint32_t phase,
                          anira_status status) noexcept ANIRA_NONBLOCKING {
    if (!m_session->m_rt->record_any(status)) { return; }
    ANIRA_LOG_RT_ERROR(anira::log_group::k_capi,
                       "%s: %s returned %d (%s); the chunk delivers zeros",
                       who,
                       phase_word(static_cast<anira_phase>(phase)),
                       static_cast<int>(status),
                       anira_status_string(status));
}

anira_status StageProcessor::run_phase(const anira_stage_ctx& ctx) noexcept ANIRA_NONBLOCKING {
    const anira_stage_fn callback = phase_slot(m_stage->desc(), ctx.phase);
    const anira_status status = callback(&ctx, m_prepared, m_stage->desc().user_data);
    if (status != ANIRA_OK) { fail(k_the_stage, ctx.phase, status); }
    return status;
}

void StageProcessor::reset_if_new_stream(const Chunk& chunk,
                                         size_t entry) noexcept ANIRA_NONBLOCKING {
    if (chunk.m_dispatch_generation == m_pre_generation) { return; }
    // The first chunk of a new stream. The stamp is adopted whether the stage resets or not,
    // so that the next chunk of this stream is no first one; the chunk's stamp, never the
    // session's atomic (a reset that lands between Core::pre_process's stamp and this read
    // belongs to the next chunk).
    m_pre_generation = chunk.m_dispatch_generation;
    if (m_stage == nullptr || m_stage->desc().reset == nullptr) { return; }
    // The chunk's context in ANIRA_PHASE_RESET over a frame without buffers: the roles answer,
    // a ring or a model tensor is refused and recorded, the stage's bug.
    const StageFrame frame = make_frame();
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_RESET, entry, frame);
    m_stage->desc().reset(&ctx, m_prepared, m_stage->desc().user_data);
}

void StageProcessor::snapshot(const std::vector<Port>& ports) noexcept ANIRA_NONBLOCKING {
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

const char* StageProcessor::mover(uint32_t phase) const noexcept ANIRA_NONBLOCKING {
    const bool filled = phase == ANIRA_PHASE_PRE_PROCESS ? m_fills_pre : m_fills_post;
    return filled ? k_the_stage : k_the_default;
}

void StageProcessor::report_hop([[maybe_unused]] uint32_t phase,
                                [[maybe_unused]] size_t tensor,
                                [[maybe_unused]] size_t channel,
                                [[maybe_unused]] size_t expected,
                                [[maybe_unused]] size_t moved) noexcept ANIRA_NONBLOCKING {
    if (!m_session->m_rt->record(ANIRA_ERROR_CONFIG)) { return; }
    // More than the hop cannot be undone; less was repaired (discarded from an input ring,
    // topped up with zeros on an output ring).
    [[maybe_unused]] const bool pre = phase == ANIRA_PHASE_PRE_PROCESS;
    [[maybe_unused]] const char* outcome =
        moved > expected ? "the stream has shifted"
                         : (pre ? "the rest was discarded" : "the rest is zeros");
    ANIRA_LOG_RT_VIOLATION(anira::log_group::k_capi,
                           "%s: %s moved %s tensor %zu, channel %zu by %zu elements, the hop "
                           "is %zu; %s",
                           mover(phase),
                           phase_word(static_cast<anira_phase>(phase)),
                           pre ? "input" : "output",
                           tensor,
                           channel,
                           moved,
                           expected,
                           outcome);
}

void StageProcessor::check_input_hops(bool report) noexcept ANIRA_NONBLOCKING {
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

void StageProcessor::check_output_hops(bool report) noexcept ANIRA_NONBLOCKING {
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

void StageProcessor::pre_process(std::vector<anira::RingBuffer>& input,
                                 std::vector<anira::BufferF>& output,
                                 anira::InferenceBackend current_inference_backend) {
    const size_t entry = entry_of_inputs(output);
    if (entry == k_no_entry) {
        // Not a struct of the bound session: the 2.x default, which needs no ctx.
        anira::PrePostProcessor::pre_process(input, output, current_inference_backend);
        return;
    }
    Chunk& chunk = *m_chunks[entry];
    // The stage's reset boundary: the first chunk of a new stream resets the stage's own state
    // before anything of the chunk is formed.
    reset_if_new_stream(chunk, entry);
    // Every Static tensor is materialised ahead of the stage: the whole tensor out of its
    // static port, under the slot's latch, into the struct's packed buffer. A State tensor has
    // no ring and is not fed here: before_inference feeds it on the inference thread, and
    // nothing on this thread touches it.
    for (size_t tensor = 0; tensor < output.size() && tensor < m_input_ports.size(); ++tensor) {
        const auto* fixed = std::get_if<StaticPort>(&m_input_ports[tensor]);
        if (fixed == nullptr) { continue; }
        fixed->m_value.read_packed(
            output[tensor].get_write_pointer(0),
            m_inference_config.get_tensor_input_size()[tensor] * sizeof(float));
    }
    const StageFrame frame = make_frame_with_inputs(chunk);
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_PRE_PROCESS, entry, frame);

    snapshot(m_input_ports);
    anira_status status = ANIRA_OK;
    if (m_fills_pre) {
        status = run_phase(ctx);
    } else {
        // The slot is NULL (or there is no stage): the default body, once per chunk.
        status = anira_stage_default_pre_process(&ctx);
        if (status != ANIRA_OK) { fail(k_the_default, ctx.phase, status); }
    }
    // Every Streamed input ring has given up exactly its hop, whatever the stage did: a
    // shortfall is discarded so that the stream stays aligned. A failed phase is already
    // recorded, so its repair is silent.
    check_input_hops(/*report=*/status == ANIRA_OK);
    // Core::pre_process reads it: a failed chunk completes as zeros and is not enqueued.
    chunk.m_stage_status = status;
}

void StageProcessor::post_process(std::vector<anira::BufferF>& input,
                                  std::vector<anira::RingBuffer>& output,
                                  anira::InferenceBackend current_inference_backend) {
    const size_t entry = entry_of_outputs(input);
    if (entry == k_no_entry) {
        anira::PrePostProcessor::post_process(input, output, current_inference_backend);
        return;
    }
    const Chunk& chunk = *m_chunks[entry];
    const StageFrame frame = make_frame_with_outputs(chunk);
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_POST_PROCESS, entry, frame);

    snapshot(m_output_ports);
    anira_status status = ANIRA_OK;
    if (m_fills_post) {
        status = run_phase(ctx);
    } else {
        status = anira_stage_default_post_process(&ctx);
        if (status != ANIRA_OK) { fail(k_the_default, ctx.phase, status); }
    }
    // Every Streamed output ring has gained exactly its hop: a ring cannot take a push back,
    // so a failed or short phase is topped up with zeros and the stream stays aligned.
    check_output_hops(/*report=*/status == ANIRA_OK);

    // Every Static tensor is captured behind the stage, the whole tensor into its static port
    // under the slot's latch (a State tensor was captured in after_inference, on the inference
    // thread). Not from a chunk that completed as zeros (dropped, failed in a stage or in the
    // engine) and not behind a failed post_process: the port holds what the model produced.
    if (status != ANIRA_OK || chunk.m_completed_as_zeros) { return; }
    for (size_t tensor = 0; tensor < input.size() && tensor < m_output_ports.size(); ++tensor) {
        auto* fixed = std::get_if<StaticPort>(&m_output_ports[tensor]);
        if (fixed == nullptr) { continue; }
        fixed->m_value.write_packed(
            input[tensor].get_read_pointer(0),
            m_inference_config.get_tensor_output_size()[tensor] * sizeof(float));
    }
}

bool StageProcessor::aliases_state(const Chunk& chunk) const noexcept ANIRA_NONBLOCKING {
    const std::vector<anira::SessionElement::PlanSlot>& plans = m_session->m_plans;
    return chunk.m_plan < plans.size() && plans[chunk.m_plan].m_state_alias;
}

void StageProcessor::bind_state(Chunk& chunk) noexcept ANIRA_NONBLOCKING {
    // A plan whose engine keeps its own aliasing sees one buffer as both halves.
    const bool alias = aliases_state(chunk);
    // The input half of every pair: the chunk's descriptor of the State input over the read
    // buffer, in the spec's dtype and shape (the buffer is packed row-major: all-zero strides)
    // and the pair's domain.
    for (size_t slot = 0; slot < chunk.m_input_tensors.size() && slot < m_input_ports.size();
         ++slot) {
        auto* state = std::get_if<StatePort>(&m_input_ports[slot]);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        StateSlot& value = *state->m_value;
        if (state->m_generation != chunk.m_dispatch_generation) {
            // The first inference of a new stream (a reset, a prepare): the state starts over.
            // The chunk's stamp, never the session's atomic: a reset that lands between a
            // chunk's stale check and its bind must not zero the read buffer for a chunk of
            // the old stream, whose promotion would then seed the new one.
            value.zero_read();
            state->m_generation = chunk.m_dispatch_generation;
        }
        anira_tensor_init_host(&chunk.m_input_tensors[slot],
                               value.read(),
                               value.dtype(),
                               static_cast<uint32_t>(value.shape().size()),
                               value.shape().data());
        chunk.m_input_tensors[slot].domain = static_cast<uint32_t>(value.domain());
    }
    // The output half: the descriptor of the State output over the write buffer of the input
    // it feeds (the port's partner names it), or over its read buffer under aliasing.
    for (size_t slot = 0; slot < chunk.m_output_tensors.size() && slot < m_output_ports.size();
         ++slot) {
        const auto* state = std::get_if<StatePort>(&m_output_ports[slot]);
        if (state == nullptr) { continue; }
        auto* input = state->m_partner < m_input_ports.size()
                          ? std::get_if<StatePort>(&m_input_ports[state->m_partner])
                          : nullptr;
        if (input == nullptr || !input->m_value.has_value()) { continue; }
        StateSlot& value = *input->m_value;
        anira_tensor_init_host(&chunk.m_output_tensors[slot],
                               alias ? value.read() : value.write(),
                               value.dtype(),
                               static_cast<uint32_t>(value.shape().size()),
                               value.shape().data());
        chunk.m_output_tensors[slot].domain = static_cast<uint32_t>(value.domain());
    }
}

void StageProcessor::promote_state(const Chunk& chunk) noexcept ANIRA_NONBLOCKING {
    // Under aliasing the engine updated the read buffer in place: it already holds the
    // produced state, and a flip would hand the next inference the stale other buffer.
    if (aliases_state(chunk)) { return; }
    for (Port& port : m_input_ports) {
        auto* state = std::get_if<StatePort>(&port);
        if (state == nullptr || !state->m_value.has_value()) { continue; }
        state->m_value->flip();
    }
}

void StageProcessor::before_inference(
    std::vector<anira::BufferF>& input,
    [[maybe_unused]] anira::InferenceBackend current_inference_backend) {
    if (!m_fills_before && !m_has_state) { return; }
    const size_t entry = entry_of_inputs(input);
    if (entry == k_no_entry) { return; }
    Chunk& chunk = *m_chunks[entry];
    // The declared state, the library's first step: every pair is bound ahead of the stage's
    // hook, which sees the read buffer as the fed state and may read or alter it (the class
    // comment).
    bind_state(chunk);
    if (!m_fills_before) { return; }
    const StageFrame frame = make_frame_with_inputs(chunk);
    const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_BEFORE_INFERENCE, entry, frame);
    // InferenceThread::do_inference reads it: a failure skips the engine call and
    // after_inference, and the chunk delivers zeros.
    chunk.m_stage_status = run_phase(ctx);
}

void StageProcessor::after_inference(
    std::vector<anira::BufferF>& output,
    [[maybe_unused]] anira::InferenceBackend current_inference_backend) {
    if (!m_fills_after && !m_has_state) { return; }
    const size_t entry = entry_of_outputs(output);
    if (entry == k_no_entry) { return; }
    Chunk& chunk = *m_chunks[entry];
    if (m_fills_after) {
        const StageFrame frame = make_frame_with_outputs(chunk);
        const anira_stage_ctx ctx = make_ctx(ANIRA_PHASE_AFTER_INFERENCE, entry, frame);
        // do_inference zeroes the outputs of a chunk whose after_inference failed.
        chunk.m_stage_status = run_phase(ctx);
    }
    // The declared state, the library's last step: every pair is promoted behind the stage's
    // hook, which saw the write buffer as the produced state. Not behind a failed hook (the
    // chunk delivers zeros; a failed before_inference and a failed engine never come here) and
    // not for a chunk that completed as zeros: the read buffer keeps the last good state.
    if (chunk.m_stage_status != ANIRA_OK || chunk.m_completed_as_zeros) { return; }
    promote_state(chunk);
}

}  // namespace anira::capi

// NOLINTEND(bugprone-exception-escape)
