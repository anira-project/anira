#include "LegacyAdapter.h"

#include <anira/InferenceConfig.h>
#include <anira/abi/engine.h>
#include <anira/abi/status.h>
#include <anira/backends/BackendBase.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/Logger.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "Adapter.h"

namespace anira::engine {

namespace {

// The element count of one 2.x buffer: one packed block of channels x samples.
size_t elements_of(const anira::BufferF& buffer) noexcept {
    return buffer.get_num_channels() * buffer.get_num_samples();
}

}  // namespace

LegacyLoaded::LegacyLoaded(anira::BackendBase& backend) : m_backend(&backend) {}

LegacyLoaded::LegacyLoaded(anira::InferenceConfig& config)
    : m_owned(std::make_unique<anira::BackendBase>(config)), m_backend(m_owned.get()) {}

LegacyLoaded::~LegacyLoaded() = default;

void LegacyLoaded::do_load(const Model& model) {
    // A caller's backend and the roundtrip carry their own copy of the configuration; the
    // record's tensors are the struct's buffers they run over.
    static_cast<void>(model);
    m_backend->prepare();
}

std::unique_ptr<Prepared> LegacyLoaded::do_prepare(const PrepareRequest& request) {
    return std::make_unique<LegacyPrepared>(*this, request.m_exclusive);
}

anira_status LegacyPrepared::process(const anira_engine_ctx& call, ChunkBuffers* chunk) noexcept {
    if (m_backend == nullptr || chunk == nullptr || chunk->m_inputs == nullptr ||
        chunk->m_outputs == nullptr) {
        return ANIRA_ERROR_INVALID_STATE;
    }
    std::vector<anira::BufferF>& inputs = *chunk->m_inputs;
    std::vector<anira::BufferF>& outputs = *chunk->m_outputs;

    // A descriptor over other memory than the struct's buffer (a State half) is copied into
    // the buffer the 2.x virtual reads; a descriptor the adapter cannot read as the buffer's
    // packed block is anira's own bug, and the chunk fails rather than running on stale data.
    for (uint32_t slot = 0;
         call.inputs != nullptr && slot < call.num_inputs && slot < inputs.size();
         ++slot) {
        const size_t elements = elements_of(inputs[slot]);
        const float* data = host_f32_packed(call.inputs[slot], elements);
        if (data == nullptr) { return ANIRA_ERROR_INTERNAL; }
        if (data == inputs[slot].data()) { continue; }
        std::memcpy(inputs[slot].data(), data, elements * sizeof(float));
    }

    try {
        m_backend->process(inputs, outputs, nullptr);
    } catch (const std::exception& e) {
        // The 2.x virtual threw: the chunk fails with ENGINE, which the scheduler records and
        // logs once per latch; the text of the throw is logged here, unlatched, like the 2.x
        // processors' own records without a session.
        ANIRA_LOG_RT_ERROR(anira::log_group::k_scheduler,
                           "a 2.x backend threw out of process: %s; the chunk delivers zeros",
                           e.what());
        return ANIRA_ERROR_ENGINE;
    } catch (...) {
        ANIRA_LOG_RT_ERROR(anira::log_group::k_scheduler,
                           "a 2.x backend threw a non-std exception out of process; the chunk "
                           "delivers zeros");
        return ANIRA_ERROR_ENGINE;
    }

    // What the 2.x virtual wrote into a buffer whose descriptor names other memory goes there.
    for (uint32_t slot = 0;
         call.outputs != nullptr && slot < call.num_outputs && slot < outputs.size();
         ++slot) {
        const size_t elements = elements_of(outputs[slot]);
        float* data = host_f32_packed(call.outputs[slot], elements);
        if (data == nullptr) { return ANIRA_ERROR_INTERNAL; }
        if (data == outputs[slot].data()) { continue; }
        std::memcpy(data, outputs[slot].data(), elements * sizeof(float));
    }
    return ANIRA_OK;
}

}  // namespace anira::engine
