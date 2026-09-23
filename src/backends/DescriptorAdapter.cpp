#include "DescriptorAdapter.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../capi/engine.h"
#include "../utils/StatusError.h"
#include "Adapter.h"

namespace anira::backend {

namespace {

// The engine-side template of every slot of one side: the spec's dtype and the engine's
// extents of the record (the entry's layout applied) at the pinned window, all-zero strides,
// ANIRA_DOMAIN_HOST, no memory. What process will be handed, minus the data.
std::vector<anira_tensor> templates_of(const std::vector<TensorInfo>& slots) {
    std::vector<anira_tensor> templates(slots.size());
    for (size_t slot = 0; slot < slots.size(); ++slot) {
        anira_tensor_init_host(&templates[slot],
                               nullptr,
                               slots[slot].m_dtype,
                               static_cast<uint32_t>(slots[slot].m_dims.size()),
                               slots[slot].m_dims.data());
    }
    return templates;
}

// The name each slot of one side binds to: the entry's tensors record where it names the
// slot, else the canonical name. Pointers into the record, which outlives the prepare call.
std::vector<const char*> names_of(const std::vector<TensorInfo>& slots) {
    std::vector<const char*> names;
    names.reserve(slots.size());
    for (const TensorInfo& slot : slots) {
        names.push_back(slot.m_engine_name.empty() ? slot.m_name.c_str()
                                                   : slot.m_engine_name.c_str());
    }
    return names;
}

}  // namespace

DescriptorAdapter::DescriptorAdapter(std::shared_ptr<const anira::capi::EngineCarrier> carrier,
                                     std::string id,
                                     uint32_t row,
                                     std::shared_ptr<const anira_model_config> model)
    : m_carrier(std::move(carrier)), m_id(std::move(id)), m_row(row), m_model(std::move(model)) {}

DescriptorAdapter::~DescriptorAdapter() {
    unprepare();
}

uint32_t DescriptorAdapter::flags() const noexcept {
    return m_carrier->desc().flags;
}

void DescriptorAdapter::unprepare() noexcept {
    if (!m_unprepare_owed) { return; }
    m_unprepare_owed = false;
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.unprepare != nullptr) { desc.unprepare(m_prepared, desc.user_data); }
    m_prepared = nullptr;
}

void DescriptorAdapter::do_prepare(const Model& model) {
    // A second prepare starts over: what the first one loaded goes back to the engine first.
    unprepare();
    m_prepared = nullptr;
    // The engine received the names and binds the slots itself.
    Bindings bindings;
    bindings.m_inputs.assign(model.m_inputs.size(), ANIRA_BINDING_ENGINE);
    bindings.m_outputs.assign(model.m_outputs.size(), ANIRA_BINDING_ENGINE);
    set_bindings(std::move(bindings));

    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.prepare == nullptr) { return; }  // nothing to prepare: prepared stays NULL

    const std::vector<anira_tensor> inputs = templates_of(model.m_inputs);
    const std::vector<anira_tensor> outputs = templates_of(model.m_outputs);
    const std::vector<const char*> input_names = names_of(model.m_inputs);
    const std::vector<const char*> output_names = names_of(model.m_outputs);
    anira_engine_prepare_info info = ANIRA_ENGINE_PREPARE_INFO_INIT;
    info.row = m_row;
    info.model = m_model.get();
    info.inputs = inputs.data();
    info.outputs = outputs.data();
    info.input_names = input_names.data();
    info.output_names = output_names.data();
    info.num_inputs = static_cast<uint32_t>(inputs.size());
    info.num_outputs = static_cast<uint32_t>(outputs.size());
    // A session-exclusive prepared model runs one inference at a time (the dispatch gate), so
    // one instance is all its process calls ever claim; a shared one takes the record's count.
    info.instances =
        model.m_session_exclusive ? 1U : (model.m_instances > 0 ? model.m_instances : 1U);

    void* prepared = nullptr;
    const anira_status status = desc.prepare(&info, desc.user_data, &prepared);
    if (status != ANIRA_OK) {
        // The refused prepare owes no unprepare.
        throw StatusError(status,
                          "the engine '" + m_id + "' refused prepare: it returned " +
                              std::to_string(static_cast<int>(status)) + " (" +
                              anira_status_string(status) + ")");
    }
    m_prepared = prepared;
    m_unprepare_owed = true;
}

anira_status DescriptorAdapter::process(const anira_engine_ctx& ctx,
                                        ChunkBuffers* /*chunk*/) noexcept {
    const anira_engine_desc& desc = m_carrier->desc();
    return desc.process(&ctx, m_prepared, desc.user_data);
}

void DescriptorAdapter::reset(const anira_engine_ctx& ctx) noexcept {
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.reset != nullptr) { desc.reset(&ctx, m_prepared, desc.user_data); }
}

}  // namespace anira::backend
