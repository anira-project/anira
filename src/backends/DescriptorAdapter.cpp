#include "DescriptorAdapter.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../capi/engine.h"
#include "../capi/words.h"
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
// slot, else the canonical name. Pointers into the record, which outlives the load call.
std::vector<const char*> names_of(const std::vector<TensorInfo>& slots) {
    std::vector<const char*> names;
    names.reserve(slots.size());
    for (const TensorInfo& slot : slots) {
        names.push_back(slot.m_engine_name.empty() ? slot.m_name.c_str()
                                                   : slot.m_engine_name.c_str());
    }
    return names;
}

// "the engine 'x' refused load: it returned 5 (ANIRA_ERROR_MODEL_LOAD)", the level named by its
// phase's word.
std::string refused(const std::string& id, anira_phase phase, anira_status status) {
    std::string message = "the engine '";
    message += id;
    message += "' refused ";
    message += anira::capi::phase_word(phase);
    message += ": it returned ";
    message += std::to_string(static_cast<int>(status));
    message += " (";
    message += anira_status_string(status);
    message += ")";
    return message;
}

}  // namespace

// ---- DescriptorLoaded -----------------------------------------------------------------------

DescriptorLoaded::DescriptorLoaded(std::shared_ptr<const anira::capi::EngineCarrier> carrier,
                                   uint32_t row,
                                   std::shared_ptr<const anira_model_config> model)
    : m_carrier(std::move(carrier)), m_row(row), m_model(std::move(model)) {}

const std::string& DescriptorLoaded::id() const noexcept {
    return m_carrier->id();
}

DescriptorLoaded::~DescriptorLoaded() {
    unload();
}

void DescriptorLoaded::init(const anira_init_info& info) {
    const anira_status status = m_carrier->ensure_init(info);
    if (status != ANIRA_OK) { throw StatusError(status, refused(id(), ANIRA_PHASE_INIT, status)); }
}

uint32_t DescriptorLoaded::flags() const noexcept {
    return m_carrier->desc().flags;
}

void DescriptorLoaded::unload() noexcept {
    if (!m_unload_owed) { return; }
    m_unload_owed = false;
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.unload != nullptr) { desc.unload(m_loaded_pointer, desc.user_data); }
    m_loaded_pointer = nullptr;
}

void DescriptorLoaded::require_initialised() const {
    if (m_carrier->initialised()) { return; }
    throw StatusError(ANIRA_ERROR_INVALID_STATE,
                      "the engine '" + id() + "' was never initialised: init runs before load");
}

void DescriptorLoaded::do_load(const Model& model) {
    // A second load starts over: what the first one loaded goes back to the engine first.
    unload();
    m_loaded_pointer = nullptr;
    // The engine received the names and binds the slots itself.
    Bindings bindings;
    bindings.m_inputs.assign(model.m_inputs.size(), ANIRA_BINDING_ENGINE);
    bindings.m_outputs.assign(model.m_outputs.size(), ANIRA_BINDING_ENGINE);
    set_bindings(std::move(bindings));

    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.load == nullptr) { return; }  // nothing to load: loaded stays NULL

    const std::vector<anira_tensor> inputs = templates_of(model.m_inputs);
    const std::vector<anira_tensor> outputs = templates_of(model.m_outputs);
    const std::vector<const char*> input_names = names_of(model.m_inputs);
    const std::vector<const char*> output_names = names_of(model.m_outputs);
    anira_engine_load_info info = ANIRA_ENGINE_LOAD_INFO_INIT;
    info.row = m_row;
    info.model = m_model.get();
    info.inputs = inputs.data();
    info.outputs = outputs.data();
    info.input_names = input_names.data();
    info.output_names = output_names.data();
    info.num_inputs = static_cast<uint32_t>(inputs.size());
    info.num_outputs = static_cast<uint32_t>(outputs.size());
    // The shared call slots: the record's count, 0 for a model run by exclusive sessions alone
    // (their calls run on what their prepare builds and claim no slot).
    info.instances = model.m_instances;
    // The provider this load is for: a value of the enum, or DEFAULT beside the custom name.
    info.provider = static_cast<uint32_t>(model.m_provider);
    info.provider_id = model.m_provider_id.empty() ? nullptr : model.m_provider_id.c_str();
    // The provider options of this backend: the set the context's "provider_options"
    // extension carries for it, on the record only for an engine whose descriptor lists the
    // kind (the handler attached them then); two parallel arrays of the pairs.
    std::vector<const char*> option_keys;
    std::vector<const char*> option_values;
    option_keys.reserve(model.m_options.size());
    option_values.reserve(model.m_options.size());
    for (const auto& [key, value] : model.m_options) {
        option_keys.push_back(key.c_str());
        option_values.push_back(value.c_str());
    }
    info.option_keys = option_keys.empty() ? nullptr : option_keys.data();
    info.option_values = option_values.empty() ? nullptr : option_values.data();
    info.num_options = static_cast<uint32_t>(option_keys.size());

    void* loaded = nullptr;
    const anira_status status = desc.load(&info, desc.user_data, &loaded);
    // The refused load owes no unload.
    if (status != ANIRA_OK) { throw StatusError(status, refused(id(), ANIRA_PHASE_LOAD, status)); }
    m_loaded_pointer = loaded;
    m_unload_owed = true;
}

std::unique_ptr<Prepared> DescriptorLoaded::do_prepare(const PrepareRequest& request) {
    return std::make_unique<DescriptorPrepared>(*this, request);
}

// ---- DescriptorPrepared ---------------------------------------------------------------------

DescriptorPrepared::DescriptorPrepared(DescriptorLoaded& loaded, const PrepareRequest& request)
    : Prepared(loaded, request.m_exclusive), m_carrier(&loaded.carrier()) {
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.prepare == nullptr) { return; }  // nothing to prepare: prepared stays NULL
    if (request.m_info == nullptr) {
        // A custom engine is prepared by the C handler with its record; the 2.x path adds no
        // engine, so a request without one is anira's own bug.
        throw StatusError(ANIRA_ERROR_INTERNAL,
                          "the engine '" + loaded.id() +
                              "' was asked to prepare a session without a prepare record");
    }
    void* prepared = nullptr;
    const anira_status status =
        desc.prepare(request.m_info, loaded.engine_loaded(), desc.user_data, &prepared);
    // The refused prepare owes no unprepare.
    if (status != ANIRA_OK) {
        throw StatusError(status, refused(loaded.id(), ANIRA_PHASE_PREPARE, status));
    }
    m_prepared = prepared;
    m_unprepare_owed = true;
}

DescriptorPrepared::~DescriptorPrepared() {
    if (!m_unprepare_owed) { return; }
    m_unprepare_owed = false;
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.unprepare != nullptr) { desc.unprepare(m_prepared, desc.user_data); }
    m_prepared = nullptr;
}

anira_status DescriptorPrepared::process(const anira_engine_ctx& call,
                                         ChunkBuffers* /*chunk*/) noexcept {
    const anira_engine_desc& desc = m_carrier->desc();
    return desc.process(&call, m_prepared, desc.user_data);
}

void DescriptorPrepared::reset(const anira_engine_ctx& call) noexcept {
    const anira_engine_desc& desc = m_carrier->desc();
    if (desc.reset != nullptr) { desc.reset(&call, m_prepared, desc.user_data); }
}

}  // namespace anira::backend
