// anira/abi/engine.h: the carrier of a registered engine's descriptor and what a pipeline's
// registrations mean to the validator (the adapter that runs a registered engine is the engine
// room's DescriptorAdapter, src/backends/DescriptorAdapter.h).
#include "engine.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ext_registry.h"
#include "translate.h"

namespace anira::capi {

EngineCarrier::EngineCarrier(std::string id, const anira_engine_desc& desc)
    : m_id(std::move(id)), m_desc(desc) {
    m_kinds.reserve(desc.num_consumed_kinds);
    for (uint32_t i = 0; i < desc.num_consumed_kinds; ++i) {
        m_kinds.emplace_back(desc.consumed_kinds[i]);
    }
    m_kind_pointers.reserve(m_kinds.size());
    for (const std::string& kind : m_kinds) { m_kind_pointers.push_back(kind.c_str()); }
    // The descriptor names this carrier's strings from here on: the caller's may die.
    m_desc.consumed_kinds = m_kind_pointers.empty() ? nullptr : m_kind_pointers.data();
}

EngineCarrier::~EngineCarrier() {
    // Exactly once: the carrier dies with the last pipeline or handler that shares it, after
    // every unprepare of a prepared model that ran on it.
    if (m_desc.release != nullptr) { m_desc.release(m_desc.user_data); }
}

EngineFacts engine_facts(const std::vector<std::shared_ptr<const EngineCarrier>>& engines) {
    EngineFacts facts;
    for (const std::shared_ptr<const EngineCarrier>& engine : engines) {
        facts.m_ids.push_back(engine->id());
        if (engine->consumed_kinds().empty()) { continue; }
        facts.m_consumers.push_back(ExtConsumer{.m_name = engine->id().c_str(),
                                                .m_engine = ANIRA_ENGINE_NONE,
                                                .m_engine_id = engine->id(),
                                                .m_consumed = engine->consumed_kinds()});
    }
    return facts;
}

}  // namespace anira::capi
