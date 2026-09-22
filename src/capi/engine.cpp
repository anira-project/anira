// anira/abi/engine.h: the carrier of a registered engine's descriptor, the registration alone
// (the adapter that runs a registered engine arrives with the engine room of src/backends).
#include "engine.h"

#include <anira/abi/engine.h>

#include <cstdint>
#include <string>
#include <utility>

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

}  // namespace anira::capi
