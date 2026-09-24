// anira/abi/engine.h: the carrier of a registered engine's descriptor and what a pipeline's
// registrations mean to the validator (the adapter that runs a registered engine is the engine
// room's DescriptorAdapter, src/backends/DescriptorAdapter.h).
#include "engine.h"

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/status.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ext_registry.h"
#include "validate.h"
#include "words.h"

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
    m_providers.reserve(desc.num_providers);
    for (uint32_t i = 0; i < desc.num_providers; ++i) {
        m_providers.emplace_back(desc.providers[i]);
    }
    m_provider_pointers.reserve(m_providers.size());
    for (const std::string& provider : m_providers) {
        m_provider_pointers.push_back(provider.c_str());
    }
    m_desc.providers = m_provider_pointers.empty() ? nullptr : m_provider_pointers.data();
}

bool EngineCarrier::serves(anira_provider provider, std::string_view provider_id) const noexcept {
    if (provider == ANIRA_PROVIDER_DEFAULT && provider_id.empty()) { return true; }
    for (const std::string& word : m_providers) {
        if (provider != ANIRA_PROVIDER_DEFAULT) {
            if (provider_of_word(word) == provider) { return true; }
        } else if (word == provider_id) {
            return true;
        }
    }
    return false;
}

EngineCarrier::~EngineCarrier() {
    // Exactly once: the carrier dies with the last handle, pipeline, handler or loaded model
    // that shares it, after every unload of a loaded model that ran on it, whether or not init
    // ever ran.
    if (m_desc.release != nullptr) { m_desc.release(m_desc.user_data); }
}

anira_status EngineCarrier::ensure_init(const anira_init_info& info) const {
    const std::scoped_lock<std::mutex> lock(m_init_mutex);
    if (m_initialised) { return ANIRA_OK; }
    if (m_desc.init != nullptr) {
        const anira_status status = m_desc.init(&info, m_desc.user_data);
        if (status != ANIRA_OK) { return status; }
    }
    m_initialised = true;
    return ANIRA_OK;
}

bool EngineCarrier::initialised() const {
    const std::scoped_lock<std::mutex> lock(m_init_mutex);
    return m_initialised;
}

anira_status EngineCarrier::query(const anira_init_info& info, uint64_t& available) const {
    const uint64_t all =
        m_providers.size() >= 64 ? ~uint64_t{0} : (uint64_t{1} << m_providers.size()) - 1;
    available = all;
    if (m_desc.query == nullptr) { return ANIRA_OK; }
    uint64_t answer = 0;
    const anira_status status = m_desc.query(&info, m_desc.user_data, &answer);
    if (status != ANIRA_OK) {
        available = 0;
        return status;
    }
    available = answer & all;
    return ANIRA_OK;
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
