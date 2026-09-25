#ifndef ANIRA_TEST_ABI_PAIR_READ_H
#define ANIRA_TEST_ABI_PAIR_READ_H

// The (value, id) pairs of anira/abi/config.h read for an expectation: one call of the status
// getter, both halves and the status in one comparable record, the id as a string ("" for a
// NULL id, which the pair rule gives every value but CUSTOM). The refusals of the getters
// themselves are asserted on the C calls directly (test_ReadBack.cpp).

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <cstdint>
#include <ostream>
#include <string>

namespace anira_test {

struct EngineRead {
    anira_status status = ANIRA_STATUS_FORCE32;
    anira_engine engine = ANIRA_ENGINE_FORCE32;
    std::string id;

    bool operator==(const EngineRead& other) const = default;
};

struct ProviderRead {
    anira_status status = ANIRA_STATUS_FORCE32;
    anira_provider provider = ANIRA_PROVIDER_FORCE32;
    std::string id;

    bool operator==(const ProviderRead& other) const = default;
};

inline void PrintTo(const EngineRead& read, std::ostream* os) {
    *os << "{status " << static_cast<int>(read.status) << ", engine "
        << static_cast<int>(read.engine) << ", id '" << read.id << "'}";
}

inline void PrintTo(const ProviderRead& read, std::ostream* os) {
    *os << "{status " << static_cast<int>(read.status) << ", provider "
        << static_cast<int>(read.provider) << ", id '" << read.id << "'}";
}

/// anira_model_config_model_engine of entry `index`.
inline EngineRead model_engine(const anira_model_config* config, uint32_t index) {
    EngineRead read;
    const char* id = nullptr;
    read.status = anira_model_config_model_engine(config, index, &read.engine, &id);
    read.id = id != nullptr ? id : "";
    return read;
}

/// anira_model_config_default_engine.
inline EngineRead default_engine(const anira_model_config* config) {
    EngineRead read;
    const char* id = nullptr;
    read.status = anira_model_config_default_engine(config, &read.engine, &id);
    read.id = id != nullptr ? id : "";
    return read;
}

/// anira_model_config_model_provider of entry `index`.
inline ProviderRead model_provider(const anira_model_config* config, uint32_t index) {
    ProviderRead read;
    const char* id = nullptr;
    read.status = anira_model_config_model_provider(config, index, &read.provider, &id);
    read.id = id != nullptr ? id : "";
    return read;
}

/// anira_model_config_default_provider.
inline ProviderRead default_provider(const anira_model_config* config) {
    ProviderRead read;
    const char* id = nullptr;
    read.status = anira_model_config_default_provider(config, &read.provider, &id);
    read.id = id != nullptr ? id : "";
    return read;
}

}  // namespace anira_test

#endif  // ANIRA_TEST_ABI_PAIR_READ_H
