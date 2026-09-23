/*
 * The spellings of anira_provider: the suffix of a model entry's "engine" word in JSON, the
 * strings of an engine descriptor's providers list, the words of the plan report's log lines.
 * One vocabulary for every place a provider is spelled; any other word is a custom provider's
 * name in the engine's own vocabulary.
 */
#pragma once

#include <anira/abi/enums.h>

#include <array>
#include <optional>
#include <string_view>
#include <utility>

namespace anira::capi {

inline constexpr std::array<std::pair<const char*, anira_provider>, 7> k_provider_words{{
    {"default", ANIRA_PROVIDER_DEFAULT},
    {"cuda", ANIRA_PROVIDER_CUDA},
    {"webgpu", ANIRA_PROVIDER_WEBGPU},
    {"directml", ANIRA_PROVIDER_DIRECTML},
    {"coreml", ANIRA_PROVIDER_COREML},
    {"xnnpack", ANIRA_PROVIDER_XNNPACK},
    {"vulkan", ANIRA_PROVIDER_VULKAN},
}};

/// The provider a word spells; nothing for a word the enum does not name (a custom provider).
inline std::optional<anira_provider> provider_of_word(std::string_view word) noexcept {
    for (const auto& [name, value] : k_provider_words) {
        if (word == name) { return value; }
    }
    return std::nullopt;
}

/// The word of a provider of the enum; "unknown" for a value the enum does not name.
inline const char* provider_word(anira_provider provider) noexcept {
    for (const auto& [name, value] : k_provider_words) {
        if (value == provider) { return name; }
    }
    return "unknown";
}

}  // namespace anira::capi
