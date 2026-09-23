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
#include <string>
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

/// Whether a value is one the enum names.
inline bool known_provider(anira_provider provider) noexcept {
    for (const auto& [name, value] : k_provider_words) {
        if (value == provider) { return true; }
    }
    return false;
}

/// The name of a provider in a message: a custom provider's own name where there is one, the
/// enum's word else ("default" for a neutral plan).
inline std::string provider_label(anira_provider provider, std::string_view provider_id) {
    return provider_id.empty() ? std::string(provider_word(provider)) : std::string(provider_id);
}

/// The spellings of the built-in engines: the "engine" word of a model entry in JSON, the
/// words of every message that names an engine.
inline constexpr std::array<std::pair<const char*, anira_engine>, 5> k_engine_words{{
    {"onnxruntime", ANIRA_ENGINE_ONNXRUNTIME},
    {"libtorch", ANIRA_ENGINE_LIBTORCH},
    {"tflite", ANIRA_ENGINE_TFLITE},
    {"litert", ANIRA_ENGINE_LITERT},
    {"executorch", ANIRA_ENGINE_EXECUTORCH},
}};

/// The built-in engine a word spells; nothing for a word the enum does not name (a custom
/// engine's id).
inline std::optional<anira_engine> engine_of_word(std::string_view word) noexcept {
    for (const auto& [name, value] : k_engine_words) {
        if (word == name) { return value; }
    }
    return std::nullopt;
}

/// The word of a built-in engine; "none" for ANIRA_ENGINE_NONE and any other value.
inline const char* engine_word(anira_engine engine) noexcept {
    for (const auto& [name, value] : k_engine_words) {
        if (value == engine) { return name; }
    }
    return "none";
}

}  // namespace anira::capi
