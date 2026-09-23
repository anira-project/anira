/*
 * The words anira spells its enums with, one table each, for every place the word appears.
 * The providers: the suffix of a model entry's "engine" word in JSON, the strings of an engine
 * descriptor's providers list, the words of the plan report's log lines; any other word is a
 * custom provider's name in the engine's own vocabulary. The built-in engines: the "engine"
 * word of a model entry and of every message that names one. The phases: how every message
 * about a slot names its phase.
 */
#pragma once

#include <anira/abi/enums.h>
#include <anira/abi/export.h>

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

/// The words of anira_phase, the lower-case names of its values: how every message anira writes
/// about a slot names the phase (a stage phase that fails or that an accessor refuses, a refused
/// init, load or prepare of a stage or an engine, a failed engine call).
inline constexpr std::array<std::pair<const char*, anira_phase>, 12> k_phase_words{{
    {"pre_process", ANIRA_PHASE_PRE_PROCESS},
    {"post_process", ANIRA_PHASE_POST_PROCESS},
    {"before_inference", ANIRA_PHASE_BEFORE_INFERENCE},
    {"inference", ANIRA_PHASE_INFERENCE},
    {"after_inference", ANIRA_PHASE_AFTER_INFERENCE},
    {"prepare", ANIRA_PHASE_PREPARE},
    {"release", ANIRA_PHASE_RELEASE},
    {"reset", ANIRA_PHASE_RESET},
    {"unprepare", ANIRA_PHASE_UNPREPARE},
    {"init", ANIRA_PHASE_INIT},
    {"load", ANIRA_PHASE_LOAD},
    {"unload", ANIRA_PHASE_UNLOAD},
}};

/// The word of a phase; "unknown phase" for a value the enum does not name. Real-time safe: the
/// stage's failure records and the engine call's failure record call it on the driving and the
/// inference thread.
inline const char* phase_word(anira_phase phase) noexcept ANIRA_NONBLOCKING {
    for (const auto& [name, value] : k_phase_words) {
        if (value == phase) { return name; }
    }
    return "unknown phase";
}

}  // namespace anira::capi
