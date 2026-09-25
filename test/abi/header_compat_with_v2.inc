// The v2 C++ umbrella, anira.hpp and the compat shim in one translation unit, in both orders:
// header_compat_with_v2.cpp (the umbrella first) and header_compat_with_v2_reverse.cpp (which
// defines ANIRA_COMPAT_WITH_V2_SHIM_FIRST) include this body. The 2.x classes live in namespace
// anira, the shim's twins of them in anira::v2: a name the two declare in one namespace, or a
// macro redefined between the headers, shows up here. Include hygiene is not the point of this
// TU, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_COMPAT_WITH_V2_SHIM_FIRST)
#include <anira/compat/v2.hpp>
// The umbrella second; the comment keeps clang-format from sorting anira.h above.
#include <anira/anira.h>
// anira.hpp last, already included through the shim.
#include <anira/anira.hpp>
#define ANIRA_COMPAT_WITH_V2_PROBE anira_header_compat_with_v2_reverse
#else
#include <anira/anira.h>

#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#define ANIRA_COMPAT_WITH_V2_PROBE anira_header_compat_with_v2
#endif

#include <cstddef>
#include <type_traits>

namespace {

// Each 2.x class and its shim twin, side by side and distinct.
static_assert(!std::is_same_v<anira::InferenceConfig, anira::v2::InferenceConfig>);
static_assert(!std::is_same_v<anira::HostConfig, anira::v2::HostConfig>);
static_assert(!std::is_same_v<anira::ModelData, anira::v2::ModelData>);
static_assert(!std::is_same_v<anira::LogConfig, anira::v2::LogConfig>);
static_assert(!std::is_same_v<anira::ContextConfig, anira::v2::ContextConfig>);

[[maybe_unused]] std::size_t ANIRA_COMPAT_WITH_V2_PROBE() {
    return sizeof(anira::InferenceConfig) + sizeof(anira::v2::InferenceConfig) +
           sizeof(anira::JsonConfigLoader) + sizeof(anira::v2::JsonConfigLoader);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
