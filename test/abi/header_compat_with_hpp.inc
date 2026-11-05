// anira.hpp, the C umbrella and the compat shim in one translation unit, in both orders:
// header_compat_with_hpp.cpp (the C umbrella and anira.hpp first) and
// header_compat_with_hpp_reverse.cpp (which defines ANIRA_COMPAT_WITH_HPP_SHIM_FIRST) include this
// body. anira.hpp's classes live in namespace anira, the shim's 2.x classes in anira::v2: a name
// the two declare in one namespace, or a macro redefined between the headers, shows up here.
// Include hygiene is not the point of this TU, so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_COMPAT_WITH_HPP_SHIM_FIRST)
#include <anira/compat/v2.hpp>
// The C umbrella and anira.hpp last, already included through the shim; the comment keeps
// clang-format from sorting them above.
#include <anira/anira.h>

#include <anira/anira.hpp>
#define ANIRA_COMPAT_WITH_HPP_PROBE anira_header_compat_with_hpp_reverse
#else
#include <anira/anira.h>

#include <anira/anira.hpp>
// The shim after the C umbrella and anira.hpp.
#include <anira/compat/v2.hpp>
#define ANIRA_COMPAT_WITH_HPP_PROBE anira_header_compat_with_hpp
#endif

#include <cstddef>
#include <type_traits>

namespace {

// The 3.x context config and the shim's 2.x twin, side by side and distinct.
static_assert(!std::is_same_v<anira::ContextConfig, anira::v2::ContextConfig>);
// The 3.x handler and the shim's 2.x handler.
static_assert(!std::is_same_v<anira::InferenceHandler, anira::v2::InferenceHandler>);

[[maybe_unused]] std::size_t ANIRA_COMPAT_WITH_HPP_PROBE() {
    return sizeof(anira::ContextConfig) + sizeof(anira::v2::InferenceConfig) +
           sizeof(anira::v2::JsonConfigLoader);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
