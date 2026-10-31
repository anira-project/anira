// The core's C++ headers still installed (core_headers.inc), anira.hpp, the C umbrella and the
// compat shim in one translation unit, in both orders: header_compat_with_v2.cpp (the core's
// headers first) and header_compat_with_v2_reverse.cpp (which defines
// ANIRA_COMPAT_WITH_V2_SHIM_FIRST) include this body. The core's classes live in namespace anira,
// the shim's 2.x classes in anira::v2: a name the two declare in one namespace, or a macro
// redefined between the headers, shows up here. Include hygiene is not the point of this TU, so the
// include-cleaner check is off for the file. NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_COMPAT_WITH_V2_SHIM_FIRST)
#include <anira/compat/v2.hpp>
// The core's headers second; the comment keeps clang-format from sorting them above.
#include "core_headers.inc"
// The C umbrella and anira.hpp last, already included through the shim.
#include <anira/anira.h>

#include <anira/anira.hpp>
#define ANIRA_COMPAT_WITH_V2_PROBE anira_header_compat_with_v2_reverse
#else
#include "core_headers.inc"
// The C umbrella, anira.hpp and the shim after the core's headers.
#include <anira/anira.h>

#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#define ANIRA_COMPAT_WITH_V2_PROBE anira_header_compat_with_v2
#endif

#include <cstddef>
#include <type_traits>

namespace {

// The 3.x context config and the shim's 2.x twin, side by side and distinct; the ring of the
// core and the shim's 2.x ring view.
static_assert(!std::is_same_v<anira::ContextConfig, anira::v2::ContextConfig>);
static_assert(!std::is_same_v<anira::RingBuffer, anira::v2::RingBuffer>);
// The 3.x handler and the shim's 2.x handler.
static_assert(!std::is_same_v<anira::InferenceHandler, anira::v2::InferenceHandler>);

[[maybe_unused]] std::size_t ANIRA_COMPAT_WITH_V2_PROBE() {
    return sizeof(anira::RingBuffer) + sizeof(anira::v2::InferenceConfig) +
           sizeof(anira::v2::JsonConfigLoader);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
