// The core's C++ headers still installed (core_headers.inc) and anira.hpp in one translation unit,
// in both orders: header_cxx20_with_v2.cpp (the core's headers first) and
// header_cxx20_with_v2_reverse.cpp (which defines ANIRA_CXX20_WITH_V2_HPP_FIRST) include this body.
// The two share namespace anira and, through anira/system/Exports.h and anira/abi/export.h, one
// ANIRA_API: a name both declare (a class libanira defines and anira.hpp defines inline would be an
// ODR violation on every link), or a macro redefined between them, shows up here as a compile
// error. The library's scheduler thread is anira::scheduler::InferenceThread, the 3.x class
// anira::InferenceThread. Include hygiene is not the point of this TU, so the include-cleaner check
// is off for the file. NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_CXX20_WITH_V2_HPP_FIRST)
#include <anira/anira.hpp>
// The core's headers second; the comment keeps clang-format from sorting them above.
#include "core_headers.inc"
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2_reverse
#else
#include "core_headers.inc"
// anira.hpp second.
#include <anira/anira.hpp>
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2
#endif

#include <cstddef>
#include <type_traits>

namespace {

// The two thread classes and the two handler faces, side by side and distinct.
static_assert(!std::is_same_v<anira::InferenceThread, anira::scheduler::InferenceThread>);
static_assert(std::is_class_v<anira::InferenceHandler>);
static_assert(std::is_class_v<anira::InferenceManager>);

[[maybe_unused]] std::size_t ANIRA_CXX20_WITH_V2_PROBE() {
    // A class of the core and a 3.x aggregate side by side, both in namespace anira.
    return sizeof(anira::LatencyCalculator::Rational) + sizeof(anira::Hard) +
           sizeof(anira::InferenceHandler);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
