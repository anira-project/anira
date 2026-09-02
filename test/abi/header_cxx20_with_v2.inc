// The v2 C++ umbrella and anira.hpp in one translation unit, in both orders:
// header_cxx20_with_v2.cpp (the umbrella first) and header_cxx20_with_v2_reverse.cpp
// (which defines ANIRA_CXX20_WITH_V2_HPP_FIRST) include this body. The two share
// namespace anira and, through anira/system/Exports.h and anira/abi/export.h, one
// ANIRA_API: a name both declare, or a macro redefined between them, shows up here.
// Include hygiene is not the point of this TU (the umbrella is), so the include-cleaner
// check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_CXX20_WITH_V2_HPP_FIRST)
#include <anira/anira.hpp>
// The umbrella second; the comment keeps clang-format from sorting anira.h above.
#include <anira/anira.h>
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2_reverse
#else
#include <anira/anira.h>

#include <anira/anira.hpp>
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2
#endif

#include <cstddef>

namespace {

[[maybe_unused]] std::size_t ANIRA_CXX20_WITH_V2_PROBE() {
    // A 2.x class and a 3.x aggregate side by side, both in namespace anira.
    return sizeof(anira::InferenceConfig) + sizeof(anira::Hard);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
