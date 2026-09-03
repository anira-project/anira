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
// The bridge last: it includes both.
#include <anira/compat/v3_to_v2.h>
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2_reverse
#else
#include <anira/anira.h>

#include <anira/anira.hpp>
#include <anira/compat/v3_to_v2.h>
#define ANIRA_CXX20_WITH_V2_PROBE anira_header_cxx20_with_v2
#endif

#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

[[maybe_unused]] std::size_t ANIRA_CXX20_WITH_V2_PROBE() {
    // A 2.x class and a 3.x aggregate side by side, both in namespace anira.
    return sizeof(anira::InferenceConfig) + sizeof(anira::Hard);
}

// The bridge takes the model config by lvalue only (a bytes entry is borrowed).
template <class Model, class = void>
struct BridgeTakes : std::false_type {};
template <class Model>
struct BridgeTakes<Model,
                   std::void_t<decltype(anira::v3compat::to_inference_config(
                       std::declval<Model>(), std::declval<const anira::Hard&>()))>>
    : std::true_type {};
static_assert(BridgeTakes<anira::ModelConfig&>::value);
static_assert(BridgeTakes<const anira::ModelConfig&>::value);
static_assert(!BridgeTakes<anira::ModelConfig>::value, "an rvalue model config is refused");

}  // namespace
// NOLINTEND(misc-include-cleaner)
