// The core's C++ headers still installed (core_headers.inc) and the C ABI headers, the C umbrella
// among them, in one translation unit, in both orders: header_coexist_v2_first.cpp and
// header_coexist_abi_first.cpp (which defines ANIRA_COEXIST_ABI_FIRST) include this body.
// anira/system/Exports.h includes anira/abi/export.h, so the two spell one ANIRA_API; a second
// definition anywhere shows up here as a macro redefinition error. Include hygiene is not the point
// of this TU, so the include-cleaner check is off for the file. NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_COEXIST_ABI_FIRST)
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <anira/anira.h>
// The core's headers last.
#include "core_headers.inc"
#define ANIRA_COEXIST_PROBE anira_header_coexist_abi_first
#else
#include "core_headers.inc"
// The core's headers first; this comment keeps clang-format from regrouping the C headers
// above it, which is the point of this branch.
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <anira/anira.h>
#define ANIRA_COEXIST_PROBE anira_header_coexist_v2_first
#endif

#include <cstddef>

namespace {

[[maybe_unused]] std::size_t ANIRA_COEXIST_PROBE() {
    const anira_error err = ANIRA_ERROR_INIT;
    // A class of the core and a C record side by side; anira::RingBuffer is anira_ring.
    const anira_engine engine = ANIRA_ENGINE_CUSTOM;
    return sizeof(anira::RingBuffer) + sizeof(anira::LatencyCalculator::Rational) +
           static_cast<std::size_t>(err.status) + static_cast<std::size_t>(engine);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
