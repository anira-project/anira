// The v2 C++ umbrella and the C ABI headers in one translation unit, in both orders:
// header_coexist_v2_first.cpp and header_coexist_abi_first.cpp (which defines
// ANIRA_COEXIST_ABI_FIRST) include this body. anira/system/Exports.h includes
// anira/abi/export.h, so the two spell one ANIRA_API; a second definition anywhere shows
// up here as a macro redefinition error. Include hygiene is not the point of this TU (the
// umbrella is), so the include-cleaner check is off for the file.
// NOLINTBEGIN(misc-include-cleaner)
#if defined(ANIRA_COEXIST_ABI_FIRST)
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <anira/anira.h>
#define ANIRA_COEXIST_PROBE anira_header_coexist_abi_first
#else
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
    // A v2 type and a v3 record side by side; anira::LogLevel and anira_log_level coexist.
    const anira::LogLevel v2_level = anira::LogLevel::Warning;
    const anira_log_level v3_level = ANIRA_LOG_WARNING;
    return sizeof(anira::InferenceConfig) + static_cast<std::size_t>(err.status) +
           static_cast<std::size_t>(v2_level) + static_cast<std::size_t>(v3_level);
}

}  // namespace
// NOLINTEND(misc-include-cleaner)
