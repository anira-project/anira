#ifndef ANIRA_EXPORTS_H
#define ANIRA_EXPORTS_H

/**
 * @file Exports.h
 * @brief ANIRA_API — the export decoration of anira's public API.
 *
 * anira is compiled with hidden symbol visibility; ANIRA_API is the allowlist that
 * marks what a shared libanira exports (dllexport/dllimport on Windows,
 * visibility("default") elsewhere), so that nothing else — above all the backend
 * runtimes linked into it — ever appears in its export table. The platform switch
 * lives in the self-contained C ABI header anira/abi/export.h, which this header
 * includes: the v2 C++ headers and the C headers spell one and the same macro.
 *
 * Two macros steer it, both set by anira's CMake build (tanh_apply_symbol_policy):
 *
 *  - ANIRA_STATIC: anira is built and consumed as a static library. Defined PUBLIC,
 *    so consumers see it through the CMake package. ANIRA_API is then empty on every
 *    platform: a static anira has no export table of its own — its objects become
 *    part of the consumer, and a plugin embedding it must not export anira's API
 *    (dllimport would look for __imp_ stubs a static library never provides, and
 *    default visibility would leak the whole API into the plugin's export table).
 *
 *  - ANIRA_BUILDING: defined PRIVATE while compiling anira itself. Selects dllexport
 *    over dllimport on Windows.
 *
 * ANIRA_STATIC_DEFINE, the previous spelling of ANIRA_STATIC, is still honoured for
 * hand-written build systems.
 */

// The legacy spelling must be mapped before abi/export.h decides.
#if defined(ANIRA_STATIC_DEFINE) && !defined(ANIRA_STATIC)
#define ANIRA_STATIC
#endif

#include <anira/abi/export.h>

#endif  // ANIRA_EXPORTS_H
