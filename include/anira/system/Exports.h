#ifndef ANIRA_EXPORTS_H
#define ANIRA_EXPORTS_H

/**
 * @file Exports.h
 * @brief ANIRA_API — the export decoration of anira's public API.
 *
 * anira is compiled with hidden symbol visibility; ANIRA_API is the allowlist that
 * marks what a shared libanira exports (dllexport/dllimport on Windows,
 * visibility("default") elsewhere — the platform switch is tanh-lib's
 * tanh/core/ExportMacros.h), so that nothing else — above all the backend runtimes
 * linked into it — ever appears in its export table.
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

#include <tanh/core/ExportMacros.h>

#if defined(ANIRA_STATIC_DEFINE) && !defined(ANIRA_STATIC)
#define ANIRA_STATIC
#endif

#if defined(ANIRA_STATIC)
#define ANIRA_API
#elif defined(ANIRA_BUILDING)
#define ANIRA_API THL_DECL_EXPORT
#else
#define ANIRA_API THL_DECL_IMPORT
#endif

#endif  // ANIRA_EXPORTS_H
