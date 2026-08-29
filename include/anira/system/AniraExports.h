#ifndef ANIRA_ANIRAEXPORTS_H
#define ANIRA_ANIRAEXPORTS_H

/**
 * @file AniraExports.h
 * @brief ANIRA_API — the export decoration of anira's public API.
 *
 * anira is compiled with hidden symbol visibility; ANIRA_API is the allowlist that
 * marks what a shared libanira exports (dllexport/dllimport on Windows,
 * visibility("default") elsewhere), so that nothing else — above all the backend
 * runtimes linked into it — ever appears in its export table.
 *
 * Two macros steer it, both set by anira's CMake build:
 *
 *  - ANIRA_STATIC: anira is built and consumed as a static library. Defined PUBLIC,
 *    so consumers see it through the CMake package. ANIRA_API is then empty on every
 *    platform: a static anira has no export table of its own — its objects become
 *    part of the consumer, and a plugin embedding it must not export anira's API
 *    (dllimport would look for __imp_ stubs a static library never provides, and
 *    default visibility would leak the whole API into the plugin's export table).
 *
 *  - ANIRA_BUILDING: defined PRIVATE while compiling anira itself. Selects dllexport
 *    over dllimport on Windows. Elsewhere the decoration is the same on both sides
 *    on purpose: inline members and other vague-linkage entities a consumer
 *    instantiates from these headers keep default visibility and are coalesced with
 *    libanira's copies at load time instead of becoming private duplicates.
 *
 * ANIRA_STATIC_DEFINE, the previous spelling of ANIRA_STATIC, is still honoured for
 * hand-written build systems.
 */

#if defined(ANIRA_STATIC_DEFINE) && !defined(ANIRA_STATIC)
#define ANIRA_STATIC
#endif

#ifdef ANIRA_STATIC
#define ANIRA_API  // static: no decoration, ever
#elif defined(_WIN32)
#ifdef ANIRA_BUILDING  // set only while compiling anira itself
#define ANIRA_API __declspec(dllexport)
#else
#define ANIRA_API __declspec(dllimport)
#endif
#ifdef _MSC_VER
// C4251 ("class needs to have dll-interface to be used by clients"): anira's exported
// classes hold std:: members by design; both sides use the same toolchain.
#pragma warning(disable : 4251)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ANIRA_API __attribute__((visibility("default")))
#else
#define ANIRA_API
#endif

#endif  // ANIRA_ANIRAEXPORTS_H
