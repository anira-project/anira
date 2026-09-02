/*
 * anira/abi/version.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_VERSION_H
#define ANIRA_ABI_VERSION_H

/**
 * @file version.h
 * @brief ABI version packing and negotiation, and the library's release identity.
 *
 * ANIRA_ABI_MAJOR / ANIRA_ABI_MINOR come from the generated build_info.h, derived from the git
 * tag by cmake/build-info.cmake: 0.N through the pre-releases, X.Y from v3.0.0 on.
 * anira_check_abi is the one negotiation call: exact match while the major is 0, same major and
 * a library minor at least the header's afterwards.
 */

#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/build_info.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief Packs an ABI pair into one uint32_t: major in the high 16 bits, minor in the low 16.
 */
#define ANIRA_MAKE_ABI_VERSION(major, minor) (((uint32_t)(major) << 16) | (uint32_t)(minor))

/**
 * @brief The packed ABI version this header was compiled against; what descriptors carry in
 * abi_version and what anira_check_abi takes.
 */
#define ANIRA_ABI_VERSION ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR, ANIRA_ABI_MINOR)

/**
 * @brief The major of a packed ABI version.
 */
#define ANIRA_ABI_VERSION_MAJOR(v) ((uint32_t)(v) >> 16)

/**
 * @brief The minor of a packed ABI version.
 */
#define ANIRA_ABI_VERSION_MINOR(v) ((uint32_t)(v) & 0xffffu)

/**
 * @brief The ANIRA_ABI_VERSION the library was built with.
 * @return The packed pair; unpack with ANIRA_ABI_VERSION_MAJOR / ANIRA_ABI_VERSION_MINOR.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.1
 */
ANIRA_API uint32_t ANIRA_CALL anira_abi_version(void) ANIRA_NONBLOCKING;

/**
 * @brief Negotiates the header the caller compiled against with the library it loaded: ANIRA_OK
 * iff the majors are equal and the library's minor is at least the header's; while the
 * major is 0 (before v3.0.0) the pair must match exactly.
 * @param header_abi_version The caller's ANIRA_ABI_VERSION.
 * @return ANIRA_OK, or ANIRA_ERROR_ABI_VERSION.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.1
 */
ANIRA_API anira_status ANIRA_CALL anira_check_abi(uint32_t header_abi_version) ANIRA_NONBLOCKING;

/**
 * @brief The library's release version, packed as ANIRA_MAKE_VERSION(major, minor, patch).
 * @return The packed semver triple of the library.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.1
 */
ANIRA_API uint32_t ANIRA_CALL anira_version(void) ANIRA_NONBLOCKING;

/**
 * @brief The library's full version string (git describe without the leading v), e.g.
 * "3.0.0-alpha.1-12-gabc123"; what ANIRA_VERSION_STRING was for the build that produced
 * the library.
 * @return A NUL-terminated string in static storage.
 * @par Thread contract
 * [thread-safe] ANIRA_NONBLOCKING
 * @since ABI 0.1
 */
ANIRA_API const char* ANIRA_CALL anira_version_string(void) ANIRA_NONBLOCKING;

/**
 * @brief Feature detection for dlopen hosts: the address of a promised entry point of this
 * build, or NULL when the name is unknown, NULL, or the entry is not in this build.
 * @param name The entry point's name, e.g. "anira_abi_version".
 * @return The function's address cast to void*, or NULL.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.1
 */
ANIRA_API void* ANIRA_CALL anira_get_proc_address(const char* name);

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_VERSION_H */
