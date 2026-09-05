/*
 * anira/abi/core.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_CORE_H
#define ANIRA_ABI_CORE_H

/**
 * @file core.h
 * @brief The core of this copy of anira: its steady clock and the shutdown family.
 *
 * One core per copy of anira (a shared library has one, every static embedding its own) holds
 * what every context and handler shares: the inference thread pool and queue, the backend
 * processor pools, the real-time log queue and its drain. It is created by the first call that
 * needs it and lives until the copy is unloaded; a host never creates or destroys it, it only
 * asks it to shut its threads down (anira_shutdown) or to free itself once nothing uses it
 * (anira_release_core_if_idle), both for a static embedding about to be unloaded. anira_now_ms
 * / anira_now_ns are the steady clock deadlines are spelled in.
 */

#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/enums.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief The steady clock in milliseconds, for deadlines and submit timestamps; the same clock
 * as anira_now_ns.
 * @return Milliseconds since an unspecified steady epoch.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API double ANIRA_CALL anira_now_ms(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief The steady clock in nanoseconds; the one allowlisted 64-bit return on an
 * ANIRA_NONBLOCKING declaration.
 * @return Nanoseconds since an unspecified steady epoch.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.2
 */
ANIRA_API uint64_t ANIRA_CALL anira_now_ns(void) ANIRA_NOEXCEPT ANIRA_NONBLOCKING;

/**
 * @brief Stops and joins the core's inference threads and its log drain thread and flushes the
 * queue, for a static embedding that is about to be unloaded (called from clap_deinit or
 * ExitDll). Idempotent, never creates the core, and effective only when no context and
 * no handler exist in this copy: otherwise nothing happens, so one client of a shared
 * library cannot silence another's sessions.
 * @return ANIRA_OK (also without a core); ANIRA_ERROR_INVALID_STATE while a context or a
 *         handler lives.
 * @par Thread contract
 * [main-thread & !loader-lock]
 * @since ABI 0.2
 */
ANIRA_API anira_status ANIRA_CALL anira_shutdown(void) ANIRA_NOEXCEPT;

/**
 * @brief Frees the core when nothing uses it: no context, no handler, no pool thread, no
 * user-driven inference thread and, on WebAssembly, no inference loop still running.
 * Never blocks; the unload hook's call.
 * @return Nonzero when the core was freed.
 * @par Thread contract
 * [main-thread]
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_release_core_if_idle(void) ANIRA_NOEXCEPT;

/**
 * @brief Whether this copy of anira holds a core.
 * @return Nonzero while the core exists.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.2
 */
ANIRA_API anira_bool ANIRA_CALL anira_has_core(void) ANIRA_NOEXCEPT;

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_CORE_H */
