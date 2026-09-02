/*
 * anira/abi/log.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_LOG_H
#define ANIRA_ABI_LOG_H

/**
 * @file log.h
 * @brief The log sink descriptor, the frozen log record and the logging entry points.
 *
 * Logging crosses the ABI as anira_log_desc in and anira_log_record out. Real-time paths write
 * fixed-size records into a lock-free queue that a drain thread or the host (anira_drain_log)
 * delivers to the sinks; control paths log synchronously. The sink runs on whichever thread
 * logs, never on the driver thread, possibly with anira's lifecycle lock held, and therefore
 * must not call anira.
 */

#include <stddef.h>
#include <stdint.h>
#include <anira/abi/export.h>
#include <anira/abi/enums.h>
#include <anira/abi/version.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

/**
 * @brief anira_log_record flag: the record came through the real-time queue.
 */
#define ANIRA_LOG_RECORD_REALTIME 1u

/**
 * @brief anira_log_record flag: the record reports a contract violation raised by the C layer
 * or the wrapper.
 */
#define ANIRA_LOG_RECORD_CONTRACT_VIOLATION 2u

/**
 * @brief One delivered log record, valid for the duration of the callback. Tier 1: 56 bytes,
 * frozen; a projection of the private logger's record.
 */
typedef struct anira_log_record {
    uint32_t level;  /**< anira_log_level. */
    uint32_t flags;  /**< ANIRA_LOG_RECORD_* bits. */
    uint32_t dropped_before;  /**< Records the queue dropped before this one. */
    uint32_t reserved;  /**< Zero. */
    uint64_t sequence;  /**< Monotonic sequence number. */
    int64_t timestamp_ms;  /**< Wall-clock UTC epoch, milliseconds. */
    uint64_t monotonic_ns;  /**< Steady-clock epoch, nanoseconds. */
    ANIRA_PTR(const char, group);  /**< "anira.<component>", NUL-terminated. */
    ANIRA_PTR(const char, message);  /**< The message, NUL-terminated. */
} anira_log_record;

/**
 * @brief A log sink. Runs on whichever thread logs (the caller of a [main-thread] entry, the
 * drain thread for real-time records, the caller of a destroy for the final flush),
 * never on the driver thread; may run with anira's lifecycle lock held and must not call
 * anira.
 * @param record The record; valid until the callback returns.
 * @param user_data The descriptor's user_data.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 */
typedef void (ANIRA_CALL* anira_log_fn)(const anira_log_record* record, void* user_data);

/**
 * @brief The log block of a machine config, handed once to anira_machine_config_set_log; the C
 * one-shot equivalent of the five scalar setters.
 */
typedef struct anira_log_desc {
    uint32_t struct_size;  /**< sizeof(anira_log_desc) of the caller's header. */
    uint32_t abi_version;  /**< ANIRA_ABI_VERSION the caller compiled against. */
    void* user_data;  /**< Passed to callback. */
    anira_log_fn callback;  /**< The sink; NULL keeps the platform sink only. */
    uint32_t level;  /**< anira_log_level; the most verbose request across machines wins. */
    uint32_t drain;  /**< anira_log_drain. */
    /**
     * Real-time queue records, clamped to [64, 65536]; fixed for the life of the core.
     */
    uint32_t queue_capacity;
    uint32_t drain_interval_ms;  /**< Drain-thread period in milliseconds. */
    uint32_t flags;  /**< ANIRA_LOG_FLAG_* bits. */
    uint32_t reserved;  /**< Zero. */
} anira_log_desc;
/**
 * @brief The defaults: no sink, WARNING, the drain thread every 10 ms, 512 records.
 */
#define ANIRA_LOG_DESC_INIT ANIRA_INIT(anira_log_desc, sizeof(anira_log_desc), ANIRA_ABI_VERSION, NULL, NULL, ANIRA_LOG_WARNING, ANIRA_LOG_DRAIN_THREAD, 512, 10, 0, 0)

/**
 * @brief Delivers the queued real-time records of this copy's core to the sinks; the host's
 * pump under ANIRA_LOG_DRAIN_MANUAL (and on Wasm, where no drain thread exists). Returns
 * 0 while no core exists.
 * @return The number of records delivered.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 * @since ABI 0.1
 */
ANIRA_API size_t ANIRA_CALL anira_drain_log(void);

/**
 * @brief Real-time logging for callers on an ANIRA_NONBLOCKING path: writes "<message> [arg0
 * arg1]" into the real-time queue without formatting or allocating; a full queue drops
 * and counts. A no-op while no core exists.
 * @param level Severity.
 * @param group "anira.<component>" or the host's own group; static storage.
 * @param static_message A static string, never a format string.
 * @param arg0 First argument, appended as "[arg0 arg1]".
 * @param arg1 Second argument.
 * @par Thread contract
 * [thread-safe] [callback-safe] ANIRA_NONBLOCKING
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_log_rt(anira_log_level level,
                                       const char* group,
                                       const char* static_message,
                                       int32_t arg0,
                                       int32_t arg1) ANIRA_NONBLOCKING;

/**
 * @brief Control-path logging through anira's private logger; formats and allocates like any
 * other control-path call and reaches every sink synchronously.
 * @param level Severity.
 * @param group "anira.<component>" or the host's own group.
 * @param message The message; copied.
 * @par Thread contract
 * [thread-safe, !audio-thread]
 * @since ABI 0.1
 */
ANIRA_API void ANIRA_CALL anira_log(anira_log_level level, const char* group, const char* message);

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_LOG_H */
