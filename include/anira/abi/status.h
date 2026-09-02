/*
 * anira/abi/status.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_STATUS_H
#define ANIRA_ABI_STATUS_H

/**
 * @file status.h
 * @brief Status codes, the anira_error record and the opaque handle types.
 *
 * Every fallible function returns anira_status; positive values are informational successes and
 * the set a function may return is not frozen, so the stable failure test is ANIRA_FAILED(s),
 * never s != ANIRA_OK. Functions that can produce a message take a nullable anira_error* as
 * their last parameter and fill it on failure, with every out-parameter left untouched.
 */

#include <stdint.h>
#include <anira/abi/export.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming)

/**
 * @brief Return status of every fallible entry point: 0 is success, positive values are
 * informational successes, negative values are failures.
 */
typedef enum anira_status {
    ANIRA_OK = 0,  /**< Success. */
    /**
     * Success; a version 2 JSON document was auto-upgraded (section 8.4).
     */
    ANIRA_SUCCESS_UPGRADED = 1,
    ANIRA_INCOMPLETE = 2,  /**< Success; the enumeration buffer was too short and holds what fit. */
    ANIRA_TIMEOUT = 3,  /**< A wait ran out of time. */
    ANIRA_PENDING = 4,  /**< Not yet complete. */
    ANIRA_ERROR_UNKNOWN = -1,  /**< Unclassified failure. */
    /**
     * A single argument is out of range or NULL where a value is required.
     */
    ANIRA_ERROR_INVALID_ARGUMENT = -2,
    ANIRA_ERROR_INVALID_STATE = -3,  /**< The object is not in a state that allows the call. */
    ANIRA_ERROR_OUT_OF_MEMORY = -4,  /**< An allocation failed. */
    ANIRA_ERROR_NOT_SUPPORTED = -5,  /**< Backend, domain or extension kind not in this build. */
    ANIRA_ERROR_NO_SUCH_FILE = -6,  /**< A path does not resolve to a readable file. */
    ANIRA_ERROR_MODEL_LOAD = -7,  /**< The engine refused the model. */
    ANIRA_ERROR_ENGINE = -8,  /**< The engine failed at run time. */
    /**
     * Prepare-time legality (section 2), and the real-time refusals recorded in
     * anira_handler_rt_error: a submitted tensor's dtype or axis-tag sequence, a ring or tensor
     * accessor's dtype, a non-F32 Static slot met by the float scalars.
     */
    ANIRA_ERROR_CONFIG = -9,
    /**
     * An extension was set that nothing in the pipeline consumes (section 1b).
     */
    ANIRA_ERROR_EXTENSION_UNCONSUMED = -10,
    /**
     * The extension kind is not registered in this build.
     */
    ANIRA_ERROR_EXTENSION_UNKNOWN = -11,
    /**
     * No edge connects the tensor's domain with the selected backend (section 7).
     */
    ANIRA_ERROR_EDGE_UNREACHABLE = -12,
    ANIRA_ERROR_BUDGET = -13,  /**< Hard-contract validation failed. */
    ANIRA_ERROR_CAPACITY = -14,  /**< Real-time: no free ticket slot, or a ring is full. */
    ANIRA_ERROR_TICKET_STALE = -15,  /**< The ticket names a recycled slot. */
    ANIRA_ERROR_WRONG_CONTRACT = -16,  /**< A Hard entry under an Async contract or vice versa. */
    ANIRA_ERROR_NOT_PREPARED = -17,  /**< The handler has not been prepared. */
    /**
     * A JSON document is malformed or names an unknown value; the message carries the key path.
     */
    ANIRA_ERROR_JSON = -18,
    ANIRA_ERROR_ABI_VERSION = -19,  /**< anira_check_abi: the header and the library disagree. */
    /**
     * A (buf, cap, out_len) text buffer is too small; out_len holds the required size.
     */
    ANIRA_ERROR_BUFFER_TOO_SMALL = -20,
    ANIRA_ERROR_DEVICE = -21,  /**< A device descriptor was refused or a device call failed. */
    /**
     * The extension kind is known but not at that version.
     */
    ANIRA_ERROR_EXTENSION_VERSION = -22,
    /**
     * An exception the firewall could not classify; a bug in anira.
     */
    ANIRA_ERROR_INTERNAL = -100,
    ANIRA_STATUS_FORCE32 = 0x7fffffff
} anira_status;

/**
 * @brief The only stable failure test: true for every negative status. A minor may add positive
 * (informational) statuses, so never compare against ANIRA_OK.
 */
#define ANIRA_FAILED(s) ((int32_t)(s) < 0)

/**
 * @brief The complement of ANIRA_FAILED.
 */
#define ANIRA_SUCCEEDED(s) ((int32_t)(s) >= 0)

/**
 * @brief Capacity of anira_error::message including the terminating NUL.
 */
#define ANIRA_ERROR_MESSAGE_CAPACITY 512

/**
 * @brief The opaque handle types. Config handles are value-like (copied at
 * anira_handler_create, destroyable right after); the runtime handles are refcounted
 * internally. Every anira_<x>_destroy is NULL-safe.
 */
typedef struct anira_tensor_spec anira_tensor_spec;
typedef struct anira_model_config anira_model_config;
typedef struct anira_machine_config anira_machine_config;
typedef struct anira_contract anira_contract;
typedef struct anira_job_options anira_job_options;
typedef struct anira_pipeline anira_pipeline;
typedef struct anira_machine anira_machine;
typedef struct anira_capabilities anira_capabilities;
typedef struct anira_inference_thread anira_inference_thread;
typedef struct anira_handler anira_handler;
typedef struct anira_plan_report anira_plan_report;

/**
 * @brief Caller-owned error record, filled by the callee on failure: the status and a
 * NUL-terminated message. Tier 1: 520 bytes, frozen; identical on every target, so
 * anira::Result<T> value-initialises it and TypeScript allocates it with
 * anira_sizeof(ANIRA_STRUCT_ERROR).
 */
typedef struct anira_error {
    int32_t status;  /**< The anira_status of the failure; ANIRA_OK after ANIRA_ERROR_INIT. */
    uint32_t reserved;  /**< Padding; zero. */
    /**
     * UTF-8, NUL-terminated; empty when no message was produced.
     */
    char message[ANIRA_ERROR_MESSAGE_CAPACITY];
} anira_error;
/**
 * @brief An anira_error with ANIRA_OK and an empty message.
 */
#define ANIRA_ERROR_INIT ANIRA_INIT(anira_error, ANIRA_OK, 0u, {0})

/**
 * @brief Static text for a status code, for diagnostics; never NULL. An unknown value yields
 * "unknown status".
 * @param status Any status, known or not.
 * @return A NUL-terminated string in static storage.
 * @par Thread contract
 * [thread-safe]
 * @since ABI 0.1
 */
ANIRA_API const char* ANIRA_CALL anira_status_string(anira_status status);

// NOLINTEND(readability-identifier-naming)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_STATUS_H */
