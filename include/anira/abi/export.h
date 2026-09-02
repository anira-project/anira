/*
 * anira/abi/export.h -- generated from abi/anira.yml by tools/abi/gen.py.
 * Do not edit: change the registry and run `python3 tools/abi/gen.py --repo . --write`.
 */
#ifndef ANIRA_ABI_EXPORT_H
#define ANIRA_ABI_EXPORT_H

/**
 * @file export.h
 * @brief Export, calling-convention and attribute macros of the C ABI.
 *
 * Self-contained (stdint.h only). ANIRA_API is the export allowlist: hidden visibility
 * everywhere, dllexport/dllimport on Windows, empty inside a static embedding. ANIRA_CALL pins
 * __cdecl on 32-bit Windows. ANIRA_NONBLOCKING marks the real-time entries and callback slots
 * for clang's function-effects analysis and RealtimeSanitizer. ANIRA_INIT spells a compound
 * literal in C and a braced temporary in C++; ANIRA_PTR is the 8-byte pointer slot of every
 * Tier-1 record.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#if defined(__EMSCRIPTEN__)
#define ANIRA_API __attribute__((visibility("default"))) /* plus the generated -sEXPORTED_FUNCTIONS */
#elif defined(ANIRA_STATIC)
#define ANIRA_API /* hidden inside the embedding plugin; PUBLIC on static targets */
#elif defined(_WIN32)
#if defined(ANIRA_BUILDING)
#define ANIRA_API __declspec(dllexport) /* dllexport is the allowlist on PE */
#else
#define ANIRA_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ANIRA_API __attribute__((visibility("default"))) /* plus the version script (SYMBOL anira_*) */
#else
#define ANIRA_API
#endif

#if defined(_WIN32) && !defined(_WIN64)
#define ANIRA_CALL __cdecl /* fixed forever, as CLAP_ABI */
#else
#define ANIRA_CALL
#endif

#if defined(__clang__) && defined(__has_attribute)
#if __has_attribute(nonblocking)
#define ANIRA_NONBLOCKING __attribute__((nonblocking))
#endif
#endif
#ifndef ANIRA_NONBLOCKING
#define ANIRA_NONBLOCKING
#endif

#if defined(__cplusplus)
#define ANIRA_NOEXCEPT noexcept
#define ANIRA_INIT(type, ...) (type{__VA_ARGS__})
#else
#define ANIRA_NOEXCEPT
#define ANIRA_INIT(type, ...) ((type){__VA_ARGS__})
#endif

/* An 8-byte pointer slot on ILP32 and LP64 alike: zero the struct before use, the high
   half of the slot is otherwise undefined on 32-bit targets. */
#define ANIRA_PTR(T, name) \
    union {                \
        T* name;           \
        uint64_t name##_bits; \
    }

#if defined(_MSC_VER) && !defined(__clang__)
#define ANIRA_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__GNUC__) || defined(__clang__)
#define ANIRA_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define ANIRA_DEPRECATED(msg)
#endif

// NOLINTEND(readability-identifier-naming, modernize-use-using, bugprone-macro-parentheses)

#ifdef __cplusplus
}
#endif

#endif /* ANIRA_ABI_EXPORT_H */
