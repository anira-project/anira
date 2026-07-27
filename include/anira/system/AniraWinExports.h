#ifndef ANIRA_ANIRAWINEXPORTS_H
#define ANIRA_ANIRAWINEXPORTS_H

// When anira is built and consumed as a STATIC library, ANIRA_STATIC_DEFINE is
// defined (PUBLIC) by the build. In that case no dllexport/dllimport decoration
// must be applied — otherwise consumers look for __imp_ symbols that a static
// lib never provides (LNK2001).
//
// On ELF/Mach-O, anira is compiled with -fvisibility=hidden and ANIRA_API marks
// the public API as default-visibility — the same allowlist model Windows DLLs
// enforce via dllexport. Everything not annotated (backend runtimes such as
// ONNX Runtime above all) stays private to the library/plugin binary, so a host
// application that ships its own copy of a backend runtime can never interpose
// or weak-coalesce against ours (see the symbol-visibility note in the build).
#if defined(_WIN32) && !defined(ANIRA_STATIC_DEFINE)
#ifdef ANIRA_EXPORTS
#define ANIRA_API __declspec(dllexport)
#pragma warning(disable : 4251)
#else
#define ANIRA_API __declspec(dllimport)
#pragma warning(disable : 4251)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define ANIRA_API __attribute__((visibility("default")))
#else
#define ANIRA_API
#endif

#endif  // ANIRA_ANIRAWINEXPORTS_H
