#ifndef ANIRA_ANIRAWINEXPORTS_H
#define ANIRA_ANIRAWINEXPORTS_H

// When anira is built and consumed as a STATIC library, ANIRA_STATIC_DEFINE is
// defined (PUBLIC) by the build. In that case no dllexport/dllimport decoration
// must be applied — otherwise consumers look for __imp_ symbols that a static
// lib never provides (LNK2001).
#if defined(_WIN32) && !defined(ANIRA_STATIC_DEFINE)
#ifdef ANIRA_EXPORTS
#define ANIRA_API __declspec(dllexport)
#pragma warning (disable: 4251)
#else
#define ANIRA_API __declspec(dllimport)
#pragma warning (disable: 4251)
#endif
#else
#define ANIRA_API
#endif

#endif // ANIRA_ANIRAWINEXPORTS_H