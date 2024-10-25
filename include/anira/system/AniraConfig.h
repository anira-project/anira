#ifndef ANIRA_CONFIG_H
#define ANIRA_CONFIG_H

#if defined(_WIN32)
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

#endif // ANIRA_CONFIG_H