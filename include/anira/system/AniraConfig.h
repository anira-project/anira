#ifndef ANIRA_CONFIG_H
#define ANIRA_CONFIG_H

#if defined(_WIN32)
#ifdef ANIRA_EXPORTS
#define ANIRA_API __declspec(dllexport)
#else
#define ANIRA_API __declspec(dllimport)
#endif
#else
#define ANIRA_API
#endif

#endif // ANIRA_CONFIG_H