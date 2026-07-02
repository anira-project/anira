#ifndef ANIRA_LOGGER_H
#define ANIRA_LOGGER_H

#include <iostream>

inline bool is_logging_enabled() {
#ifdef ENABLE_LOGGING
    return true;
#else
    return false;
#endif
}

#define LOG_INFO \
    if (is_logging_enabled()) (std::cout)
#define LOG_ERROR \
    if (is_logging_enabled()) (std::cerr)

#endif  // ANIRA_LOGGER_H
