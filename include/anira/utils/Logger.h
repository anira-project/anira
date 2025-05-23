#ifndef ANIRA_LOGGER_H
#define ANIRA_LOGGER_H

#include <iostream>

inline bool isLoggingEnabled() {
#ifdef ENABLE_LOGGING
    return true;
#else
    return false;
#endif
}

#define LOG_INFO  if (isLoggingEnabled()) (std::cout)
#define LOG_ERROR if (isLoggingEnabled()) (std::cerr)

#endif //ANIRA_LOGGER_H
