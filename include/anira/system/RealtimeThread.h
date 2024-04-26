#ifndef ANIRA_SYSTEM_REALTIMETHREAD_H
#define ANIRA_SYSTEM_REALTIMETHREAD_H

#if WIN32
    #include <windows.h>
#elif __linux__
    #include <pthread.h>
    #include <sys/resource.h>
#elif __APPLE__
    #include <pthread.h>
    #include <sys/qos.h>
#endif
#include <thread>
#include <iostream>

#include "AniraConfig.h"

namespace anira {
namespace system {

class ANIRA_API RealtimeThread {
public:
    RealtimeThread();
    ~RealtimeThread();
    
    void start();
    void stop();

    virtual void run() = 0;

    static void elevateToRealTimePriority(std::thread::native_handle_type thread_native_handle, bool is_main_process = false);
    bool shouldExit();

private:
    std::thread thread;
    std::atomic<bool> m_should_exit;
};

} // namespace system
} // namespace anira

#endif // ANIRA_SYSTEM_REALTIMETHREAD_H