#ifndef ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H
#define ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H

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

#include "AniraWinExports.h"

namespace anira {

class ANIRA_API HighPriorityThread {
public:
    HighPriorityThread();
    ~HighPriorityThread();
    
    void start();
    void stop();

    virtual void run() = 0;

    static void elevate_priority(std::thread::native_handle_type thread_native_handle, bool is_main_process = false);
    bool should_exit();
    bool is_running();

private:
    std::thread m_thread;
    std::atomic<bool> m_should_exit;
};

} // namespace anira

#endif // ANIRA_SYSTEM_HIGHPRIORITYTHREAD_H