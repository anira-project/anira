#include <anira/system/RealtimeThread.h>

namespace anira {
namespace system {

RealtimeThread::RealtimeThread() : m_should_exit(false){
}

RealtimeThread::~RealtimeThread() {
    stop();
}

void RealtimeThread::start() {
    m_should_exit = false;
    #if __linux__
        pthread_attr_t thread_attr;
        pthread_attr_init(&thread_attr);
        pthread_attr_setinheritsched(&thread_attr, PTHREAD_EXPLICIT_SCHED);
        pthread_setattr_default_np(&thread_attr);
    #endif
        thread = std::thread(&RealtimeThread::run, this);
    #if __linux__
        pthread_attr_destroy(&thread_attr);
    #endif
        elevateToRealTimePriority(thread.native_handle());
    }

void RealtimeThread::stop() {
    m_should_exit = true;
    if (thread.joinable()) thread.join();
}   

void RealtimeThread::elevateToRealTimePriority(std::thread::native_handle_type thread_native_handle, bool is_main_process) {
#if WIN32
    if (is_main_process) {
        if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
           std::cerr << "Failed to set real-time priority for process. Error: " << GetLastError() << std::endl;
        }
    }

    int priorities[] = {THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_ABOVE_NORMAL};
    for (int priority : priorities) {
        if (SetThreadPriority(thread_native_handle, priority)) {
            return;
        } else {
            std::cerr << "Failed to set thread priority for Thread. Current priority: " << priority << std::endl;
        }
    }
#elif __linux__
    int ret;

    if (!is_main_process) {
        int attr_inheritsched;
        pthread_attr_t thread_attr;
        ret = pthread_getattr_np(thread_native_handle, &thread_attr);
        ret = pthread_attr_getinheritsched(&thread_attr, &attr_inheritsched);
        if(ret != 0) {
            std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;\
        }
        if (attr_inheritsched != PTHREAD_EXPLICIT_SCHED) {
            std::cerr << "Thread scheduling policy is not PTHREAD_EXPLICIT_SCHED. Possibly thread attributes get inherited from the main process." << std::endl;
        }
        pthread_attr_destroy(&thread_attr);
    }

    int sch_policy;
    struct sched_param sch_params;

    ret = pthread_getschedparam(thread_native_handle, &sch_policy, &sch_params);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    sch_params.sched_priority = 85;

    ret = pthread_setschedparam(thread_native_handle, SCHED_FIFO, &sch_params); 
    if(ret != 0) {
        std::cerr << "Failed to set Thread scheduling policy to SCHED_FIFO and increase the sched_priority to " << sch_params.sched_priority << ". Error : " << errno << std::endl;
        std::cout << "Give rtprio privileges to the user by adding the user to the realtime/audio group. Or run the application as root." << std::endl;
    } else {
        return;
    }

    std::cout << "Instead, trying to set increased nice value for SCHED_OTHER..." << std::endl;
    ret = setpriority(PRIO_PROCESS, 0, -10);

    if(ret != 0) {
        std::cerr << "Failed to set increased nice value. Error : " << errno << std::endl;
        std::cout << "Using default nice value: " << getpriority(PRIO_PROCESS, 0) << std::endl;
    }
    return;
#endif
}

bool RealtimeThread::shouldExit() {
    return m_should_exit;
}

} // namespace system
} // namespace anira