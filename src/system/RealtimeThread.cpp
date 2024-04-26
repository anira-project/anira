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

void RealtimeThread::elevateToRealTimePriority(std::thread::native_handle_type thread_native_handle) {
#if WIN32
    int priorities[] = {THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_ABOVE_NORMAL};

    for (int priority : priorities) {
        if (SetThreadPriority(thread_native_handle, priority)) {
            return;
        } else {
            std::cerr << "Failed to set thread priority " << priority << std::endl;
        }
    }
#elif __linux__
    int sch_policy;
    struct sched_param sch_params;

    int ret = pthread_getschedparam(thread_native_handle, &sch_policy, &sch_params);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    sch_params.sched_priority = 80;

    ret = pthread_setschedparam(thread_native_handle, SCHED_FIFO, &sch_params); 
    if(ret != 0) {
        std::cerr << "Failed to set Thread scheduling policy and params : " << errno << std::endl;
        std::cout << "Try running the application as root or with sudo, or add the user to the realtime/audio group" << std::endl;
    }

    ret = pthread_getschedparam(thread_native_handle, &sch_policy, &sch_params);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    if (sch_policy != SCHED_FIFO) {
        std::cerr << "Failed to set thread scheduling policy to SCHED_FIFO" << std::endl;
    }
    if (sch_params.sched_priority != 80) {
        std::cerr << "Failed to set thread scheduling priority to 80" << std::endl;
    }

    int attr_inheritsched;
    pthread_attr_t thread_attr;
    ret = pthread_getattr_np(thread_native_handle, &thread_attr);
    ret = pthread_attr_getinheritsched(&thread_attr, &attr_inheritsched);
    if(ret != 0) {
        std::cerr << "Failed to get Thread scheduling policy and params : " << errno << std::endl;
    }

    pthread_attr_destroy(&thread_attr);
#endif
}

bool RealtimeThread::shouldExit() {
    return m_should_exit;
}

} // namespace system
} // namespace anira