#include <anira/system/HighPriorityThread.h>
#include <anira/utils/Logger.h>

#include <cerrno>

// POSIX threading/scheduling headers are only available (and only used) on the
// non-Windows code paths below.
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <sys/resource.h>
#endif

namespace anira {

HighPriorityThread::HighPriorityThread() : m_should_exit(false) {}

HighPriorityThread::~HighPriorityThread() {
    stop();
}

void HighPriorityThread::start() {
    if (!m_thread.joinable()) {
        m_should_exit = false;
// glibc-only: set the process-default thread attributes so spawned threads use
// explicit (non-inherited) scheduling. Android's bionic libc lacks
// pthread_setattr_default_np / pthread_attr_setinheritsched, so this is skipped
// there — elevate_priority() still raises the thread to SCHED_FIFO below.
#if defined(__linux__) && !defined(__ANDROID__)
        // pthread_attr_t comes portably from <pthread.h>, not the glibc-internal <bits/...>
        // NOLINTNEXTLINE(misc-include-cleaner)
        pthread_attr_t thread_attr;
        pthread_attr_init(&thread_attr);
        pthread_attr_setinheritsched(&thread_attr, PTHREAD_EXPLICIT_SCHED);
        pthread_setattr_default_np(&thread_attr);
#endif

        m_thread = std::thread(&HighPriorityThread::run, this);

#if defined(__linux__) && !defined(__ANDROID__)
        pthread_attr_destroy(&thread_attr);
#endif

        elevate_priority(m_thread.native_handle());
        m_is_running = true;
    }
}

void HighPriorityThread::stop() {
    m_should_exit = true;
    if (m_thread.joinable()) {
        m_thread.join();
        m_is_running = false;
    }
}

void HighPriorityThread::elevate_priority(std::thread::native_handle_type thread_native_handle,
                                          bool is_main_process) {
#if WIN32
    if (is_main_process) {
        if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
            ANIRA_LOG_ERROR(log_group::k_system,
                            "Failed to set real-time priority for process. Error: %lu",
                            GetLastError());
        }
    }

    int priorities[] = {THREAD_PRIORITY_TIME_CRITICAL,
                        THREAD_PRIORITY_HIGHEST,
                        THREAD_PRIORITY_ABOVE_NORMAL};
    for (int priority : priorities) {
        if (SetThreadPriority(thread_native_handle, priority)) {
            return;
        } else {
            ANIRA_LOG_ERROR(log_group::k_system,
                            "Failed to set thread priority for Thread. Current priority: %d",
                            priority);
        }
    }
#elif __linux__
    int ret;

    // The inherit-scheduling sanity check relies on glibc-only APIs
    // (pthread_attr_getinheritsched, PTHREAD_EXPLICIT_SCHED) that bionic doesn't
    // provide, so it's compiled out on Android. The SCHED_FIFO elevation below
    // uses portable POSIX calls and runs on Android too.
#if !defined(__ANDROID__)
    if (!is_main_process) {
        int attr_inheritsched;
        // pthread_attr_t comes portably from <pthread.h>, not the glibc-internal <bits/...>
        // NOLINTNEXTLINE(misc-include-cleaner)
        pthread_attr_t thread_attr;
        pthread_getattr_np(thread_native_handle, &thread_attr);
        ret = pthread_attr_getinheritsched(&thread_attr, &attr_inheritsched);
        if (ret != 0) {
            ANIRA_LOG_ERROR(log_group::k_system,
                            "Failed to get Thread scheduling policy and params : %d",
                            errno);
        }
        if (attr_inheritsched != PTHREAD_EXPLICIT_SCHED) {
            ANIRA_LOG_ERROR(log_group::k_system,
                            "Thread scheduling policy is not PTHREAD_EXPLICIT_SCHED. Possibly "
                            "thread attributes get inherited from the main process.");
        }
        pthread_attr_destroy(&thread_attr);
    }
#else
    (void)is_main_process;
#endif

    int sch_policy;
    // sched_param comes portably from <sched.h>, not the glibc-internal <bits/...>
    // NOLINTNEXTLINE(misc-include-cleaner)
    struct sched_param sch_params;

    ret = pthread_getschedparam(thread_native_handle, &sch_policy, &sch_params);
    if (ret != 0) {
        ANIRA_LOG_ERROR(log_group::k_system,
                        "Failed to get Thread scheduling policy and params : %d",
                        errno);
    }

    // Pipewire uses SCHED_FIFO 60 and juce plugin host uses SCHED_FIFO 55 better stay below
    sch_params.sched_priority = 50;

    ret = pthread_setschedparam(thread_native_handle, SCHED_FIFO, &sch_params);
    if (ret != 0) {
        ANIRA_LOG_ERROR(log_group::k_system,
                        "Failed to set Thread scheduling policy to SCHED_FIFO and increase "
                        "the sched_priority to %d. Error : %d",
                        sch_params.sched_priority,
                        errno);
        ANIRA_LOG_WARNING(log_group::k_system,
                          "Give rtprio privileges to the user by adding the user to the "
                          "realtime/audio group. Or run the application as root.");
        ANIRA_LOG_WARNING(log_group::k_system,
                          "Instead, trying to set increased nice value for SCHED_OTHER...");

        ret = setpriority(PRIO_PROCESS, 0, -10);
        if (ret != 0) {
            ANIRA_LOG_ERROR(log_group::k_system,
                            "Failed to set increased nice value. Error : %d",
                            errno);
            ANIRA_LOG_WARNING(log_group::k_system,
                              "Using default nice value: %d",
                              getpriority(PRIO_PROCESS, 0));
        }
    }

    return;
#elif __APPLE__
    int ret;

    ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (ret != 0) {
        ANIRA_LOG_ERROR(log_group::k_system,
                        "Failed to set Thread QOS class to QOS_CLASS_USER_INTERACTIVE. Error : %d",
                        ret);
    } else {
        return;
    }

    ANIRA_LOG_ERROR(log_group::k_system,
                    "Failed to set Thread QOS class and relative priority. Error: %d",
                    ret);

    qos_class_t qos_class;
    int relative_priority;
    pthread_get_qos_class_np(pthread_self(), &qos_class, &relative_priority);

    ANIRA_LOG_WARNING(log_group::k_system,
                      "Fallback to default QOS class and relative priority: %u %d",
                      static_cast<unsigned>(qos_class),
                      relative_priority);
    return;
#endif
}

bool HighPriorityThread::should_exit() {
    return m_should_exit.load();
}

bool HighPriorityThread::is_running() {
    return m_thread.joinable() && m_is_running.load();
}

}  // namespace anira