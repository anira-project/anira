#ifndef ANIRA_ANIRACONTEXTCONFIG_H
#define ANIRA_ANIRACONTEXTCONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include "anira/utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"

namespace anira {

enum SynchronizationType {
    SEMAPHORE,
    ATOMIC
};

struct ANIRA_API AniraContextConfig {
    AniraContextConfig(
            int num_threads = ((int) std::thread::hardware_concurrency() / 2 > 0) ? (int) std::thread::hardware_concurrency() / 2 : 1, bool use_host_threads = false) :
            m_num_threads(num_threads),
            m_use_host_threads(use_host_threads)
    {
#ifdef USE_LIBTORCH
        m_enabled_backends.push_back(InferenceBackend::LIBTORCH);
#endif
#ifdef USE_ONNXRUNTIME
        m_enabled_backends.push_back(InferenceBackend::ONNX);
#endif
#ifdef USE_TFLITE
        m_enabled_backends.push_back(InferenceBackend::TFLITE);
#endif
#ifdef USE_SEMAPHORE
        m_synchronization_type = SynchronizationType::SEMAPHORE;
#else
        m_synchronization_type = SynchronizationType::ATOMIC;
#endif
    }

    int m_num_threads;
    bool m_use_host_threads;
    std::string m_anira_version = ANIRA_VERSION;
    std::vector<InferenceBackend> m_enabled_backends;
    SynchronizationType m_synchronization_type;
    

    bool operator==(const AniraContextConfig& other) const {
        return
            m_num_threads == other.m_num_threads &&
            m_use_host_threads == other.m_use_host_threads &&
            m_anira_version == other.m_anira_version &&
            m_enabled_backends == other.m_enabled_backends &&
            m_synchronization_type == other.m_synchronization_type;

    }

    bool operator!=(const AniraContextConfig& other) const {
        return !(*this == other);
    }

};

} // namespace anira

#endif //ANIRA_ANIRACONTEXTCONFIG_H