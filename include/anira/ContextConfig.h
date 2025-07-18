#ifndef ANIRA_CONTEXTCONFIG_H
#define ANIRA_CONTEXTCONFIG_H

#include <array>
#include <string>
#include <vector>
#include <thread>
#include <functional>
#include "anira/utils/InferenceBackend.h"
#include "anira/system/AniraWinExports.h"

namespace anira {

struct ANIRA_API ContextConfig {
    ContextConfig(
            unsigned int num_threads = (std::thread::hardware_concurrency() / 2 > 0) ? std::thread::hardware_concurrency() / 2 : 1) :
            m_num_threads(num_threads)
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
    }

    unsigned int m_num_threads;
    std::string m_anira_version = ANIRA_VERSION;
    std::vector<InferenceBackend> m_enabled_backends;
    

    bool operator==(const ContextConfig& other) const {
        return
            m_num_threads == other.m_num_threads &&
            m_anira_version == other.m_anira_version &&
            m_enabled_backends == other.m_enabled_backends;
    }

    bool operator!=(const ContextConfig& other) const {
        return !(*this == other);
    }

};

} // namespace anira

#endif //ANIRA_CONTEXTCONFIG_H