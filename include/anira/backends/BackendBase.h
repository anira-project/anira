#ifndef ANIRA_BACKENDBASE_H
#define ANIRA_BACKENDBASE_H

#include "../InferenceConfig.h"
#include "../utils/Buffer.h"
#include "../system/AniraWinExports.h"
#include <memory>

namespace anira {

class SessionElement; // Forward declaration as we have a circular dependency

class ANIRA_API BackendBase {
public:
    BackendBase(InferenceConfig& inference_config);
    virtual void prepare();
    virtual void process(BufferF& input, BufferF& output, [[maybe_unused]] std::shared_ptr<SessionElement> session);

    InferenceConfig& m_inference_config;
};

} // namespace anira

#endif //ANIRA_BACKENDBASE_H
