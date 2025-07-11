#ifndef ANIRA_CLEAR_CUSTOM_PROCESSOR_H
#define ANIRA_CLEAR_CUSTOM_PROCESSOR_H

#include <anira/anira.h>

class ClearCustomProcessor : public anira::BackendBase {
public:
    ClearCustomProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::BufferF &input, anira::BufferF &output, std::shared_ptr<anira::SessionElement>) override {
    }
};

#endif // ANIRA_CLEAR_CUSTOM_PROCESSOR_H