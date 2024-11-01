#ifndef ANIRA_CLEAR_NONE_PROCESSOR_H
#define ANIRA_CLEAR_NONE_PROCESSOR_H

#include <anira/anira.h>

class ClearNoneProcessor : public anira::BackendBase {
public:
    ClearNoneProcessor(anira::InferenceConfig& inference_config) : anira::BackendBase(inference_config) {}

    void process(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
    }
};

#endif // ANIRA_CLEAR_NONE_PROCESSOR_H