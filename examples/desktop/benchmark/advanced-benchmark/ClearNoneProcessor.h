#ifndef ANIRA_CLEAR_NONE_PROCESSOR_H
#define ANIRA_CLEAR_NONE_PROCESSOR_H

#include <anira/anira.h>

class ClearNoneProcessor : public anira::BackendBase {
public:
    ClearNoneProcessor(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void process(anira::AudioBufferF &input, anira::AudioBufferF &output) override {
    }
};

#endif // ANIRA_CLEAR_NONE_PROCESSOR_H