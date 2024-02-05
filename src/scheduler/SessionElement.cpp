#include <aari/scheduler/SessionElement.h>

SessionElement::SessionElement(int newSessionID, PrePostProcessor& ppP, InferenceConfig& config) :
    sessionID(newSessionID),
    prePostProcessor(ppP),
    inferenceConfig(config)
{
}
