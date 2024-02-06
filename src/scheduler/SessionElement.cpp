#include <anira/scheduler/SessionElement.h>

namespace anira {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& ppP, InferenceConfig& config) :
    sessionID(newSessionID),
    prePostProcessor(ppP),
    inferenceConfig(config)
{
}

} // namespace anira