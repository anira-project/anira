#include <aari/scheduler/SessionElement.h>

namespace aari {

SessionElement::SessionElement(int newSessionID, PrePostProcessor& ppP, InferenceConfig& config) :
    sessionID(newSessionID),
    prePostProcessor(ppP),
    inferenceConfig(config)
{
}

} // namespace aari