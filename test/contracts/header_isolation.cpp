// Compiled by the anira_header_isolation CTest with anira's own include directory
// (plus those of its public dependencies) and nothing else on the include path: no
// engine include directory. It fails to compile as soon as a public header includes
// an engine header again — the engine state belongs in each engine adapter's own
// translation unit under src/engines (src/engines/Adapter.h). Every public
// header is covered: the C umbrella, the C++ face, the 2.x compat shim and the core's C++
// headers still installed, listed one by one.
// NOLINTBEGIN(misc-include-cleaner)
#include <anira/anira.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/HighPriorityThread.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/Semaphore.h>

#include <anira/anira.hpp>
#include <anira/compat/v2.hpp>
#include <type_traits>

static_assert(std::is_class_v<anira::RingBuffer>);
static_assert(std::is_class_v<anira::InferenceHandler>);
static_assert(std::is_class_v<anira::v2::InferenceHandler>);
// NOLINTEND(misc-include-cleaner)
