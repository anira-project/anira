// The 2.x C++ headers that are still installed, every one: what the 2.x umbrella included before
// <anira/anira.h> became the C umbrella (the 2.x handler is gone). Included by the coexistence
// gates (header_coexist.inc, header_cxx20_with_v2.inc, header_compat_with_v2.inc) where the 2.x
// umbrella stood. Not a header: no include guard, the caller's NOLINT region around it.
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/HighPriorityThread.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/Semaphore.h>
