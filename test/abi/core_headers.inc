// The core's C++ headers that are still installed, every one: the scheduler's, the thread's and
// the two utilities theirs include, public until a later pre-release makes them private (the
// 2.x classes the 2.x umbrella included are gone; anira/compat/v2.hpp has their twins).
// Included by the coexistence gates (header_coexist.inc, header_cxx20_with_v2.inc,
// header_compat_with_v2.inc) where the 2.x umbrella stood. Not a header: no include guard, the
// caller's NOLINT region around it.
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/scheduler/LatencyCalculator.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/system/HighPriorityThread.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/Semaphore.h>
