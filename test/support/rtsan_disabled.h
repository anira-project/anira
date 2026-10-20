#ifndef ANIRA_TEST_SUPPORT_RTSAN_DISABLED_H
#define ANIRA_TEST_SUPPORT_RTSAN_DISABLED_H

// RealtimeSanitizer off for the calling thread while an object of this type lives, on an
// ANIRA_WITH_RTSAN build; nothing otherwise. For a test that does something real-time unsafe
// ON PURPOSE inside a nonblocking call (a processor that throws, the live 2.x class whose
// nonblocking declarations wait in non-realtime mode), so that exactly that call is exempt and
// the rest of the test stays checked. Never to silence a finding of the code under test.

#ifdef ANIRA_WITH_RTSAN
#include <sanitizer/rtsan_interface.h>
#endif

namespace anira_test {

#ifdef ANIRA_WITH_RTSAN
using RtsanDisabled = __rtsan::ScopedDisabler;
#else
struct RtsanDisabled {
    RtsanDisabled() = default;
    RtsanDisabled(const RtsanDisabled&) = delete;
    RtsanDisabled& operator=(const RtsanDisabled&) = delete;
    RtsanDisabled(RtsanDisabled&&) = delete;
    RtsanDisabled& operator=(RtsanDisabled&&) = delete;
    ~RtsanDisabled() = default;
};
#endif

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_RTSAN_DISABLED_H
