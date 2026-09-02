// Reclaims the context core once every test in the binary has run.
//
// The Context is immortal by design, so at process exit it still owns the inference
// queue and LeakSanitizer reports it — after the tests themselves have passed. Rather
// than suppress that, do what a well-behaved host does on the way out and call the
// public reclaim API, which frees the core only when nothing references it. Registered
// for every test binary by anira_add_test_binary().

#include <anira/scheduler/Context.h>
#include <gtest/gtest.h>

namespace {

class CoreReclaimEnvironment : public ::testing::Environment {
public:
    // Not an assertion: a binary that never created a session has no core to release,
    // and release_core_if_idle() reports that with the same false.
    void TearDown() override { anira::Context::release_core_if_idle(); }
};

// gtest owns the pointer; the binaries use gtest_main, so this registers statically.
const ::testing::Environment* const k_core_reclaim_env =
    ::testing::AddGlobalTestEnvironment(new CoreReclaimEnvironment);

}  // namespace
