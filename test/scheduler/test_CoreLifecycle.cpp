#include <anira/abi/enums.h>
#include <anira/abi/status.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "../abi/handler_support.h"
#include "capi/handles.h"  // IWYU pragma: keep - the body of anira_context_config
#include "engines/Adapter.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "scheduler/InferenceThread.h"

using namespace anira;

// The pool-lifetime policy the session registry enforces (issue #104): inference
// threads exist exactly while sessions exist. Building the pool is part of
// registering the first session; stopping and joining it is part of unregistering
// the last one — synchronously, inside the same critical section — so a plugin host
// may unload the plugin's library the moment its last handler is gone.
// Plus the backstops: shutdown() (the library-unload hook / module-exit call) and
// release_core_if_idle() (the core's memory is reclaimed only when nothing uses it).

namespace {

// One "plugin instance": a context asking for a number of pool threads and a handler over the
// mono pass-through, prepared on construction so its pool threads are running.
using anira_test::Instance;

// The pool's threads count themselves as active when they enter their loop, i.e.
// asynchronously after prepare(); only the join side (count 0) is synchronous, and
// the tests assert that side without waiting.
bool wait_for_num_inference_threads(unsigned int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (Core::get_num_inference_threads() != expected) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace

// ANIRA_THREADS_AUTO, the default thread count of a context config, is half the hardware
// threads and at least one (Core::default_num_threads), whatever this machine has: the size
// the first registration of a generation reports for its pool before it exists, and the size
// of the pool its first session builds.
TEST(CoreLifecycleTest, AutoThreadsAreHalfTheHardwareThreadsAtLeastOne) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    const unsigned int hardware = std::thread::hardware_concurrency();
    const unsigned int expected = std::max(hardware / 2, 1U);
    EXPECT_EQ(Core::default_num_threads(), expected) << hardware << " hardware threads";

    const anira_context_config config;
    ASSERT_EQ(config.m_num_threads, ANIRA_THREADS_AUTO) << "the config's default";
    EXPECT_EQ(Core::prospective_thread_pool_size(config), expected);
    {
        const Instance instance(ANIRA_THREADS_AUTO);
        EXPECT_EQ(Core::get_thread_pool_size(), expected);
        EXPECT_TRUE(wait_for_num_inference_threads(expected));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0U);
}

// No polling: the threads are joined before the last handler's destructor returns.
TEST(CoreLifecycleTest, LastReleaseJoinsPoolSynchronously) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    {
        const Instance instance_a;
        EXPECT_TRUE(wait_for_num_inference_threads(2));
        EXPECT_TRUE(Core::has_inference_threads());
        {
            const Instance instance_b;
            EXPECT_EQ(Core::get_num_sessions(), 2);
            EXPECT_TRUE(wait_for_num_inference_threads(2));
        }
        // Not the last one: the pool stays.
        EXPECT_EQ(Core::get_num_sessions(), 1);
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    EXPECT_EQ(Core::get_num_sessions(), 0);
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
    EXPECT_FALSE(Core::has_inference_threads());
}

// After the registry emptied, the next first session's configuration is applied
// afresh — the previous generation's does not linger.
TEST(CoreLifecycleTest, NextGenerationAppliesItsOwnConfig) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    {
        const Instance instance(2);
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
    {
        const Instance instance(3);
        EXPECT_TRUE(wait_for_num_inference_threads(3));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
}

// While sessions exist, a later configuration only ever shrinks the pool (and never
// to zero: 0 means "no preference"), it does not grow it — the rule of the previous
// implementation, kept unchanged.
TEST(CoreLifecycleTest, LaterSessionOnlyShrinksThePool) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    const Instance instance_a(3);
    ASSERT_TRUE(wait_for_num_inference_threads(3));
    {
        const Instance instance_b(1);
        EXPECT_TRUE(wait_for_num_inference_threads(1));
    }
    EXPECT_TRUE(wait_for_num_inference_threads(1));
    {
        const Instance instance_c(4);
        EXPECT_TRUE(wait_for_num_inference_threads(1));
    }
    {
        const Instance instance_d(0);
        EXPECT_TRUE(wait_for_num_inference_threads(1));
    }
    EXPECT_EQ(Core::get_num_sessions(), 1);
}

TEST(CoreLifecycleTest, ShutdownIsIdempotentAndNeverAllocates) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    // Start from "no core": with no session, no pool and no user thread the core is
    // idle, so this either frees it or finds none.
    Core::release_core_if_idle();
    ASSERT_FALSE(Core::has_core());

    Core::shutdown();
    EXPECT_FALSE(Core::has_core()) << "shutdown() allocated a core just to find nothing to do";
    Core::shutdown();
    EXPECT_FALSE(Core::has_core());

    // Queries do not allocate either.
    EXPECT_EQ(Core::get_num_sessions(), 0);
    EXPECT_TRUE(Core::get_sessions().empty());
    EXPECT_FALSE(Core::has_inference_threads());
    EXPECT_FALSE(Core::has_core());
}

// The backstop for a host that unloads with a live instance: the pool is joined, the
// session stays registered and is released cleanly afterwards, and the next
// generation gets a fresh pool.
TEST(CoreLifecycleTest, ShutdownWithLiveSessionJoinsPoolAndSessionReleasesCleanly) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    {
        const Instance instance;
        ASSERT_TRUE(wait_for_num_inference_threads(2));

        Core::shutdown();
        EXPECT_EQ(Core::get_num_inference_threads(), 0u);
        EXPECT_FALSE(Core::has_inference_threads());
        EXPECT_EQ(Core::get_num_sessions(), 1);

        Core::shutdown();  // idempotent
        EXPECT_EQ(Core::get_num_sessions(), 1);
    }
    EXPECT_EQ(Core::get_num_sessions(), 0);
    {
        const Instance instance;
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
}

// The core is never destroyed while something uses it, and is re-created on demand.
TEST(CoreLifecycleTest, ReleaseCoreIfIdleFreesOnlyWhenIdle) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    {
        const Instance instance;
        EXPECT_TRUE(Core::has_core());
        EXPECT_FALSE(Core::release_core_if_idle());
        EXPECT_TRUE(Core::has_core());
        EXPECT_EQ(Core::get_num_sessions(), 1);
    }
    // Idle now — but not freed on its own: the core is immortal while the library is
    // loaded; only the unload hook (or this explicit call) reclaims it.
    EXPECT_TRUE(Core::has_core());
    EXPECT_TRUE(Core::release_core_if_idle());
    EXPECT_FALSE(Core::has_core());
    EXPECT_FALSE(Core::release_core_if_idle());

    // Queries without a core answer without allocating one.
    EXPECT_EQ(Core::get_num_sessions(), 0);
    EXPECT_FALSE(Core::has_core());

    {
        const Instance instance;
        EXPECT_TRUE(Core::has_core());
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
}

// A user-managed inference thread references the queue inside the core, so it keeps
// the core alive even with no session registered.
TEST(CoreLifecycleTest, ReleaseCoreIfIdleKeepsCoreWhileUserThreadRuns) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    auto user_thread = Core::make_inference_thread();
    ASSERT_NE(user_thread, nullptr);
    EXPECT_TRUE(Core::has_core());

    user_thread->start();
    ASSERT_TRUE(wait_for_num_inference_threads(1));
    EXPECT_FALSE(Core::release_core_if_idle());
    EXPECT_TRUE(Core::has_core());

    user_thread->stop();
    ASSERT_TRUE(wait_for_num_inference_threads(0));
    user_thread.reset();
    EXPECT_TRUE(Core::release_core_if_idle());
    EXPECT_FALSE(Core::has_core());
}

// The core hands out one engine object per built-in engine while anything holds it: what the
// engine builds at its init (an environment with the level in effect on its logger, a
// runtime's initialisation) is shared by every loaded model of the engine, which hold the
// object, and dies with the last holder, on that holder's thread (never at process exit). The
// same object answers every call while it is held, the next call after the last holder let go
// makes a new one, an engine the build does not carry has none, the objects alone do not keep
// the core, and a freed core knows none of them.
TEST(CoreLifecycleTest, OneEngineObjectPerBuiltInEngineWhileSomethingHoldsIt) {
    Core::release_core_if_idle();
    ASSERT_FALSE(Core::has_core());
    EXPECT_EQ(Core::builtin_engine(ANIRA_ENGINE_NONE), nullptr);
    std::vector<std::shared_ptr<anira::engine::BuiltinEngine>> objects;
    for (const anira_engine engine : {ANIRA_ENGINE_LIBTORCH,
                                      ANIRA_ENGINE_ONNXRUNTIME,
                                      ANIRA_ENGINE_TFLITE,
                                      ANIRA_ENGINE_LITERT,
                                      ANIRA_ENGINE_EXECUTORCH}) {
        std::shared_ptr<anira::engine::BuiltinEngine> object = Core::builtin_engine(engine);
        if (object == nullptr) { continue; }  // not in this build
        EXPECT_EQ(object->kind(), engine);
        EXPECT_FALSE(object->initialised()) << "no session loaded a model of the engine";
        EXPECT_EQ(Core::builtin_engine(engine), object) << "one object per engine while held";
        const std::weak_ptr<anira::engine::BuiltinEngine> watch = object;
        object.reset();
        EXPECT_TRUE(watch.expired()) << "the last holder frees the object";
        objects.push_back(Core::builtin_engine(engine));
        ASSERT_NE(objects.back(), nullptr) << "the next call makes a new one";
        EXPECT_EQ(Core::builtin_engine(engine), objects.back());
    }
    EXPECT_TRUE(Core::has_core()) << "the first call made the core";
    EXPECT_TRUE(Core::release_core_if_idle()) << "the objects alone do not keep the core";
    EXPECT_FALSE(Core::has_core());
    for (const std::shared_ptr<anira::engine::BuiltinEngine>& object : objects) {
        EXPECT_NE(Core::builtin_engine(object->kind()), object)
            << "a freed core knows none of the objects";
    }
    objects.clear();
    Core::release_core_if_idle();
}
