#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

// The pool-lifetime policy the session registry enforces (issue #104): inference
// threads exist exactly while sessions exist. Building the pool is part of
// registering the first session; stopping and joining it is part of unregistering
// the last one — synchronously, inside the same critical section — so a plugin host
// may unload the plugin's library the moment its last InferenceHandler is gone.
// Plus the backstops: shutdown() (the library-unload hook / module-exit call) and
// release_core_if_idle() (the core's memory is reclaimed only when nothing uses it).

namespace {

InferenceConfig make_inference_config() {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        1.f,
        0,
        false,
        0.f,
        2);
}

// One "plugin instance", prepared on construction so its pool threads are running.
struct Instance {
    explicit Instance(const CoreConfig& core_config = CoreConfig(2))
        : m_handler(m_pp_processor, m_inference_config, core_config) {
        m_handler.prepare(HostConfig(512, 48000));
    }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler;
};

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
        const Instance instance(CoreConfig(2));
        EXPECT_TRUE(wait_for_num_inference_threads(2));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
    {
        const Instance instance(CoreConfig(3));
        EXPECT_TRUE(wait_for_num_inference_threads(3));
    }
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
}

// While sessions exist, a later configuration only ever shrinks the pool (and never
// to zero: 0 means "no preference"), it does not grow it — the rule of the previous
// implementation, kept unchanged.
TEST(CoreLifecycleTest, LaterSessionOnlyShrinksThePool) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    const Instance instance_a(CoreConfig(3));
    ASSERT_TRUE(wait_for_num_inference_threads(3));
    {
        const Instance instance_b(CoreConfig(1));
        EXPECT_TRUE(wait_for_num_inference_threads(1));
    }
    EXPECT_TRUE(wait_for_num_inference_threads(1));
    {
        const Instance instance_c(CoreConfig(4));
        EXPECT_TRUE(wait_for_num_inference_threads(1));
    }
    {
        const Instance instance_d(CoreConfig(0));
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

// The deprecated two-step API keeps working for one minor release: the staged
// configuration is the one the session is created with.
TEST(CoreLifecycleTest, DeprecatedGetInstanceStagesConfigForCreateSession) {
    ASSERT_EQ(Core::get_num_sessions(), 0);
    InferenceConfig inference_config = make_inference_config();
    PrePostProcessor pp_processor(inference_config);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    auto core = Core::get_instance(CoreConfig(3));
    ASSERT_NE(core, nullptr);
    auto session = core->create_session(pp_processor, inference_config, nullptr);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(Core::get_num_sessions(), 1);
    core->prepare_session(session, HostConfig(512, 48000));
    EXPECT_TRUE(wait_for_num_inference_threads(3));

    Core::release_session(session);
    EXPECT_EQ(Core::get_num_sessions(), 0);
    EXPECT_EQ(Core::get_num_inference_threads(), 0u);
}
