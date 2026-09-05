#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/scheduler/Core.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

using namespace anira;

// Reproduces the unsynchronized session-lifecycle defect: the core
// singleton's static state (m_sessions vector, shared inference thread pool,
// the m_core shared_ptr, and the "last session releases the pool" teardown
// in release_session) is mutated by get_instance / create_session /
// release_session / prepare_session without any serialization. Two handler
// lifecycles overlapping on different threads therefore corrupt that state:
// under ThreadSanitizer this manifests as races on m_sessions and the
// SessionElement shared_ptrs, both releasers entering the pool teardown
// through the decrement-then-check window, and SEGVs (double-release in
// __shared_count::__release_shared; null thread handle in HighPriorityThread
// teardown). Without a sanitizer the same corruption can surface as
// intermittent crashes.
//
// Real-world trigger: a host application creating/destroying two anira-based
// plugin instances concurrently or in quick succession — e.g. a DAW loading or
// closing a session with several instances, or moving/duplicating an insert
// (Pro Tools does both from worker threads).

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

// One "plugin instance": config + processor + handler, prepared on
// construction so the session is registered with the shared core.
struct Instance {
    Instance() { m_handler.prepare(HostConfig(512, 48000)); }

    InferenceConfig m_inference_config = make_inference_config();
    PrePostProcessor m_pp_processor{m_inference_config};
    InferenceHandler m_handler{m_pp_processor, m_inference_config, CoreConfig(2)};
};

// Spin gate so two threads hit their critical action as close to
// simultaneously as the scheduler allows.
class StartGate {
public:
    void arrive_and_wait() {
        m_arrived.fetch_add(1, std::memory_order_acq_rel);
        while (!m_go.load(std::memory_order_acquire)) {}
    }
    void wait_for_arrivals(int n) {
        while (m_arrived.load(std::memory_order_acquire) < n) { std::this_thread::yield(); }
    }
    void open() { m_go.store(true, std::memory_order_release); }

private:
    std::atomic<int> m_arrived{0};
    std::atomic<bool> m_go{false};
};

constexpr int k_targeted_iterations = 25;
constexpr int k_churn_iterations = 50;

}  // namespace

// Two live sessions released at the same instant: both release_session calls
// race the m_sessions erase, and the fetch_sub / "== 0" teardown window lets
// both threads tear down the shared thread pool and the singleton.
TEST(ConcurrentLifecycleTest, ConcurrentDestroy) {
    for (int i = 0; i < k_targeted_iterations; ++i) {
        auto instance_a = std::make_unique<Instance>();
        auto instance_b = std::make_unique<Instance>();

        StartGate gate;
        std::thread thread_a([&] {
            gate.arrive_and_wait();
            instance_a.reset();
        });
        std::thread thread_b([&] {
            gate.arrive_and_wait();
            instance_b.reset();
        });
        gate.wait_for_arrivals(2);
        gate.open();
        thread_a.join();
        thread_b.join();

        ASSERT_EQ(Core::get_num_sessions(), 0)
            << "session bookkeeping corrupted after concurrent destroy, iteration " << i;
        ASSERT_EQ(Core::get_num_inference_threads(), 0u)
            << "inference threads survived the last release, iteration " << i;
        ASSERT_EQ(Core::get_num_inference_threads(), 0u)
            << "inference threads survived the last release, iteration " << i;
    }
}

// The instance-move shape: the last session tearing down the shared pool and
// singleton while a replacement instance rebuilds them on another thread.
TEST(ConcurrentLifecycleTest, DestroyCreateOverlap) {
    for (int i = 0; i < k_targeted_iterations; ++i) {
        auto instance_a = std::make_unique<Instance>();

        StartGate gate;
        std::thread thread_a([&] {
            gate.arrive_and_wait();
            instance_a.reset();
        });
        std::thread thread_b([&] {
            gate.arrive_and_wait();
            const Instance instance_b;
        });
        gate.wait_for_arrivals(2);
        gate.open();
        thread_a.join();
        thread_b.join();

        ASSERT_EQ(Core::get_num_sessions(), 0)
            << "session bookkeeping corrupted after destroy/create overlap, iteration " << i;
        ASSERT_EQ(Core::get_num_inference_threads(), 0u)
            << "inference threads survived the last release, iteration " << i;
        ASSERT_EQ(Core::get_num_inference_threads(), 0u)
            << "inference threads survived the last release, iteration " << i;
    }
}

// Broad randomized overlap: two threads independently churning full instance
// lifecycles, covering create-vs-create, create-vs-destroy and
// destroy-vs-destroy interleavings without staged timing.
TEST(ConcurrentLifecycleTest, ParallelChurn) {
    auto churn = [] {
        for (int i = 0; i < k_churn_iterations; ++i) { const Instance instance; }
    };

    std::thread thread_a(churn);
    std::thread thread_b(churn);
    thread_a.join();
    thread_b.join();

    ASSERT_EQ(Core::get_num_sessions(), 0) << "session bookkeeping corrupted after parallel churn";
    ASSERT_EQ(Core::get_num_inference_threads(), 0u)
        << "inference threads survived the last release";
    ASSERT_EQ(Core::get_num_inference_threads(), 0u)
        << "inference threads survived the last release";
}
