#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <concurrentqueue.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../abi/handler_support.h"
#include "../support/log_record_collector.h"
#include "capi/handles.h"
#include "gtest/gtest.h"
#include "scheduler/Core.h"
#include "scheduler/QueueCapacity.h"
#include "scheduler/SessionElement.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2  // GetProcessMemoryInfo is kernel32's K32GetProcessMemoryInfo
#endif
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

// The core's inference queue (Core::State::m_next_inference) holds what the sessions that exist
// reserve, nothing more (issue #200): it is built with no blocks, each session's prepare gives
// its producer token the blocks for every job of its struct pool (one for an exclusive session)
// through one enqueue_bulk of placeholders (src/scheduler/QueueCapacity.h), and the drain
// requeues the other sessions' jobs with the allocating enqueue. Checked here: the reservation's
// size at every tail offset, with consumers taking the placeholders as they come; the memory the
// core and the sessions take; a session that fills its whole struct pool with nobody consuming,
// every job run afterwards; and a drain that requeues more than 1,024 jobs of another session.

using namespace anira;

namespace {

constexpr double k_mib = 1024.0 * 1024.0;

/// The process's resident memory in bytes, or -1 where this test cannot read it.
long long resident_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) == 0) { return -1; }
    return static_cast<long long>(counters.WorkingSetSize);
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),  // NOLINT(*-reinterpret-cast) the API's
                  &count) != KERN_SUCCESS) {
        return -1;
    }
    return static_cast<long long>(info.resident_size);
#elif defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long long pages = 0;
    long long resident = 0;
    if (!(statm >> pages >> resident)) { return -1; }
    return resident * static_cast<long long>(sysconf(_SC_PAGESIZE));
#else
    return -1;
#endif
}

/// The process's virtual size in bytes (Linux: VmSize), or -1 elsewhere.
long long virtual_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    long long pages = 0;
    if (!(statm >> pages)) { return -1; }
    return pages * static_cast<long long>(sysconf(_SC_PAGESIZE));
#else
    return -1;
#endif
}

/// A core nobody uses, freed, so that the next user creates it anew; false when another test's
/// objects still hold it (a binary run in one launch).
bool fresh_core() {
    Core::shutdown();
    return Core::release_core_if_idle() || !Core::has_core();
}

/// One mono Streamed tensor of `window` samples on either side, hop = window, on the custom row
/// with `instances` shared instances; stateful when asked (its session is exclusive).
anira::ModelConfig hop_model(int64_t window, uint32_t instances, bool stateful = false) {
    anira::ModelConfig model = anira_test::custom_model(instances);
    model.input(anira_test::stream_tensor("in", 1, window, window));
    model.output(anira_test::stream_tensor("out", 1, window, window));
    if (stateful) { model.state(ANIRA_MODEL_STATEFUL); }
    return model;
}

/// One call of `block` samples of 0.25 through the handler (its output, which is not looked at,
/// goes to the same buffer).
anira_status process_block(anira_handler* handler, size_t block) {
    std::vector<float> samples(block, 0.25F);
    const std::array<float*, 1> channels{samples.data()};
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(block)};
    anira_tensor io{};
    anira_tensor_init_host_planar(&io,
                                  static_cast<const void*>(channels.data()),
                                  1,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    size_t delivered = 0;
    return anira_handler_process(handler, &io, 0, &io, 0, &delivered);
}

/// A thread of the host's own (the context has no pool) that the test steps by hand: runs every
/// job the queue holds, on the calling thread, and returns how many it ran.
struct HostThread {
    explicit HostThread(const anira_test::Context& context) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_inference_thread_create(context.m_context, &m_thread, &err), ANIRA_OK)
            << err.message;
    }
    ~HostThread() { anira_inference_thread_destroy(m_thread); }
    HostThread(const HostThread&) = delete;
    HostThread& operator=(const HostThread&) = delete;
    HostThread(HostThread&&) = delete;
    HostThread& operator=(HostThread&&) = delete;

    size_t run_queue() const {
        size_t ran = 0;
        while (anira_inference_thread_execute(m_thread) != 0U) { ++ran; }
        return ran;
    }

    anira_inference_thread* m_thread = nullptr;
};

/// A gate engine that counts the inferences inside its process at once.
class InsideGate : public anira_test::GateEngine {
public:
    anira_status process(const anira_engine_ctx& ctx) noexcept override {
        m_inside.fetch_add(1);
        const anira_status status = GateEngine::process(ctx);
        m_inside.fetch_sub(1);
        return status;
    }

    /// Waits until `count` inferences are inside, at most 10 s; false at the bound.
    bool wait_inside(int count) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (m_inside.load() < count) {
            if (std::chrono::steady_clock::now() > deadline) { return false; }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return true;
    }

    std::atomic<int> m_inside{0};
};

}  // namespace

// ---- the reservation's size -------------------------------------------------------------------

TEST(InferenceQueue, TheReservationSizeIsOneBlockPerThirtyTwoBeyondTheFirstItem) {
    constexpr size_t k_block = InferenceQueue::BLOCK_SIZE;
    EXPECT_EQ(k_block, 32U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(1), 1U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(2), 33U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(33), 33U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(34), 65U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(512), 513U);
    EXPECT_EQ(detail::reservation_size<InferenceQueue>(513), 513U);
}

// A producer reserved for `capacity` items takes that many at every tail offset by try_enqueue,
// which never allocates. The placeholders are reserved while two threads dequeue everything they
// find, the way a running pool takes jobs: the bulk enqueue links every block before a consumer
// can take one. One placeholder fewer leaves a ring that some offset overflows.
TEST(InferenceQueue, AReservedProducerTakesItsCapacityAtEveryTailOffset) {
    for (const size_t capacity : {size_t{1},
                                  size_t{2},
                                  size_t{4},
                                  size_t{32},
                                  size_t{33},
                                  size_t{34},
                                  size_t{65},
                                  size_t{512},
                                  size_t{513}}) {
        InferenceQueue queue(0);
        moodycamel::ProducerToken token(queue);
        std::atomic<bool> stop{false};
        std::atomic<size_t> taken{0};
        std::vector<std::thread> consumers;
        for (int i = 0; i < 2; ++i) {
            consumers.emplace_back([&queue, &stop, &taken] {
                InferenceData data;
                while (!stop.load()) {
                    if (queue.try_dequeue(data)) { taken.fetch_add(1); }
                }
            });
        }
        const size_t count = detail::reservation_size<InferenceQueue>(capacity);
        ASSERT_TRUE(detail::reserve_capacity(queue, token, count, InferenceData{}));
        while (taken.load() < count) { std::this_thread::yield(); }
        stop.store(true);
        for (std::thread& consumer : consumers) { consumer.join(); }

        // Every tail offset: a round of `capacity` items, then one item more shifts the tail.
        size_t failures = 0;
        InferenceData data;
        for (size_t round = 0; round < InferenceQueue::BLOCK_SIZE + 1; ++round) {
            for (size_t i = 0; i < capacity; ++i) {
                if (!queue.try_enqueue(token, InferenceData{})) { ++failures; }
            }
            while (queue.try_dequeue(data)) {}
            if (!queue.try_enqueue(token, InferenceData{})) { ++failures; }
            while (queue.try_dequeue(data)) {}
        }
        EXPECT_EQ(failures, 0U) << "capacity " << capacity << ", " << count << " placeholders";

        if (capacity < 2) { continue; }
        InferenceQueue short_queue(0);
        moodycamel::ProducerToken short_token(short_queue);
        ASSERT_TRUE(detail::reserve_capacity(short_queue, short_token, count - 1, InferenceData{}));
        while (short_queue.try_dequeue(data)) {}
        size_t short_failures = 0;
        for (size_t round = 0; round < InferenceQueue::BLOCK_SIZE + 1; ++round) {
            for (size_t i = 0; i < capacity; ++i) {
                if (!short_queue.try_enqueue(short_token, InferenceData{})) { ++short_failures; }
            }
            while (short_queue.try_dequeue(data)) {}
            if (!short_queue.try_enqueue(short_token, InferenceData{})) { ++short_failures; }
            while (short_queue.try_dequeue(data)) {}
        }
        EXPECT_GT(short_failures, 0U) << "capacity " << capacity << ": " << count - 1
                                      << " placeholders were enough, the size is not the least";
    }
}

// ---- memory ------------------------------------------------------------------------------------

// The core costs little resident memory: issue #200 measured 329 MiB at the first context (the
// queue's 314,328 preallocated blocks); the core without a queue of its own takes about 1 MiB.
// The bound leaves room for Debug and the sanitizer trees. register_context alone: no engine is
// probed, so the engines of the build do not count.
TEST(InferenceQueue, TheCoreTakesLittleResidentMemory) {
    if (!fresh_core()) { GTEST_SKIP() << "another test's objects hold the core"; }
    const long long before = resident_bytes();
    if (before < 0) { GTEST_SKIP() << "no resident-memory reading on this platform"; }
    anira_context_config* config = nullptr;
    ASSERT_EQ(anira_context_config_create(&config, nullptr), ANIRA_OK);
    ASSERT_EQ(anira_context_config_set_log_drain(config, ANIRA_LOG_DRAIN_MANUAL, 10), ANIRA_OK);
    Core::register_context(*config);
    const long long after = resident_bytes();
    Core::unregister_context();
    anira_context_config_destroy(config);
    const double taken = static_cast<double>(after - before) / k_mib;
    std::printf("the core takes %.2f MiB resident\n", taken);
    EXPECT_LT(taken, 16.0);
}

// A session's producer reserves its blocks, not a block index sized for a 10M-slot pool (8 MiB
// of virtual memory per session before #200): 64 sessions beside a running one cost less than
// 64 MiB of address space. Linux: VmSize.
TEST(InferenceQueue, SixtyFourSessionsTakeLittleVirtualMemory) {
    if (virtual_bytes() < 0) { GTEST_SKIP() << "no virtual-size reading on this platform"; }
    const anira_test::Context context(2);
    anira_test::Handler first(context, hop_model(512, 2));
    ASSERT_EQ(first.prepare(anira_test::explicit_contract()), ANIRA_OK) << first.m_err.message;
    const long long before = virtual_bytes();
    std::vector<std::unique_ptr<anira_test::Handler>> handlers;
    for (int i = 0; i < 64; ++i) {
        handlers.push_back(std::make_unique<anira_test::Handler>(context, hop_model(512, 2)));
        ASSERT_EQ(handlers.back()->prepare(anira_test::explicit_contract()), ANIRA_OK)
            << handlers.back()->m_err.message;
    }
    const long long after = virtual_bytes();
    for (int block = 0; block < 8; ++block) {
        for (const auto& handler : handlers) {
            const anira_status status = process_block(handler->m_handler, anira_test::k_block);
            EXPECT_TRUE(status == ANIRA_OK || status == ANIRA_MISSED) << status;
        }
    }
    handlers.clear();
    const double taken = static_cast<double>(after - before) / k_mib;
    std::printf("64 sessions take %.1f MiB of address space\n", taken);
    EXPECT_LT(taken, 64.0);
}

// ---- capacity ----------------------------------------------------------------------------------

// A stateless session with a struct pool in the hundreds fills every struct with nobody
// consuming (the context has no pool): every job is in the queue at once and none is dropped;
// the host's thread then runs every one of them.
TEST(InferenceQueue, AStatelessSessionFillsItsWholePoolWithoutADrop) {
    anira_test::RecordCollector collector;
    InsideGate engine;
    const anira_test::Context context(0);
    {
        anira_test::Handler handler(context, hop_model(64, 1), {}, {}, engine.engine());
        const anira_test::DestroyFirst destroy_first(handler, &engine);
        constexpr uint32_t k_host_block = 8192;
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_host_block,
                                                                anira_test::k_rate,
                                                                ANIRA_MISS_ZEROS,
                                                                0.0,
                                                                /*budget_ms=*/0.5)),
                  ANIRA_OK)
            << handler.m_err.message;
        const uint32_t pool = anira_handler_num_entries(handler.m_handler);
        const uint32_t hops_per_call = k_host_block / 64;
        ASSERT_GE(pool, 4 * hops_per_call) << "the geometry no longer makes a large pool";
        const uint32_t calls = pool / hops_per_call;
        for (uint32_t call = 0; call < calls; ++call) {
            static_cast<void>(process_block(handler.m_handler, k_host_block));
        }
        EXPECT_EQ(engine.m_calls.load(), 0) << "nothing consumes before the host's thread runs";
        const HostThread host(context);
        EXPECT_EQ(host.run_queue(), static_cast<size_t>(calls) * hops_per_call);
        EXPECT_EQ(engine.m_calls.load(), static_cast<int>(calls * hops_per_call))
            << "a job was dropped instead of queued (pool " << pool << ")";
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "Could not enqueue", "rt"), 0U);
#endif
}

// More than 1,024 jobs of one session are queued while another session is prepared: its drain
// takes every one of them out and requeues it with the allocating enqueue (a tokenless
// try_enqueue stopped at 1,024 and zero-filled the rest); the host's thread then runs them all.
TEST(InferenceQueue, ADrainRequeuesMoreThan1024JobsOfAnotherSession) {
    anira_test::RecordCollector collector;
    InsideGate busy_engine;
    const anira_test::Context context(0);
    {
        anira_test::Handler busy(context, hop_model(8, 1), {}, {}, busy_engine.engine());
        const anira_test::DestroyFirst destroy_busy(busy, &busy_engine);
        constexpr uint32_t k_host_block = 8192;
        ASSERT_EQ(busy.prepare(anira_test::explicit_contract(k_host_block,
                                                             anira_test::k_rate,
                                                             ANIRA_MISS_ZEROS,
                                                             0.0,
                                                             /*budget_ms=*/0.05)),
                  ANIRA_OK)
            << busy.m_err.message;
        const uint32_t pool = anira_handler_num_entries(busy.m_handler);
        const uint32_t hops_per_call = k_host_block / 8;
        ASSERT_GE(pool, 2 * hops_per_call) << "the geometry no longer makes a large pool";
        for (int call = 0; call < 2; ++call) {
            static_cast<void>(process_block(busy.m_handler, k_host_block));
        }
        const size_t queued = 2 * static_cast<size_t>(hops_per_call);
        ASSERT_GT(queued, 1024U);

        // Another session's prepare drains: the busy session's jobs go out and back in.
        anira_test::Handler other(context, hop_model(512, 1));
        ASSERT_EQ(other.prepare(anira_test::explicit_contract()), ANIRA_OK) << other.m_err.message;

        const HostThread host(context);
        EXPECT_EQ(host.run_queue(), queued);
        EXPECT_EQ(busy_engine.m_calls.load(), static_cast<int>(queued))
            << "a requeued job was zero-filled instead of run";
        other.destroy();
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "Could not requeue", "rt"), 0U);
#endif
}

// The reservation while a running pool takes every job it finds: the pool's threads take the
// placeholders as they are enqueued, and the session still has its whole pool's room. The pool
// is then parked inside a gated engine of another session, so nothing consumes, and the session
// fills every struct of its pool without a drop.
TEST(InferenceQueue, AReservationHoldsWhileThePoolTakesThePlaceholders) {
    anira_test::RecordCollector collector;
    InsideGate parking;
    anira_test::GateEngine counting;
    const anira_test::Context context(2);
    {
        anira_test::Handler parked(context, hop_model(512, 2), {}, {}, parking.engine());
        const anira_test::DestroyFirst destroy_parked(parked, &parking);
        ASSERT_EQ(parked.prepare(anira_test::explicit_contract(1024)), ANIRA_OK)
            << parked.m_err.message;

        // Reserved while both pool threads poll the empty queue.
        anira_test::Handler filled(context, hop_model(64, 1), {}, {}, counting.engine());
        const anira_test::DestroyFirst destroy_filled(filled, &counting);
        constexpr uint32_t k_host_block = 8192;
        ASSERT_EQ(filled.prepare(anira_test::explicit_contract(k_host_block,
                                                               anira_test::k_rate,
                                                               ANIRA_MISS_ZEROS,
                                                               0.0,
                                                               /*budget_ms=*/0.5)),
                  ANIRA_OK)
            << filled.m_err.message;

        // Park both pool threads inside the other session's engine.
        parking.m_open.store(false);
        static_cast<void>(process_block(parked.m_handler, 1024));
        ASSERT_TRUE(parking.wait_inside(2)) << "the pool did not park";

        const uint32_t pool = anira_handler_num_entries(filled.m_handler);
        const uint32_t hops_per_call = k_host_block / 64;
        const uint32_t calls = pool / hops_per_call;
        for (uint32_t call = 0; call < calls; ++call) {
            static_cast<void>(process_block(filled.m_handler, k_host_block));
        }
        EXPECT_EQ(counting.m_calls.load(), 0);
        parking.m_open.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (counting.m_calls.load() < static_cast<int>(calls * hops_per_call) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(counting.m_calls.load(), static_cast<int>(calls * hops_per_call))
            << "a job was dropped instead of queued (pool " << pool << ")";
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "Could not enqueue", "rt"), 0U);
#endif
}

// ---- the pending queue of an exclusive session (#199)
// --------------------------------------------

namespace {

/// Waits until the engine ran `count` inferences, at most 20 s.
void wait_for_calls(const anira_test::GateEngine& engine, int count) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (engine.m_calls.load() < count && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

}  // namespace

// An exclusive session parks every chunk of a call in its pending queue until the dispatch gate
// lets one go. The queue was sized for the session's instance count (one), so it held 33 to 64
// entries and completed every chunk beyond them as zeros. Reserved for the struct pool, it holds
// the 128 chunks of a call of 8192 samples at hop 64, after a re-prepare mid-stream as well, and
// a whole file of 200 hops of 2048 in one call: every chunk reaches the engine.
TEST(InferenceQueue, AStatefulSessionParksEveryChunkOfALargeCall) {
    anira_test::RecordCollector collector;
    anira_test::GateEngine engine;
    const anira_test::Context context(2);
    {
        anira_test::Handler handler(context,
                                    hop_model(64, 1, /*stateful=*/true),
                                    {},
                                    {},
                                    engine.engine());
        const anira_test::DestroyFirst destroy_first(handler, &engine);
        constexpr uint32_t k_host_block = 8192;
        constexpr int k_hops_per_call = static_cast<int>(k_host_block / 64);
        for (int round = 0; round < 2; ++round) {
            ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_host_block,
                                                                    anira_test::k_rate,
                                                                    ANIRA_MISS_ZEROS,
                                                                    0.0,
                                                                    /*budget_ms=*/0.5)),
                      ANIRA_OK)
                << handler.m_err.message;
            const int before = engine.m_calls.load();
            for (int call = 0; call < 4; ++call) {
                static_cast<void>(process_block(handler.m_handler, k_host_block));
                const int expected = before + ((call + 1) * k_hops_per_call);
                wait_for_calls(engine, expected);
                ASSERT_EQ(engine.m_calls.load(), expected)
                    << "round " << round << ", call " << call << ": a parked chunk was dropped";
            }
        }
    }
    {
        anira_test::GateEngine file_engine;
        anira_test::Handler handler(context,
                                    hop_model(2048, 1, /*stateful=*/true),
                                    {},
                                    {},
                                    file_engine.engine());
        const anira_test::DestroyFirst destroy_first(handler, &file_engine);
        constexpr uint32_t k_file = 200 * 2048;
        ASSERT_EQ(handler.prepare(anira_test::explicit_contract(k_file, anira_test::k_rate)),
                  ANIRA_OK)
            << handler.m_err.message;
        static_cast<void>(process_block(handler.m_handler, k_file));
        wait_for_calls(file_engine, 200);
        EXPECT_EQ(file_engine.m_calls.load(), 200) << "a whole-file call dropped chunks";
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "Could not enqueue pending", "rt"), 0U);
#endif
}

// ---- a dropped chunk ---------------------------------------------------------------------------

// A chunk a queue has no room for completes as zeros at its stream position and is a dropped
// hop: rt_error says ANIRA_ERROR_CAPACITY, as for a hop with no free struct. Forced white-box on
// an exclusive session's pending queue, its one consumer never asked: the same struct parked
// past the room its prepare reserved, which holds the whole struct pool first.
TEST(InferenceQueue, AChunkAQueueHasNoRoomForIsADroppedHop) {
    const anira_test::Context context(0);
    anira_test::Handler handler(context, hop_model(64, 1, /*stateful=*/true));
    ASSERT_EQ(handler.prepare(anira_test::explicit_contract(2048,
                                                            anira_test::k_rate,
                                                            ANIRA_MISS_ZEROS,
                                                            0.0,
                                                            /*budget_ms=*/0.5)),
              ANIRA_OK)
        << handler.m_err.message;
    ASSERT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    const std::shared_ptr<SessionElement> session = anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    const size_t pool = session->m_struct_pool.size();
    const std::shared_ptr<SessionElement::ThreadSafeStruct> chunk = session->m_struct_pool[0];
    size_t parked = 0;
    while (anira_handler_rt_error(handler.m_handler) == ANIRA_OK && parked <= 2 * pool + 64) {
        session->enqueue_pending_dispatch(chunk);
        ++parked;
    }
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_CAPACITY);
    EXPECT_GT(parked, pool) << "the pending queue held fewer entries than the struct pool";
}
