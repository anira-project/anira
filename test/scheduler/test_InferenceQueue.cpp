#include <anira/abi/config.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/thread.h>
#include <concurrentqueue.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "../support/log_record_collector.h"
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
        consumers.reserve(2);
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

// ---- a refused dispatch and the chunks parked behind it (#252) ---------------------------------

namespace {

using Chunk = SessionElement::ThreadSafeStruct;

/// A gate engine that keeps the first sample of every window it ran, in run order, and passes
/// the window through.
class WindowEngine : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const float* in = anira_tensor_data_f32(&ctx.inputs[0]);
        if (in == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        m_windows.push_back(static_cast<int64_t>(in[0]));
        return anira_test::passthrough(ctx);
    }

    std::vector<int64_t> m_windows;  ///< written by the one thread that steps the jobs
};

/// An exclusive session on a context without a pool, its jobs stepped by hand on a thread of the
/// host's: nothing runs, dispatches or lifts unless the test's own thread does it, so every count
/// below is exact. A call is k_hops hops; the first call writes every sample as its position in
/// the stream plus one (window k starts with k * k_hop + 1, and a zero is no sample of it), every
/// later call zeros.
struct SteppedStatefulStream {
    static constexpr size_t k_hop = 64;
    static constexpr size_t k_hops = 8;
    static constexpr size_t k_call = k_hop * k_hops;

    SteppedStatefulStream() {
        EXPECT_EQ(m_handler.prepare(anira_test::explicit_contract(static_cast<uint32_t>(k_call),
                                                                  anira_test::k_rate,
                                                                  ANIRA_MISS_ZEROS,
                                                                  0.0,
                                                                  /*budget_ms=*/0.5)),
                  ANIRA_OK)
            << m_handler.m_err.message;
        m_session = anira_test::session_of(m_handler.m_handler);
    }

    /// One call of the driving thread; its output block is appended to m_out.
    void call() {
        std::vector<float> block(k_call, 0.0F);
        if (m_out.empty()) {
            for (size_t i = 0; i < k_call; ++i) { block[i] = static_cast<float>(i + 1); }
        }
        const std::array<float*, 1> channels{block.data()};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_call)};
        anira_tensor io{};
        anira_tensor_init_host_planar(&io,
                                      static_cast<const void*>(channels.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        size_t delivered = 0;
        static_cast<void>(anira_handler_process(m_handler.m_handler, &io, 0, &io, 0, &delivered));
        m_out.insert(m_out.end(), block.begin(), block.end());
    }

    /// A call whose chunks all park behind a gate that nobody holds afterwards: the busy bit is
    /// set by hand for the call and the gate then freed without a look at the queue. That is
    /// what a release that left without the look leaves behind, which no path does any more;
    /// the driving thread takes a free gate only, so its dispatch finds a second chunk parked in
    /// this state alone.
    void call_parking_every_chunk() {
        const uint64_t held =
            m_session->m_stateful_dispatch_gate.load() | SessionElement::k_dispatch_busy;
        m_session->m_stateful_dispatch_gate.store(held);
        call();
        m_session->release_dispatch(held);
    }

    /// A call of the driving thread that forms no chunk: it collects, which kicks the chain.
    void kick() const {
        size_t available = 0;
        EXPECT_EQ(anira_handler_get_available_samples(m_handler.m_handler, 0, 0, &available),
                  ANIRA_OK);
    }

    /// The chunks waiting in the pending queue.
    size_t parked() const { return m_session->m_dispatch_pending.size_approx(); }

    /// Takes the parked chunks out, oldest first (white-box: nothing dispatches meanwhile).
    std::vector<std::shared_ptr<Chunk>> take_parked() const {
        std::vector<std::shared_ptr<Chunk>> chunks;
        std::shared_ptr<Chunk> chunk;
        while (m_session->m_dispatch_pending.try_dequeue(chunk)) {
            chunks.push_back(std::move(chunk));
        }
        return chunks;
    }

    bool gate_free() const {
        return (m_session->m_stateful_dispatch_gate.load() & SessionElement::k_dispatch_busy) == 0;
    }

    /// The first sample of window `hop` of the first call.
    static int64_t window(size_t hop) { return static_cast<int64_t>((hop * k_hop) + 1); }

    /// Calls (their jobs run between them) until the first call's hops are out, then the whole
    /// output: the first call's samples at the latency, zeros in the hops of `dropped` and
    /// everywhere else.
    void expect_stream(const std::vector<size_t>& dropped) {
        const size_t latency = anira_handler_get_latency(m_handler.m_handler, 0);
        ASSERT_GE(latency, k_call) << "a hop of the first call would be due inside that call";
        while (m_out.size() < latency + k_call) {
            call();
            static_cast<void>(m_host.run_queue());
        }
        for (size_t i = 0; i < m_out.size(); ++i) {
            float expected = 0.0F;
            if (i >= latency && i - latency < k_call) {
                const size_t hop = (i - latency) / k_hop;
                const bool is_dropped = std::ranges::find(dropped, hop) != dropped.end();
                expected = is_dropped ? 0.0F : static_cast<float>(i - latency + 1);
            }
            ASSERT_EQ(m_out[i], expected)
                << "output sample " << i << " (latency " << latency << ")";
        }
    }

    WindowEngine m_engine;  ///< before the handler, which dies first
    anira_test::Context m_context{0};
    anira_test::Handler m_handler{m_context,
                                  hop_model(static_cast<int64_t>(k_hop), 1, /*stateful=*/true),
                                  {},
                                  {},
                                  m_engine.engine()};
    HostThread m_host{m_context};
    std::shared_ptr<SessionElement> m_session;
    std::vector<float> m_out;  ///< every output block so far, in order
};

/// The inference queue's refusal of a session's dispatches, forced white-box. The queue is the
/// core's own and private to it, and a session's producer in it does not run out of the room its
/// prepare reserved, so the session's producer token is exchanged for the token of a queue of the
/// test's whose producer holds no block: a try_enqueue through that token finds no block ahead
/// and none to take, and never allocates one, on whichever queue it is called (moodycamel's
/// ExplicitProducer::enqueue). lift() exchanges the tokens back.
///
/// The refusal does not end by itself, as a real one does not before a consumer made room. A
/// dispatch that is to be refused once ends it from inside: lifting() gives a parked chunk a
/// pointer whose last owner lifts the refusal, and the last owner of a chunk that is refused is
/// the dispatching thread, behind that chunk's drop and ahead of its next look at the pending
/// queue. Declared behind the session's handler: the destructor lifts ahead of the handler's
/// destroy, which uses the token.
class Refusal {
public:
    explicit Refusal(std::shared_ptr<SessionElement> session)
        : m_state(std::make_shared<State>(std::move(session))) {}
    ~Refusal() {
        m_state->lift();
        m_state->m_lifting.reset();
    }
    Refusal(const Refusal&) = delete;
    Refusal& operator=(const Refusal&) = delete;
    Refusal(Refusal&&) = delete;
    Refusal& operator=(Refusal&&) = delete;

    void lift() const noexcept { m_state->lift(); }
    bool lifted() const noexcept { return m_state->m_session == nullptr; }

    /// `chunk` behind a pointer whose last owner lifts the refusal. That owner may be inside a
    /// real-time call (the driving thread's dispatch), where nothing is freed: this object
    /// keeps the pointer's control block until its own end.
    std::shared_ptr<Chunk> lifting(const std::shared_ptr<Chunk>& chunk) const {
        std::shared_ptr<Chunk> pointer(chunk.get(), [state = m_state, chunk](Chunk*) noexcept {
            state->lift();
        });
        m_state->m_lifting = pointer;
        return pointer;
    }

private:
    struct State {
        explicit State(std::shared_ptr<SessionElement> session) : m_session(std::move(session)) {
            m_session->m_producer_token.swap(m_no_room);
        }

        void lift() noexcept {
            if (m_session == nullptr) { return; }
            m_session->m_producer_token.swap(m_no_room);
            m_session.reset();
        }

        std::shared_ptr<SessionElement> m_session;  ///< none once lifted
        InferenceQueue m_queue{0};                  ///< no block in its pool
        moodycamel::ProducerToken m_no_room{m_queue};
        std::weak_ptr<Chunk> m_lifting;  ///< the control block of lifting()'s pointer
    };

    std::shared_ptr<State> m_state;
};

/// Parks the stream's waiting chunks again in their order, the oldest behind the refusal's
/// lifting pointer: the dispatch of that chunk is refused, and the refusal ends with it. Returns
/// that chunk.
std::shared_ptr<Chunk> refuse_the_oldest_once(const SteppedStatefulStream& stream,
                                              const Refusal& refusal) {
    const std::vector<std::shared_ptr<Chunk>> parked = stream.take_parked();
    for (size_t i = 0; i < parked.size(); ++i) {
        stream.m_session->enqueue_pending_dispatch(i == 0 ? refusal.lifting(parked[i]) : parked[i]);
    }
    return parked.empty() ? nullptr : parked.front();
}

}  // namespace

// An inference thread that finished a chunk dispatches the next parked one. When the inference
// queue refuses that one, it is a dropped hop: zeros at its stream position, ANIRA_ERROR_CAPACITY
// in rt_error. The thread then looks at the pending queue again, and the chunk behind the refused
// one goes to the queue; it used to free the gate and leave, so the chunks behind stayed parked
// until the driving thread's next call. Here the driving thread makes no further call: one job
// runs chunk 0 and its thread dispatches (chunk 1 refused, once; chunk 2 handed over), and the
// host's thread then runs every chunk behind the refused one.
TEST(InferenceQueue, AnInferenceThreadDispatchesTheChunkBehindARefusedOne) {
    SteppedStatefulStream stream;
    ASSERT_NE(stream.m_session, nullptr);
    constexpr size_t k_hops = SteppedStatefulStream::k_hops;
    stream.call();
    ASSERT_EQ(stream.parked(), k_hops - 1) << "chunk 0 is queued, the others wait behind it";
    {
        const Refusal refusal(stream.m_session);
        const std::shared_ptr<Chunk> refused = refuse_the_oldest_once(stream, refusal);
        ASSERT_NE(refused, nullptr);
        ASSERT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_OK);

        ASSERT_NE(anira_inference_thread_execute(stream.m_host.m_thread), 0U);
        EXPECT_EQ(stream.m_engine.m_windows, std::vector<int64_t>{stream.window(0)});
        EXPECT_TRUE(refusal.lifted()) << "the dispatch of chunk 1 was not refused";
        EXPECT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_ERROR_CAPACITY);
        EXPECT_TRUE(refused->m_completed_as_zeros);
        EXPECT_EQ(stream.parked(), k_hops - 3) << "chunk 2 was not handed to the queue";
    }
    EXPECT_EQ(stream.m_host.run_queue(), k_hops - 2)
        << "the chunks behind the refused one wait for a call of the driving thread";
    std::vector<int64_t> ran{stream.window(0)};
    for (size_t hop = 2; hop < k_hops; ++hop) { ran.push_back(stream.window(hop)); }
    EXPECT_EQ(stream.m_engine.m_windows, ran);
    EXPECT_TRUE(stream.gate_free());
    stream.expect_stream({1});
}

// The same on the driving thread, in the one state in which its dispatch has a chunk behind the
// one it takes (SteppedStatefulStream::call_parking_every_chunk). One kick: chunk 0 is refused,
// once, and chunk 1 goes to the queue; the kick used to free the gate behind the refused chunk
// and return, and every chunk behind it waited for the next call.
TEST(InferenceQueue, TheDrivingThreadDispatchesTheChunkBehindARefusedOne) {
    SteppedStatefulStream stream;
    ASSERT_NE(stream.m_session, nullptr);
    constexpr size_t k_hops = SteppedStatefulStream::k_hops;
    stream.call_parking_every_chunk();
    ASSERT_EQ(stream.parked(), k_hops);
    ASSERT_TRUE(stream.gate_free());
    {
        const Refusal refusal(stream.m_session);
        const std::shared_ptr<Chunk> refused = refuse_the_oldest_once(stream, refusal);
        ASSERT_NE(refused, nullptr);
        ASSERT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_OK);

        stream.kick();
        EXPECT_TRUE(refusal.lifted()) << "the dispatch of chunk 0 was not refused";
        EXPECT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(stream.parked(), k_hops - 2) << "chunk 1 was not handed to the queue";
    }
    EXPECT_TRUE(stream.m_engine.m_windows.empty());
    EXPECT_EQ(stream.m_host.run_queue(), k_hops - 1)
        << "the chunks behind the refused one wait for the next call of the driving thread";
    std::vector<int64_t> ran;
    for (size_t hop = 1; hop < k_hops; ++hop) { ran.push_back(stream.window(hop)); }
    EXPECT_EQ(stream.m_engine.m_windows, ran);
    EXPECT_TRUE(stream.gate_free());
    stream.expect_stream({0});
}

// While the queue goes on refusing, every parked chunk is a dropped hop in its turn, in stream
// order, and the dispatch ends with the pending queue empty and the gate free: nothing waits
// behind a free gate, and each refused chunk gave its count on the session back. On an inference
// thread (behind chunk 0, which ran) and on the driving thread (the state made by hand, as
// above); each used to drop one chunk and leave the others parked.
TEST(InferenceQueue, AnInferenceThreadDropsEveryParkedChunkWhileTheQueueRefuses) {
    SteppedStatefulStream stream;
    ASSERT_NE(stream.m_session, nullptr);
    constexpr size_t k_hops = SteppedStatefulStream::k_hops;
    stream.call();
    ASSERT_EQ(stream.parked(), k_hops - 1);
    std::vector<size_t> dropped;
    {
        const Refusal refusal(stream.m_session);
        ASSERT_NE(anira_inference_thread_execute(stream.m_host.m_thread), 0U);
        EXPECT_FALSE(refusal.lifted());
        EXPECT_EQ(stream.parked(), 0U) << "a chunk stayed parked behind the refused one";
        EXPECT_TRUE(stream.gate_free());
        EXPECT_EQ(stream.m_session->m_active_inferences->load(), 0);
        EXPECT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_ERROR_CAPACITY);
        // Chunk k of the first call is struct k of the pool.
        for (size_t hop = 1; hop < k_hops; ++hop) {
            EXPECT_TRUE(stream.m_session->m_struct_pool[hop]->m_completed_as_zeros) << hop;
            dropped.push_back(hop);
        }
    }
    EXPECT_EQ(stream.m_host.run_queue(), 0U);
    EXPECT_EQ(stream.m_engine.m_windows, std::vector<int64_t>{stream.window(0)});
    stream.expect_stream(dropped);
}

TEST(InferenceQueue, TheDrivingThreadDropsEveryParkedChunkWhileTheQueueRefuses) {
    SteppedStatefulStream stream;
    ASSERT_NE(stream.m_session, nullptr);
    constexpr size_t k_hops = SteppedStatefulStream::k_hops;
    stream.call_parking_every_chunk();
    ASSERT_EQ(stream.parked(), k_hops);
    ASSERT_TRUE(stream.gate_free());
    std::vector<size_t> dropped;
    {
        const Refusal refusal(stream.m_session);
        stream.kick();
        EXPECT_FALSE(refusal.lifted());
        EXPECT_EQ(stream.parked(), 0U) << "a chunk stayed parked behind the refused one";
        EXPECT_TRUE(stream.gate_free());
        EXPECT_EQ(stream.m_session->m_active_inferences->load(), 0);
        EXPECT_EQ(anira_handler_rt_error(stream.m_handler.m_handler), ANIRA_ERROR_CAPACITY);
        for (size_t hop = 0; hop < k_hops; ++hop) { dropped.push_back(hop); }
    }
    EXPECT_EQ(stream.m_host.run_queue(), 0U);
    EXPECT_TRUE(stream.m_engine.m_windows.empty());
    stream.expect_stream(dropped);
}
