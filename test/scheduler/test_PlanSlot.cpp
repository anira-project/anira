// The plan slot a chunk carries. A session holds its plans as objects of their own
// (SessionElement::m_plans, by pointer). When a chunk is formed it is stamped with its plan's
// index, with the plan's slot and with a copy of what the slot reports (SessionElement::
// stamp_plan); from then on no step of the chunk reads the table: the inference thread uses the
// slot, the processor's phases the copies.
//
// A slot counts the chunks whose worker side is not finished (PlanSlot::active_chunks): up at
// the stamp, down exactly once when no inference thread touches the chunk or the slot any more,
// on every path a chunk can take. The count never waits for the driving thread: a chunk whose
// inference finished does not count, collected or not. These tests drive each path and read the
// count; they wait for a count or an order, never for a time.
//
// The sessions are C handlers' over rows on two gate engines: a chunk is held inside an engine's
// call while its gate is closed.

#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "gtest/gtest.h"
#include "scheduler/InferenceManager.h"
#include "scheduler/InstanceWaitList.h"
#include "scheduler/SessionElement.h"

using namespace anira;

namespace {

using Chunk = SessionElement::ThreadSafeStruct;
using Slot = SessionElement::PlanSlot;

constexpr size_t k_block = 512;
constexpr auto k_wait = std::chrono::seconds(20);
constexpr const char* k_engine_a = "org.example.slot_a";
constexpr const char* k_engine_b = "org.example.slot_b";
/// What engine A serves: the CPU path and a custom provider, two plans of one row.
constexpr const char* k_npu = "com.example.slot_npu";
constexpr std::array<const char*, 2> k_served{"cpu", k_npu};

/// Polls until `holds` answers true; false after k_wait. The tests wait for a count or an order
/// with it, and a passing test never waits long.
template <class Predicate>
bool eventually(Predicate holds) {
    const auto deadline = std::chrono::steady_clock::now() + k_wait;
    while (!holds()) {
        if (std::chrono::steady_clock::now() >= deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
}

/// A gate engine that also counts the calls that entered it (GateEngine::m_calls counts the
/// calls that left it): entered above left means a chunk is inside the engine.
class EnteredGate : public anira_test::GateEngine {
public:
    anira_status process(const anira_engine_ctx& ctx) noexcept override {
        m_entered.fetch_add(1);
        return GateEngine::process(ctx);
    }

    std::atomic<int> m_entered{0};
};

/// An engine object under `id`, over a gate engine's descriptor, serving `providers` (none: the
/// CPU path alone); the pipelines it is added to hold it.
anira_custom_engine* engine_under(anira_test::GateEngine& gate,
                                  const char* id,
                                  std::span<const char* const> providers = {}) {
    anira_engine_desc desc = gate.desc();
    desc.providers = providers.empty() ? nullptr : providers.data();
    desc.num_providers = static_cast<uint32_t>(providers.size());
    anira_custom_engine* engine = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_custom_engine_create(id, &desc, &engine, &err), ANIRA_OK) << err.message;
    return engine;
}

/// Engine A on the CPU path and on k_npu: two plans of one row.
std::vector<anira_backend_id> two_providers_of_a() {
    return {{.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_CUSTOM,
             .provider = ANIRA_PROVIDER_CPU,
             .engine_id = k_engine_a},
            {.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_CUSTOM,
             .provider = ANIRA_PROVIDER_CUSTOM,
             .engine_id = k_engine_a,
             .provider_id = k_npu}};
}

/// What a rig is built from. One plan per row under the default candidate set.
struct RigOptions {
    std::vector<const char*> m_rows{k_engine_a};        ///< each row's engine, in order
    std::vector<anira_backend_id> m_candidates;         ///< none: the default set
    uint32_t m_instances = 2;                           ///< the model's shared instances
    bool m_stateful = false;                            ///< an exclusive session
    uint32_t m_threads = 2;                             ///< the context's pool
    uint32_t m_warmup_iterations = 0;                   ///< the contract's fixed warm-up
    const anira_processor_desc* m_processor = nullptr;  ///< none: the default processor
};

/// A handler over a mono 512 pass-through with the gate engines A and B on its pipeline,
/// prepared under an explicit budget that gives the session room for several chunks at once.
class SlotRig {
public:
    explicit SlotRig(const RigOptions& options = {}) : m_context(options.m_threads) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        anira_custom_engine* const a = engine_under(m_a, k_engine_a, k_served);
        anira_custom_engine* const b = engine_under(m_b, k_engine_b);
        anira_test::add_engine(m_pipeline, a);
        anira_test::add_engine(m_pipeline, b);
        anira_custom_engine_destroy(a);
        anira_custom_engine_destroy(b);

        anira::ModelConfig model;
        for (size_t i = 0; i < options.m_rows.size(); ++i) {
            model.add_model_path(options.m_rows[i], "row-" + std::to_string(i));
        }
        model.input(anira_test::stream_tensor("in", 1, k_block, k_block));
        model.output(anira_test::stream_tensor("out", 1, k_block, k_block));
        model.max_instances(options.m_instances);
        if (options.m_stateful) { model.state(ANIRA_MODEL_STATEFUL); }

        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(
                      m_pipeline,
                      variants.data(),
                      1,
                      options.m_candidates.empty() ? nullptr : options.m_candidates.data(),
                      static_cast<uint32_t>(options.m_candidates.size()),
                      &err),
                  ANIRA_OK)
            << err.message;
        if (options.m_processor != nullptr) {
            EXPECT_EQ(anira_pipeline_add_processor(m_pipeline, options.m_processor, &err), ANIRA_OK)
                << err.message;
        }
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err), ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_prepare(m_handler,
                                        anira_test::explicit_contract(k_block,
                                                                      anira_test::k_rate,
                                                                      ANIRA_MISS_ZEROS,
                                                                      0.0,
                                                                      /*budget_ms=*/40.0,
                                                                      options.m_warmup_iterations)
                                            .native(),
                                        &err),
                  ANIRA_OK)
            << err.message;
    }
    ~SlotRig() {
        destroy_handler();
        anira_pipeline_destroy(m_pipeline);
    }
    SlotRig(const SlotRig&) = delete;
    SlotRig& operator=(const SlotRig&) = delete;
    SlotRig(SlotRig&&) = delete;
    SlotRig& operator=(SlotRig&&) = delete;

    anira_handler* handler() const { return m_handler; }
    InferenceManager& manager() const { return *m_handler->m_manager; }
    std::shared_ptr<SessionElement> session() const { return anira_test::session_of(m_handler); }
    /// The slot of a plan, held: it outlives the session for as long as the caller keeps it.
    std::shared_ptr<Slot> slot(size_t plan) const { return session()->m_plans.at(plan); }

    /// Opens both gates and destroys the handler (the session's release); idempotent.
    void destroy_handler() {
        m_a.m_open.store(true);
        m_b.m_open.store(true);
        anira_handler_destroy(m_handler);
        m_handler = nullptr;
    }

    /// The handler's destroy alone, for a test that holds a gate closed across it.
    void destroy_handler_behind_the_gates() {
        anira_handler_destroy(m_handler);
        m_handler = nullptr;
    }

    /// One block through anira_handler_process, in place: one hop, so one chunk.
    void process_block() {
        std::vector<float> io(k_block, 0.5F);
        const std::array<float*, 1> channels{io.data()};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block)};
        anira_tensor block{};
        anira_tensor_init_host_planar(&block,
                                      static_cast<const void*>(channels.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        size_t delivered = 0;
        static_cast<void>(anira_handler_process(m_handler, &block, 0, &block, 0, &delivered));
    }

    EnteredGate m_a;  ///< before the handler, which dies first
    EnteredGate m_b;

private:
    anira_test::Context m_context;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
};

/// A helper thread joined at scope exit, behind the gate that may hold what it waits for.
struct JoinedBehind {
    JoinedBehind(std::thread thread, anira_test::GateEngine& gate)
        : m_thread(std::move(thread)), m_gate(gate) {}
    ~JoinedBehind() { join(); }
    JoinedBehind(const JoinedBehind&) = delete;
    JoinedBehind& operator=(const JoinedBehind&) = delete;
    JoinedBehind(JoinedBehind&&) = delete;
    JoinedBehind& operator=(JoinedBehind&&) = delete;

    void join() {
        m_gate.m_open.store(true);
        if (m_thread.joinable()) { m_thread.join(); }
    }

    std::thread m_thread;
    anira_test::GateEngine& m_gate;
};

/// A free chunk of the session's pool taken, stamped with `plan` and the session's generation,
/// as Core::pre_process forms one, and parked in the exclusive session's pending queue without
/// a dispatch: the dispatch gate stays free, which a stream reaches only through a race.
std::shared_ptr<Chunk> park_without_dispatch(SessionElement& session, uint32_t plan) {
    for (const std::shared_ptr<Chunk>& chunk : session.m_struct_pool) {
        if (!chunk->m_free.exchange(false)) { continue; }
        session.stamp_plan(*chunk, plan);
        chunk->m_dispatch_generation = session.m_generation.load();
        session.enqueue_pending_dispatch(chunk);
        return chunk;
    }
    return nullptr;
}

/// The chunks of the pool that are out (not free) and still carry their slot, and the ones that
/// are out with their slot taken.
struct OutChunks {
    std::vector<std::shared_ptr<Chunk>> m_with_slot;
    size_t m_taken = 0;
};

OutChunks out_chunks(const SessionElement& session) {
    OutChunks out;
    for (const std::shared_ptr<Chunk>& chunk : session.m_struct_pool) {
        if (chunk->m_free.load()) { continue; }
        if (chunk->m_slot != nullptr) {
            out.m_with_slot.push_back(chunk);
        } else {
            ++out.m_taken;
        }
    }
    return out;
}

bool no_chunk_carries_a_slot(const SessionElement& session) {
    return std::ranges::all_of(session.m_struct_pool, [](const std::shared_ptr<Chunk>& chunk) {
        return chunk->m_slot == nullptr;
    });
}

// ---- a processor that records the backend every phase is told ---------------------------------

/// The backend a phase's context reports: the two values and the two id strings, as pointers.
struct Backend {
    anira_engine m_engine = ANIRA_ENGINE_NONE;
    anira_provider m_provider = ANIRA_PROVIDER_NONE;
    const char* m_engine_id = nullptr;
    const char* m_provider_id = nullptr;
};

Backend backend_of(const anira_processor_ctx* ctx) noexcept {
    Backend backend;
    static_cast<void>(anira_processor_engine(ctx, &backend.m_engine, &backend.m_engine_id));
    static_cast<void>(anira_processor_provider(ctx, &backend.m_provider, &backend.m_provider_id));
    return backend;
}

/// pre_process and post_process run inside the handler's real-time process call: plain fields of
/// the driving thread, no allocation. The inference thread's hook publishes its record with the
/// count.
struct BackendRecorder {
    Backend m_pre;
    size_t m_num_pre = 0;
    Backend m_post;
    size_t m_num_post = 0;
    Backend m_before;
    std::atomic<int> m_num_before{0};
};

anira_status ANIRA_CALL backend_pre(const anira_processor_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) noexcept {
    auto& recorder = *static_cast<BackendRecorder*>(user_data);
    recorder.m_pre = backend_of(ctx);
    ++recorder.m_num_pre;
    return anira_processor_default_pre_process(ctx);
}

anira_status ANIRA_CALL backend_post(const anira_processor_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    auto& recorder = *static_cast<BackendRecorder*>(user_data);
    recorder.m_post = backend_of(ctx);
    ++recorder.m_num_post;
    return anira_processor_default_post_process(ctx);
}

anira_status ANIRA_CALL backend_before(const anira_processor_ctx* ctx,
                                       void* /*prepared*/,
                                       void* user_data) noexcept {
    auto& recorder = *static_cast<BackendRecorder*>(user_data);
    recorder.m_before = backend_of(ctx);
    recorder.m_num_before.fetch_add(1);
    return ANIRA_OK;
}

anira_processor_desc backend_processor(BackendRecorder& recorder) {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.user_data = &recorder;
    desc.realtime_phases =
        ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS) | ANIRA_PHASE_BIT(ANIRA_PHASE_POST_PROCESS);
    desc.pre_process = backend_pre;
    desc.post_process = backend_post;
    desc.before_inference = backend_before;
    return desc;
}

/// A pre_process that fails every chunk: the chunk is never submitted.
anira_status ANIRA_CALL failing_pre(const anira_processor_ctx* /*ctx*/,
                                    void* /*prepared*/,
                                    void* /*user_data*/) noexcept {
    return ANIRA_ERROR_INVALID_STATE;
}

anira_processor_desc failing_pre_processor() {
    anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
    desc.realtime_phases = ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS);
    desc.pre_process = failing_pre;
    return desc;
}

}  // namespace

// ---- the table and the stamp -------------------------------------------------------------------

// Idle: no slot is needed by anyone. A chunk that ran and was collected leaves no count either.
TEST(PlanSlot, TheCountIsZeroWhileNoChunkIsOut) {
    SlotRig rig({.m_rows = {k_engine_a, k_engine_b}});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_EQ(session->m_plans.size(), 2U);
    EXPECT_NE(session->m_plans[0].get(), session->m_plans[1].get()) << "a slot per plan";
    for (const std::shared_ptr<Slot>& slot : session->m_plans) {
        ASSERT_NE(slot, nullptr);
        EXPECT_EQ(slot->active_chunks(), 0U);
    }

    const size_t before = anira_test::available(rig.handler());
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_calls.load() == 1; }));
    anira_test::wait_for_block(rig.handler(), before);
    ASSERT_TRUE(eventually([&] { return session->m_plans[0]->active_chunks() == 0U; }));
    EXPECT_EQ(session->m_plans[1]->active_chunks(), 0U);
    EXPECT_TRUE(no_chunk_carries_a_slot(*session));
}

// What a stamped chunk carries, read off a chunk that waits in the inference queue behind the
// one inside the engine (one thread, so nothing touches it): the table's own slot object, and
// the slot's backend as copies. The chunk inside the engine has no slot any more: its job took
// it at its start.
TEST(PlanSlot, AStampedChunkCarriesTheTablesSlotAndACopyOfWhatItReports) {
    SlotRig rig({.m_instances = 1, .m_threads = 1});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_GE(session->m_struct_pool.size(), 2U);
    const std::shared_ptr<Slot> slot = rig.slot(0);

    rig.m_a.m_open.store(false);
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    rig.process_block();
    EXPECT_EQ(slot->active_chunks(), 2U) << "one inside the engine, one queued";

    const OutChunks out = out_chunks(*session);
    EXPECT_EQ(out.m_taken, 1U) << "the job inside the engine took its chunk's slot";
    ASSERT_EQ(out.m_with_slot.size(), 1U) << "the queued chunk still carries its slot";
    const Chunk& queued = *out.m_with_slot[0];
    EXPECT_EQ(queued.m_plan, 0U);
    EXPECT_EQ(queued.m_slot, slot.get()) << "the table's slot object, not a copy of it";
    EXPECT_EQ(queued.m_engine, ANIRA_ENGINE_CUSTOM);
    EXPECT_EQ(queued.m_engine, slot->m_engine);
    EXPECT_EQ(queued.m_provider, slot->m_provider);
    EXPECT_EQ(queued.m_engine_id, slot->m_engine_id) << "the session's string, by pointer";
    ASSERT_NE(queued.m_engine_id, nullptr);
    EXPECT_STREQ(queued.m_engine_id, k_engine_a);
    EXPECT_EQ(queued.m_engine_id, session->intern_id(k_engine_a)) << "one copy per string";
    EXPECT_EQ(queued.m_provider_id, nullptr) << "a provider of the enum has no name";
    EXPECT_EQ(queued.m_state_alias, slot->m_state_alias);

    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_calls.load(), 2);
    EXPECT_TRUE(no_chunk_carries_a_slot(*session));
}

// ---- the count, path by path -------------------------------------------------------------------

// The normal path: the count is up exactly while the chunk is the worker side's, here inside
// the engine, and it comes down when the inference ended, with nobody collecting the chunk.
TEST(PlanSlot, AChunkCountsWhileItIsInsideTheEngineAndNotBehindIt) {
    SlotRig rig;
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    const std::shared_ptr<Slot> slot = rig.slot(0);
    ASSERT_EQ(slot->active_chunks(), 0U);

    rig.m_a.m_open.store(false);
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    EXPECT_EQ(slot->active_chunks(), 1U);
    EXPECT_EQ(rig.m_a.m_calls.load(), 0) << "the chunk is held inside the engine";
    EXPECT_EQ(slot->active_chunks(), 1U) << "and counts for as long as it is";

    // No call of the driving thread from here: the count does not wait for a collection.
    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    EXPECT_EQ(out_chunks(*session).m_taken, 1U) << "the chunk is still out, not collected";
}

// Each slot counts its own chunks: a chunk held on plan 0 keeps plan 0's count while chunks of
// plan 1, selected meanwhile, come and go.
TEST(PlanSlot, ASwitchLeavesTheCountOnTheSlotTheChunkWasStampedWith) {
    SlotRig rig({.m_rows = {k_engine_a, k_engine_b}});
    const std::shared_ptr<Slot> slot_a = rig.slot(0);
    const std::shared_ptr<Slot> slot_b = rig.slot(1);

    rig.m_a.m_open.store(false);
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    ASSERT_EQ(anira_handler_set_plan(rig.handler(), 1), ANIRA_OK);
    EXPECT_EQ(slot_a->active_chunks(), 1U) << "a selection moves no count";
    EXPECT_EQ(slot_b->active_chunks(), 0U);

    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_b.m_calls.load() == 1; }));
    ASSERT_TRUE(eventually([&] { return slot_b->active_chunks() == 0U; }));
    EXPECT_EQ(slot_a->active_chunks(), 1U) << "the chunk inside engine A still counts on its slot";

    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot_a->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    EXPECT_EQ(rig.m_b.m_calls.load(), 1);
}

// A chunk whose model has no free instance waits on the model's wait list: it counts while it
// waits, until the thread that holds the instance took it and ran it.
TEST(PlanSlot, AChunkWaitingForAnInstanceCountsUntilItRan) {
    SlotRig rig({.m_instances = 1});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_GE(session->m_struct_pool.size(), 3U);
    const std::shared_ptr<Slot> slot = rig.slot(0);
    ASSERT_NE(slot->m_waiting, nullptr);
    const InstanceWaitList& waiting = *slot->m_waiting;

    rig.m_a.m_open.store(false);
    for (int block = 0; block < 3; ++block) { rig.process_block(); }
    // Settled: one chunk inside the engine and two on the wait list. A thread that parks a chunk
    // counts once more until it is done with the list, so the count may read high for a moment.
    ASSERT_TRUE(
        eventually([&] { return waiting.num_waiting() == 2U && slot->active_chunks() == 3U; }));
    EXPECT_EQ(rig.m_a.m_entered.load(), 1);

    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_calls.load(), 3) << "the waiting chunks ran";
    EXPECT_EQ(waiting.num_waiting(), 0U);
}

// A reset while a chunk is held inside the engine makes the chunks behind it stale: the
// inference thread skips them without an engine call, and each gives its count back there.
// Once with the chunks behind it on the model's wait list (two threads, one instance), once in
// the inference queue (one thread).
TEST(PlanSlot, AStaleChunkAnInferenceThreadSkipsGivesItsCountBack) {
    for (const uint32_t threads : {2U, 1U}) {
        SCOPED_TRACE("threads " + std::to_string(threads));
        SlotRig rig({.m_instances = 1, .m_threads = threads});
        const std::shared_ptr<SessionElement> session = rig.session();
        ASSERT_NE(session, nullptr);
        ASSERT_GE(session->m_struct_pool.size(), 3U);
        const std::shared_ptr<Slot> slot = rig.slot(0);

        rig.m_a.m_open.store(false);
        rig.process_block();
        ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
        rig.process_block();
        rig.process_block();
        if (threads > 1) {
            ASSERT_NE(slot->m_waiting, nullptr);
            const InstanceWaitList& waiting = *slot->m_waiting;
            ASSERT_TRUE(eventually(
                [&] { return waiting.num_waiting() == 2U && slot->active_chunks() == 3U; }));
        } else {
            EXPECT_EQ(slot->active_chunks(), 3U) << "one inside the engine, two queued";
        }

        anira_handler_reset(rig.handler());
        EXPECT_EQ(slot->active_chunks(), 3U) << "a reset waits for nobody and frees no slot";

        rig.m_a.m_open.store(true);
        ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
        EXPECT_EQ(rig.m_a.m_entered.load(), 1) << "a stale chunk reached the engine";
    }
}

// A chunk whose pre_process fails is never submitted: it completes as zeros on the driving
// thread, and its count is back when the call returns.
TEST(PlanSlot, AFailedPreProcessGivesTheCountBackAtOnce) {
    const anira_processor_desc processor = failing_pre_processor();
    SlotRig rig({.m_processor = &processor});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    const std::shared_ptr<Slot> slot = rig.slot(0);

    rig.process_block();
    EXPECT_EQ(slot->active_chunks(), 0U);
    EXPECT_EQ(anira_handler_rt_error(rig.handler()), ANIRA_ERROR_INVALID_STATE);
    // The chunk was stamped (it carries plan 0's backend) and its slot taken off it again.
    const bool stamped =
        std::ranges::any_of(session->m_struct_pool, [](const std::shared_ptr<Chunk>& chunk) {
            return chunk->m_engine == ANIRA_ENGINE_CUSTOM && chunk->m_completed_as_zeros;
        });
    EXPECT_TRUE(stamped);
    EXPECT_TRUE(no_chunk_carries_a_slot(*session));
    EXPECT_EQ(rig.m_a.m_entered.load(), 0) << "no inference thread got the chunk";
}

// An engine that fails the chunk: the job ends like any other, and so does its count.
TEST(PlanSlot, AFailingEngineGivesTheCountBack) {
    SlotRig rig;
    const std::shared_ptr<Slot> slot = rig.slot(0);

    rig.m_a.m_fail.store(true);
    rig.m_a.m_open.store(false);
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    EXPECT_EQ(slot->active_chunks(), 1U);

    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_calls.load(), 1);
    EXPECT_EQ(anira_handler_rt_error(rig.handler()), ANIRA_ERROR_ENGINE);
}

// An exclusive session parks its chunks in its pending queue behind the one in flight: they
// count there. A reset while the dispatch gate is held leaves them to the inference thread,
// which filters the stale ones out at its next task boundary and gives their counts back.
TEST(PlanSlot, PendingChunksFilteredAsStaleGiveTheirCountsBack) {
    SlotRig rig({.m_instances = 1, .m_stateful = true});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_TRUE(session->exclusive());
    ASSERT_GE(session->m_struct_pool.size(), 3U);
    const std::shared_ptr<Slot> slot = rig.slot(0);

    rig.m_a.m_open.store(false);
    for (int block = 0; block < 3; ++block) { rig.process_block(); }
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    EXPECT_EQ(slot->active_chunks(), 3U) << "one inside the engine, two pending";
    EXPECT_EQ(out_chunks(*session).m_with_slot.size(), 2U);

    anira_handler_reset(rig.handler());
    EXPECT_EQ(slot->active_chunks(), 3U) << "the gate is held: the reset leaves the queue alone";

    rig.m_a.m_open.store(true);
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_entered.load(), 1) << "a stale pending chunk reached the engine";
    EXPECT_TRUE(no_chunk_carries_a_slot(*session));
}

// The reset's own discard, which runs when no chunk is in flight (the gate is free): every
// pending chunk goes back to the free pool on the driving thread, its count with it.
TEST(PlanSlot, PendingChunksDiscardedByAResetGiveTheirCountsBack) {
    const SlotRig rig({.m_instances = 1, .m_stateful = true});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_GE(session->m_struct_pool.size(), 2U);
    const std::shared_ptr<Slot> slot = rig.slot(0);

    const std::shared_ptr<Chunk> first = park_without_dispatch(*session, 0);
    const std::shared_ptr<Chunk> second = park_without_dispatch(*session, 0);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(slot->active_chunks(), 2U);
    EXPECT_EQ(first->m_slot, slot.get());

    anira_handler_reset(rig.handler());
    EXPECT_EQ(slot->active_chunks(), 0U);
    EXPECT_TRUE(first->m_free.load());
    EXPECT_TRUE(second->m_free.load());
    EXPECT_TRUE(no_chunk_carries_a_slot(*session));
    EXPECT_EQ(rig.m_a.m_entered.load(), 0);
}

// A prepare flushes what is left in the pending queue before it rebuilds the chunks; the slots
// stay, so each flushed chunk gives its count back.
TEST(PlanSlot, PendingChunksFlushedForAPrepareGiveTheirCountsBack) {
    const SlotRig rig({.m_instances = 1, .m_stateful = true});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    const std::shared_ptr<Slot> slot = rig.slot(0);

    const std::shared_ptr<Chunk> parked = park_without_dispatch(*session, 0);
    ASSERT_NE(parked, nullptr);
    EXPECT_EQ(slot->active_chunks(), 1U);

    session->force_reset_dispatch_chain();
    EXPECT_EQ(slot->active_chunks(), 0U);
    EXPECT_EQ(parked->m_slot, nullptr);
    // The flush leaves the struct to the rebuild a prepare does next: back to the pool by hand.
    parked->m_free.store(true);
}

// A session released while an exclusive chunk still waits for its turn: nothing dispatches it
// any more, and the release leaves no count on a slot that outlives the session.
TEST(PlanSlot, AReleaseLeavesNoCountOfAPendingChunk) {
    SlotRig rig({.m_instances = 1, .m_stateful = true});
    const std::shared_ptr<Slot> slot = rig.slot(0);
    {
        const std::shared_ptr<SessionElement> session = rig.session();
        ASSERT_NE(session, nullptr);
        ASSERT_NE(park_without_dispatch(*session, 0), nullptr);
    }
    EXPECT_EQ(slot->active_chunks(), 1U);

    rig.destroy_handler();
    EXPECT_EQ(slot->active_chunks(), 0U);
    EXPECT_EQ(rig.m_a.m_entered.load(), 0);
}

// A session released with jobs in the inference queue: the release's drain completes them as
// zeros without an inference thread, their counts go back there, and the drain then waits for
// the one inside the engine.
TEST(PlanSlot, AReleaseWithJobsQueuedGivesTheirCountsBack) {
    SlotRig rig({.m_instances = 1, .m_threads = 1});
    const std::shared_ptr<Slot> slot = rig.slot(0);
    {
        const std::shared_ptr<SessionElement> session = rig.session();
        ASSERT_NE(session, nullptr);
        ASSERT_GE(session->m_struct_pool.size(), 3U);
    }

    rig.m_a.m_open.store(false);
    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == 1; }));
    rig.process_block();
    rig.process_block();
    ASSERT_EQ(slot->active_chunks(), 3U) << "one inside the engine, two queued";

    {
        JoinedBehind release(std::thread([&rig] { rig.destroy_handler_behind_the_gates(); }),
                             rig.m_a);
        // The drain took the two queued jobs; the one inside the engine still counts.
        EXPECT_TRUE(eventually([&] { return slot->active_chunks() == 1U; }));
        EXPECT_EQ(rig.m_a.m_calls.load(), 0) << "the engine is still held";
        release.join();
    }
    EXPECT_EQ(slot->active_chunks(), 0U);
    EXPECT_EQ(rig.m_a.m_entered.load(), 1) << "a drained job reached the engine";
}

// The warm-up forms a chunk of its own (Core::warm_up_inference) and stamps it like the
// driving thread does: it counts while its inference runs, and no longer.
TEST(PlanSlot, AWarmUpChunkCountsUntilItsInferenceEnded) {
    SlotRig rig({.m_warmup_iterations = 1});
    const std::shared_ptr<Slot> slot = rig.slot(0);
    // The prepare's own pass is over (or was answered from a figure the core kept).
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    const int entered = rig.m_a.m_entered.load();

    rig.m_a.m_open.store(false);
    {
        JoinedBehind pass(std::thread([&rig] { static_cast<void>(rig.manager().warm_up_plan(0)); }),
                          rig.m_a);
        EXPECT_TRUE(eventually([&] { return rig.m_a.m_entered.load() == entered + 1; }));
        EXPECT_EQ(slot->active_chunks(), 1U);
        pass.join();
    }
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    EXPECT_EQ(rig.m_a.m_entered.load(), entered + 1) << "a fixed pass of one inference";
}

// ---- the copies --------------------------------------------------------------------------------

// The backend a post_process is told is the chunk's own copy, and the id strings are the
// session's: both still hold when the chunk's inference finished long ago and its slot no longer
// says what it said (here rewritten by hand while nothing counts on it, as a slot that left the
// table might be).
TEST(PlanSlot, APostProcessReportsTheBackendOfTheStampWhateverBecameOfTheSlot) {
    BackendRecorder recorder;
    const anira_processor_desc processor = backend_processor(recorder);
    SlotRig rig({.m_candidates = two_providers_of_a(), .m_processor = &processor});
    const std::shared_ptr<SessionElement> session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_EQ(session->m_plans.size(), 2U);
    ASSERT_EQ(anira_handler_set_plan(rig.handler(), 1), ANIRA_OK) << "engine A on its provider";
    const std::shared_ptr<Slot> slot = rig.slot(1);

    rig.process_block();
    ASSERT_TRUE(eventually([&] { return rig.m_a.m_calls.load() == 1; }));
    ASSERT_TRUE(eventually([&] { return slot->active_chunks() == 0U; }));
    ASSERT_EQ(recorder.m_num_pre, 1U);
    ASSERT_EQ(recorder.m_num_before.load(), 1);
    ASSERT_EQ(recorder.m_num_post, 0U) << "nobody collected the chunk yet";
    ASSERT_TRUE(no_chunk_carries_a_slot(*session));

    // Nothing counts on the slot: what it says is rewritten, its strings dropped.
    slot->m_engine = ANIRA_ENGINE_NONE;
    slot->m_provider = ANIRA_PROVIDER_NONE;
    slot->m_engine_id = nullptr;
    slot->m_provider_id = nullptr;

    ASSERT_TRUE(eventually([&] {
        static_cast<void>(anira_test::available(rig.handler()));  // collects: runs post_process
        return recorder.m_num_post == 1U;
    }));
    for (const Backend* told : {&recorder.m_pre, &recorder.m_before, &recorder.m_post}) {
        EXPECT_EQ(told->m_engine, ANIRA_ENGINE_CUSTOM);
        EXPECT_EQ(told->m_provider, ANIRA_PROVIDER_CUSTOM);
        ASSERT_NE(told->m_engine_id, nullptr);
        ASSERT_NE(told->m_provider_id, nullptr);
        EXPECT_STREQ(told->m_engine_id, k_engine_a);
        EXPECT_STREQ(told->m_provider_id, k_npu);
    }
    EXPECT_EQ(recorder.m_post.m_engine_id, recorder.m_pre.m_engine_id) << "one string, every phase";
    EXPECT_EQ(recorder.m_post.m_provider_id, recorder.m_pre.m_provider_id);
    EXPECT_EQ(recorder.m_post.m_engine_id, session->intern_id(k_engine_a)) << "the session's own";
    EXPECT_EQ(recorder.m_post.m_provider_id, session->intern_id(k_npu));
}
