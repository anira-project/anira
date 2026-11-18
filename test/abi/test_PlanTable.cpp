// The plan table of a running stream, replaced by a control thread (src/scheduler/
// SessionElement.h: PlanTable, publish_plans, synchronize_plans, claim_plans; src/utils/
// RcuReaders.h). The driving thread stamps every chunk from a published version of the table,
// read through a lease it holds for one entry; a control thread changes the list that owns the
// plan slots (Core::add_plan_slot, take_plan_slot), publishes it again, waits for the leases
// and for the chunks that still count on a slot it took out, and frees that slot itself.
//
// No C entry adds or removes a plan yet: the writer here is the test, white-box, over a
// handler with two custom rows on two engines that multiply by a factor of their own, so the
// stream says which engine ran every chunk, and a processor that records what every chunk was
// stamped with. The slots the test appends are built by hand on the loaded model of one of the
// two plans and report a variant of their own (a marker).

#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>
#include <tanh/core/threading/RCU.h>

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

#include "../support/log_record_collector.h"
#include "engines/Adapter.h"
#include "float_face.h"
#include "handler_support.h"
#include "scheduler/Core.h"
#include "scheduler/InferenceManager.h"
#include "scheduler/SessionElement.h"
#include "utils/RcuReaders.h"

namespace {

using anira::Core;
using anira::SessionElement;
using anira_test::BackgroundDriver;
using anira_test::RecordCollector;
using PlanSlot = anira::SessionElement::PlanSlot;

constexpr size_t k_hop = 64;       // the window of the stream, and the contract's block
constexpr double k_rate = 6400.0;  // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr const char* k_engine_a = "org.example.table_a";
constexpr const char* k_engine_b = "org.example.table_b";
constexpr float k_factor_a = 1.0F;  // what engine A multiplies its input by
constexpr float k_factor_b = 2.0F;  // and engine B
/// The variant a slot built by hand reports: this plus its number.
constexpr uint32_t k_first_marker = 100;
constexpr size_t k_num_slots = anira::RcuReaders<SessionElement::PlanTable>::k_num_slots;

/// An engine whose output is its input times a factor of its own: the stream names the engine
/// that ran a chunk.
class FactorEngine : public anira_test::GateEngine {
public:
    explicit FactorEngine(float factor) : m_factor(factor) {}

    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const float* in = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        for (size_t i = 0; i < anira_test::elements_of(ctx.outputs[0]); ++i) {
            out[i] = in[i] * m_factor;
        }
        return ANIRA_OK;
    }

private:
    float m_factor;
};

/// What every chunk of the stream was stamped with, in the order the chunks were formed: the
/// engine and the variant pre_process was told. pre_process runs inside the real-time process
/// call, so the room is fixed and made up front; a chunk past it is not recorded. The arrays
/// are the driving thread's (read once it is parked); the marker counts are atomics, read by
/// the test while the stream runs.
struct StampRecorder {
    static constexpr size_t k_capacity = size_t{1} << 17;
    static constexpr size_t k_markers = 4096;

    std::vector<uint8_t> m_engine = std::vector<uint8_t>(k_capacity, 0);     ///< 0: A, 1: B, 2: ?
    std::vector<uint32_t> m_variant = std::vector<uint32_t>(k_capacity, 0);  ///< the stamp's
    size_t m_count = 0;
    /// Per marker: the chunks stamped with variant k_first_marker + its index.
    std::array<std::atomic<uint32_t>, k_markers> m_marker_chunks{};
    /// A hold between two chunks of one call: armed (m_hold_next), the next pre_process says
    /// that it is inside (m_holding) and spins there, on the driving thread and inside the
    /// entry, until it is released (m_release). A spin and no sleep: the phase is real-time
    /// code.
    std::atomic<bool> m_hold_next{false};
    std::atomic<bool> m_holding{false};
    std::atomic<bool> m_release{false};

    float factor_of(size_t chunk) const {
        if (m_engine.at(chunk) == 0) { return k_factor_a; }
        return m_engine.at(chunk) == 1 ? k_factor_b : 0.0F;
    }
};

anira_status ANIRA_CALL record_stamp(const anira_processor_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) noexcept {
    auto& recorder = *static_cast<StampRecorder*>(user_data);
    anira_engine engine = ANIRA_ENGINE_NONE;
    const char* id = nullptr;
    uint8_t index = 2;
    if (anira_processor_engine(ctx, &engine, &id) == ANIRA_OK && id != nullptr) {
        if (std::strcmp(id, k_engine_a) == 0) { index = 0; }
        if (std::strcmp(id, k_engine_b) == 0) { index = 1; }
    }
    if (recorder.m_count < StampRecorder::k_capacity) {
        recorder.m_engine[recorder.m_count] = index;
        recorder.m_variant[recorder.m_count] = ctx->variant;
        ++recorder.m_count;
    }
    if (ctx->variant >= k_first_marker &&
        ctx->variant - k_first_marker < StampRecorder::k_markers) {
        recorder.m_marker_chunks[ctx->variant - k_first_marker].fetch_add(1);
    }
    if (recorder.m_hold_next.exchange(false)) {
        recorder.m_holding.store(true, std::memory_order_release);
        while (!recorder.m_release.load(std::memory_order_acquire)) {}
        recorder.m_holding.store(false, std::memory_order_release);
    }
    return anira_processor_default_pre_process(ctx);
}

/// Waits until `done` answers true; false after anira_test::k_wait_s.
template <typename Predicate>
bool wait_for(Predicate done) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(anira_test::k_wait_s);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

/// A prepared handler over one mono stream of k_hop samples per inference with two custom
/// rows, plan 0 on engine A and plan 1 on engine B, and the stamp recorder as its processor;
/// and the control side of a table that is replaced, as a later control entry will do it.
class TableRig {
public:
    explicit TableRig(bool stateful = false, size_t host_block = k_hop) : m_context(2) {
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &err), ANIRA_OK) << err.message;
        add_engine(m_a, k_engine_a);
        add_engine(m_b, k_engine_b);
        anira::ModelConfig model;
        model.add_model_path(k_engine_a, "row-a");
        model.add_model_path(k_engine_b, "row-b");
        model.input(anira_test::stream_tensor("in", 1, k_hop, k_hop));
        model.output(anira_test::stream_tensor("out", 1, k_hop, k_hop));
        model.max_instances(2);
        if (stateful) { model.state(ANIRA_MODEL_STATEFUL); }
        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline, variants.data(), 1, nullptr, 0, &err),
                  ANIRA_OK)
            << err.message;
        anira_processor_desc processor = ANIRA_PROCESSOR_DESC_INIT;
        processor.user_data = &m_recorder;
        processor.realtime_phases = ANIRA_PHASE_BIT(ANIRA_PHASE_PRE_PROCESS);
        processor.pre_process = record_stamp;
        EXPECT_EQ(anira_pipeline_add_processor(m_pipeline, &processor, &err), ANIRA_OK)
            << err.message;
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &err), ANIRA_OK)
            << err.message;
        const anira::ContractHandle contract =
            anira_test::explicit_contract(static_cast<uint32_t>(host_block),
                                          k_rate,
                                          ANIRA_MISS_ZEROS,
                                          0.0,
                                          1.0);
        EXPECT_EQ(anira_handler_prepare(m_handler, contract.native(), &err), ANIRA_OK)
            << err.message;
        m_session = anira_test::session_of(m_handler);
    }
    ~TableRig() {
        m_a.m_open.store(true);
        m_b.m_open.store(true);
        m_session.reset();
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    TableRig(const TableRig&) = delete;
    TableRig& operator=(const TableRig&) = delete;
    TableRig(TableRig&&) = delete;
    TableRig& operator=(TableRig&&) = delete;

    anira_handler* handler() const { return m_handler; }
    const std::shared_ptr<SessionElement>& session() const { return m_session; }
    StampRecorder& recorder() { return m_recorder; }

    /// A plan slot built by hand: on the loaded model of plan `donor`, with a prepared handle
    /// of its own over it (the test's engines have no prepare slot, so no record is needed),
    /// reporting the variant `marker`.
    std::shared_ptr<PlanSlot> make_slot(uint32_t donor, uint32_t marker) const {
        const PlanSlot& from = *m_session->m_plans.at(donor);
        auto slot = std::make_shared<PlanSlot>();
        slot->m_loaded = from.m_loaded;
        slot->m_waiting = from.m_waiting;
        slot->m_registered = from.m_registered;
        slot->m_variant = marker;
        slot->m_engine = from.m_engine;
        slot->m_provider = from.m_provider;
        slot->m_engine_id = from.m_engine_id;  // interned in this session
        slot->m_provider_id = from.m_provider_id;
        slot->m_state_alias = from.m_state_alias;
        slot->m_prepared = slot->m_loaded->prepare(
            anira::engine::PrepareRequest{.m_exclusive = m_session->exclusive(),
                                          .m_info = nullptr});
        return slot;
    }

    /// A slot joins the table: into the list, then published. Its index.
    uint32_t append(std::shared_ptr<PlanSlot> slot) const {
        const uint32_t index = Core::add_plan_slot(m_session, std::move(slot));
        m_session->publish_plans();
        return index;
    }

    /// A plan leaves the table: out of the list, published, and every lease that could still
    /// stamp it waited for. The slot, which the caller frees (free_slot).
    std::shared_ptr<PlanSlot> empty(uint32_t plan) const {
        std::shared_ptr<PlanSlot> taken = Core::take_plan_slot(m_session, plan);
        m_session->publish_plans();
        m_session->synchronize_plans();
        return taken;
    }

    /// Frees a slot that left the table, by hand: once no chunk counts on it its prepared
    /// handle goes back, the pool forgets its model when nothing else holds it, and the slot
    /// dies here.
    void free_slot(std::shared_ptr<PlanSlot>& slot) const {
        Core::wait_until_unused(m_session, *slot);
        EXPECT_EQ(slot->active_chunks(), 0U);
        slot->m_prepared.reset();
        const std::shared_ptr<anira::engine::Loaded> loaded = slot->m_loaded;
        const std::weak_ptr<PlanSlot> watch = slot;
        slot.reset();
        EXPECT_TRUE(watch.expired()) << "something still holds the slot";
        Core::release_loaded_if_unheld(loaded);
    }

    FactorEngine m_a{k_factor_a};  ///< before the handler, which dies first
    FactorEngine m_b{k_factor_b};

private:
    void add_engine(anira_test::GateEngine& gate, const char* id) const {
        const anira_engine_desc desc = gate.desc();
        anira_custom_engine* engine = nullptr;
        anira_error err = ANIRA_ERROR_INIT;
        EXPECT_EQ(anira_custom_engine_create(id, &desc, &engine, &err), ANIRA_OK) << err.message;
        anira_test::add_engine(m_pipeline, engine);
        anira_custom_engine_destroy(engine);
    }

    anira_test::Context m_context;
    StampRecorder m_recorder;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
    std::shared_ptr<SessionElement> m_session;
};

/// Every chunk ran the plan it was stamped with: block c + latency of the stream carries the
/// factor of the engine chunk c was stamped with (the driver is parked).
void expect_chunks_ran_their_stamp(const BackgroundDriver& driver, const StampRecorder& recorder) {
    const std::vector<float>& factors = driver.factors();
    const size_t latency = driver.latency_blocks();
    ASSERT_EQ(recorder.m_count, factors.size()) << "one chunk per block";
    for (size_t block = 0; block < factors.size() && block < latency; ++block) {
        ASSERT_EQ(factors[block], 0.0F) << "the latency's silence, block " << block;
    }
    for (size_t chunk = 0; chunk + latency < factors.size(); ++chunk) {
        ASSERT_EQ(factors[chunk + latency], recorder.factor_of(chunk))
            << "chunk " << chunk << " was stamped with engine "
            << static_cast<int>(recorder.m_engine[chunk]) << ", variant "
            << recorder.m_variant[chunk];
    }
}

/// One hop pushed, of a constant value.
anira_status push_hop(anira_handler* handler, float value) {
    std::vector<float> block(k_hop, value);
    const std::array<const float*, 1> planes{block.data()};
    const anira_tensor in = anira_test::planar_f32(planes.data(), 1, k_hop);
    return anira_handler_push_data(handler, &in, 0);
}

// A lease is neither copied nor moved, so the tests that hold every slot of a pool hold four
// leases by name.
static_assert(k_num_slots == 4, "the tests below hold every slot of a pool: four leases by name");

}  // namespace

// ==== the reader pool =========================================================================

// A lease is the version of its claim for as long as it lives, and the next claim sees the
// newest one. A pool has k_num_slots slots: one more lease than that is none, and a lease that
// dies gives its slot back.
TEST(RcuReaders, ALeaseHoldsTheVersionOfItsClaimAndEverySlotBusyGivesNone) {
    thl::RCU<int> published(1);
    const anira::RcuReaders<int> readers(published);
    {
        const auto first = readers.claim();
        if (!first.has_value()) {
            ADD_FAILURE() << "no lease on a pool nobody reads";
            return;
        }
        EXPECT_EQ(first->data(), 1);
        published.replace([](int& value) { value = 2; });
        EXPECT_EQ(first->data(), 1) << "the version of the claim";
        const auto second = readers.claim();
        if (!second.has_value()) {
            ADD_FAILURE() << "no second lease";
            return;
        }
        EXPECT_EQ(second->data(), 2) << "the newest version";
    }
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        {
            const auto a = readers.claim();
            const auto b = readers.claim();
            const auto c = readers.claim();
            const auto d = readers.claim();
            ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());
            EXPECT_FALSE(readers.claim().has_value()) << "every slot is busy";
            EXPECT_FALSE(readers.claim().has_value());
        }
        EXPECT_TRUE(readers.claim().has_value()) << "the leases gave their slots back";
    }
    // Nothing is out: synchronize has nobody to wait for.
    published.replace([](int& value) { value = 3; });
    published.synchronize();
    const auto last = readers.claim();
    if (!last.has_value()) {
        ADD_FAILURE() << "no lease behind the synchronize";
        return;
    }
    EXPECT_EQ(last->data(), 3);
}

// More readers than slots, on threads that come and go between the slots, beside a writer that
// frees what it published by hand behind synchronize(), as a plan slot is freed: a reader
// never sees a freed object (the address sanitizer's report, and the flag here), a claim that
// finds every slot busy gets none and tries again, and the thread sanitizer sees no race on a
// slot's hand-over.
TEST(RcuReaders, ManyReadersOverFewSlotsNeverSeeAFreedObject) {
    constexpr int k_readers = 8;
    constexpr int k_publications = 2000;
    struct Heavy {
        std::atomic<bool> m_alive{true};
        int m_value = 0;
    };
    struct Published {
        Heavy* m_heavy = nullptr;  // not owned: the writer frees it behind synchronize()
    };
    auto current = std::make_unique<Heavy>();
    thl::RCU<Published> published(Published{.m_heavy = current.get()});
    const anira::RcuReaders<Published> readers(published);

    std::atomic<bool> stop{false};
    std::atomic<int> freed_seen{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> busy{0};
    std::array<std::atomic<uint64_t>, k_readers> rounds{};
    std::vector<std::thread> threads;
    threads.reserve(k_readers);
    for (size_t reader = 0; reader < k_readers; ++reader) {
        threads.emplace_back([&, reader] {
            while (!stop.load(std::memory_order_acquire)) {
                {
                    const auto lease = readers.claim();
                    if (lease.has_value()) {
                        const Heavy* const heavy = lease->data().m_heavy;
                        if (!heavy->m_alive.load(std::memory_order_acquire) || heavy->m_value < 0) {
                            freed_seen.fetch_add(1);
                        }
                        reads.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        busy.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                rounds.at(reader).fetch_add(1, std::memory_order_release);
            }
        });
    }
    for (int publication = 1; publication <= k_publications; ++publication) {
        auto next = std::make_unique<Heavy>();
        next->m_value = publication;
        published.replace([&next](Published& value) { value.m_heavy = next.get(); });
        published.synchronize();
        // No lease holds the version that named `current` any more.
        current->m_alive.store(false, std::memory_order_release);
        current->m_value = -1;
        current = std::move(next);
    }
    // Every reader has gone round behind the last publication.
    for (size_t reader = 0; reader < k_readers; ++reader) {
        const uint64_t seen = rounds.at(reader).load();
        while (rounds.at(reader).load(std::memory_order_acquire) == seen) {
            std::this_thread::yield();
        }
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }
    EXPECT_EQ(freed_seen.load(), 0);
    EXPECT_GT(reads.load(), 0U);
}

// ==== the published table =====================================================================

// The session publishes the list it was built with, and a read lease holds one version of the
// table: a slot appended and an entry emptied behind the claim are in the next claim's table,
// not in this one's. select_plan checks an index against the newest table.
TEST(PlanTable, ALeaseHoldsTheTableOfItsClaim) {
    const TableRig rig;
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_EQ(session->m_plans.size(), 2U);
    EXPECT_FALSE(session->select_plan(2)) << "the table has two plans";
    std::shared_ptr<PlanSlot> appended = rig.make_slot(0, k_first_marker);
    PlanSlot* const appended_address = appended.get();
    {
        const auto before = session->claim_plans();
        if (!before.has_value()) {
            ADD_FAILURE() << "no lease on a table nobody reads";
            return;
        }
        const std::vector<PlanSlot*>& slots_before = before->data().m_slots;
        ASSERT_EQ(slots_before.size(), 2U);
        EXPECT_EQ(slots_before[0], session->m_plans[0].get());
        EXPECT_EQ(slots_before[1], session->m_plans[1].get());

        EXPECT_EQ(rig.append(std::move(appended)), 2U);
        EXPECT_EQ(slots_before.size(), 2U) << "the table of the claim";
        EXPECT_TRUE(session->select_plan(2)) << "the newest table has the index";
        EXPECT_TRUE(session->select_plan(0));
        const auto behind = session->claim_plans();
        if (!behind.has_value()) {
            ADD_FAILURE() << "no second lease";
            return;
        }
        ASSERT_EQ(behind->data().m_slots.size(), 3U);
        EXPECT_EQ(behind->data().m_slots[2], appended_address);
    }
    // The leases are back: the entry is emptied and the leases waited for.
    std::shared_ptr<PlanSlot> taken = rig.empty(2);
    ASSERT_EQ(taken.get(), appended_address);
    EXPECT_EQ(session->m_plans.size(), 3U) << "the index stays";
    EXPECT_EQ(session->m_plans[2], nullptr);
    EXPECT_EQ(Core::take_plan_slot(session, 2), nullptr) << "an empty entry";
    EXPECT_EQ(Core::take_plan_slot(session, 9), nullptr) << "no such index";
    {
        const auto emptied = session->claim_plans();
        if (!emptied.has_value()) {
            ADD_FAILURE() << "no lease behind the synchronize";
            return;
        }
        ASSERT_EQ(emptied->data().m_slots.size(), 3U);
        EXPECT_EQ(emptied->data().m_slots[2], nullptr);
    }
    EXPECT_TRUE(session->select_plan(2)) << "an index of the table; the stamp says what runs";
    EXPECT_TRUE(session->select_plan(0));
    rig.free_slot(taken);
    EXPECT_GT(session->m_chunk_alignment, 0U) << "what prepare built the chunks with";
}

// The stamp when the table has no plan under the selection. An index beyond the table keeps
// the plan of the chunk before it and changes nothing else. An index whose plan was taken out
// does the same, moves the selection back to the plan the chunk runs and records
// ANIRA_ERROR_NOT_FOUND, logged once. When the plan of the chunk before is gone as well, the
// chunk runs the first plan the table has. The stream stays ordered and complete throughout,
// and every chunk runs what it was stamped with.
TEST(PlanTable, AChunkWithoutAPlanUnderTheSelectionRunsThePlanBeforeIt) {
    anira_drain_log();
    RecordCollector collector;
    TableRig rig;
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    BackgroundDriver driver(h, k_hop);
    StampRecorder& recorder = rig.recorder();
    // What the chunks from `first` on were stamped with: the engine (0: A, 1: B) and variant.
    const auto expect_stamps = [&](size_t first, size_t count, int engine, uint32_t variant) {
        ASSERT_TRUE(driver.wait());
        ASSERT_EQ(recorder.m_count, first + count);
        for (size_t chunk = first; chunk < first + count; ++chunk) {
            EXPECT_EQ(recorder.m_engine[chunk], engine) << "chunk " << chunk;
            EXPECT_EQ(recorder.m_variant[chunk], variant) << "chunk " << chunk;
        }
    };

    driver.step(3);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(0, 3, 0, 0));

    // An index beyond the table: a plan of a table this one is not. Silently plan 0.
    h->m_selected_plan.store(7);
    driver.step(3);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(3, 3, 0, 0));
    EXPECT_EQ(anira_handler_get_plan(h), 7U) << "the selection stands";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    ASSERT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
    driver.step(3);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(6, 3, 1, 0));

    // Plan 0 selected, and taken out before a chunk is formed on it: the stream stays on plan
    // 1, the selection goes back there, and the refusal is recorded and logged once.
    ASSERT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    std::shared_ptr<PlanSlot> plan_0 = rig.empty(0);
    ASSERT_NE(plan_0, nullptr);
    driver.step(3);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(9, 3, 1, 0));
    EXPECT_EQ(anira_handler_get_plan(h), 1U) << "back on the plan the stream runs";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_FOUND);
    rig.free_slot(plan_0);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "the selected plan 0 was removed", "rt"), 1U);
#else
    static_cast<void>(collector);
#endif
    // The latch re-armed by hand (anira_handler_reset would start the stream over).
    static_cast<void>(h->m_rt.rearm());
    ASSERT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // A third plan, on engine B's model. Then the emptied plan 0 is selected again and plan 1,
    // the plan of the chunk before, is taken out too: the chunk runs the first plan the table
    // has, the third.
    EXPECT_EQ(rig.append(rig.make_slot(1, k_first_marker)), 2U);
    ASSERT_TRUE(session->select_plan(0));
    std::shared_ptr<PlanSlot> plan_1 = rig.empty(1);
    ASSERT_NE(plan_1, nullptr);
    driver.step(3);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(12, 3, 1, k_first_marker));
    EXPECT_EQ(anira_handler_get_plan(h), 2U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_FOUND);
    rig.free_slot(plan_1);
    EXPECT_EQ(recorder.m_marker_chunks[0].load(), 3U);

    driver.step(4);
    ASSERT_NO_FATAL_FAILURE(expect_stamps(15, 4, 1, k_first_marker));
    ASSERT_NO_FATAL_FAILURE(driver.expect_continuous());
    ASSERT_NO_FATAL_FAILURE(expect_chunks_ran_their_stamp(driver, recorder));
    EXPECT_EQ(rig.m_a.m_calls.load(), 6) << "the chunks stamped with plan 0";
    EXPECT_EQ(rig.m_b.m_calls.load(), 13);
}

// One lease per entry, not per chunk: a call that forms two chunks stamps both from the table
// its lease holds. The contract's block is two hops, so one anira_handler_process forms two
// chunks; the call is held between them (inside the first chunk's pre_process), plan 1, which
// it runs, is taken out of the list and the table published without it. The second chunk is
// still stamped with plan 1, the synchronize behind the publication waits for the call, and
// the slot is freed once both chunks ran. The next call reads the table without the plan.
TEST(PlanTable, AnEntryStampsEveryChunkOfItsCallFromTheTableOfItsLease) {
    TableRig rig(/*stateful=*/false, /*host_block=*/2 * k_hop);
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    StampRecorder& recorder = rig.recorder();
    ASSERT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
    std::vector<float> block(2 * k_hop, 0.5F);
    const std::array<float*, 1> planes{block.data()};
    const anira_tensor io = anira_test::planar_f32(planes.data(), 1, block.size());

    recorder.m_hold_next.store(true);
    anira_status status = ANIRA_ERROR_UNKNOWN;
    std::thread driving([&] { status = anira_handler_process(h, &io, 0, &io, 0, nullptr); });
    const bool held = wait_for([&] { return recorder.m_holding.load(); });
    PlanSlot* const plan_1 = session->m_plans[1].get();
    const uint32_t stamped_while_held = plan_1->active_chunks();
    // Plan 1 leaves the list and the table is published without it, under the call's feet.
    std::shared_ptr<PlanSlot> taken = Core::take_plan_slot(session, 1);
    session->publish_plans();
    std::atomic<bool> synchronized{false};
    std::thread synchronizer([&] {
        session->synchronize_plans();
        synchronized.store(true, std::memory_order_release);
    });
    for (int yields = 0; yields < 100; ++yields) { std::this_thread::yield(); }
    const bool synchronized_while_held = synchronized.load();
    recorder.m_release.store(true, std::memory_order_release);
    driving.join();
    synchronizer.join();
    ASSERT_TRUE(held) << "the call never reached its first chunk's pre_process";
    EXPECT_EQ(stamped_while_held, 1U) << "the first chunk, stamped with plan 1";
    EXPECT_FALSE(synchronized_while_held) << "the call was inside, its lease out";
    EXPECT_FALSE(ANIRA_FAILED(status)) << status;

    // Both chunks of the call ran plan 1: the second one was stamped from the lease's table,
    // which has the plan, and nothing was refused.
    ASSERT_EQ(recorder.m_count, 2U) << "two hops, two chunks";
    EXPECT_EQ(recorder.m_engine[0], 1);
    EXPECT_EQ(recorder.m_engine[1], 1) << "stamped from the table published behind the claim";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 1U);
    ASSERT_NE(taken, nullptr);
    rig.free_slot(taken);
    EXPECT_EQ(rig.m_b.m_calls.load(), 2);

    // The next call claims a lease on the table without plan 1, which is still selected: its
    // chunks run the first plan the table has, the selection follows, and the latch says so.
    EXPECT_FALSE(ANIRA_FAILED(anira_handler_process(h, &io, 0, &io, 0, nullptr)));
    ASSERT_EQ(recorder.m_count, 4U);
    EXPECT_EQ(recorder.m_engine[2], 0);
    EXPECT_EQ(recorder.m_engine[3], 0);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_FOUND);
}

// synchronize_plans() waits for an entry that is inside its call. The driving thread runs
// anira_handler_process_wait on plan 0, whose engine is held on a closed gate: the call has
// stamped its chunk and waits for it, its read lease out. A table published now is not seen by
// that call, and the synchronize behind the publication returns only once the call has, which
// is when the gate opens.
TEST(PlanTable, SynchronizeWaitsForAnEntryInsideItsCall) {
    TableRig rig;
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    BackgroundDriver driver(h, k_hop, BackgroundDriver::Entry::ProcessWait);
    driver.step(2);
    ASSERT_TRUE(driver.wait());
    rig.m_a.m_open.store(false);
    driver.step(1);
    PlanSlot* const plan_0 = session->m_plans[0].get();
    ASSERT_TRUE(wait_for([&] { return driver.inside_call() && plan_0->active_chunks() == 1; }))
        << "the call never got as far as its wait";

    // Plan 1 leaves the list and the table is published without it; the call inside keeps the
    // table of its lease.
    ASSERT_EQ(anira_handler_get_plan(h), 0U);
    std::shared_ptr<PlanSlot> plan_1 = Core::take_plan_slot(session, 1);
    ASSERT_NE(plan_1, nullptr);
    session->publish_plans();
    std::atomic<bool> synchronized{false};
    std::thread synchronizer([&] {
        session->synchronize_plans();
        synchronized.store(true, std::memory_order_release);
    });
    // The gate is closed: the call cannot have returned, so neither has the synchronize.
    for (int yields = 0; yields < 100; ++yields) { std::this_thread::yield(); }
    EXPECT_FALSE(synchronized.load());
    EXPECT_TRUE(driver.inside_call());
    rig.m_a.m_open.store(true);
    synchronizer.join();
    EXPECT_TRUE(synchronized.load());
    ASSERT_TRUE(driver.wait());
    rig.free_slot(plan_1);

    driver.step(3);
    ASSERT_TRUE(driver.wait());
    ASSERT_NO_FATAL_FAILURE(driver.expect_continuous());
    ASSERT_NO_FATAL_FAILURE(expect_chunks_ran_their_stamp(driver, rig.recorder()));
    EXPECT_EQ(rig.m_b.m_calls.load(), 0);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// The writer beside a free-running driving thread. While the stream runs, this thread
// publishes the same table again, appends a slot, selects it until chunks were stamped with
// it, takes the slot out again, waits for the leases and for its chunks and frees it by hand,
// a hundred and twenty times over. In every other cycle another plan is selected before the
// slot is taken out; in the others the slot leaves while it is selected and the stream runs on
// it, and the stream goes on with the first plan the table has. The stream stays ordered and
// complete, every chunk runs the plan it was stamped with, and no chunk is stamped with a slot
// behind the synchronize of its removal (the address sanitizer would report the stamp of a
// freed one).
TEST(PlanTable, TheTableIsReplacedBesideARunningStream) {
    constexpr uint32_t k_cycles = 120;
    TableRig rig;
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    StampRecorder& recorder = rig.recorder();
    BackgroundDriver driver(h, k_hop);
    driver.run_free(StampRecorder::k_capacity);
    std::vector<uint32_t> stamped_at_synchronize(k_cycles, 0);
    for (uint32_t cycle = 0; cycle < k_cycles; ++cycle) {
        SCOPED_TRACE("cycle " + std::to_string(cycle));
        // The same table, published again.
        for (int again = 0; again < 3; ++again) {
            session->publish_plans();
            session->synchronize_plans();
        }
        // A further slot, on the model of plan 0 or of plan 1 in turn, selected until three
        // chunks were stamped with it.
        std::shared_ptr<PlanSlot> slot = rig.make_slot(cycle % 2, k_first_marker + cycle);
        const uint32_t index = rig.append(slot);
        ASSERT_EQ(index, 2 + cycle) << "an index is never used twice";
        ASSERT_TRUE(session->select_plan(index));
        ASSERT_TRUE(wait_for([&] { return recorder.m_marker_chunks.at(cycle).load() >= 3; }))
            << "no chunk was stamped with the appended slot";
        // Every other cycle another plan is selected first, so the plan that leaves is not
        // selected; else it leaves under the stream that runs on it.
        if (cycle % 2 == 0) { ASSERT_EQ(anira_handler_set_plan(h, (cycle / 2) % 2), ANIRA_OK); }
        std::shared_ptr<PlanSlot> taken = rig.empty(index);
        ASSERT_EQ(taken, slot);
        stamped_at_synchronize.at(cycle) = recorder.m_marker_chunks.at(cycle).load();
        slot.reset();
        rig.free_slot(taken);
    }
    // The last slot left while it was selected: one whole call behind that moves the selection.
    const size_t done = driver.done();
    ASSERT_TRUE(wait_for([&] { return driver.done() >= done + 2; }));
    driver.stop();
    ASSERT_TRUE(driver.wait());
    for (uint32_t cycle = 0; cycle < k_cycles; ++cycle) {
        EXPECT_EQ(recorder.m_marker_chunks.at(cycle).load(), stamped_at_synchronize.at(cycle))
            << "cycle " << cycle << ": a chunk was stamped behind the synchronize of its removal";
        EXPECT_GE(stamped_at_synchronize.at(cycle), 3U);
    }
    ASSERT_NO_FATAL_FAILURE(driver.expect_continuous());
    ASSERT_NO_FATAL_FAILURE(expect_chunks_ran_their_stamp(driver, recorder));
    // A slot that left while it was selected is the one refusal the stream may have recorded.
    const anira_status last = anira_handler_rt_error(h);
    EXPECT_TRUE(last == ANIRA_OK || last == ANIRA_ERROR_NOT_FOUND) << last;
    EXPECT_LT(anira_handler_get_plan(h), 2U) << "back on a plan the table has";
    ASSERT_EQ(session->m_plans.size(), 2U + k_cycles);
    for (size_t index = 2; index < session->m_plans.size(); ++index) {
        EXPECT_EQ(session->m_plans[index], nullptr) << "entry " << index;
    }
    // An inference thread gives a chunk's count back right behind its done signal.
    EXPECT_TRUE(wait_for([&] {
        return session->m_plans[0]->active_chunks() == 0 &&
               session->m_plans[1]->active_chunks() == 0;
    }));
}

// The getters and the selection entries beside a table that is replaced under a running
// stream: eight threads call them all while this one appends and empties slots nobody selects.
// A getter never answers the unprepared value; a selection entry returns ANIRA_OK or, when
// more of them overlap than the plan rows have reader slots, ANIRA_ERROR_CAPACITY; the stream
// stays ordered and complete on the two plans the entries select.
TEST(PlanTable, TheGettersAndTheSelectionEntriesBesideAReplacedTable) {
    constexpr int k_threads = 8;
    constexpr uint32_t k_cycles = 40;
    TableRig rig;
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    const uint32_t latency = anira_handler_get_latency(h, 0);
    const uint32_t entries = anira_handler_num_entries(h);
    const anira_plan_report* const report = anira_handler_plan_report(h);
    ASSERT_GT(latency, 0U);
    ASSERT_GT(entries, 0U);
    ASSERT_NE(report, nullptr);

    BackgroundDriver driver(h, k_hop);
    driver.run_free(StampRecorder::k_capacity);
    std::atomic<bool> stop{false};
    std::atomic<int> wrong{0};
    std::atomic<uint64_t> selected{0};
    std::atomic<uint64_t> busy{0};
    std::array<std::atomic<uint64_t>, k_threads> rounds{};
    std::vector<std::thread> threads;
    threads.reserve(k_threads);
    for (size_t thread = 0; thread < k_threads; ++thread) {
        threads.emplace_back([&, thread] {
            anira_backend_id on_a = ANIRA_BACKEND_ID_INIT;
            on_a.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
            on_a.engine_id = k_engine_a;
            anira_backend_id on_b = on_a;
            on_b.engine_id = k_engine_b;
            const auto count = [&](anira_status status) {
                if (status == ANIRA_OK) {
                    selected.fetch_add(1, std::memory_order_relaxed);
                } else if (status == ANIRA_ERROR_CAPACITY) {
                    busy.fetch_add(1, std::memory_order_relaxed);
                } else {
                    wrong.fetch_add(1);
                }
            };
            uint64_t round = 0;
            while (!stop.load(std::memory_order_acquire)) {
                count(anira_handler_set_plan(h, static_cast<uint32_t>((round + thread) % 2)));
                count(anira_handler_set_model(h, 0));
                count(anira_handler_set_backend(h, round % 2 == 0 ? &on_a : &on_b));
                if (anira_handler_get_plan(h) > 1U) { wrong.fetch_add(1); }
                if (anira_handler_get_latency(h, 0) != latency) { wrong.fetch_add(1); }
                std::array<uint32_t, 1> all{};
                uint32_t num = 1;
                if (anira_handler_get_latencies(h, &num, all.data()) != ANIRA_OK ||
                    all[0] != latency) {
                    wrong.fetch_add(1);
                }
                if (anira_handler_num_entries(h) != entries) { wrong.fetch_add(1); }
                if (anira_handler_plan_report(h) != report) { wrong.fetch_add(1); }
                ++round;
                rounds.at(thread).fetch_add(1, std::memory_order_release);
            }
        });
    }
    for (uint32_t cycle = 0; cycle < k_cycles; ++cycle) {
        SCOPED_TRACE("cycle " + std::to_string(cycle));
        session->publish_plans();
        session->synchronize_plans();
        const uint32_t index = rig.append(rig.make_slot(cycle % 2, k_first_marker + cycle));
        ASSERT_EQ(index, 2 + cycle);
        std::shared_ptr<PlanSlot> taken = rig.empty(index);
        ASSERT_NE(taken, nullptr);
        rig.free_slot(taken);
    }
    // Every thread has gone round behind the last cycle.
    for (size_t thread = 0; thread < k_threads; ++thread) {
        const uint64_t seen = rounds.at(thread).load();
        ASSERT_TRUE(wait_for([&] { return rounds.at(thread).load() != seen; }));
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& thread : threads) { thread.join(); }
    driver.stop();
    ASSERT_TRUE(driver.wait());
    EXPECT_EQ(wrong.load(), 0);
    EXPECT_GT(selected.load(), 0U);
    ASSERT_NO_FATAL_FAILURE(driver.expect_continuous());
    ASSERT_NO_FATAL_FAILURE(expect_chunks_ran_their_stamp(driver, rig.recorder()));
    for (uint32_t cycle = 0; cycle < k_cycles; ++cycle) {
        EXPECT_EQ(rig.recorder().m_marker_chunks.at(cycle).load(), 0U)
            << "nobody selected the slot of cycle " << cycle;
    }
    // Back-pressure is a return value: nothing of it reaches the latch.
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << busy.load() << " calls found no slot";
}

// Every reader slot busy is ANIRA_ERROR_CAPACITY, behind the checks and before anything moved.
// With k_num_slots leases on the session's table out, every entry of the driving thread that
// has a status returns it, delivers nothing and leaves the stream as it was; with the leases
// back the same calls run. The same for the selection entries over the plan rows, the other
// pool. It is back-pressure: nothing is recorded in the latch. reset has no status and reads
// no table: it resets all the same. A selection by axis whose selected plan is no row of the
// rows it reads gives up after a few reads, with the same status.
TEST(PlanTable, EveryReaderSlotBusyIsCapacity) {
    const TableRig rig;
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    std::vector<float> block(k_hop, 0.25F);
    const std::array<float*, 1> planes{block.data()};
    const anira_tensor io = anira_test::planar_f32(planes.data(), 1, k_hop);
    const size_t primed = anira_test::available(h);

    {
        // Every reader slot of the session's table is out.
        const auto a = session->claim_plans();
        const auto b = session->claim_plans();
        const auto c = session->claim_plans();
        const auto d = session->claim_plans();
        ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());
        EXPECT_FALSE(session->claim_plans().has_value());
        size_t delivered = 77;
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, &delivered), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(delivered, 0U);
        delivered = 77;
        EXPECT_EQ(anira_handler_process_multi(h, &io, 1, &io, 1, &delivered), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(delivered, 0U);
        EXPECT_EQ(anira_handler_push_data(h, &io, 0), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_push_data_multi(h, &io, 1), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_pop_data(h, &io, 0, &delivered), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_pop_data_multi(h, &io, 1, &delivered), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_process_wait(h, &io, 0, &io, 0, &delivered, 0.0),
                  ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_process_multi_wait(h, &io, 1, &io, 1, &delivered, 0.0),
                  ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_pop_data_wait(h, &io, 0, &delivered, 0.0), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_pop_data_multi_wait(h, &io, 1, &delivered, 0.0),
                  ANIRA_ERROR_CAPACITY);
        size_t count = 77;
        EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &count), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(count, 0U);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << "back-pressure is not recorded";
        // A check that fails is still the first answer, and is recorded as ever.
        EXPECT_EQ(anira_handler_push_data(h, &io, 5), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
        // reset has no status to refuse with and reads no table: it runs.
        anira_handler_reset(h);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        // The selection entries read the plan rows, another pool: they run.
        EXPECT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
        EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_OK);
    }
    EXPECT_EQ(rig.m_a.m_calls.load() + rig.m_b.m_calls.load(), 0) << "nothing was submitted";
    EXPECT_EQ(anira_test::available(h), primed) << "nothing moved, and the leases are back";
    EXPECT_EQ(anira_handler_push_data(h, &io, 0), ANIRA_OK);

    {
        // Every reader slot of the handler's plan rows is out.
        const auto a = h->m_plan_row_readers.claim();
        const auto b = h->m_plan_row_readers.claim();
        const auto c = h->m_plan_row_readers.claim();
        const auto d = h->m_plan_row_readers.claim();
        ASSERT_TRUE(a.has_value() && b.has_value() && c.has_value() && d.has_value());
        EXPECT_EQ(anira_handler_set_plan(h, 0), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_ERROR_CAPACITY);
        anira_backend_id backend = ANIRA_BACKEND_ID_INIT;
        backend.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
        backend.engine_id = k_engine_a;
        EXPECT_EQ(anira_handler_set_backend(h, &backend), ANIRA_ERROR_CAPACITY);
        EXPECT_EQ(anira_handler_get_plan(h), 1U) << "the selection as it was";
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        // A check that fails is the first answer here too.
        EXPECT_EQ(anira_handler_set_backend(h, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
        // The getters read no table, and the driving thread's entries the other pool.
        EXPECT_GT(anira_handler_get_latency(h, 0), 0U);
        EXPECT_NE(anira_handler_plan_report(h), nullptr);
        size_t count = 0;
        EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &count), ANIRA_OK);
    }
    EXPECT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    anira_handler_reset(h);

    // The selected plan is no row of the rows a selection by axis reads (a white-box store:
    // only a table published behind the read can have such a selection): the entry reads again,
    // then gives up. set_plan reads no selection and runs.
    h->m_selected_plan.store(9);
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_ERROR_CAPACITY);
    EXPECT_EQ(anira_handler_get_plan(h), 9U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_OK);
}

// A stopped driving thread does not pin a plan of an exclusive session. Three hops are pushed
// on plan 1 with its engine held on a closed gate (one chunk inside the engine, two waiting for
// their turn), and the driving thread never calls again. The plan is taken out, the leases are
// waited for, and the wait for its chunks returns once the gate opens: the inference threads
// run the three chunks one after the other, no call of the driving thread needed. Then the
// slot is freed, and the stream delivers the three chunks, in order.
TEST(PlanTable, AStoppedDrivingThreadDoesNotPinAPlanOfAnExclusiveSession) {
    TableRig rig(/*stateful=*/true);
    anira_handler* const h = rig.handler();
    const std::shared_ptr<SessionElement>& session = rig.session();
    ASSERT_NE(session, nullptr);
    ASSERT_TRUE(session->exclusive());
    const size_t primed = anira_test::available(h);
    ASSERT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
    rig.m_b.m_open.store(false);
    for (int hop = 1; hop <= 3; ++hop) {
        ASSERT_EQ(push_hop(h, static_cast<float>(hop)), ANIRA_OK);
    }
    PlanSlot* const plan_1 = session->m_plans[1].get();
    ASSERT_EQ(plan_1->active_chunks(), 3U) << "three chunks stamped, none finished";

    // The last call of the driving thread was the third push. Plan 0 is selected and plan 1
    // taken out.
    ASSERT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    std::shared_ptr<PlanSlot> taken = rig.empty(1);
    ASSERT_EQ(taken.get(), plan_1);
    EXPECT_EQ(taken->active_chunks(), 3U) << "the gate is closed: the chunks still count";
    std::atomic<bool> unused{false};
    std::thread waiter([&] {
        Core::wait_until_unused(session, *taken);
        unused.store(true, std::memory_order_release);
    });
    // The gate is closed: no chunk of the plan has finished, so the wait cannot have returned.
    EXPECT_EQ(rig.m_b.m_calls.load(), 0);
    EXPECT_FALSE(unused.load());
    rig.m_b.m_open.store(true);
    waiter.join();
    EXPECT_TRUE(unused.load());
    EXPECT_EQ(taken->active_chunks(), 0U);
    EXPECT_EQ(rig.m_b.m_calls.load(), 3) << "every chunk of the plan ran, without a driver call";
    rig.free_slot(taken);

    // The driving thread again: it collects the three chunks, engine B's, in the order pushed,
    // behind what the receive ring was primed with.
    ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_available(h, primed + (3 * k_hop)));
    std::vector<float> expected(primed, 0.0F);
    for (int hop = 1; hop <= 3; ++hop) {
        expected.insert(expected.end(), k_hop, static_cast<float>(hop) * k_factor_b);
    }
    for (size_t block = 0; block < 3; ++block) {
        std::vector<float> out(k_hop, -1.0F);
        const std::array<float*, 1> planes{out.data()};
        const anira_tensor tensor = anira_test::planar_f32(planes.data(), 1, k_hop);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_pop_data(h, &tensor, 0, &delivered), ANIRA_OK);
        ASSERT_EQ(delivered, k_hop);
        const std::span<const float> wanted(expected.data() + (block * k_hop), k_hop);
        ASSERT_NO_FATAL_FAILURE(anira_test::expect_same_block(out, wanted, block));
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(rig.m_a.m_calls.load(), 0);
}
