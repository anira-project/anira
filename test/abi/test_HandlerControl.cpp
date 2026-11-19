// What a C handler keeps of its own so that a control call and the entries of a running stream
// may overlap (src/capi/handler.h): the slot signature (what a slot number names, read by every
// real-time entry in place of the pipeline copy), the atomics the value getters answer from (so
// that none of them reaches into the session a prepare replaces) and the control lock that
// serialises anira_handler_prepare and anira_handler_destroy. Engine-free: the one model path is
// the custom row on the pass-through engine a Handler adds (output i is input i), so what goes
// in comes back, delayed by the latency.

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../support/log_record_collector.h"
#include "capi/handler.h"
#include "capi/processor.h"
#include "capi/validate.h"
#include "float_face.h"
#include "handler_support.h"
#include "scheduler/InferenceManager.h"
#include "scheduler/SessionElement.h"

namespace {

using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::Handler;
using anira_test::RecordCollector;
using anira_test::whole_f32;

constexpr uint32_t k_hop = 8;      // the window of the streamed tensor, and the contract's block
constexpr double k_rate = 800.0;   // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr size_t k_unset = 77777;  // what a delivered count holds before its call
constexpr size_t k_values = 4;     // the elements of the Static tensor
using Values = std::array<float, k_values>;
constexpr std::array<int64_t, 2> k_values_shape{1, static_cast<int64_t>(k_values)};

/// Slot 0 of either side: a mono stream of k_hop samples per inference. Slot 1: a Static tensor
/// of four values. Two slots per side, so a single-tensor form goes through the staging array.
ModelConfig two_slot_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, k_hop)
        .window(k_hop, k_hop, 0);
    model.input(in);
    TensorSpec values_in("values_in", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    values_in.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_ANY, k_values);
    model.input(values_in);
    TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, 1)
        .axis(2, ANIRA_AXIS_TIME, k_hop)
        .window(k_hop, k_hop, 0);
    model.output(out);
    TensorSpec values_out("values_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    values_out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_ANY, k_values);
    model.output(values_out);
    model.max_instances(2);
    return model;
}

/// The contract of every case: blocks of one hop, an explicit budget, no warm-up.
anira::ContractHandle hop_contract(anira_miss_policy on_miss = ANIRA_MISS_ZEROS) {
    return anira_test::explicit_contract(k_hop, k_rate, on_miss, 0.0, 1.0);
}

/// One hop through the stream's single form, in place, waited for: the block the call hands
/// back in `block`.
void process_hop(anira_handler* handler, std::vector<float>& block) {
    const std::array<float*, 1> planes{block.data()};
    const anira_tensor io = anira_test::planar_f32(planes.data(), 1, block.size());
    const size_t before = anira_test::available(handler);
    size_t delivered = k_unset;
    ASSERT_EQ(anira_handler_process(handler, &io, 0, &io, 0, &delivered), ANIRA_OK);
    ASSERT_EQ(delivered, block.size());
    ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_block(handler, before));
}

/// `blocks` hops of the ramp from block `first` on, each compared with the delay line's answer.
void run_stream(anira_handler* handler, anira_test::DelayLine& line, size_t first, size_t blocks) {
    for (size_t k = first; k < first + blocks; ++k) {
        std::vector<float> block = anira_test::ramp(k, k_hop);
        const std::vector<float> expected = line.next(block);
        ASSERT_NO_FATAL_FAILURE(process_hop(handler, block));
        ASSERT_NO_FATAL_FAILURE(anira_test::expect_same_block(block, expected, k));
    }
}

/// Every byte of a tensor descriptor.
std::vector<unsigned char> bytes_of(const anira_tensor& tensor) {
    std::vector<unsigned char> bytes(sizeof(anira_tensor));
    std::memcpy(bytes.data(), &tensor, sizeof(anira_tensor));
    return bytes;
}

/// Every entry a host calls while the stream runs, on a prepared handler over two_slot_model():
/// the stream through the single form and the multi form, the Static entries, the ring state,
/// two refusals whose records name their slot, the getters, the selection entries, reset and a
/// _wait twin.
void drive_every_entry(anira_handler* h, RecordCollector& collector) {
    const uint32_t latency = anira_handler_get_latency(h, 0);
    const size_t primed = anira_test::available(h);

    // The stream through the single form, whose one tensor is staged in the handler's array.
    anira_test::DelayLine line(latency);
    ASSERT_NO_FATAL_FAILURE(run_stream(h, line, 1, 4));
    EXPECT_EQ(h->m_staged_inputs[0].shape[1], 0) << "the staged slot is empty again";
    EXPECT_EQ(h->m_staged_outputs[0].shape[1], 0);

    // The Static entries, and the multi form that carries both slots: the value that went in
    // with a block comes back with that block's inference.
    Values sent{1.0F, 2.0F, 3.0F, 4.0F};
    const anira_tensor sent_tensor = whole_f32(sent.data(), k_values_shape);
    ASSERT_EQ(anira_handler_set_static_input(h, 1, &sent_tensor), ANIRA_OK);
    {
        std::vector<float> block = anira_test::ramp(5, k_hop);
        const std::vector<float> expected = line.next(block);
        const std::array<float*, 1> planes{block.data()};
        Values carried{5.0F, 6.0F, 7.0F, 8.0F};
        Values received{};
        const std::array<anira_tensor, 2> inputs{anira_test::planar_f32(planes.data(), 1, k_hop),
                                                 whole_f32(carried.data(), k_values_shape)};
        const std::array<anira_tensor, 2> outputs{anira_test::planar_f32(planes.data(), 1, k_hop),
                                                  whole_f32(received.data(), k_values_shape)};
        std::array<size_t, 2> delivered{k_unset, k_unset};
        const size_t before = anira_test::available(h);
        ASSERT_EQ(
            anira_handler_process_multi(h, inputs.data(), 2, outputs.data(), 2, delivered.data()),
            ANIRA_OK);
        EXPECT_EQ(delivered[0], k_hop);
        EXPECT_EQ(delivered[1], k_values);
        ASSERT_NO_FATAL_FAILURE(anira_test::wait_for_block(h, before));
        ASSERT_NO_FATAL_FAILURE(anira_test::expect_same_block(block, expected, 5));
        Values stored{};
        const anira_tensor stored_tensor = whole_f32(stored.data(), k_values_shape);
        ASSERT_EQ(anira_handler_get_static_output(h, 1, &stored_tensor), ANIRA_OK);
        EXPECT_EQ(stored, carried) << "the pass-through's Static output of that inference";
    }

    // The ring state: a stream's count, 0 for the Static output, the refusals.
    size_t count = k_unset;
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &count), ANIRA_OK);
    EXPECT_EQ(count, primed) << "every block's inference was waited for";
    count = k_unset;
    EXPECT_EQ(anira_handler_get_available_samples(h, 1, 0, &count), ANIRA_OK);
    EXPECT_EQ(count, 0U) << "a Static output has no ring";
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 1, &count), ANIRA_ERROR_INVALID_ARGUMENT)
        << "the stream has one channel";
    EXPECT_EQ(anira_handler_get_available_samples(h, 2, 0, &count), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_handler_reset(h);
    line.reset();

    // Two refusals whose records name the slot: a block above host_block_max on the stream, a
    // tensor of another shape on the Static slot. A reset between them re-arms the latch.
    {
        std::vector<float> large(static_cast<size_t>(2) * k_hop, 0.0F);
        const std::array<float*, 1> planes{large.data()};
        const anira_tensor io = anira_test::planar_f32(planes.data(), 1, large.size());
        EXPECT_EQ(anira_handler_process(h, &io, 0, &io, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
        anira_handler_reset(h);
        std::array<float, 6> six{};
        const anira_tensor wrong = whole_f32(six.data(), {1, 6});
        EXPECT_EQ(anira_handler_set_static_input(h, 1, &wrong), ANIRA_ERROR_INVALID_ARGUMENT);
        anira_handler_reset(h);
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "input slot 0 'in' carries 16 samples", "rt"),
              1U);
    EXPECT_EQ(
        anira_test::count_records(collector, "Static input slot 1 'values_in' does not fit", "rt"),
        1U);
#else
    static_cast<void>(collector);
#endif

    // The getters and the selection entries.
    EXPECT_EQ(anira_handler_get_latency(h, 0), latency);
    EXPECT_EQ(anira_handler_get_latency(h, 1), 0U);
    std::array<uint32_t, 2> latencies{77, 77};
    uint32_t num_latencies = 2;
    EXPECT_EQ(anira_handler_get_latencies(h, &num_latencies, latencies.data()), ANIRA_OK);
    EXPECT_EQ(latencies, (std::array<uint32_t, 2>{latency, 0U}));
    EXPECT_GT(anira_handler_num_entries(h), 0U);
    EXPECT_NE(anira_handler_plan_report(h), nullptr);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    EXPECT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_plan(h, 1), ANIRA_ERROR_CONFIG) << "one plan";
    EXPECT_EQ(anira_handler_set_model(h, 0), ANIRA_OK);
    EXPECT_EQ(anira_handler_set_model(h, 1), ANIRA_ERROR_NOT_FOUND);
    anira_backend_id backend = ANIRA_BACKEND_ID_INIT;
    backend.engine = static_cast<uint32_t>(ANIRA_ENGINE_CUSTOM);
    backend.engine_id = anira_test::k_custom;
    EXPECT_EQ(anira_handler_set_backend(h, &backend), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // The stream again behind the resets, then one block through a _wait twin.
    ASSERT_NO_FATAL_FAILURE(run_stream(h, line, 10, 3));
    {
        std::vector<float> block = anira_test::ramp(13, k_hop);
        const std::array<float*, 1> planes{block.data()};
        const anira_tensor io = anira_test::planar_f32(planes.data(), 1, k_hop);
        size_t delivered = k_unset;
        EXPECT_EQ(anira_handler_process_wait(h,
                                             &io,
                                             0,
                                             &io,
                                             0,
                                             &delivered,
                                             1000.0 * anira_test::k_wait_s),
                  ANIRA_OK);
        EXPECT_EQ(delivered, k_hop);
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

}  // namespace

// ==== the slot signature ======================================================================

// The signature is the handler's own record: the canonical names from anira_handler_create (its
// own strings, not the pipeline copy's), the empty tensors from anira_handler_prepare (rank 2,
// {channels, 0}, the slot's dtype, no memory), and the staging arrays of the single forms
// starting as copies of them.
TEST(AbiSlotSignature, CreateFixesTheNamesAndPrepareTheEmptyTensors) {
    const Context context;
    Handler handler(context, two_slot_model());
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const anira::capi::SlotSignature& signature = h->m_signature;
    EXPECT_EQ(signature.m_inputs.m_names, (std::vector<std::string>{"in", "values_in"}));
    EXPECT_EQ(signature.m_outputs.m_names, (std::vector<std::string>{"out", "values_out"}));
    EXPECT_EQ(&signature.side(true), &signature.m_inputs);
    EXPECT_EQ(&signature.side(false), &signature.m_outputs);
    ASSERT_EQ(h->m_pipeline.m_variants.size(), 1U);
    EXPECT_NE(signature.m_inputs.m_names[0].c_str(),
              h->m_pipeline.m_variants[0].m_inputs[0].m_name.c_str())
        << "the handler's own copy of the name";
    EXPECT_TRUE(signature.m_inputs.m_empty.empty()) << "prepare builds the empty tensors";
    EXPECT_TRUE(signature.m_outputs.m_empty.empty());

    ASSERT_EQ(handler.prepare(hop_contract()), ANIRA_OK) << handler.m_err.message;
    for (const bool input : {true, false}) {
        SCOPED_TRACE(input ? "inputs" : "outputs");
        const std::vector<anira_tensor>& empty = signature.side(input).m_empty;
        ASSERT_EQ(empty.size(), 2U);
        for (const anira_tensor& tensor : empty) {
            EXPECT_EQ(tensor.ndim, 2U);
            EXPECT_EQ(tensor.shape[0], 1) << "one channel, and 1 for a slot without a host block";
            EXPECT_EQ(tensor.shape[1], 0) << "an empty tensor";
            EXPECT_EQ(tensor.dtype, static_cast<anira_dtype>(ANIRA_DTYPE_F32));
            EXPECT_EQ(tensor.domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        }
        const std::vector<anira_tensor>& staged = input ? h->m_staged_inputs : h->m_staged_outputs;
        ASSERT_EQ(staged.size(), 2U);
        EXPECT_NE(staged.data(), empty.data()) << "an array of the driving thread's own";
        for (size_t slot = 0; slot < 2; ++slot) {
            EXPECT_EQ(bytes_of(staged[slot]), bytes_of(empty[slot])) << "slot " << slot;
        }
    }
}

// Nothing a host calls while the stream runs reads the pipeline copy or a variant's derivation:
// with the copy's variants and the handler's per-variant derivations taken away behind the
// prepare (what a removed first variant leaves of them), every Hard entry, the Static entries,
// the ring state, the selection entries, the getters, reset and a _wait twin still do their
// job, and a refusal's record still names the slot by its canonical name, off the signature.
TEST(AbiSlotSignature, NoEntryOfARunningStreamReadsThePipelineCopy) {
    // The records an earlier test of the binary left in the real-time queue go first.
    anira_drain_log();
    RecordCollector collector;
    const Context context;
    Handler handler(context, two_slot_model());
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    ASSERT_EQ(handler.prepare(hop_contract()), ANIRA_OK) << handler.m_err.message;

    // Taken away for the stream, and put back behind it.
    std::vector<anira_model_config> variants = std::move(h->m_pipeline.m_variants);
    h->m_pipeline.m_variants.clear();
    std::vector<anira::capi::Derived> derivations = std::move(h->m_variant_derived);
    h->m_variant_derived.clear();
    drive_every_entry(h, collector);
    h->m_pipeline.m_variants = std::move(variants);
    h->m_variant_derived = std::move(derivations);
}

// ==== the value getters =======================================================================

// The five value getters answer from the handler's own atomics: the selection is the handler's
// atomic, which the session refers to; the latencies and the entry count are copies prepare
// stores at every sizing; the report is a published pointer. Unprepared they answer 0, NULL and
// ANIRA_ERROR_NOT_PREPARED, before the first prepare and behind a refused one.
TEST(AbiGetters, TheValueGettersAnswerFromTheHandlersOwnAtomics) {
    const Context context;
    Handler handler(context, two_slot_model());
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const auto expect_unprepared = [h](const char* when) {
        SCOPED_TRACE(when);
        EXPECT_EQ(anira_handler_get_plan(h), 0U);
        EXPECT_EQ(anira_handler_get_latency(h, 0), 0U);
        uint32_t count = 0;
        EXPECT_EQ(anira_handler_get_latencies(h, &count, nullptr), ANIRA_ERROR_NOT_PREPARED);
        EXPECT_EQ(anira_handler_num_entries(h), 0U);
        EXPECT_EQ(anira_handler_plan_report(h), nullptr);
        EXPECT_EQ(h->m_published_report.load(), nullptr);
        EXPECT_EQ(h->m_num_entries.load(), 0U);
        for (const std::atomic<uint32_t>& latency : h->m_latencies) {
            EXPECT_EQ(latency.load(), 0U);
        }
    };
    ASSERT_EQ(h->m_latencies.size(), 2U) << "one figure per output slot, from the constructor";
    expect_unprepared("before the first prepare");

    ASSERT_EQ(handler.prepare(hop_contract()), ANIRA_OK) << handler.m_err.message;
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(h);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(&session->m_current_plan, &h->m_selected_plan)
        << "the session's selection is the handler's atomic";
    ASSERT_FALSE(session->m_latency.empty());
    EXPECT_GT(session->m_latency[0], 0U);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        const uint32_t expected =
            slot < session->m_latency.size() ? static_cast<uint32_t>(session->m_latency[slot]) : 0U;
        EXPECT_EQ(h->m_latencies[slot].load(), expected) << "slot " << slot;
        EXPECT_EQ(anira_handler_get_latency(h, slot), expected) << "slot " << slot;
    }
    EXPECT_EQ(anira_handler_get_latency(h, 2), 0U) << "no such slot";
    std::array<uint32_t, 2> latencies{};
    uint32_t count = 1;
    EXPECT_EQ(anira_handler_get_latencies(h, &count, latencies.data()), ANIRA_INCOMPLETE);
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(latencies[0], session->m_latency[0]);
    EXPECT_GT(anira_handler_num_entries(h), 0U);
    EXPECT_EQ(anira_handler_num_entries(h), h->m_pp->num_entries());
    EXPECT_EQ(anira_handler_num_entries(h), h->m_num_entries.load());
    EXPECT_EQ(anira_handler_plan_report(h), h->m_reports.back().get());
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    ASSERT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    EXPECT_EQ(h->m_manager->get_plan(), anira_handler_get_plan(h));

    // A refused prepare unprepares the handler, and the getters with it: a CALLBACK miss policy
    // without its function.
    EXPECT_EQ(handler.prepare(hop_contract(ANIRA_MISS_CALLBACK)), ANIRA_ERROR_CONFIG);
    expect_unprepared("behind a refused prepare");

    ASSERT_EQ(handler.prepare(hop_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(anira_handler_plan_report(h), h->m_reports.back().get());
    EXPECT_GT(anira_handler_get_latency(h, 0), 0U);
}

// The getters never reach into the session, so they may run while a prepare on another thread
// replaces it: four threads read every value getter while this one prepares the handler again
// and again under one contract. Every answer is the unprepared one (0, NULL,
// ANIRA_ERROR_NOT_PREPARED) or the prepared one, which is the same after every prepare; the
// thread sanitizer reports a getter that reads what the prepare writes.
TEST(AbiGetters, AGetterBesideAPrepareAnswersUnpreparedOrPrepared) {
    constexpr int k_readers = 4;
    constexpr int k_prepares = 6;
    const Context context;
    Handler handler(context, two_slot_model());
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const anira::ContractHandle contract = hop_contract();
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    const uint32_t latency = anira_handler_get_latency(h, 0);
    const uint32_t entries = anira_handler_num_entries(h);
    const anira_plan_report* const report = anira_handler_plan_report(h);
    ASSERT_GT(latency, 0U);
    ASSERT_GT(entries, 0U);
    ASSERT_NE(report, nullptr);

    std::atomic<bool> stop{false};
    std::atomic<int> wrong{0};
    std::array<std::atomic<uint64_t>, k_readers> rounds{};
    std::vector<std::thread> readers;
    readers.reserve(k_readers);
    for (size_t reader = 0; reader < k_readers; ++reader) {
        readers.emplace_back([&, reader] {
            while (!stop.load(std::memory_order_acquire)) {
                if (anira_handler_get_plan(h) != 0U) { wrong.fetch_add(1); }
                const uint32_t one = anira_handler_get_latency(h, 0);
                if (one != 0U && one != latency) { wrong.fetch_add(1); }
                std::array<uint32_t, 2> all{};
                uint32_t count = 2;
                const anira_status status = anira_handler_get_latencies(h, &count, all.data());
                if (status == ANIRA_OK) {
                    // A prepare between the flag and the loads may have zeroed a figure.
                    if ((all[0] != 0U && all[0] != latency) || all[1] != 0U) { wrong.fetch_add(1); }
                } else if (status != ANIRA_ERROR_NOT_PREPARED) {
                    wrong.fetch_add(1);
                }
                const uint32_t seen_entries = anira_handler_num_entries(h);
                if (seen_entries != 0U && seen_entries != entries) { wrong.fetch_add(1); }
                const anira_plan_report* const seen_report = anira_handler_plan_report(h);
                if (seen_report != nullptr && seen_report != report) { wrong.fetch_add(1); }
                if (anira_handler_rt_error(h) != ANIRA_OK) { wrong.fetch_add(1); }
                rounds.at(reader).fetch_add(1, std::memory_order_release);
            }
        });
    }
    // Every reader goes round at least once between two prepares, and once behind the last.
    const auto wait_for_rounds = [&rounds] {
        std::array<uint64_t, k_readers> seen{};
        for (size_t i = 0; i < seen.size(); ++i) { seen.at(i) = rounds.at(i).load(); }
        for (size_t i = 0; i < seen.size(); ++i) {
            while (rounds.at(i).load(std::memory_order_acquire) == seen.at(i)) {
                std::this_thread::yield();
            }
        }
    };
    wait_for_rounds();
    anira_status last = ANIRA_OK;
    for (int k = 0; k < k_prepares && last == ANIRA_OK; ++k) {
        last = handler.prepare(contract);
        wait_for_rounds();
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& reader : readers) { reader.join(); }
    EXPECT_EQ(last, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(wrong.load(), 0);
    EXPECT_EQ(anira_handler_get_latency(h, 0), latency) << "every prepare sizes the same stream";
    EXPECT_EQ(anira_handler_num_entries(h), entries);
    EXPECT_EQ(anira_handler_plan_report(h), report);
}

// ==== the control lock ========================================================================

namespace {

/// A processor that fills prepare and unprepare alone: its prepare notes whether the handler's
/// control lock records the thread it runs on, says that it is inside, and waits there while
/// the gate is closed; both callbacks append to the order of events.
struct GatedPrepare {
    std::atomic<bool> m_open{true};
    std::atomic<bool> m_inside{false};
    std::atomic<int> m_held_by_caller{0};  ///< prepares that found their own thread as holder
    std::atomic<int> m_held_by_other{0};   ///< prepares that found another one, or nobody
    std::mutex m_book;
    std::vector<std::string> m_events;

    void note(const char* event) {
        const std::scoped_lock<std::mutex> lock(m_book);
        m_events.emplace_back(event);
    }
    std::vector<std::string> events() {
        const std::scoped_lock<std::mutex> lock(m_book);
        return m_events;
    }

    static anira_status ANIRA_CALL prepare(const anira_prepare_info* info,
                                           void* user_data,
                                           void** out_prepared) {
        auto* self = static_cast<GatedPrepare*>(user_data);
        self->note("prepare enters");
        if (info->handler->m_control_holder.load() == std::this_thread::get_id()) {
            self->m_held_by_caller.fetch_add(1);
        } else {
            self->m_held_by_other.fetch_add(1);
        }
        self->m_inside.store(true);
        while (!self->m_open.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        *out_prepared = nullptr;
        self->note("prepare returns");
        return ANIRA_OK;
    }
    static void ANIRA_CALL unprepare(void* /*prepared*/, void* user_data) {
        static_cast<GatedPrepare*>(user_data)->note("unprepare");
    }

    anira_processor_desc desc() {
        anira_processor_desc desc = ANIRA_PROCESSOR_DESC_INIT;
        desc.user_data = this;
        desc.prepare = prepare;
        desc.unprepare = unprepare;
        return desc;
    }
};

}  // namespace

// A prepare runs under the handler's control lock, and the handler records the thread that
// holds it: inside a callback of the prepare the holder is the calling thread, and nobody
// behind the prepare, whichever thread prepares.
TEST(AbiControlLock, APrepareHoldsTheLockAndRecordsItsThread) {
    const Context context;
    GatedPrepare gated;
    const std::array<anira_processor_desc, 1> processors{gated.desc()};
    Handler handler(context, two_slot_model(), {}, processors);
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->m_control_holder.load(), std::thread::id()) << "nobody holds it";
    ASSERT_EQ(handler.prepare(hop_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(gated.m_held_by_caller.load(), 1) << "the preparing thread, inside its callback";
    EXPECT_EQ(h->m_control_holder.load(), std::thread::id()) << "given back with the lock";
    anira_status status = ANIRA_ERROR_UNKNOWN;
    std::thread other([&] { status = handler.prepare(hop_contract()); });
    other.join();
    EXPECT_EQ(status, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(gated.m_held_by_caller.load(), 2) << "the other thread, inside its callback";
    EXPECT_EQ(gated.m_held_by_other.load(), 0);
    EXPECT_EQ(h->m_control_holder.load(), std::thread::id());
}

// anira_handler_destroy waits for a prepare on another thread: with the prepare held inside its
// last callback, a destroy from a second thread does nothing until the prepare has returned, so
// the processor sees its prepare return, then its unprepare, once, and the prepare succeeds.
// (Without the lock the destroy would release the session under the prepare's feet, and the
// unprepare, owed only once the prepare callback has returned, would never be called.)
TEST(AbiControlLock, ADestroyWaitsForAPrepareOnAnotherThread) {
    const Context context;
    GatedPrepare gated;
    gated.m_open.store(false);
    const std::array<anira_processor_desc, 1> processors{gated.desc()};
    Handler handler(context, two_slot_model(), {}, processors);
    anira_handler* const h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const anira::ContractHandle contract = hop_contract();
    anira_status prepared = ANIRA_ERROR_UNKNOWN;
    anira_error prepare_err = ANIRA_ERROR_INIT;
    std::thread preparing(
        [&] { prepared = anira_handler_prepare(h, contract.native(), &prepare_err); });
    while (!gated.m_inside.load()) { std::this_thread::yield(); }
    // The prepare is inside its callback and holds the lock: the destroy waits for it.
    std::atomic<bool> destroying{false};
    std::thread destroyer([&] {
        destroying.store(true);
        handler.destroy();
    });
    while (!destroying.load()) { std::this_thread::yield(); }
    EXPECT_EQ(gated.events(), (std::vector<std::string>{"prepare enters"}));
    gated.m_open.store(true);
    preparing.join();
    destroyer.join();
    EXPECT_EQ(prepared, ANIRA_OK) << prepare_err.message;
    EXPECT_EQ(gated.events(),
              (std::vector<std::string>{"prepare enters", "prepare returns", "unprepare"}));
    EXPECT_EQ(handler.m_handler, nullptr);
}

// Two threads that prepare one handler at once take turns: no prepare runs inside another's
// callbacks, each finds itself as the holder, and all succeed.
TEST(AbiControlLock, TwoPreparesOfOneHandlerTakeTurns) {
    constexpr int k_rounds = 4;
    const Context context;
    GatedPrepare gated;
    const std::array<anira_processor_desc, 1> processors{gated.desc()};
    const Handler handler(context, two_slot_model(), {}, processors);
    anira_handler* const h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const anira::ContractHandle contract = hop_contract();
    std::array<anira_status, 2> statuses{ANIRA_OK, ANIRA_OK};
    std::vector<std::thread> threads;
    threads.reserve(statuses.size());
    for (anira_status& status : statuses) {
        threads.emplace_back([h, &contract, &status] {
            for (int round = 0; round < k_rounds && status == ANIRA_OK; ++round) {
                anira_error err = ANIRA_ERROR_INIT;
                status = anira_handler_prepare(h, contract.native(), &err);
            }
        });
    }
    for (std::thread& thread : threads) { thread.join(); }
    EXPECT_EQ(statuses, (std::array<anira_status, 2>{ANIRA_OK, ANIRA_OK}));
    // Every prepare's callbacks in one piece: a prepare enters and returns, and the next one
    // unprepares it before it enters itself.
    const std::vector<std::string> events = gated.events();
    ASSERT_EQ(events.size(), (static_cast<size_t>(3) * 2 * k_rounds) - 1);
    for (size_t i = 0; i < events.size(); ++i) {
        const char* expected = "unprepare";
        if (i % 3 == 0) { expected = "prepare enters"; }
        if (i % 3 == 1) { expected = "prepare returns"; }
        EXPECT_EQ(events.at(i), expected) << "event " << i;
    }
    EXPECT_EQ(gated.m_held_by_caller.load(), 2 * k_rounds);
    EXPECT_EQ(gated.m_held_by_other.load(), 0);
}
