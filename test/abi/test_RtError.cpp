// anira/abi/handler.h: anira_handler_rt_error and its latches. The first occurrence of a
// kind records once and later ones are counted; prepare and reset re-arm, logging the count;
// the drain's summary reports a persisting condition; the scheduler's sites latch once per
// prepare; a throwing inference zero-fills, records ENGINE and keeps the pool; CAPACITY never
// latches; the violation record carries its flags and the drain thread delivers it.
#include <anira/abi/context.h>
#include <anira/abi/core.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/thread.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceThread.h>
#include <anira/utils/RtLatch.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "../support/log_record_collector.h"
#include "handler_support.h"

namespace {

using anira_test::attach_processor;
using anira_test::Context;
using anira_test::count_records;
using anira_test::custom_candidates;
using anira_test::expect_all;
using anira_test::expect_same_block;
using anira_test::explicit_contract;
using anira_test::find_record;
using anira_test::gain_with_custom;
using anira_test::GateBackend;
using anira_test::Handler;
using anira_test::k_block;
using anira_test::ramp;
using anira_test::RecordCollector;
using anira_test::SummaryInterval;
using anira_test::ThrowingBackend;
using anira_test::wait_for_block;

constexpr const char* k_missing_samples_site =
    anira::k_rt_site_names[static_cast<size_t>(anira::RtSite::MissingSamples)];

/// A context whose Warning and Info records pass the process-global level.
struct DebugContext : Context {
    explicit DebugContext(anira_log_drain drain = ANIRA_LOG_DRAIN_MANUAL, uint32_t interval_ms = 10)
        : Context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_DEBUG, drain, interval_ms) {}
};

/// The persistent INVALID_ARGUMENT: a process call with a tensor index the model has not.
size_t bad_process(anira_handler* handler) {
    float sample = 0.0F;
    float* ptrs[1] = {&sample};
    return anira_handler_process(handler, ptrs, 1, 99);
}

/// One in-place gain block, waited, returned for the assertions.
std::vector<float> waited_block(anira_handler* handler, size_t block_index) {
    std::vector<float> block = ramp(block_index);
    float* ptrs[1] = {block.data()};
    const size_t prev = anira_handler_get_available_samples(handler, 0, 0);
    EXPECT_EQ(anira_handler_process(handler, ptrs, k_block, 0), k_block) << "block " << block_index;
    wait_for_block(handler, prev);
    return block;
}

}  // namespace

// ============================================================================================
// The handler's latch
// ============================================================================================

TEST(AbiRtError, TheFirstOccurrenceRecordsOnceAndLaterOnesAreCounted) {
    const DebugContext context;
    anira_drain_log();
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_handler* h = handler.m_handler;

    std::vector<float> block(k_block, 0.0F);
    float* ptrs[1] = {block.data()};
    for (int i = 0; i < 5; ++i) { EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), 0U); }
    for (int i = 0; i < 3; ++i) { EXPECT_EQ(anira_handler_pop_data(h, ptrs, k_block, 0), 0U); }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "handler not prepared", "rt"), 1U);
    const RecordCollector::Record record = find_record(collector, "handler not prepared", "rt");
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
    EXPECT_EQ(record.m_group, "anira.capi");
    EXPECT_EQ(record.m_message, "anira_handler_process: handler not prepared")
        << "the first refusing entry names the record";
#endif

    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector,
                            "anira_handler_prepare: 7 real-time failures were suppressed since "
                            "the last prepare or reset",
                            "native"),
              1U)
        << "8 refusals minus the one logged";
#endif
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

TEST(AbiRtError, EachKindLatchesSeparatelyAndRtErrorIsLastWins) {
    const DebugContext context;
    anira_drain_log();
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    for (int i = 0; i < 3; ++i) { EXPECT_EQ(bad_process(h), 0U); }
    for (int i = 0; i < 2; ++i) { EXPECT_EQ(anira_handler_set_plan(h, 99), ANIRA_ERROR_CONFIG); }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_CONFIG);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_process: invalid argument", "rt"), 1U);
    EXPECT_EQ(count_records(collector, "anira_handler_set_plan: configuration error", "rt"), 1U);
#endif
    EXPECT_EQ(bad_process(h), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT) << "last-wins";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_process: invalid argument", "rt"), 1U);
    EXPECT_EQ(count_records(collector, "anira_handler_set_plan: configuration error", "rt"), 1U);
#endif
}

TEST(AbiRtError, ResetReArmsAndLogsTheSuppressedCount) {
    const DebugContext context;
    anira_drain_log();
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    for (int i = 0; i < 3; ++i) { EXPECT_EQ(bad_process(h), 0U); }
    for (int i = 0; i < 2; ++i) { EXPECT_EQ(anira_handler_set_plan(h, 99), ANIRA_ERROR_CONFIG); }
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector,
                            "anira_handler_reset: 3 real-time failures were suppressed since the "
                            "last prepare or reset",
                            "rt"),
              1U)
        << "5 refusals, 2 logged";
    const RecordCollector::Record record = find_record(collector, "anira_handler_reset:", "rt");
    EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_INFO));
    EXPECT_EQ(record.m_group, "anira.capi");
#endif
    EXPECT_EQ(bad_process(h), 0U);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_process: invalid argument", "rt"), 2U)
        << "the kind was re-armed";
#endif
    // A reset with nothing suppressed logs nothing.
    anira_handler_reset(h);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "anira_handler_reset:", "rt"), 1U);
#endif
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// ============================================================================================
// The summary
// ============================================================================================

TEST(AbiRtError, TheDrainSummaryReportsAPersistentCondition) {
    const SummaryInterval interval(50);
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    {
        const DebugContext context(ANIRA_LOG_DRAIN_THREAD, 1);
        RecordCollector collector;
        Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
        anira_handler* h = handler.m_handler;

        std::atomic<bool> stop{false};
        std::thread hammer([&] {
            while (!stop.load()) {
                bad_process(h);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
#ifdef ENABLE_LOGGING
        EXPECT_TRUE(collector.wait_for("handler of session", "native"));
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
#endif
        stop.store(true);
        hammer.join();
#ifdef ENABLE_LOGGING
        const RecordCollector::Record record =
            find_record(collector, "handler of session", "native");
        EXPECT_NE(record.m_message.find("real-time failures still occurring"), std::string::npos)
            << record.m_message;
        EXPECT_NE(record.m_message.find("(last status invalid argument)"), std::string::npos)
            << record.m_message;
        EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_WARNING));
        EXPECT_EQ(record.m_group, "anira.scheduler");
        // A counter that did not grow is not reported.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const size_t settled = count_records(collector, "handler of session", "native");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_EQ(count_records(collector, "handler of session", "native"), settled);
        // The re-arm zeroes both counters: nothing grew, nothing is reported.
        anira_handler_reset(h);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        EXPECT_EQ(count_records(collector, "handler of session", "native"), settled);
#endif
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    }
    {
        // MANUAL drain: the host's pump runs the summary too.
        const DebugContext context(ANIRA_LOG_DRAIN_MANUAL);
        anira_drain_log();
        RecordCollector collector;
        Handler handler(context, model, candidates);
        ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
        anira_handler* h = handler.m_handler;
#ifdef ENABLE_LOGGING
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            bad_process(h);
            anira_drain_log();
            if (collector.has("handler of session", "native")) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_TRUE(collector.has("handler of session", "native"));
        const RecordCollector::Record record =
            find_record(collector, "handler of session", "native");
        EXPECT_NE(record.m_message.find("real-time failures still occurring"), std::string::npos)
            << record.m_message;
#endif
    }
}

TEST(AbiRtError, SiteLatchesLogOncePerPrepareAndSummarise) {
    const SummaryInterval interval(50);
    const DebugContext context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    anira_drain_log();
    RecordCollector collector;
    GateBackend gate(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, gate));
    gate.m_open.store(false);

    std::vector<float> block(k_block, 0.25F);
    float* ptrs[1] = {block.data()};
    const auto starved_call = [&] {
        anira_handler_process(h, ptrs, k_block, 0);
        anira_drain_log();
    };
    // The first call delivers the priming block, the rest starve: the S7 site latches once.
    for (int i = 0; i < 30; ++i) { starved_call(); }
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "Missing samples", "rt"), 1U);
    // The summary's site half, run from anira_drain_log().
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!collector.has(k_missing_samples_site, "native") &&
           std::chrono::steady_clock::now() < deadline) {
        starved_call();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(collector.has(k_missing_samples_site, "native"));
    const RecordCollector::Record summary =
        find_record(collector, k_missing_samples_site, "native");
    EXPECT_NE(summary.m_message.find("is still failing"), std::string::npos) << summary.m_message;
    for (int i = 0; i < 10; ++i) { starved_call(); }
    EXPECT_EQ(count_records(collector, "Missing samples", "rt"), 1U)
        << "the summary never re-arms a site";
#endif

    // A prepare re-arms the site, logging the count suppressed since the last prepare.
    gate.m_open.store(true);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
#ifdef ENABLE_LOGGING
    // "real-time condition 'missing samples': N occurrences suppressed since the last prepare"
    // (the summary's site half says "' is still failing" instead).
    const std::string rearm_site =
        std::string("real-time condition '") + k_missing_samples_site + "': ";
    EXPECT_GE(count_records(collector, rearm_site.c_str(), "native"), 1U);
    const RecordCollector::Record rearm = find_record(collector, rearm_site.c_str(), "native");
    EXPECT_NE(rearm.m_message.find("suppressed since the last prepare"), std::string::npos)
        << rearm.m_message;
    EXPECT_EQ(rearm.m_level, static_cast<uint32_t>(ANIRA_LOG_WARNING));
    EXPECT_EQ(rearm.m_group, "anira.scheduler");
#endif
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, gate));
    gate.m_open.store(false);
    for (int i = 0; i < 6; ++i) { starved_call(); }
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "Missing samples", "rt"), 2U);
#endif
    gate.m_open.store(true);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK) << "a miss records nothing";
}

// ============================================================================================
// A failed inference
// ============================================================================================

TEST(AbiRtError, EngineAfterAThrowingInferenceZeroFillsAndKeepsTheThread) {
    const DebugContext context;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    ASSERT_EQ(handler.prepare(explicit_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    anira_drain_log();
    RecordCollector collector;
    ThrowingBackend backend(h->m_inference_config);
    ASSERT_NO_FATAL_FAILURE(attach_processor(h, backend));

    {
        std::vector<float> block = ramp(1);
        float* ptrs[1] = {block.data()};
        const size_t prev = anira_handler_get_available_samples(h, 0, 0);
        EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), k_block);
        expect_all(block, 0.0F, "block 1: the priming zeros");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (anira_handler_rt_error(h) != ANIRA_ERROR_ENGINE &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_ENGINE)
            << "the inference thread records it";
        wait_for_block(h, prev);
    }
    expect_all(waited_block(h, 2), 0.0F, "block 2: the failed inference's zeros");
    for (size_t k = 3; k <= 5; ++k) { waited_block(h, k); }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector, "inference failed in session", "rt"), 1U);
    const RecordCollector::Record record =
        find_record(collector, "inference failed in session", "rt");
    EXPECT_NE(record.m_message.find("test backend: inference failed"), std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("delivering zeros"), std::string::npos) << record.m_message;
    EXPECT_EQ(record.m_flags & ANIRA_LOG_RECORD_CONTRACT_VIOLATION, 0U);
    EXPECT_EQ(record.m_level, static_cast<uint32_t>(ANIRA_LOG_ERROR));
    EXPECT_EQ(record.m_group, "anira.scheduler");
#endif
    // The pool is intact.
    EXPECT_EQ(anira_num_inference_threads(), 2U);
    EXPECT_EQ(anira::InferenceThread::get_num_loop_active(), 2U);
    EXPECT_TRUE(anira::Core::has_inference_threads());

    // The pass-through resumes: block k delivers ramp(k - 1).
    backend.m_throw.store(false);
    expect_all(waited_block(h, 6), 0.0F, "block 6: inference 5 threw");
    expect_same_block(waited_block(h, 7), ramp(6), 7);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_ENGINE) << "nothing cleared it";
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(count_records(collector,
                            "anira_handler_reset: 4 real-time failures were suppressed since the "
                            "last prepare or reset",
                            "rt"),
              1U)
        << "5 throws, 1 logged";
#endif
}

// ============================================================================================
// The latch itself, the flags
// ============================================================================================

TEST(AbiRtLatch, CapacityNeverLatches) {
    anira::RtLatch latch;
    EXPECT_EQ(anira::rt_kind_bit(ANIRA_ERROR_CAPACITY), 0U);
    EXPECT_FALSE(latch.record(ANIRA_ERROR_CAPACITY));
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);
    EXPECT_EQ(latch.m_latched.load(), 0U);
    EXPECT_EQ(latch.m_suppressed.load(), 0U);

    EXPECT_TRUE(latch.record(ANIRA_ERROR_WRONG_CONTRACT));
    EXPECT_FALSE(latch.record(ANIRA_ERROR_WRONG_CONTRACT));
    EXPECT_EQ(latch.m_suppressed.load(), 1U);
    EXPECT_TRUE(latch.record(ANIRA_ERROR_ENGINE));
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_ENGINE);

    latch.m_reported.store(1);
    EXPECT_EQ(latch.rearm(), 1U);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);
    EXPECT_EQ(latch.m_reported.load(), 0U);
    EXPECT_TRUE(latch.record(ANIRA_ERROR_WRONG_CONTRACT)) << "re-armed";

    for (anira_status status : {ANIRA_ERROR_WRONG_CONTRACT,
                                ANIRA_ERROR_NOT_PREPARED,
                                ANIRA_ERROR_CONFIG,
                                ANIRA_ERROR_INVALID_STATE,
                                ANIRA_ERROR_INVALID_ARGUMENT,
                                ANIRA_ERROR_ENGINE}) {
        EXPECT_NE(anira::rt_kind_bit(status), 0U) << status;
    }
    for (anira_status status : {ANIRA_OK, ANIRA_TIMEOUT, ANIRA_ERROR_INTERNAL}) {
        EXPECT_EQ(anira::rt_kind_bit(status), 0U) << status;
    }

    // The per-site form.
    anira::RtLatch site;
    EXPECT_TRUE(site.first());
    EXPECT_FALSE(site.first());
    EXPECT_FALSE(site.first());
    EXPECT_EQ(site.rearm(), 2U);
    EXPECT_TRUE(site.first());
}

#if !defined(__EMSCRIPTEN__)
TEST(AbiRtError, AViolationRecordCarriesTheFlagsAndTheDrainDeliversIt) {
    const DebugContext context(ANIRA_LOG_DRAIN_THREAD, 1);
    RecordCollector collector;
    const anira::ModelConfig model = gain_with_custom();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    Handler handler(context, model, candidates);
    anira_handler* h = handler.m_handler;
    std::vector<float> block(k_block, 0.0F);
    float* ptrs[1] = {block.data()};
    EXPECT_EQ(anira_handler_process(h, ptrs, k_block, 0), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
#ifdef ENABLE_LOGGING
    EXPECT_TRUE(collector.wait_for("handler not prepared", "rt")) << "the drain thread delivers";
    const RecordCollector::Record record = find_record(collector, "handler not prepared", "rt");
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
#endif
}
#endif
