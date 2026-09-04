#include <anira/abi/status.h>
#include <anira/utils/Logger.h>
#include <gtest/gtest.h>

#include <cstring>
#include <mutex>
#include <string>

#include "capi/capi_internal.h"

using anira::capi::firewall_probe;

TEST(AbiFirewall, SuccessWritesTheOutParameterAndLeavesErrAlone) {
    anira_error err = ANIRA_ERROR_INIT;
    int value = -1;
    EXPECT_EQ(firewall_probe(0, ANIRA_OK, nullptr, &err, &value), ANIRA_OK);
    EXPECT_EQ(value, 42);
    EXPECT_EQ(err.status, ANIRA_OK);
    EXPECT_EQ(err.message[0], '\0');
}

TEST(AbiFirewall, BadAllocBecomesOutOfMemory) {
    anira_error err = ANIRA_ERROR_INIT;
    int value = -1;
    EXPECT_EQ(firewall_probe(1, ANIRA_OK, nullptr, &err, &value), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(err.status, ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_STREQ(err.message, "out of memory");
    EXPECT_EQ(value, -1) << "out-parameters are written only on success";
}

TEST(AbiFirewall, StatusErrorCarriesItsStatusAndMessage) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "models[0].engine: unknown", &err, nullptr),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(err.status, ANIRA_ERROR_JSON);
    EXPECT_STREQ(err.message, "models[0].engine: unknown");
}

TEST(AbiFirewall, InvalidArgumentBecomesConfig) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(3, ANIRA_OK, "bad shape", &err, nullptr), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(err.status, ANIRA_ERROR_CONFIG);
    EXPECT_STREQ(err.message, "bad shape");
}

TEST(AbiFirewall, OtherExceptionsBecomeInternal) {
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(4, ANIRA_OK, "engine exploded", &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_STREQ(err.message, "engine exploded");
    err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(5, ANIRA_OK, nullptr, &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_STREQ(err.message, "unknown exception");
}

TEST(AbiFirewall, NullErrIsAccepted) {
    EXPECT_EQ(firewall_probe(1, ANIRA_OK, nullptr, nullptr, nullptr), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_DEVICE, "x", nullptr, nullptr), ANIRA_ERROR_DEVICE);
}

TEST(AbiFirewall, LongMessagesAreTruncatedAndTerminated) {
    anira_error err = ANIRA_ERROR_INIT;
    const std::string long_message(700, 'm');
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_CONFIG, long_message.c_str(), &err, nullptr),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(std::strlen(err.message), static_cast<size_t>(ANIRA_ERROR_MESSAGE_CAPACITY - 1));
    EXPECT_EQ(err.message[ANIRA_ERROR_MESSAGE_CAPACITY - 1], '\0');
}

// ---- the error strategy: what the firewall logs, and what it never logs ------------------

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/abi/log.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <vector>

#include "../support/log_record_collector.h"

namespace {

using anira::capi::firewall_probe_void;
using anira::capi::set_trace_failures;
using anira_test::RecordCollector;

// The firewall logs on the sync path, which the collector sees immediately.
int native_records(RecordCollector& collector, const char* fragment) {
    int count = 0;
    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const RecordCollector::Record& record : collector.m_records) {
        if (record.m_source == "native" && record.m_message.find(fragment) != std::string::npos) {
            ++count;
        }
    }
    return count;
}

int native_record_count(RecordCollector& collector) {
    int count = 0;
    const std::scoped_lock<std::mutex> lock(collector.m_mutex);
    for (const RecordCollector::Record& record : collector.m_records) {
        if (record.m_source == "native") { ++count; }
    }
    return count;
}

struct TraceOff {
    TraceOff() { set_trace_failures(false); }
    ~TraceOff() { set_trace_failures(false); }
    TraceOff(const TraceOff&) = delete;
    TraceOff& operator=(const TraceOff&) = delete;
};

/// A collector that starts empty: a freshly set callback receives the logger's early-buffered
/// records of earlier tests (and, on a leg that runs every suite in one process, of other
/// suites), which are not under test here.
struct FreshCollector : RecordCollector {
    FreshCollector() { m_records.clear(); }
};

}  // namespace

TEST(AbiFirewallLogging, AClassifiedStatusLogsNothing) {
    const TraceOff off;
    FreshCollector collector;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(1, ANIRA_OK, nullptr, &err, nullptr), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "models[0]: bad", &err, nullptr),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(firewall_probe(3, ANIRA_OK, "bad shape", &err, nullptr), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(native_record_count(collector), 0) << "a returned failure is not logged";
}

TEST(AbiFirewallLogging, InternalLogsExactlyOnceWithTheEntryName) {
    const TraceOff off;
    FreshCollector collector;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(4, ANIRA_OK, "unexpected", &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_STREQ(err.message, "unexpected");
    EXPECT_EQ(native_records(collector, "firewall_probe: unexpected"), 1);
    EXPECT_EQ(native_record_count(collector), 1);
    EXPECT_EQ(firewall_probe(5, ANIRA_OK, nullptr, &err, nullptr), ANIRA_ERROR_INTERNAL);
    EXPECT_EQ(native_records(collector, "firewall_probe: unknown exception"), 1);
}

TEST(AbiFirewallLogging, AVoidEntryReportsASwallowedFailureOnce) {
    const TraceOff off;
    FreshCollector collector;
    firewall_probe_void(0, nullptr, false);
    EXPECT_EQ(native_record_count(collector), 0) << "success logs nothing";
    firewall_probe_void(3, "destroy went wrong", false);
    EXPECT_EQ(native_records(collector, "firewall_probe_void: destroy went wrong"), 1);
    EXPECT_EQ(native_record_count(collector), 1);
}

TEST(AbiFirewallLogging, TheQuietHandlerNeverLogs) {
    const TraceOff off;
    FreshCollector collector;
    firewall_probe_void(4, "sink threw", true);
    EXPECT_EQ(native_record_count(collector), 0) << "a throwing sink must not recurse";
}

TEST(AbiFirewallLogging, TheTraceFlagEmitsTheErrorMessageOnce) {
    const TraceOff off;
    FreshCollector collector;
    anira_error err = ANIRA_ERROR_INIT;
    set_trace_failures(true);
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "models[0].engine: unknown", &err, nullptr),
              ANIRA_ERROR_JSON);
    EXPECT_EQ(native_records(collector, "firewall_probe: "), 1);
    EXPECT_EQ(native_records(collector, "models[0].engine: unknown"), 1)
        << "the record carries the same bytes as the error message";
    EXPECT_EQ(native_records(collector, anira_status_string(ANIRA_ERROR_JSON)), 1);
    // Without an error record the trace still carries the message.
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "no record", nullptr, nullptr), ANIRA_ERROR_JSON);
    EXPECT_EQ(native_records(collector, "no record"), 1);
    set_trace_failures(false);
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "silent again", &err, nullptr), ANIRA_ERROR_JSON);
    EXPECT_EQ(native_records(collector, "silent again"), 0);
}

TEST(AbiFirewallLogging, AFailingEntryDrainsTheRealTimeQueueFirst) {
    const TraceOff off;
    // Manual drain: nothing pumps the queue but anira_drain_log, or a failing entry.
    anira::CoreConfig core_config(1, anira::WaitStrategy::Blocking, anira::LogLevel::Error);
    core_config.m_log.m_drain = anira::LogDrain::Manual;
    anira::InferenceConfig inference_config(
        std::vector<anira::ModelData>{
            anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
        std::vector<anira::TensorShape>{anira::TensorShape({{1, 1, 512}}, {{1, 1, 512}})},
        1.f,
        0,
        false,
        0.f,
        2);
    anira::PrePostProcessor pp_processor(inference_config);
    anira::InferenceHandler handler(pp_processor, inference_config, core_config);
    handler.prepare(anira::HostConfig(512, 48000));

    FreshCollector collector;
    anira_log_rt(ANIRA_LOG_ERROR, "anira.test", "queued before the failure", 7, 8);
    EXPECT_FALSE(collector.has("queued before the failure")) << "still in the queue";
    anira_error err = ANIRA_ERROR_INIT;
    anira_model_config* config = nullptr;
    const char* text = "{not json";
    EXPECT_EQ(anira_model_config_from_json(text, 9, nullptr, &config, &err), ANIRA_ERROR_JSON);
    EXPECT_TRUE(collector.has("queued before the failure"))
        << "the failing entry delivered the real-time record before returning";
}

TEST(AbiFirewallLogging, ASinkThatFailsAnEntryDoesNotRecurse) {
    const TraceOff off;
    // A callback sink that calls a failing entry from inside the sink: the depth guard keeps
    // the nested failure from draining (and from re-entering this sink) again.
    static int nested_calls = 0;
    nested_calls = 0;
    struct ReentrantSink {
        static void on_record(const anira_log_record* record, void* /*user_data*/) {
            if (std::strstr(record->message, "outer failure") == nullptr) { return; }
            if (nested_calls++ > 0) { return; }
            anira_error inner = ANIRA_ERROR_INIT;
            anira_model_config* config = nullptr;
            static_cast<void>(anira_model_config_from_json("{", 1, nullptr, &config, &inner));
        }
    };
    const anira::detail::LogSinkId sink =
        anira::detail::add_log_sink(&ReentrantSink::on_record, nullptr, ANIRA_LOG_DEBUG);
    set_trace_failures(true);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(firewall_probe(2, ANIRA_ERROR_JSON, "outer failure", &err, nullptr),
              ANIRA_ERROR_JSON);
    set_trace_failures(false);
    anira::detail::remove_log_sink(sink);
    EXPECT_EQ(nested_calls, 1) << "the inner failure ran once and did not re-enter the sink";
}
