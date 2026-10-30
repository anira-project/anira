#ifndef ANIRA_TEST_COPY_ORACLE_H
#define ANIRA_TEST_COPY_ORACLE_H

// The tools of the copy-path recording (test/scheduler/test_CopyPathOracle.cpp over the tensor
// stems of InferenceManager, test/abi/test_HandlerCopyOracle.cpp over the Hard entries of the C
// ABI, both through an anira::PlanarFloatAdapter; the transcripts were recorded under the
// float*** functions and the _f32 entries those calls replaced, and are not re-recorded): the
// host memory of a call, the input ramps, the transcript a scenario writes
// and its comparison with the golden transcript, and the clocked gate that makes a scenario
// a pure function of its calls.
//
// A transcript is text: one header line per call ("#<n> <call> ...") and one line per host
// run the call could have written. A float is printed bit-exactly and readably
// (format_value), so a golden transcript is a literal a reviewer can read, and a difference
// names the call and the run it happened in.
//
// Determinism. The session's model is a gate (test/abi/handler_support.h: a GateEngine added to
// the C handler's pipeline) that a scenario keeps closed while it calls the library: an
// inference submitted by a call is held on its inference thread, so no call ever sees a result
// of its own submissions, whatever the machine load or the sanitizer. Between two calls
// settle() opens the gate, collects until the session has no submitted inference left (the
// driver-thread list SessionElement::m_time_stamps is empty) and closes the gate again: the
// next call finds every earlier result in the receive rings, in submission order. A scenario
// that wants a starved block leaves the settle out. Nothing is synchronised by a sleep of a
// chosen length and nothing is measured against the wall clock; the one wait is a poll with a
// failure timeout that a passing run never reaches.
//
// Two kinds of call wait inside the library and cannot run against a closed gate. The waiting
// stems of InferenceManager are recorded on a non-real-time session, where every collection
// waits for every submitted inference whatever the budget; the _wait twins of the C entries
// with ANIRA_WAIT_FOREVER, the gate open for the call. Either way the call returns with every
// submitted inference collected, its own included: as much a pure function of the calls. A
// finite budget or timeout makes the block a matter of the wall clock and is not recorded.
//
// The sessions. Every recording runs on a C handler (HandlerSession below) whose one model
// entry is the custom row and whose engine is the gate. The C entries' rigs call the Hard
// entries; the stems' rigs (test_CopyPathOracle, test_TensorStems, the referee of
// test_HandlerCopyOracle) call the tensor stems of the handler's own session directly, below
// the C entries and their block rule, so any count goes there, as it went on the 2.x session
// the scheduler's transcripts were recorded on. The models are the 2.x configurations of the
// recordings, figure for figure: hop 8 at 800 Hz, a budget of 1 ms, no warm-up, two instances
// on two threads.

#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/processor.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/RingBuffer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "capi/port.h"

namespace anira_test::oracle {

/// What every float of a call's output memory holds before the call: a value no ramp and no
/// miss policy produces, printed as ".".
inline constexpr float k_untouched = -7.0F;

/// Floats past the requested count that every run carries and the transcript prints, so a
/// write beyond the request shows.
inline constexpr size_t k_guard = 2;

/// Set in the environment to a file path, the scenarios append their transcripts to that
/// file as the literals of a golden header and compare nothing. It is how the golden headers
/// were produced; read their comments before ever using it again.
inline constexpr const char* k_record_env = "ANIRA_COPY_ORACLE_RECORD";

/// One MSVC string literal holds 16380 bytes; a transcript stays below that.
inline constexpr size_t k_max_transcript_bytes = 15000;

/// A float, bit-exact: "." for k_untouched, the integer for an integral value below 2^24
/// (the negative zero is "-0"), and the bit pattern in hex for anything else.
inline std::string format_value(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t untouched_bits = 0;
    std::memcpy(&untouched_bits, &k_untouched, sizeof(untouched_bits));
    if (bits == untouched_bits) { return "."; }
    constexpr float k_exact_limit = 16777216.0F;
    if (std::isfinite(value) && std::fabs(value) < k_exact_limit) {
        const auto whole = static_cast<int32_t>(value);
        if (static_cast<float>(whole) == value) {
            if (whole == 0 && std::signbit(value)) { return "-0"; }
            return std::to_string(whole);
        }
    }
    constexpr std::string_view k_digits = "0123456789abcdef";
    std::string text = "0x";
    for (int shift = 28; shift >= 0; shift -= 4) { text += k_digits[(bits >> shift) & 0xFU]; }
    return text;
}

/// "[8,3]"
inline std::string format_counts(std::span<const size_t> counts) {
    std::string text = "[";
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i > 0) { text += ","; }
        text += std::to_string(counts[i]);
    }
    return text + "]";
}

/// A run of a call: the first count values, then " |", then the guard values.
inline std::string format_run(std::span<const float> run, size_t count) {
    std::string text;
    for (size_t i = 0; i < run.size(); ++i) {
        if (i == count) { text += " |"; }
        text += " ";
        text += format_value(run[i]);
    }
    if (count == run.size()) { text += " |"; }
    return text;
}

/// Sample `position` (0-based, counted over the whole scenario) of a streamed input: every
/// slot and every channel its own ramp, exactly representable, never 0 and never k_untouched.
inline float stream_value(size_t slot, size_t channel, size_t position) {
    return static_cast<float>((slot * 10000) + (channel * 1000) + position + 1);
}

/// Value `index` of a Static input in the call `call` of a scenario.
inline float static_value(size_t slot, size_t call, size_t index) {
    return static_cast<float>(900000 + (slot * 10000) + (call * 10) + index);
}

/// The host memory of one slot for one call: `channels` runs of `count + k_guard` floats, every
/// float k_untouched until the scenario or the library writes it.
class SlotBlock {
public:
    SlotBlock(size_t channels, size_t count) {
        m_runs.assign(channels, std::vector<float>(count + k_guard, k_untouched));
        m_planes.reserve(channels);
        for (std::vector<float>& run : m_runs) { m_planes.push_back(run.data()); }
    }
    SlotBlock(const SlotBlock&) = delete;
    SlotBlock& operator=(const SlotBlock&) = delete;
    SlotBlock(SlotBlock&&) = default;
    SlotBlock& operator=(SlotBlock&&) = default;

    /// One pointer per channel: what a float*** call takes for the slot.
    float** planes() { return m_planes.data(); }
    size_t channels() const { return m_runs.size(); }
    std::vector<float>& run(size_t channel) { return m_runs[channel]; }
    const std::vector<float>& run(size_t channel) const { return m_runs[channel]; }

private:
    std::vector<std::vector<float>> m_runs;
    std::vector<float*> m_planes;
};

/// The channels of one slot of a prepared C handler: a Streamed slot's stream port says (the
/// channel count a host block of it carries), every other slot is one run of values.
inline size_t channels_of(const std::vector<anira::capi::Port>& ports, size_t slot) {
    const anira::capi::StreamPort* stream = anira::capi::stream_port(ports, slot);
    return stream != nullptr ? stream->m_channels : 1;
}

/// Whether a slot of a prepared C handler streams (it has a ring).
inline bool is_streamed(const std::vector<anira::capi::Port>& ports, size_t slot) {
    return anira::capi::stream_port(ports, slot) != nullptr;
}

/// The input memory of one call, one block per input slot of the handler's ports: a streamed
/// slot continues its ramps from `positions` (the samples pushed so far, advanced here), a
/// Static slot carries static_value(slot, call, index). Slot 0 is `slot0_length` floats long
/// when that is more than its count: the memory an in-place call shares with output slot 0.
inline std::vector<SlotBlock> make_inputs(const std::vector<anira::capi::Port>& input_ports,
                                          std::vector<size_t>& positions,
                                          const std::vector<size_t>& in_counts,
                                          size_t slot0_length,
                                          size_t call) {
    std::vector<SlotBlock> inputs;
    inputs.reserve(in_counts.size());
    for (size_t slot = 0; slot < in_counts.size(); ++slot) {
        const size_t length = slot == 0 ? std::max(slot0_length, in_counts[0]) : in_counts[slot];
        inputs.emplace_back(channels_of(input_ports, slot), length);
        const bool streamed = is_streamed(input_ports, slot);
        for (size_t channel = 0; channel < inputs[slot].channels(); ++channel) {
            for (size_t i = 0; i < in_counts[slot]; ++i) {
                inputs[slot].run(channel)[i] =
                    streamed ? stream_value(slot, channel, positions[slot] + i)
                             : static_value(slot, call, i);
            }
        }
        if (streamed) { positions[slot] += in_counts[slot]; }
    }
    return inputs;
}

using Snapshot = std::vector<std::vector<std::vector<float>>>;

inline Snapshot snapshot(const std::vector<SlotBlock>& blocks) {
    Snapshot copy;
    for (const SlotBlock& block : blocks) {
        copy.emplace_back();
        for (size_t channel = 0; channel < block.channels(); ++channel) {
            copy.back().push_back(block.run(channel));
        }
    }
    return copy;
}

/// An input is read, never written: every input run is what it was before the call, the
/// shared slot 0 of an in-place call apart.
inline void expect_inputs_untouched(const std::vector<SlotBlock>& inputs,
                                    const Snapshot& before,
                                    bool in_place,
                                    size_t call) {
    for (size_t slot = 0; slot < inputs.size(); ++slot) {
        if (in_place && slot == 0) { continue; }
        for (size_t channel = 0; channel < inputs[slot].channels(); ++channel) {
            EXPECT_EQ(inputs[slot].run(channel), before[slot][channel])
                << "call " << call << ": input " << slot << ", channel " << channel;
        }
    }
}

/// What a scenario writes and the golden transcript is compared with.
class Transcript {
public:
    void add(std::string line) { m_lines.push_back(std::move(line)); }

    /// The lines of one slot block under a label: "  out0.c1: 1001 1002 | . ."
    void add_block(const std::string& label, const SlotBlock& block, size_t count) {
        for (size_t channel = 0; channel < block.channels(); ++channel) {
            add("  " + label + ".c" + std::to_string(channel) + ":" +
                format_run(block.run(channel), count));
        }
    }

    const std::vector<std::string>& lines() const { return m_lines; }

private:
    std::vector<std::string> m_lines;
};

/// The file of k_record_env, or NULL when nothing is recorded.
inline const char* record_path() {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once per scenario, by the test's one thread
#endif
    return std::getenv(k_record_env);  // NOLINT(concurrency-mt-unsafe)
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

inline std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    while (!text.empty()) {
        const size_t end = text.find('\n');
        if (end == std::string_view::npos) {
            lines.push_back(text);
            break;
        }
        lines.push_back(text.substr(0, end));
        text.remove_prefix(end + 1);
    }
    // The raw literal opens with a line break.
    if (!lines.empty() && lines.front().empty()) { lines.erase(lines.begin()); }
    return lines;
}

/// Compares the transcript with the golden one line by line; a difference reports the call it
/// happened in. Under k_record_env it appends the transcript to that file as the golden literal
/// and compares nothing.
inline void expect_golden(const char* name, std::string_view golden, const Transcript& transcript) {
    const std::vector<std::string>& actual = transcript.lines();
    if (const char* path = record_path(); path != nullptr) {
        size_t bytes = 0;
        for (const std::string& line : actual) { bytes += line.size() + 1; }
        EXPECT_LT(bytes, k_max_transcript_bytes)
            << name << ": too long for one string literal, split the scenario";
        std::ofstream file(path, std::ios::app);
        file << "\ninline constexpr std::string_view " << name << " = R\"oracle(\n";
        for (const std::string& line : actual) { file << line << "\n"; }
        file << ")oracle\";\n";
        EXPECT_TRUE(file.good()) << "cannot write " << path;
        return;
    }
    const std::vector<std::string_view> expected = split_lines(golden);
    EXPECT_EQ(actual.size(), expected.size()) << name << ": the number of transcript lines";
    constexpr size_t k_max_reports = 6;
    size_t reports = 0;
    std::string_view call = "(before the first call)";
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        if (!expected[i].empty() && expected[i].front() == '#') { call = expected[i]; }
        if (actual[i] == expected[i]) { continue; }
        ADD_FAILURE() << name << ", line " << i << ", in the call\n    " << call
                      << "\n  recorded: " << expected[i] << "\n  now:      " << actual[i];
        if (++reports == k_max_reports) {
            ADD_FAILURE() << name << ": further differences are not reported";
            break;
        }
    }
}

/// The clocked gate (see the file comment). collect() runs one non-waiting collection of the
/// session, which is what get_available_samples does on both faces.
template <typename Collect>
void settle(GateEngine& gate, const anira::SessionElement& session, Collect&& collect) {
    gate.m_open.store(true);
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        std::forward<Collect>(collect)();
        if (session.m_time_stamps.empty()) { break; }
        if (std::chrono::steady_clock::now() > start + std::chrono::seconds(k_wait_s)) {
            ADD_FAILURE() << "timeout while collecting " << session.m_time_stamps.size()
                          << " submitted inferences";
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    gate.m_open.store(false);
}

/// A gate engine whose model spreads the one channel of input 0 over every channel of output
/// 0: channel c of the output is the input plus 1000 * c, so a delivered block tells its
/// channels apart although the input is mono (the pass-through alone delivers zeros when the
/// two element counts differ).
class FanOutEngine : public GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const anira_status status = passthrough(ctx);
        if (status != ANIRA_OK) { return status; }
        const float* in = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const size_t samples = elements_of(ctx.inputs[0]);
        const size_t channels = elements_of(ctx.outputs[0]) / samples;
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t i = 0; i < samples; ++i) {
                out[(channel * samples) + i] = in[i] + static_cast<float>(1000 * channel);
            }
        }
        return ANIRA_OK;
    }
};

/// A gate engine whose model is a generator: the pass-through, then sample i of output 0 is
/// element 0 of input 0 plus i.
class ParamRampEngine : public GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const anira_status status = passthrough(ctx);
        if (status != ANIRA_OK) { return status; }
        const float* param = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (param == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        for (size_t i = 0; i < elements_of(ctx.outputs[0]); ++i) {
            out[i] = param[0] + static_cast<float>(i);
        }
        return ANIRA_OK;
    }
};

/// What the custom row's engine computes behind the gate.
enum class Model { PassThrough, FanOut, ParamRamp };

/// The gate engine of a Model.
inline std::unique_ptr<GateEngine> make_gate(Model model) {
    switch (model) {
        case Model::FanOut: return std::make_unique<FanOutEngine>();
        case Model::ParamRamp: return std::make_unique<ParamRampEngine>();
        case Model::PassThrough: break;
    }
    return std::make_unique<GateEngine>();
}

// ---- the models and the contract ---------------------------------------------------------------

/// The window of every streamed tensor, and the block of most scenarios.
inline constexpr uint32_t k_hop = 8;
/// A hop lasts 10 ms, the explicit budget is 1 ms: the latency is small, a transcript short.
inline constexpr double k_rate = 800.0;

/// A streamed [1, channels, k_hop] tensor whose window is the hop.
inline anira::TensorSpec streamed_spec(std::string_view name, int64_t channels) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, channels)
        .axis(2, ANIRA_AXIS_TIME, k_hop)
        .window(k_hop, k_hop, 0);
    return spec;
}

/// A Static [1, count] tensor.
inline anira::TensorSpec static_spec(std::string_view name, int64_t count) {
    anira::TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_ANY, 1).axis(1, ANIRA_AXIS_ANY, count);
    return spec;
}

/// The custom row alone, which the gate engine runs, and two instances.
inline anira::ModelConfig gate_model() {
    anira::ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    model.max_instances(2);
    return model;
}

/// `channels` channels in, as many out: the gate passes the input through.
inline anira::ModelConfig pass_through_model(size_t channels) {
    anira::ModelConfig model = gate_model();
    model.input(streamed_spec("in", static_cast<int64_t>(channels)));
    model.output(streamed_spec("out", static_cast<int64_t>(channels)));
    return model;
}

/// Two channels in, two channels out.
inline anira::ModelConfig stereo_model() {
    return pass_through_model(2);
}

/// A three-channel stream and three Static values on either side.
inline anira::ModelConfig multi_model() {
    anira::ModelConfig model = gate_model();
    model.input(streamed_spec("in", 3));
    model.input(static_spec("values_in", 3));
    model.output(streamed_spec("out", 3));
    model.output(static_spec("values_out", 3));
    return model;
}

/// One channel in, two out (FanOutEngine): a BYPASS miss has one input channel for two output
/// channels.
inline anira::ModelConfig mono_to_stereo_model() {
    anira::ModelConfig model = gate_model();
    model.input(streamed_spec("in", 1));
    model.output(streamed_spec("out", 2));
    return model;
}

/// Four Static values in, one streamed channel out (ParamRampEngine): the anchor is the output,
/// the session is driven by its pops.
inline anira::ModelConfig generator_model() {
    anira::ModelConfig model = gate_model();
    model.input(static_spec("param", 4));
    model.output(streamed_spec("out", 1));
    return model;
}

/// The Hard contract of the recordings: blocks of 1 to `block_max` samples at k_rate (the twin
/// of the 2.x HostConfig(block_max, k_rate, true)), an explicit budget of 1 ms, no warm-up, the
/// miss policy and the wait ratio.
inline anira::ContractHandle recording_contract(anira_miss_policy policy,
                                                uint32_t block_max = k_hop,
                                                double wait_ratio = 0.0) {
    anira::ContractHandle contract =
        explicit_contract(block_max, k_rate, policy, wait_ratio, /*budget_ms=*/1.0);
    contract.hard_geometry(1, block_max, k_rate);
    return contract;
}

// ---- the session -------------------------------------------------------------------------------

/// A C handler prepared for a recording (see "The sessions" in the file comment): the context
/// (two threads), the gate engine of the custom row, the handler over a pipeline of that row
/// alone (and the processors of `processors`), prepared under `contract`, and the session it
/// made. The stems' rigs drive manager() directly. The gate opens and the handler is destroyed
/// (which drains the in-flight work) before the gate dies.
class HandlerSession {
public:
    HandlerSession(const anira::ModelConfig& model,
                   const anira::ContractHandle& contract,
                   Model engine = Model::PassThrough,
                   std::span<const anira_processor_desc> processors = {})
        : m_gate(make_gate(engine))
        , m_handler(m_context, model, m_candidates, processors, m_gate->engine()) {
        const anira_status prepared = m_handler.prepare(contract);
        EXPECT_EQ(prepared, ANIRA_OK) << m_handler.m_err.message;
        if (prepared != ANIRA_OK) { return; }
        m_session = session_of(m_handler.m_handler);
    }
    ~HandlerSession() {
        m_gate->m_open.store(true);
        m_handler.destroy();
    }
    HandlerSession(const HandlerSession&) = delete;
    HandlerSession& operator=(const HandlerSession&) = delete;
    HandlerSession(HandlerSession&&) = delete;
    HandlerSession& operator=(HandlerSession&&) = delete;

    /// Prepared, and its session found.
    bool ready() const { return m_session != nullptr; }
    anira_handler* handler() const { return m_handler.m_handler; }
    /// The copy path: the tensor stems of the handler's own session.
    anira::InferenceManager& manager() const { return *m_handler.m_handler->m_manager; }
    anira::SessionElement& session() const { return *m_session; }
    GateEngine& gate() const { return *m_gate; }
    std::vector<anira::capi::Port>& input_ports() const {
        return m_handler.m_handler->m_input_ports;
    }
    std::vector<anira::capi::Port>& output_ports() const {
        return m_handler.m_handler->m_output_ports;
    }

private:
    Context m_context;
    /// The custom rows only: the models name no engine.
    std::vector<anira_backend_id> m_candidates{{.struct_size = sizeof(anira_backend_id),
                                                .engine = ANIRA_ENGINE_CUSTOM,
                                                .provider = ANIRA_PROVIDER_CPU,
                                                .engine_id = k_custom}};
    std::unique_ptr<GateEngine> m_gate;  ///< before the handler, which dies first
    Handler m_handler;
    std::shared_ptr<anira::SessionElement> m_session;
};

/// "send=[3/3] recv=[16/16,-]": the samples waiting per channel in every send and receive
/// ring, "-" for a Static slot (it has no ring). Read from the rings, nothing is collected.
inline std::string format_rings(const anira::SessionElement& session) {
    const auto side = [](const std::vector<anira::RingBuffer>& rings) {
        std::string text = "[";
        for (size_t slot = 0; slot < rings.size(); ++slot) {
            if (slot > 0) { text += ","; }
            const size_t channels = rings[slot].get_num_channels();
            if (channels == 0) { text += "-"; }
            for (size_t channel = 0; channel < channels; ++channel) {
                if (channel > 0) { text += "/"; }
                text += std::to_string(rings[slot].get_available_samples(channel));
            }
        }
        return text + "]";
    };
    return "send=" + side(session.m_send_buffer) + " recv=" + side(session.m_receive_buffer);
}

/// The first line of every transcript: what prepare left (the latency per output, the capacity
/// of every send and receive ring, the struct pool, the rings).
inline std::string prepared_line(const std::vector<unsigned int>& latencies,
                                 const anira::SessionElement& session) {
    std::string line = "latency=[";
    for (size_t i = 0; i < latencies.size(); ++i) {
        line += (i > 0 ? "," : "") + std::to_string(latencies[i]);
    }
    line += "] send_capacity=[";
    for (size_t i = 0; i < session.m_send_buffer.size(); ++i) {
        line += (i > 0 ? "," : "") + std::to_string(session.m_send_buffer[i].get_num_samples());
    }
    line += "] recv_capacity=[";
    for (size_t i = 0; i < session.m_receive_buffer.size(); ++i) {
        line += (i > 0 ? "," : "") + std::to_string(session.m_receive_buffer[i].get_num_samples());
    }
    return line + "] structs=" + std::to_string(session.m_num_structs) + " " +
           format_rings(session);
}

}  // namespace anira_test::oracle

#endif  // ANIRA_TEST_COPY_ORACLE_H
