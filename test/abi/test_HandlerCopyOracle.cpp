// The recording of the host<->ring copy path under the _f32 Hard entries of anira/abi/handler.h:
// the twin of test/scheduler/test_CopyPathOracle.cpp over anira_handler_process_f32, _inplace
// and _multi, anira_handler_push_data_f32 and _multi, anira_handler_pop_data_f32 and _multi,
// compared with the transcripts of handler_copy_oracle_golden.h. The golden transcripts were
// written by the float copy core as it stood before it was rewritten over anira_tensor, so a
// core that passes reproduces that one bit for bit under the C entries: the status, the
// delivered count of a single form, the num_out array of a multi form (written on ANIRA_OK
// only), anira_handler_rt_error, every float a call wrote and every float it left alone.
// test/support/copy_oracle.h says what a transcript is and why a scenario is a pure function
// of its calls.

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../support/copy_oracle.h"
#include "handler_copy_oracle_golden.h"
#include "handler_support.h"

namespace {

namespace oracle = anira_test::oracle;
using anira::ModelConfig;
using anira::TensorSpec;

constexpr uint32_t k_hop = 8;      // the window of every streamed tensor, and the contract's block
constexpr double k_rate = 800.0;   // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr size_t k_unset = 77777;  // what a delivered count holds before its call

// ---- the models --------------------------------------------------------------------------------
// Engine-free: the one model path is the custom row, and the session's custom backend is the
// gate, which runs BackendBase::process (tensor i of the output is tensor i of the input).

TensorSpec streamed(std::string_view name, int64_t channels) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, channels)
        .axis(2, ANIRA_AXIS_TIME, k_hop)
        .window(k_hop, k_hop, 0);
    return spec;
}

TensorSpec static_values(std::string_view name, int64_t count) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_ANY, 1).axis(1, ANIRA_AXIS_ANY, count);
    return spec;
}

/// Two channels in, two channels out.
ModelConfig stereo_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", 2));
    model.output(streamed("out", 2));
    model.max_instances(2);
    return model;
}

/// A three-channel stream and three Static values on either side.
ModelConfig multi_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", 3));
    model.input(static_values("values_in", 3));
    model.output(streamed("out", 3));
    model.output(static_values("values_out", 3));
    model.max_instances(2);
    return model;
}

/// Four Static values in, one streamed channel out: the anchor is the output.
ModelConfig generator_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(static_values("param", 4));
    model.output(streamed("out", 1));
    model.max_instances(2);
    return model;
}

// ---- the rig -----------------------------------------------------------------------------------

struct CallOptions {
    bool m_in_place = false;     ///< output slot 0 is the memory of input slot 0 (multi form)
    bool m_null_unused = false;  ///< a slot whose count is 0 is handed over as NULL (multi forms)
    bool m_null_delivered = false;  ///< the single forms' delivered pointer is NULL
    bool m_settle = true;           ///< false leaves the call's inferences at the gate
    bool m_wait = false;            ///< the _wait twin with ANIRA_WAIT_FOREVER, the gate open
};

constexpr CallOptions k_plain{.m_in_place = false,
                              .m_null_unused = false,
                              .m_null_delivered = false,
                              .m_settle = true,
                              .m_wait = false};
constexpr CallOptions k_in_place{.m_in_place = true,
                                 .m_null_unused = false,
                                 .m_null_delivered = false,
                                 .m_settle = true,
                                 .m_wait = false};
constexpr CallOptions k_null_unused{.m_in_place = false,
                                    .m_null_unused = true,
                                    .m_null_delivered = false,
                                    .m_settle = true,
                                    .m_wait = false};
constexpr CallOptions k_null_delivered{.m_in_place = false,
                                       .m_null_unused = false,
                                       .m_null_delivered = true,
                                       .m_settle = true,
                                       .m_wait = false};
constexpr CallOptions k_gate_closed{.m_in_place = false,
                                    .m_null_unused = false,
                                    .m_null_delivered = false,
                                    .m_settle = false,
                                    .m_wait = false};
constexpr CallOptions k_wait{.m_in_place = false,
                             .m_null_unused = false,
                             .m_null_delivered = false,
                             .m_settle = true,
                             .m_wait = true};
constexpr CallOptions k_wait_in_place{.m_in_place = true,
                                      .m_null_unused = false,
                                      .m_null_delivered = false,
                                      .m_settle = true,
                                      .m_wait = true};

/// What the session's custom backend computes behind the gate.
enum class Model { PassThrough, ParamRamp };

class HandlerRig {
public:
    HandlerRig(const ModelConfig& model,
               anira_miss_policy policy,
               Model backend = Model::PassThrough)
        : m_handler(m_context, model, m_candidates) {
        // Blocks of 1 to k_hop samples: the twin of HostConfig(k_hop, k_rate, true).
        anira::ContractHandle contract =
            anira_test::explicit_contract(k_hop, k_rate, policy, 0.0, 1.0);
        contract.hard_geometry(1, k_hop, k_rate);
        const anira_status prepared = m_handler.prepare(contract);
        EXPECT_EQ(prepared, ANIRA_OK) << m_handler.m_err.message;
        if (prepared != ANIRA_OK) { return; }
        anira_handler* handler = m_handler.m_handler;
        if (backend == Model::ParamRamp) {
            m_gate = std::make_unique<oracle::ParamRampGate>(handler->m_inference_config);
        } else {
            m_gate = std::make_unique<anira_test::GateBackend>(handler->m_inference_config);
        }
        m_session = anira_test::session_of(handler);
        if (m_session == nullptr) { return; }
        m_session->m_custom_processor = m_gate.get();  // attach_processor, the gate closed
        m_gate->m_open.store(false);
        m_positions.assign(handler->m_num_inputs, 0);

        std::string line = "latency=[";
        for (uint32_t i = 0; i < handler->m_num_outputs; ++i) {
            line += (i > 0 ? "," : "") + std::to_string(anira_handler_get_latency(handler, i));
        }
        line += "] send_capacity=[";
        for (size_t i = 0; i < m_session->m_send_buffer.size(); ++i) {
            line +=
                (i > 0 ? "," : "") + std::to_string(m_session->m_send_buffer[i].get_num_samples());
        }
        line += "] recv_capacity=[";
        for (size_t i = 0; i < m_session->m_receive_buffer.size(); ++i) {
            line += (i > 0 ? "," : "") +
                    std::to_string(m_session->m_receive_buffer[i].get_num_samples());
        }
        m_transcript.add(line + "] structs=" + std::to_string(m_session->m_num_structs) + " " +
                         oracle::format_rings(*m_session));
    }
    /// The gate opens and the handler is destroyed (which drains the in-flight work) before
    /// the gate dies: what anira_test::DestroyFirst does for a stack backend.
    ~HandlerRig() {
        if (m_gate != nullptr) { m_gate->m_open.store(true); }
        m_handler.destroy();
    }
    HandlerRig(const HandlerRig&) = delete;
    HandlerRig& operator=(const HandlerRig&) = delete;

    /// anira_handler_process_f32: one slot, separate memory.
    void process(size_t num_in, size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        oracle::SlotBlock in = input_block(slot, num_in, call);
        const std::vector<std::vector<float>> in_before = runs_of(in);
        oracle::SlotBlock out(output_channels(slot), num_out);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        anira_handler* const handler = m_handler.m_handler;
        open_gate_for(options);
        const anira_status status = options.m_wait
                                        ? anira_handler_process_f32_wait(handler,
                                                                         in.planes(),
                                                                         num_in,
                                                                         out.planes(),
                                                                         num_out,
                                                                         ANIRA_WAIT_FOREVER,
                                                                         slot,
                                                                         count)
                                        : anira_handler_process_f32(handler,
                                                                    in.planes(),
                                                                    num_in,
                                                                    out.planes(),
                                                                    num_out,
                                                                    slot,
                                                                    count);
        m_gate->m_open.store(false);
        add_header(entry("process_f32", options) + " slot=" + std::to_string(slot) +
                       " in=" + std::to_string(num_in) + " out=" + std::to_string(num_out),
                   status,
                   delivered_text(delivered, options),
                   call);
        m_transcript.add_block("out" + std::to_string(slot), out, num_out);
        EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_process_f32_inplace: one slot, one memory.
    void process_inplace(size_t num_samples, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        oracle::SlotBlock data = input_block(slot, num_samples, call);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        anira_handler* const handler = m_handler.m_handler;
        open_gate_for(options);
        const anira_status status = options.m_wait
                                        ? anira_handler_process_f32_inplace_wait(handler,
                                                                                 data.planes(),
                                                                                 num_samples,
                                                                                 ANIRA_WAIT_FOREVER,
                                                                                 slot,
                                                                                 count)
                                        : anira_handler_process_f32_inplace(handler,
                                                                            data.planes(),
                                                                            num_samples,
                                                                            slot,
                                                                            count);
        m_gate->m_open.store(false);
        add_header(entry("process_f32_inplace", options) + " slot=" + std::to_string(slot) +
                       " n=" + std::to_string(num_samples),
                   status,
                   delivered_text(delivered, options),
                   call);
        m_transcript.add_block("io" + std::to_string(slot), data, num_samples);
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_process_f32_multi: every slot; num_out is the caller's request array.
    void process_multi(const std::vector<size_t>& in_counts,
                       const std::vector<size_t>& out_counts,
                       CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        const size_t shared_length = options.m_in_place ? out_counts[0] : 0;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(config(), m_positions, in_counts, shared_length, call);
        const oracle::Snapshot inputs_before = oracle::snapshot(inputs);
        std::vector<const float* const*> in_planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            in_planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        std::vector<oracle::SlotBlock> outputs;
        std::vector<float* const*> out_planes;
        make_outputs(out_counts, options, inputs, outputs, out_planes);
        std::vector<size_t> num_out = out_counts;
        anira_handler* const handler = m_handler.m_handler;
        open_gate_for(options);
        const anira_status status = options.m_wait
                                        ? anira_handler_process_f32_multi_wait(handler,
                                                                               in_planes.data(),
                                                                               in_counts.data(),
                                                                               out_planes.data(),
                                                                               num_out.data(),
                                                                               ANIRA_WAIT_FOREVER)
                                        : anira_handler_process_f32_multi(handler,
                                                                          in_planes.data(),
                                                                          in_counts.data(),
                                                                          out_planes.data(),
                                                                          num_out.data());
        m_gate->m_open.store(false);
        std::string what = entry("process_f32_multi", options) +
                           " in=" + oracle::format_counts(in_counts) +
                           " out=" + oracle::format_counts(out_counts);
        if (options.m_in_place) { what += " in-place"; }
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "num_out=" + oracle::format_counts(num_out), call);
        add_outputs(out_counts, options, inputs, outputs, out_planes);
        oracle::expect_inputs_untouched(inputs, inputs_before, options.m_in_place, call);
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_push_data_f32: one slot.
    void push(size_t num_in, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        oracle::SlotBlock in = input_block(slot, num_in, call);
        const std::vector<std::vector<float>> in_before = runs_of(in);
        const anira_status status =
            anira_handler_push_data_f32(m_handler.m_handler, in.planes(), num_in, slot);
        add_header("push_data_f32 slot=" + std::to_string(slot) + " in=" + std::to_string(num_in),
                   status,
                   "",
                   call);
        EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_push_data_f32_multi: every input slot.
    void push_multi(const std::vector<size_t>& in_counts, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(config(), m_positions, in_counts, 0, call);
        const oracle::Snapshot inputs_before = oracle::snapshot(inputs);
        std::vector<const float* const*> in_planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            in_planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        const anira_status status = anira_handler_push_data_f32_multi(m_handler.m_handler,
                                                                      in_planes.data(),
                                                                      in_counts.data());
        std::string what = "push_data_f32_multi in=" + oracle::format_counts(in_counts);
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "", call);
        oracle::expect_inputs_untouched(inputs, inputs_before, false, call);
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_pop_data_f32: one slot.
    void pop(size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        oracle::SlotBlock out(output_channels(slot), num_out);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        anira_handler* const handler = m_handler.m_handler;
        open_gate_for(options);
        const anira_status status =
            options.m_wait
                ? anira_handler_pop_data_f32_wait(handler,
                                                  out.planes(),
                                                  num_out,
                                                  ANIRA_WAIT_FOREVER,
                                                  slot,
                                                  count)
                : anira_handler_pop_data_f32(handler, out.planes(), num_out, slot, count);
        m_gate->m_open.store(false);
        add_header(entry("pop_data_f32", options) + " slot=" + std::to_string(slot) +
                       " out=" + std::to_string(num_out),
                   status,
                   delivered_text(delivered, options),
                   call);
        m_transcript.add_block("out" + std::to_string(slot), out, num_out);
        if (options.m_settle) { settle(); }
    }

    /// anira_handler_pop_data_f32_multi: every output slot.
    void pop_multi(const std::vector<size_t>& out_counts, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> no_inputs;
        std::vector<oracle::SlotBlock> outputs;
        std::vector<float* const*> out_planes;
        make_outputs(out_counts, options, no_inputs, outputs, out_planes);
        std::vector<size_t> num_out = out_counts;
        anira_handler* const handler = m_handler.m_handler;
        open_gate_for(options);
        const anira_status status =
            options.m_wait
                ? anira_handler_pop_data_f32_multi_wait(handler,
                                                        out_planes.data(),
                                                        num_out.data(),
                                                        ANIRA_WAIT_FOREVER)
                : anira_handler_pop_data_f32_multi(handler, out_planes.data(), num_out.data());
        m_gate->m_open.store(false);
        std::string what =
            entry("pop_data_f32_multi", options) + " out=" + oracle::format_counts(out_counts);
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "num_out=" + oracle::format_counts(num_out), call);
        add_outputs(out_counts, options, no_inputs, outputs, out_planes);
        if (options.m_settle) { settle(); }
    }

    void reset() {
        if (m_session == nullptr) { return; }
        anira_handler_reset(m_handler.m_handler);
        m_transcript.add("#" + std::to_string(++m_calls) + " reset " +
                         oracle::format_rings(*m_session));
    }

    /// Opens the gate until every submitted inference is collected.
    void settle() {
        if (m_session == nullptr) { return; }
        oracle::settle(*m_gate, *m_session, [this] { anira_test::available(m_handler.m_handler); });
        m_transcript.add("  settled: " + oracle::format_rings(*m_session));
    }

    const oracle::Transcript& transcript() const { return m_transcript; }

private:
    /// A _wait twin waits without limit, so its gate is open while it runs: it returns with
    /// every submitted inference collected, its own included, and the gate closes again
    /// behind it with nothing in flight.
    void open_gate_for(const CallOptions& options) {
        if (options.m_wait) { m_gate->m_open.store(true); }
    }

    static std::string entry(const std::string& name, const CallOptions& options) {
        return options.m_wait ? name + "_wait" : name;
    }

    const anira::InferenceConfig& config() const { return m_handler.m_handler->m_inference_config; }

    size_t output_channels(uint32_t slot) const {
        return config().get_postprocess_output_channels()[slot];
    }

    /// The memory of one input slot, its ramp continued (the single forms).
    oracle::SlotBlock input_block(uint32_t slot, size_t count, size_t call) {
        oracle::SlotBlock block(config().get_preprocess_input_channels()[slot], count);
        const bool is_streamed = config().get_preprocess_input_size()[slot] > 0;
        for (size_t channel = 0; channel < block.channels(); ++channel) {
            for (size_t i = 0; i < count; ++i) {
                block.run(channel)[i] =
                    is_streamed ? oracle::stream_value(slot, channel, m_positions[slot] + i)
                                : oracle::static_value(slot, call, i);
            }
        }
        if (is_streamed) { m_positions[slot] += count; }
        return block;
    }

    static std::vector<std::vector<float>> runs_of(const oracle::SlotBlock& block) {
        std::vector<std::vector<float>> runs;
        runs.reserve(block.channels());
        for (size_t channel = 0; channel < block.channels(); ++channel) {
            runs.push_back(block.run(channel));
        }
        return runs;
    }

    void make_outputs(const std::vector<size_t>& out_counts,
                      const CallOptions& options,
                      std::vector<oracle::SlotBlock>& inputs,
                      std::vector<oracle::SlotBlock>& outputs,
                      std::vector<float* const*>& out_planes) const {
        outputs.reserve(out_counts.size());
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            outputs.emplace_back(config().get_postprocess_output_channels()[slot],
                                 out_counts[slot]);
            const bool unused = options.m_null_unused && out_counts[slot] == 0;
            const bool shared = options.m_in_place && slot == 0 && !inputs.empty();
            float* const* planes = shared ? inputs[0].planes() : outputs[slot].planes();
            out_planes.push_back(unused ? nullptr : planes);
        }
    }

    void add_outputs(const std::vector<size_t>& out_counts,
                     const CallOptions& options,
                     const std::vector<oracle::SlotBlock>& inputs,
                     const std::vector<oracle::SlotBlock>& outputs,
                     const std::vector<float* const*>& out_planes) {
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            if (out_planes[slot] == nullptr) { continue; }
            if (options.m_in_place && slot == 0 && !inputs.empty()) {
                m_transcript.add_block("io0", inputs[0], out_counts[0]);
            } else {
                m_transcript.add_block("out" + std::to_string(slot),
                                       outputs[slot],
                                       out_counts[slot]);
            }
        }
    }

    static std::string delivered_text(size_t delivered, const CallOptions& options) {
        if (options.m_null_delivered) { return "delivered=null"; }
        return "delivered=" +
               (delivered == k_unset ? std::string("unset") : std::to_string(delivered));
    }

    /// Not anira_status_string: its wording is documentation and may change.
    static std::string status_text(anira_status status) {
        if (status == ANIRA_OK) { return "OK"; }
        if (status == ANIRA_MISSED) { return "MISSED"; }
        return "status(" + std::to_string(static_cast<int>(status)) + ")";
    }

    void add_header(const std::string& what,
                    anira_status status,
                    const std::string& counts,
                    size_t call) {
        std::string header = "#" + std::to_string(call) + " " + what + " -> " + status_text(status);
        if (!counts.empty()) { header += " " + counts; }
        header += " rt_error=" + status_text(anira_handler_rt_error(m_handler.m_handler));
        m_transcript.add(header + " " + oracle::format_rings(*m_session));
    }

    anira_test::Context m_context;
    /// The custom rows only: the models name no engine.
    std::vector<anira_backend_id> m_candidates{{.struct_size = sizeof(anira_backend_id),
                                                .engine = ANIRA_ENGINE_NONE,
                                                .provider = ANIRA_PROVIDER_DEFAULT,
                                                .engine_id = nullptr}};
    anira_test::Handler m_handler;
    std::unique_ptr<anira_test::GateBackend> m_gate;
    std::shared_ptr<anira::SessionElement> m_session;
    std::vector<size_t> m_positions;  ///< per input slot, the samples pushed so far
    size_t m_calls = 0;
    oracle::Transcript m_transcript;
};

// ---- the scenarios -----------------------------------------------------------------------------
// The scripts of test/scheduler/test_CopyPathOracle.cpp, through the C entries. The figures
// (latency, ring capacities, structs) are printed by the golden transcripts.

/// Blocks below, at and above the hop, a block of 0 samples, separate memory and in place, a
/// NULL delivered pointer, two calls whose counts differ (a request the ring cannot serve, then
/// a request below the pushed count), the catch-up, a reset in the middle of the stream, and a
/// push larger than the send ring.
void run_block_sizes(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy);
    rig.process(8, 8);
    rig.process(8, 8);
    rig.process(3, 3);
    rig.process(3, 3);
    rig.process(3, 3);
    rig.process(0, 0);
    rig.process(13, 13);
    rig.process_inplace(8);
    rig.process_inplace(3);
    rig.process_inplace(13);
    rig.process_inplace(0);
    rig.process(8, 8, 0, k_null_delivered);
    rig.process(8, 40);
    rig.process_inplace(8, 0, k_null_delivered);
    rig.process(8, 3);
    rig.process(8, 8);
    rig.reset();
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.process(20, 13);
    rig.process(8, 8);
    rig.process_inplace(8);
    oracle::expect_golden(name, golden, rig.transcript());
}

/// The starved block: the gate stays closed from call 3 on, so the ring runs dry while four
/// inferences wait. A delivered block above the hold capacity, then misses at a request equal
/// to the hold capacity, above it, in place, below the pushed count and on a pop; the gate
/// opens, the catch-up discards the late blocks; a reset, after which HOLD_LAST holds nothing.
void run_starved_blocks(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy);
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.process(13, 13, 0, k_gate_closed);
    rig.process(8, 8, 0, k_gate_closed);
    rig.process(3, 13, 0, k_gate_closed);
    rig.process_inplace(8, 0, k_gate_closed);
    rig.process(5, 3, 0, k_gate_closed);
    rig.pop(8, 0, k_gate_closed);
    rig.settle();
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.reset();
    rig.pop(20);
    rig.process(8, 8);
    rig.process_inplace(8);
    oracle::expect_golden(name, golden, rig.transcript());
}

/// The split calls: pushes and pops of unequal sizes, a pop of 0 samples, a pop the ring cannot
/// serve, a pop without a delivered pointer, a pop while the pushed block waits at the gate.
void run_push_pop(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy);
    rig.push(8);
    rig.push(3);
    rig.push(5);
    rig.pop(5);
    rig.pop(11);
    rig.pop(0);
    rig.pop(30);
    rig.push(8);
    rig.pop(8, 0, k_null_delivered);
    rig.push(8, 0, k_gate_closed);
    rig.pop(8);
    rig.pop(13);
    oracle::expect_golden(name, golden, rig.transcript());
}

/// Two slots on either side, one of them Static: the multi forms with both slots, with one
/// slot and the other handed over as NULL, a Static request above and below the value count,
/// in place; the single forms on the Static slot and on the streamed one; the split multi
/// forms; a starved request on both outputs at once (num_out is left at the requests), the
/// catch-up and a reset.
void run_multi_slot(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(multi_model(), policy);
    rig.process_multi({8, 3}, {8, 3});
    rig.process_multi({8, 3}, {8, 3});
    rig.process_multi({8, 0}, {8, 0}, k_null_unused);
    rig.process_multi({0, 3}, {0, 3}, k_null_unused);
    rig.process_multi({8, 3}, {8, 5});
    rig.process_multi({8, 2}, {8, 2}, k_in_place);
    rig.process(3, 3, 1);
    rig.process(3, 5, 1);
    rig.process_inplace(2, 1);
    rig.process(8, 8, 0);
    rig.push_multi({8, 3});
    rig.push_multi({0, 3}, k_null_unused);
    rig.push(3, 1);
    rig.pop_multi({8, 3});
    rig.pop_multi({0, 3}, k_null_unused);
    rig.pop_multi({8, 0}, k_null_unused);
    rig.pop(5, 1);
    rig.process_multi({8, 3}, {40, 5});
    rig.process_multi({8, 3}, {40, 2}, k_in_place);
    rig.pop_multi({40, 3});
    rig.pop(40, 0);
    rig.process_multi({8, 3}, {8, 3});
    rig.reset();
    rig.process_multi({8, 3}, {8, 3});
    rig.process_multi({8, 3}, {8, 3});
    oracle::expect_golden(name, golden, rig.transcript());
}

}  // namespace

TEST(AbiHandlerCopyOracle, StereoBlockSizes) {
    run_block_sizes(ANIRA_MISS_ZEROS,
                    "k_handler_block_sizes_zeros",
                    oracle::k_handler_block_sizes_zeros);
    run_block_sizes(ANIRA_MISS_HOLD_LAST,
                    "k_handler_block_sizes_hold_last",
                    oracle::k_handler_block_sizes_hold_last);
    run_block_sizes(ANIRA_MISS_BYPASS,
                    "k_handler_block_sizes_bypass",
                    oracle::k_handler_block_sizes_bypass);
}

TEST(AbiHandlerCopyOracle, StereoStarvedBlocks) {
    run_starved_blocks(ANIRA_MISS_ZEROS,
                       "k_handler_starved_zeros",
                       oracle::k_handler_starved_zeros);
    run_starved_blocks(ANIRA_MISS_HOLD_LAST,
                       "k_handler_starved_hold_last",
                       oracle::k_handler_starved_hold_last);
    run_starved_blocks(ANIRA_MISS_BYPASS,
                       "k_handler_starved_bypass",
                       oracle::k_handler_starved_bypass);
}

TEST(AbiHandlerCopyOracle, StereoPushPop) {
    run_push_pop(ANIRA_MISS_ZEROS, "k_handler_push_pop_zeros", oracle::k_handler_push_pop_zeros);
    run_push_pop(ANIRA_MISS_HOLD_LAST,
                 "k_handler_push_pop_hold_last",
                 oracle::k_handler_push_pop_hold_last);
}

TEST(AbiHandlerCopyOracle, MultiSlotWithStaticTensors) {
    run_multi_slot(ANIRA_MISS_ZEROS, "k_handler_multi_zeros", oracle::k_handler_multi_zeros);
    run_multi_slot(ANIRA_MISS_HOLD_LAST,
                   "k_handler_multi_hold_last",
                   oracle::k_handler_multi_hold_last);
    run_multi_slot(ANIRA_MISS_BYPASS, "k_handler_multi_bypass", oracle::k_handler_multi_bypass);
}

// The five _wait twins with ANIRA_WAIT_FOREVER, which waits for every submitted inference
// whatever the machine does (an explicit timeout would make the block a matter of the wall
// clock): the delivered block carries the call's own inference, a request the ring cannot
// serve even then is a miss, and a multi twin leaves num_out at the requests on it.
TEST(AbiHandlerCopyOracle, WaitingTwins) {
    HandlerRig rig(multi_model(), ANIRA_MISS_HOLD_LAST);
    rig.process(8, 8, 0, k_wait);
    rig.process(8, 8, 0, k_wait);
    rig.process_inplace(8, 0, k_wait);
    rig.process_inplace(3, 0, k_wait);
    rig.process_multi({8, 3}, {8, 3}, k_wait);
    rig.process_multi({5, 3}, {5, 5}, k_wait_in_place);
    rig.push(8);
    rig.pop(8, 0, k_wait);
    rig.push_multi({8, 3});
    rig.pop_multi({8, 3}, k_wait);
    rig.process(8, 40, 0, k_wait);
    rig.process_multi({8, 3}, {40, 5}, k_wait);
    rig.pop(40, 0, k_wait);
    rig.pop_multi({40, 3}, k_wait);
    rig.process_multi({8, 3}, {8, 3}, k_wait);
    oracle::expect_golden("k_handler_waiting_twins",
                          oracle::k_handler_waiting_twins,
                          rig.transcript());
}

// A generator: the Static input is pushed, the pops drive the inferences, a process form does
// both. (BYPASS is refused on a generator at prepare; ZEROS is the policy here.)
TEST(AbiHandlerCopyOracle, GeneratorPulledByItsPops) {
    HandlerRig rig(generator_model(), ANIRA_MISS_ZEROS, Model::ParamRamp);
    rig.push(4);
    rig.pop(8);
    rig.pop(8);
    rig.pop(3);
    rig.process(4, 13);
    rig.process(2, 8);
    rig.pop(20);
    rig.pop(8);
    rig.reset();
    rig.pop(8);
    rig.pop(8);
    oracle::expect_golden("k_handler_generator", oracle::k_handler_generator, rig.transcript());
}
