// The recording of the host<->ring copy path under the float face of anira/abi/handler.h,
// compared with the transcripts of handler_copy_oracle_golden.h: the twin of
// test/scheduler/test_CopyPathOracle.cpp one level up. The golden transcripts were written by
// the float copy core as it stood before it was rewritten over anira_tensor, under the twelve
// anira_handler_*_f32 entries the C ABI carried until they were removed. Every line of a
// Streamed slot, every ring level and every status of a Streamed call is still that recording,
// byte for byte, up to the first call the block rule made the scripts rewrite (see "The scripts
// keep every call within the declared block" below). The lines of the Static slots were
// recorded again, once, when the C ABI changed what a Static tensor is by decision (see "The
// Static slots" below): an oracle guards refactors, not decisions. Those entries were the tensor
// entries of the same names over planar float32 tensors, so the driver makes every recorded
// call the way their bodies did: it presents the float blocks through an
// anira::PlanarFloatAdapter (anira_test::FloatFace, float_face.h) and calls the tensor entry of
// the same name (anira_handler_process, _multi, push_data, pop_data, their _multi forms and the
// _wait twins). A core that passes reproduces the recorded one bit for bit under the C entries.
//
// What a transcript line holds, and where each word comes from now:
//
//   observed from the library, as before
//     the status the tensor entry returned; rt_error= (anira_handler_rt_error after the
//     call); the ring fill levels (format_rings over the session); every float the call
//     delivered and every guard float it left alone (add_block over the caller's memory);
//     the input memory compared around the call; delivered= of a single form, which is the
//     size_t the tensor entry wrote (or "unset" / "null" when it was handed none); and the
//     counts a multi tensor entry wrote into its `delivered` array.
//
//   reproduced by the driver
//     num_out=[..] of a multi form. The removed _f32 multi forms took one array that was the
//     request going in and the delivered counts coming out, written on ANIRA_OK only and left
//     at the request on ANIRA_MISSED and on every refusal. The tensor multi entries have no
//     such array: `delivered` is a pure out parameter. The driver keeps the request array
//     itself and copies the entry's delivered counts into it on ANIRA_OK and only then
//     (deliver_num_out), which is the rule the removed bodies applied. So the numbers printed
//     on an OK line are still the library's; that the array is left alone on a miss is the
//     driver's doing and pins nothing about the library any more.
//
//   labels
//     the call names a header line prints ("process_f32", "pop_data_f32_multi_wait", ...)
//     are the recorded labels of the scenario's calls, kept verbatim so that the golden file
//     stays byte-identical. They no longer name symbols.
//
// The Static slots. A Static tensor travels whole, in the spec's shape, through
// anira_handler_set_static_input / anira_handler_get_static_output and as the element of a
// _multi form; it has no count, no clamp and no miss policy, and a single form refuses its
// slot. The scripts below are the recorded ones, call for call, and the driver maps what they
// say about a Static slot onto that:
//   - a Static count of a multi form above 0 (3, the 5 over 3 values, the partial 2) carries
//     the whole tensor, and the label and num_out print the element count;
//   - a single form on a Static slot is driven through the Static entries instead, where it
//     stands, so that no refusal latches rt_error for the Streamed lines behind it: process ->
//     "set_static_input+get_static_output" (on the generator, whose output is Streamed,
//     "set_static_input+pop_data_f32"), push_data -> "set_static_input", pop_data ->
//     "get_static_output"; the status printed is the first that is not OK, else the last;
//   - the row of a Static output is the stored value (the latest the model produced), on a
//     delivered block and on a missed one alike.
// Nothing else of a transcript may differ from the recording of the float core: the status
// word, rt_error=, send=, recv=, settled: and every row of a Streamed slot.
//
// test/support/copy_oracle.h says what a transcript is and why a scenario is a pure function
// of its calls.
//
// The scripts keep every call within the declared block. A Hard entry refuses a host block
// larger than one block of the geometry's block_max on its stream (ANIRA_ERROR_INVALID_ARGUMENT,
// nothing moves), so the recorded calls that passed more (13, 20, 30, 40 samples on a block of
// 8, where the ring kept the tail or the request missed) were rewritten within the block, and
// their transcripts recorded again, once, with that change: every transcript of the golden file,
// each up to its first rewritten call byte for byte the earlier recording (the block-size
// scenario declares two hops, so its transcripts differ from the first line). The 2.x copy
// path (anira::InferenceManager through the 2.x float face, the PlanarFloatAdapter and the
// NonStreamableRouter that anira::InferenceHandler owns) takes any count and is untouched, so it
// is the referee of the rewritten transcripts: V2Rig below drives every script call for call
// through that face, and AbiHandlerCopyOracle.MatchesTheV2CopyPath compares what both paths
// observed (see "The 2.x twin").

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../support/copy_oracle.h"
#include "capi/port.h"
#include "float_face.h"
#include "handler_copy_oracle_golden.h"
#include "handler_support.h"
#include "scheduler/NonStreamableRouter.h"
#include "scheduler/PlanarFloatAdapter.h"

namespace {

namespace oracle = anira_test::oracle;
using anira::ModelConfig;
using anira::TensorSpec;

constexpr uint32_t k_hop = 8;      // the window of every streamed tensor, and the contract's block
constexpr double k_rate = 800.0;   // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr size_t k_unset = 77777;  // what a delivered count holds before its call
/// The block of the block-size scenario: two hops, so that a block may be above the hop.
constexpr uint32_t k_two_hops = 2 * k_hop;

// ---- the models --------------------------------------------------------------------------------
// Engine-free: the one model path is the custom row, and its engine is the gate added to the
// pipeline, which runs the pass-through (tensor i of the output is tensor i of the input).

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
constexpr CallOptions k_in_place_gate_closed{.m_in_place = true,
                                             .m_null_unused = false,
                                             .m_null_delivered = false,
                                             .m_settle = false,
                                             .m_wait = false};

/// What the custom row's engine computes behind the gate.
enum class Model { PassThrough, ParamRamp };

/// The gate engine of a Model.
std::unique_ptr<anira_test::GateEngine> gate_of(Model model) {
    if (model == Model::ParamRamp) { return std::make_unique<oracle::ParamRampEngine>(); }
    return std::make_unique<anira_test::GateEngine>();
}

/// "OK", "MISSED" or "status(<n>)". Not anira_status_string: its wording is documentation and
/// may change.
std::string status_text(anira_status status) {
    if (status == ANIRA_OK) { return "OK"; }
    if (status == ANIRA_MISSED) { return "MISSED"; }
    return "status(" + std::to_string(static_cast<int>(status)) + ")";
}

const char* policy_name(anira_miss_policy policy) {
    switch (policy) {
        case ANIRA_MISS_ZEROS: return "ZEROS";
        case ANIRA_MISS_HOLD_LAST: return "HOLD_LAST";
        case ANIRA_MISS_BYPASS: return "BYPASS";
        default: return "another policy";
    }
}

/// The first line of both transcripts: what prepare left.
std::string prepared_line(const std::vector<unsigned int>& latencies,
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
           oracle::format_rings(session);
}

class HandlerRig {
public:
    /// Blocks of 1 to `block_max` samples: the twin of HostConfig(block_max, k_rate, true).
    HandlerRig(const ModelConfig& model,
               anira_miss_policy policy,
               Model engine = Model::PassThrough,
               uint32_t block_max = k_hop)
        : m_gate(gate_of(engine)), m_handler(m_context, model, m_candidates, {}, m_gate->engine()) {
        anira::ContractHandle contract =
            anira_test::explicit_contract(block_max, k_rate, policy, 0.0, 1.0);
        contract.hard_geometry(1, block_max, k_rate);
        const anira_status prepared = m_handler.prepare(contract);
        EXPECT_EQ(prepared, ANIRA_OK) << m_handler.m_err.message;
        if (prepared != ANIRA_OK) { return; }
        anira_handler* handler = m_handler.m_handler;
        m_face = std::make_unique<anira_test::FloatFace>(handler);
        m_session = anira_test::session_of(handler);
        if (m_session == nullptr) { return; }
        m_gate->m_open.store(false);
        m_positions.assign(handler->m_num_inputs, 0);

        std::vector<unsigned int> latencies;
        latencies.reserve(handler->m_num_outputs);
        for (uint32_t i = 0; i < handler->m_num_outputs; ++i) {
            latencies.push_back(anira_handler_get_latency(handler, i));
        }
        m_transcript.add(prepared_line(latencies, *m_session));
        m_neutral.add(prepared_line(latencies, *m_session));
    }
    /// The gate opens and the handler is destroyed (which drains the in-flight work) before
    /// the gate dies: what anira_test::DestroyFirst does for a Handler on the stack.
    ~HandlerRig() {
        m_gate->m_open.store(true);
        m_handler.destroy();
    }
    HandlerRig(const HandlerRig&) = delete;
    HandlerRig& operator=(const HandlerRig&) = delete;

    /// "process_f32": anira_handler_process, one slot, separate memory.
    void process(size_t num_in, size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        if (static_input(slot) != nullptr) {
            set_then_get(slot, num_out, /*in_place=*/false, options);
            return;
        }
        const size_t call = ++m_calls;
        oracle::SlotBlock in = input_block(slot, num_in, call);
        const std::vector<std::vector<float>> in_before = runs_of(in);
        oracle::SlotBlock out(output_channels(slot), num_out);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        open_gate_for(options);
        const anira_status status =
            options.m_wait
                ? m_face->process_wait(in.planes(),
                                       num_in,
                                       out.planes(),
                                       num_out,
                                       slot,
                                       count,
                                       ANIRA_WAIT_FOREVER)
                : m_face->process(in.planes(), num_in, out.planes(), num_out, slot, count);
        m_gate->m_open.store(false);
        add_header(entry("process_f32", options) + " slot=" + std::to_string(slot) +
                       " in=" + std::to_string(num_in) + " out=" + std::to_string(num_out),
                   status,
                   delivered_text(delivered, options),
                   call);
        add_neutral(call, status, streamed_text(delivered, options));
        add_block("out" + std::to_string(slot), out, num_out, /*streamed=*/true);
        EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
        if (options.m_settle) { settle(); }
    }

    /// "process_f32_inplace": anira_handler_process in place, one slot, one memory.
    void process_inplace(size_t num_samples, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        if (static_input(slot) != nullptr) {
            set_then_get(slot, num_samples, /*in_place=*/true, options);
            return;
        }
        const size_t call = ++m_calls;
        oracle::SlotBlock data = input_block(slot, num_samples, call);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        open_gate_for(options);
        const anira_status status =
            options.m_wait ? m_face->process_inplace_wait(data.planes(),
                                                          num_samples,
                                                          slot,
                                                          count,
                                                          ANIRA_WAIT_FOREVER)
                           : m_face->process_inplace(data.planes(), num_samples, slot, count);
        m_gate->m_open.store(false);
        add_header(entry("process_f32_inplace", options) + " slot=" + std::to_string(slot) +
                       " n=" + std::to_string(num_samples),
                   status,
                   delivered_text(delivered, options),
                   call);
        add_neutral(call, status, streamed_text(delivered, options));
        add_block("io" + std::to_string(slot), data, num_samples, /*streamed=*/true);
        if (options.m_settle) { settle(); }
    }

    /// "process_f32_multi": anira_handler_process_multi, every slot. num_out is the request
    /// array of the removed form, kept by the driver (deliver_num_out).
    void process_multi(const std::vector<size_t>& recorded_in,
                       const std::vector<size_t>& recorded_out,
                       CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> in_counts = whole(recorded_in, /*inputs=*/true);
        const std::vector<size_t> out_counts = whole(recorded_out, /*inputs=*/false);
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
        std::vector<size_t> delivered(out_counts.size(), k_unset);
        open_gate_for(options);
        const anira_status status = options.m_wait ? m_face->process_multi_wait(in_planes.data(),
                                                                                in_counts.data(),
                                                                                out_planes.data(),
                                                                                out_counts.data(),
                                                                                delivered.data(),
                                                                                ANIRA_WAIT_FOREVER)
                                                   : m_face->process_multi(in_planes.data(),
                                                                           in_counts.data(),
                                                                           out_planes.data(),
                                                                           out_counts.data(),
                                                                           delivered.data());
        m_gate->m_open.store(false);
        deliver_num_out(status, delivered, num_out);
        std::string what = entry("process_f32_multi", options) +
                           " in=" + oracle::format_counts(in_counts) +
                           " out=" + oracle::format_counts(out_counts);
        if (options.m_in_place) { what += " in-place"; }
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "num_out=" + oracle::format_counts(num_out), call);
        add_neutral(call, status, streamed_text(delivered));
        add_outputs(out_counts, options, inputs, outputs, out_planes);
        oracle::expect_inputs_untouched(inputs, inputs_before, options.m_in_place, call);
        if (options.m_settle) { settle(); }
    }

    /// "push_data_f32": anira_handler_push_data, one slot.
    void push(size_t num_in, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        oracle::SlotBlock in = input_block(slot, num_in, call);
        const std::vector<std::vector<float>> in_before = runs_of(in);
        if (const anira::capi::StaticSlot* store = static_input(slot); store != nullptr) {
            const anira_tensor tensor = anira_test::whole_f32(in.planes()[0], store->shape());
            const anira_status status =
                anira_handler_set_static_input(m_handler.m_handler, slot, &tensor);
            add_header("set_static_input slot=" + std::to_string(slot), status, "", call);
            add_neutral(call, status, "static");
            EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
            if (options.m_settle) { settle(); }
            return;
        }
        const anira_status status = m_face->push_data(in.planes(), num_in, slot);
        add_header("push_data_f32 slot=" + std::to_string(slot) + " in=" + std::to_string(num_in),
                   status,
                   "",
                   call);
        add_neutral(call, status, "push");
        EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
        if (options.m_settle) { settle(); }
    }

    /// "push_data_f32_multi": anira_handler_push_data_multi, every input slot.
    void push_multi(const std::vector<size_t>& recorded_in, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> in_counts = whole(recorded_in, /*inputs=*/true);
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(config(), m_positions, in_counts, 0, call);
        const oracle::Snapshot inputs_before = oracle::snapshot(inputs);
        std::vector<const float* const*> in_planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            in_planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        const anira_status status = m_face->push_data_multi(in_planes.data(), in_counts.data());
        std::string what = "push_data_f32_multi in=" + oracle::format_counts(in_counts);
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "", call);
        add_neutral(call, status, "push");
        oracle::expect_inputs_untouched(inputs, inputs_before, false, call);
        if (options.m_settle) { settle(); }
    }

    /// "pop_data_f32": anira_handler_pop_data, one slot.
    void pop(size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        if (const anira::capi::StaticSlot* store = static_output(slot); store != nullptr) {
            const size_t call = ++m_calls;
            oracle::SlotBlock out(1, store->num_elements());
            const anira_tensor tensor = anira_test::whole_f32(out.planes()[0], store->shape());
            const anira_status status =
                anira_handler_get_static_output(m_handler.m_handler, slot, &tensor);
            add_header("get_static_output slot=" + std::to_string(slot), status, "", call);
            add_neutral(call, status, "static");
            add_block("out" + std::to_string(slot), out, store->num_elements(), false);
            if (options.m_settle) { settle(); }
            return;
        }
        const size_t call = ++m_calls;
        oracle::SlotBlock out(output_channels(slot), num_out);
        size_t delivered = k_unset;
        size_t* const count = options.m_null_delivered ? nullptr : &delivered;
        open_gate_for(options);
        const anira_status status =
            options.m_wait
                ? m_face->pop_data_wait(out.planes(), num_out, slot, count, ANIRA_WAIT_FOREVER)
                : m_face->pop_data(out.planes(), num_out, slot, count);
        m_gate->m_open.store(false);
        add_header(entry("pop_data_f32", options) + " slot=" + std::to_string(slot) +
                       " out=" + std::to_string(num_out),
                   status,
                   delivered_text(delivered, options),
                   call);
        add_neutral(call, status, streamed_text(delivered, options));
        add_block("out" + std::to_string(slot), out, num_out, /*streamed=*/true);
        if (options.m_settle) { settle(); }
    }

    /// "pop_data_f32_multi": anira_handler_pop_data_multi, every output slot.
    void pop_multi(const std::vector<size_t>& recorded_out, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> out_counts = whole(recorded_out, /*inputs=*/false);
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> no_inputs;
        std::vector<oracle::SlotBlock> outputs;
        std::vector<float* const*> out_planes;
        make_outputs(out_counts, options, no_inputs, outputs, out_planes);
        std::vector<size_t> num_out = out_counts;
        std::vector<size_t> delivered(out_counts.size(), k_unset);
        open_gate_for(options);
        const anira_status status =
            options.m_wait
                ? m_face->pop_data_multi_wait(out_planes.data(),
                                              out_counts.data(),
                                              delivered.data(),
                                              ANIRA_WAIT_FOREVER)
                : m_face->pop_data_multi(out_planes.data(), out_counts.data(), delivered.data());
        m_gate->m_open.store(false);
        deliver_num_out(status, delivered, num_out);
        std::string what =
            entry("pop_data_f32_multi", options) + " out=" + oracle::format_counts(out_counts);
        if (options.m_null_unused) { what += " null-unused"; }
        add_header(what, status, "num_out=" + oracle::format_counts(num_out), call);
        add_neutral(call, status, streamed_text(delivered));
        add_outputs(out_counts, options, no_inputs, outputs, out_planes);
        if (options.m_settle) { settle(); }
    }

    void reset() {
        if (m_session == nullptr) { return; }
        anira_handler_reset(m_handler.m_handler);
        const std::string line =
            "#" + std::to_string(++m_calls) + " reset " + oracle::format_rings(*m_session);
        m_transcript.add(line);
        m_neutral.add(line);
    }

    /// Opens the gate until every submitted inference is collected.
    void settle() {
        if (m_session == nullptr) { return; }
        oracle::settle(*m_gate, *m_session, [this] { anira_test::available(m_handler.m_handler); });
        m_transcript.add("  settled: " + oracle::format_rings(*m_session));
        m_neutral.add("  settled: " + oracle::format_rings(*m_session));
    }

    const oracle::Transcript& transcript() const { return m_transcript; }
    /// What both copy paths observe (see "The 2.x twin").
    const oracle::Transcript& neutral() const { return m_neutral; }

private:
    /// The store of a Static slot, NULL for a Streamed one.
    const anira::capi::StaticSlot* static_input(uint32_t slot) const {
        return anira::capi::static_slot(m_handler.m_handler->m_input_ports, slot);
    }
    const anira::capi::StaticSlot* static_output(uint32_t slot) const {
        return anira::capi::static_slot(m_handler.m_handler->m_output_ports, slot);
    }

    /// The recorded counts of a multi form with every Static count above 0 replaced by the
    /// slot's element count: a carried Static tensor is the whole tensor.
    std::vector<size_t> whole(const std::vector<size_t>& recorded, bool inputs) const {
        std::vector<size_t> counts = recorded;
        for (uint32_t slot = 0; slot < counts.size(); ++slot) {
            const anira::capi::StaticSlot* store =
                inputs ? static_input(slot) : static_output(slot);
            if (store != nullptr && counts[slot] > 0) { counts[slot] = store->num_elements(); }
        }
        return counts;
    }

    /// A recorded single process form whose input slot is Static, driven through the Static
    /// entries where it stands: the set of the whole input, then the get of the whole output
    /// (in place: into the same memory), or, where the output of that slot is Streamed (the
    /// generator), the pop of the recorded request.
    void set_then_get(uint32_t slot, size_t num_out, bool in_place, const CallOptions& options) {
        const size_t call = ++m_calls;
        anira_handler* handler = m_handler.m_handler;
        const anira::capi::StaticSlot* in_store = static_input(slot);
        oracle::SlotBlock in = input_block(slot, in_store->num_elements(), call);
        const std::vector<std::vector<float>> in_before = runs_of(in);
        const anira_tensor in_tensor = anira_test::whole_f32(in.planes()[0], in_store->shape());
        anira_status status = anira_handler_set_static_input(handler, slot, &in_tensor);
        const anira::capi::StaticSlot* out_store = static_output(slot);
        if (out_store != nullptr) {
            oracle::SlotBlock separate(1, out_store->num_elements());
            oracle::SlotBlock& out = in_place ? in : separate;
            const anira_tensor out_tensor =
                anira_test::whole_f32(out.planes()[0], out_store->shape());
            const anira_status got = anira_handler_get_static_output(handler, slot, &out_tensor);
            if (status == ANIRA_OK) { status = got; }
            add_header("set_static_input+get_static_output slot=" + std::to_string(slot),
                       status,
                       "",
                       call);
            add_neutral(call, status, "static");
            add_block((in_place ? "io" : "out") + std::to_string(slot),
                      out,
                      out_store->num_elements(),
                      /*streamed=*/false);
            if (!in_place) {
                EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
            }
        } else {
            oracle::SlotBlock out(output_channels(slot), num_out);
            size_t delivered = k_unset;
            size_t* const count = options.m_null_delivered ? nullptr : &delivered;
            open_gate_for(options);
            const anira_status popped =
                options.m_wait
                    ? m_face->pop_data_wait(out.planes(), num_out, slot, count, ANIRA_WAIT_FOREVER)
                    : m_face->pop_data(out.planes(), num_out, slot, count);
            m_gate->m_open.store(false);
            if (status == ANIRA_OK) { status = popped; }
            add_header("set_static_input+" + entry("pop_data_f32", options) +
                           " slot=" + std::to_string(slot) + " out=" + std::to_string(num_out),
                       status,
                       delivered_text(delivered, options),
                       call);
            add_neutral(call, status, streamed_text(delivered, options));
            add_block("out" + std::to_string(slot), out, num_out, /*streamed=*/true);
            EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
        }
        if (options.m_settle) { settle(); }
    }

    /// A _wait twin waits without limit, so its gate is open while it runs: it returns with
    /// every submitted inference collected, its own included, and the gate closes again
    /// behind it with nothing in flight.
    void open_gate_for(const CallOptions& options) {
        if (options.m_wait) { m_gate->m_open.store(true); }
    }

    /// The count protocol of the removed _f32 multi forms, reproduced here (see the file
    /// comment): num_out goes in as the request and takes the counts the tensor entry
    /// delivered on ANIRA_OK only; ANIRA_MISSED and every refusal leave it at the request.
    static void deliver_num_out(anira_status status,
                                const std::vector<size_t>& delivered,
                                std::vector<size_t>& num_out) {
        if (status != ANIRA_OK) { return; }
        num_out = delivered;
    }

    static std::string entry(const std::string& name, const CallOptions& options) {
        return options.m_wait ? name + "_wait" : name;
    }

    const anira::InferenceConfig& config() const { return m_handler.m_handler->m_inference_config; }

    size_t output_channels(uint32_t slot) const {
        return config().get_postprocess_output_channels()[slot];
    }

    /// The memory of one input slot, its ramp continued (the single forms and the Static
    /// entries, whose count is the slot's element count).
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
            const bool streamed = config().get_postprocess_output_size()[slot] > 0;
            if (options.m_in_place && slot == 0 && !inputs.empty()) {
                add_block("io0", inputs[0], out_counts[0], streamed);
            } else {
                add_block("out" + std::to_string(slot), outputs[slot], out_counts[slot], streamed);
            }
        }
    }

    static std::string delivered_text(size_t delivered, const CallOptions& options) {
        if (options.m_null_delivered) { return "delivered=null"; }
        return "delivered=" +
               (delivered == k_unset ? std::string("unset") : std::to_string(delivered));
    }

    /// The neutral word of a single form's count: what the 2.x twin prints for it.
    static std::string streamed_text(size_t delivered, const CallOptions& options) {
        if (options.m_null_delivered) { return "streamed=?"; }
        return "streamed=[" + std::to_string(delivered) + "]";
    }

    /// The neutral word of a multi form's counts: those of its Streamed output slots.
    std::string streamed_text(const std::vector<size_t>& delivered) const {
        std::vector<size_t> streamed;
        for (size_t slot = 0; slot < delivered.size(); ++slot) {
            if (config().get_postprocess_output_size()[slot] > 0) {
                streamed.push_back(delivered[slot]);
            }
        }
        return "streamed=" + oracle::format_counts(streamed);
    }

    /// A call's header in the neutral transcript: the status and the rings it left.
    void add_neutral(size_t call, anira_status status, const std::string& counts) {
        m_neutral.add("#" + std::to_string(call) + " " + status_text(status) + " " + counts + " " +
                      oracle::format_rings(*m_session));
    }

    /// The rows of an output the call was handed; the neutral transcript takes a Streamed
    /// slot's alone (a Static value is what the C ABI made of it, see the file comment).
    void add_block(const std::string& label,
                   const oracle::SlotBlock& block,
                   size_t count,
                   bool streamed) {
        m_transcript.add_block(label, block, count);
        if (streamed) { m_neutral.add_block(label, block, count); }
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
                                                .engine = ANIRA_ENGINE_CUSTOM,
                                                .provider = ANIRA_PROVIDER_CPU,
                                                .engine_id = anira_test::k_custom}};
    std::unique_ptr<anira_test::GateEngine> m_gate;  ///< before the handler, which dies first
    anira_test::Handler m_handler;
    std::unique_ptr<anira_test::FloatFace> m_face;  ///< built after prepare; borrows m_handler
    std::shared_ptr<anira::SessionElement> m_session;

    std::vector<size_t> m_positions;  ///< per input slot, the samples pushed so far
    size_t m_calls = 0;
    oracle::Transcript m_transcript;
    oracle::Transcript m_neutral;
};

// ---- the 2.x twin ------------------------------------------------------------------------------
// The referee of the transcripts: every call of a script driven through the 2.x copy path, the
// anira::InferenceManager of a 2.x configuration (the twin of the C model, the twins of
// test/scheduler/test_CopyPathOracle.cpp) through the float face anira::InferenceHandler owns
// (the PlanarFloatAdapter presents the channel pointers, the NonStreamableRouter stores and
// reads back the Static slots around every stem call), which takes any count. V2Rig has the
// call surface of HandlerRig and makes each call as the C entry it mirrors does: the same stem
// (process_nowait, process_wait, push_data, pop_data, pop_data_wait) over the same counts and
// the same memory, the gate open for a _wait twin; a single form carries its slot and leaves
// the others out; a Static slot carried alone (the C Static entries) goes through the router
// with every streamed count 0. Its neutral transcript holds what both paths observe: the first
// line of the golden transcripts, per call its status (MISSED is the manager's miss flag), the
// delivered counts of its Streamed outputs and the rings it left, the rows of every Streamed
// output it was handed, and every "settled:" and reset line. A Static value is left out: the
// C ABI changed what a Static tensor is by decision (the file comment), and a pass-through
// model never lets it reach a Streamed slot (the generator reads element 0, which both carry).

/// The 2.x configuration of a C model of this file, its every figure the C model's.
anira::InferenceConfig v2_config(anira::TensorShape shape, anira::ProcessingSpec spec) {
    return {std::vector<anira::ModelData>{
                anira::ModelData("placeholder", anira::InferenceBackend::CUSTOM)},
            std::vector<anira::TensorShape>{std::move(shape)},
            std::move(spec),
            1.F,    // max_inference_time, in ms: the contract's explicit budget
            0,      // warm_up
            false,  // session_exclusive_processor
            0.F,    // blocking_ratio: the contract's wait ratio
            2};     // num_parallel_processors: max_instances
}

anira::InferenceConfig v2_stereo_config() {
    return v2_config(anira::TensorShape({{1, 2, k_hop}}, {{1, 2, k_hop}}),
                     anira::ProcessingSpec({2}, {2}));
}

anira::InferenceConfig v2_multi_config() {
    return v2_config(anira::TensorShape({{1, 3, k_hop}, {1, 3}}, {{1, 3, k_hop}, {1, 3}}),
                     anira::ProcessingSpec({3, 1}, {3, 1}, {k_hop, 0}, {k_hop, 0}));
}

anira::InferenceConfig v2_generator_config() {
    return v2_config(anira::TensorShape({{1, 4}}, {{1, 1, k_hop}}),
                     anira::ProcessingSpec({1}, {1}, {0}, {k_hop}));
}

std::unique_ptr<anira_test::GateBackend> v2_gate_of(Model model, anira::InferenceConfig& config) {
    if (model == Model::ParamRamp) { return std::make_unique<oracle::ParamRampGate>(config); }
    return std::make_unique<anira_test::GateBackend>(config);
}

class V2Rig {
public:
    V2Rig(anira::InferenceConfig config,
          anira_miss_policy policy,
          Model model = Model::PassThrough,
          uint32_t block_max = k_hop)
        : m_config(std::move(config))
        , m_pp(m_config)
        , m_gate(v2_gate_of(model, m_config))
        , m_manager(m_pp, m_config, m_gate.get(), anira::CoreConfig(2))
        , m_positions(m_config.get_tensor_input_shape().size(), 0) {
        m_manager.set_miss_policy(policy);
        m_manager.prepare(
            anira::HostConfig(static_cast<float>(block_max), static_cast<float>(k_rate), true));
        m_adapter.prepare(m_config);
        m_router.prepare();
        for (const std::shared_ptr<anira::SessionElement>& session : anira::Core::get_sessions()) {
            if (session->m_session_id == m_manager.get_session_id()) { m_session = session; }
        }
        if (m_session == nullptr) {
            ADD_FAILURE() << "the manager's session is not registered";
            return;
        }
        m_gate->m_open.store(false);
        m_neutral.add(prepared_line(m_manager.get_latency(), *m_session));
    }
    ~V2Rig() { m_gate->m_open.store(true); }  // before the manager releases the session
    V2Rig(const V2Rig&) = delete;
    V2Rig& operator=(const V2Rig&) = delete;
    V2Rig(V2Rig&&) = delete;
    V2Rig& operator=(V2Rig&&) = delete;

    void process(size_t num_in, size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        call_single(num_in, num_out, slot, /*in_place=*/false, options);
    }

    void process_inplace(size_t num_samples, uint32_t slot = 0, CallOptions options = k_plain) {
        call_single(num_samples, num_samples, slot, /*in_place=*/true, options);
    }

    void process_multi(const std::vector<size_t>& recorded_in,
                       const std::vector<size_t>& recorded_out,
                       CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> in_counts = whole(recorded_in, /*inputs=*/true);
        const std::vector<size_t> out_counts = whole(recorded_out, /*inputs=*/false);
        const size_t call = ++m_calls;
        const size_t shared_length = options.m_in_place ? out_counts[0] : 0;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, shared_length, call);
        std::vector<const float* const*> in_planes = input_planes(inputs, in_counts, options);
        std::vector<oracle::SlotBlock> outputs;
        std::vector<float* const*> out_planes;
        make_outputs(out_counts, options, inputs, outputs, out_planes);
        const anira_tensor* in_tensors =
            m_adapter.present_inputs(in_planes.data(), in_counts.data());
        const anira_tensor* out_tensors =
            m_adapter.present_outputs(out_planes.data(), out_counts.data());
        open_gate_for(options);
        anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
        const size_t* delivered =
            options.m_wait ? m_router.process_wait(in_tensors,
                                                   out_tensors,
                                                   std::chrono::steady_clock::duration::max(),
                                                   outcome)
                           : m_router.process_nowait(in_tensors, out_tensors);
        m_gate->m_open.store(false);
        add_neutral(call, streamed_text(delivered));
        add_outputs(out_counts, options, inputs, outputs, out_planes);
        if (options.m_settle) { settle(); }
    }

    void push(size_t num_in, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        const bool is_static = !streamed_input(slot);
        std::vector<size_t> in_counts(m_config.get_tensor_input_shape().size(), 0);
        in_counts[slot] = is_static ? m_config.get_tensor_input_size()[slot] : num_in;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, 0, call);
        const std::vector<const float* const*> in_planes =
            input_planes(inputs, in_counts, k_null_unused);
        m_router.push_data(m_adapter.present_inputs(in_planes.data(), in_counts.data()));
        m_neutral.add("#" + std::to_string(call) + " OK " + (is_static ? "static" : "push") + " " +
                      oracle::format_rings(*m_session));
        if (options.m_settle) { settle(); }
    }

    void push_multi(const std::vector<size_t>& recorded_in, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> in_counts = whole(recorded_in, /*inputs=*/true);
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, 0, call);
        const std::vector<const float* const*> in_planes = input_planes(inputs, in_counts, options);
        m_router.push_data(m_adapter.present_inputs(in_planes.data(), in_counts.data()));
        m_neutral.add("#" + std::to_string(call) + " OK push " + oracle::format_rings(*m_session));
        if (options.m_settle) { settle(); }
    }

    void pop(size_t num_out, uint32_t slot = 0, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        if (!streamed_output(slot)) {
            pop_static(slot);
            m_neutral.add("#" + std::to_string(call) + " OK static " +
                          oracle::format_rings(*m_session));
            if (options.m_settle) { settle(); }
            return;
        }
        oracle::SlotBlock out(m_config.get_postprocess_output_channels()[slot], num_out);
        const size_t count = pop_streamed(slot, out, num_out, options);
        add_neutral(call, single_text(count, options));
        m_neutral.add_block("out" + std::to_string(slot), out, num_out);
        if (options.m_settle) { settle(); }
    }

    void pop_multi(const std::vector<size_t>& recorded_out, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const std::vector<size_t> out_counts = whole(recorded_out, /*inputs=*/false);
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> no_inputs;
        std::vector<oracle::SlotBlock> outputs;
        std::vector<float* const*> out_planes;
        make_outputs(out_counts, options, no_inputs, outputs, out_planes);
        const anira_tensor* out_tensors =
            m_adapter.present_outputs(out_planes.data(), out_counts.data());
        open_gate_for(options);
        anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
        const size_t* delivered =
            options.m_wait ? m_router.pop_data_wait(out_tensors,
                                                    std::chrono::steady_clock::duration::max(),
                                                    outcome)
                           : m_router.pop_data(out_tensors);
        m_gate->m_open.store(false);
        add_neutral(call, streamed_text(delivered));
        add_outputs(out_counts, options, no_inputs, outputs, out_planes);
        if (options.m_settle) { settle(); }
    }

    void reset() {
        if (m_session == nullptr) { return; }
        m_manager.reset();
        m_neutral.add("#" + std::to_string(++m_calls) + " reset " +
                      oracle::format_rings(*m_session));
    }

    void settle() {
        if (m_session == nullptr) { return; }
        oracle::settle(*m_gate, *m_session, [this] { m_manager.get_available_samples(0, 0); });
        m_neutral.add("  settled: " + oracle::format_rings(*m_session));
    }

    const oracle::Transcript& neutral() const { return m_neutral; }

private:
    bool streamed_input(size_t slot) const {
        return m_config.get_preprocess_input_size()[slot] > 0;
    }
    bool streamed_output(size_t slot) const {
        return m_config.get_postprocess_output_size()[slot] > 0;
    }

    /// HandlerRig::whole(): a carried Static slot is its whole tensor.
    std::vector<size_t> whole(const std::vector<size_t>& recorded, bool inputs) const {
        std::vector<size_t> counts = recorded;
        for (size_t slot = 0; slot < counts.size(); ++slot) {
            const bool streamed = inputs ? streamed_input(slot) : streamed_output(slot);
            if (streamed || counts[slot] == 0) { continue; }
            counts[slot] = inputs ? m_config.get_tensor_input_size()[slot]
                                  : m_config.get_tensor_output_size()[slot];
        }
        return counts;
    }

    /// A single form (HandlerRig::process / process_inplace): the slot's block in, the slot's
    /// block out, every other slot left out. A Static input slot is HandlerRig::set_then_get:
    /// the whole value stored, then the whole value read back, or, where the slot's output is
    /// Streamed (the generator), the pop of the request.
    void call_single(size_t num_in,
                     size_t num_out,
                     uint32_t slot,
                     bool in_place,
                     const CallOptions& options) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        const size_t num_inputs = m_config.get_tensor_input_shape().size();
        const size_t num_outputs = m_config.get_tensor_output_shape().size();
        if (!streamed_input(slot)) {
            std::vector<size_t> in_counts(num_inputs, 0);
            in_counts[slot] = m_config.get_tensor_input_size()[slot];
            std::vector<oracle::SlotBlock> inputs =
                oracle::make_inputs(m_config, m_positions, in_counts, 0, call);
            const std::vector<const float* const*> in_planes =
                input_planes(inputs, in_counts, k_null_unused);
            m_router.push_data(m_adapter.present_inputs(in_planes.data(), in_counts.data()));
            if (!streamed_output(slot)) {
                pop_static(slot);
                m_neutral.add("#" + std::to_string(call) + " OK static " +
                              oracle::format_rings(*m_session));
            } else {
                oracle::SlotBlock out(m_config.get_postprocess_output_channels()[slot], num_out);
                const size_t count = pop_streamed(slot, out, num_out, options);
                add_neutral(call, single_text(count, options));
                m_neutral.add_block("out" + std::to_string(slot), out, num_out);
            }
            if (options.m_settle) { settle(); }
            return;
        }
        std::vector<size_t> in_counts(num_inputs, 0);
        in_counts[slot] = num_in;
        std::vector<size_t> out_counts(num_outputs, 0);
        out_counts[slot] = num_out;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, 0, call);
        const std::vector<const float* const*> in_planes =
            input_planes(inputs, in_counts, k_null_unused);
        oracle::SlotBlock separate(m_config.get_postprocess_output_channels()[slot], num_out);
        oracle::SlotBlock& out = in_place ? inputs[slot] : separate;
        std::vector<float* const*> out_planes(num_outputs, nullptr);
        out_planes[slot] = out.planes();
        const anira_tensor* in_tensors =
            m_adapter.present_inputs(in_planes.data(), in_counts.data());
        const anira_tensor* out_tensors =
            m_adapter.present_outputs(out_planes.data(), out_counts.data());
        open_gate_for(options);
        anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
        const size_t* delivered =
            options.m_wait ? m_router.process_wait(in_tensors,
                                                   out_tensors,
                                                   std::chrono::steady_clock::duration::max(),
                                                   outcome)
                           : m_router.process_nowait(in_tensors, out_tensors);
        m_gate->m_open.store(false);
        add_neutral(call, single_text(delivered[slot], options));
        m_neutral.add_block((in_place ? "io" : "out") + std::to_string(slot), out, num_out);
        if (options.m_settle) { settle(); }
    }

    /// The pop of one Streamed output slot, the others left out; the delivered count.
    size_t pop_streamed(uint32_t slot,
                        oracle::SlotBlock& out,
                        size_t num_out,
                        const CallOptions& options) {
        const size_t num_outputs = m_config.get_tensor_output_shape().size();
        std::vector<size_t> out_counts(num_outputs, 0);
        out_counts[slot] = num_out;
        std::vector<float* const*> out_planes(num_outputs, nullptr);
        out_planes[slot] = out.planes();
        const anira_tensor* out_tensors =
            m_adapter.present_outputs(out_planes.data(), out_counts.data());
        open_gate_for(options);
        anira::Core::WaitOutcome outcome = anira::Core::WaitOutcome::Done;
        const size_t* delivered =
            options.m_wait ? m_router.pop_data_wait(out_tensors,
                                                    std::chrono::steady_clock::duration::max(),
                                                    outcome)
                           : m_router.pop_data(out_tensors);
        m_gate->m_open.store(false);
        return delivered[slot];
    }

    /// The read of one Static output slot alone (anira_handler_get_static_output).
    void pop_static(uint32_t slot) {
        const size_t num_outputs = m_config.get_tensor_output_shape().size();
        std::vector<size_t> out_counts(num_outputs, 0);
        out_counts[slot] = m_config.get_tensor_output_size()[slot];
        oracle::SlotBlock out(1, out_counts[slot]);
        std::vector<float* const*> out_planes(num_outputs, nullptr);
        out_planes[slot] = out.planes();
        m_router.pop_data(m_adapter.present_outputs(out_planes.data(), out_counts.data()));
    }

    void open_gate_for(const CallOptions& options) {
        if (options.m_wait) { m_gate->m_open.store(true); }
    }

    static std::vector<const float* const*> input_planes(std::vector<oracle::SlotBlock>& inputs,
                                                         const std::vector<size_t>& in_counts,
                                                         const CallOptions& options) {
        std::vector<const float* const*> planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        return planes;
    }

    void make_outputs(const std::vector<size_t>& out_counts,
                      const CallOptions& options,
                      std::vector<oracle::SlotBlock>& inputs,
                      std::vector<oracle::SlotBlock>& outputs,
                      std::vector<float* const*>& out_planes) const {
        outputs.reserve(out_counts.size());
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            outputs.emplace_back(m_config.get_postprocess_output_channels()[slot],
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
            if (out_planes[slot] == nullptr || !streamed_output(slot)) { continue; }
            if (options.m_in_place && slot == 0 && !inputs.empty()) {
                m_neutral.add_block("io0", inputs[0], out_counts[0]);
            } else {
                m_neutral.add_block("out" + std::to_string(slot), outputs[slot], out_counts[slot]);
            }
        }
    }

    static std::string single_text(size_t delivered, const CallOptions& options) {
        if (options.m_null_delivered) { return "streamed=?"; }
        return "streamed=[" + std::to_string(delivered) + "]";
    }

    std::string streamed_text(const size_t* delivered) const {
        std::vector<size_t> streamed;
        for (size_t slot = 0; slot < m_config.get_tensor_output_shape().size(); ++slot) {
            if (streamed_output(slot)) { streamed.push_back(delivered[slot]); }
        }
        return "streamed=" + oracle::format_counts(streamed);
    }

    void add_neutral(size_t call, const std::string& counts) {
        const anira_status status = m_manager.last_block_missed() ? ANIRA_MISSED : ANIRA_OK;
        m_neutral.add("#" + std::to_string(call) + " " + status_text(status) + " " + counts + " " +
                      oracle::format_rings(*m_session));
    }

    anira::InferenceConfig m_config;
    anira::PrePostProcessor m_pp;
    std::unique_ptr<anira_test::GateBackend> m_gate;  ///< before the manager, which dies first
    anira::InferenceManager m_manager;
    anira::PlanarFloatAdapter m_adapter;
    anira::NonStreamableRouter m_router{m_manager, m_pp, m_config};
    std::shared_ptr<anira::SessionElement> m_session;
    std::vector<size_t> m_positions;  ///< per input slot, the samples pushed so far
    size_t m_calls = 0;
    oracle::Transcript m_neutral;
};

/// The two neutral transcripts line by line; a difference names the call it happened in.
void expect_same_observations(const char* name,
                              const oracle::Transcript& handler,
                              const oracle::Transcript& v2) {
    const std::vector<std::string>& c_lines = handler.lines();
    const std::vector<std::string>& v2_lines = v2.lines();
    EXPECT_EQ(c_lines.size(), v2_lines.size()) << name << ": the number of lines";
    constexpr size_t k_max_reports = 6;
    size_t reports = 0;
    std::string_view call = "(before the first call)";
    for (size_t i = 0; i < c_lines.size() && i < v2_lines.size(); ++i) {
        if (!c_lines[i].empty() && c_lines[i].front() == '#') { call = c_lines[i]; }
        if (c_lines[i] == v2_lines[i]) { continue; }
        ADD_FAILURE() << name << ", line " << i << ", in the call\n    " << call
                      << "\n  C entries: " << c_lines[i] << "\n  2.x path:  " << v2_lines[i];
        if (++reports == k_max_reports) {
            ADD_FAILURE() << name << ": further differences are not reported";
            break;
        }
    }
}

// ---- the scenarios -----------------------------------------------------------------------------
// The scripts of test/scheduler/test_CopyPathOracle.cpp, through the C entries, with every call
// kept within the declared block (the file comment). The figures (latency, ring capacities,
// structs) are printed by the golden transcripts. Each script runs on a HandlerRig against its
// golden transcript and on both rigs against each other (MatchesTheV2CopyPath).

/// Blocks below, at and above the hop (the block is two hops, k_two_hops), a block of 0
/// samples, separate memory and in place, a NULL delivered pointer, two calls whose counts
/// differ (a request above the pushed count the ring serves, then one it cannot serve, a
/// request below the pushed count), the catch-up, a reset in the middle of the stream, and the
/// whole declared block in one call. (Recorded on the block of one hop with a push larger than
/// the send ring, 20 samples, and a request of 40; a call carries one declared block at most.)
template <typename Rig>
void block_sizes_script(Rig& rig) {
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
    rig.process(8, 16);
    rig.process(8, 16);
    rig.process_inplace(8, 0, k_null_delivered);
    rig.process(8, 3);
    rig.process(8, 8);
    rig.reset();
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.process(16, 13);
    rig.process(8, 8);
    rig.process_inplace(8);
}

/// The starved block: the gate stays closed from call 3 on, so the ring runs dry while four
/// inferences wait. Two delivered blocks, at the hold capacity (the block) and below it, then
/// misses at a request equal to the hold capacity, above the pushed count, in place, below the
/// pushed count and on a pop; the gate opens, the catch-up discards the late blocks; a reset,
/// after which the ring holds its priming again. (Recorded with a delivered block of 13 above
/// the hold capacity and a pop of 20 after the reset; a call carries one declared block at
/// most, so no block exceeds the hold capacity.)
template <typename Rig>
void starved_blocks_script(Rig& rig) {
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.process(8, 8, 0, k_gate_closed);
    rig.process(5, 5, 0, k_gate_closed);
    rig.process(8, 8, 0, k_gate_closed);
    rig.process(3, 8, 0, k_gate_closed);
    rig.process_inplace(8, 0, k_gate_closed);
    rig.process(5, 3, 0, k_gate_closed);
    rig.pop(8, 0, k_gate_closed);
    rig.settle();
    rig.process(8, 8);
    rig.process_inplace(8);
    rig.reset();
    rig.pop(8);
    rig.process(8, 8);
    rig.process_inplace(8);
}

/// The split calls: pushes and pops of unequal sizes, a pop of 0 samples, a pop without a
/// delivered pointer, a pop while the pushed block waits at the gate, and the pops that drain
/// the ring until one it cannot serve. (Recorded with pops of 11, 30 and 13; a call carries one
/// declared block at most, so the ring is drained block by block.)
template <typename Rig>
void push_pop_script(Rig& rig) {
    rig.push(8);
    rig.push(3);
    rig.push(5);
    rig.pop(5);
    rig.pop(7);
    rig.pop(0);
    rig.push(8);
    rig.pop(8, 0, k_null_delivered);
    rig.push(8, 0, k_gate_closed);
    rig.pop(8);
    rig.pop(8);
    rig.pop(8);
    rig.pop(8);
}

/// Two slots on either side, one of them Static: the multi forms with both slots, with one
/// slot and the other handed over as NULL, a Static request above and below the value count,
/// in place; the single forms on the Static slot and on the streamed one; the split multi
/// forms; a starved request on both outputs at once (num_out is left at the requests), in
/// place and on the split forms, the catch-up and a reset. (Recorded with requests of 40 the
/// ring could not serve; the ring is starved at the closed gate instead.)
template <typename Rig>
void multi_slot_script(Rig& rig) {
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
    rig.process_multi({8, 3}, {8, 5}, k_gate_closed);
    rig.process_multi({8, 3}, {8, 5}, k_gate_closed);
    rig.process_multi({8, 3}, {8, 2}, k_in_place_gate_closed);
    rig.pop_multi({8, 3}, k_gate_closed);
    rig.pop(8, 0, k_gate_closed);
    rig.settle();
    rig.process_multi({8, 3}, {8, 3});
    rig.reset();
    rig.process_multi({8, 3}, {8, 3});
    rig.process_multi({8, 3}, {8, 3});
}

/// The five _wait twins with ANIRA_WAIT_FOREVER, which waits for every submitted inference
/// whatever the machine does (an explicit timeout would make the block a matter of the wall
/// clock): the delivered block carries the call's own inference; once the pops have drained
/// the ring, a request no inference covers is a miss even then, on all four forms that pop (a
/// multi twin leaves num_out at the requests), and the next whole block recovers. (Recorded
/// with requests of 40; a call carries one declared block at most.)
template <typename Rig>
void waiting_twins_script(Rig& rig) {
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
    rig.pop(8, 0, k_wait);
    rig.pop(8, 0, k_wait);
    rig.process(0, 8, 0, k_wait);
    rig.process_multi({0, 3}, {8, 5}, k_wait);
    rig.pop(8, 0, k_wait);
    rig.pop_multi({8, 3}, k_wait);
    rig.process_multi({8, 3}, {8, 3}, k_wait);
}

/// A generator: the Static input is pushed, the pops drive the inferences, a process form does
/// both, a reset. (BYPASS is refused on a generator at prepare; ZEROS is the policy here.)
/// (Recorded with a process of 13 and a pop of 20; a call carries one declared block at most.)
template <typename Rig>
void generator_script(Rig& rig) {
    rig.push(4);
    rig.pop(8);
    rig.pop(8);
    rig.pop(3);
    rig.process(4, 8);
    rig.process(2, 8);
    rig.pop(5);
    rig.pop(8);
    rig.reset();
    rig.pop(8);
    rig.pop(8);
}

void run_block_sizes(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy, Model::PassThrough, k_two_hops);
    block_sizes_script(rig);
    oracle::expect_golden(name, golden, rig.transcript());
}

void run_starved_blocks(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy);
    starved_blocks_script(rig);
    oracle::expect_golden(name, golden, rig.transcript());
}

void run_push_pop(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(stereo_model(), policy);
    push_pop_script(rig);
    oracle::expect_golden(name, golden, rig.transcript());
}

void run_multi_slot(anira_miss_policy policy, const char* name, std::string_view golden) {
    HandlerRig rig(multi_model(), policy);
    multi_slot_script(rig);
    oracle::expect_golden(name, golden, rig.transcript());
}

/// One script on both copy paths, one after the other (the C handler is destroyed before the
/// 2.x manager is built), and their neutral transcripts compared.
template <typename Script>
void expect_twins(const char* name,
                  const ModelConfig& model,
                  anira::InferenceConfig v2_config,
                  anira_miss_policy policy,
                  Model engine,
                  uint32_t block_max,
                  Script script) {
    oracle::Transcript handler;
    {
        HandlerRig rig(model, policy, engine, block_max);
        script(rig);
        handler = rig.neutral();
    }
    oracle::Transcript v2;
    {
        V2Rig rig(std::move(v2_config), policy, engine, block_max);
        script(rig);
        v2 = rig.neutral();
    }
    expect_same_observations(name, handler, v2);
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

// The _wait twins (waiting_twins_script).
TEST(AbiHandlerCopyOracle, WaitingTwins) {
    HandlerRig rig(multi_model(), ANIRA_MISS_HOLD_LAST);
    waiting_twins_script(rig);
    oracle::expect_golden("k_handler_waiting_twins",
                          oracle::k_handler_waiting_twins,
                          rig.transcript());
}

// A generator pulled by its pops (generator_script).
TEST(AbiHandlerCopyOracle, GeneratorPulledByItsPops) {
    HandlerRig rig(generator_model(), ANIRA_MISS_ZEROS, Model::ParamRamp);
    generator_script(rig);
    oracle::expect_golden("k_handler_generator", oracle::k_handler_generator, rig.transcript());
}

// Every script on the C entries and on the 2.x copy path, which takes any count and is
// untouched by the C ABI: what the two observe is the same, call for call (V2Rig). It is the
// evidence that the golden transcripts are the copy path's and not a recording of whatever the
// C entries do.
TEST(AbiHandlerCopyOracle, MatchesTheV2CopyPath) {
    const auto block_sizes = [](auto& rig) { block_sizes_script(rig); };
    const auto starved = [](auto& rig) { starved_blocks_script(rig); };
    const auto push_pop = [](auto& rig) { push_pop_script(rig); };
    const auto multi_slot = [](auto& rig) { multi_slot_script(rig); };
    for (const anira_miss_policy policy :
         {ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST, ANIRA_MISS_BYPASS}) {
        SCOPED_TRACE(policy_name(policy));
        expect_twins("block sizes",
                     stereo_model(),
                     v2_stereo_config(),
                     policy,
                     Model::PassThrough,
                     k_two_hops,
                     block_sizes);
        expect_twins("starved blocks",
                     stereo_model(),
                     v2_stereo_config(),
                     policy,
                     Model::PassThrough,
                     k_hop,
                     starved);
        expect_twins("multi slot",
                     multi_model(),
                     v2_multi_config(),
                     policy,
                     Model::PassThrough,
                     k_hop,
                     multi_slot);
        expect_twins("push pop",
                     stereo_model(),
                     v2_stereo_config(),
                     policy,
                     Model::PassThrough,
                     k_hop,
                     push_pop);
    }
    expect_twins("waiting twins",
                 multi_model(),
                 v2_multi_config(),
                 ANIRA_MISS_HOLD_LAST,
                 Model::PassThrough,
                 k_hop,
                 [](auto& rig) { waiting_twins_script(rig); });
    expect_twins("generator",
                 generator_model(),
                 v2_generator_config(),
                 ANIRA_MISS_ZEROS,
                 Model::ParamRamp,
                 k_hop,
                 [](auto& rig) { generator_script(rig); });
}
