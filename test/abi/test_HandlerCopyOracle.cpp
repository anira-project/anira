// The recording of the host<->ring copy path under the float face of anira/abi/handler.h,
// compared with the transcripts of handler_copy_oracle_golden.h: the twin of
// test/scheduler/test_CopyPathOracle.cpp one level up. The golden transcripts were written by
// the float copy core as it stood before it was rewritten over anira_tensor, under the twelve
// anira_handler_*_f32 entries the C ABI carried until they were removed. Every line of a
// Streamed slot, every ring level and every status of a Streamed call is still that recording,
// byte for byte. The lines of the Static slots were recorded again, once, when the C ABI changed
// what a Static tensor is by decision (see "The Static slots" below): an oracle guards
// refactors, not decisions. Those entries were the tensor entries of the same names over planar
// float32 tensors, so the driver makes every recorded call the way their bodies did: it presents
// the float blocks through an anira::PlanarFloatAdapter (anira_test::FloatFace, float_face.h) and
// calls the tensor entry of the same name (anira_handler_process, _multi, push_data, pop_data,
// their _multi forms and the _wait twins). A core that passes reproduces the recorded one bit
// for bit under the C entries.
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

#include <anira/InferenceConfig.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
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
#include "capi/port.h"
#include "float_face.h"
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
        m_face = std::make_unique<anira_test::FloatFace>(handler);
        if (backend == Model::ParamRamp) {
            m_gate = std::make_unique<oracle::ParamRampGate>(handler->m_inference_config);
        } else {
            m_gate = std::make_unique<anira_test::GateBackend>(handler->m_inference_config);
        }
        m_session = anira_test::session_of(handler);
        if (m_session == nullptr) { return; }
        anira_test::attach_processor(handler, *m_gate);  // the gate closed
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
        m_transcript.add_block("out" + std::to_string(slot), out, num_out);
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
        m_transcript.add_block("io" + std::to_string(slot), data, num_samples);
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
            add_header("set_static_input slot=" + std::to_string(slot),
                       anira_handler_set_static_input(m_handler.m_handler, slot, &tensor),
                       "",
                       call);
            EXPECT_EQ(runs_of(in), in_before) << "call " << call << ": the input was written";
            if (options.m_settle) { settle(); }
            return;
        }
        const anira_status status = m_face->push_data(in.planes(), num_in, slot);
        add_header("push_data_f32 slot=" + std::to_string(slot) + " in=" + std::to_string(num_in),
                   status,
                   "",
                   call);
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
            add_header("get_static_output slot=" + std::to_string(slot),
                       anira_handler_get_static_output(m_handler.m_handler, slot, &tensor),
                       "",
                       call);
            m_transcript.add_block("out" + std::to_string(slot), out, store->num_elements());
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
        m_transcript.add_block("out" + std::to_string(slot), out, num_out);
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
            m_transcript.add_block((in_place ? "io" : "out") + std::to_string(slot),
                                   out,
                                   out_store->num_elements());
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
            m_transcript.add_block("out" + std::to_string(slot), out, num_out);
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
    std::unique_ptr<anira_test::FloatFace> m_face;  ///< built after prepare; borrows m_handler
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
