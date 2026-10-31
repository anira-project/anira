// The session-sizing cases of the scheduler suites (test_SessionElement, test_InferenceManager):
// a model described by the facts a session is sized from, per slot its element count, its
// channels and the samples it moves per inference, the host blocks, the budget, the wait
// ratio and the instances, built into a C model config and contract and prepared on a C
// handler over the custom row (its pass-through engine; no case runs an inference). What the
// suites read back is the prepared session: the latencies, the struct pool and the ring sizes.
//
// A Streamed slot of e elements over c channels moving h samples is [Channel c, Time e / c]
// with a window of h and no overlap (anira_test::stream_tensor); a Static slot is one axis of
// e elements. Those are the facts the 2.x configurations of these cases spelled as a tensor
// shape and a processing spec, so every expected figure is the one they were written for.
#ifndef ANIRA_TEST_SUPPORT_SESSION_CASES_H
#define ANIRA_TEST_SUPPORT_SESSION_CASES_H

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"

namespace anira_test {

/// One tensor of a case: `m_elements` elements over `m_channels` channels, moving `m_hop`
/// samples per inference; a hop of -1 moves every sample of a channel (elements / channels),
/// a hop of 0 makes the slot Static.
struct SizingSlot {
    int64_t m_elements = 0;
    int64_t m_channels = 1;
    int64_t m_hop = -1;
};

/// Which tensor anchors the host blocks; none: the first Streamed input, else the first Streamed
/// output.
struct SizingAnchor {
    bool m_is_input = true;
    size_t m_index = 0;
};

/// The host blocks of a case: blocks of `m_block` samples of the anchor at `m_host_rate`, smaller
/// blocks admitted or not.
struct SizingHostBlocks {
    double m_block = 0.0;
    double m_host_rate = 0.0;
    bool m_allow_smaller = false;
    std::optional<SizingAnchor> m_anchor;
};

/// Host blocks of `block` samples of the anchor at `host_rate`.
inline SizingHostBlocks sizing_host_blocks(double block,
                                           double host_rate,
                                           bool allow_smaller = false,
                                           std::optional<SizingAnchor> anchor = std::nullopt) {
    return SizingHostBlocks{.m_block = block,
                            .m_host_rate = host_rate,
                            .m_allow_smaller = allow_smaller,
                            .m_anchor = anchor};
}

/// The anchor at input `index`.
inline SizingAnchor input_anchor(size_t index) {
    return SizingAnchor{.m_is_input = true, .m_index = index};
}

/// The tensor names of a case: "in0", "in1", ... and "out0", ...
inline std::string sizing_name(bool is_input, size_t index) {
    return (is_input ? "in" : "out") + std::to_string(index);
}

inline anira::TensorSpec sizing_spec(bool is_input, size_t index, const SizingSlot& slot) {
    const std::string name = sizing_name(is_input, index);
    if (slot.m_hop == 0) { return static_tensor(name, {slot.m_elements}); }
    const int64_t time = slot.m_elements / slot.m_channels;
    return stream_tensor(name, slot.m_channels, time, slot.m_hop < 0 ? time : slot.m_hop);
}

/// The model of a case over the custom row, with `instances` instances; `latencies` are the
/// models' own (internal) latencies of the outputs, 0 for every output without one.
inline anira::ModelConfig sizing_model(const std::vector<SizingSlot>& inputs,
                                       const std::vector<SizingSlot>& outputs,
                                       uint32_t instances,
                                       const std::optional<SizingAnchor>& anchor = std::nullopt,
                                       const std::vector<int64_t>& latencies = {}) {
    anira::ModelConfig model = custom_model(instances);
    for (size_t i = 0; i < inputs.size(); ++i) { model.input(sizing_spec(true, i, inputs[i])); }
    for (size_t i = 0; i < outputs.size(); ++i) {
        anira::TensorSpec spec = sizing_spec(false, i, outputs[i]);
        if (i < latencies.size() && latencies[i] > 0) { spec.latency(latencies[i]); }
        model.output(spec);
    }
    if (anchor.has_value()) { model.anchor(sizing_name(anchor->m_is_input, anchor->m_index)); }
    return model;
}

/// The contract of a case: its host blocks (host_block_min half the block when smaller blocks are
/// admitted, the block otherwise), an explicit budget, no warm-up, ZEROS on a miss (the 2.x
/// manager's policy; BYPASS asks for channel counts the cases do not keep) and the wait ratio.
inline anira::ContractHandle sizing_contract(const SizingHostBlocks& blocks,
                                             double budget_ms,
                                             double wait_ratio) {
    anira::Hard hard{.host_block_min = blocks.m_allow_smaller ? blocks.m_block / 2 : blocks.m_block,
                     .host_block_max = blocks.m_block,
                     .host_rate = blocks.m_host_rate,
                     .budget = std::chrono::duration<double, std::milli>(budget_ms)};
    hard.warmup = ANIRA_WARMUP_FIXED;
    hard.warmup_iterations = 0;
    hard.on_miss = ANIRA_MISS_ZEROS;
    hard.wait_ratio = wait_ratio;
    return anira::ContractHandle(hard);
}

/// What a prepared session of a case is sized at.
struct SizingResult {
    std::vector<unsigned int> m_latency;
    size_t m_num_structs = 0;
    std::vector<size_t> m_send;
    std::vector<size_t> m_receive;
};

/// A case prepared: a context of two threads and the handler over the model, prepared under
/// the contract. prepared() is the status of the prepare.
class SizingSession {
public:
    SizingSession(const anira::ModelConfig& model, const anira::ContractHandle& contract)
        : m_handler(m_context, model), m_status(m_handler.prepare(contract)) {}

    anira_status prepared() const { return m_status; }
    const char* message() const { return m_handler.m_err.message; }
    anira_handler* handler() const { return m_handler.m_handler; }

    SizingResult measure() const {
        const std::shared_ptr<anira::SessionElement> session = session_of(m_handler.m_handler);
        if (session == nullptr) { return {}; }
        return SizingResult{.m_latency = session->m_latency,
                            .m_num_structs = session->m_num_structs,
                            .m_send = session->m_send_buffer_size,
                            .m_receive = session->m_receive_buffer_size};
    }

private:
    Context m_context{2};
    Handler m_handler;
    anira_status m_status;
};

}  // namespace anira_test

#endif  // ANIRA_TEST_SUPPORT_SESSION_CASES_H
