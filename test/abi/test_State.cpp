// Declared state passing (ANIRA_ROLE_STATE, anira_tensor_spec_set_state_source), engine-free.
//
// The model is the engine-free twin of example-models' StatefulAccumulatorNetwork, a custom-row
// backend that is a pure function of its inputs:
//
//     processed       = data + state_in[..., 0]
//     state_out[...,0] = state_in[..., 0] + sum(data)       per channel, over the window
//     state_out[...,1] = state_in[..., 1] + 1
//
// with the state shaped [1, channels, 2]. It keeps nothing between two calls: whatever carries
// over from one inference to the next travelled through anira's feed and capture. So the
// expected stream is a closed form (block k = its input plus the sum of the blocks 0..k-1, the
// counter reads k + 1), a missing feed or a missing capture is wrong in every block after the
// first, and integer-valued ramps stay exact in float32. The engine-backed cases share the
// oracle.
//
// One slot space: a slot is the tensor's position in the model config's list of its side, the
// one number the host's entries, a stage, the backend, the latency vector and the plan report's
// rows use. The role decides which entries take it: no Hard entry carries a State tensor.

#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "../support/log_record_collector.h"
#include "capi/port.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::RecordCollector;
using anira_test::whole_f32;

constexpr uint32_t k_hop = 8;     // the window of the streamed tensors, and the contract's block
constexpr double k_rate = 800.0;  // a hop lasts 10 ms; the explicit budget below is 1 ms
constexpr int64_t k_channels = 2;
constexpr size_t k_state_width = 2;  // [carry, counter] per channel

// ---- the model
// -----------------------------------------------------------------------------------

/// Where the accumulator's four tensors stand in the model's lists (their slots), and the
/// Static pair of the wider model (absent: k_none).
constexpr size_t k_none = SIZE_MAX;
struct Positions {
    size_t m_state_in = 0;
    size_t m_data = 1;
    size_t m_processed = 0;
    size_t m_state_out = 1;
    size_t m_values_in = k_none;
    size_t m_values_out = k_none;
};

TensorSpec streamed(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
        .axis(2, ANIRA_AXIS_TIME, k_hop);
    spec.window(k_hop, k_hop, 0);
    return spec;
}

/// One half of the pair: [1, channels, 2], the Channel tag on a State spec. `source` names the
/// output half and is set on the input half only.
TensorSpec state(std::string_view name, const char* source = nullptr) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATE);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
        .axis(2, ANIRA_AXIS_ANY, static_cast<int64_t>(k_state_width));
    if (source != nullptr) {
        EXPECT_EQ(anira_tensor_spec_set_state_source(spec.native(), source), ANIRA_OK);
    }
    return spec;
}

TensorSpec static_values(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_ANY, 3);
    return spec;
}

/// The example model's order: the state FIRST on the input side and LAST on the output side.
/// `data` is input slot 1 and `processed_data` is output slot 0: the stream does not stand at
/// the same position on both sides, which is what the two-slot process form is for.
ModelConfig accumulator_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(state("state_in", "state_out"));
    model.input(streamed("data"));
    model.output(streamed("processed_data"));
    model.output(state("state_out"));
    return model;
}
constexpr Positions k_accumulator{};

/// The state in the MIDDLE of the input side and FIRST on the output side:
///   inputs   values_in (Static, slot 0)   state_in (State, slot 1)   data (Streamed, slot 2)
///   outputs  state_out (State, slot 0)   processed_data (Streamed, slot 1)   values_out (Static,
///   slot 2)
ModelConfig wide_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(static_values("values_in"));
    model.input(state("state_in", "state_out"));
    model.input(streamed("data"));
    model.output(state("state_out"));
    model.output(streamed("processed_data"));
    model.output(static_values("values_out"));
    return model;
}
constexpr Positions k_wide{.m_state_in = 1,
                           .m_data = 2,
                           .m_processed = 1,
                           .m_state_out = 0,
                           .m_values_in = 0,
                           .m_values_out = 2};

// ---- the backend
// -----------------------------------------------------------------------------------

/// StatefulAccumulatorNetwork, engine-free. It holds no state of its own between two calls.
/// It can hold an inference (the gate), fail one (the throw), and it records the order and the
/// overlap of its calls: the first sample of `data` is the block's index in the tests that ask.
class AccumulatorBackend : public anira::BackendBase {
public:
    AccumulatorBackend(anira::InferenceConfig& config, Positions positions)
        : anira::BackendBase(config), m_positions(positions) {}
    ~AccumulatorBackend() override { m_open.store(true); }
    AccumulatorBackend(const AccumulatorBackend&) = delete;
    AccumulatorBackend& operator=(const AccumulatorBackend&) = delete;
    AccumulatorBackend(AccumulatorBackend&&) = delete;
    AccumulatorBackend& operator=(AccumulatorBackend&&) = delete;

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        const int inside = m_inside.fetch_add(1) + 1;
        int widest = m_widest.load();
        while (inside > widest && !m_widest.compare_exchange_weak(widest, inside)) {}
        while (!m_open.load()) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
        if (m_sleep_us > 0) { std::this_thread::sleep_for(std::chrono::microseconds(m_sleep_us)); }
        if (m_throw.exchange(false)) {
            m_inside.fetch_sub(1);
            throw std::runtime_error("test backend: inference failed");
        }
        const float* state_in = input[m_positions.m_state_in].get_read_pointer(0);
        const float* data = input[m_positions.m_data].get_read_pointer(0);
        float* processed = output[m_positions.m_processed].get_write_pointer(0);
        float* state_out = output[m_positions.m_state_out].get_write_pointer(0);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            const float carry = state_in[channel * k_state_width];
            float sum = 0.0F;
            for (size_t i = 0; i < k_hop; ++i) {
                const float sample = data[(channel * k_hop) + i];
                processed[(channel * k_hop) + i] = sample + carry;
                sum += sample;
            }
            state_out[channel * k_state_width] = carry + sum;
            state_out[(channel * k_state_width) + 1] = state_in[(channel * k_state_width) + 1] + 1;
        }
        if (m_positions.m_values_in != k_none) {
            const float* values_in = input[m_positions.m_values_in].get_read_pointer(0);
            float* values_out = output[m_positions.m_values_out].get_write_pointer(0);
            for (size_t i = 0; i < 3; ++i) { values_out[i] = values_in[i] + 0.5F; }
        }
        {
            const std::scoped_lock<std::mutex> lock(m_mutex);
            m_order.push_back(data[0]);
        }
        m_calls.fetch_add(1);
        m_inside.fetch_sub(1);
    }

    std::vector<float> order() {
        const std::scoped_lock<std::mutex> lock(m_mutex);
        return m_order;
    }

    std::atomic<bool> m_open{true};
    std::atomic<bool> m_throw{false};  ///< fails the next inference, once
    std::atomic<int> m_calls{0};
    std::atomic<int> m_widest{0};  ///< the most calls that were inside process() at once
    int m_sleep_us = 0;            ///< set before the first block

private:
    Positions m_positions;
    std::atomic<int> m_inside{0};
    std::mutex m_mutex;
    std::vector<float> m_order;
};

// ---- the rig -----------------------------------------------------------------------------------

anira::ContractHandle contract_of(anira_miss_policy policy,
                                  uint32_t block_max = k_hop,
                                  anira_miss_fn miss_fn = nullptr,
                                  void* miss_user_data = nullptr) {
    anira::ContractHandle contract = anira_test::explicit_contract(k_hop, k_rate, policy, 0.0, 1.0);
    contract.hard_geometry(1, block_max, k_rate);
    contract.hard_miss_fn(miss_fn, miss_user_data);
    return contract;
}

std::vector<anira_backend_id> custom_candidates() {
    return {{.struct_size = sizeof(anira_backend_id),
             .engine = ANIRA_ENGINE_NONE,
             .provider = ANIRA_PROVIDER_DEFAULT,
             .engine_id = nullptr}};
}

/// A handler over the custom row, with an optional stage chain, prepared, the accumulator
/// attached. Input slot `data_slot()` and output slot `processed_slot()` are the two streams:
/// the tensors' positions in the model's lists.
class Rig {
public:
    explicit Rig(const ModelConfig& model,
                 Positions positions = k_accumulator,
                 const std::vector<anira_stage_desc>& stages = {},
                 anira_miss_policy policy = ANIRA_MISS_ZEROS,
                 uint32_t block_max = k_hop,
                 anira_miss_fn miss_fn = nullptr,
                 void* miss_user_data = nullptr,
                 anira_log_level level = ANIRA_LOG_ERROR)
        : m_context(4, ANIRA_WAIT_SPIN_BACKOFF, level)
        , m_positions(positions)
        , m_policy(policy)
        , m_block_max(block_max)
        , m_miss_fn(miss_fn)
        , m_miss_user_data(miss_user_data) {
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message;
        const std::array<const anira_model_config*, 1> variants{model.native()};
        const std::vector<anira_backend_id> candidates = custom_candidates();
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants.data(),
                                               1,
                                               candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &m_err),
                  ANIRA_OK)
            << m_err.message;
        for (const anira_stage_desc& stage : stages) {
            EXPECT_EQ(anira_pipeline_add_stage(m_pipeline, &stage, &m_err), ANIRA_OK)
                << m_err.message;
        }
        EXPECT_EQ(anira_handler_create(m_context.m_context, m_pipeline, &m_handler, &m_err),
                  ANIRA_OK)
            << m_err.message;
        if (m_handler != nullptr) { prepare(); }
    }
    ~Rig() {
        if (m_backend != nullptr) { m_backend->m_open.store(true); }
        anira_handler_destroy(m_handler);  // drains the in-flight work before the backend dies
        anira_pipeline_destroy(m_pipeline);
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;
    Rig(Rig&&) = delete;
    Rig& operator=(Rig&&) = delete;

    void prepare() {
        if (m_backend != nullptr) { m_backend->m_open.store(true); }
        const anira::ContractHandle contract =
            contract_of(m_policy, m_block_max, m_miss_fn, m_miss_user_data);
        m_err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(m_handler, contract.native(), &m_err), ANIRA_OK)
            << m_err.message;
        // The previous session is gone: its backend may die now.
        m_backend =
            std::make_unique<AccumulatorBackend>(m_handler->m_inference_config, m_positions);
        m_session = anira_test::session_of(m_handler);
        ASSERT_NE(m_session, nullptr);
        m_session->m_custom_processor = m_backend.get();
    }

    bool ready() const { return m_session != nullptr && m_backend != nullptr; }
    anira_handler* get() const { return m_handler; }
    AccumulatorBackend& backend() { return *m_backend; }
    anira::SessionElement& session() { return *m_session; }
    uint32_t data_slot() const { return static_cast<uint32_t>(m_positions.m_data); }
    uint32_t processed_slot() const { return static_cast<uint32_t>(m_positions.m_processed); }

    /// One host block of `samples` per channel through the waiting two-slot single form, each
    /// stream at its own position (`data` is input 1 and `processed_data` output 0 in the
    /// accumulator): the call returns with its own inferences collected, so the stream is a
    /// pure function of the calls.
    anira_status block(std::span<const float> in, std::span<float> out, size_t samples) {
        const anira_tensor in_tensor =
            whole_f32(in.data(), {k_channels, static_cast<int64_t>(samples)});
        const anira_tensor out_tensor =
            whole_f32(out.data(), {k_channels, static_cast<int64_t>(samples)});
        return anira_handler_process_wait(m_handler,
                                          &in_tensor,
                                          data_slot(),
                                          &out_tensor,
                                          processed_slot(),
                                          nullptr,
                                          ANIRA_WAIT_FOREVER);
    }

private:
    Context m_context;
    Positions m_positions;
    anira_miss_policy m_policy;
    uint32_t m_block_max;
    anira_miss_fn m_miss_fn;
    void* m_miss_user_data;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    std::unique_ptr<AccumulatorBackend> m_backend;
    std::shared_ptr<anira::SessionElement> m_session;
};

// ---- the closed form
// -----------------------------------------------------------------------------

/// Sample n of channel c of the input stream: small integers, distinct per channel.
float input_sample(size_t channel, size_t n) {
    return static_cast<float>((n % 7) + 1 + (channel * 3));
}

/// What the accumulator delivers for it when the state is fed back: the sample plus the sum of
/// every sample of the hops before its own. `first` is the stream position the state started
/// over at (0, or where a reset landed).
float expected_sample(size_t channel, size_t n, size_t first = 0) {
    float carry = 0.0F;
    const size_t hop_start = first + (((n - first) / k_hop) * k_hop);
    for (size_t m = first; m < hop_start; ++m) { carry += input_sample(channel, m); }
    return input_sample(channel, n) + carry;
}

/// The two streams of a run, per channel, as the host saw them.
struct Streams {
    std::array<std::vector<float>, static_cast<size_t>(k_channels)> m_in;
    std::array<std::vector<float>, static_cast<size_t>(k_channels)> m_out;
};

/// Drives `sizes.size()` host blocks of the input stream, continuing at stream position
/// `position`, and appends both sides to `streams`.
void drive(Rig& rig, std::span<const size_t> sizes, size_t& position, Streams& streams) {
    for (const size_t samples : sizes) {
        std::vector<float> in(static_cast<size_t>(k_channels) * samples);
        std::vector<float> out(in.size(), -1.0F);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            for (size_t i = 0; i < samples; ++i) {
                in[(channel * samples) + i] = input_sample(channel, position + i);
            }
        }
        ASSERT_EQ(rig.block(in, out, samples), ANIRA_OK);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            for (size_t i = 0; i < samples; ++i) {
                streams.m_in.at(channel).push_back(in[(channel * samples) + i]);
                streams.m_out.at(channel).push_back(out[(channel * samples) + i]);
            }
        }
        position += samples;
    }
}

/// The delivered stream is the latency's zeros, then the closed form. `first` as above, in
/// samples of the delivered stream's model time (the stream restarts there).
void expect_closed_form(const Streams& streams, size_t latency, size_t first = 0) {
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        const std::vector<float>& out = streams.m_out.at(channel);
        ASSERT_GT(out.size(), latency + k_hop) << "the run is too short to show a fed state";
        for (size_t n = 0; n < out.size(); ++n) {
            const float expected =
                n < latency ? 0.0F : expected_sample(channel, n - latency, first);
            ASSERT_EQ(out[n], expected) << "channel " << channel << ", sample " << n;
        }
    }
}

std::vector<size_t> hops(size_t count) {
    return std::vector<size_t>(count, k_hop);
}

// ---- one slot space
// ------------------------------------------------------------------------------

/// What the stage of OneSlotSpacePerSide saw in before_inference, per inference. Real-time code
/// (the inference thread): a fixed array of atomics.
struct SlotProbe {
    static constexpr size_t k_capacity = 8;
    std::array<std::atomic<float>, k_capacity> m_static_first{};  ///< model_inputs[0][0]
    std::array<std::atomic<float>, k_capacity> m_data_first{};    ///< model_inputs[2][0]
    std::atomic<size_t> m_calls{0};
    std::atomic<uint32_t> m_other_counts{0};  ///< ctxs whose tensor counts were not 3 and 3
    /// anira_plan_slot::role of every slot of plan 0, as the stage's prepare read them from the
    /// plan report (the control thread: plain members).
    std::array<uint32_t, 3> m_input_roles{ANIRA_ROLE_FORCE32,
                                          ANIRA_ROLE_FORCE32,
                                          ANIRA_ROLE_FORCE32};
    std::array<uint32_t, 3> m_output_roles{ANIRA_ROLE_FORCE32,
                                           ANIRA_ROLE_FORCE32,
                                           ANIRA_ROLE_FORCE32};
    uint32_t m_prepares = 0;
};

// A stage asks what each tensor is before it runs: the role of every slot, from the report its
// prepare receives.
anira_status ANIRA_CALL probe_roles(anira_handler* /*handler*/,
                                    const anira_plan_report* report,
                                    void* user_data) {
    auto* probe = static_cast<SlotProbe*>(user_data);
    for (const anira_bool inputs : {anira_bool{1}, anira_bool{0}}) {
        std::array<anira_plan_slot, 3> slots{ANIRA_PLAN_SLOT_INIT,
                                             ANIRA_PLAN_SLOT_INIT,
                                             ANIRA_PLAN_SLOT_INIT};
        uint32_t rows = 3;
        const anira_status status = anira_plan_report_slots(report,
                                                            0,
                                                            inputs,
                                                            sizeof(anira_plan_slot),
                                                            &rows,
                                                            slots.data());
        if (status != ANIRA_OK || rows != 3) { return ANIRA_ERROR_INVALID_STATE; }
        std::array<uint32_t, 3>& roles = inputs != 0 ? probe->m_input_roles : probe->m_output_roles;
        for (size_t slot = 0; slot < 3; ++slot) { roles.at(slot) = slots.at(slot).role; }
    }
    ++probe->m_prepares;
    return ANIRA_OK;
}

// anira_plan_slot as a caller compiled before the tail field `role` has it: the record ends
// behind `reason`.
// NOLINTBEGIN(readability-identifier-naming) the C record's own names
struct PlanSlotWithoutRole {
    uint32_t struct_size;
    uint32_t slot;
    uint32_t is_input;
    uint32_t domain_in;
    uint32_t domain_out;
    uint32_t edge_class;
    uint32_t allocate_class;
    uint32_t wait_strategy;
    const char* recipe;
    const char* reason;
};
// NOLINTEND(readability-identifier-naming)
static_assert(sizeof(PlanSlotWithoutRole) == offsetof(anira_plan_slot, role));

anira_status ANIRA_CALL probe_slots(const anira_stage_ctx* ctx, void* user_data) ANIRA_NONBLOCKING {
    auto* probe = static_cast<SlotProbe*>(user_data);
    const size_t index = probe->m_calls.fetch_add(1);
    if (ctx->num_inputs != 3 || ctx->num_outputs != 3) {
        probe->m_other_counts.fetch_add(1);
        return ANIRA_OK;
    }
    if (index < SlotProbe::k_capacity) {
        // The slots the host uses: the Static tensor it set at slot 0, the stream it pushed at
        // slot 2.
        probe->m_static_first.at(index).store(anira_tensor_data_f32(&ctx->model_inputs[0])[0]);
        probe->m_data_first.at(index).store(anira_tensor_data_f32(&ctx->model_inputs[2])[0]);
    }
    return ANIRA_OK;
}

// A slot is the tensor's position in the model config's list of its side, State tensors
// included, and that one number is the host entries', the Static entries', a stage's, the
// latency vector's and the report rows'. The model has its State tensor in the MIDDLE of the
// input list and FIRST in the output list, so a numbering that skipped it would show everywhere.
TEST(AbiState, OneSlotSpacePerSide) {
    anira_drain_log();
    RecordCollector collector;
    SlotProbe probe;
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.name = "slot-probe";
    stage.user_data = &probe;
    stage.prepare = &probe_roles;
    stage.before_inference = &probe_slots;
    Rig rig(wide_model(),
            k_wide,
            {stage},
            ANIRA_MISS_ZEROS,
            k_hop,
            nullptr,
            nullptr,
            ANIRA_LOG_DEBUG);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();

    // Three tensors per side, three slots per side.
    EXPECT_EQ(h->m_num_inputs, 3U);
    EXPECT_EQ(h->m_num_outputs, 3U);
    EXPECT_EQ(rig.data_slot(), 2U);
    EXPECT_EQ(rig.processed_slot(), 1U);
    // One port per tensor, indexed by slot, the arm by the spec's role. The two halves of the
    // pair name each other; they hold no value: no entry of the handler carries them, and the
    // chain's materialisation never touches them.
    using anira::capi::port_role;
    ASSERT_EQ(h->m_input_ports.size(), 3U);
    ASSERT_EQ(h->m_output_ports.size(), 3U);
    EXPECT_EQ(port_role(h->m_input_ports[0]), ANIRA_ROLE_STATIC);
    EXPECT_EQ(port_role(h->m_input_ports[1]), ANIRA_ROLE_STATE);
    EXPECT_EQ(port_role(h->m_input_ports[2]), ANIRA_ROLE_STREAMED);
    EXPECT_EQ(port_role(h->m_output_ports[0]), ANIRA_ROLE_STATE);
    EXPECT_EQ(port_role(h->m_output_ports[1]), ANIRA_ROLE_STREAMED);
    EXPECT_EQ(port_role(h->m_output_ports[2]), ANIRA_ROLE_STATIC);
    const auto* state_in = std::get_if<anira::capi::StatePort>(&h->m_input_ports[1]);
    const auto* state_out = std::get_if<anira::capi::StatePort>(&h->m_output_ports[0]);
    ASSERT_NE(state_in, nullptr);
    ASSERT_NE(state_out, nullptr);
    EXPECT_EQ(state_in->m_partner, 0U);
    EXPECT_EQ(state_out->m_partner, 1U);
    EXPECT_EQ(anira::capi::static_slot(h->m_input_ports, 1), nullptr);
    EXPECT_EQ(anira::capi::static_slot(h->m_output_ports, 0), nullptr);
    EXPECT_NE(anira::capi::static_slot(h->m_input_ports, 0), nullptr);
    EXPECT_NE(anira::capi::static_slot(h->m_output_ports, 2), nullptr);
    // A stream port carries what a host block of the slot must have, and the session's ring.
    const anira::capi::StreamPort* data = anira::capi::stream_port(h->m_input_ports, 2);
    const anira::capi::StreamPort* processed = anira::capi::stream_port(h->m_output_ports, 1);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(processed, nullptr);
    EXPECT_EQ(data->m_ring_dtype, static_cast<anira_dtype>(ANIRA_DTYPE_F32));
    EXPECT_EQ(data->m_ring, &h->m_manager->session().m_send_buffer[2]);
    EXPECT_EQ(processed->m_ring, &h->m_manager->session().m_receive_buffer[1]);
    EXPECT_EQ(anira::capi::stream_port(h->m_input_ports, 0), nullptr);
    EXPECT_EQ(anira::capi::stream_port(h->m_input_ports, 3), nullptr) << "beyond the side's count";

    // The latency vector: one entry per output tensor, 0 for one that is not Streamed.
    uint32_t count = 0;
    ASSERT_EQ(anira_handler_get_latencies(h, &count, nullptr), ANIRA_OK);
    ASSERT_EQ(count, 3U);
    std::array<uint32_t, 3> latencies{77, 77, 77};
    ASSERT_EQ(anira_handler_get_latencies(h, &count, latencies.data()), ANIRA_OK);
    EXPECT_EQ(latencies[0], 0U) << "output 0 is the State tensor";
    EXPECT_EQ(latencies[1], static_cast<uint32_t>(rig.session().m_latency[1]))
        << "output 1 is the stream";
    EXPECT_GT(latencies[1], 0U);
    EXPECT_EQ(latencies[2], 0U) << "output 2 is the Static tensor";
    EXPECT_EQ(anira_handler_get_latency(h, 0), 0U);
    EXPECT_EQ(anira_handler_get_latency(h, 1), latencies[1]);
    EXPECT_EQ(anira_handler_get_latency(h, 2), 0U);
    EXPECT_EQ(anira_handler_get_latency(h, 3), 0U) << "out of range";
    // The ring state likewise: a slot without a ring reads 0 and records nothing.
    size_t available = 77;
    EXPECT_EQ(anira_handler_get_available_samples(h, 0, 0, &available), ANIRA_OK);
    EXPECT_EQ(available, 0U) << "the State output has no ring";
    EXPECT_EQ(anira_handler_get_available_samples(h, 1, 0, &available), ANIRA_OK);
    EXPECT_EQ(available, latencies[1]) << "right after prepare: the reported latency";
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // The plan report: one slot row per tensor of each side, numbered by position.
    const anira_plan_report* report = anira_handler_plan_report(h);
    ASSERT_NE(report, nullptr);
    for (const anira_bool inputs : {anira_bool{1}, anira_bool{0}}) {
        uint32_t rows = 0;
        ASSERT_EQ(
            anira_plan_report_slots(report, 0, inputs, sizeof(anira_plan_slot), &rows, nullptr),
            ANIRA_OK);
        ASSERT_EQ(rows, 3U);
        std::array<anira_plan_slot, 3> slots{ANIRA_PLAN_SLOT_INIT,
                                             ANIRA_PLAN_SLOT_INIT,
                                             ANIRA_PLAN_SLOT_INIT};
        ASSERT_EQ(anira_plan_report_slots(report,
                                          0,
                                          inputs,
                                          sizeof(anira_plan_slot),
                                          &rows,
                                          slots.data()),
                  ANIRA_OK);
        for (uint32_t slot = 0; slot < 3; ++slot) { EXPECT_EQ(slots.at(slot).slot, slot); }
        // The role of every tensor, which decides the entries that take its slot: Static, State,
        // Streamed on the input side; State, Streamed, Static on the output side.
        const std::array<uint32_t, 3> expected_roles =
            inputs != 0
                ? std::array<uint32_t, 3>{ANIRA_ROLE_STATIC, ANIRA_ROLE_STATE, ANIRA_ROLE_STREAMED}
                : std::array<uint32_t, 3>{ANIRA_ROLE_STATE, ANIRA_ROLE_STREAMED, ANIRA_ROLE_STATIC};
        for (uint32_t slot = 0; slot < 3; ++slot) {
            EXPECT_EQ(slots.at(slot).role, expected_roles.at(slot))
                << (inputs != 0 ? "input " : "output ") << slot;
        }
        // The stage's prepare read the same from the report it was handed.
        EXPECT_EQ(inputs != 0 ? probe.m_input_roles : probe.m_output_roles, expected_roles);

        // A caller whose header ends before `role` still gets its rows, at its own stride: the
        // fields it knows, and nothing behind the last record it asked for.
        std::array<PlanSlotWithoutRole, 4> old_rows{};
        constexpr uint32_t k_untouched = 0xA5A5A5A5U;
        old_rows.at(3).struct_size = k_untouched;
        old_rows.at(3).slot = k_untouched;
        uint32_t old_count = 3;
        ASSERT_EQ(anira_plan_report_slots(report,
                                          0,
                                          inputs,
                                          sizeof(PlanSlotWithoutRole),
                                          &old_count,
                                          reinterpret_cast<anira_plan_slot*>(old_rows.data())),
                  ANIRA_OK);
        EXPECT_EQ(old_count, 3U);
        for (uint32_t slot = 0; slot < 3; ++slot) {
            EXPECT_EQ(old_rows.at(slot).slot, slot);
            EXPECT_EQ(old_rows.at(slot).is_input, inputs);
            EXPECT_EQ(old_rows.at(slot).edge_class, static_cast<uint32_t>(ANIRA_EDGE_ZERO_COPY));
            EXPECT_STREQ(old_rows.at(slot).recipe, "host");
            EXPECT_EQ(old_rows.at(slot).reason, nullptr);
        }
        EXPECT_EQ(old_rows.at(3).struct_size, k_untouched);
        EXPECT_EQ(old_rows.at(3).slot, k_untouched);
    }
    EXPECT_EQ(probe.m_prepares, 1U);
    // The report's log lines name the slot and the canonical name of its tensor, State included.
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "input 0 'values_in'", "native"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "input 1 'state_in'", "native"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "input 2 'data'", "native"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "output 0 'state_out'", "native"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "output 1 'processed_data'", "native"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "output 2 'values_out'", "native"), 1U);
#endif

    // The host entries and a stage agree on the numbers: the Static value set at input slot 0
    // and the stream pushed at input slot 2 are what the stage reads at model_inputs[0] and [2],
    // and the stream is popped at output slot 1.
    std::array<float, 3> values{9.0F, 2.0F, 3.0F};
    const anira_tensor whole = whole_f32(values.data(), {1, 3});
    ASSERT_EQ(anira_handler_set_static_input(h, 0, &whole), ANIRA_OK);
    const std::vector<size_t> sizes = hops(4);
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    expect_closed_form(streams, latencies[1]);
    ASSERT_EQ(probe.m_calls.load(), 4U);
    EXPECT_EQ(probe.m_other_counts.load(), 0U) << "three tensors per side, State included";
    for (size_t hop = 0; hop < 4; ++hop) {
        EXPECT_EQ(probe.m_static_first.at(hop).load(), 9.0F) << "hop " << hop;
        EXPECT_EQ(probe.m_data_first.at(hop).load(), input_sample(0, hop * k_hop)) << "hop " << hop;
    }
    std::array<float, 3> captured{};
    const anira_tensor captured_tensor = whole_f32(captured.data(), {1, 3});
    EXPECT_EQ(anira_handler_get_static_output(h, 2, &captured_tensor), ANIRA_OK);
    EXPECT_EQ(captured, (std::array<float, 3>{9.5F, 2.5F, 3.5F}));
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    // The role decides which entries take a slot. A single form takes a Streamed slot only: a
    // Static and a State slot are refused like a number out of range. One record per kind.
    std::array<float, static_cast<size_t>(k_channels) * k_hop> samples{};
    const anira_tensor block = whole_f32(samples.data(), {k_channels, k_hop});
    EXPECT_EQ(anira_handler_push_data(h, &block, 0), ANIRA_ERROR_INVALID_ARGUMENT) << "Static";
    EXPECT_EQ(anira_handler_push_data(h, &block, 1), ANIRA_ERROR_INVALID_ARGUMENT) << "State";
    EXPECT_EQ(anira_handler_push_data(h, &block, 3), ANIRA_ERROR_INVALID_ARGUMENT) << "range";
    EXPECT_EQ(anira_handler_pop_data(h, &block, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "State";
    EXPECT_EQ(anira_handler_pop_data(h, &block, 2, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "Static";
    EXPECT_EQ(anira_handler_pop_data(h, &block, 3, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "range";
    // The two-slot form refuses each side on its own.
    EXPECT_EQ(anira_handler_process(h, &block, 1, &block, 1, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a State in_slot beside the good out_slot";
    EXPECT_EQ(anira_handler_process(h, &block, 2, &block, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a State out_slot beside the good in_slot";
    // The Static entries take a Static slot only.
    EXPECT_EQ(anira_handler_set_static_input(h, 1, &whole), ANIRA_ERROR_INVALID_ARGUMENT)
        << "State";
    EXPECT_EQ(anira_handler_set_static_input(h, 2, &whole), ANIRA_ERROR_INVALID_ARGUMENT)
        << "Streamed";
    EXPECT_EQ(anira_handler_set_static_input(h, 3, &whole), ANIRA_ERROR_INVALID_ARGUMENT)
        << "range";
    EXPECT_EQ(anira_handler_get_static_output(h, 0, &whole), ANIRA_ERROR_INVALID_ARGUMENT)
        << "State";
    EXPECT_EQ(anira_handler_get_static_output(h, 1, &whole), ANIRA_ERROR_INVALID_ARGUMENT)
        << "Streamed";
    // A multi form takes one tensor per tensor of the side: another count is refused.
    std::vector<anira_tensor> two(2, block);
    EXPECT_EQ(anira_handler_push_data_multi(h, two.data(), 2), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi(h, two.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "anira_handler_push_data: ", "rt"), 1U);
#endif
}

// The position of a State tensor in an array of a multi form must be the empty tensor (a rank
// of 1 or more with an extent of 0): anything else refuses the whole call in the same
// validate-everything-first walk as the Static elements, so nothing is set and nothing pushed.
TEST(AbiState, AStatePositionInAMultiArrayMustBeEmpty) {
    anira_drain_log();
    RecordCollector collector;
    Rig rig(wide_model(), k_wide, {}, ANIRA_MISS_ZEROS, k_hop, nullptr, nullptr, ANIRA_LOG_DEBUG);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();

    std::vector<float> in(static_cast<size_t>(k_channels) * k_hop, 1.0F);
    std::vector<float> out(in.size(), -1.0F);
    std::array<float, 3> values_in{4.0F, 5.0F, 6.0F};
    std::array<float, 3> values_out{-1.0F, -1.0F, -1.0F};
    std::array<float, static_cast<size_t>(k_channels) * k_state_width> state{};
    const anira_tensor empty = whole_f32(static_cast<float*>(nullptr), {1, 0});
    const anira_tensor a_state =
        whole_f32(state.data(), {1, k_channels, static_cast<int64_t>(k_state_width)});
    // Slot order: inputs {values_in, state_in, data}, outputs {state_out, processed_data,
    // values_out}.
    std::vector<anira_tensor> inputs{whole_f32(values_in.data(), {1, 3}),
                                     a_state,
                                     whole_f32(in.data(), {k_channels, k_hop})};
    std::vector<anira_tensor> outputs{empty,
                                      whole_f32(out.data(), {k_channels, k_hop}),
                                      whole_f32(values_out.data(), {1, 3})};
    std::vector<size_t> delivered(3, 77);
    const auto stored_input = [&] {
        std::array<float, 3> stored{};
        anira::capi::static_slot(h->m_input_ports, 0)->read_packed(stored.data(), sizeof(stored));
        return stored;
    };

    // A whole state tensor at the State position of the inputs: refused, and the Static element
    // in front of it was not set, the stream behind it not pushed.
    EXPECT_EQ(anira_handler_process_multi(h, inputs.data(), 3, outputs.data(), 3, delivered.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, (std::vector<size_t>{0, 0, 0}));
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(stored_input(), (std::array<float, 3>{})) << "a refused call set nothing";
    EXPECT_EQ(rig.session().m_send_buffer[2].get_available_samples(0), 0U)
        << "a refused call pushed samples";
    EXPECT_EQ(anira_handler_push_data_multi(h, inputs.data(), 3), ANIRA_ERROR_INVALID_ARGUMENT);
    // A zeroed record is rank 0: not the empty tensor.
    inputs[1] = anira_tensor{};
    EXPECT_EQ(anira_handler_push_data_multi(h, inputs.data(), 3), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a zeroed record has no axis with an extent of 0";
    // The output side likewise, in every form that takes outputs.
    inputs[1] = empty;
    outputs[0] = a_state;
    delivered.assign(3, 77);
    EXPECT_EQ(anira_handler_process_multi(h, inputs.data(), 3, outputs.data(), 3, delivered.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, (std::vector<size_t>{0, 0, 0}));
    EXPECT_EQ(stored_input(), (std::array<float, 3>{}))
        << "the inputs were good, the output refused the call: nothing set";
    EXPECT_EQ(anira_handler_pop_data_multi(h, outputs.data(), 3, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process_multi_wait(h,
                                               inputs.data(),
                                               3,
                                               outputs.data(),
                                               3,
                                               nullptr,
                                               ANIRA_WAIT_FOREVER),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi_wait(h, outputs.data(), 3, nullptr, 0.0),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rig.session().m_send_buffer[2].get_available_samples(0), 0U);
    EXPECT_EQ(state, (std::array<float, static_cast<size_t>(k_channels) * k_state_width>{}))
        << "the memory a State element names is never touched";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    // One latched record for the kind, naming the slot and the canonical name.
    EXPECT_EQ(anira_test::count_records(collector, "input slot 1 'state_in'", "rt"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "is not the empty tensor", "rt"), 1U);
#endif

    // Any empty tensor is accepted there, whatever its rank: nothing else of it is read.
    anira_handler_reset(h);
    outputs[0] = whole_f32(static_cast<float*>(nullptr), {0});
    delivered.assign(3, 77);
    EXPECT_EQ(anira_handler_process_multi_wait(h,
                                               inputs.data(),
                                               3,
                                               outputs.data(),
                                               3,
                                               delivered.data(),
                                               ANIRA_WAIT_FOREVER),
              ANIRA_OK);
    EXPECT_EQ(delivered, (std::vector<size_t>{0, k_hop, 3})) << "a State position delivers nothing";
    EXPECT_EQ(stored_input(), values_in);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// The multi forms under a State spec: the caller's arrays, one tensor per tensor of the side
// (heap arrays of exactly that length: ASan is the guard against a loop past it), the State
// position an empty tensor, the Static elements through the store, the streams against the
// closed form, `delivered` by slot with 0 at the State position.
TEST(AbiState, MultiFormsTakeTheCallersArrays) {
    const Rig rig(wide_model(), k_wide);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    const size_t latency = anira_handler_get_latency(h, rig.processed_slot());

    Streams streams;
    std::array<float, 3> last_values{};
    for (size_t k = 0; k < 6; ++k) {
        std::vector<float> in(static_cast<size_t>(k_channels) * k_hop);
        std::vector<float> out(in.size(), -1.0F);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            for (size_t i = 0; i < k_hop; ++i) {
                in[(channel * k_hop) + i] = input_sample(channel, (k * k_hop) + i);
            }
        }
        const std::array<float, 3> values_in{static_cast<float>(k), 2.0F, 3.0F};
        std::array<float, 3> values_out{-1.0F, -1.0F, -1.0F};
        // Slot order: inputs {values_in, state_in, data}, outputs {state_out, processed_data,
        // values_out}; the State positions hold the empty tensor.
        const anira_tensor empty = whole_f32(static_cast<float*>(nullptr), {1, 0});
        const std::vector<anira_tensor> inputs{whole_f32(values_in.data(), {1, 3}),
                                               empty,
                                               whole_f32(in.data(), {k_channels, k_hop})};
        const std::vector<anira_tensor> outputs{empty,
                                                whole_f32(out.data(), {k_channels, k_hop}),
                                                whole_f32(values_out.data(), {1, 3})};
        std::vector<size_t> delivered(3, 77);
        ASSERT_EQ(anira_handler_process_multi_wait(h,
                                                   inputs.data(),
                                                   3,
                                                   outputs.data(),
                                                   3,
                                                   delivered.data(),
                                                   ANIRA_WAIT_FOREVER),
                  ANIRA_OK);
        EXPECT_EQ(delivered[0], 0U) << "the State position";
        EXPECT_EQ(delivered[1], k_hop);
        EXPECT_EQ(delivered[2], 3U);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            for (size_t i = 0; i < k_hop; ++i) {
                streams.m_in.at(channel).push_back(in[(channel * k_hop) + i]);
                streams.m_out.at(channel).push_back(out[(channel * k_hop) + i]);
            }
        }
        last_values = values_out;
    }
    expect_closed_form(streams, latency);
    EXPECT_EQ(last_values, (std::array<float, 3>{5.5F, 2.5F, 3.5F}))
        << "the Static pair beside the state: values_in + 0.5 of the last inference";
    // A multi pop that carries no stream pops nothing.
    std::vector<anira_tensor> none{whole_f32(static_cast<float*>(nullptr), {1, 0}),
                                   whole_f32(static_cast<float*>(nullptr), {k_channels, 0}),
                                   whole_f32(static_cast<float*>(nullptr), {1, 0})};
    std::vector<size_t> delivered(3, 77);
    EXPECT_EQ(anira_handler_pop_data_multi(h, none.data(), 3, delivered.data()), ANIRA_OK);
    EXPECT_EQ(delivered, (std::vector<size_t>{0, 0, 0}));
}

// ---- the feedback
// --------------------------------------------------------------------------------

// Many blocks, of sizes above and below the hop: the closed form on every sample, and the
// counter of the state reads the number of inferences.
TEST(AbiState, StateFeedsBack) {
    Rig rig(accumulator_model(), k_accumulator, {}, ANIRA_MISS_ZEROS, 2 * k_hop);
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), 0);
    const std::vector<size_t> sizes{8, 3, 13, 16, 1, 7, 8, 16, 5, 11, 8, 8, 16, 2, 6};
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    expect_closed_form(streams, latency);

    // White-box: the session's one buffer holds what the last inference left.
    const int calls = rig.backend().m_calls.load();
    ASSERT_GT(calls, 10);
    ASSERT_EQ(rig.session().m_state.size(), 1U);
    ASSERT_EQ(rig.session().m_state[0].size(), static_cast<size_t>(k_channels) * k_state_width);
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        EXPECT_EQ(rig.session().m_state[0][(channel * k_state_width) + 1],
                  static_cast<float>(calls))
            << "the counter reads k + 1 after inference k";
        float sum = 0.0F;
        for (size_t n = 0; n < static_cast<size_t>(calls) * k_hop; ++n) {
            sum += input_sample(channel, n);
        }
        EXPECT_EQ(rig.session().m_state[0][channel * k_state_width], sum);
    }
}

// A State spec may stand anywhere in either list: first and last (the example model's order),
// and in the middle of a list. The same closed form either way, through the two-slot single
// form with each stream at its own position.
TEST(AbiState, StateSpecsMayStandAnywhere) {
    const std::vector<size_t> sizes = hops(8);
    {
        // State FIRST in and LAST out: `data` is input slot 1, `processed_data` output slot 0.
        // One index shared by both lists could not name this pair of streams.
        Rig rig(accumulator_model(), k_accumulator);
        ASSERT_TRUE(rig.ready());
        EXPECT_EQ(rig.get()->m_num_inputs, 2U);
        EXPECT_EQ(rig.get()->m_num_outputs, 2U);
        EXPECT_EQ(rig.data_slot(), 1U);
        EXPECT_EQ(rig.processed_slot(), 0U);
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, anira_handler_get_latency(rig.get(), 0));
    }
    {
        // State in the MIDDLE of the inputs and FIRST of the outputs: `data` is input slot 2,
        // `processed_data` output slot 1.
        Rig rig(wide_model(), k_wide);
        ASSERT_TRUE(rig.ready());
        EXPECT_EQ(rig.data_slot(), 2U);
        EXPECT_EQ(rig.processed_slot(), 1U);
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, anira_handler_get_latency(rig.get(), rig.processed_slot()));
    }
}

// ---- reset, prepare
// ------------------------------------------------------------------------------

// anira_handler_reset re-initialises the state: the same input yields the first run's samples
// bit for bit.
TEST(AbiState, ResetReinitialisesState) {
    Rig rig(accumulator_model());
    ASSERT_TRUE(rig.ready());
    const std::vector<size_t> sizes = hops(6);
    size_t position = 0;
    Streams first;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));
    EXPECT_NE(rig.session().m_state[0][0], 0.0F);

    anira_handler_reset(rig.get());
    position = 0;
    Streams second;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, second));
    EXPECT_EQ(second.m_out, first.m_out);
    expect_closed_form(second, anira_handler_get_latency(rig.get(), 0));
}

// A reset that lands while an inference of the old stream is between its feed and its capture:
// the stale capture writes the state under the old generation, and the first inference of the
// new stream starts over on zeros. Nothing on the driving thread touches the state, so the
// reset races nothing (TSan runs this case ten times).
TEST(AbiState, AResetAcrossAnInferenceInFlightNeverSeedsTheNewStream) {
    Rig rig(accumulator_model());
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    const std::vector<size_t> sizes = hops(4);
    size_t position = 0;
    Streams first;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));

    // One more block of the old stream, held inside the engine call: fed, not yet captured.
    const int before = rig.backend().m_calls.load();
    rig.backend().m_open.store(false);
    std::array<float, static_cast<size_t>(k_channels) * k_hop> held{};
    held.fill(1000.0F);
    const anira_tensor held_tensor = whole_f32(held.data(), {k_channels, k_hop});
    ASSERT_EQ(anira_handler_push_data(h, &held_tensor, rig.data_slot()), ANIRA_OK);
    const auto start = std::chrono::steady_clock::now();
    while (rig.session().m_active_inferences.load() == 0) {
        ASSERT_LT(std::chrono::steady_clock::now(), start + std::chrono::seconds(10));
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    anira_handler_reset(h);
    rig.backend().m_open.store(true);
    while (rig.backend().m_calls.load() == before) {
        ASSERT_LT(std::chrono::steady_clock::now(), start + std::chrono::seconds(10));
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    position = 0;
    Streams second;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, second));
    EXPECT_EQ(second.m_out, first.m_out) << "the held inference's state (sums of 1000) leaked";
    expect_closed_form(second, anira_handler_get_latency(h, 0));
}

// A second prepare builds a new session: a zeroed state, the first run's samples again.
TEST(AbiState, PrepareReinitialisesState) {
    Rig rig(accumulator_model());
    ASSERT_TRUE(rig.ready());
    const std::vector<size_t> sizes = hops(5);
    size_t position = 0;
    Streams first;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));
    ASSERT_NO_FATAL_FAILURE(rig.prepare());
    ASSERT_TRUE(rig.ready());
    for (const float value : rig.session().m_state[0]) { EXPECT_EQ(value, 0.0F); }
    position = 0;
    Streams second;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, second));
    EXPECT_EQ(second.m_out, first.m_out);
    expect_closed_form(second, anira_handler_get_latency(rig.get(), 0));
}

// ---- a failed inference
// --------------------------------------------------------------------------

// An engine failure delivers zeros for its block and skips the capture: the next block continues
// from the last good state, as if the failed block's input had never been summed.
TEST(AbiState, FailedInferenceKeepsState) {
    Rig rig(accumulator_model());
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), 0);
    constexpr size_t k_failed = 3;  // the inference of hop 3 throws
    Streams streams;
    size_t position = 0;
    for (size_t k = 0; k < 7; ++k) {
        if (k == k_failed) { rig.backend().m_throw.store(true); }
        const std::vector<size_t> one = hops(1);
        ASSERT_NO_FATAL_FAILURE(drive(rig, one, position, streams));
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(rig.backend().m_calls.load(), 6) << "six inferences captured, one failed";
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        float carry = 0.0F;
        for (size_t hop = 0; hop < 7; ++hop) {
            float sum = 0.0F;
            for (size_t i = 0; i < k_hop; ++i) {
                const size_t n = (hop * k_hop) + i;
                const float sample = input_sample(channel, n);
                sum += sample;
                if (n + latency >= streams.m_out.at(channel).size()) { continue; }
                const float expected = hop == k_failed ? 0.0F : sample + carry;
                ASSERT_EQ(streams.m_out.at(channel)[n + latency], expected)
                    << "channel " << channel << ", hop " << hop << ", sample " << i;
            }
            if (hop != k_failed) { carry += sum; }
        }
    }
    EXPECT_EQ(rig.session().m_state[0][1], 6.0F) << "the counter skipped the failed inference";
}

// ---- forced Stateful
// -----------------------------------------------------------------------------

// A model that calls itself stateless, with a pair: the session is session-exclusive, and
// several inferences submitted at once run one at a time and in order (anira_plan_info has no
// lanes field to read this from). Out of order, or overlapping, the closed form would break.
TEST(AbiState, StatePairForcesStateful) {
    ModelConfig model = accumulator_model();
    model.state(ANIRA_MODEL_STATELESS);
    model.max_instances(4);
    Rig rig(model, k_accumulator, {}, ANIRA_MISS_ZEROS, 4 * k_hop);
    ASSERT_TRUE(rig.ready());
    EXPECT_TRUE(rig.get()->m_inference_config.m_session_exclusive_processor);
    rig.backend().m_sleep_us = 300;

    // Host blocks of four hops: four inferences per call, on a pool of four threads.
    const size_t latency = anira_handler_get_latency(rig.get(), 0);
    const std::vector<size_t> sizes(6, static_cast<size_t>(4) * k_hop);
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    expect_closed_form(streams, latency);
    EXPECT_EQ(rig.backend().m_widest.load(), 1) << "two inferences of the session overlapped";
    EXPECT_EQ(rig.backend().m_calls.load(), 24);
}

// ---- a stage
// ---------------------------------------------------------------------------------------

/// What the stage of StageSeesAndAltersState saw. The callbacks are real-time code (the
/// inference thread, no lock and no allocation: RTSan runs this case), so the log is a fixed
/// array of atomics the test reads when the stream has settled.
struct StageLog {
    static constexpr size_t k_capacity = 16;
    std::array<std::atomic<float>, k_capacity> m_fed{};       ///< state_in[0][0] as fed
    std::array<std::atomic<float>, k_capacity> m_produced{};  ///< state_out[0][0] as the engine
                                                              ///< left it
    std::atomic<size_t> m_num_fed{0};
    std::atomic<size_t> m_num_produced{0};
    std::atomic<uint32_t> m_other_counts{0};  ///< ctxs whose tensor counts were not 2 and 2
};
constexpr float k_offset = 1000.0F;

anira_status ANIRA_CALL alter_before(const anira_stage_ctx* ctx,
                                     void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<StageLog*>(user_data);
    // Input slot 0 is the State tensor, for the stage as for the host (whose `data` is slot 1).
    float* state_in = anira_tensor_data_f32(&ctx->model_inputs[0]);
    const size_t index = log->m_num_fed.fetch_add(1);
    if (index < StageLog::k_capacity) { log->m_fed.at(index).store(state_in[0]); }
    if (ctx->num_inputs != 2 || ctx->num_outputs != 2) { log->m_other_counts.fetch_add(1); }
    state_in[0] += k_offset;  // what the engine gets for channel 0
    return ANIRA_OK;
}

anira_status ANIRA_CALL alter_after(const anira_stage_ctx* ctx, void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<StageLog*>(user_data);
    // Output slot 1 is the State tensor, not yet captured.
    float* state_out = anira_tensor_data_f32(&ctx->model_outputs[1]);
    const size_t index = log->m_num_produced.fetch_add(1);
    if (index < StageLog::k_capacity) { log->m_produced.at(index).store(state_out[0]); }
    state_out[0] -= k_offset;  // what is captured
    return ANIRA_OK;
}

// The feed runs ahead of every before_inference stage and the capture behind every
// after_inference stage: a stage reads the fed state at the tensor's slot, what it writes is
// what the engine gets, and what it leaves in the state output is what the next inference is
// fed. The host names `data` input slot 1 in the same numbering.
TEST(AbiState, StageSeesAndAltersState) {
    StageLog log;
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.name = "state-probe";
    stage.user_data = &log;
    stage.before_inference = &alter_before;
    stage.after_inference = &alter_after;
    Rig rig(accumulator_model(), k_accumulator, {stage});
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), 0);
    const std::vector<size_t> sizes = hops(6);
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));

    ASSERT_EQ(log.m_num_fed.load(), 6U);
    ASSERT_EQ(log.m_num_produced.load(), 6U);
    EXPECT_EQ(log.m_other_counts.load(), 0U) << "two tensors per side, State included";
    float carry = 0.0F;
    for (size_t hop = 0; hop < 6; ++hop) {
        // Fed before the stage ran: the closed form's carry, without the stage's offset, so
        // the capture took what the after stage left.
        EXPECT_EQ(log.m_fed.at(hop).load(), carry) << "hop " << hop;
        for (size_t i = 0; i < k_hop; ++i) { carry += input_sample(0, (hop * k_hop) + i); }
        // The engine saw the altered state, and the stage saw the engine's state output.
        EXPECT_EQ(log.m_produced.at(hop).load(), carry + k_offset) << "hop " << hop;
    }
    for (size_t n = latency; n < streams.m_out[0].size(); ++n) {
        EXPECT_EQ(streams.m_out[0][n], expected_sample(0, n - latency) + k_offset);
        EXPECT_EQ(streams.m_out[1][n], expected_sample(1, n - latency)) << "channel 1 untouched";
    }
}

// ---- the miss function
// ---------------------------------------------------------------------------

/// What anira_miss_fn was handed.
struct MissLog {
    uint32_t m_calls = 0;
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    const anira_tensor* m_inputs = nullptr;   ///< the array itself
    const anira_tensor* m_outputs = nullptr;  ///< the array itself
    std::vector<int64_t> m_input_requests;    ///< shape[1] of every input element
    std::vector<int64_t> m_output_requests;   ///< shape[1] of every output element
    std::vector<float> m_stored;  ///< the Static output element as the function found it
};

anira_status ANIRA_CALL record_miss(anira_handler* /*handler*/,
                                    const anira_tensor* inputs,
                                    uint32_t num_inputs,
                                    const anira_tensor* outputs,
                                    uint32_t num_outputs,
                                    void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<MissLog*>(user_data);
    ++log->m_calls;
    log->m_num_inputs = num_inputs;
    log->m_num_outputs = num_outputs;
    log->m_inputs = inputs;
    log->m_outputs = outputs;
    log->m_input_requests.clear();
    log->m_output_requests.clear();
    // Every element the counts promise is read: under ASan a count past the caller's heap
    // array is a report.
    for (uint32_t i = 0; i < num_inputs; ++i) {
        log->m_input_requests.push_back(inputs[i].shape[1]);
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
        log->m_output_requests.push_back(outputs[i].shape[1]);
    }
    // The wide model's output slots: state_out 0, processed_data 1, values_out 2.
    log->m_stored.clear();
    if (num_outputs == 3 && outputs[2].shape[1] == 3) {
        const float* stored = anira_tensor_data_f32(&outputs[2]);
        log->m_stored.assign(stored, stored + 3);
    }
    // The stream: a recognisable fill.
    if (num_outputs == 3 && outputs[1].shape[1] > 0) {
        float* samples = anira_tensor_data_f32(&outputs[1]);
        for (int64_t i = 0; i < outputs[1].shape[0] * outputs[1].shape[1]; ++i) {
            samples[i] = 9.0F;
        }
    }
    return ANIRA_OK;
}

// anira_miss_fn gets the arrays the copy path was handed, one tensor per tensor of the side: the
// caller's OWN arrays from a multi form (the same pointers, the State position the empty tensor
// the caller put there), the handler's staged arrays from a two-slot single form, empty inputs
// from a pop.
TEST(AbiState, MissFunctionReceivesTheCallersArrays) {
    MissLog log;
    log.m_input_requests.reserve(8);
    log.m_output_requests.reserve(8);
    log.m_stored.reserve(8);
    Rig rig(wide_model(), k_wide, {}, ANIRA_MISS_CALLBACK, k_hop, &record_miss, &log);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    rig.backend().m_open.store(false);  // every inference is held: the ring starves

    // A multi form, until a block misses (the latency's zeros are delivered first). Slot order:
    // inputs {values_in, state_in, data}, outputs {state_out, processed_data, values_out}.
    const std::array<float, 3> values_in{1.0F, 2.0F, 3.0F};
    anira_status status = ANIRA_OK;
    const std::vector<float> in(static_cast<size_t>(k_channels) * k_hop, 1.0F);
    std::vector<float> out(static_cast<size_t>(k_channels) * k_hop);
    std::array<float, 3> values_out{};
    const anira_tensor empty = whole_f32(static_cast<float*>(nullptr), {1, 0});
    const std::vector<anira_tensor> inputs{whole_f32(values_in.data(), {1, 3}),
                                           empty,
                                           whole_f32(in.data(), {k_channels, k_hop})};
    const std::vector<anira_tensor> outputs{empty,
                                            whole_f32(out.data(), {k_channels, k_hop}),
                                            whole_f32(values_out.data(), {1, 3})};
    for (int i = 0; i < 8 && status == ANIRA_OK; ++i) {
        status = anira_handler_process_multi(h, inputs.data(), 3, outputs.data(), 3, nullptr);
    }
    ASSERT_EQ(status, ANIRA_MISSED);
    ASSERT_EQ(log.m_calls, 1U);
    EXPECT_EQ(log.m_inputs, inputs.data()) << "the caller's own array, not a staged copy";
    EXPECT_EQ(log.m_outputs, outputs.data()) << "the caller's own array, not a staged copy";
    EXPECT_EQ(log.m_num_inputs, 3U);
    EXPECT_EQ(log.m_num_outputs, 3U);
    EXPECT_EQ(log.m_input_requests, (std::vector<int64_t>{3, 0, k_hop}))
        << "{values_in, state_in, data}";
    EXPECT_EQ(log.m_output_requests, (std::vector<int64_t>{0, k_hop, 3}))
        << "{state_out, processed_data, values_out}";
    EXPECT_EQ(log.m_stored, (std::vector<float>{0.0F, 0.0F, 0.0F})) << "the stored value";
    EXPECT_EQ(out[0], 9.0F);

    // The two-slot single form: the handler's own arrays, as long as the model's lists, the
    // caller's descriptor in its slot of each side and empty tensors beside it.
    const anira_tensor in_tensor = whole_f32(out.data(), {k_channels, k_hop});
    const anira_tensor out_tensor = whole_f32(out.data(), {k_channels, k_hop});
    ASSERT_EQ(anira_handler_process(h,
                                    &in_tensor,
                                    rig.data_slot(),
                                    &out_tensor,
                                    rig.processed_slot(),
                                    nullptr),
              ANIRA_MISSED);
    ASSERT_EQ(log.m_calls, 2U);
    EXPECT_EQ(log.m_inputs, h->m_input_tensors.data());
    EXPECT_EQ(log.m_outputs, h->m_output_tensors.data());
    EXPECT_EQ(log.m_num_inputs, 3U);
    EXPECT_EQ(log.m_num_outputs, 3U);
    EXPECT_EQ(log.m_input_requests, (std::vector<int64_t>{0, 0, k_hop}));
    EXPECT_EQ(log.m_output_requests, (std::vector<int64_t>{0, k_hop, 0}));

    // A pop: empty inputs, the caller's descriptor in its output slot.
    ASSERT_EQ(anira_handler_pop_data(h, &out_tensor, rig.processed_slot(), nullptr), ANIRA_MISSED);
    ASSERT_EQ(log.m_calls, 3U);
    EXPECT_EQ(log.m_num_inputs, 3U);
    EXPECT_EQ(log.m_num_outputs, 3U);
    EXPECT_EQ(log.m_input_requests, (std::vector<int64_t>{0, 0, 0})) << "a pop passes empty inputs";
    EXPECT_EQ(log.m_output_requests, (std::vector<int64_t>{0, k_hop, 0}));

    // Between two calls the handler's arrays hold no request: a multi pop that carries no
    // stream asks for nothing, so nothing misses.
    std::vector<anira_tensor> none{whole_f32(static_cast<float*>(nullptr), {1, 0}),
                                   whole_f32(static_cast<float*>(nullptr), {k_channels, 0}),
                                   whole_f32(static_cast<float*>(nullptr), {1, 0})};
    EXPECT_EQ(anira_handler_pop_data_multi(h, none.data(), 3, nullptr), ANIRA_OK);
    EXPECT_EQ(log.m_calls, 3U);
    rig.backend().m_open.store(true);
}

// ---- validation
// ------------------------------------------------------------------------------------

/// anira_handler_create over `model`: the status, and the message in `message`.
anira_status create_status(const ModelConfig& model, std::string& message) {
    const Context context;
    message.clear();
    anira_pipeline* pipeline = nullptr;
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK);
    const std::array<const anira_model_config*, 1> variants{model.native()};
    const std::vector<anira_backend_id> candidates = custom_candidates();
    EXPECT_EQ(anira_pipeline_add_inference(pipeline,
                                           variants.data(),
                                           1,
                                           candidates.data(),
                                           static_cast<uint32_t>(candidates.size()),
                                           &err),
              ANIRA_OK)
        << err.message;
    anira_handler* created = nullptr;
    const anira_status status = anira_handler_create(context.m_context, pipeline, &created, &err);
    message = err.message;
    anira_handler_destroy(created);
    anira_pipeline_destroy(pipeline);
    return status;
}

/// The accumulator's four specs, each open to one alteration before it joins the model.
struct Parts {
    TensorSpec m_state_in = state("state_in", "state_out");
    TensorSpec m_data = streamed("data");
    TensorSpec m_processed = streamed("processed_data");
    TensorSpec m_state_out = state("state_out");

    ModelConfig model() const {
        ModelConfig model;
        model.add_model_path(anira_test::k_custom, "custom-processor");
        model.input(m_state_in);
        model.input(m_data);
        model.output(m_processed);
        model.output(m_state_out);
        return model;
    }
};

void expect_refused(const ModelConfig& model,
                    anira_status expected,
                    std::string_view tensor,
                    std::string_view fragment) {
    std::string message;
    EXPECT_EQ(create_status(model, message), expected) << message;
    EXPECT_NE(message.find("'" + std::string(tensor) + "'"), std::string::npos)
        << "the message names the tensor '" << tensor << "': " << message;
    EXPECT_NE(message.find(fragment), std::string::npos) << message;
}

// Every rule of the pairing, at anira_handler_create, with its status and a message that names
// the tensor.
TEST(AbiState, ThePairingIsValidatedAtCreate) {
    {
        std::string message;
        EXPECT_EQ(create_status(Parts{}.model(), message), ANIRA_OK) << message;
    }
    {
        Parts parts;
        parts.m_state_in = state("state_in");
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "needs a state_source");
    }
    {
        Parts parts;
        parts.m_state_in = state("state_in", "nowhere");
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "names no output tensor");
    }
    {
        // An input's name is no output's.
        Parts parts;
        parts.m_state_in = state("state_in", "data");
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "names no output tensor");
    }
    {
        Parts parts;
        parts.m_state_in = state("state_in", "processed_data");
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "is a Streamed output");
    }
    {
        // Two inputs fed from one output.
        ModelConfig model = Parts{}.model();
        model.input(state("state_in_2", "state_out"));
        expect_refused(model, ANIRA_ERROR_CONFIG, "state_out", "more than one State input");
    }
    {
        // An output nobody is fed from.
        ModelConfig model = Parts{}.model();
        model.output(state("orphan"));
        expect_refused(model, ANIRA_ERROR_CONFIG, "orphan", "no State input names");
    }
    {
        // The source on the output half (the C setter cannot know the side).
        Parts parts;
        parts.m_state_out = state("state_out", "state_out");
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_out", "is set on an output");
    }
    {
        Parts parts;
        parts.m_state_out = TensorSpec("state_out", ANIRA_DTYPE_I32, ANIRA_ROLE_STATE);
        parts.m_state_out.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
            .axis(2, ANIRA_AXIS_ANY, static_cast<int64_t>(k_state_width));
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "have one dtype");
    }
    {
        // Another extent, and another rank.
        Parts parts;
        parts.m_state_out.axis(2, ANIRA_AXIS_ANY, 3);
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "have one shape");
        Parts ranks;
        ranks.m_state_out.axis(3, ANIRA_AXIS_ANY, 1);
        expect_refused(ranks.model(), ANIRA_ERROR_CONFIG, "state_in", "have one shape");
    }
    {
        Parts parts;
        parts.m_state_in.axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_state_width));
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "has no Time axis");
    }
    {
        Parts parts;
        parts.m_state_in.window(2, 2, 0);
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_in", "has no window");
    }
    {
        Parts parts;
        parts.m_state_out.time_ratio(1, 1);
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_out", "has no time ratio");
    }
    {
        Parts parts;
        parts.m_state_out.latency(4);
        expect_refused(parts.model(), ANIRA_ERROR_CONFIG, "state_out", "has no latency");
    }
    {
        ModelConfig model = Parts{}.model();
        model.anchor("state_in");
        expect_refused(model, ANIRA_ERROR_CONFIG, "state_in", "the anchor is a Streamed tensor");
    }
    {
        // Both halves of another dtype: equal, and not supported before PR 7.
        Parts parts;
        for (TensorSpec* half : {&parts.m_state_in, &parts.m_state_out}) {
            const bool input = half == &parts.m_state_in;
            *half = TensorSpec(input ? "state_in" : "state_out", ANIRA_DTYPE_I32, ANIRA_ROLE_STATE);
            half->axis(0, ANIRA_AXIS_BATCH, 1)
                .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
                .axis(2, ANIRA_AXIS_ANY, static_cast<int64_t>(k_state_width));
        }
        ASSERT_EQ(anira_tensor_spec_set_state_source(parts.m_state_in.native(), "state_out"),
                  ANIRA_OK);
        expect_refused(parts.model(), ANIRA_ERROR_NOT_SUPPORTED, "state_in", "float32");
    }
    {
        // A transposed layout on a state tensor, like on every tensor today; a view prepares.
        const Parts parts;
        ModelConfig transposed;
        const uint32_t row = transposed.add_model_path(anira_test::k_custom, "custom-processor");
        transposed.input(parts.m_state_in).input(parts.m_data);
        transposed.output(parts.m_processed).output(parts.m_state_out);
        transposed.tensor_layout(row, "state_in", std::array{0U, 2U, 1U});
        expect_refused(transposed, ANIRA_ERROR_NOT_SUPPORTED, "state_in", "transpose");

        ModelConfig view;
        const uint32_t view_row = view.add_model_path(anira_test::k_custom, "custom-processor");
        view.input(parts.m_state_in).input(parts.m_data);
        view.output(parts.m_processed).output(parts.m_state_out);
        view.tensor_layout(view_row, "state_in", std::array{1U, 0U, 2U});
        std::string message;
        EXPECT_EQ(create_status(view, message), ANIRA_OK) << message;
    }
}

// The contract's half of the rules fires at prepare: a ring dtype that names a state tensor.
TEST(AbiState, ARingDtypeOnAStateTensorIsRefusedAtPrepare) {
    const Context context;
    const ModelConfig model = accumulator_model();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    anira_test::Handler handler(context, model, candidates);
    ASSERT_NE(handler.m_handler, nullptr);
    anira::ContractHandle contract = contract_of(ANIRA_MISS_ZEROS);
    contract.hard_ring_dtype("state_out", ANIRA_DTYPE_F32);
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_CONFIG);
    const std::string message = handler.m_err.message;
    EXPECT_NE(message.find("'state_out'"), std::string::npos) << message;
    EXPECT_NE(message.find("State tensor"), std::string::npos) << message;
}

// anira_tensor_spec_set_state_source: a NULL spec, a NULL or empty name, a spec of another role.
TEST(AbiState, TheSetterRefusals) {
    TensorSpec half = state("state_in");
    EXPECT_EQ(anira_tensor_spec_set_state_source(nullptr, "state_out"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_state_source(half.native(), nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_tensor_spec_set_state_source(half.native(), ""), ANIRA_ERROR_INVALID_ARGUMENT);
    TensorSpec stream = streamed("data");
    EXPECT_EQ(anira_tensor_spec_set_state_source(stream.native(), "state_out"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    TensorSpec values = static_values("values");
    EXPECT_EQ(anira_tensor_spec_set_state_source(values.native(), "state_out"),
              ANIRA_ERROR_INVALID_ARGUMENT);
    // A second call replaces the name.
    EXPECT_EQ(anira_tensor_spec_set_state_source(half.native(), "first"), ANIRA_OK);
    EXPECT_EQ(anira_tensor_spec_set_state_source(half.native(), "state_out"), ANIRA_OK);
    Parts parts;
    parts.m_state_in = std::move(half);
    std::string message;
    EXPECT_EQ(create_status(parts.model(), message), ANIRA_OK) << message;
}

}  // namespace
