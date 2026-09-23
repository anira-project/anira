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
// first, and integer-valued ramps stay exact in float32. The engine-backed cases at the end of
// the file run the bundled file of the model itself (stateful_accumulator.model.json, the
// stereo export at a 64-sample window) on every engine of the build it names, against the
// same oracle.
//
// One slot space: a slot is the tensor's position in the model config's list of its side, the
// one number the host's entries, a stage, the backend, the latency vector and the plan report's
// rows use. The role decides which entries take it: no Hard entry carries a State tensor.

#include <anira/InferenceConfig.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/lifecycle.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/backends/BackendBase.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "capi/handles.h"
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
/// output half and is set on the input half only; `dtype` is the pair's (float32, or int32 on
/// the registered engine).
TensorSpec state(std::string_view name,
                 const char* source = nullptr,
                 anira_dtype dtype = ANIRA_DTYPE_F32) {
    TensorSpec spec(name, dtype, ANIRA_ROLE_STATE);
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
/// the same position on both sides, which is what the two-slot process form is for. One row on
/// `engine_id` (the engine-free custom row by default; a registered engine's id otherwise,
/// whose path anira never opens), the pair in `state_dtype`.
ModelConfig accumulator_model(const char* engine_id = anira_test::k_custom,
                              anira_dtype state_dtype = ANIRA_DTYPE_F32) {
    ModelConfig model;
    model.add_model_path(engine_id, "custom-processor");
    model.input(state("state_in", "state_out", state_dtype));
    model.input(streamed("data"));
    model.output(streamed("processed_data"));
    model.output(state("state_out", nullptr, state_dtype));
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

/// The value of the State input at `slot`, read off the handler's port at quiescence (the
/// input half of a pair holds it; the session has no buffer of its own): the read buffer, what
/// the last promotion left, zeros after a prepare or at a new generation. Empty for a slot that
/// is no State input. TSan is the check that no inference touches the buffer meanwhile.
template <typename Element = float>
std::vector<Element> state_values(const anira_handler* handler, size_t slot) {
    const auto* port = slot < handler->m_input_ports.size()
                           ? std::get_if<anira::capi::StatePort>(&handler->m_input_ports[slot])
                           : nullptr;
    if (port == nullptr || !port->m_value.has_value()) { return {}; }
    const anira::capi::StateSlot& value = *port->m_value;
    std::vector<Element> values(value.num_elements());
    std::memcpy(values.data(),
                value.read(),
                std::min(value.num_bytes(), values.size() * sizeof(Element)));
    return values;
}

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
        anira_test::attach_processor(m_handler, *m_backend);
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
/// over at (0, or where a reset landed); `hop` the model's window (k_hop for the engine-free
/// twin, the file's window for the bundled model).
float expected_sample(size_t channel, size_t n, size_t first = 0, size_t hop = k_hop) {
    float carry = 0.0F;
    const size_t hop_start = first + (((n - first) / hop) * hop);
    for (size_t m = first; m < hop_start; ++m) { carry += input_sample(channel, m); }
    return input_sample(channel, n) + carry;
}

/// The two streams of a run, per channel, as the host saw them.
struct Streams {
    std::array<std::vector<float>, static_cast<size_t>(k_channels)> m_in;
    std::array<std::vector<float>, static_cast<size_t>(k_channels)> m_out;
};

/// Drives `sizes.size()` host blocks of the input stream, continuing at stream position
/// `position`, and appends both sides to `streams`. A rig is the engine-free Rig or the
/// EngineRig over the bundled file: both take one block through the two-slot single form.
template <typename RigType>
void drive(RigType& rig, std::span<const size_t> sizes, size_t& position, Streams& streams) {
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

/// The delivered stream is the latency's zeros, then the closed form. `first` and `hop` as
/// above, `first` in samples of the delivered stream's model time (the stream restarts there).
void expect_closed_form(const Streams& streams,
                        size_t latency,
                        size_t first = 0,
                        size_t hop = k_hop) {
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        const std::vector<float>& out = streams.m_out.at(channel);
        ASSERT_GT(out.size(), latency + hop) << "the run is too short to show a fed state";
        for (size_t n = 0; n < out.size(); ++n) {
            const float expected =
                n < latency ? 0.0F : expected_sample(channel, n - latency, first, hop);
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
    std::array<std::atomic<float>, k_capacity> m_static_first{};  ///< input slot 0, element 0
    std::array<std::atomic<float>, k_capacity> m_data_first{};    ///< input slot 2, element 0
    std::atomic<size_t> m_calls{0};
    std::atomic<uint32_t> m_other_counts{0};  ///< ctxs whose tensor counts were not 3 and 3
    /// What the context accessors answered in before_inference, per slot: the status and the
    /// role of each side, and the status of the input tensor accessor (the last call's; every
    /// call answers the same). The probe asks the legal questions of its phase only:
    /// before_inference exposes the model's inputs, of every role, and no ring, and a refused
    /// question would be recorded.
    std::array<std::atomic<int32_t>, 3> m_input_role_status{};
    std::array<std::atomic<int32_t>, 3> m_output_role_status{};
    std::array<std::atomic<uint32_t>, 3> m_ctx_input_roles{};
    std::array<std::atomic<uint32_t>, 3> m_ctx_output_roles{};
    std::array<std::atomic<int32_t>, 3> m_input_tensor_status{};
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
// prepare record carries.
anira_status ANIRA_CALL probe_roles(const anira_prepare_info* info,
                                    void* user_data,
                                    void** /*out_prepared*/) {
    auto* probe = static_cast<SlotProbe*>(user_data);
    const anira_plan_report* const report = info->report;
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

anira_status ANIRA_CALL probe_slots(const anira_stage_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) ANIRA_NONBLOCKING {
    auto* probe = static_cast<SlotProbe*>(user_data);
    const size_t index = probe->m_calls.fetch_add(1);
    if (ctx->num_inputs != 3 || ctx->num_outputs != 3) {
        probe->m_other_counts.fetch_add(1);
        return ANIRA_OK;
    }
    for (uint32_t slot = 0; slot < 3; ++slot) {
        anira_role input_role = ANIRA_ROLE_FORCE32;
        anira_role output_role = ANIRA_ROLE_FORCE32;
        probe->m_input_role_status.at(slot).store(anira_stage_input_role(ctx, slot, &input_role));
        probe->m_output_role_status.at(slot).store(
            anira_stage_output_role(ctx, slot, &output_role));
        probe->m_ctx_input_roles.at(slot).store(static_cast<uint32_t>(input_role));
        probe->m_ctx_output_roles.at(slot).store(static_cast<uint32_t>(output_role));
        anira_tensor tensor;
        probe->m_input_tensor_status.at(slot).store(anira_stage_input_tensor(ctx, slot, &tensor));
    }
    if (index < SlotProbe::k_capacity) {
        // The slots the host uses: the Static tensor it set at slot 0, the stream it pushed at
        // slot 2.
        anira_tensor values;
        anira_tensor data;
        if (anira_stage_input_tensor(ctx, 0, &values) != ANIRA_OK ||
            anira_stage_input_tensor(ctx, 2, &data) != ANIRA_OK) {
            return ANIRA_ERROR_INTERNAL;
        }
        probe->m_static_first.at(index).store(anira_tensor_data_f32(&values)[0]);
        probe->m_data_first.at(index).store(anira_tensor_data_f32(&data)[0]);
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
    // pair name each other; the input half holds the value, in the spec's shape and dtype, the
    // output half none: no entry of the handler carries them, and the chain's materialisation
    // never touches them.
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
    const anira::capi::StateSlot* value =
        state_in->m_value.has_value() ? &*state_in->m_value : nullptr;
    ASSERT_NE(value, nullptr);
    EXPECT_FALSE(state_out->m_value.has_value());
    EXPECT_EQ(value->dtype(), static_cast<anira_dtype>(ANIRA_DTYPE_F32));
    EXPECT_EQ(value->shape(),
              (std::vector<int64_t>{1, k_channels, static_cast<int64_t>(k_state_width)}));
    EXPECT_EQ(value->num_bytes(), static_cast<size_t>(k_channels) * k_state_width * sizeof(float));
    EXPECT_EQ(state_in->m_generation, anira::capi::StatePort::k_no_generation) << "not bound yet";
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
    // and the stream pushed at input slot 2 are what the stage reads from
    // anira_stage_input_tensor at slots 0 and 2, and the stream is popped at output slot 1.
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
    // The capture went from output slot 0 into input slot 1, the partner the output names: the
    // input half's value holds the sums of the four hops and the counter, which no adjacent
    // numbering could have found (the pair stands at different positions of the two lists).
    {
        const std::vector<float> state = state_values(h, 1);
        ASSERT_EQ(state.size(), static_cast<size_t>(k_channels) * k_state_width);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            float sum = 0.0F;
            for (size_t n = 0; n < size_t{4} * k_hop; ++n) { sum += input_sample(channel, n); }
            EXPECT_EQ(state[channel * k_state_width], sum) << "channel " << channel;
            EXPECT_EQ(state[(channel * k_state_width) + 1], 4.0F) << "channel " << channel;
        }
        EXPECT_NE(state_in->m_generation, anira::capi::StatePort::k_no_generation) << "bound";
    }
    for (size_t hop = 0; hop < 4; ++hop) {
        EXPECT_EQ(probe.m_static_first.at(hop).load(), 9.0F) << "hop " << hop;
        EXPECT_EQ(probe.m_data_first.at(hop).load(), input_sample(0, hop * k_hop)) << "hop " << hop;
    }
    // A stage asks the ctx what each tensor is, and gets the port's arm: the answer of the plan
    // report's role column, in the same numbering. before_inference exposes the model's inputs,
    // of every role: every question the probe asked answered ANIRA_OK, and it asked for no ring
    // and no output there, so nothing is recorded (anira_handler_rt_error below).
    for (size_t slot = 0; slot < 3; ++slot) {
        SCOPED_TRACE("slot " + std::to_string(slot));
        EXPECT_EQ(probe.m_input_role_status.at(slot).load(), ANIRA_OK);
        EXPECT_EQ(probe.m_output_role_status.at(slot).load(), ANIRA_OK);
        EXPECT_EQ(probe.m_ctx_input_roles.at(slot).load(),
                  static_cast<uint32_t>(port_role(h->m_input_ports[slot])));
        EXPECT_EQ(probe.m_ctx_output_roles.at(slot).load(),
                  static_cast<uint32_t>(port_role(h->m_output_ports[slot])));
        EXPECT_EQ(probe.m_ctx_input_roles.at(slot).load(), probe.m_input_roles.at(slot));
        EXPECT_EQ(probe.m_ctx_output_roles.at(slot).load(), probe.m_output_roles.at(slot));
        EXPECT_EQ(probe.m_input_tensor_status.at(slot).load(), ANIRA_OK);
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

    // White-box: the port of the State input holds what the last inference left.
    const int calls = rig.backend().m_calls.load();
    ASSERT_GT(calls, 10);
    const std::vector<float> state = state_values(rig.get(), k_accumulator.m_state_in);
    ASSERT_EQ(state.size(), static_cast<size_t>(k_channels) * k_state_width);
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        EXPECT_EQ(state[(channel * k_state_width) + 1], static_cast<float>(calls))
            << "the counter reads k + 1 after inference k";
        float sum = 0.0F;
        for (size_t n = 0; n < static_cast<size_t>(calls) * k_hop; ++n) {
            sum += input_sample(channel, n);
        }
        EXPECT_EQ(state[channel * k_state_width], sum);
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
    EXPECT_NE(state_values(rig.get(), k_accumulator.m_state_in).at(0), 0.0F);

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

// A second prepare builds a new session and zeroes the state: the port of the State input is
// the same object (the handler's, built at create; nothing but prepare and the capture writes
// its value), the value is zeros again and the port has forgotten the generation it was last
// fed under; the first run's samples again.
TEST(AbiState, PrepareReinitialisesState) {
    Rig rig(accumulator_model());
    ASSERT_TRUE(rig.ready());
    const std::vector<size_t> sizes = hops(5);
    size_t position = 0;
    Streams first;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));
    const auto* port =
        std::get_if<anira::capi::StatePort>(&rig.get()->m_input_ports[k_accumulator.m_state_in]);
    ASSERT_NE(port, nullptr);
    ASSERT_TRUE(port->m_value.has_value());
    EXPECT_NE(port->m_generation, anira::capi::StatePort::k_no_generation) << "bound";
    EXPECT_NE(state_values(rig.get(), k_accumulator.m_state_in).at(0), 0.0F);
    ASSERT_NO_FATAL_FAILURE(rig.prepare());
    ASSERT_TRUE(rig.ready());
    EXPECT_EQ(
        std::get_if<anira::capi::StatePort>(&rig.get()->m_input_ports[k_accumulator.m_state_in]),
        port)
        << "the port survives the re-prepare";
    EXPECT_EQ(port->m_generation, anira::capi::StatePort::k_no_generation);
    const std::vector<float> zeros = state_values(rig.get(), k_accumulator.m_state_in);
    ASSERT_EQ(zeros.size(), static_cast<size_t>(k_channels) * k_state_width);
    for (const float value : zeros) { EXPECT_EQ(value, 0.0F); }
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
    EXPECT_EQ(state_values(rig.get(), k_accumulator.m_state_in).at(1), 6.0F)
        << "the counter skipped the failed inference";
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
                                     void* /*prepared*/,
                                     void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<StageLog*>(user_data);
    // Input slot 0 is the State tensor, for the stage as for the host (whose `data` is slot 1).
    anira_tensor tensor;
    if (anira_stage_input_tensor(ctx, 0, &tensor) != ANIRA_OK) { return ANIRA_ERROR_INTERNAL; }
    float* state_in = anira_tensor_data_f32(&tensor);
    const size_t index = log->m_num_fed.fetch_add(1);
    if (index < StageLog::k_capacity) { log->m_fed.at(index).store(state_in[0]); }
    if (ctx->num_inputs != 2 || ctx->num_outputs != 2) { log->m_other_counts.fetch_add(1); }
    state_in[0] += k_offset;  // what the engine gets for channel 0
    return ANIRA_OK;
}

anira_status ANIRA_CALL alter_after(const anira_stage_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<StageLog*>(user_data);
    // Output slot 1 is the State tensor, not yet captured.
    anira_tensor tensor;
    if (anira_stage_output_tensor(ctx, 1, &tensor) != ANIRA_OK) { return ANIRA_ERROR_INTERNAL; }
    float* state_out = anira_tensor_data_f32(&tensor);
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

// ---- the host-end phases and a State slot
// ---------------------------------------------------------------------------

/// "Not asked": what a question the stage did not ask leaves in its status slot.
constexpr int32_t k_not_asked = static_cast<int32_t>(ANIRA_STATUS_FORCE32);

/// What the context accessors answered in pre_process and post_process, per slot (the last
/// call's; every call answers the same). Written on the driving thread. The stage asks the
/// role of every slot first. A well-behaved one (m_ask_state false) then asks only what the
/// slot has, the ring and the model end of the Streamed slot; one that asks anyway
/// (m_ask_state true) asks the State slot for both in each host-end phase, which is the
/// stage's bug and is recorded.
struct HostEndLog {
    bool m_ask_state = false;
    anira_status m_recorded = ANIRA_OK;  ///< anira_handler_rt_error after the run
    std::array<std::atomic<uint32_t>, 2> m_pre_roles{};
    std::array<std::atomic<uint32_t>, 2> m_post_roles{};
    std::array<std::atomic<int32_t>, 2> m_pre_input_status{k_not_asked, k_not_asked};
    std::array<std::atomic<int32_t>, 2> m_post_output_status{k_not_asked, k_not_asked};
    std::array<std::atomic<int32_t>, 2> m_pre_input_rings{k_not_asked, k_not_asked};
    std::array<std::atomic<int32_t>, 2> m_post_output_rings{k_not_asked, k_not_asked};
    std::atomic<uint32_t> m_pre_dtype_of_state{1};  ///< the record a refused accessor leaves
    std::atomic<int> m_pre_ring_of_state{1};        ///< 0 when the refused accessor left NULL
};

/// Asks every slot in the two host-end phases as the log says, then runs the default body of
/// the phase.
anira_status ANIRA_CALL ask_host_end(const anira_stage_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<HostEndLog*>(user_data);
    if (ctx->num_inputs != 2 || ctx->num_outputs != 2) { return ANIRA_ERROR_INTERNAL; }
    const bool pre = ctx->phase == ANIRA_PHASE_PRE_PROCESS;
    if (!pre && ctx->phase != ANIRA_PHASE_POST_PROCESS) { return ANIRA_OK; }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        anira_role role = ANIRA_ROLE_FORCE32;
        const anira_status asked = pre ? anira_stage_input_role(ctx, slot, &role)
                                       : anira_stage_output_role(ctx, slot, &role);
        if (asked != ANIRA_OK) { return asked; }
        (pre ? log->m_pre_roles : log->m_post_roles).at(slot).store(static_cast<uint32_t>(role));
        // A State slot has no host end: a correct stage asks nothing else of it here.
        if (role == ANIRA_ROLE_STATE && !log->m_ask_state) { continue; }
        anira_ring* ring = nullptr;
        anira_tensor tensor;
        const anira_status has_ring = pre ? anira_stage_input_ring(ctx, slot, &ring)
                                          : anira_stage_output_ring(ctx, slot, &ring);
        const anira_status exposed = pre ? anira_stage_input_tensor(ctx, slot, &tensor)
                                         : anira_stage_output_tensor(ctx, slot, &tensor);
        (pre ? log->m_pre_input_rings : log->m_post_output_rings).at(slot).store(has_ring);
        (pre ? log->m_pre_input_status : log->m_post_output_status).at(slot).store(exposed);
        if (pre && slot == 0) {
            log->m_pre_dtype_of_state.store(tensor.dtype);
            log->m_pre_ring_of_state.store(ring != nullptr ? 1 : 0);
        }
    }
    return pre ? anira_stage_default_pre_process(ctx) : anira_stage_default_post_process(ctx);
}

/// Runs the accumulator with ask_host_end over `log` for four hops and checks the stream.
void run_host_end_probe(HostEndLog& log) {
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.user_data = &log;
    stage.flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST;
    stage.pre_process = &ask_host_end;
    stage.post_process = &ask_host_end;
    Rig rig(accumulator_model(), k_accumulator, {stage});
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), 0);
    const std::vector<size_t> sizes = hops(4);
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    // The stream is the closed form either way: the stage composes the defaults, and a refused
    // accessor fails no chunk.
    expect_closed_form(streams, latency);
    // The roles, on both sides: input slot 0 is the State half, slot 1 the stream; output slot
    // 0 the stream, slot 1 the State half. The Streamed slot answers as always.
    EXPECT_EQ(log.m_pre_roles.at(0).load(), static_cast<uint32_t>(ANIRA_ROLE_STATE));
    EXPECT_EQ(log.m_pre_roles.at(1).load(), static_cast<uint32_t>(ANIRA_ROLE_STREAMED));
    EXPECT_EQ(log.m_post_roles.at(0).load(), static_cast<uint32_t>(ANIRA_ROLE_STREAMED));
    EXPECT_EQ(log.m_post_roles.at(1).load(), static_cast<uint32_t>(ANIRA_ROLE_STATE));
    EXPECT_EQ(log.m_pre_input_rings.at(1).load(), ANIRA_OK);
    EXPECT_EQ(log.m_pre_input_status.at(1).load(), ANIRA_OK);
    EXPECT_EQ(log.m_post_output_rings.at(0).load(), ANIRA_OK);
    EXPECT_EQ(log.m_post_output_status.at(0).load(), ANIRA_OK);
    // What the State slot answered is the caller's to check, beside the latch's word.
    log.m_recorded = anira_handler_rt_error(rig.get());
}

// A State slot has no host end: it is fed behind pre_process and captured ahead of
// post_process. A stage that asks the role first sees ANIRA_ROLE_STATE and asks nothing else of
// the slot in the two host-end phases, and nothing is recorded. A stage that asks the State
// slot for its ring or its model end there gets ANIRA_ERROR_INVALID_STATE (no ring, an all-zero
// record), and its bug is recorded: anira_handler_rt_error, and one latched record per kind
// naming the entry, the stage, the slot and the phase.
TEST(AbiState, TheHostEndPhasesDoNotExposeAStateSlot) {
    anira_drain_log();
    RecordCollector collector;
    {
        HostEndLog log;
        ASSERT_NO_FATAL_FAILURE(run_host_end_probe(log));
        EXPECT_EQ(log.m_recorded, ANIRA_OK)
            << "a stage that asks only what the slot has records nothing";
        EXPECT_EQ(log.m_pre_input_rings.at(0).load(), k_not_asked);
        EXPECT_EQ(log.m_pre_input_status.at(0).load(), k_not_asked);
        EXPECT_EQ(log.m_post_output_rings.at(1).load(), k_not_asked);
        EXPECT_EQ(log.m_post_output_status.at(1).load(), k_not_asked);
    }
    {
        HostEndLog log;
        log.m_ask_state = true;
        ASSERT_NO_FATAL_FAILURE(run_host_end_probe(log));
        EXPECT_EQ(log.m_recorded, ANIRA_ERROR_INVALID_STATE)
            << "asking a State slot for a host end is the stage's bug, recorded";
        EXPECT_EQ(log.m_pre_input_rings.at(0).load(), ANIRA_ERROR_INVALID_STATE);
        EXPECT_EQ(log.m_pre_ring_of_state.load(), 0) << "no ring";
        EXPECT_EQ(log.m_pre_input_status.at(0).load(), ANIRA_ERROR_INVALID_STATE);
        EXPECT_EQ(log.m_pre_dtype_of_state.load(), 0U) << "an all-zero record";
        EXPECT_EQ(log.m_post_output_rings.at(1).load(), ANIRA_ERROR_INVALID_STATE);
        EXPECT_EQ(log.m_post_output_status.at(1).load(), ANIRA_ERROR_INVALID_STATE);
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    // One record per kind: the first refusal, the ring of the State input in pre_process,
    // naming the entry, the slot and the phase; the others of the kind are counted.
    EXPECT_EQ(anira_test::count_records(collector, "has no ring in", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "has no ring in", "rt");
    EXPECT_NE(record.m_message.find("anira_stage_input_ring: the stage: slot 0 has no ring in "
                                    "pre_process"),
              std::string::npos)
        << record.m_message;
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(anira_test::count_records(collector, "has no model tensor in", "rt"), 0U)
        << "the same kind: counted, not logged";
#endif
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
        // Both halves of another dtype, equal: legal, since a pair lives in the handler's own
        // buffers in the spec's dtype and never in the queue (AnInt32StatePair runs one); the
        // float32 rule of the queue stands on the Streamed tensor beside it.
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
        std::string message;
        EXPECT_EQ(create_status(parts.model(), message), ANIRA_OK) << message;
        Parts stream;
        stream.m_data = TensorSpec("data", ANIRA_DTYPE_I32, ANIRA_ROLE_STREAMED);
        stream.m_data.axis(0, ANIRA_AXIS_BATCH, 1)
            .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
            .axis(2, ANIRA_AXIS_TIME, k_hop);
        stream.m_data.window(k_hop, k_hop, 0);
        expect_refused(stream.model(), ANIRA_ERROR_NOT_SUPPORTED, "data", "float32");
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

// The host-end domain is declared per tensor of either side, a State half included: host
// memory is accepted on both halves, anything else is refused at prepare naming the half.
TEST(AbiState, AHostDomainOnAStateTensorResolvesAtPrepare) {
    const Context context;
    const ModelConfig model = accumulator_model();
    const std::vector<anira_backend_id> candidates = custom_candidates();
    anira_test::Handler handler(context, model, candidates);
    ASSERT_NE(handler.m_handler, nullptr);
    anira::ContractHandle host = contract_of(ANIRA_MISS_ZEROS);
    host.host_domain("state_in", ANIRA_DOMAIN_HOST).host_domain("state_out", ANIRA_DOMAIN_HOST);
    EXPECT_EQ(handler.prepare(host), ANIRA_OK) << handler.m_err.message;
    anira::ContractHandle device = contract_of(ANIRA_MISS_ZEROS);
    device.host_domain("state_in", ANIRA_DOMAIN_CUDA);
    EXPECT_EQ(handler.prepare(device), ANIRA_ERROR_NOT_SUPPORTED);
    const std::string message = handler.m_err.message;
    EXPECT_NE(message.find("the host domain of 'state_in'"), std::string::npos) << message;
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

// ---- the registered engine
// ------------------------------------------------------------------------

/// The accumulator as a registered C engine over the descriptors (anira/abi/engine.h): the
/// function of AccumulatorBackend, its slots found by the canonical names of its load record
/// (so the specs may stand in any order) and read back through ctx->loaded, the state in the
/// spec's dtype (float32, or int32), the stream float32. Its prepare hands back nothing (a
/// NULL prepared is legal: the engine keeps nothing per handler), and it keeps nothing between
/// two calls, so whatever carries over travelled through the pair's two buffers. process runs
/// on the inference thread: atomics and the tensors alone.
struct AccumulatorEngine {
    struct Loaded {
        uint32_t m_state_in = 0;
        uint32_t m_data = 0;
        uint32_t m_processed = 0;
        uint32_t m_state_out = 0;
    };
    static constexpr size_t k_capacity = 64;
    std::atomic<int> m_calls{0};
    std::atomic<anira_status> m_fail_next{ANIRA_OK};  ///< what the next process returns, once
    std::atomic<int> m_bad_prepared{0};               ///< a process with a prepared pointer
    int m_loaded = 0;
    int m_unloaded = 0;
    int m_prepared = 0;
    int m_unprepared = 0;
    int m_released = 0;
    /// Per call, before the engine writes: element 0 of the State input and of the State output
    /// as found in their buffers (the read side and the write side of the pair).
    std::array<std::atomic<float>, k_capacity> m_read_first{};
    std::array<std::atomic<float>, k_capacity> m_write_first{};
};

anira_status ANIRA_CALL accumulator_load(const anira_engine_load_info* info,
                                         void* user_data,
                                         void** out_loaded) {
    auto* engine = static_cast<AccumulatorEngine*>(user_data);
    ++engine->m_loaded;
    const auto find =
        [](const char* const* names, uint32_t count, std::string_view wanted, uint32_t& slot) {
            for (uint32_t i = 0; i < count; ++i) {
                if (std::string_view(names[i]) == wanted) {
                    slot = i;
                    return true;
                }
            }
            return false;
        };
    auto loaded = std::make_unique<AccumulatorEngine::Loaded>();
    if (!find(info->input_names, info->num_inputs, "state_in", loaded->m_state_in) ||
        !find(info->input_names, info->num_inputs, "data", loaded->m_data) ||
        !find(info->output_names, info->num_outputs, "processed_data", loaded->m_processed) ||
        !find(info->output_names, info->num_outputs, "state_out", loaded->m_state_out)) {
        return ANIRA_ERROR_CONFIG;
    }
    *out_loaded = loaded.release();
    return ANIRA_OK;
}

void ANIRA_CALL accumulator_unload(void* loaded, void* user_data) {
    ++static_cast<AccumulatorEngine*>(user_data)->m_unloaded;
    delete static_cast<AccumulatorEngine::Loaded*>(loaded);
}

anira_status ANIRA_CALL accumulator_prepare(const anira_prepare_info* info,
                                            void* loaded,
                                            void* user_data,
                                            void** out_prepared) {
    auto* engine = static_cast<AccumulatorEngine*>(user_data);
    ++engine->m_prepared;
    if (info == nullptr || loaded == nullptr || out_prepared == nullptr) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    *out_prepared = nullptr;  // nothing per handler: process finds its slots in ctx->loaded
    return ANIRA_OK;
}

/// The accumulator over one call's memory, the state in `State`.
template <typename State>
void accumulate(const State* state_in,
                const float* data,
                float* processed,
                State* state_out,
                size_t hop) {
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        const State carry = state_in[channel * k_state_width];
        State sum = 0;
        for (size_t i = 0; i < hop; ++i) {
            const float sample = data[(channel * hop) + i];
            processed[(channel * hop) + i] = sample + static_cast<float>(carry);
            sum += static_cast<State>(sample);
        }
        state_out[channel * k_state_width] = carry + sum;
        state_out[(channel * k_state_width) + 1] = state_in[(channel * k_state_width) + 1] + 1;
    }
}

anira_status ANIRA_CALL accumulator_process(const anira_engine_ctx* ctx,
                                            void* prepared,
                                            void* user_data) {
    auto* engine = static_cast<AccumulatorEngine*>(user_data);
    if (prepared != nullptr) { engine->m_bad_prepared.fetch_add(1); }
    const auto* slots = static_cast<const AccumulatorEngine::Loaded*>(ctx->loaded);
    if (slots == nullptr) { return ANIRA_ERROR_INTERNAL; }
    const auto call = static_cast<size_t>(engine->m_calls.fetch_add(1));
    const anira_tensor& state_in = ctx->inputs[slots->m_state_in];
    const anira_tensor& data = ctx->inputs[slots->m_data];
    const anira_tensor& processed = ctx->outputs[slots->m_processed];
    const anira_tensor& state_out = ctx->outputs[slots->m_state_out];
    const size_t hop = data.ndim == 3 ? static_cast<size_t>(data.shape[2]) : 0;
    const float* samples = anira_tensor_data_f32(&data);
    float* out = anira_tensor_data_f32(&processed);
    if (samples == nullptr || out == nullptr || hop == 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    if (state_in.dtype == ANIRA_DTYPE_F32) {
        const float* in = anira_tensor_data_f32(&state_in);
        float* next = anira_tensor_data_f32(&state_out);
        if (in == nullptr || next == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        if (call < AccumulatorEngine::k_capacity) {
            engine->m_read_first.at(call).store(in[0]);
            engine->m_write_first.at(call).store(next[0]);
        }
        accumulate(in, samples, out, next, hop);
    } else if (state_in.dtype == ANIRA_DTYPE_I32) {
        const auto* in = static_cast<const int32_t*>(anira_tensor_data(&state_in, ANIRA_DTYPE_I32));
        auto* next = static_cast<int32_t*>(anira_tensor_data(&state_out, ANIRA_DTYPE_I32));
        if (in == nullptr || next == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        if (call < AccumulatorEngine::k_capacity) {
            engine->m_read_first.at(call).store(static_cast<float>(in[0]));
            engine->m_write_first.at(call).store(static_cast<float>(next[0]));
        }
        accumulate(in, samples, out, next, hop);
    } else {
        return ANIRA_ERROR_CONFIG;
    }
    return engine->m_fail_next.exchange(ANIRA_OK);
}

void ANIRA_CALL accumulator_unprepare(void* /*prepared*/, void* user_data) {
    ++static_cast<AccumulatorEngine*>(user_data)->m_unprepared;
}

void ANIRA_CALL accumulator_release(void* user_data) {
    ++static_cast<AccumulatorEngine*>(user_data)->m_released;
}

anira_engine_desc accumulator_desc(AccumulatorEngine& engine) {
    anira_engine_desc desc = ANIRA_ENGINE_DESC_INIT;
    desc.user_data = &engine;
    desc.process = accumulator_process;
    desc.prepare = accumulator_prepare;
    desc.unprepare = accumulator_unprepare;
    desc.load = accumulator_load;
    desc.unload = accumulator_unload;
    desc.release = accumulator_release;
    return desc;
}

/// The id the registered accumulator runs under, and a second one for the plan switch.
constexpr const char* k_accumulator_id = "org.example.accumulator";
constexpr const char* k_other_accumulator_id = "org.example.accumulator.other";

/// One registration for the rig: the id and the engine behind it.
struct Registration {
    const char* m_id;
    AccumulatorEngine* m_engine;
};

/// A handler over a model whose rows name registered accumulator engines (registered on the
/// pipeline before the inference stage, the engines outliving the rig), with an optional stage,
/// prepared with the accumulator's contract. The streams stand at the accumulator's positions.
class RegisteredRig {
public:
    explicit RegisteredRig(const ModelConfig& model,
                           std::span<const Registration> registrations,
                           const std::vector<anira_stage_desc>& stages = {},
                           uint32_t block_max = k_hop)
        : m_context(4), m_block_max(block_max) {
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message;
        for (const Registration& registration : registrations) {
            const anira_engine_desc desc = accumulator_desc(*registration.m_engine);
            anira_custom_engine* engine = nullptr;
            EXPECT_EQ(anira_custom_engine_create(&desc, &engine, &m_err), ANIRA_OK)
                << m_err.message;
            EXPECT_EQ(anira_pipeline_add_engine(m_pipeline, registration.m_id, engine, &m_err),
                      ANIRA_OK)
                << m_err.message;
            anira_custom_engine_destroy(engine);
        }
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
    ~RegisteredRig() {
        anira_handler_destroy(m_handler);
        anira_pipeline_destroy(m_pipeline);
    }
    RegisteredRig(const RegisteredRig&) = delete;
    RegisteredRig& operator=(const RegisteredRig&) = delete;
    RegisteredRig(RegisteredRig&&) = delete;
    RegisteredRig& operator=(RegisteredRig&&) = delete;

    void prepare() {
        const anira::ContractHandle contract = contract_of(ANIRA_MISS_ZEROS, m_block_max);
        m_err = ANIRA_ERROR_INIT;
        ASSERT_EQ(anira_handler_prepare(m_handler, contract.native(), &m_err), ANIRA_OK)
            << m_err.message;
        m_prepared = true;
    }

    bool ready() const { return m_handler != nullptr && m_prepared; }
    anira_handler* get() const { return m_handler; }
    static constexpr uint32_t data_slot() { return static_cast<uint32_t>(k_accumulator.m_data); }
    static constexpr uint32_t processed_slot() {
        return static_cast<uint32_t>(k_accumulator.m_processed);
    }

    /// One host block through the waiting two-slot single form (Rig::block).
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
    uint32_t m_block_max;
    anira_pipeline* m_pipeline = nullptr;
    anira_handler* m_handler = nullptr;
    anira_error m_err = ANIRA_ERROR_INIT;
    bool m_prepared = false;
};

/// What a stage's two hooks saw of the State pair, per inference: the memory the State input
/// (slot 0) named in before_inference and the memory the State output (slot 1) named in
/// after_inference. Real-time code (the inference thread): fixed arrays of atomics.
struct PointerLog {
    static constexpr size_t k_capacity = 64;
    std::array<std::atomic<const void*>, k_capacity> m_in{};
    std::array<std::atomic<const void*>, k_capacity> m_out{};
    std::atomic<size_t> m_num_in{0};
    std::atomic<size_t> m_num_out{0};
    std::atomic<uint32_t> m_refused{0};  ///< accessor calls that did not answer ANIRA_OK
};

/// The memory a descriptor names.
const void* memory_of(const anira_tensor& tensor) {
    return static_cast<const unsigned char*>(tensor.handle.host.ptr) +
           static_cast<size_t>(tensor.byte_offset);
}

anira_status ANIRA_CALL log_state_in(const anira_stage_ctx* ctx,
                                     void* /*prepared*/,
                                     void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<PointerLog*>(user_data);
    anira_tensor tensor;
    if (anira_stage_input_tensor(ctx, 0, &tensor) != ANIRA_OK) {
        log->m_refused.fetch_add(1);
        return ANIRA_OK;
    }
    const size_t index = log->m_num_in.fetch_add(1);
    if (index < PointerLog::k_capacity) { log->m_in.at(index).store(memory_of(tensor)); }
    return ANIRA_OK;
}

anira_status ANIRA_CALL log_state_out(const anira_stage_ctx* ctx,
                                      void* /*prepared*/,
                                      void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<PointerLog*>(user_data);
    anira_tensor tensor;
    if (anira_stage_output_tensor(ctx, 1, &tensor) != ANIRA_OK) {
        log->m_refused.fetch_add(1);
        return ANIRA_OK;
    }
    const size_t index = log->m_num_out.fetch_add(1);
    if (index < PointerLog::k_capacity) { log->m_out.at(index).store(memory_of(tensor)); }
    return ANIRA_OK;
}

anira_stage_desc logging_stage(PointerLog& log) {
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.user_data = &log;
    stage.before_inference = &log_state_in;
    stage.after_inference = &log_state_out;
    return stage;
}

/// The pointer identity of the ping-pong over `inferences` successful inferences, all logged:
/// what inference k produced is what inference k + 1 reads (in[k + 1] == out[k]), no
/// inference reads what it writes, and exactly two addresses alternate, each on a 64-byte
/// boundary.
void expect_two_alternating_buffers(const PointerLog& log, size_t inferences) {
    ASSERT_LE(inferences, PointerLog::k_capacity);
    ASSERT_EQ(log.m_num_in.load(), inferences);
    ASSERT_EQ(log.m_num_out.load(), inferences);
    EXPECT_EQ(log.m_refused.load(), 0U);
    std::vector<const void*> addresses;
    for (size_t k = 0; k < inferences; ++k) {
        const void* in = log.m_in.at(k).load();
        const void* out = log.m_out.at(k).load();
        ASSERT_NE(in, nullptr) << "inference " << k;
        ASSERT_NE(out, nullptr) << "inference " << k;
        EXPECT_NE(in, out) << "inference " << k << " reads what it writes";
        if (k + 1 < inferences) {
            EXPECT_EQ(log.m_in.at(k + 1).load(), out)
                << "inference " << k + 1 << " does not read what inference " << k << " wrote";
        }
        for (const void* address : {in, out}) {
            if (std::ranges::find(addresses, address) == addresses.end()) {
                addresses.push_back(address);
            }
            EXPECT_EQ(reinterpret_cast<uintptr_t>(address) % 64, 0U) << "unaligned";
        }
    }
    EXPECT_EQ(addresses.size(), 2U) << "the pair alternates between exactly two buffers";
}

// The state travels by pointer: through a stage's hooks, the State input of inference k + 1
// names the memory the State output of inference k named, no inference reads what it writes,
// and exactly two buffers alternate, on the registered C accumulator (which binds the
// descriptors as they are). The closed form holds meanwhile.
TEST(AbiState, TheStateAlternatesBetweenTwoBuffers) {
    AccumulatorEngine engine;
    PointerLog log;
    const std::array<Registration, 1> registrations{
        {{.m_id = k_accumulator_id, .m_engine = &engine}}};
    RegisteredRig rig(accumulator_model(k_accumulator_id), registrations, {logging_stage(log)});
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), RegisteredRig::processed_slot());
    const std::vector<size_t> sizes = hops(8);
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    expect_closed_form(streams, latency);
    EXPECT_EQ(engine.m_calls.load(), 8);
    ASSERT_NO_FATAL_FAILURE(expect_two_alternating_buffers(log, 8));
    // The port's read buffer is the last promoted state: the memory the last inference wrote.
    const auto* port = std::get_if<anira::capi::StatePort>(&rig.get()->m_input_ports[0]);
    ASSERT_NE(port, nullptr);
    const anira::capi::StateSlot* value = port->m_value.has_value() ? &*port->m_value : nullptr;
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->read(), log.m_out.at(7).load());
    EXPECT_EQ(state_values(rig.get(), 0).at(1), 8.0F);
}

// A failed inference does not promote: the pair keeps its read buffer, so the next inference
// reads the same memory the failed one read, and the closed form continues as if the failed
// block's input had never been summed (the failed block itself delivers zeros).
TEST(AbiState, AFailingInferenceDoesNotPromote) {
    AccumulatorEngine engine;
    PointerLog log;
    const std::array<Registration, 1> registrations{
        {{.m_id = k_accumulator_id, .m_engine = &engine}}};
    RegisteredRig rig(accumulator_model(k_accumulator_id), registrations, {logging_stage(log)});
    ASSERT_TRUE(rig.ready());
    const size_t latency = anira_handler_get_latency(rig.get(), RegisteredRig::processed_slot());
    constexpr size_t k_failed = 3;  // the inference of hop 3 fails
    Streams streams;
    size_t position = 0;
    for (size_t k = 0; k < 7; ++k) {
        if (k == k_failed) { engine.m_fail_next.store(ANIRA_ERROR_ENGINE); }
        const std::vector<size_t> one = hops(1);
        ASSERT_NO_FATAL_FAILURE(drive(rig, one, position, streams));
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(engine.m_calls.load(), 7);
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
    EXPECT_EQ(state_values(rig.get(), 0).at(1), 6.0F) << "the counter skipped the failed one";
    // The pointers: before_inference ran for every inference, after_inference for the six
    // that succeeded (a failed engine skips it). The failed one read the buffer its
    // predecessor wrote, and its successor read that same buffer again: no flip.
    ASSERT_EQ(log.m_num_in.load(), 7U);
    ASSERT_EQ(log.m_num_out.load(), 6U);
    EXPECT_EQ(log.m_in.at(k_failed).load(), log.m_out.at(k_failed - 1).load());
    EXPECT_EQ(log.m_in.at(k_failed + 1).load(), log.m_in.at(k_failed).load())
        << "the failed inference promoted its state";
    // The successes around it alternate as always: out[j] names the successful inference j's
    // write buffer, which the next inference reads.
    const std::array<size_t, 6> successes{0, 1, 2, 4, 5, 6};
    for (size_t j = 0; j < successes.size(); ++j) {
        const size_t k = successes.at(j);
        EXPECT_NE(log.m_in.at(k).load(), log.m_out.at(j).load()) << "inference " << k;
        if (k + 1 < 7) {
            EXPECT_EQ(log.m_in.at(k + 1).load(), log.m_out.at(j).load()) << "inference " << k;
        }
    }
}

// A reset re-initialises the read side alone, at the first inference of the new stream: the
// State input of that inference is zeros, while the write side still holds what it held (a
// canary written into it at quiescence), since the engine overwrites it anyway.
TEST(AbiState, AResetZeroesTheReadSideOnly) {
    AccumulatorEngine engine;
    const std::array<Registration, 1> registrations{
        {{.m_id = k_accumulator_id, .m_engine = &engine}}};
    RegisteredRig rig(accumulator_model(k_accumulator_id), registrations);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    const size_t latency = anira_handler_get_latency(h, RegisteredRig::processed_slot());
    const std::vector<size_t> sizes = hops(4);
    size_t position = 0;
    Streams first;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));
    ASSERT_EQ(engine.m_calls.load(), 4);
    EXPECT_NE(engine.m_read_first.at(3).load(), 0.0F) << "the fourth inference read a carry";
    // Quiescence (every process_wait returned with its inferences collected): the canary into
    // the write side, white-box.
    auto* port = std::get_if<anira::capi::StatePort>(&h->m_input_ports[0]);
    ASSERT_NE(port, nullptr);
    anira::capi::StateSlot* value = port->m_value.has_value() ? &*port->m_value : nullptr;
    ASSERT_NE(value, nullptr);
    constexpr float k_canary = 12345.0F;
    static_cast<float*>(value->write())[0] = k_canary;
    EXPECT_NE(static_cast<const float*>(value->read())[0], 0.0F);

    anira_handler_reset(h);
    position = 0;
    Streams second;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, second));
    ASSERT_EQ(engine.m_calls.load(), 8);
    EXPECT_EQ(engine.m_read_first.at(4).load(), 0.0F) << "the read side was zeroed";
    EXPECT_EQ(engine.m_write_first.at(4).load(), k_canary) << "the write side was not";
    EXPECT_EQ(second.m_out, first.m_out);
    expect_closed_form(second, latency);
}

// The state is the handler's, not a plan's: two registered accumulator engines on two rows,
// anira_handler_set_plan mid-stream, and the closed form continues across the switch, both
// engines having run their share.
TEST(AbiState, StateSurvivesPlanSwitch) {
    AccumulatorEngine first;
    AccumulatorEngine second;
    const std::array<Registration, 2> registrations{
        {{.m_id = k_accumulator_id, .m_engine = &first},
         {.m_id = k_other_accumulator_id, .m_engine = &second}}};
    ModelConfig model = accumulator_model(k_accumulator_id);
    model.add_model_path(k_other_accumulator_id, "custom-processor");
    RegisteredRig rig(model, registrations);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    EXPECT_EQ(anira_plan_report_num_plans(anira_handler_plan_report(h)), 2U);
    EXPECT_EQ(anira_handler_get_plan(h), 0U);
    const size_t latency = anira_handler_get_latency(h, RegisteredRig::processed_slot());
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, hops(4), position, streams));
    ASSERT_EQ(anira_handler_set_plan(h, 1), ANIRA_OK);
    ASSERT_NO_FATAL_FAILURE(drive(rig, hops(4), position, streams));
    ASSERT_EQ(anira_handler_set_plan(h, 0), ANIRA_OK);
    ASSERT_NO_FATAL_FAILURE(drive(rig, hops(4), position, streams));
    expect_closed_form(streams, latency);
    EXPECT_EQ(first.m_calls.load(), 8);
    EXPECT_EQ(second.m_calls.load(), 4);
    EXPECT_EQ(state_values(h, 0).at(1), 12.0F) << "the counter counts both engines' inferences";
}

// A pair of another dtype than float32 (int32 here) on the registered engine: the two buffers
// take the spec's dtype, the engine reads and writes int32 through the descriptors, and the
// closed form holds on the float32 stream around it.
TEST(AbiState, AnInt32StatePair) {
    AccumulatorEngine engine;
    const std::array<Registration, 1> registrations{
        {{.m_id = k_accumulator_id, .m_engine = &engine}}};
    RegisteredRig rig(accumulator_model(k_accumulator_id, ANIRA_DTYPE_I32),
                      registrations,
                      {},
                      2 * k_hop);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    const auto* port = std::get_if<anira::capi::StatePort>(&h->m_input_ports[0]);
    ASSERT_NE(port, nullptr);
    const anira::capi::StateSlot* value = port->m_value.has_value() ? &*port->m_value : nullptr;
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->dtype(), static_cast<anira_dtype>(ANIRA_DTYPE_I32));
    EXPECT_EQ(value->num_bytes(),
              static_cast<size_t>(k_channels) * k_state_width * sizeof(int32_t));
    const size_t latency = anira_handler_get_latency(h, RegisteredRig::processed_slot());
    const std::vector<size_t> sizes{8, 3, 13, 16, 1, 7, 8, 16, 5, 11, 8, 8, 16, 2, 6};
    size_t position = 0;
    Streams streams;
    ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
    expect_closed_form(streams, latency);
    const int calls = engine.m_calls.load();
    ASSERT_GT(calls, 10);
    const std::vector<int32_t> state = state_values<int32_t>(h, 0);
    ASSERT_EQ(state.size(), static_cast<size_t>(k_channels) * k_state_width);
    for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
        EXPECT_EQ(state[(channel * k_state_width) + 1], calls) << "channel " << channel;
        int32_t sum = 0;
        for (size_t n = 0; n < static_cast<size_t>(calls) * k_hop; ++n) {
            sum += static_cast<int32_t>(input_sample(channel, n));
        }
        EXPECT_EQ(state[channel * k_state_width], sum) << "channel " << channel;
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// ---- the bundled model on the engines
// -------------------------------------------------------------

#if defined(USE_ONNXRUNTIME) || defined(USE_LIBTORCH) || defined(USE_EXECUTORCH) || \
    defined(USE_TFLITE) || defined(USE_LITERT)

// stateful_accumulator.model.json: the stereo export of example-models' StatefulAccumulatorNetwork
// (the model of the twin above, a [1, 2, 2] state around a [1, 2, 64] stream, the state first
// in the inputs and last in the outputs) with a row for every engine. The TFLite export orders
// its outputs [state_out, processed_data] in the file (the README of example-models); its
// signature keeps the declared mapping under the keys output_0 / output_1, which the tflite row
// reaches by position (the signature runner lists them in key order) and the litert row by
// name (LiteRT lists them in the file's order, so the row carries a tensors record). The
// closed form below holds on every engine because the binding is checked against the shapes
// at prepare: a swapped pair would be refused, not captured as the state.

constexpr uint32_t k_file_hop = 64;      // the file's window (min = max = 64, no overlap)
constexpr double k_file_rate = 48000.0;  // the host's rate; a contract file carries none

/// The engines of this build the file names, in the order of anira_enabled_backends.
std::vector<anira_engine> file_engines() {
    std::vector<anira_engine> out;
    for (const anira::BackendId& id : anira::enabled_backends()) {
        const auto engine = static_cast<anira_engine>(id.engine);
        if (engine == ANIRA_ENGINE_LIBTORCH || engine == ANIRA_ENGINE_ONNXRUNTIME ||
            engine == ANIRA_ENGINE_EXECUTORCH || engine == ANIRA_ENGINE_TFLITE ||
            engine == ANIRA_ENGINE_LITERT) {
            out.push_back(engine);
        }
    }
    return out;
}

const char* engine_word(anira_engine engine) {
    switch (engine) {
        case ANIRA_ENGINE_LIBTORCH: return "libtorch";
        case ANIRA_ENGINE_ONNXRUNTIME: return "onnxruntime";
        case ANIRA_ENGINE_EXECUTORCH: return "executorch";
        case ANIRA_ENGINE_TFLITE: return "tflite";
        case ANIRA_ENGINE_LITERT: return "litert";
        default: return "another engine";
    }
}

/// The engine of the plan the handler runs: a case tells that it ran on the engine it asked for.
uint32_t plan_engine(const anira_handler* handler) {
    return handler->m_report.m_plans.at(anira_handler_get_plan(handler)).engine;
}

/// The bundled file on one engine of this build: the handler over the model file with that one
/// candidate, prepared with the contract file at the host geometry (blocks of 1 to `block_max`
/// samples at 48 kHz), the two streams driven through the waiting two-slot single form (`data`
/// is input slot 1 and `processed_data` output slot 0: the file's order).
class EngineRig {
public:
    explicit EngineRig(anira_engine engine,
                       uint32_t block_max = k_file_hop,
                       std::vector<anira_stage_desc> stages = {})
        : m_model(ModelConfig::from_file(k_stateful_accumulator_model_json))
        , m_candidates{anira_backend_id{.struct_size = sizeof(anira_backend_id),
                                        .engine = static_cast<uint32_t>(engine),
                                        .provider = ANIRA_PROVIDER_DEFAULT,
                                        .engine_id = nullptr}}
        , m_stages(std::move(stages))
        , m_handler(m_context, m_model, m_candidates, m_stages)
        , m_block_max(block_max) {
        prepare();
    }
    ~EngineRig() = default;
    EngineRig(const EngineRig&) = delete;
    EngineRig& operator=(const EngineRig&) = delete;
    EngineRig(EngineRig&&) = delete;
    EngineRig& operator=(EngineRig&&) = delete;

    void prepare() {
        anira::ContractHandle contract =
            anira::ContractHandle::from_file(k_stateful_accumulator_contract_json);
        contract.hard_geometry(1, m_block_max, k_file_rate);
        ASSERT_EQ(m_handler.prepare(contract), ANIRA_OK) << m_handler.m_err.message;
        m_session = anira_test::session_of(m_handler.m_handler);
        ASSERT_NE(m_session, nullptr);
    }

    bool ready() const { return m_session != nullptr; }
    anira_handler* get() const { return m_handler.m_handler; }
    anira::SessionElement& session() { return *m_session; }
    static constexpr uint32_t data_slot() { return 1; }
    static constexpr uint32_t processed_slot() { return 0; }
    static constexpr uint32_t state_slot() { return 0; }  ///< the State input, first in the file

    /// One host block of `samples` per channel, each stream at its own slot; the call returns
    /// with its own inferences collected.
    anira_status block(std::span<const float> in, std::span<float> out, size_t samples) {
        const anira_tensor in_tensor =
            whole_f32(in.data(), {k_channels, static_cast<int64_t>(samples)});
        const anira_tensor out_tensor =
            whole_f32(out.data(), {k_channels, static_cast<int64_t>(samples)});
        return anira_handler_process_wait(get(),
                                          &in_tensor,
                                          data_slot(),
                                          &out_tensor,
                                          processed_slot(),
                                          nullptr,
                                          ANIRA_WAIT_FOREVER);
    }

    /// Waits until every hop the stream holds has run: the output ring is back at the latency
    /// (what it held right after prepare), so the state is what the last inference left.
    void settle() {
        anira_test::wait_for_available(get(),
                                       anira_handler_get_latency(get(), processed_slot()),
                                       processed_slot());
    }

private:
    Context m_context{4};
    ModelConfig m_model;
    std::array<anira_backend_id, 1> m_candidates;
    std::vector<anira_stage_desc> m_stages;
    anira_test::Handler m_handler;
    uint32_t m_block_max;
    std::shared_ptr<anira::SessionElement> m_session;
};

// The bundled model on every engine of this build the file names, against the closed form of
// the engine-free twin: host blocks above and below the hop, the state fed back by anira
// across the engine's inferences, and the port's value left with the sums and the count.
TEST(AbiState, TheBundledModelFollowsTheClosedFormOnEveryEngine) {
    for (const anira_engine engine : file_engines()) {
        SCOPED_TRACE(engine_word(engine));
        EngineRig rig(engine, 2 * k_file_hop);
        ASSERT_TRUE(rig.ready());
        const anira_handler* h = rig.get();
        ASSERT_EQ(plan_engine(h), static_cast<uint32_t>(engine));
        const size_t latency = anira_handler_get_latency(h, EngineRig::processed_slot());
        // Rounds of 704 samples (eleven hops, in blocks of 8 to 128 samples), enough of them to
        // run eight hops past the latency (ten hops at this geometry).
        constexpr std::array<size_t, 12> k_round{64, 32, 96, 64, 16, 48, 128, 64, 8, 56, 64, 64};
        std::vector<size_t> sizes;
        for (size_t total = 0; total < latency + (size_t{8} * k_file_hop);) {
            for (const size_t samples : k_round) {
                sizes.push_back(samples);
                total += samples;
            }
        }
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, latency, 0, k_file_hop);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

        // White-box: the port of the State input holds what the engine's last inference left,
        // the sum of every sample of the stream and the number of inferences.
        ASSERT_NO_FATAL_FAILURE(rig.settle());
        const size_t inferences = position / k_file_hop;
        const std::vector<float> state = state_values(h, EngineRig::state_slot());
        ASSERT_EQ(state.size(), static_cast<size_t>(k_channels) * k_state_width);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            float sum = 0.0F;
            for (size_t n = 0; n < inferences * k_file_hop; ++n) {
                sum += input_sample(channel, n);
            }
            EXPECT_EQ(state[channel * k_state_width], sum) << "channel " << channel;
            EXPECT_EQ(state[(channel * k_state_width) + 1], static_cast<float>(inferences))
                << "channel " << channel;
        }
    }
}

// anira_handler_reset on the engine: the state starts over at zeros, and the same input yields
// the first run's samples bit for bit.
TEST(AbiState, TheBundledModelResetsToZerosOnEveryEngine) {
    for (const anira_engine engine : file_engines()) {
        SCOPED_TRACE(engine_word(engine));
        EngineRig rig(engine);
        ASSERT_TRUE(rig.ready());
        anira_handler* h = rig.get();
        const size_t latency = anira_handler_get_latency(h, EngineRig::processed_slot());
        // Enough hops to see four of them behind the latency's zeros (six at this geometry).
        const std::vector<size_t> sizes((latency / k_file_hop) + 4, k_file_hop);
        size_t position = 0;
        Streams first;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, first));
        ASSERT_NO_FATAL_FAILURE(rig.settle());
        EXPECT_NE(state_values(h, EngineRig::state_slot()).at(0), 0.0F);

        anira_handler_reset(h);
        position = 0;
        Streams second;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, second));
        EXPECT_EQ(second.m_out, first.m_out);
        expect_closed_form(second, latency, 0, k_file_hop);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    }
}

// The contract file asks for one warm-up inference. At this point of the runtime it runs
// inside the engine's processor when the processor loads, on the processor's own buffers,
// ahead of anira's feed and capture, so it reaches neither the state nor the stream:
// after prepare the state is zeros, the first hop is the identity, and after n hops the
// counter reads n, not n + 1. The case stands so that a warm-up that runs through the session
// keeps the property.
TEST(AbiState, TheWarmupDoesNotLeakIntoTheStateOnAnyEngine) {
    for (const anira_engine engine : file_engines()) {
        SCOPED_TRACE(engine_word(engine));
        EngineRig rig(engine);
        ASSERT_TRUE(rig.ready());
        const anira_handler* h = rig.get();
        EXPECT_EQ(h->m_inference_config.m_warm_up, 1U) << "the contract file's fixed warm-up";
        const std::vector<float> fresh = state_values(h, EngineRig::state_slot());
        ASSERT_EQ(fresh.size(), static_cast<size_t>(k_channels) * k_state_width);
        for (const float value : fresh) { EXPECT_EQ(value, 0.0F); }

        const size_t latency = anira_handler_get_latency(h, EngineRig::processed_slot());
        // Enough hops to see the first one behind the latency's zeros.
        const std::vector<size_t> sizes((latency / k_file_hop) + 3, k_file_hop);
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, latency, 0, k_file_hop);
        ASSERT_NO_FATAL_FAILURE(rig.settle());
        const std::vector<float> state = state_values(h, EngineRig::state_slot());
        ASSERT_EQ(state.size(), static_cast<size_t>(k_channels) * k_state_width);
        for (size_t channel = 0; channel < static_cast<size_t>(k_channels); ++channel) {
            EXPECT_EQ(state[(channel * k_state_width) + 1], static_cast<float>(sizes.size()))
                << "channel " << channel;
        }
    }
}

// The ping-pong on every engine of this build the file names: through a stage's hooks the
// descriptors a built-in adapter is handed alternate between the pair's two buffers exactly as
// on the registered engine, whether the adapter binds them in place (ONNX Runtime both ways,
// LibTorch's input) or copies them (LibTorch's output, ExecuTorch, TFLite, LiteRT); the closed
// form holds.
TEST(AbiState, TheStateAlternatesBetweenTwoBuffersOnEveryEngine) {
    for (const anira_engine engine : file_engines()) {
        SCOPED_TRACE(engine_word(engine));
        PointerLog log;
        EngineRig rig(engine, k_file_hop, {logging_stage(log)});
        ASSERT_TRUE(rig.ready());
        const anira_handler* h = rig.get();
        ASSERT_EQ(plan_engine(h), static_cast<uint32_t>(engine));
        const size_t latency = anira_handler_get_latency(h, EngineRig::processed_slot());
        // Enough hops to see six of them behind the latency's zeros.
        const std::vector<size_t> sizes((latency / k_file_hop) + 6, k_file_hop);
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, latency, 0, k_file_hop);
        ASSERT_NO_FATAL_FAILURE(rig.settle());
        ASSERT_NO_FATAL_FAILURE(expect_two_alternating_buffers(log, sizes.size()));
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    }
}

#ifdef USE_ONNXRUNTIME

/// The bundled accumulator's ONNX export.
std::string accumulator_onnx_path() {
    const ModelConfig file = ModelConfig::from_file(k_stateful_accumulator_model_json);
    for (const anira::capi::ModelEntry& row : file.native()->m_models) {
        if (row.m_engine == ANIRA_ENGINE_ONNXRUNTIME) { return row.m_path; }
    }
    return {};
}

/// The bundled accumulator declared under other canonical names and in another order than
/// the export's (the stream first on both sides), with or without the tensors record that
/// names the export's tensors.
ModelConfig renamed_accumulator(bool with_names) {
    ModelConfig model;
    const uint32_t row = model.add_model_path(ANIRA_ENGINE_ONNXRUNTIME, accumulator_onnx_path());
    TensorSpec audio("audio", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    audio.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_file_hop));
    audio.window(k_file_hop, k_file_hop, 0);
    TensorSpec audio_out("audio_out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    audio_out.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, k_channels)
        .axis(2, ANIRA_AXIS_TIME, static_cast<int64_t>(k_file_hop));
    audio_out.window(k_file_hop, k_file_hop, 0);
    model.input(audio);
    model.input(state("carry", "carry_out"));
    model.output(audio_out);
    model.output(state("carry_out"));
    if (with_names) {
        model.tensor_name(row, "audio", "data");
        model.tensor_name(row, "carry", "state_in");
        model.tensor_name(row, "audio_out", "processed_data");
        model.tensor_name(row, "carry_out", "state_out");
    }
    return model;
}

/// The renamed accumulator on ONNX Runtime: the streams at slot 0 on both sides.
class RenamedRig {
public:
    explicit RenamedRig(bool with_names)
        : m_model(renamed_accumulator(with_names)), m_handler(m_context, m_model, m_candidates) {}

    anira_status prepare() {
        anira::ContractHandle contract =
            anira::ContractHandle::from_file(k_stateful_accumulator_contract_json);
        contract.hard_geometry(1, k_file_hop, k_file_rate);
        return m_handler.prepare(contract);
    }
    anira_handler* get() const { return m_handler.m_handler; }
    const char* message() const { return m_handler.m_err.message; }

    anira_status block(std::span<const float> in, std::span<float> out, size_t samples) {
        const anira_tensor in_tensor =
            whole_f32(in.data(), {k_channels, static_cast<int64_t>(samples)});
        const anira_tensor out_tensor =
            whole_f32(out.data(), {k_channels, static_cast<int64_t>(samples)});
        return anira_handler_process_wait(get(),
                                          &in_tensor,
                                          0,
                                          &out_tensor,
                                          0,
                                          nullptr,
                                          ANIRA_WAIT_FOREVER);
    }

private:
    Context m_context{4};
    ModelConfig m_model;
    std::array<anira_backend_id, 1> m_candidates{
        anira_backend_id{.struct_size = sizeof(anira_backend_id),
                         .engine = ANIRA_ENGINE_ONNXRUNTIME,
                         .provider = ANIRA_PROVIDER_DEFAULT,
                         .engine_id = nullptr}};
    anira_test::Handler m_handler;
};

// A State model bound by name: the bundled accumulator declared in another order and under
// other canonical names, its tensors record naming the export's tensors, runs the closed form
// on ONNX Runtime (the state pair bound to the graph's state_in and state_out by name); the
// same declaration without the record binds by position, which the shape check refuses at
// prepare naming the slot, in place of running the stream into the state.
TEST(AbiState, AStateModelBoundByNameOnOnnxRuntime) {
    {
        RenamedRig rig(true);
        ASSERT_EQ(rig.prepare(), ANIRA_OK) << rig.message();
        const anira_handler* h = rig.get();
        ASSERT_EQ(plan_engine(h), static_cast<uint32_t>(ANIRA_ENGINE_ONNXRUNTIME));
        const size_t latency = anira_handler_get_latency(h, 0);
        const std::vector<size_t> sizes((latency / k_file_hop) + 6, k_file_hop);
        size_t position = 0;
        Streams streams;
        ASSERT_NO_FATAL_FAILURE(drive(rig, sizes, position, streams));
        expect_closed_form(streams, latency, 0, k_file_hop);
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        const std::vector<anira_plan_slot> rows = [h] {
            uint32_t count = 0;
            const anira_plan_report* report = anira_handler_plan_report(h);
            EXPECT_EQ(
                anira_plan_report_slots(report, 0, 1, sizeof(anira_plan_slot), &count, nullptr),
                ANIRA_OK);
            std::vector<anira_plan_slot> slots(count, ANIRA_PLAN_SLOT_INIT);
            EXPECT_EQ(anira_plan_report_slots(report,
                                              0,
                                              1,
                                              sizeof(anira_plan_slot),
                                              &count,
                                              slots.data()),
                      ANIRA_OK);
            return slots;
        }();
        ASSERT_EQ(rows.size(), 2U);
        EXPECT_EQ(rows[0].binding, static_cast<uint32_t>(ANIRA_BINDING_NAME));
        EXPECT_EQ(rows[1].binding, static_cast<uint32_t>(ANIRA_BINDING_NAME));
    }
    {
        RenamedRig rig(false);
        EXPECT_EQ(rig.prepare(), ANIRA_ERROR_CONFIG) << rig.message();
        const std::string message = rig.message();
        EXPECT_NE(message.find("'audio'"), std::string::npos) << message;
        EXPECT_NE(message.find("state_in"), std::string::npos) << message;
    }
}

#endif  // USE_ONNXRUNTIME

#endif  // an engine the file names

}  // namespace
