// anira/abi/stage.h: the ring accessors, the stage context and its six accessors, the descriptor
// and its carrier, the default bodies and the processor a C-created handler runs them in.
//
// A phase callback runs inside an ANIRA_NONBLOCKING entry (pre_process, post_process) or on an
// inference thread (before_inference, after_inference), so every callback here is declared
// ANIRA_NONBLOCKING, holds no gtest assertion (a failing EXPECT allocates, and under
// RealtimeSanitizer that aborts the process in place of failing the test), allocates nothing
// and writes only into the state its test owns. Every EXPECT runs after the calls. The streams
// are driven in lockstep (one block, then the wait for its inference), so every case is
// deterministic whatever the pool does.
#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/build_info.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/stage.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/abi/version.h>
#include <anira/backends/BackendBase.h>
#include <anira/compat/v3_to_v2.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/Buffer.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>
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
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/log_record_collector.h"
#include "capi/port.h"
#include "capi/stage.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::Context;
using anira_test::explicit_contract;
using anira_test::Handler;
using anira_test::k_custom;
using anira_test::RecordCollector;

constexpr size_t k_hop = anira_test::k_block;

// ---- models ------------------------------------------------------------------------------------

/// A mono stream through the engine-free custom row: [1, 1, window] in, of which `overlap` are
/// history, and one hop out. Without an overlap BackendBase::process is an exact pass-through.
ModelConfig stream_model(int64_t window = static_cast<int64_t>(k_hop), int64_t overlap = 0) {
    ModelConfig model;
    model.add_model_path(k_custom, "custom-processor");
    TensorSpec in("in", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    in.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, window);
    in.window(window, window, overlap);
    model.input(in);
    const int64_t hop = window - overlap;
    TensorSpec out("out", ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    out.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 1).axis(2, ANIRA_AXIS_TIME, hop);
    out.window(hop, hop, 0);
    model.output(out);
    return model;
}

/// A contract whose miss policy needs nothing of the rings' dtypes.
anira::ContractHandle zeros_contract(uint32_t block = static_cast<uint32_t>(k_hop)) {
    return explicit_contract(block, anira_test::k_rate, ANIRA_MISS_ZEROS);
}

// ---- a pipeline with stages --------------------------------------------------------------------

/// anira_test::Handler's twin with a stage: the descriptors (at most one) go in before the
/// handler is created, and the pipeline can be destroyed ahead of the handler.
struct StagedHandler {
    StagedHandler(const Context& context,
                  const ModelConfig& model,
                  const std::vector<anira_stage_desc>& stages,
                  std::span<const anira_backend_id> candidates = {}) {
        EXPECT_EQ(anira_pipeline_create(&m_pipeline, &m_err), ANIRA_OK) << m_err.message;
        const std::array<const anira_model_config*, 1> variants{model.native()};
        EXPECT_EQ(anira_pipeline_add_inference(m_pipeline,
                                               variants.data(),
                                               1,
                                               candidates.empty() ? nullptr : candidates.data(),
                                               static_cast<uint32_t>(candidates.size()),
                                               &m_err),
                  ANIRA_OK)
            << m_err.message;
        for (const anira_stage_desc& stage : stages) {
            EXPECT_EQ(anira_pipeline_add_stage(m_pipeline, &stage, &m_err), ANIRA_OK)
                << m_err.message;
        }
        m_create_status = anira_handler_create(context.m_context, m_pipeline, &m_handler, &m_err);
    }
    ~StagedHandler() { destroy(); }
    StagedHandler(const StagedHandler&) = delete;
    StagedHandler& operator=(const StagedHandler&) = delete;
    StagedHandler(StagedHandler&&) = delete;
    StagedHandler& operator=(StagedHandler&&) = delete;

    anira_status prepare(const anira::ContractHandle& contract) {
        m_err = ANIRA_ERROR_INIT;
        return anira_handler_prepare(m_handler, contract.native(), &m_err);
    }

    void destroy() {
        anira_handler_destroy(m_handler);
        m_handler = nullptr;
        anira_pipeline_destroy(m_pipeline);
        m_pipeline = nullptr;
    }

    anira_handler* m_handler = nullptr;
    anira_pipeline* m_pipeline = nullptr;
    anira_status m_create_status = ANIRA_OK;
    anira_error m_err = ANIRA_ERROR_INIT;
};

/// A descriptor with the user_data and both real-time bits: every phase body of this file
/// allocates nothing and blocks on nothing (the file comment), so the descriptors promise it,
/// and a Hard contract takes a filled pre_process or post_process only with the promise. A
/// case about the promise itself clears the bits.
anira_stage_desc promising_stage(void* user_data) {
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.user_data = user_data;
    stage.flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST | ANIRA_STAGE_FLAG_REALTIME_HOOKS;
    return stage;
}

// ---- a stream in lockstep ----------------------------------------------------------------------

/// Drives a prepared mono handler one block at a time, in place, and waits for the block's
/// inference before the next one: m_in and m_out are the two streams.
struct Stream {
    explicit Stream(anira_handler* handler) : m_handler(handler), m_face(handler) {}

    /// One block of the ramp; `settle` is the change the block leaves in the output ring (0
    /// for an aligned stream, the hop when a stage pushed one hop too many).
    void block(size_t settle = 0) {
        std::vector<float> data = anira_test::ramp(m_next++, k_hop);
        m_in.insert(m_in.end(), data.begin(), data.end());
        const size_t prev = anira_test::available(m_handler);
        const std::array<float*, 1> channels{data.data()};
        size_t delivered = 0;
        m_status = m_face.process_inplace(channels.data(), k_hop, 0, &delivered);
        anira_test::wait_for_available(m_handler, prev + settle);
        m_out.insert(m_out.end(), data.begin(), data.end());
    }

    void blocks(size_t count) {
        for (size_t i = 0; i < count; ++i) {
            block();
            if (::testing::Test::HasFatalFailure()) { return; }
        }
    }

    /// What the output stream holds when chunk `failed` (counted from 0; none when negative)
    /// delivered zeros and every other chunk passed through: the latency, then the input.
    std::vector<float> expected(int failed = -1) const {
        const size_t latency = anira_handler_get_latency(m_handler, 0);
        std::vector<float> out(m_in.size(), 0.0F);
        for (size_t n = latency; n < out.size(); ++n) {
            const size_t source = n - latency;
            const bool zeroed = failed >= 0 && source / k_hop == static_cast<size_t>(failed);
            out[n] = zeroed ? 0.0F : m_in[source];
        }
        return out;
    }

    anira_handler* m_handler;
    anira_test::FloatFace m_face;
    std::vector<float> m_in;
    std::vector<float> m_out;
    anira_status m_status = ANIRA_OK;
    size_t m_next = 1;
};

void expect_same_stream(const std::vector<float>& actual, const std::vector<float>& wanted) {
    ASSERT_EQ(actual.size(), wanted.size());
    for (size_t n = 0; n < actual.size(); ++n) {
        ASSERT_EQ(actual[n], wanted[n]) << "sample " << n << " (chunk " << n / k_hop << ")";
    }
}

// ---- the recording stage -----------------------------------------------------------------------

constexpr size_t k_phases = 5;   // indexed by anira_stage_phase up to AFTER_INFERENCE (4)
constexpr size_t k_tensors = 4;  // the tensors per side the record has room for

/// "Not asked": what a question a probe did not ask leaves in its status slot.
constexpr anira_status k_not_asked = ANIRA_STATUS_FORCE32;

/// What one stage saw and what it is told to do. The inference-thread phases write their own
/// entries only, and the test reads after the block's inference was collected.
struct Probe {
    std::array<std::atomic<int>, k_phases> m_calls{};
    std::array<anira_stage_ctx, k_phases> m_ctx{};
    // What the six context accessors answered, per phase and slot. The probe asks like a
    // correct stage: the role of every slot in every phase; the input ring in pre_process and
    // the output ring in post_process, of a Streamed slot only; the model end of a side in the
    // phases that expose it, and not of a State slot in the two host-end phases. A question
    // not asked leaves k_not_asked in its status and the zero in its answer, so the record is
    // the table of what a phase exposes and what a stage may ask.
    std::array<std::array<anira_status, k_tensors>, k_phases> m_input_role_status{};
    std::array<std::array<anira_status, k_tensors>, k_phases> m_output_role_status{};
    std::array<std::array<anira_role, k_tensors>, k_phases> m_input_roles{};
    std::array<std::array<anira_role, k_tensors>, k_phases> m_output_roles{};
    std::array<std::array<anira_status, k_tensors>, k_phases> m_input_ring_status{};
    std::array<std::array<anira_status, k_tensors>, k_phases> m_output_ring_status{};
    std::array<std::array<anira_ring*, k_tensors>, k_phases> m_input_rings{};
    std::array<std::array<anira_ring*, k_tensors>, k_phases> m_output_rings{};
    std::array<std::array<anira_status, k_tensors>, k_phases> m_input_status{};
    std::array<std::array<anira_status, k_tensors>, k_phases> m_output_status{};
    std::array<std::array<anira_tensor, k_tensors>, k_phases> m_inputs{};
    std::array<std::array<anira_tensor, k_tensors>, k_phases> m_outputs{};
    std::array<std::thread::id, k_phases> m_thread{};
    float m_static_input = -1.0F;  ///< element 0 of model input 1 as pre_process found it
    // What the stage does in each phase: nothing, the default body (`m_defaults` times), and
    // the status it returns on its `m_fail_call`-th call of `m_fail_phase` (counted from 0).
    int m_defaults = 0;
    uint32_t m_fail_phase = ANIRA_PHASE_RELEASE;
    int m_fail_call = -1;
    anira_status m_fail_status = ANIRA_ERROR_ENGINE;
    std::atomic<int>* m_order = nullptr;  ///< a counter shared by several probes
    std::array<int, k_phases> m_seen_order{};
    float m_scale = 1.0F;  ///< post_process multiplies model output 0 by it before the default
};

anira_status ANIRA_CALL probe_phase(const anira_stage_ctx* ctx,
                                    void* /*prepared*/,
                                    void* user_data) ANIRA_NONBLOCKING {
    auto* probe = static_cast<Probe*>(user_data);
    const uint32_t phase = ctx->phase;
    if (phase >= k_phases) { return ANIRA_ERROR_INTERNAL; }
    const int call = probe->m_calls.at(phase).fetch_add(1);
    probe->m_ctx.at(phase) = *ctx;
    probe->m_thread.at(phase) = std::this_thread::get_id();
    if (probe->m_order != nullptr) { probe->m_seen_order.at(phase) = probe->m_order->fetch_add(1); }
    const bool pre = phase == ANIRA_PHASE_PRE_PROCESS;
    const bool post = phase == ANIRA_PHASE_POST_PROCESS;
    for (uint32_t slot = 0; slot < ctx->num_inputs && slot < k_tensors; ++slot) {
        anira_role& role = probe->m_input_roles.at(phase).at(slot);
        probe->m_input_role_status.at(phase).at(slot) = anira_stage_input_role(ctx, slot, &role);
        probe->m_input_ring_status.at(phase).at(slot) =
            pre && role == ANIRA_ROLE_STREAMED
                ? anira_stage_input_ring(ctx, slot, &probe->m_input_rings.at(phase).at(slot))
                : k_not_asked;
        const bool exposed =
            (pre && role != ANIRA_ROLE_STATE) || phase == ANIRA_PHASE_BEFORE_INFERENCE;
        probe->m_input_status.at(phase).at(slot) =
            exposed ? anira_stage_input_tensor(ctx, slot, &probe->m_inputs.at(phase).at(slot))
                    : k_not_asked;
    }
    for (uint32_t slot = 0; slot < ctx->num_outputs && slot < k_tensors; ++slot) {
        anira_role& role = probe->m_output_roles.at(phase).at(slot);
        probe->m_output_role_status.at(phase).at(slot) = anira_stage_output_role(ctx, slot, &role);
        probe->m_output_ring_status.at(phase).at(slot) =
            post && role == ANIRA_ROLE_STREAMED
                ? anira_stage_output_ring(ctx, slot, &probe->m_output_rings.at(phase).at(slot))
                : k_not_asked;
        const bool exposed =
            (post && role != ANIRA_ROLE_STATE) || phase == ANIRA_PHASE_AFTER_INFERENCE;
        probe->m_output_status.at(phase).at(slot) =
            exposed ? anira_stage_output_tensor(ctx, slot, &probe->m_outputs.at(phase).at(slot))
                    : k_not_asked;
    }
    if (phase == ANIRA_PHASE_PRE_PROCESS && ctx->num_inputs > 1) {
        const float* gain = anira_tensor_data_f32(&probe->m_inputs.at(phase).at(1));
        if (gain != nullptr) { probe->m_static_input = gain[0]; }
    }
    if (phase == probe->m_fail_phase && call == probe->m_fail_call) { return probe->m_fail_status; }
    if (phase == ANIRA_PHASE_POST_PROCESS && probe->m_scale != 1.0F) {
        const anira_tensor& tensor = probe->m_outputs.at(phase).at(0);
        float* out = anira_tensor_data_f32(&tensor);
        const size_t count = anira_tensor_num_elements(&tensor);
        for (size_t n = 0; out != nullptr && n < count; ++n) { out[n] *= probe->m_scale; }
    }
    anira_status status = ANIRA_OK;
    for (int i = 0; i < probe->m_defaults && status == ANIRA_OK; ++i) {
        if (phase == ANIRA_PHASE_PRE_PROCESS) { status = anira_stage_default_pre_process(ctx); }
        if (phase == ANIRA_PHASE_POST_PROCESS) { status = anira_stage_default_post_process(ctx); }
    }
    return status;
}

// ---- prepare and release -----------------------------------------------------------------------

/// What the stage's prepare saw of its record and what it does: the counts, the first
/// template and name of either side (copied: the arrays die with the call), the status it
/// returns and the pointer it hands back.
struct Lifetime {
    int m_prepared = 0;
    int m_released = 0;
    anira_handler* m_handler = nullptr;
    const anira_plan_report* m_report = nullptr;
    uint32_t m_plans = 0;
    unsigned int m_latency = 0;
    uint32_t m_struct_size = 0;
    uint32_t m_num_entries = 0;
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    anira_tensor m_first_input{};
    anira_tensor m_first_output{};
    std::string m_first_input_name;
    std::string m_first_output_name;
    anira_status m_return = ANIRA_OK;
    void* m_hand_back = nullptr;  ///< what prepare writes into out_prepared
};

anira_status ANIRA_CALL on_prepare(const anira_stage_prepare_info* info,
                                   void* user_data,
                                   void** out_prepared) {
    auto* lifetime = static_cast<Lifetime*>(user_data);
    ++lifetime->m_prepared;
    lifetime->m_handler = info->handler;
    lifetime->m_report = info->report;
    lifetime->m_plans = anira_plan_report_num_plans(info->report);
    // A getter: the handler answers while its stage prepares.
    lifetime->m_latency = anira_handler_get_latency(info->handler, 0);
    lifetime->m_struct_size = info->struct_size;
    lifetime->m_num_entries = info->num_entries;
    lifetime->m_num_inputs = info->num_inputs;
    lifetime->m_num_outputs = info->num_outputs;
    if (info->num_inputs > 0 && info->inputs != nullptr && info->input_names != nullptr) {
        lifetime->m_first_input = info->inputs[0];
        lifetime->m_first_input_name = info->input_names[0];
    }
    if (info->num_outputs > 0 && info->outputs != nullptr && info->output_names != nullptr) {
        lifetime->m_first_output = info->outputs[0];
        lifetime->m_first_output_name = info->output_names[0];
    }
    *out_prepared = lifetime->m_hand_back;
    return lifetime->m_return;
}

void ANIRA_CALL on_release(void* user_data) {
    ++static_cast<Lifetime*>(user_data)->m_released;
}

// ---- a backend whose output needs the window's history -----------------------------------------

/// out[n] = in[n] + in[n + overlap] over a [1, 1, 2 * hop] window: wrong in every sample when
/// the head of the window is not the ring's history.
class WindowSumBackend : public anira::BackendBase {
public:
    explicit WindowSumBackend(anira::InferenceConfig& config) : anira::BackendBase(config) {}

    void process(std::vector<anira::BufferF>& input,
                 std::vector<anira::BufferF>& output,
                 [[maybe_unused]] std::shared_ptr<anira::SessionElement> session) override {
        const size_t hop = output[0].get_num_samples();
        const float* in = input[0].get_read_pointer(0);
        float* out = output[0].get_write_pointer(0);
        for (size_t n = 0; n < hop; ++n) { out[n] = in[n] + in[n + hop]; }
    }
};

// ---- a converting stage ------------------------------------------------------------------------

/// pre_process of a stage over an int16 input ring, written with the context accessors only: it
/// asks what slot 0 is, takes its ring and its model end, pops one hop of int16 into its scratch
/// (user_data) and writes float32 into the model input. Nothing in anira converts.
anira_status ANIRA_CALL int16_to_float(const anira_stage_ctx* ctx,
                                       void* /*prepared*/,
                                       void* user_data) ANIRA_NONBLOCKING {
    auto* scratch = static_cast<std::array<int16_t, k_hop>*>(user_data);
    anira_role role = ANIRA_ROLE_FORCE32;
    const anira_status asked = anira_stage_input_role(ctx, 0, &role);
    if (asked != ANIRA_OK) { return asked; }
    if (role != ANIRA_ROLE_STREAMED) { return ANIRA_ERROR_CONFIG; }
    anira_ring* ring = nullptr;
    const anira_status has_ring = anira_stage_input_ring(ctx, 0, &ring);
    if (has_ring != ANIRA_OK) { return has_ring; }
    anira_tensor tensor;
    const anira_status exposed = anira_stage_input_tensor(ctx, 0, &tensor);
    if (exposed != ANIRA_OK) { return exposed; }
    float* samples = anira_tensor_data_f32(&tensor);
    if (samples == nullptr || anira_ring_dtype(ring) != ANIRA_DTYPE_I16) {
        return ANIRA_ERROR_CONFIG;
    }
    if (anira_ring_pop_block(ring, 0, scratch->data(), ANIRA_DTYPE_I16, k_hop) != k_hop) {
        return ANIRA_ERROR_INTERNAL;
    }
    for (size_t n = 0; n < k_hop; ++n) {
        samples[n] = static_cast<float>(scratch->at(n)) / 32768.0F;
    }
    return ANIRA_OK;
}

// ---- a context without a handler ---------------------------------------------------------------

/// Whether every byte of the record is zero: what a refused tensor accessor leaves, like a
/// refused anira_tensor_init_* factory. The record holds unions, so tidy refuses a plain memory
/// compare over it (bugprone-suspicious-memory-comparison); an array of unsigned char compares
/// byte for byte.
bool all_zero(const anira_tensor& tensor) {
    std::array<unsigned char, sizeof(anira_tensor)> bytes{};
    std::memcpy(bytes.data(), &tensor, sizeof(anira_tensor));
    return bytes == std::array<unsigned char, sizeof(anira_tensor)>{};
}

/// A ctx over a hand-filled StageFrame (src/capi/stage.h): one Streamed float32 tensor of
/// `shape` at slot 0 of each side, both over `ring`, and with `with_state` a State tensor of
/// the same shape at slot 1 of each side; the memory of the model ends owned here. What the
/// processor fills per call, for the default bodies and the context accessors without a handler.
struct HandBuiltCtx {
    HandBuiltCtx(uint32_t phase,
                 anira::RingBuffer& ring,
                 const std::vector<int64_t>& shape,
                 bool with_state = false)
        : m_ports(with_state ? 2 : 1) {
        size_t elements = 1;
        for (const int64_t extent : shape) { elements *= static_cast<size_t>(extent); }
        for (size_t slot = 0; slot < m_ports.size(); ++slot) {
            m_buffers.emplace_back(1, elements);
            m_buffers[slot].clear();
        }
        // The chunk's descriptors as the session builds them: the spec's shape, float32, over
        // the buffer.
        m_tensors.resize(m_ports.size());
        for (size_t slot = 0; slot < m_ports.size(); ++slot) {
            anira_tensor_init_host(&m_tensors[slot],
                                   m_buffers[slot].data(),
                                   ANIRA_DTYPE_F32,
                                   static_cast<uint32_t>(shape.size()),
                                   shape.data());
        }
        std::get<anira::capi::StreamPort>(m_ports[0]).m_ring = &ring;
        if (with_state) { m_ports[1].emplace<anira::capi::StatePort>(); }
        m_frame.m_input_ports = &m_ports;
        m_frame.m_output_ports = &m_ports;
        expose(phase);
        m_ctx.num_inputs = static_cast<uint32_t>(m_ports.size());
        m_ctx.num_outputs = static_cast<uint32_t>(m_ports.size());
        m_ctx.frame = &m_frame;
    }
    HandBuiltCtx(const HandBuiltCtx&) = delete;  // the ctx names this object's frame
    HandBuiltCtx& operator=(const HandBuiltCtx&) = delete;
    HandBuiltCtx(HandBuiltCtx&&) = delete;
    HandBuiltCtx& operator=(HandBuiltCtx&&) = delete;
    ~HandBuiltCtx() = default;

    /// The phase, and the descriptors of the side that phase exposes, as the processor's entry
    /// points fill them.
    void expose(uint32_t phase) {
        const bool inputs =
            phase == ANIRA_PHASE_PRE_PROCESS || phase == ANIRA_PHASE_BEFORE_INFERENCE;
        m_ctx.phase = phase;
        m_frame.m_input_tensors = inputs ? &m_tensors : nullptr;
        m_frame.m_output_tensors = inputs ? nullptr : &m_tensors;
    }

    float* data() { return m_buffers[0].get_write_pointer(0); }

    std::vector<anira::capi::Port> m_ports;
    std::vector<anira::BufferF> m_buffers;
    std::vector<anira_tensor> m_tensors;
    anira::capi::StageFrame m_frame;
    anira_stage_ctx m_ctx{};
};

}  // namespace

// ============================================================================================
// anira_pipeline_add_stage
// ============================================================================================

TEST(AbiStage, AddStageRefusals) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    Lifetime lifetime;
    anira_stage_desc stage = promising_stage(&lifetime);
    stage.release = on_release;

    EXPECT_EQ(anira_pipeline_add_stage(nullptr, &stage, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);

    anira_stage_desc bad = stage;
    bad.struct_size = static_cast<uint32_t>(offsetof(anira_stage_desc, user_data));
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "struct_size"), nullptr) << err.message;

    // A flags bit the header does not define, low or high.
    bad = stage;
    bad.flags = 4U;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "flags"), nullptr) << err.message;
    bad.flags = 0x80000000U | ANIRA_STAGE_FLAG_REALTIME_PRE_POST;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);

    bad = stage;
    bad.num_consumed_kinds = 1;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "consumed_kinds"), nullptr) << err.message;
    const std::array<const char*, 1> null_kind{nullptr};
    bad.consumed_kinds = null_kind.data();
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);

    bad = stage;
    bad.abi_version = ANIRA_MAKE_ABI_VERSION(ANIRA_ABI_MAJOR + 1U, 0U);
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_ABI_VERSION);

    // A refused call creates no carrier: nothing is released, now or at the destroy.
    anira_pipeline_destroy(pipeline);
    EXPECT_EQ(lifetime.m_released, 0);
}

// A pipeline holds one stage: the second add is refused by name, creates no carrier and never
// calls its release; the first stage's release fires once, with the pipeline.
TEST(AbiStage, ASecondStageIsRefused) {
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    Lifetime first;
    anira_stage_desc stage = promising_stage(&first);
    stage.release = on_release;
    ASSERT_EQ(anira_pipeline_add_stage(pipeline, &stage, &err), ANIRA_OK) << err.message;

    Lifetime second;
    anira_stage_desc other = promising_stage(&second);
    other.release = on_release;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &other, &err), ANIRA_ERROR_INVALID_STATE);
    EXPECT_NE(std::strstr(err.message, "already has a stage"), nullptr) << err.message;
    // The same descriptor a second time is a second stage as well.
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &stage, &err), ANIRA_ERROR_INVALID_STATE);
    // A malformed descriptor is reported as such, not as a second stage.
    anira_stage_desc bad = other;
    bad.flags = 4U;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);

    anira_pipeline_destroy(pipeline);
    EXPECT_EQ(first.m_released, 1);
    EXPECT_EQ(second.m_released, 0);
}

// A caller compiled against a shorter header hands over the three leading slots only: the
// rest reads as ANIRA_STAGE_DESC_INIT, so the stage has no callback and no flags.
TEST(AbiStage, AShortDescriptorReadsAsTheDefaults) {
    const Context context;
    Lifetime lifetime;
    anira_stage_desc stage = promising_stage(&lifetime);
    stage.prepare = on_prepare;  // beyond the struct_size below: not read
    stage.struct_size =
        static_cast<uint32_t>(offsetof(anira_stage_desc, user_data) + sizeof(void*));
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(lifetime.m_prepared, 0);
    ASSERT_NE(handler.m_handler->m_pipeline.m_stage, nullptr);
    EXPECT_EQ(handler.m_handler->m_pipeline.m_stage->desc().flags, 0U);
    EXPECT_EQ(handler.m_handler->m_pipeline.m_stage->desc().prepare, nullptr);
}

// ============================================================================================
// The carrier: prepare and release
// ============================================================================================

TEST(AbiStage, PrepareSeesTheReportReleaseFiresOnce) {
    const Context context;
    Lifetime lifetime;
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    const ModelConfig model = stream_model();
    const std::array<const anira_model_config*, 1> variants{model.native()};
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err),
              ANIRA_OK)
        << err.message;
    {
        // The descriptor dies with this scope: the carrier copied it.
        anira_stage_desc stage = promising_stage(&lifetime);
        stage.prepare = on_prepare;
        stage.release = on_release;
        ASSERT_EQ(anira_pipeline_add_stage(pipeline, &stage, &err), ANIRA_OK) << err.message;
    }

    // Two handlers and the pipeline share one carrier.
    anira_handler* first = nullptr;
    anira_handler* second = nullptr;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &first, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &second, &err), ANIRA_OK)
        << err.message;
    ASSERT_NE(first->m_pipeline.m_stage, nullptr);
    EXPECT_EQ(first->m_pipeline.m_stage.get(), second->m_pipeline.m_stage.get());
    EXPECT_EQ(lifetime.m_prepared, 0) << "prepare belongs to anira_handler_prepare";

    const anira::ContractHandle contract = zeros_contract();
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(lifetime.m_prepared, 1);
    EXPECT_EQ(lifetime.m_handler, first);
    EXPECT_EQ(lifetime.m_report, anira_handler_plan_report(first));
    EXPECT_EQ(lifetime.m_plans, 1U);
    EXPECT_EQ(lifetime.m_latency, anira_handler_get_latency(first, 0));
    EXPECT_GT(lifetime.m_latency, 0U) << "the handler counts as prepared while the stage prepares";
    // The rest of the record: the library's struct_size, the entry count, the two lists
    // with their canonical names, and a template of the model end of every slot (the
    // spec's dtype and shape at the pinned window, the slot's host domain, all-zero strides,
    // no memory).
    EXPECT_EQ(lifetime.m_struct_size, sizeof(anira_stage_prepare_info));
    EXPECT_EQ(lifetime.m_num_entries, anira_handler_num_entries(first));
    EXPECT_GT(lifetime.m_num_entries, 0U);
    EXPECT_EQ(lifetime.m_num_inputs, 1U);
    EXPECT_EQ(lifetime.m_num_outputs, 1U);
    EXPECT_EQ(lifetime.m_first_input_name, "in");
    EXPECT_EQ(lifetime.m_first_output_name, "out");
    for (const anira_tensor* model_end : {&lifetime.m_first_input, &lifetime.m_first_output}) {
        EXPECT_EQ(model_end->domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
        EXPECT_EQ(model_end->dtype, ANIRA_DTYPE_F32);
        ASSERT_EQ(model_end->ndim, 3U);
        EXPECT_EQ(model_end->shape[0], 1);
        EXPECT_EQ(model_end->shape[1], 1);
        EXPECT_EQ(model_end->shape[2], static_cast<int64_t>(k_hop));
        EXPECT_EQ(model_end->strides[0], 0);
        EXPECT_EQ(model_end->strides[2], 0);
        EXPECT_EQ(model_end->handle.host.ptr, nullptr) << "a template, never dereferenced";
    }
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(lifetime.m_prepared, 2) << "once per prepare";

    // A stage that refuses fails the prepare with its status, and the handler is unprepared
    // afterwards.
    lifetime.m_return = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(err.message, "the stage refused prepare"), nullptr) << err.message;
    EXPECT_EQ(anira_handler_plan_report(first), nullptr);
    lifetime.m_return = ANIRA_OK;

    anira_pipeline_destroy(pipeline);
    EXPECT_EQ(lifetime.m_released, 0);
    anira_handler_destroy(first);
    EXPECT_EQ(lifetime.m_released, 0);
    anira_handler_destroy(second);
    EXPECT_EQ(lifetime.m_released, 1) << "exactly once, with the last carrier";
}

// ============================================================================================
// The shared lifecycle: the prepared pointer per handler, unprepare, reset
// ============================================================================================

namespace {

/// What one handler's prepare handed back: a scratch of its own, whose identity the phase calls
/// and the unprepare of that handler must receive. Allocated in prepare (main thread), never on
/// a per-call path.
struct PerHandler {
    anira_handler* m_handler = nullptr;
    std::atomic<int> m_phase_calls{0};
    std::atomic<int> m_unprepared{0};
};

/// The registration the handlers of a pipeline share: every prepare hands back a fresh
/// PerHandler, every unprepare hands one back, in order.
struct Registration {
    std::vector<std::unique_ptr<PerHandler>> m_prepared;  ///< in prepare order
    std::vector<void*> m_unprepared;                      ///< what unprepare received, in order
    std::atomic<int> m_calls_without_prepared{0};         ///< phase calls with a NULL prepared
    anira_status m_return = ANIRA_OK;                     ///< what prepare returns
    int m_prepare_calls = 0;
    int m_released = 0;
};

anira_status ANIRA_CALL per_handler_prepare(const anira_stage_prepare_info* info,
                                            void* user_data,
                                            void** out_prepared) {
    auto* registration = static_cast<Registration*>(user_data);
    ++registration->m_prepare_calls;
    if (registration->m_return != ANIRA_OK) { return registration->m_return; }
    auto prepared = std::make_unique<PerHandler>();
    prepared->m_handler = info->handler;
    *out_prepared = prepared.get();
    registration->m_prepared.push_back(std::move(prepared));
    return ANIRA_OK;
}

anira_status ANIRA_CALL per_handler_phase(const anira_stage_ctx* ctx,
                                          void* prepared,
                                          void* user_data) ANIRA_NONBLOCKING {
    auto* registration = static_cast<Registration*>(user_data);
    if (prepared == nullptr) {
        registration->m_calls_without_prepared.fetch_add(1);
    } else {
        static_cast<PerHandler*>(prepared)->m_phase_calls.fetch_add(1);
    }
    if (ctx->phase == ANIRA_PHASE_PRE_PROCESS) { return anira_stage_default_pre_process(ctx); }
    if (ctx->phase == ANIRA_PHASE_POST_PROCESS) { return anira_stage_default_post_process(ctx); }
    return ANIRA_OK;
}

void ANIRA_CALL per_handler_unprepare(void* prepared, void* user_data) {
    auto* registration = static_cast<Registration*>(user_data);
    registration->m_unprepared.push_back(prepared);
    if (prepared != nullptr) { static_cast<PerHandler*>(prepared)->m_unprepared.fetch_add(1); }
}

void ANIRA_CALL per_handler_release(void* user_data) {
    ++static_cast<Registration*>(user_data)->m_released;
}

/// Every slot filled: the four phases, prepare, unprepare and release over one registration.
anira_stage_desc per_handler_stage(Registration& registration) {
    anira_stage_desc stage = promising_stage(&registration);
    stage.pre_process = per_handler_phase;
    stage.post_process = per_handler_phase;
    stage.before_inference = per_handler_phase;
    stage.after_inference = per_handler_phase;
    stage.prepare = per_handler_prepare;
    stage.unprepare = per_handler_unprepare;
    stage.release = per_handler_release;
    return stage;
}

}  // namespace

// One registration, two handlers of one pipeline: each prepare hands back its own pointer, and
// every phase call of a handler receives that handler's (a per-chunk scratch behind it is never
// the other handler's), while user_data is the shared registration. Each handler's unprepare
// gets its own pointer back; release fires with the last carrier, after both.
TEST(AbiStage, TwoHandlersOfOnePipelineGetTheirOwnPreparedPointer) {
    const Context context;
    Registration registration;
    anira_error err = ANIRA_ERROR_INIT;
    anira_pipeline* pipeline = nullptr;
    ASSERT_EQ(anira_pipeline_create(&pipeline, &err), ANIRA_OK) << err.message;
    const ModelConfig model = stream_model();
    const std::array<const anira_model_config*, 1> variants{model.native()};
    ASSERT_EQ(anira_pipeline_add_inference(pipeline, variants.data(), 1, nullptr, 0, &err),
              ANIRA_OK)
        << err.message;
    const anira_stage_desc stage = per_handler_stage(registration);
    ASSERT_EQ(anira_pipeline_add_stage(pipeline, &stage, &err), ANIRA_OK) << err.message;
    anira_handler* first = nullptr;
    anira_handler* second = nullptr;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &first, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &second, &err), ANIRA_OK)
        << err.message;
    anira_pipeline_destroy(pipeline);

    const anira::ContractHandle contract = zeros_contract();
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(anira_handler_prepare(second, contract.native(), &err), ANIRA_OK) << err.message;
    ASSERT_EQ(registration.m_prepared.size(), 2U);
    PerHandler& of_first = *registration.m_prepared[0];
    PerHandler& of_second = *registration.m_prepared[1];
    EXPECT_EQ(of_first.m_handler, first);
    EXPECT_EQ(of_second.m_handler, second);
    EXPECT_EQ(first->m_stage_prepared, &of_first);
    EXPECT_EQ(second->m_stage_prepared, &of_second);
    EXPECT_NE(first->m_stage_prepared, second->m_stage_prepared);

    // Three chunks through the first handler, two through the second: the four phase calls of
    // a chunk land on its handler's prepared pointer and on no other.
    Stream first_stream(first);
    first_stream.blocks(3);
    Stream second_stream(second);
    second_stream.blocks(2);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    expect_same_stream(first_stream.m_out, first_stream.expected());
    expect_same_stream(second_stream.m_out, second_stream.expected());
    EXPECT_EQ(of_first.m_phase_calls.load(), 4 * 3);
    EXPECT_EQ(of_second.m_phase_calls.load(), 4 * 2);
    EXPECT_EQ(registration.m_calls_without_prepared.load(), 0);
    EXPECT_EQ(anira_handler_rt_error(first), ANIRA_OK);
    EXPECT_EQ(anira_handler_rt_error(second), ANIRA_OK);

    anira_handler_destroy(first);
    EXPECT_EQ(of_first.m_unprepared.load(), 1);
    EXPECT_EQ(of_second.m_unprepared.load(), 0);
    EXPECT_EQ(registration.m_released, 0);
    anira_handler_destroy(second);
    EXPECT_EQ(of_second.m_unprepared.load(), 1);
    ASSERT_EQ(registration.m_unprepared.size(), 2U);
    EXPECT_EQ(registration.m_unprepared[0], &of_first);
    EXPECT_EQ(registration.m_unprepared[1], &of_second);
    EXPECT_EQ(registration.m_released, 1) << "after every unprepare, with the last carrier";
}

// Three prepares of one handler: three prepares and three unprepares, two at the re-prepares
// (once the old session is released, before the new prepare) and one at the destroy, each
// receiving what the matching prepare handed back, in order; the phase calls of every session
// landed on that session's pointer.
TEST(AbiStage, UnprepareOncePerPrepare) {
    const Context context;
    Registration registration;
    const anira_stage_desc stage = per_handler_stage(registration);
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    const anira::ContractHandle contract = zeros_contract();
    for (size_t round = 0; round < 3; ++round) {
        SCOPED_TRACE("prepare " + std::to_string(round));
        ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        ASSERT_EQ(registration.m_prepared.size(), round + 1);
        EXPECT_EQ(registration.m_unprepared.size(), round) << "the previous prepare was undone";
        EXPECT_EQ(handler.m_handler->m_stage_prepared, registration.m_prepared.back().get());
        Stream stream(handler.m_handler);
        stream.blocks(2);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
    }
    EXPECT_EQ(registration.m_prepare_calls, 3);
    handler.destroy();
    ASSERT_EQ(registration.m_unprepared.size(), 3U);
    for (size_t i = 0; i < 3; ++i) {
        SCOPED_TRACE("prepare " + std::to_string(i));
        EXPECT_EQ(registration.m_unprepared[i], registration.m_prepared[i].get());
        EXPECT_EQ(registration.m_prepared[i]->m_unprepared.load(), 1);
        EXPECT_EQ(registration.m_prepared[i]->m_phase_calls.load(), 4 * 2);
    }
    EXPECT_EQ(registration.m_calls_without_prepared.load(), 0);
    EXPECT_EQ(registration.m_released, 1);
}

// A prepare the stage refuses is never unprepared (nothing was prepared) and leaves the handler
// unprepared; a successful prepare before it is unprepared all the same, once, when the failed
// prepare releases the old session, and so is one that a later prepare fails ahead of the
// stage's prepare.
TEST(AbiStage, ARefusedPrepareCallsNoUnprepare) {
    const Context context;
    Registration registration;
    const anira_stage_desc stage = per_handler_stage(registration);
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    const anira::ContractHandle contract = zeros_contract();
    // Refused from the start: nothing to unprepare, ever.
    registration.m_return = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(handler.m_err.message, "the stage refused prepare"), nullptr)
        << handler.m_err.message;
    EXPECT_EQ(registration.m_prepare_calls, 1);
    EXPECT_TRUE(registration.m_unprepared.empty());
    EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";
    // A successful prepare, then a refused one: the failed prepare releases the old session and
    // undoes the successful prepare; the refused one is never undone.
    registration.m_return = ANIRA_OK;
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(registration.m_prepared.size(), 1U);
    registration.m_return = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_EQ(registration.m_prepare_calls, 3);
    ASSERT_EQ(registration.m_unprepared.size(), 1U);
    EXPECT_EQ(registration.m_unprepared[0], registration.m_prepared[0].get());
    EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";
    // A prepare that fails before the stage's prepare runs (a contract the handler refuses:
    // ANIRA_MISS_CALLBACK without a function) undoes the previous one too, since the handler is
    // unprepared afterwards.
    registration.m_return = ANIRA_OK;
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(registration.m_prepared.size(), 2U);
    const anira::ContractHandle refused =
        explicit_contract(static_cast<uint32_t>(k_hop), anira_test::k_rate, ANIRA_MISS_CALLBACK);
    EXPECT_TRUE(ANIRA_FAILED(handler.prepare(refused))) << handler.m_err.message;
    EXPECT_EQ(registration.m_prepare_calls, 4) << "the stage's prepare did not run";
    ASSERT_EQ(registration.m_unprepared.size(), 2U);
    EXPECT_EQ(registration.m_unprepared[1], registration.m_prepared[1].get());
    EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";
    handler.destroy();
    EXPECT_EQ(registration.m_unprepared.size(), 2U) << "nothing outstanding at the destroy";
    EXPECT_EQ(registration.m_released, 1);
}

namespace {

/// What the stage's reset slot saw, per call: the context, the thread, the pre_process calls
/// since the previous reset, the answers of the accessors for slot 0 (the role answers, a ring
/// or a model tensor is refused) and the entry of the pre_process that followed.
struct ResetLog {
    static constexpr size_t k_capacity = 8;
    std::array<anira_stage_ctx, k_capacity> m_ctx{};
    std::array<std::thread::id, k_capacity> m_thread{};
    std::array<uint32_t, k_capacity> m_pre_since_last{};
    std::array<anira_status, k_capacity> m_role_status{};
    std::array<anira_role, k_capacity> m_role{};
    std::array<anira_status, k_capacity> m_ring_status{};
    std::array<anira_status, k_capacity> m_input_tensor_status{};
    std::array<anira_status, k_capacity> m_output_tensor_status{};
    std::array<uint32_t, k_capacity> m_pre_entry_after{};
    std::atomic<size_t> m_resets{0};
    std::atomic<uint32_t> m_pre_calls{0};
    std::atomic<uint32_t> m_pre_since_reset{0};
    std::atomic<bool> m_reset_pending{false};  ///< a reset ran, no pre_process followed yet
    std::atomic<int> m_foreign_prepared{0};    ///< resets whose prepared was not m_prepared
    void* m_prepared = nullptr;                ///< what prepare hands back
};

anira_status ANIRA_CALL reset_log_prepare(const anira_stage_prepare_info* /*info*/,
                                          void* user_data,
                                          void** out_prepared) {
    *out_prepared = static_cast<ResetLog*>(user_data)->m_prepared;
    return ANIRA_OK;
}

void ANIRA_CALL reset_log_reset(const anira_stage_ctx* ctx,
                                void* prepared,
                                void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<ResetLog*>(user_data);
    const size_t index = log->m_resets.fetch_add(1);
    if (prepared != log->m_prepared) { log->m_foreign_prepared.fetch_add(1); }
    if (index >= ResetLog::k_capacity) { return; }
    log->m_ctx.at(index) = *ctx;
    log->m_thread.at(index) = std::this_thread::get_id();
    log->m_pre_since_last.at(index) = log->m_pre_since_reset.exchange(0);
    anira_role role = ANIRA_ROLE_FORCE32;
    log->m_role_status.at(index) = anira_stage_input_role(ctx, 0, &role);
    log->m_role.at(index) = role;
    anira_ring* ring = nullptr;
    log->m_ring_status.at(index) = anira_stage_input_ring(ctx, 0, &ring);
    anira_tensor tensor;
    log->m_input_tensor_status.at(index) = anira_stage_input_tensor(ctx, 0, &tensor);
    log->m_output_tensor_status.at(index) = anira_stage_output_tensor(ctx, 0, &tensor);
    log->m_reset_pending.store(true);
}

anira_status ANIRA_CALL reset_log_pre_process(const anira_stage_ctx* ctx,
                                              void* /*prepared*/,
                                              void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<ResetLog*>(user_data);
    log->m_pre_calls.fetch_add(1);
    log->m_pre_since_reset.fetch_add(1);
    if (log->m_reset_pending.exchange(false)) {
        const size_t index = log->m_resets.load() - 1;
        if (index < ResetLog::k_capacity) { log->m_pre_entry_after.at(index) = ctx->entry; }
    }
    return anira_stage_default_pre_process(ctx);
}

}  // namespace

// The stage's reset runs for the first chunk of a new stream, before that chunk's pre_process
// and on its thread: once after prepare, once after anira_handler_reset, once after a
// re-prepare, never mid-stream. Its context is that chunk's in ANIRA_PHASE_RESET (the entry the
// pre_process that follows reports), the role answers, a ring or a model tensor is refused and
// recorded (the stage's bug), and it receives this handler's prepared pointer.
TEST(AbiStage, ResetRunsBeforeTheFirstPreProcessOfANewStream) {
    const Context context;
    ResetLog log;
    int scratch = 0;
    log.m_prepared = &scratch;
    anira_stage_desc stage = promising_stage(&log);
    stage.pre_process = reset_log_pre_process;
    stage.prepare = reset_log_prepare;
    stage.reset = reset_log_reset;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    const anira::ContractHandle contract = zeros_contract();
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    EXPECT_EQ(log.m_resets.load(), 0U) << "prepare resets nothing itself: the first chunk does";
    {
        Stream stream(h);
        stream.blocks(3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
    }
    EXPECT_EQ(log.m_resets.load(), 1U) << "once, for the first chunk; never mid-stream";
    EXPECT_EQ(log.m_pre_since_last.at(0), 0U) << "before the first pre_process";
    // anira_handler_reset is one generation bump: the next chunk is the first of a new stream.
    anira_handler_reset(h);
    EXPECT_EQ(log.m_resets.load(), 1U) << "the reset entry calls the stage nothing";
    {
        Stream stream(h);
        stream.blocks(3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
    }
    EXPECT_EQ(log.m_resets.load(), 2U);
    EXPECT_EQ(log.m_pre_since_last.at(1), 3U) << "the three chunks of the first stream between";
    // A re-prepare: a new session and a new stream.
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(log.m_resets.load(), 2U);
    {
        Stream stream(h);
        stream.blocks(2);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
    }
    EXPECT_EQ(log.m_resets.load(), 3U);
    EXPECT_EQ(log.m_pre_calls.load(), 8U);
    EXPECT_EQ(log.m_foreign_prepared.load(), 0);
    for (size_t i = 0; i < 3; ++i) {
        SCOPED_TRACE("reset " + std::to_string(i));
        const anira_stage_ctx& ctx = log.m_ctx.at(i);
        EXPECT_EQ(ctx.phase, static_cast<uint32_t>(ANIRA_PHASE_RESET));
        EXPECT_EQ(ctx.num_inputs, 1U);
        EXPECT_EQ(ctx.num_outputs, 1U);
        EXPECT_EQ(ctx.ticket, ANIRA_TICKET_INVALID);
        EXPECT_EQ(ctx.entry, log.m_pre_entry_after.at(i)) << "the chunk whose pre_process followed";
        EXPECT_LT(ctx.entry, anira_handler_num_entries(h));
        EXPECT_NE(ctx.frame_bits, 0U);
        EXPECT_EQ(log.m_thread.at(i), std::this_thread::get_id()) << "the thread of pre_process";
        EXPECT_EQ(log.m_role_status.at(i), ANIRA_OK);
        EXPECT_EQ(log.m_role.at(i), ANIRA_ROLE_STREAMED);
        EXPECT_EQ(log.m_ring_status.at(i), ANIRA_ERROR_INVALID_STATE);
        EXPECT_EQ(log.m_input_tensor_status.at(i), ANIRA_ERROR_INVALID_STATE);
        EXPECT_EQ(log.m_output_tensor_status.at(i), ANIRA_ERROR_INVALID_STATE);
    }
    // The refused accessors of the last reset were recorded on the handler, as in any phase
    // (the re-prepare re-armed the latch ahead of it).
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_STATE);
}

// ============================================================================================
// The context per phase
// ============================================================================================

TEST(AbiStage, CtxPerPhase) {
    const Context context;
    Probe probe;
    probe.m_defaults = 1;  // the stage owns pre_process and post_process and calls the defaults
    anira_stage_desc stage = promising_stage(&probe);
    stage.pre_process = probe_phase;
    stage.post_process = probe_phase;
    stage.before_inference = probe_phase;
    stage.after_inference = probe_phase;
    // The engines the bundled gain files load on, plus the custom row the handler starts on.
    const std::vector<anira_backend_id> candidates = anira_test::custom_candidates();
    StagedHandler handler(context, anira_test::gain_with_custom(), {stage}, candidates);
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(anira_test::file_contract(k_gain_contract_json, k_hop)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    // The Static input: the whole tensor in the spec's shape ([1]), through the handler's store.
    float gain_value = 0.5F;
    const std::array<int64_t, 1> gain_shape{1};
    anira_tensor gain_tensor{};
    anira_tensor_init_host(&gain_tensor, &gain_value, ANIRA_DTYPE_F32, 1, gain_shape.data());
    ASSERT_EQ(anira_handler_set_static_input(h, 1, &gain_tensor), ANIRA_OK);

    Stream stream(h);
    stream.blocks(3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(stream.m_status, ANIRA_OK);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    expect_same_stream(stream.m_out, stream.expected());

    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(h);
    ASSERT_NE(session, nullptr);
    const uint32_t plan = anira_handler_get_plan(h);
    const anira_plan_info& info = h->m_report.m_plans.at(plan);
    const std::array<uint32_t, 4> phases{ANIRA_PHASE_PRE_PROCESS,
                                         ANIRA_PHASE_BEFORE_INFERENCE,
                                         ANIRA_PHASE_AFTER_INFERENCE,
                                         ANIRA_PHASE_POST_PROCESS};
    for (const uint32_t phase : phases) {
        SCOPED_TRACE("phase " + std::to_string(phase));
        EXPECT_EQ(probe.m_calls.at(phase).load(), 3);
        const anira_stage_ctx& ctx = probe.m_ctx.at(phase);
        EXPECT_EQ(ctx.phase, phase);
        EXPECT_EQ(ctx.engine, info.engine);
        EXPECT_EQ(ctx.provider, info.provider);
        EXPECT_EQ(ctx.variant, 0U);
        EXPECT_EQ(ctx.num_inputs, 2U);
        EXPECT_EQ(ctx.num_outputs, 2U);
        EXPECT_EQ(ctx.ticket, ANIRA_TICKET_INVALID);
        // The entry: below the count, and the same in all four phases of a chunk (the record
        // of every phase is the last chunk's, the streams being driven in lockstep).
        EXPECT_LT(ctx.entry, anira_handler_num_entries(h));
        EXPECT_EQ(ctx.entry, probe.m_ctx.at(ANIRA_PHASE_PRE_PROCESS).entry);
        // The frame is anira's; the reserved slots are NULL, their high halves included.
        EXPECT_NE(ctx.frame_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr0_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr1_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr2_bits, 0U);
        // What a phase exposes, asked through the six accessors like a correct stage: the role
        // of every slot in every phase; the input ring in pre_process and the output ring in
        // post_process, the session's own, of the Streamed tensor alone (the Static one has
        // none, and the probe does not ask); the model's inputs in pre_process and
        // before_inference, its outputs in the two others, of either role, and no tensor of a
        // side outside its phases. Every question asked answered ANIRA_OK, and nothing was
        // recorded: anira_handler_rt_error read ANIRA_OK above.
        const bool pre = phase == ANIRA_PHASE_PRE_PROCESS;
        const bool post = phase == ANIRA_PHASE_POST_PROCESS;
        const bool inputs = pre || phase == ANIRA_PHASE_BEFORE_INFERENCE;
        EXPECT_EQ(probe.m_input_ring_status.at(phase).at(0), pre ? ANIRA_OK : k_not_asked);
        EXPECT_EQ(probe.m_input_rings.at(phase).at(0),
                  pre ? session->m_send_buffer.data() : nullptr);
        EXPECT_EQ(probe.m_output_ring_status.at(phase).at(0), post ? ANIRA_OK : k_not_asked);
        EXPECT_EQ(probe.m_output_rings.at(phase).at(0),
                  post ? session->m_receive_buffer.data() : nullptr);
        EXPECT_EQ(probe.m_input_ring_status.at(phase).at(1), k_not_asked);
        EXPECT_EQ(probe.m_output_ring_status.at(phase).at(1), k_not_asked);
        EXPECT_EQ(probe.m_input_rings.at(phase).at(1), nullptr);
        EXPECT_EQ(probe.m_output_rings.at(phase).at(1), nullptr);
        for (size_t slot = 0; slot < 2; ++slot) {
            SCOPED_TRACE("slot " + std::to_string(slot));
            const anira_role role = slot == 0 ? ANIRA_ROLE_STREAMED : ANIRA_ROLE_STATIC;
            EXPECT_EQ(probe.m_input_role_status.at(phase).at(slot), ANIRA_OK);
            EXPECT_EQ(probe.m_output_role_status.at(phase).at(slot), ANIRA_OK);
            EXPECT_EQ(probe.m_input_roles.at(phase).at(slot), role);
            EXPECT_EQ(probe.m_output_roles.at(phase).at(slot), role);
            EXPECT_EQ(probe.m_input_status.at(phase).at(slot), inputs ? ANIRA_OK : k_not_asked);
            EXPECT_EQ(probe.m_output_status.at(phase).at(slot), inputs ? k_not_asked : ANIRA_OK);
            EXPECT_EQ(probe.m_inputs.at(phase).at(slot).dtype, inputs ? ANIRA_DTYPE_F32 : 0U);
            EXPECT_EQ(probe.m_outputs.at(phase).at(slot).dtype, inputs ? 0U : ANIRA_DTYPE_F32);
            EXPECT_EQ(anira_tensor_data_f32(&probe.m_inputs.at(phase).at(slot)) != nullptr, inputs);
            EXPECT_EQ(anira_tensor_data_f32(&probe.m_outputs.at(phase).at(slot)) != nullptr,
                      !inputs);
        }
    }
    // The tensors: the spec's shape and dtype over a struct's own buffer, host memory.
    const anira_tensor& audio = probe.m_inputs.at(ANIRA_PHASE_PRE_PROCESS).at(0);
    EXPECT_EQ(audio.domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(audio.dtype, ANIRA_DTYPE_F32);
    ASSERT_EQ(audio.ndim, 3U);
    EXPECT_EQ(audio.shape[0], 1);
    EXPECT_EQ(audio.shape[1], 1);
    EXPECT_EQ(audio.shape[2], static_cast<int64_t>(k_hop));
    bool input_is_a_struct_buffer = false;
    bool output_is_a_struct_buffer = false;
    for (const auto& chunk : session->m_inference_queue) {
        input_is_a_struct_buffer =
            input_is_a_struct_buffer ||
            audio.handle.host.ptr == chunk->m_tensor_input_data.at(0).get_read_pointer(0);
        output_is_a_struct_buffer =
            output_is_a_struct_buffer ||
            probe.m_outputs.at(ANIRA_PHASE_POST_PROCESS).at(0).handle.host.ptr ==
                chunk->m_tensor_output_data.at(0).get_read_pointer(0);
    }
    EXPECT_TRUE(input_is_a_struct_buffer);
    EXPECT_TRUE(output_is_a_struct_buffer);
    const anira_tensor& gain = probe.m_inputs.at(ANIRA_PHASE_PRE_PROCESS).at(1);
    ASSERT_EQ(gain.ndim, 1U);
    EXPECT_EQ(gain.shape[0], 1);
    EXPECT_EQ(probe.m_static_input, 0.5F)
        << "every Static input is materialised ahead of the stages";
    // pre_process and post_process run on the thread that drives the entry, the two hooks on
    // an inference thread.
    EXPECT_EQ(probe.m_thread.at(ANIRA_PHASE_PRE_PROCESS), std::this_thread::get_id());
    EXPECT_EQ(probe.m_thread.at(ANIRA_PHASE_POST_PROCESS), std::this_thread::get_id());
    EXPECT_NE(probe.m_thread.at(ANIRA_PHASE_BEFORE_INFERENCE), std::this_thread::get_id());
    EXPECT_NE(probe.m_thread.at(ANIRA_PHASE_AFTER_INFERENCE), std::this_thread::get_id());
}

namespace {

/// What the stage of ASlotBeyondTheCountIsRecordedWithTheStagesName got for a slot no tensor
/// has. Written on an inference thread.
struct Beyond {
    std::atomic<int32_t> m_role_status{ANIRA_OK};
    std::atomic<uint32_t> m_role{0};
    std::atomic<int32_t> m_ring_status{ANIRA_OK};
    std::atomic<int> m_rings{0};
    std::atomic<int32_t> m_status{ANIRA_OK};
    std::atomic<uint32_t> m_dtype{1};
};

anira_status ANIRA_CALL ask_beyond(const anira_stage_ctx* ctx,
                                   void* /*prepared*/,
                                   void* user_data) ANIRA_NONBLOCKING {
    auto* beyond = static_cast<Beyond*>(user_data);
    const uint32_t slot = ctx->num_inputs;  // one beyond the last
    anira_role role = ANIRA_ROLE_STREAMED;
    beyond->m_role_status.store(anira_stage_input_role(ctx, slot, &role));
    beyond->m_role.store(static_cast<uint32_t>(role));
    anira_ring* ring = nullptr;
    beyond->m_ring_status.store(anira_stage_input_ring(ctx, slot, &ring));
    if (ring != nullptr) { beyond->m_rings.fetch_add(1); }
    anira_tensor tensor;
    beyond->m_status.store(anira_stage_input_tensor(ctx, slot, &tensor));
    beyond->m_dtype.store(tensor.dtype);
    return ANIRA_OK;
}

}  // namespace

// A slot at the side's count, asked in before_inference: the refusal reaches the handler's
// rt_error from the inference thread, and the one latched record names the entry and the stage
// that asked. The stage returned ANIRA_OK, so the chunk is not failed and the stream is whole.
TEST(AbiStage, ASlotBeyondTheCountIsRecordedWithTheStagesName) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    Beyond beyond;
    anira_stage_desc stage = promising_stage(&beyond);
    stage.before_inference = ask_beyond;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    expect_same_stream(stream.m_out, stream.expected());
    EXPECT_EQ(beyond.m_role_status.load(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(beyond.m_role.load(), static_cast<uint32_t>(ANIRA_ROLE_FORCE32)) << "no role";
    EXPECT_EQ(beyond.m_ring_status.load(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(beyond.m_rings.load(), 0) << "no ring";
    EXPECT_EQ(beyond.m_status.load(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(beyond.m_dtype.load(), 0U) << "an all-zero record";
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "is out of range", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "is out of range", "rt");
    EXPECT_NE(record.m_message.find("anira_stage_input_role: the stage: slot 1 is out of range, "
                                    "the model has 1 input tensors"),
              std::string::npos)
        << record.m_message;
#endif
}

namespace {

// A stage's prepare may call the two Static entries: no driver thread runs during prepare, and
// the store is the handler's, so the value is what the first inference sees.
anira_status ANIRA_CALL set_gain_in_prepare(const anira_stage_prepare_info* info,
                                            void* /*user_data*/,
                                            void** /*out_prepared*/) {
    anira_handler* const handler = info->handler;
    float gain_value = 0.25F;
    const std::array<int64_t, 1> gain_shape{1};
    anira_tensor gain_tensor{};
    anira_tensor_init_host(&gain_tensor, &gain_value, ANIRA_DTYPE_F32, 1, gain_shape.data());
    const anira_status set = anira_handler_set_static_input(handler, 1, &gain_tensor);
    if (set != ANIRA_OK) { return set; }
    // The output store answers too: zeros, nothing has been captured yet.
    float gain_out = -1.0F;
    anira_tensor out_tensor{};
    anira_tensor_init_host(&out_tensor, &gain_out, ANIRA_DTYPE_F32, 1, gain_shape.data());
    const anira_status got = anira_handler_get_static_output(handler, 1, &out_tensor);
    return got != ANIRA_OK || gain_out == 0.0F ? got : ANIRA_ERROR_INTERNAL;
}

}  // namespace

TEST(AbiStage, AStagesPrepareMaySetAStaticInput) {
    const Context context;
    Probe probe;
    probe.m_defaults = 1;
    anira_stage_desc stage = promising_stage(&probe);
    stage.pre_process = probe_phase;
    stage.prepare = set_gain_in_prepare;
    const std::vector<anira_backend_id> candidates = anira_test::custom_candidates();
    StagedHandler handler(context, anira_test::gain_with_custom(), {stage}, candidates);
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(anira_test::file_contract(k_gain_contract_json, k_hop)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    Stream stream(h);
    stream.blocks(2);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    EXPECT_EQ(probe.m_static_input, 0.25F) << "what prepare stored is what pre_process found";
}

// ============================================================================================
// The ring accessors
// ============================================================================================

TEST(AbiStage, RingAccessorsF32AndI16) {
    const Context context;  // the real-time log queue is the core's
    anira_drain_log();
    RecordCollector collector;
    anira::RtLatch latch;
    anira::RingOwner owner;
    owner.m_rt = &latch;

    // A float32 ring: push, pop, history, fill, discard.
    anira::RingBuffer f32;
    ASSERT_TRUE(f32.initialize_with_positions(2, 16, ANIRA_DTYPE_F32));
    f32.set_owner(4, &owner);
    EXPECT_EQ(anira_ring_dtype(&f32), ANIRA_DTYPE_F32);
    EXPECT_EQ(anira_ring_num_channels(&f32), 2U);
    const std::array<float, 4> in{1.0F, 2.0F, 3.0F, 4.0F};
    EXPECT_EQ(anira_ring_push_block(&f32, 1, in.data(), ANIRA_DTYPE_F32, 4), 4U);
    EXPECT_EQ(anira_ring_available(&f32, 1), 4U);
    EXPECT_EQ(anira_ring_available(&f32, 0), 0U);
    std::array<float, 4> out{};
    EXPECT_EQ(anira_ring_pop_block(&f32, 1, out.data(), ANIRA_DTYPE_F32, 2), 2U);
    EXPECT_EQ(out[0], 1.0F);
    EXPECT_EQ(out[1], 2.0F);
    EXPECT_EQ(anira_ring_available_past(&f32, 1), 2U);
    EXPECT_EQ(anira_ring_peek_past_block(&f32, 1, out.data(), ANIRA_DTYPE_F32, 2), 2U);
    EXPECT_EQ(out[1], 2.0F) << "the element popped last is the last of the history";
    EXPECT_EQ(anira_ring_discard(&f32, 1, 8), 2U) << "at most the available ones";
    const float fill = 0.25F;
    EXPECT_EQ(anira_ring_push_fill(&f32, 0, &fill, ANIRA_DTYPE_F32, 3), 3U);
    EXPECT_EQ(anira_ring_pop_block(&f32, 0, out.data(), ANIRA_DTYPE_F32, 3), 3U);
    EXPECT_EQ(out[2], 0.25F);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);

    // An int16 ring holds int16, and says so.
    anira::RingBuffer i16;
    ASSERT_TRUE(i16.initialize_with_positions(1, 16, ANIRA_DTYPE_I16));
    i16.set_owner(4, &owner);
    EXPECT_EQ(anira_ring_dtype(&i16), ANIRA_DTYPE_I16);
    const std::array<int16_t, 4> words{-3, 7, 32767, -32768};
    EXPECT_EQ(anira_ring_push_block(&i16, 0, words.data(), ANIRA_DTYPE_I16, 4), 4U);
    std::array<int16_t, 4> popped{};
    EXPECT_EQ(anira_ring_pop_block(&i16, 0, popped.data(), ANIRA_DTYPE_I16, 4), 4U);
    EXPECT_EQ(popped, words);

    // Another dtype than the ring's: 0, nothing moved, CONFIG; nothing converts, either way.
    EXPECT_EQ(anira_ring_push_block(&i16, 0, in.data(), ANIRA_DTYPE_F32, 4), 0U);
    EXPECT_EQ(anira_ring_available(&i16, 0), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_ring_push_block(&i16, 0, words.data(), ANIRA_DTYPE_I16, 4), 4U);
    out.fill(-1.0F);
    EXPECT_EQ(anira_ring_pop_block(&i16, 0, out.data(), ANIRA_DTYPE_F32, 4), 0U);
    EXPECT_EQ(anira_ring_peek_past_block(&i16, 0, out.data(), ANIRA_DTYPE_F32, 4), 0U);
    EXPECT_EQ(anira_ring_pop_windows(&i16, 0, out.data(), ANIRA_DTYPE_F32, 2, 2, 0, 1), 0U);
    EXPECT_EQ(anira_ring_push_fill(&i16, 0, &fill, ANIRA_DTYPE_F32, 4), 0U);
    EXPECT_EQ(out[0], -1.0F) << "a refused pop writes nothing";
    EXPECT_EQ(anira_ring_available(&i16, 0), 4U) << "a refused pop pops nothing";
    EXPECT_EQ(anira_ring_discard(&i16, 0, 1), 1U) << "discard takes no dtype";

    // A channel out of range and NULL memory: 0 and INVALID_ARGUMENT. A NULL ring: 0, and no
    // word to record on.
    latch.rearm();
    EXPECT_EQ(anira_ring_available(&f32, 2), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_ARGUMENT);
    latch.rearm();
    EXPECT_EQ(anira_ring_pop_block(&f32, 0, nullptr, ANIRA_DTYPE_F32, 1), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_ARGUMENT);
    latch.rearm();
    EXPECT_EQ(anira_ring_dtype(nullptr), 0U);
    EXPECT_EQ(anira_ring_num_channels(nullptr), 0U);
    EXPECT_EQ(anira_ring_available(nullptr, 0), 0U);
    EXPECT_EQ(anira_ring_available_past(nullptr, 0), 0U);
    EXPECT_EQ(anira_ring_pop_block(nullptr, 0, out.data(), ANIRA_DTYPE_F32, 1), 0U);
    EXPECT_EQ(anira_ring_peek_past_block(nullptr, 0, out.data(), ANIRA_DTYPE_F32, 1), 0U);
    EXPECT_EQ(anira_ring_push_block(nullptr, 0, in.data(), ANIRA_DTYPE_F32, 1), 0U);
    EXPECT_EQ(anira_ring_push_fill(nullptr, 0, &fill, ANIRA_DTYPE_F32, 1), 0U);
    EXPECT_EQ(anira_ring_discard(nullptr, 0, 1), 0U);
    EXPECT_EQ(anira_ring_pop_windows(nullptr, 0, out.data(), ANIRA_DTYPE_F32, 1, 1, 0, 1), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);

    // A ring outside a session has no owner: the refusal stands, nothing is recorded.
    anira::RingBuffer orphan;
    ASSERT_TRUE(orphan.initialize_with_positions(1, 8, ANIRA_DTYPE_I16));
    EXPECT_EQ(anira_ring_push_block(&orphan, 0, in.data(), ANIRA_DTYPE_F32, 4), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);

    anira_drain_log();
#ifdef ENABLE_LOGGING
    // One record per kind, naming the entry.
    EXPECT_EQ(anira_test::count_records(collector, "nothing converts", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "nothing converts", "rt");
    EXPECT_NE(record.m_message.find("anira_ring_push_block: the stage: dtype"), std::string::npos)
        << record.m_message;
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(anira_test::count_records(collector, "is out of range", "rt"), 1U);
#endif
}

// anira_ring_pop_windows is the 2.x batched pop (PrePostProcessor::pop_samples_from_buffer with
// num_batches), element for element.
TEST(AbiStage, PopWindowsEqualsTheBatchedPop) {
    constexpr size_t k_new = 3;
    constexpr size_t k_old = 2;
    constexpr size_t k_offset = 1;
    constexpr size_t k_batches = 4;
    constexpr size_t k_total = k_offset + (k_batches * (k_new + k_old));
    std::array<anira::RingBuffer, 2> rings;
    for (anira::RingBuffer& ring : rings) {
        ring.initialize_with_positions(1, 64);
        // Two elements of history, then the unread run.
        for (int n = 0; n < 2 + static_cast<int>(k_batches * k_new); ++n) {
            ring.push_sample(0, static_cast<float>(n + 1));
        }
        (void)ring.pop_sample(0);
        (void)ring.pop_sample(0);
    }
    const std::vector<anira::ModelData> model_data = {
        {"unused-by-this-test", anira::InferenceBackend::CUSTOM}};
    const std::vector<anira::TensorShape> shapes = {{{{1, 1, 8}}, {{1, 1, 8}}}};
    anira::InferenceConfig config{model_data, shapes, anira::ProcessingSpec(), 5.0F};
    anira::PrePostProcessor v2(config);
    anira::BufferF expected(1, k_total);
    expected.clear();
    v2.pop_samples_from_buffer(rings[0], expected, k_new, k_old, k_offset, k_batches);

    std::array<float, k_total> windows{};
    EXPECT_EQ(anira_ring_pop_windows(&rings[1],
                                     0,
                                     windows.data(),
                                     ANIRA_DTYPE_F32,
                                     k_new,
                                     k_old,
                                     k_offset,
                                     static_cast<uint32_t>(k_batches)),
              k_batches * (k_new + k_old));
    for (size_t n = 0; n < k_total; ++n) {
        EXPECT_EQ(windows.at(n), expected.get_sample(0, n)) << "element " << n;
    }
    EXPECT_EQ(anira_ring_available(&rings[1], 0), rings[0].available(0));
    EXPECT_EQ(windows.at(k_offset), 1.0F) << "a window starts with its history";
}

// ============================================================================================
// The context accessors over a hand-built context
// ============================================================================================

// The six accessors over every phase of a hand-built context. A legal question answers
// ANIRA_OK with the role, the ring or the descriptor. A ring or a model tensor the slot does not
// have in this phase is the stage's bug, not an answer: ANIRA_ERROR_INVALID_STATE with the
// out-parameter reset, recorded into the latch with one record per kind naming the entry, the
// slot and the phase (the first of a kind is logged, the others counted).
// A slot out of range and a NULL out are ANIRA_ERROR_INVALID_ARGUMENT, recorded the same way; a
// NULL ctx and a ctx without a frame are refused with nothing to record into. The default
// bodies over the same slots ask nothing a slot lacks and record nothing.
TEST(AbiStage, ContextAccessorsAnswerAndRefuse) {
    const Context context;  // the real-time log queue is the core's
    anira_drain_log();
    RecordCollector collector;
    anira::RtLatch latch;
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(1, 16, ANIRA_DTYPE_F32));
    ring.set_owner(4, nullptr);
    // Slot 0 Streamed over `ring`, slot 1 a State tensor, on both sides.
    HandBuiltCtx built(ANIRA_PHASE_PRE_PROCESS, ring, {1, 1, 4}, /*with_state=*/true);
    built.m_frame.m_rt = &latch;
    const anira_stage_ctx& ctx = built.m_ctx;
    anira_role role = ANIRA_ROLE_FORCE32;
    anira_ring* out_ring = nullptr;
    anira_tensor tensor;

    // pre_process, the legal questions: the role of every slot, the input ring and the model's
    // input of the Streamed slot.
    EXPECT_EQ(anira_stage_input_role(&ctx, 0, &role), ANIRA_OK);
    EXPECT_EQ(role, ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_output_role(&ctx, 0, &role), ANIRA_OK);
    EXPECT_EQ(role, ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_input_role(&ctx, 1, &role), ANIRA_OK);
    EXPECT_EQ(role, ANIRA_ROLE_STATE);
    EXPECT_EQ(anira_stage_output_role(&ctx, 1, &role), ANIRA_OK);
    EXPECT_EQ(role, ANIRA_ROLE_STATE);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0, &out_ring), ANIRA_OK);
    EXPECT_EQ(out_ring, &ring);
    ASSERT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(tensor.domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    ASSERT_EQ(tensor.ndim, 3U);
    EXPECT_EQ(tensor.shape[2], 4);
    EXPECT_EQ(tensor.release, nullptr) << "borrowed";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), built.data());
    EXPECT_EQ(latch.rt_error(), ANIRA_OK) << "every question so far was a legal one";

    // pre_process, the illegal questions: no output ring and no output here, no ring and no
    // host end of a State slot. Each is INVALID_STATE with the out-parameter reset, and
    // recorded: the first of the kind is logged, the others are counted.
    out_ring = &ring;
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(out_ring, nullptr);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(latch.m_suppressed.load(), 0U);
    out_ring = &ring;
    EXPECT_EQ(anira_stage_input_ring(&ctx, 1, &out_ring), ANIRA_ERROR_INVALID_STATE)
        << "a State slot has no ring";
    EXPECT_EQ(out_ring, nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 1U);
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_TRUE(all_zero(tensor)) << "an all-zero record";
    EXPECT_EQ(tensor.dtype, 0U);
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 2U);
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 1, &tensor), ANIRA_ERROR_INVALID_STATE)
        << "a State input has no host end in pre_process";
    EXPECT_TRUE(all_zero(tensor));
    EXPECT_EQ(latch.rearm(), 3U);

    // The other phases: the role stays, the rings and the tensors follow the phase. Per phase
    // the legal questions, then the illegal ones, recorded and counted.
    built.expose(ANIRA_PHASE_BEFORE_INFERENCE);
    EXPECT_EQ(anira_stage_input_role(&ctx, 0, &role), ANIRA_OK);
    EXPECT_EQ(role, ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 1, &tensor), ANIRA_OK)
        << "before_inference exposes a State input";
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE)
        << "no ring outside pre_process";
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(latch.rearm(), 2U);

    built.expose(ANIRA_PHASE_AFTER_INFERENCE);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 1, &tensor), ANIRA_OK)
        << "after_inference exposes a State output";
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);
    // The model tensor's refusal first this time, so that its record is the one logged.
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(latch.rearm(), 2U);

    built.expose(ANIRA_PHASE_POST_PROCESS);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0, &out_ring), ANIRA_OK);
    EXPECT_EQ(out_ring, &ring);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0, &out_ring), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(out_ring, nullptr);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 1, &out_ring), ANIRA_ERROR_INVALID_STATE)
        << "a State slot has no ring";
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 1, &tensor), ANIRA_ERROR_INVALID_STATE)
        << "a State output has no host end in post_process";
    EXPECT_TRUE(all_zero(tensor));
    EXPECT_EQ(latch.rearm(), 3U);

    // A slot at or beyond the side's count: no role, no ring, an all-zero record, and
    // INVALID_ARGUMENT, recorded; a NULL out likewise. The first of the kind is the one that is
    // logged, the others are counted.
    built.expose(ANIRA_PHASE_PRE_PROCESS);
    role = ANIRA_ROLE_STREAMED;
    EXPECT_EQ(anira_stage_input_role(&ctx, 2, &role), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(role, ANIRA_ROLE_FORCE32) << "no role";
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.m_suppressed.load(), 0U);
    role = ANIRA_ROLE_STREAMED;
    EXPECT_EQ(anira_stage_output_role(&ctx, 2, &role), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(role, ANIRA_ROLE_FORCE32);
    EXPECT_EQ(latch.m_suppressed.load(), 1U);
    out_ring = &ring;
    EXPECT_EQ(anira_stage_input_ring(&ctx, 7, &out_ring), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(out_ring, nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 2U);
    out_ring = &ring;
    EXPECT_EQ(anira_stage_output_ring(&ctx, 7, &out_ring), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(out_ring, nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 3U);
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 2, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(all_zero(tensor));
    EXPECT_EQ(latch.m_suppressed.load(), 4U);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 2, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.m_suppressed.load(), 5U);
    // A NULL out, of each kind of accessor.
    EXPECT_EQ(anira_stage_input_role(&ctx, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.rearm(), 8U);

    // A NULL ctx and a ctx without a frame: the refusal stands, the out-parameter is reset, and
    // there is no latch to record into.
    anira_stage_ctx bare{};
    bare.phase = ANIRA_PHASE_PRE_PROCESS;
    bare.num_inputs = 1;
    bare.num_outputs = 1;
    const std::array<const anira_stage_ctx*, 2> without_a_frame{nullptr, &bare};
    for (const anira_stage_ctx* refused : without_a_frame) {
        role = ANIRA_ROLE_STREAMED;
        EXPECT_EQ(anira_stage_input_role(refused, 0, &role), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(role, ANIRA_ROLE_FORCE32);
        role = ANIRA_ROLE_STREAMED;
        EXPECT_EQ(anira_stage_output_role(refused, 0, &role), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(role, ANIRA_ROLE_FORCE32);
        out_ring = &ring;
        EXPECT_EQ(anira_stage_input_ring(refused, 0, &out_ring), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(out_ring, nullptr);
        out_ring = &ring;
        EXPECT_EQ(anira_stage_output_ring(refused, 0, &out_ring), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(out_ring, nullptr);
        std::memset(&tensor, 0xff, sizeof(tensor));
        EXPECT_EQ(anira_stage_input_tensor(refused, 0, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_TRUE(all_zero(tensor));
        EXPECT_EQ(anira_stage_output_tensor(refused, 0, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(anira_stage_input_role(refused, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(anira_stage_input_ring(refused, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(anira_stage_input_tensor(refused, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(anira_stage_default_pre_process(&bare), ANIRA_ERROR_INVALID_ARGUMENT);
    bare.phase = ANIRA_PHASE_POST_PROCESS;
    EXPECT_EQ(anira_stage_default_post_process(&bare), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);

    // The default bodies over the same two slots: the State slot is skipped, never asked for a
    // ring it lacks, so a default records nothing whatever the pipeline's slots are.
    built.expose(ANIRA_PHASE_PRE_PROCESS);
    EXPECT_EQ(anira_stage_default_pre_process(&ctx), ANIRA_OK);
    built.expose(ANIRA_PHASE_POST_PROCESS);
    EXPECT_EQ(anira_stage_default_post_process(&ctx), ANIRA_OK);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK) << "a default body asks nothing a slot lacks";

    anira_drain_log();
#ifdef ENABLE_LOGGING
    // One record per kind and re-arm, the first refusal of the kind, naming the entry, the
    // slot and the phase. The INVALID_STATE kind was re-armed per phase: the
    // ring's refusal came first in three phases, the model tensor's in after_inference.
    EXPECT_EQ(anira_test::count_records(collector, "has no ring in", "rt"), 3U);
    const RecordCollector::Record no_ring =
        anira_test::find_record(collector, "has no ring in", "rt");
    EXPECT_NE(no_ring.m_message.find(
                  "anira_stage_output_ring: the stage: slot 0 has no ring in pre_process"),
              std::string::npos)
        << no_ring.m_message;
    EXPECT_EQ(no_ring.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(anira_test::count_records(collector, "has no model tensor in", "rt"), 1U);
    const RecordCollector::Record no_tensor =
        anira_test::find_record(collector, "has no model tensor in", "rt");
    EXPECT_NE(no_tensor.m_message.find("anira_stage_input_tensor: the stage: slot 0 has no "
                                       "model tensor in after_inference"),
              std::string::npos)
        << no_tensor.m_message;
    EXPECT_EQ(anira_test::count_records(collector, "is out of range", "rt"), 1U);
    const RecordCollector::Record out_of_range =
        anira_test::find_record(collector, "is out of range", "rt");
    EXPECT_NE(out_of_range.m_message.find("anira_stage_input_role: the stage: slot 2 is out of "
                                          "range, the model has 2 input tensors"),
              std::string::npos)
        << out_of_range.m_message;
    EXPECT_EQ(out_of_range.m_flags,
              ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(anira_test::count_records(collector, "NULL out", "rt"), 0U)
        << "the same kind as the slot out of range: counted, not logged";
#endif
}

// ============================================================================================
// The default bodies over a hand-built context
// ============================================================================================

// test_PrePostProcessor's HistoryFillsTheWindowHead over anira_stage_default_pre_process: a
// window of 8 with a hop of 4 is the previous hop followed by the new one.
TEST(AbiStage, DefaultPreProcessFillsTheWindowHeadWithHistory) {
    constexpr size_t k_window = 8;
    constexpr size_t k_step = 4;
    anira::RingBuffer ring;
    ring.initialize_with_positions(1, 32);
    ring.set_owner(k_step, nullptr);
    HandBuiltCtx built(ANIRA_PHASE_PRE_PROCESS, ring, {1, 1, static_cast<int64_t>(k_window)});
    const float* window = built.data();

    for (size_t n = 0; n < k_step; ++n) { ring.push_sample(0, static_cast<float>(n + 1)); }
    EXPECT_EQ(anira_stage_default_pre_process(&built.m_ctx), ANIRA_OK);
    for (size_t n = 0; n < k_step; ++n) {
        EXPECT_EQ(window[n], 0.0F) << "no history yet, sample " << n;
        EXPECT_EQ(window[k_step + n], static_cast<float>(n + 1));
    }
    for (size_t n = 0; n < k_step; ++n) { ring.push_sample(0, static_cast<float>(k_step + n + 1)); }
    EXPECT_EQ(anira_stage_default_pre_process(&built.m_ctx), ANIRA_OK);
    for (size_t n = 0; n < k_window; ++n) {
        EXPECT_EQ(window[n], static_cast<float>(n + 1)) << "window sample " << n;
    }

    // Another phase than its own, and a NULL context.
    built.expose(ANIRA_PHASE_POST_PROCESS);
    EXPECT_EQ(anira_stage_default_pre_process(&built.m_ctx), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_stage_default_pre_process(nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_stage_default_post_process(nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
}

// test_PrePostProcessor's MultichannelTensorIsNotAWindow: a [1, 4, 1] tensor holds more
// elements than its hop, but per channel exactly the hop, so no history is mixed in.
TEST(AbiStage, DefaultPreProcessTreatsAMultichannelTensorAsChannels) {
    anira::RingBuffer ring;
    ring.initialize_with_positions(4, 8);
    ring.set_owner(1, nullptr);
    HandBuiltCtx built(ANIRA_PHASE_PRE_PROCESS, ring, {1, 4, 1});
    for (size_t channel = 0; channel < 4; ++channel) {
        ring.push_sample(channel, static_cast<float>(channel + 1));
    }
    EXPECT_EQ(anira_stage_default_pre_process(&built.m_ctx), ANIRA_OK);
    for (size_t n = 0; n < 4; ++n) { EXPECT_EQ(built.data()[n], static_cast<float>(n + 1)); }
}

// A ring of another dtype than its tensor: that slot stays untouched, the status is CONFIG,
// and the default records nothing itself.
TEST(AbiStage, DefaultBodiesRefuseARingOfAnotherDtype) {
    anira::RtLatch latch;
    anira::RingOwner owner;
    owner.m_rt = &latch;
    anira::RingBuffer in_ring;
    anira::RingBuffer out_ring;
    ASSERT_TRUE(in_ring.initialize_with_positions(1, 16, ANIRA_DTYPE_I16));
    ASSERT_TRUE(out_ring.initialize_with_positions(1, 16, ANIRA_DTYPE_I16));
    in_ring.set_owner(4, &owner);
    out_ring.set_owner(4, &owner);
    const std::array<int16_t, 4> words{1, 2, 3, 4};
    ASSERT_EQ(in_ring.push_block(0, words.data(), ANIRA_DTYPE_I16, 4), 4U);

    HandBuiltCtx pre(ANIRA_PHASE_PRE_PROCESS, in_ring, {1, 1, 4});
    pre.m_frame.m_rt = &latch;
    for (size_t n = 0; n < 4; ++n) { pre.data()[n] = 9.0F; }
    EXPECT_EQ(anira_stage_default_pre_process(&pre.m_ctx), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(in_ring.available(0), 4U);
    EXPECT_EQ(pre.data()[0], 9.0F);

    HandBuiltCtx post(ANIRA_PHASE_POST_PROCESS, out_ring, {1, 1, 4});
    post.m_frame.m_rt = &latch;
    EXPECT_EQ(anira_stage_default_post_process(&post.m_ctx), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(out_ring.available(0), 0U);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK) << "the status is returned, not recorded";
}

// ============================================================================================
// The stage in its phases
// ============================================================================================

// No stage fills pre_process or post_process: the default body of each runs, once per chunk.
// A stage that fills them and calls the default itself gets the same stream, and the default
// does not run a second time behind it (the rings would move two hops).
// The default body runs iff the descriptor's slot is NULL: a stage of the hooks alone leaves
// both defaults running (the stream arrives), a stage that fills pre_process and post_process
// owns them and composes the default itself (once, so the count check stays quiet).
TEST(AbiStage, TheDefaultBodyRunsIffTheSlotIsNull) {
    const Context context;
    {
        Probe probe;
        anira_stage_desc stage = promising_stage(&probe);
        stage.before_inference = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(6);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_BEFORE_INFERENCE).load(), 6);
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 0);
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    }
    {
        Probe probe;
        probe.m_defaults = 1;
        anira_stage_desc stage = promising_stage(&probe);
        stage.pre_process = probe_phase;
        stage.post_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(6);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 6);
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_POST_PROCESS).load(), 6);
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK)
            << "one hop per ring per chunk: the default did not run behind the stage";
    }
}

// Composition is explicit, in the stage's own code: a post_process that works on the model
// tensor first and then calls the default push moves exactly one hop of what it left there.
TEST(AbiStage, OneStageComposesTheDefaultBehindItsOwnWork) {
    const Context context;
    Probe probe;
    probe.m_scale = 2.0F;  // works on the tensor ...
    probe.m_defaults = 1;  // ... then pushes it through the default
    anira_stage_desc stage = promising_stage(&probe);
    stage.post_process = probe_phase;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(4);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::vector<float> wanted = stream.expected();
    for (float& sample : wanted) { sample *= 2.0F; }
    expect_same_stream(stream.m_out, wanted);
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_POST_PROCESS).load(), 4);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
}

// ============================================================================================
// The count check of the two ring-moving phases
// ============================================================================================

// A pre_process that pops nothing: anira discards the hop, so the stream stays aligned, and
// says so once. The model input was never written, so the chunks deliver zeros.
TEST(AbiStage, APreProcessThatPopsNothingIsRepairedAndRecorded) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    Probe probe;  // fills pre_process, calls no default
    anira_stage_desc stage = promising_stage(&probe);
    stage.pre_process = probe_phase;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(4);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_CONFIG);
    const std::shared_ptr<anira::SessionElement> session =
        anira_test::session_of(handler.m_handler);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->m_send_buffer.at(0).available(0), 0U) << "every hop was given up";
    expect_same_stream(stream.m_out, std::vector<float>(stream.m_in.size(), 0.0F));
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "pre_process moved", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "pre_process moved", "rt");
    EXPECT_NE(record.m_message.find("the stage: pre_process moved"), std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("input tensor 0, channel 0 by 0 elements, the hop is 512"),
              std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("the rest was discarded"), std::string::npos)
        << record.m_message;
#endif
}

// A pre_process that pops the ring twice (its own pop and the default's): more than the hop
// left the ring, which nothing can undo; the hop-count check is the guard of a buggy stage.
TEST(AbiStage, APreProcessThatPopsTwiceShiftsTheStreamAndIsRecorded) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    Probe probe;
    probe.m_defaults = 2;  // two pops of one chunk
    anira_stage_desc stage = promising_stage(&probe);
    stage.pre_process = probe_phase;
    StagedHandler handler(context, stream_model(), {stage});
    // Two hops per block: the ring holds both when the first chunk is formed.
    ASSERT_EQ(handler.prepare(zeros_contract(static_cast<uint32_t>(2 * k_hop))), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    anira_test::FloatFace face(h);
    std::vector<float> data(2 * k_hop, 1.0F);
    const std::array<float*, 1> channels{data.data()};
    size_t delivered = 0;
    EXPECT_EQ(face.process_inplace(channels.data(), 2 * k_hop, 0, &delivered), ANIRA_OK);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 1)
        << "one chunk took both hops: the second chunk of the block never formed";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    const RecordCollector::Record record =
        anira_test::find_record(collector, "pre_process moved", "rt");
    EXPECT_NE(record.m_message.find("the stage: pre_process moved"), std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("by 1024 elements, the hop is 512"), std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("the stream has shifted"), std::string::npos)
        << record.m_message;
#endif
}

// A post_process that pushes nothing is topped up with zeros; one that pushes twice has
// shifted the output stream. Both are CONFIG with one record.
TEST(AbiStage, APostProcessThatPushesTheWrongCountIsRecorded) {
    const Context context;
    {
        anira_drain_log();
        RecordCollector collector;
        Probe probe;  // fills post_process, calls no default
        anira_stage_desc stage = promising_stage(&probe);
        stage.post_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(4);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_CONFIG);
        expect_same_stream(stream.m_out, std::vector<float>(stream.m_in.size(), 0.0F));
        anira_drain_log();
#ifdef ENABLE_LOGGING
        const RecordCollector::Record record =
            anira_test::find_record(collector, "post_process moved", "rt");
        EXPECT_NE(record.m_message.find("the stage: post_process moved"), std::string::npos)
            << record.m_message;
        EXPECT_NE(record.m_message.find("output tensor 0, channel 0 by 0 elements"),
                  std::string::npos)
            << record.m_message;
        EXPECT_NE(record.m_message.find("the rest is zeros"), std::string::npos)
            << record.m_message;
#endif
    }
    {
        Probe probe;
        probe.m_defaults = 2;  // two pushes of one chunk
        anira_stage_desc stage = promising_stage(&probe);
        stage.post_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.block(/*settle=*/k_hop);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_CONFIG);
    }
}

// ============================================================================================
// A failing phase
// ============================================================================================

TEST(AbiStage, FailingPreProcessDeliversZerosAtTheStreamPosition) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    Probe probe;
    probe.m_defaults = 1;
    probe.m_fail_phase = ANIRA_PHASE_PRE_PROCESS;
    probe.m_fail_call = 2;                            // the third chunk, before it pops anything
    probe.m_fail_status = ANIRA_ERROR_OUT_OF_MEMORY;  // a status without a kind bit of its own
    anira_stage_desc stage = promising_stage(&probe);
    stage.pre_process = probe_phase;
    stage.before_inference = probe_phase;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(6);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // The failed chunk is zeros where it stands, every other one is intact and in place: the
    // ring gave up the failed chunk's hop all the same.
    expect_same_stream(stream.m_out, stream.expected(/*failed=*/2));
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 6);
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_BEFORE_INFERENCE).load(), 5)
        << "the failed chunk was never submitted";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "the stage: pre_process returned", "rt"), 1U);
    EXPECT_EQ(anira_test::count_records(collector, "the hop is", "rt"), 0U)
        << "the repair of a failed phase is silent";
#endif
}

TEST(AbiStage, FailingHooksDeliverZeros) {
    const Context context;
    for (const uint32_t phase : {static_cast<uint32_t>(ANIRA_PHASE_BEFORE_INFERENCE),
                                 static_cast<uint32_t>(ANIRA_PHASE_AFTER_INFERENCE)}) {
        SCOPED_TRACE("phase " + std::to_string(phase));
        Probe probe;
        probe.m_fail_phase = phase;
        probe.m_fail_call = 1;
        probe.m_fail_status = ANIRA_ERROR_ENGINE;
        anira_stage_desc stage = promising_stage(&probe);
        stage.before_inference = probe_phase;
        stage.after_inference = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(5);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected(/*failed=*/1));
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_ENGINE);
        // A failed before_inference skips the rest of the chunk, after_inference included.
        const int after = probe.m_calls.at(ANIRA_PHASE_AFTER_INFERENCE).load();
        EXPECT_EQ(after, phase == ANIRA_PHASE_BEFORE_INFERENCE ? 4 : 5);
    }
}

TEST(AbiStage, FailingPostProcessKeepsTheStreamAligned) {
    const Context context;
    Probe probe;
    probe.m_defaults = 1;
    probe.m_fail_phase = ANIRA_PHASE_POST_PROCESS;
    probe.m_fail_call = 1;  // fails before it pushes: anira tops the ring up with zeros
    probe.m_fail_status = ANIRA_ERROR_INVALID_STATE;
    anira_stage_desc stage = promising_stage(&probe);
    stage.post_process = probe_phase;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(5);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    expect_same_stream(stream.m_out, stream.expected(/*failed=*/1));
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_INVALID_STATE);
}

// ============================================================================================
// The real-time promise: anira_stage_desc.flags against the placement
// ============================================================================================

// Under a Hard contract a filled pre_process or post_process runs on the driving thread: the
// stage must promise ANIRA_STAGE_FLAG_REALTIME_PRE_POST, else prepare is CONFIG naming the flag,
// and the handler stays unprepared. The hooks need no promise (an inference thread), and the
// promise alone, without the hooks' bit, suffices for the two host-end phases.
TEST(AbiStage, TheRealTimePromiseIsCheckedAtPrepare) {
    const Context context;
    const anira::ContractHandle contract = zeros_contract();
    {
        Probe probe;
        probe.m_defaults = 1;
        anira_stage_desc stage = promising_stage(&probe);
        stage.flags = 0;
        stage.pre_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::strstr(handler.m_err.message, "the stage: pre_process is filled"), nullptr)
            << handler.m_err.message;
        EXPECT_NE(std::strstr(handler.m_err.message, "pre_process"), nullptr)
            << handler.m_err.message;
        EXPECT_NE(std::strstr(handler.m_err.message, "ANIRA_STAGE_FLAG_REALTIME_PRE_POST"), nullptr)
            << handler.m_err.message;
        EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 0);
    }
    {
        // post_process alone, with the hooks' bit only: the wrong promise.
        Probe probe;
        probe.m_defaults = 1;
        anira_stage_desc stage = promising_stage(&probe);
        stage.flags = ANIRA_STAGE_FLAG_REALTIME_HOOKS;
        stage.post_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_CONFIG);
        EXPECT_NE(std::strstr(handler.m_err.message, "post_process"), nullptr)
            << handler.m_err.message;
    }
    {
        // The hooks alone promise nothing and run: the promise is about the driving thread.
        Probe probe;
        anira_stage_desc stage = promising_stage(&probe);
        stage.flags = 0;
        stage.before_inference = probe_phase;
        stage.after_inference = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
        EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_AFTER_INFERENCE).load(), 3);
    }
    {
        Probe probe;
        probe.m_defaults = 1;
        anira_stage_desc stage = promising_stage(&probe);
        stage.flags = ANIRA_STAGE_FLAG_REALTIME_PRE_POST;
        stage.pre_process = probe_phase;
        stage.post_process = probe_phase;
        StagedHandler handler(context, stream_model(), {stage});
        ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        Stream stream(handler.m_handler);
        stream.blocks(3);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        expect_same_stream(stream.m_out, stream.expected());
        EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
    }
}

// ============================================================================================
// The entry: anira_stage_ctx.entry and anira_handler_num_entries
// ============================================================================================

namespace {

/// What a stage saw of the entries: per phase, the entry and the block's first sample of every
/// call, in call order. The inference-thread phases write their own rows only.
struct EntryLog {
    static constexpr size_t k_capacity = 16;
    struct Row {
        std::atomic<uint32_t> m_entry{UINT32_MAX};
        std::atomic<float> m_first{-1.0F};
    };
    std::array<std::array<Row, k_capacity>, k_phases> m_rows{};
    std::array<std::atomic<size_t>, k_phases> m_calls{};
    uint32_t m_num_entries = 0;  ///< what the stage's prepare read
};

anira_status ANIRA_CALL log_entry_prepare(const anira_stage_prepare_info* info,
                                          void* user_data,
                                          void** /*out_prepared*/) {
    static_cast<EntryLog*>(user_data)->m_num_entries = anira_handler_num_entries(info->handler);
    return ANIRA_OK;
}

/// pre_process calls the default first, so that the model input holds the block; every phase
/// then takes the entry and the first sample of the tensor its phase exposes (the model is a
/// pass-through: the output holds the block too). post_process ends with the default push.
anira_status ANIRA_CALL log_entry(const anira_stage_ctx* ctx,
                                  void* /*prepared*/,
                                  void* user_data) ANIRA_NONBLOCKING {
    auto* log = static_cast<EntryLog*>(user_data);
    const uint32_t phase = ctx->phase;
    if (phase >= k_phases) { return ANIRA_ERROR_INTERNAL; }
    if (phase == ANIRA_PHASE_PRE_PROCESS) {
        const anira_status popped = anira_stage_default_pre_process(ctx);
        if (popped != ANIRA_OK) { return popped; }
    }
    const bool inputs = phase == ANIRA_PHASE_PRE_PROCESS || phase == ANIRA_PHASE_BEFORE_INFERENCE;
    anira_tensor tensor;
    const anira_status exposed = inputs ? anira_stage_input_tensor(ctx, 0, &tensor)
                                        : anira_stage_output_tensor(ctx, 0, &tensor);
    if (exposed != ANIRA_OK) { return exposed; }
    const float* data = anira_tensor_data_f32(&tensor);
    const size_t call = log->m_calls.at(phase).fetch_add(1);
    if (call < EntryLog::k_capacity) {
        EntryLog::Row& row = log->m_rows.at(phase).at(call);
        row.m_entry.store(ctx->entry);
        row.m_first.store(data != nullptr ? data[0] : -1.0F);
    }
    return phase == ANIRA_PHASE_POST_PROCESS ? anira_stage_default_post_process(ctx) : ANIRA_OK;
}

/// A per-entry scratch: pre_process pops the hop into the chunk's slot of the scratch and
/// leaves the model input alone, before_inference writes twice the slot into the model input.
/// Nothing is allocated on a per-call path: prepare sizes the scratch by the record's
/// num_entries (anira_handler_num_entries says the same).
struct Scratch {
    std::vector<std::array<float, k_hop>> m_slots;  ///< one per entry, sized at prepare
    std::atomic<int> m_broken{0};                   ///< entries at or beyond the scratch
};

anira_status ANIRA_CALL scratch_prepare(const anira_stage_prepare_info* info,
                                        void* user_data,
                                        void** /*out_prepared*/) {
    auto* scratch = static_cast<Scratch*>(user_data);
    scratch->m_slots.assign(info->num_entries, {});
    return ANIRA_OK;
}

anira_status ANIRA_CALL scratch_phase(const anira_stage_ctx* ctx,
                                      void* /*prepared*/,
                                      void* user_data) ANIRA_NONBLOCKING {
    auto* scratch = static_cast<Scratch*>(user_data);
    if (ctx->entry >= scratch->m_slots.size()) {
        scratch->m_broken.fetch_add(1);
        return ANIRA_ERROR_INTERNAL;
    }
    std::array<float, k_hop>& slot = scratch->m_slots[ctx->entry];
    if (ctx->phase == ANIRA_PHASE_PRE_PROCESS) {
        anira_role role = ANIRA_ROLE_FORCE32;
        const anira_status asked = anira_stage_input_role(ctx, 0, &role);
        if (asked != ANIRA_OK) { return asked; }
        if (role != ANIRA_ROLE_STREAMED) { return ANIRA_ERROR_CONFIG; }
        anira_ring* ring = nullptr;
        const anira_status has_ring = anira_stage_input_ring(ctx, 0, &ring);
        if (has_ring != ANIRA_OK) { return has_ring; }
        return anira_ring_pop_block(ring, 0, slot.data(), ANIRA_DTYPE_F32, k_hop) == k_hop
                   ? ANIRA_OK
                   : ANIRA_ERROR_CONFIG;
    }
    if (ctx->phase == ANIRA_PHASE_BEFORE_INFERENCE) {
        anira_tensor tensor;
        const anira_status exposed = anira_stage_input_tensor(ctx, 0, &tensor);
        if (exposed != ANIRA_OK) { return exposed; }
        float* samples = anira_tensor_data_f32(&tensor);
        if (samples == nullptr || anira_tensor_num_elements(&tensor) != k_hop) {
            return ANIRA_ERROR_CONFIG;
        }
        for (size_t n = 0; n < k_hop; ++n) { samples[n] = 2.0F * slot[n]; }
    }
    return ANIRA_OK;
}

}  // namespace

// The entry is below anira_handler_num_entries, which the stage's prepare reads (0 while
// unprepared); it is the same in all four phases of a chunk and distinct across the chunks in
// flight at once. A gated backend holds the inferences, so several chunks are in flight
// together; a chunk formed after they completed reuses one of their entries.
TEST(AbiStage, TheEntryIsStableWithinAChunkAndDistinctAcrossChunksInFlight) {
    const Context context(4);
    EntryLog log;
    std::unique_ptr<anira_test::GateBackend> gate;  // outlives the handler: declared first
    anira_stage_desc stage = promising_stage(&log);
    stage.prepare = log_entry_prepare;
    stage.pre_process = log_entry;
    stage.post_process = log_entry;
    stage.before_inference = log_entry;
    stage.after_inference = log_entry;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(anira_handler_num_entries(handler.m_handler), 0U) << "unprepared";
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    const uint32_t num_entries = anira_handler_num_entries(h);
    ASSERT_GE(num_entries, 2U);
    EXPECT_EQ(log.m_num_entries, num_entries) << "the stage's prepare read the count";
    EXPECT_EQ(anira_handler_num_entries(nullptr), 0U);

    gate = std::make_unique<anira_test::GateBackend>(h->m_inference_config);
    anira_test::attach_processor(h, *gate);
    gate->m_open.store(false);
    // One block per chunk while nothing completes: every chunk in flight has its own entry.
    // The call is ANIRA_OK while the latency's zeros last and ANIRA_MISSED after (the policy
    // fills zeros), a success either way.
    const uint32_t in_flight = std::min(num_entries, 4U);
    anira_test::FloatFace face(h);
    for (uint32_t k = 0; k < in_flight; ++k) {
        std::vector<float> block = anira_test::ramp(k + 1, k_hop);
        const std::array<float*, 1> channels{block.data()};
        size_t delivered = 0;
        EXPECT_TRUE(ANIRA_SUCCEEDED(face.process_inplace(channels.data(), k_hop, 0, &delivered)));
    }
    ASSERT_EQ(log.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), in_flight);
    std::vector<bool> taken(num_entries, false);
    for (uint32_t k = 0; k < in_flight; ++k) {
        const uint32_t entry = log.m_rows.at(ANIRA_PHASE_PRE_PROCESS).at(k).m_entry.load();
        ASSERT_LT(entry, num_entries);
        EXPECT_FALSE(taken.at(entry)) << "chunk " << k << " shares entry " << entry;
        taken.at(entry) = true;
    }
    // The gate opens: every chunk completes, and the collection (post_process, on this thread)
    // runs as the available count is read.
    gate->m_open.store(true);
    const auto start = std::chrono::steady_clock::now();
    while (log.m_calls.at(ANIRA_PHASE_POST_PROCESS).load() < in_flight) {
        anira_test::available(h);  // collects the completed chunks
        ASSERT_LT(std::chrono::steady_clock::now(),
                  start + std::chrono::seconds(anira_test::k_wait_s))
            << "timeout: " << log.m_calls.at(ANIRA_PHASE_POST_PROCESS).load() << " of " << in_flight
            << " chunks collected";
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // Chunk by chunk (told apart by the block's first sample): one entry in all four phases.
    for (uint32_t k = 0; k < in_flight; ++k) {
        const EntryLog::Row& pre = log.m_rows.at(ANIRA_PHASE_PRE_PROCESS).at(k);
        const float first = pre.m_first.load();
        const uint32_t entry = pre.m_entry.load();
        for (const uint32_t phase : {static_cast<uint32_t>(ANIRA_PHASE_BEFORE_INFERENCE),
                                     static_cast<uint32_t>(ANIRA_PHASE_AFTER_INFERENCE),
                                     static_cast<uint32_t>(ANIRA_PHASE_POST_PROCESS)}) {
            SCOPED_TRACE("chunk " + std::to_string(k) + ", phase " + std::to_string(phase));
            bool found = false;
            for (uint32_t i = 0; i < in_flight; ++i) {
                const EntryLog::Row& row = log.m_rows.at(phase).at(i);
                if (row.m_first.load() != first) { continue; }
                found = true;
                EXPECT_EQ(row.m_entry.load(), entry);
            }
            EXPECT_TRUE(found) << "the chunk was not seen in this phase";
        }
    }
    // A chunk formed after the others completed reuses one of their entries.
    {
        std::vector<float> block = anira_test::ramp(in_flight + 1, k_hop);
        const std::array<float*, 1> channels{block.data()};
        size_t delivered = 0;
        EXPECT_TRUE(ANIRA_SUCCEEDED(face.process_inplace(channels.data(), k_hop, 0, &delivered)));
        ASSERT_EQ(log.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), in_flight + 1U);
        const uint32_t entry = log.m_rows.at(ANIRA_PHASE_PRE_PROCESS).at(in_flight).m_entry.load();
        ASSERT_LT(entry, num_entries);
        EXPECT_TRUE(taken.at(entry)) << "reused after the chunk completed";
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
}

// Per-chunk scratch without an allocation: pre_process pops the window into the chunk's entry
// of a scratch the stage sized in its prepare, before_inference transforms it from there into
// the model input, on another thread. The output is exact, so the entry pre_process saw is the
// entry before_inference saw, chunk by chunk.
TEST(AbiStage, PerChunkScratchByEntry) {
    const Context context;
    Scratch scratch;
    anira_stage_desc stage = promising_stage(&scratch);
    stage.prepare = scratch_prepare;
    stage.pre_process = scratch_phase;
    stage.before_inference = scratch_phase;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(scratch.m_slots.size(), anira_handler_num_entries(handler.m_handler));
    Stream stream(handler.m_handler);
    stream.blocks(6);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::vector<float> wanted = stream.expected();
    for (float& sample : wanted) { sample *= 2.0F; }
    expect_same_stream(stream.m_out, wanted);
    EXPECT_EQ(scratch.m_broken.load(), 0);
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_OK);
}

// ============================================================================================
// A pipeline without a stage is the 2.x default processor
// ============================================================================================

// Differential, on a window longer than its hop (the receptive-field fill) and a backend whose
// every output sample needs the history: the C handler, whose session sees the processor, against
// a 2.x handler over the bridged configuration, whose session sees the default processor.
TEST(AbiStage, NoStageEqualsTheV2Default) {
    const Context context;
    const auto window = static_cast<int64_t>(2 * k_hop);
    const ModelConfig model = stream_model(window, static_cast<int64_t>(k_hop));
    const anira::ContractHandle contract = zeros_contract();

    Handler c(context, model);
    ASSERT_EQ(c.prepare(contract), ANIRA_OK) << c.m_err.message;
    WindowSumBackend c_backend(c.m_handler->m_inference_config);
    anira_test::attach_processor(c.m_handler, c_backend);
    const anira_test::DestroyFirst c_guard(c);

    anira::InferenceConfig config_2x = anira::v3compat::to_inference_config(model, contract);
    anira::PrePostProcessor pp(config_2x);
    WindowSumBackend v_backend(config_2x);
    {
        anira::CoreConfig core_config(2, anira::WaitStrategy::SpinBackoff, anira::LogLevel::Error);
        core_config.m_log.m_drain = anira::LogDrain::Manual;
        anira::InferenceHandler v2(pp, config_2x, v_backend, core_config);
        v2.prepare(
            anira::HostConfig(static_cast<float>(k_hop), static_cast<float>(anira_test::k_rate)));
        v2.set_inference_backend(anira::InferenceBackend::CUSTOM);
        ASSERT_EQ(anira_handler_get_latency(c.m_handler, 0), v2.get_latency(0));

        anira_test::FloatFace face(c.m_handler);
        for (size_t k = 1; k <= 12; ++k) {
            std::vector<float> c_block = anira_test::ramp(k, k_hop);
            std::vector<float> v_block = c_block;
            const size_t prev_c = anira_test::available(c.m_handler);
            const size_t prev_v = v2.get_available_samples(0);
            const std::array<float*, 1> c_channels{c_block.data()};
            const std::array<float*, 1> v_channels{v_block.data()};
            size_t delivered = 0;
            ASSERT_EQ(face.process_inplace(c_channels.data(), k_hop, 0, &delivered), ANIRA_OK);
            anira_test::wait_for_block(c.m_handler, prev_c);
            EXPECT_EQ(v2.process(v_channels.data(), k_hop), delivered);
            anira_test::wait_for_block(v2, prev_v);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            anira_test::expect_same_block(c_block, v_block, k);
        }
    }
    EXPECT_EQ(anira_handler_rt_error(c.m_handler), ANIRA_OK);
}

// ============================================================================================
// What a stage changes at prepare
// ============================================================================================

// A stage that declares "model:entry" consumes the entry extension of the custom row, which no
// adapter of anira reads: the handler is created, and the plan's extension row's consumer reads
// "stage".
TEST(AbiStage, ConsumedKindsJoinTheWalk) {
    const Context context;
    ModelConfig model = stream_model();
    model.model_ext(0, anira::ext::Entry{"forward"});
    {
        const StagedHandler refused(context, model, {});
        EXPECT_EQ(refused.m_create_status, ANIRA_ERROR_EXTENSION_UNCONSUMED)
            << refused.m_err.message;
    }
    const std::array<const char*, 1> kinds{"model:entry"};
    anira_stage_desc stage = promising_stage(nullptr);
    stage.consumed_kinds = kinds.data();
    stage.num_consumed_kinds = 1;
    StagedHandler handler(context, model, {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    const anira_plan_report* report = anira_handler_plan_report(handler.m_handler);
    ASSERT_NE(report, nullptr);
    uint32_t count = 0;
    ASSERT_EQ(anira_plan_report_exts(report, 0, sizeof(anira_plan_ext), &count, nullptr), ANIRA_OK);
    ASSERT_EQ(count, 1U);
    anira_plan_ext row = ANIRA_PLAN_EXT_INIT;
    ASSERT_EQ(anira_plan_report_exts(report, 0, sizeof(anira_plan_ext), &count, &row), ANIRA_OK);
    EXPECT_STREQ(row.kind, "entry");
    EXPECT_STREQ(row.consumer, "stage");
    EXPECT_STREQ(row.host, "model 0");
}

// A ring dtype that differs from its spec's dtype is CONFIG at prepare, unless a stage fills
// the phase that moves that ring.
TEST(AbiStage, ARingOfAnotherDtypeNeedsAStageThatFillsThePhase) {
    const Context context;
    anira::ContractHandle contract = zeros_contract();
    contract.hard_ring_dtype("in", ANIRA_DTYPE_I16);

    Handler refused(context, stream_model());
    EXPECT_EQ(refused.prepare(contract), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::strstr(refused.m_err.message, "a stage that fills pre_process"), nullptr)
        << refused.m_err.message;

    // A stage on the other side does not help: post_process moves the output rings.
    Probe probe;
    probe.m_defaults = 1;
    anira_stage_desc post_only = promising_stage(&probe);
    post_only.post_process = probe_phase;
    StagedHandler wrong_side(context, stream_model(), {post_only});
    EXPECT_EQ(wrong_side.prepare(contract), ANIRA_ERROR_CONFIG);

    auto scratch = std::make_unique<std::array<int16_t, k_hop>>();
    anira_stage_desc convert = promising_stage(scratch.get());
    convert.pre_process = int16_to_float;
    StagedHandler handler(context, stream_model(), {convert});
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    const size_t latency = anira_handler_get_latency(h, 0);

    std::vector<float> out_stream;
    std::vector<int16_t> in_stream;
    for (size_t k = 0; k < 4; ++k) {
        std::array<int16_t, k_hop> in{};
        for (size_t n = 0; n < k_hop; ++n) {
            in.at(n) = static_cast<int16_t>(static_cast<int>((k * k_hop) + n) - 1000);
        }
        in_stream.insert(in_stream.end(), in.begin(), in.end());
        std::array<float, k_hop> out{};
        const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_hop)};
        anira_tensor in_tensor{};
        anira_tensor out_tensor{};
        anira_tensor_init_host(&in_tensor, in.data(), ANIRA_DTYPE_I16, 2, shape.data());
        anira_tensor_init_host(&out_tensor, out.data(), ANIRA_DTYPE_F32, 2, shape.data());
        const size_t prev = anira_test::available(h);
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process(h, &in_tensor, 0, &out_tensor, 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_hop);
        anira_test::wait_for_block(h, prev);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        out_stream.insert(out_stream.end(), out.begin(), out.end());
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    for (size_t n = 0; n < out_stream.size(); ++n) {
        const float wanted =
            n < latency ? 0.0F : static_cast<float>(in_stream.at(n - latency)) / 32768.0F;
        ASSERT_EQ(out_stream.at(n), wanted) << "sample " << n;
    }
}

// ============================================================================================
// The stages in front of an engine
// ============================================================================================

#if defined(USE_ONNXRUNTIME) || defined(USE_LIBTORCH) || defined(USE_EXECUTORCH) || \
    defined(USE_TFLITE)

namespace {

/// One candidate: the engine, its default provider.
std::array<anira_backend_id, 1> one_engine(anira_engine engine) {
    return {anira_backend_id{.struct_size = sizeof(anira_backend_id),
                             .engine = static_cast<uint32_t>(engine),
                             .provider = ANIRA_PROVIDER_DEFAULT,
                             .engine_id = nullptr}};
}

/// The engine of the plan the handler runs.
uint32_t plan_engine(const anira_handler* handler) {
    return handler->m_report.m_plans.at(anira_handler_get_plan(handler)).engine;
}

/// The Static gain of the bundled gain model, through the handler's store.
void set_gain(anira_handler* handler, float gain_value) {
    const std::array<int64_t, 1> gain_shape{1};
    anira_tensor gain_tensor{};
    anira_tensor_init_host(&gain_tensor, &gain_value, ANIRA_DTYPE_F32, 1, gain_shape.data());
    ASSERT_EQ(anira_handler_set_static_input(handler, 1, &gain_tensor), ANIRA_OK);
}

}  // namespace

// The converting stage of ARingOfAnotherDtypeNeedsAStageThatFillsThePhase in front of a real
// engine: the bundled gain model on every engine of this build the file runs on, its input
// ring int16, the host pushing int16 blocks, the stage writing float32 into the model input
// through the context accessors, the engine applying a gain of one half. Every output sample
// is exact: the word over 32768, halved. TFLite included: its export of the gain model orders
// the outputs [peak, processed_data] in the file, and the adapter runs the signature, whose
// runner lists the keys output_0 (the stream) and output_1 (the peak) in that order, so output
// slot 0 reads the stream; a swapped pair would fail the shape check at prepare.
TEST(AbiStage, Int16RingWithAConvertingStageOnEveryEngine) {
    const Context context;
    for (const anira_engine engine : anira_test::oracle_engines()) {
        SCOPED_TRACE("engine " + std::to_string(static_cast<unsigned>(engine)));
        auto scratch = std::make_unique<std::array<int16_t, k_hop>>();
        anira_stage_desc convert = promising_stage(scratch.get());
        convert.pre_process = int16_to_float;
        const std::array<anira_backend_id, 1> candidates = one_engine(engine);
        const ModelConfig model = ModelConfig::from_file(k_gain_model_json);
        StagedHandler handler(context, model, {convert}, candidates);
        ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
        anira::ContractHandle contract = anira_test::file_contract(k_gain_contract_json, k_hop);
        contract.hard_ring_dtype("audio_in", ANIRA_DTYPE_I16);
        // The file's BYPASS needs both rings in one dtype (nothing converts): ZEROS does not.
        contract.hard_on_miss(ANIRA_MISS_ZEROS);
        ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        ASSERT_EQ(plan_engine(h), static_cast<uint32_t>(engine));
        ASSERT_NO_FATAL_FAILURE(set_gain(h, 0.5F));
        const size_t latency = anira_handler_get_latency(h, 0);

        std::vector<float> out_stream;
        std::vector<int16_t> in_stream;
        for (size_t k = 0; k < 4; ++k) {
            std::array<int16_t, k_hop> in{};
            for (size_t n = 0; n < k_hop; ++n) {
                in.at(n) = static_cast<int16_t>(static_cast<int>((k * k_hop) + n) - 1000);
            }
            in_stream.insert(in_stream.end(), in.begin(), in.end());
            std::array<float, k_hop> out{};
            const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_hop)};
            anira_tensor in_tensor{};
            anira_tensor out_tensor{};
            anira_tensor_init_host(&in_tensor, in.data(), ANIRA_DTYPE_I16, 2, shape.data());
            anira_tensor_init_host(&out_tensor, out.data(), ANIRA_DTYPE_F32, 2, shape.data());
            const size_t prev = anira_test::available(h);
            size_t delivered = 0;
            ASSERT_EQ(anira_handler_process(h, &in_tensor, 0, &out_tensor, 0, &delivered),
                      ANIRA_OK);
            EXPECT_EQ(delivered, k_hop);
            anira_test::wait_for_block(h, prev);
            ASSERT_FALSE(::testing::Test::HasFatalFailure());
            out_stream.insert(out_stream.end(), out.begin(), out.end());
        }
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
        for (size_t n = 0; n < out_stream.size(); ++n) {
            const float wanted =
                n < latency ? 0.0F
                            : 0.5F * (static_cast<float>(in_stream.at(n - latency)) / 32768.0F);
            ASSERT_EQ(out_stream.at(n), wanted) << "sample " << n;
        }
    }
}

#endif  // an engine

#ifdef USE_LIBTORCH

namespace {

/// What the stage of TensorsKeepTheStructsMemoryAcrossLibTorchInferences recorded per
/// chunk: the memory behind model input 0 in pre_process and behind model output 0 in
/// post_process. Both phases run on the driving thread, in lockstep with the stream, so the
/// arrays are plain and the test reads them after the run.
struct Pointers {
    static constexpr size_t k_capacity = 64;
    std::array<const void*, k_capacity> m_inputs{};
    std::array<const void*, k_capacity> m_outputs{};
    size_t m_pre = 0;
    size_t m_post = 0;
};

anira_status ANIRA_CALL record_then_default(const anira_stage_ctx* ctx,
                                            void* /*prepared*/,
                                            void* user_data) ANIRA_NONBLOCKING {
    auto* pointers = static_cast<Pointers*>(user_data);
    anira_tensor tensor;
    if (ctx->phase == ANIRA_PHASE_PRE_PROCESS) {
        if (anira_stage_input_tensor(ctx, 0, &tensor) != ANIRA_OK) { return ANIRA_ERROR_INTERNAL; }
        if (pointers->m_pre < Pointers::k_capacity) {
            pointers->m_inputs.at(pointers->m_pre) = tensor.handle.host.ptr;
        }
        ++pointers->m_pre;
        return anira_stage_default_pre_process(ctx);
    }
    if (anira_stage_output_tensor(ctx, 0, &tensor) != ANIRA_OK) { return ANIRA_ERROR_INTERNAL; }
    if (pointers->m_post < Pointers::k_capacity) {
        pointers->m_outputs.at(pointers->m_post) = tensor.handle.host.ptr;
    }
    ++pointers->m_post;
    return anira_stage_default_post_process(ctx);
}

}  // namespace

// The LibTorch adapter binds the struct's memory through the descriptors (a from_blob view
// per call, LibTorchAdapter.cpp) and swaps nothing, so a struct holds one input block for its
// whole life: two uses of one struct (told apart by its output tensor) return the same input
// pointer, and the default fill through that pointer is what the engine read, since the stream
// is intact. The accessor still builds the tensor from the struct's descriptor of the moment,
// so a descriptor that did move would be followed.
TEST(AbiStage, TensorsKeepTheStructsMemoryAcrossLibTorchInferences) {
    const Context context;
    Pointers pointers;
    anira_stage_desc stage = promising_stage(&pointers);
    stage.pre_process = record_then_default;
    stage.post_process = record_then_default;
    const std::array<anira_backend_id, 1> candidates = one_engine(ANIRA_ENGINE_LIBTORCH);
    const ModelConfig model = ModelConfig::from_file(k_gain_model_json);
    StagedHandler handler(context, model, {stage}, candidates);
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(anira_test::file_contract(k_gain_contract_json, k_hop)), ANIRA_OK)
        << handler.m_err.message;
    anira_handler* h = handler.m_handler;
    ASSERT_EQ(plan_engine(h), static_cast<uint32_t>(ANIRA_ENGINE_LIBTORCH));
    ASSERT_NO_FATAL_FAILURE(set_gain(h, 1.0F));  // the identity: the stream oracle applies
    const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(h);
    ASSERT_NE(session, nullptr);
    const auto& queue = session->m_inference_queue;
    const size_t structs = queue.size();
    ASSERT_GT(structs, 0U);
    const size_t chunks = (2 * structs) + 2;  // every struct that runs comes round again
    ASSERT_LE(chunks, Pointers::k_capacity);

    Stream stream(h);
    stream.blocks(chunks);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);
    expect_same_stream(stream.m_out, stream.expected());
    ASSERT_EQ(pointers.m_pre, chunks);
    ASSERT_EQ(pointers.m_post, chunks);

    // The struct of every chunk, by the memory of its output tensor; then the input pointer of
    // a chunk against the previous chunk of the same struct.
    std::vector<size_t> last_use(structs, SIZE_MAX);
    size_t pairs = 0;
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const auto found = std::ranges::find_if(queue, [&](const auto& candidate) {
            return pointers.m_outputs.at(chunk) ==
                   candidate->m_tensor_output_data.at(0).get_read_pointer(0);
        });
        ASSERT_NE(found, queue.end()) << "chunk " << chunk << " ran on no struct of the session";
        const auto index = static_cast<size_t>(found - queue.begin());
        if (last_use.at(index) != SIZE_MAX) {
            EXPECT_EQ(pointers.m_inputs.at(chunk), pointers.m_inputs.at(last_use.at(index)))
                << "chunks " << last_use.at(index) << " and " << chunk << " on struct " << index
                << " saw two input blocks: the struct's memory was swapped away";
            ++pairs;
        }
        last_use.at(index) = chunk;
    }
    EXPECT_GE(pairs, chunks - structs) << "every chunk beyond a struct's first use is a pair";
}

#endif  // USE_LIBTORCH
