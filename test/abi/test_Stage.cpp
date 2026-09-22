// anira/abi/stage.h: the ring accessors, the stage context and its six accessors, the descriptor
// and its carrier, the default bodies and the chain a C-created handler runs them in.
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
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

/// anira_test::Handler's twin with a stage chain: the stages go in before the handler is
/// created, and the pipeline can be destroyed ahead of the handler.
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

anira_stage_desc named_stage(const char* name, void* user_data) {
    anira_stage_desc stage = ANIRA_STAGE_DESC_INIT;
    stage.name = name;
    stage.user_data = user_data;
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

/// What one stage saw and what it is told to do. The inference-thread phases write their own
/// entries only, and the test reads after the block's inference was collected.
struct Probe {
    std::array<std::atomic<int>, k_phases> m_calls{};
    std::array<anira_stage_ctx, k_phases> m_ctx{};
    // What the six context accessors answered, per phase and slot: every phase asks all six
    // for every slot, so the record is the table of what a phase exposes.
    std::array<std::array<anira_role, k_tensors>, k_phases> m_input_roles{};
    std::array<std::array<anira_role, k_tensors>, k_phases> m_output_roles{};
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

anira_status ANIRA_CALL probe_phase(const anira_stage_ctx* ctx, void* user_data) ANIRA_NONBLOCKING {
    auto* probe = static_cast<Probe*>(user_data);
    const uint32_t phase = ctx->phase;
    if (phase >= k_phases) { return ANIRA_ERROR_INTERNAL; }
    const int call = probe->m_calls.at(phase).fetch_add(1);
    probe->m_ctx.at(phase) = *ctx;
    probe->m_thread.at(phase) = std::this_thread::get_id();
    if (probe->m_order != nullptr) { probe->m_seen_order.at(phase) = probe->m_order->fetch_add(1); }
    for (uint32_t slot = 0; slot < ctx->num_inputs && slot < k_tensors; ++slot) {
        probe->m_input_roles.at(phase).at(slot) = anira_stage_input_role(ctx, slot);
        probe->m_input_rings.at(phase).at(slot) = anira_stage_input_ring(ctx, slot);
        probe->m_input_status.at(phase).at(slot) =
            anira_stage_input_tensor(ctx, slot, &probe->m_inputs.at(phase).at(slot));
    }
    for (uint32_t slot = 0; slot < ctx->num_outputs && slot < k_tensors; ++slot) {
        probe->m_output_roles.at(phase).at(slot) = anira_stage_output_role(ctx, slot);
        probe->m_output_rings.at(phase).at(slot) = anira_stage_output_ring(ctx, slot);
        probe->m_output_status.at(phase).at(slot) =
            anira_stage_output_tensor(ctx, slot, &probe->m_outputs.at(phase).at(slot));
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

struct Lifetime {
    int m_prepared = 0;
    int m_released = 0;
    anira_handler* m_handler = nullptr;
    const anira_plan_report* m_report = nullptr;
    uint32_t m_plans = 0;
    unsigned int m_latency = 0;
    anira_status m_return = ANIRA_OK;
};

anira_status ANIRA_CALL on_prepare(anira_handler* handler,
                                   const anira_plan_report* report,
                                   void* user_data) {
    auto* lifetime = static_cast<Lifetime*>(user_data);
    ++lifetime->m_prepared;
    lifetime->m_handler = handler;
    lifetime->m_report = report;
    lifetime->m_plans = anira_plan_report_num_plans(report);
    lifetime->m_latency = anira_handler_get_latency(handler, 0);  // a getter: the handler answers
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
                                       void* user_data) ANIRA_NONBLOCKING {
    auto* scratch = static_cast<std::array<int16_t, k_hop>*>(user_data);
    if (anira_stage_input_role(ctx, 0) != ANIRA_ROLE_STREAMED) { return ANIRA_ERROR_CONFIG; }
    anira_ring* ring = anira_stage_input_ring(ctx, 0);
    anira_tensor tensor;
    const anira_status exposed = anira_stage_input_tensor(ctx, 0, &tensor);
    if (exposed != ANIRA_OK) { return exposed; }
    float* samples = anira_tensor_data_f32(&tensor);
    if (ring == nullptr || samples == nullptr || anira_ring_dtype(ring) != ANIRA_DTYPE_I16) {
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
/// `shape` on each side, both over `ring`, the memory of the model end owned here. What the
/// chain fills per call, for the default bodies and the context accessors without a handler.
struct HandBuiltCtx {
    HandBuiltCtx(uint32_t phase, anira::RingBuffer& ring, std::vector<int64_t> shape) : m_ports(1) {
        size_t elements = 1;
        for (const int64_t extent : shape) { elements *= static_cast<size_t>(extent); }
        m_shapes.push_back(std::move(shape));
        m_buffers.emplace_back(1, elements);
        m_buffers[0].clear();
        std::get<anira::capi::StreamPort>(m_ports[0]).m_ring = &ring;
        m_frame.m_input_ports = &m_ports;
        m_frame.m_output_ports = &m_ports;
        m_frame.m_input_shapes = &m_shapes;
        m_frame.m_output_shapes = &m_shapes;
        expose(phase);
        m_ctx.num_inputs = 1;
        m_ctx.num_outputs = 1;
        m_ctx.frame = &m_frame;
    }
    HandBuiltCtx(const HandBuiltCtx&) = delete;  // the ctx names this object's frame
    HandBuiltCtx& operator=(const HandBuiltCtx&) = delete;
    HandBuiltCtx(HandBuiltCtx&&) = delete;
    HandBuiltCtx& operator=(HandBuiltCtx&&) = delete;
    ~HandBuiltCtx() = default;

    /// The phase, and the buffers of the side that phase exposes, as the chain's entry points
    /// fill them.
    void expose(uint32_t phase) {
        const bool inputs =
            phase == ANIRA_PHASE_PRE_PROCESS || phase == ANIRA_PHASE_BEFORE_INFERENCE;
        m_ctx.phase = phase;
        m_frame.m_model_inputs = inputs ? &m_buffers : nullptr;
        m_frame.m_model_outputs = inputs ? nullptr : &m_buffers;
    }

    float* data() { return m_buffers[0].get_write_pointer(0); }

    std::vector<anira::capi::Port> m_ports;
    std::vector<std::vector<int64_t>> m_shapes;
    std::vector<anira::BufferF> m_buffers;
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
    anira_stage_desc stage = named_stage("refused", &lifetime);
    stage.release = on_release;

    EXPECT_EQ(anira_pipeline_add_stage(nullptr, &stage, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, nullptr, &err), ANIRA_ERROR_INVALID_ARGUMENT);

    anira_stage_desc bad = stage;
    bad.struct_size = static_cast<uint32_t>(offsetof(anira_stage_desc, user_data));
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "struct_size"), nullptr) << err.message;

    bad = stage;
    bad.reserved = 1;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_NE(std::strstr(err.message, "reserved"), nullptr) << err.message;

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

    bad = stage;
    bad.domain_out = ANIRA_DOMAIN_GL_BUFFER;
    EXPECT_EQ(anira_pipeline_add_stage(pipeline, &bad, &err), ANIRA_ERROR_NOT_SUPPORTED);
    EXPECT_NE(std::strstr(err.message, "host stage"), nullptr) << err.message;

    // A refused call creates no carrier: nothing is released, now or at the destroy.
    anira_pipeline_destroy(pipeline);
    EXPECT_EQ(lifetime.m_released, 0);
}

// A caller compiled against a shorter header hands over the three leading slots only: the
// rest reads as ANIRA_STAGE_DESC_INIT, so the stage has no callback and no name.
TEST(AbiStage, AShortDescriptorReadsAsTheDefaults) {
    const Context context;
    Lifetime lifetime;
    anira_stage_desc stage = named_stage("never read", &lifetime);
    stage.prepare = on_prepare;  // beyond the struct_size below: not read
    stage.struct_size =
        static_cast<uint32_t>(offsetof(anira_stage_desc, user_data) + sizeof(void*));
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.m_create_status, ANIRA_OK) << handler.m_err.message;
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(lifetime.m_prepared, 0);
    ASSERT_EQ(handler.m_handler->m_pipeline.m_stages.size(), 1U);
    EXPECT_STREQ(handler.m_handler->m_pipeline.m_stages[0]->name(), "stage#0");
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
        // The name and the descriptor die with this scope: the carrier copied both.
        std::string name = "lifetime";
        anira_stage_desc stage = named_stage(name.c_str(), &lifetime);
        stage.prepare = on_prepare;
        stage.release = on_release;
        ASSERT_EQ(anira_pipeline_add_stage(pipeline, &stage, &err), ANIRA_OK) << err.message;
        name.assign(name.size(), 'x');
    }

    // Two handlers and the pipeline share one carrier.
    anira_handler* first = nullptr;
    anira_handler* second = nullptr;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &first, &err), ANIRA_OK)
        << err.message;
    ASSERT_EQ(anira_handler_create(context.m_context, pipeline, &second, &err), ANIRA_OK)
        << err.message;
    EXPECT_STREQ(first->m_pipeline.m_stages.at(0)->name(), "lifetime");
    EXPECT_EQ(first->m_pipeline.m_stages.at(0).get(), second->m_pipeline.m_stages.at(0).get());
    EXPECT_EQ(lifetime.m_prepared, 0) << "prepare belongs to anira_handler_prepare";

    const anira::ContractHandle contract = zeros_contract();
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(lifetime.m_prepared, 1);
    EXPECT_EQ(lifetime.m_handler, first);
    EXPECT_EQ(lifetime.m_report, anira_handler_plan_report(first));
    EXPECT_EQ(lifetime.m_plans, 1U);
    EXPECT_EQ(lifetime.m_latency, anira_handler_get_latency(first, 0));
    EXPECT_GT(lifetime.m_latency, 0U) << "the handler counts as prepared while the stage prepares";
    ASSERT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_OK) << err.message;
    EXPECT_EQ(lifetime.m_prepared, 2) << "once per prepare";

    // A stage that refuses fails the prepare with its status, by name, and the handler is
    // unprepared afterwards.
    lifetime.m_return = ANIRA_ERROR_MODEL_LOAD;
    EXPECT_EQ(anira_handler_prepare(first, contract.native(), &err), ANIRA_ERROR_MODEL_LOAD);
    EXPECT_NE(std::strstr(err.message, "stage 'lifetime'"), nullptr) << err.message;
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
// The context per phase
// ============================================================================================

TEST(AbiStage, CtxPerPhase) {
    const Context context;
    Probe probe;
    probe.m_defaults = 1;  // the stage owns pre_process and post_process and calls the defaults
    anira_stage_desc stage = named_stage("probe", &probe);
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
        EXPECT_EQ(ctx.reserved, 0U);
        // The frame is anira's; the reserved slots are NULL, their high halves included.
        EXPECT_NE(ctx.frame_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr0_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr1_bits, 0U);
        EXPECT_EQ(ctx.reserved_ptr2_bits, 0U);
        // What a phase exposes, asked through the six accessors for every slot: the role in
        // every phase; an input ring in pre_process and an output ring in post_process, the
        // session's own for the Streamed tensor and NULL for the Static one; the model's inputs
        // in pre_process and before_inference, its outputs in the two others, of either role.
        // A tensor outside its phase is ANIRA_ERROR_INVALID_STATE and an all-zero record.
        // Nothing of it is recorded: anira_handler_rt_error read ANIRA_OK above.
        const bool pre = phase == ANIRA_PHASE_PRE_PROCESS;
        const bool post = phase == ANIRA_PHASE_POST_PROCESS;
        const bool inputs = pre || phase == ANIRA_PHASE_BEFORE_INFERENCE;
        EXPECT_EQ(probe.m_input_rings.at(phase).at(0),
                  pre ? session->m_send_buffer.data() : nullptr);
        EXPECT_EQ(probe.m_output_rings.at(phase).at(0),
                  post ? session->m_receive_buffer.data() : nullptr);
        EXPECT_EQ(probe.m_input_rings.at(phase).at(1), nullptr);
        EXPECT_EQ(probe.m_output_rings.at(phase).at(1), nullptr);
        for (size_t slot = 0; slot < 2; ++slot) {
            SCOPED_TRACE("slot " + std::to_string(slot));
            const anira_role role = slot == 0 ? ANIRA_ROLE_STREAMED : ANIRA_ROLE_STATIC;
            EXPECT_EQ(probe.m_input_roles.at(phase).at(slot), role);
            EXPECT_EQ(probe.m_output_roles.at(phase).at(slot), role);
            EXPECT_EQ(probe.m_input_status.at(phase).at(slot),
                      inputs ? ANIRA_OK : ANIRA_ERROR_INVALID_STATE);
            EXPECT_EQ(probe.m_output_status.at(phase).at(slot),
                      inputs ? ANIRA_ERROR_INVALID_STATE : ANIRA_OK);
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
    std::atomic<uint32_t> m_role{0};
    std::atomic<int32_t> m_status{ANIRA_OK};
    std::atomic<uint32_t> m_dtype{1};
    std::atomic<int> m_rings{0};
};

anira_status ANIRA_CALL ask_beyond(const anira_stage_ctx* ctx, void* user_data) ANIRA_NONBLOCKING {
    auto* beyond = static_cast<Beyond*>(user_data);
    const uint32_t slot = ctx->num_inputs;  // one beyond the last
    beyond->m_role.store(static_cast<uint32_t>(anira_stage_input_role(ctx, slot)));
    if (anira_stage_input_ring(ctx, slot) != nullptr) { beyond->m_rings.fetch_add(1); }
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
    anira_stage_desc stage = named_stage("asks-beyond", &beyond);
    stage.before_inference = ask_beyond;
    StagedHandler handler(context, stream_model(), {stage});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(3);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    expect_same_stream(stream.m_out, stream.expected());
    EXPECT_EQ(beyond.m_role.load(), static_cast<uint32_t>(ANIRA_ROLE_FORCE32));
    EXPECT_EQ(beyond.m_rings.load(), 0);
    EXPECT_EQ(beyond.m_status.load(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(beyond.m_dtype.load(), 0U) << "an all-zero record";
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_INVALID_ARGUMENT);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "is out of range", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "is out of range", "rt");
    EXPECT_NE(record.m_message.find("anira_stage_input_role: stage 'asks-beyond': slot 1 is out "
                                    "of range, the model has 1 input tensors"),
              std::string::npos)
        << record.m_message;
#endif
}

namespace {

// A stage's prepare may call the two Static entries: no driver thread runs during prepare, and
// the store is the handler's, so the value is what the first inference sees.
anira_status ANIRA_CALL set_gain_in_prepare(anira_handler* handler,
                                            const anira_plan_report* /*report*/,
                                            void* /*user_data*/) {
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
    anira_stage_desc stage = named_stage("sets-gain", &probe);
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
    owner.m_stage = "accessor-test";

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
    // One record per kind, naming the entry and the running stage.
    EXPECT_EQ(anira_test::count_records(collector, "nothing converts", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "nothing converts", "rt");
    EXPECT_NE(record.m_message.find("anira_ring_push_block: stage 'accessor-test'"),
              std::string::npos)
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

// The refusals of the six accessors, and what is not one: a NULL ring and a tensor outside its
// phase are answers and record nothing; a slot out of range and a NULL out are recorded, one
// record per kind naming the entry and the running stage.
TEST(AbiStage, ContextAccessorsAnswerAndRefuse) {
    const Context context;  // the real-time log queue is the core's
    anira_drain_log();
    RecordCollector collector;
    anira::RtLatch latch;
    anira::RingBuffer ring;
    ASSERT_TRUE(ring.initialize_with_positions(1, 16, ANIRA_DTYPE_F32));
    ring.set_owner(4, nullptr);
    HandBuiltCtx built(ANIRA_PHASE_PRE_PROCESS, ring, {1, 1, 4});
    built.m_frame.m_rt = &latch;
    built.m_frame.m_stage = "asker";
    const anira_stage_ctx& ctx = built.m_ctx;

    // pre_process: the role, the input ring, the model's input; no output ring, no output.
    EXPECT_EQ(anira_stage_input_role(&ctx, 0), ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_output_role(&ctx, 0), ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0), &ring);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0), nullptr);
    anira_tensor tensor;
    ASSERT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(tensor.domain, static_cast<uint32_t>(ANIRA_DOMAIN_HOST));
    EXPECT_EQ(tensor.dtype, ANIRA_DTYPE_F32);
    ASSERT_EQ(tensor.ndim, 3U);
    EXPECT_EQ(tensor.shape[2], 4);
    EXPECT_EQ(tensor.release, nullptr) << "borrowed";
    EXPECT_EQ(anira_tensor_data_f32(&tensor), built.data());
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_TRUE(all_zero(tensor)) << "an all-zero record";
    EXPECT_EQ(tensor.dtype, 0U);
    EXPECT_EQ(anira_tensor_data_f32(&tensor), nullptr);

    // The other phases: the role stays, the rings and the tensors follow the phase.
    built.expose(ANIRA_PHASE_BEFORE_INFERENCE);
    EXPECT_EQ(anira_stage_input_role(&ctx, 0), ANIRA_ROLE_STREAMED);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0), nullptr);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0), nullptr);
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    built.expose(ANIRA_PHASE_AFTER_INFERENCE);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0), nullptr);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0), nullptr);
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_OK);
    built.expose(ANIRA_PHASE_POST_PROCESS);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 0), nullptr);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 0), &ring);
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 0, &tensor), ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, &tensor), ANIRA_OK);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK) << "none of these answers is a refusal";

    // A slot at or beyond the side's count: no role, no ring, INVALID_ARGUMENT. Every refusal
    // is recorded; the first of its kind is the one that is logged, the others are counted.
    EXPECT_EQ(anira_stage_input_role(&ctx, 1), ANIRA_ROLE_FORCE32);
    EXPECT_EQ(latch.rt_error(), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.m_suppressed.load(), 0U);
    EXPECT_EQ(anira_stage_output_role(&ctx, 1), ANIRA_ROLE_FORCE32);
    EXPECT_EQ(latch.m_suppressed.load(), 1U);
    EXPECT_EQ(anira_stage_input_ring(&ctx, 7), nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 2U);
    EXPECT_EQ(anira_stage_output_ring(&ctx, 7), nullptr);
    EXPECT_EQ(latch.m_suppressed.load(), 3U);
    std::memset(&tensor, 0xff, sizeof(tensor));
    EXPECT_EQ(anira_stage_input_tensor(&ctx, 1, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_TRUE(all_zero(tensor));
    EXPECT_EQ(latch.m_suppressed.load(), 4U);
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 1, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.m_suppressed.load(), 5U);
    // A NULL out.
    EXPECT_EQ(anira_stage_output_tensor(&ctx, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.rearm(), 6U);

    // A NULL ctx and a ctx without a frame: the refusal stands, and there is no latch to record
    // into.
    anira_stage_ctx bare{};
    bare.phase = ANIRA_PHASE_PRE_PROCESS;
    bare.num_inputs = 1;
    bare.num_outputs = 1;
    const std::array<const anira_stage_ctx*, 2> without_a_frame{nullptr, &bare};
    for (const anira_stage_ctx* refused : without_a_frame) {
        EXPECT_EQ(anira_stage_input_role(refused, 0), ANIRA_ROLE_FORCE32);
        EXPECT_EQ(anira_stage_output_role(refused, 0), ANIRA_ROLE_FORCE32);
        EXPECT_EQ(anira_stage_input_ring(refused, 0), nullptr);
        EXPECT_EQ(anira_stage_output_ring(refused, 0), nullptr);
        std::memset(&tensor, 0xff, sizeof(tensor));
        EXPECT_EQ(anira_stage_input_tensor(refused, 0, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_TRUE(all_zero(tensor));
        EXPECT_EQ(anira_stage_output_tensor(refused, 0, &tensor), ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(anira_stage_input_tensor(refused, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(anira_stage_default_pre_process(&bare), ANIRA_ERROR_INVALID_ARGUMENT);
    bare.phase = ANIRA_PHASE_POST_PROCESS;
    EXPECT_EQ(anira_stage_default_post_process(&bare), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(latch.rt_error(), ANIRA_OK);

    anira_drain_log();
#ifdef ENABLE_LOGGING
    // One record per kind: the first refusal, naming the entry and the running stage.
    EXPECT_EQ(anira_test::count_records(collector, "is out of range", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "is out of range", "rt");
    EXPECT_NE(record.m_message.find("anira_stage_input_role: stage 'asker': slot 1"),
              std::string::npos)
        << record.m_message;
    EXPECT_EQ(record.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
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
// The chain
// ============================================================================================

// No stage fills pre_process or post_process: the default body of each runs, once per chunk.
// A stage that fills them and calls the default itself gets the same stream, and the default
// does not run a second time behind it (the rings would move two hops).
TEST(AbiStage, DefaultRunsOnceWhenNoStageFillsThePhase) {
    const Context context;
    {
        Probe probe;
        anira_stage_desc stage = named_stage("hooks-only", &probe);
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
        anira_stage_desc stage = named_stage("owner", &probe);
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

// Two stages fill post_process: they run in chain order on one chunk with one context, so the
// second pushes what the first left in the tensor.
TEST(AbiStage, TwoStagesRunInChainOrderOnOneChunk) {
    const Context context;
    std::atomic<int> order{0};
    Probe scale;
    scale.m_scale = 2.0F;  // works on the tensor, moves no ring
    scale.m_order = &order;
    Probe push;
    push.m_defaults = 1;  // the one stage of the chain that moves the ring
    push.m_order = &order;
    anira_stage_desc first = named_stage("scale", &scale);
    first.post_process = probe_phase;
    anira_stage_desc second = named_stage("push", &push);
    second.post_process = probe_phase;
    StagedHandler handler(context, stream_model(), {first, second});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(4);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    std::vector<float> wanted = stream.expected();
    for (float& sample : wanted) { sample *= 2.0F; }
    expect_same_stream(stream.m_out, wanted);
    EXPECT_EQ(scale.m_seen_order.at(ANIRA_PHASE_POST_PROCESS) + 1,
              push.m_seen_order.at(ANIRA_PHASE_POST_PROCESS));
    EXPECT_EQ(scale.m_outputs.at(ANIRA_PHASE_POST_PROCESS).at(0).handle.host.ptr,
              push.m_outputs.at(ANIRA_PHASE_POST_PROCESS).at(0).handle.host.ptr)
        << "one chunk, one model tensor";
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
    anira_stage_desc stage = named_stage("idle-pre", &probe);
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
    EXPECT_EQ(anira_test::count_records(collector, "stage chain: pre_process", "rt"), 1U);
    const RecordCollector::Record record =
        anira_test::find_record(collector, "stage chain: pre_process", "rt");
    EXPECT_NE(record.m_message.find("stage 'idle-pre'"), std::string::npos) << record.m_message;
    EXPECT_NE(record.m_message.find("input tensor 0, channel 0 moved 0 elements, the hop is 512"),
              std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("the rest was discarded"), std::string::npos)
        << record.m_message;
#endif
}

// Two stages that both pop the ring: more than the hop left it, which nothing can undo.
TEST(AbiStage, TwoStagesThatBothPopShiftTheStreamAndAreRecorded) {
    const Context context;
    anira_drain_log();
    RecordCollector collector;
    Probe first;
    first.m_defaults = 1;
    Probe second;
    second.m_defaults = 1;
    anira_stage_desc a = named_stage("pop-a", &first);
    a.pre_process = probe_phase;
    anira_stage_desc b = named_stage("pop-b", &second);
    b.pre_process = probe_phase;
    StagedHandler handler(context, stream_model(), {a, b});
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
    EXPECT_EQ(first.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 1)
        << "one chunk took both hops: the second chunk of the block never formed";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    const RecordCollector::Record record =
        anira_test::find_record(collector, "stage chain: pre_process", "rt");
    EXPECT_NE(record.m_message.find("stages 'pop-a' to 'pop-b'"), std::string::npos)
        << record.m_message;
    EXPECT_NE(record.m_message.find("moved 1024 elements, the hop is 512"), std::string::npos)
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
        anira_stage_desc stage = named_stage("idle-post", &probe);
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
            anira_test::find_record(collector, "stage chain: post_process", "rt");
        EXPECT_NE(record.m_message.find("stage 'idle-post'"), std::string::npos)
            << record.m_message;
        EXPECT_NE(record.m_message.find("output tensor 0, channel 0 moved 0 elements"),
                  std::string::npos)
            << record.m_message;
        EXPECT_NE(record.m_message.find("the rest is zeros"), std::string::npos)
            << record.m_message;
#endif
    }
    {
        Probe probe;
        probe.m_defaults = 2;  // two pushes of one chunk
        anira_stage_desc stage = named_stage("double-post", &probe);
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
    Probe later;
    anira_stage_desc stage = named_stage("failing-pre", &probe);
    stage.pre_process = probe_phase;
    stage.before_inference = probe_phase;
    anira_stage_desc behind = named_stage("behind", &later);
    behind.pre_process = probe_phase;
    StagedHandler handler(context, stream_model(), {stage, behind});
    ASSERT_EQ(handler.prepare(zeros_contract()), ANIRA_OK) << handler.m_err.message;
    Stream stream(handler.m_handler);
    stream.blocks(6);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // The failed chunk is zeros where it stands, every other one is intact and in place: the
    // ring gave up the failed chunk's hop all the same.
    expect_same_stream(stream.m_out, stream.expected(/*failed=*/2));
    EXPECT_EQ(anira_handler_rt_error(handler.m_handler), ANIRA_ERROR_OUT_OF_MEMORY);
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 6);
    EXPECT_EQ(later.m_calls.at(ANIRA_PHASE_PRE_PROCESS).load(), 5)
        << "later stages of a failed phase do not run";
    EXPECT_EQ(probe.m_calls.at(ANIRA_PHASE_BEFORE_INFERENCE).load(), 5)
        << "the failed chunk was never submitted";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(
        anira_test::count_records(collector, "stage 'failing-pre': pre_process returned", "rt"),
        1U);
    EXPECT_EQ(anira_test::count_records(collector, "stage chain:", "rt"), 0U)
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
        anira_stage_desc stage = named_stage("failing-hook", &probe);
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
    anira_stage_desc stage = named_stage("failing-post", &probe);
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
// A chain without a stage is the 2.x default processor
// ============================================================================================

// Differential, on a window longer than its hop (the receptive-field fill) and a backend whose
// every output sample needs the history: the C handler, whose session sees the chain, against
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
// adapter of anira reads: the handler is created, and the plan's extension row names the stage.
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
    anira_stage_desc stage = named_stage("entry-reader", nullptr);
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
    EXPECT_STREQ(row.consumer, "entry-reader");
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
    anira_stage_desc post_only = named_stage("post-only", &probe);
    post_only.post_process = probe_phase;
    StagedHandler wrong_side(context, stream_model(), {post_only});
    EXPECT_EQ(wrong_side.prepare(contract), ANIRA_ERROR_CONFIG);

    auto scratch = std::make_unique<std::array<int16_t, k_hop>>();
    anira_stage_desc convert = named_stage("int16-to-float", scratch.get());
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
// is exact: the word over 32768, halved.
TEST(AbiStage, Int16RingWithAConvertingStageOnEveryEngine) {
    const Context context;
    for (const anira_engine engine : anira_test::oracle_engines()) {
        SCOPED_TRACE("engine " + std::to_string(static_cast<unsigned>(engine)));
        auto scratch = std::make_unique<std::array<int16_t, k_hop>>();
        anira_stage_desc convert = named_stage("int16-to-float", scratch.get());
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

/// What the stage of TensorsFollowTheStructsMemoryAcrossLibTorchInferences recorded per
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

// LibTorch swaps the memory of the struct's input buffer with its own on every inference
// (LibTorchProcessor.cpp, swap_data), so a struct holds another block each time it comes
// round. The accessor builds the tensor from the struct's memory of the moment: two uses of
// one struct (told apart by its output tensor, which is not swapped) return different input
// pointers, and the default fill through that pointer is what the engine read, since the
// stream is intact. Nothing about a descriptor may be cached across calls.
TEST(AbiStage, TensorsFollowTheStructsMemoryAcrossLibTorchInferences) {
    const Context context;
    Pointers pointers;
    anira_stage_desc stage = named_stage("pointers", &pointers);
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
            EXPECT_NE(pointers.m_inputs.at(chunk), pointers.m_inputs.at(last_use.at(index)))
                << "chunks " << last_use.at(index) << " and " << chunk << " on struct " << index
                << " saw one input block, so the swap was not followed";
            ++pairs;
        }
        last_use.at(index) = chunk;
    }
    EXPECT_GE(pairs, chunks - structs) << "every chunk beyond a struct's first use is a pair";
}

#endif  // USE_LIBTORCH
