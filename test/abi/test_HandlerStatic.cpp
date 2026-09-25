// The Static tensors of anira/abi/handler.h: anira_handler_set_static_input,
// anira_handler_get_static_output, the Static elements of the _multi forms and the handler's
// store behind them (src/capi/port.h). A Static tensor has one description everywhere:
// the whole tensor in the spec's shape and dtype. Every case is engine-free: the one model path
// is the custom row, whose engine is a gate added to the pipeline that runs the pass-through
// (tensor i of the output is tensor i of the input), so what goes in as a Static input comes
// back as the Static output of the inference that saw it.
//
// A spec's dtype is float32 until the engines bind other types, so a store of another dtype is
// reached through a white-box handler: an anira_handler whose store the test builds itself
// (the two entries are legal on an unprepared handler and touch nothing but the store and the
// latch).

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

#include "../support/copy_oracle.h"
#include "../support/log_record_collector.h"
#include "capi/handler.h"
#include "capi/port.h"
#include "float_face.h"
#include "handler_support.h"

namespace {

namespace oracle = anira_test::oracle;
using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::RecordCollector;
using anira_test::whole_f32;

constexpr uint32_t k_hop = 8;      // the window of the streamed tensor, and the contract's block
constexpr double k_rate = 800.0;   // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr size_t k_unset = 77777;  // what a delivered count holds before its call
constexpr float k_untouched = -7.0F;

// ---- the models --------------------------------------------------------------------------------

TensorSpec streamed(std::string_view name, int64_t channels) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STREAMED);
    spec.axis(0, ANIRA_AXIS_BATCH, 1)
        .axis(1, ANIRA_AXIS_CHANNEL, channels)
        .axis(2, ANIRA_AXIS_TIME, k_hop)
        .window(k_hop, k_hop, 0);
    return spec;
}

/// [batch 1][channel 2][any 3]: a Static tensor with a Channel axis of extent 2, six values.
TensorSpec two_channel_static(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_CHANNEL, 2).axis(2, ANIRA_AXIS_ANY, 3);
    return spec;
}

/// [batch 1][any 4]: a second Static tensor of another shape, four values.
TensorSpec four_values(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_STATIC);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_ANY, 4);
    return spec;
}

/// [batch 1][time 4]: a Buffer tensor, the whole submitted buffer as one tensor. A per-job
/// payload of the Async contract: valid as a spec, refused by prepare under a Hard contract.
TensorSpec buffer_values(std::string_view name) {
    TensorSpec spec(name, ANIRA_DTYPE_F32, ANIRA_ROLE_BUFFER);
    spec.axis(0, ANIRA_AXIS_BATCH, 1).axis(1, ANIRA_AXIS_TIME, 4);
    return spec;
}

constexpr std::array<int64_t, 3> k_static_shape{1, 2, 3};
constexpr std::array<int64_t, 2> k_four_shape{1, 4};
constexpr size_t k_static_elements = 6;
using StaticValues = std::array<float, k_static_elements>;

/// Slot 0: a two-channel stream. Slot 1: the two-channel Static tensor. Slot 2: the second
/// Static tensor, of another shape.
ModelConfig static_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", 2));
    model.input(two_channel_static("values_in"));
    model.input(four_values("four_in"));
    model.output(streamed("out", 2));
    model.output(two_channel_static("values_out"));
    model.output(four_values("four_out"));
    model.max_instances(2);
    return model;
}

/// static_model() with a Buffer tensor in slot 2 of either side.
ModelConfig buffer_model() {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", 2));
    model.input(two_channel_static("values_in"));
    model.input(buffer_values("buffer_in"));
    model.output(streamed("out", 2));
    model.output(two_channel_static("values_out"));
    model.output(buffer_values("buffer_out"));
    model.max_instances(2);
    return model;
}

StaticValues values_of(float first) {
    StaticValues values{};
    for (size_t i = 0; i < values.size(); ++i) { values.at(i) = first + static_cast<float>(i); }
    return values;
}

// ---- the rig -----------------------------------------------------------------------------------

/// A handler over static_model() whose custom row runs on a gate engine added to its pipeline,
/// the gate closed until settle() opens it. prepare_now false leaves the handler unprepared.
class GateRig {
public:
    explicit GateRig(anira_miss_policy policy = ANIRA_MISS_ZEROS,
                     anira_miss_fn miss_fn = nullptr,
                     void* miss_user_data = nullptr,
                     bool prepare_now = true)
        : m_handler(m_context, static_model(), m_candidates, {}, m_gate.engine())
        , m_policy(policy)
        , m_miss_fn(miss_fn)
        , m_miss_user_data(miss_user_data) {
        if (prepare_now) { prepare(); }
    }
    ~GateRig() {
        m_gate.m_open.store(true);
        m_handler.destroy();  // drains the in-flight work before the gate dies
    }
    GateRig(const GateRig&) = delete;
    GateRig& operator=(const GateRig&) = delete;
    GateRig(GateRig&&) = delete;
    GateRig& operator=(GateRig&&) = delete;

    /// Prepares (again), the gate open for the previous session's in-flight work, and closes the
    /// gate on the new session.
    void prepare() {
        m_gate.m_open.store(true);
        anira::ContractHandle contract =
            anira_test::explicit_contract(k_hop, k_rate, m_policy, 0.0, 1.0);
        contract.hard_geometry(1, k_hop, k_rate);
        contract.hard_miss_fn(m_miss_fn, m_miss_user_data);
        const anira_status prepared = m_handler.prepare(contract);
        ASSERT_EQ(prepared, ANIRA_OK) << m_handler.m_err.message;
        m_session = anira_test::session_of(get());
        ASSERT_NE(m_session, nullptr);
        m_gate.m_open.store(false);
    }

    bool ready() const { return m_session != nullptr; }
    anira_handler* get() const { return m_handler.m_handler; }
    anira_test::GateEngine& gate() { return m_gate; }

    /// Opens the gate until every submitted inference is collected.
    void settle() {
        oracle::settle(m_gate, *m_session, [this] { anira_test::available(get()); });
    }

    /// One block of the stream through the single form; the delivered status.
    anira_status stream_block(float first = 1.0F) {
        std::array<float, static_cast<size_t>(2) * k_hop> in{};
        for (size_t i = 0; i < in.size(); ++i) { in.at(i) = first + static_cast<float>(i); }
        std::array<float, static_cast<size_t>(2) * k_hop> out{};
        const anira_tensor in_tensor = whole_f32(in.data(), {2, k_hop});
        const anira_tensor out_tensor = whole_f32(out.data(), {2, k_hop});
        return anira_handler_process(get(), &in_tensor, 0, &out_tensor, 0, nullptr);
    }

private:
    anira_test::Context m_context;
    std::vector<anira_backend_id> m_candidates{{.struct_size = sizeof(anira_backend_id),
                                                .engine = ANIRA_ENGINE_NONE,
                                                .provider = ANIRA_PROVIDER_DEFAULT,
                                                .engine_id = nullptr}};
    anira_test::GateEngine m_gate;  // before the handler, which dies first
    anira_test::Handler m_handler;
    anira_miss_policy m_policy;
    anira_miss_fn m_miss_fn;
    void* m_miss_user_data;
    std::shared_ptr<anira::SessionElement> m_session;
};

/// set_static_input(1) of `values`, one streamed block, settle: the model has produced them.
void run_through(GateRig& rig, const StaticValues& values) {
    const anira_tensor tensor = whole_f32(values.data(), k_static_shape);
    ASSERT_EQ(anira_handler_set_static_input(rig.get(), 1, &tensor), ANIRA_OK);
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
}

StaticValues stored_output(anira_handler* handler) {
    StaticValues values{};
    values.fill(k_untouched);
    const anira_tensor tensor = whole_f32(values.data(), k_static_shape);
    EXPECT_EQ(anira_handler_get_static_output(handler, 1, &tensor), ANIRA_OK);
    return values;
}

// ---- the round trip ----------------------------------------------------------------------------

TEST(AbiHandlerStatic, ATwoChannelStaticTensorTravelsWhole) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    const StaticValues zeros{};
    EXPECT_EQ(stored_output(rig.get()), zeros) << "all zeros before the first capture";
    run_through(rig, values_of(10.0F));
    EXPECT_EQ(stored_output(rig.get()), values_of(10.0F));
    // A value set between two calls applies from the next submitted inference on.
    run_through(rig, values_of(20.0F));
    EXPECT_EQ(stored_output(rig.get()), values_of(20.0F));
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);
}

// A second Static tensor, of another shape, travels whole beside the first.
TEST(AbiHandlerStatic, ASecondStaticSlotOfAnotherShapeTravelsWhole) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    const std::array<float, 4> in{1.5F, 2.5F, 3.5F, 4.5F};
    const anira_tensor in_tensor = whole_f32(in.data(), k_four_shape);
    ASSERT_EQ(anira_handler_set_static_input(rig.get(), 2, &in_tensor), ANIRA_OK);
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    std::array<float, 4> out{};
    const anira_tensor out_tensor = whole_f32(out.data(), k_four_shape);
    ASSERT_EQ(anira_handler_get_static_output(rig.get(), 2, &out_tensor), ANIRA_OK);
    EXPECT_EQ(out, in);
    EXPECT_EQ(anira_handler_get_latency(rig.get(), 2), 0U) << "no ring, no stream latency";
}

// A Buffer tensor is a per-job payload, which arrives with the Async contract. The spec is
// valid (create takes it, no contract is known yet), nothing is stored for it, and prepare
// under a Hard contract refuses the model with NOT_SUPPORTED, naming the tensor.
TEST(AbiHandlerStatic, ABufferSpecUnderAHardContractIsRefusedAtPrepare) {
    const anira_test::Context context;
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_NONE,
                                                    .provider = ANIRA_PROVIDER_DEFAULT,
                                                    .engine_id = nullptr}};
    anira_test::Handler handler(context, buffer_model(), candidates);
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr) << "a Buffer spec is valid without a contract: " << handler.m_err.message;

    // The Buffer slot has a buffer port, which holds nothing: the Static entries take a Static
    // slot only, and the Static slot beside it works as ever.
    ASSERT_EQ(h->m_input_ports.size(), 3U);
    ASSERT_EQ(h->m_output_ports.size(), 3U);
    EXPECT_TRUE(std::holds_alternative<anira::capi::BufferPort>(h->m_input_ports[2]));
    EXPECT_TRUE(std::holds_alternative<anira::capi::BufferPort>(h->m_output_ports[2]));
    EXPECT_EQ(anira::capi::port_role(h->m_input_ports[2]), ANIRA_ROLE_BUFFER);
    EXPECT_EQ(anira::capi::static_slot(h->m_input_ports, 2), nullptr);
    EXPECT_EQ(anira::capi::static_slot(h->m_output_ports, 2), nullptr);
    EXPECT_NE(anira::capi::static_slot(h->m_input_ports, 1), nullptr);
    std::array<float, 4> four{1.5F, 2.5F, 3.5F, 4.5F};
    const anira_tensor four_tensor = whole_f32(four.data(), k_four_shape);
    EXPECT_EQ(anira_handler_set_static_input(h, 2, &four_tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_get_static_output(h, 2, &four_tensor), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(four, (std::array<float, 4>{1.5F, 2.5F, 3.5F, 4.5F})) << "a refused get wrote";
    const StaticValues values = values_of(10.0F);
    const anira_tensor values_tensor = whole_f32(values.data(), k_static_shape);
    EXPECT_EQ(anira_handler_set_static_input(h, 1, &values_tensor), ANIRA_OK);

    anira::ContractHandle contract =
        anira_test::explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 0.0, 1.0);
    contract.hard_geometry(1, k_hop, k_rate);
    EXPECT_EQ(handler.prepare(contract), ANIRA_ERROR_NOT_SUPPORTED);
    const std::string message = handler.m_err.message;
    EXPECT_NE(message.find("'buffer_in'"), std::string::npos) << message;
    EXPECT_NE(message.find("Buffer"), std::string::npos) << message;
    EXPECT_NE(message.find("Async contract"), std::string::npos) << message;
    // The handler stays unprepared, and says so.
    std::array<float, static_cast<size_t>(2) * k_hop> samples{};
    const anira_tensor block = whole_f32(samples.data(), {2, k_hop});
    EXPECT_EQ(anira_handler_process(h, &block, 0, &block, 0, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_get_latency(h, 0), 0U);
}

TEST(AbiHandlerStatic, ValuesSetBeforePrepareSurviveAndTheOutputSurvivesResetAndPrepare) {
    GateRig rig(ANIRA_MISS_ZEROS, nullptr, nullptr, /*prepare_now=*/false);
    // Legal from create on: the store is the handler's, built with it.
    const StaticValues early = values_of(30.0F);
    const anira_tensor tensor = whole_f32(early.data(), k_static_shape);
    ASSERT_EQ(anira_handler_set_static_input(rig.get(), 1, &tensor), ANIRA_OK);
    const StaticValues zeros{};
    EXPECT_EQ(stored_output(rig.get()), zeros) << "an unprepared handler answers, with zeros";
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);

    ASSERT_NO_FATAL_FAILURE(rig.prepare());
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    EXPECT_EQ(stored_output(rig.get()), early) << "the value set before prepare reached the model";

    anira_handler_reset(rig.get());
    EXPECT_EQ(stored_output(rig.get()), early) << "reset leaves the store alone";
    ASSERT_NO_FATAL_FAILURE(rig.prepare());
    EXPECT_EQ(stored_output(rig.get()), early) << "and so does a second prepare";
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    EXPECT_EQ(stored_output(rig.get()), early) << "the input store survived it too";
}

// ---- the strides -------------------------------------------------------------------------------

TEST(AbiHandlerStatic, AStridedTensorIsReadAndWrittenByItsStrides) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    // The six values of [1, 2, 3] laid out transposed (the any axis outermost) with a gap
    // after every value: strides {0, 2, 4} in elements, behind a lead of one element.
    constexpr size_t k_lead = 1;
    std::array<float, k_lead + 12> memory{};
    memory.fill(k_untouched);
    const StaticValues logical = values_of(40.0F);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < 3; ++i) {
            memory.at(k_lead + (channel * 2) + (i * 4)) = logical.at((channel * 3) + i);
        }
    }
    anira_tensor strided = whole_f32(memory.data(), k_static_shape);
    strided.byte_offset = k_lead * sizeof(float);
    strided.strides[0] = 1;  // an axis of extent 1: never stepped
    strided.strides[1] = 2;
    strided.strides[2] = 4;
    ASSERT_EQ(anira_handler_set_static_input(rig.get(), 1, &strided), ANIRA_OK);
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    EXPECT_EQ(stored_output(rig.get()), logical) << "read by its strides";

    std::array<float, k_lead + 12> back{};
    back.fill(k_untouched);
    anira_tensor out = strided;
    out.handle.host.ptr = back.data();
    ASSERT_EQ(anira_handler_get_static_output(rig.get(), 1, &out), ANIRA_OK);
    EXPECT_EQ(back, memory) << "written by its strides, the elements between left alone";
}

TEST(AbiHandlerStatic, HostPinnedMemoryIsAccepted) {
    const GateRig rig;
    ASSERT_TRUE(rig.ready());
    const StaticValues values = values_of(50.0F);
    anira_tensor tensor = whole_f32(values.data(), k_static_shape);
    tensor.domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST_PINNED);
    EXPECT_EQ(anira_handler_set_static_input(rig.get(), 1, &tensor), ANIRA_OK);
    StaticValues out{};
    anira_tensor out_tensor = whole_f32(out.data(), k_static_shape);
    out_tensor.domain = static_cast<uint32_t>(ANIRA_DOMAIN_HOST_PINNED);
    EXPECT_EQ(anira_handler_get_static_output(rig.get(), 1, &out_tensor), ANIRA_OK);
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);
}

// ---- the refusals ------------------------------------------------------------------------------

TEST(AbiHandlerStatic, EveryRefusalHasItsStatusAndStoresNothing) {
    RecordCollector collector;
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    run_through(rig, values_of(60.0F));
    StaticValues memory = values_of(70.0F);
    const anira_tensor good = whole_f32(memory.data(), k_static_shape);

    // A NULL handler is refused and not recorded.
    EXPECT_EQ(anira_handler_set_static_input(nullptr, 1, &good), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_get_static_output(nullptr, 1, &good), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_OK);

    const auto refused =
        [&](const anira_tensor* tensor, uint32_t slot, anira_status status, const char* what) {
            SCOPED_TRACE(what);
            anira_handler_reset(h);  // re-arms the latch: rt_error reads the next refusal alone
            EXPECT_EQ(anira_handler_set_static_input(h, slot, tensor), status);
            EXPECT_EQ(anira_handler_rt_error(h), status);
            anira_handler_reset(h);
            EXPECT_EQ(anira_handler_get_static_output(h, slot, tensor), status);
            EXPECT_EQ(anira_handler_rt_error(h), status);
        };
    refused(&good, 0, ANIRA_ERROR_INVALID_ARGUMENT, "a Streamed slot");
    refused(&good, 3, ANIRA_ERROR_INVALID_ARGUMENT, "a slot out of range");
    refused(nullptr, 1, ANIRA_ERROR_INVALID_ARGUMENT, "a NULL tensor");

    anira_tensor wrong = good;
    wrong.domain = static_cast<uint32_t>(ANIRA_DOMAIN_CUDA);
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "a device domain");
    wrong = good;
    wrong.flags |= static_cast<uint32_t>(ANIRA_TENSOR_PLANAR);
    refused(&wrong, 1, ANIRA_ERROR_NOT_SUPPORTED, "ANIRA_TENSOR_PLANAR");
    wrong = good;
    wrong.flags |= 0x40000000U;
    refused(&wrong, 1, ANIRA_ERROR_NOT_SUPPORTED, "an unknown flag");
    wrong = whole_f32(memory.data(), {1, 6});
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "another rank: the flat description");
    wrong = whole_f32(memory.data(), {1, 3, 2});
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "another shape");
    wrong = whole_f32(memory.data(), {1, 2, 2});
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "a partial tensor");
    wrong = whole_f32(memory.data(), {1, 0, 3});
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "an empty tensor is another shape");
    const anira_tensor zeroed{};
    refused(&zeroed, 1, ANIRA_ERROR_INVALID_ARGUMENT, "a zeroed record");
    wrong = good;
    wrong.dtype = ANIRA_DTYPE_I32;
    refused(&wrong, 1, ANIRA_ERROR_CONFIG, "another dtype");
    wrong = good;
    wrong.handle.host.ptr = nullptr;
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "NULL memory");
    wrong = good;
    wrong.byte_offset = 1;
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "misaligned memory");
    wrong = good;
    wrong.strides[0] = 1;
    wrong.strides[1] = 3;
    wrong.strides[2] = -1;
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "a negative stride");
    // The order is pinned by a tensor wrong in two ways: the shape before the dtype.
    wrong = whole_f32(memory.data(), {1, 6});
    wrong.dtype = ANIRA_DTYPE_I32;
    refused(&wrong, 1, ANIRA_ERROR_INVALID_ARGUMENT, "the shape is checked before the dtype");

    // ANIRA_TENSOR_READ_ONLY: fine on an input, refused on the output anira writes.
    anira_handler_reset(h);
    const StaticValues constant = values_of(60.0F);
    const anira_tensor read_only = whole_f32(constant.data(), k_static_shape);
    EXPECT_EQ(anira_handler_set_static_input(h, 1, &read_only), ANIRA_OK);
    EXPECT_EQ(anira_handler_get_static_output(h, 1, &read_only), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);

    // No refused call stored or wrote anything.
    EXPECT_EQ(memory, values_of(70.0F));
    anira_handler_reset(h);
    EXPECT_EQ(stored_output(h), values_of(60.0F));
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    EXPECT_EQ(stored_output(h), values_of(60.0F)) << "the input store kept its last good value";

    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_GE(anira_test::count_records(collector, "Static input slot 1", "rt"), 1U);
    EXPECT_GE(anira_test::count_records(collector, "Static output slot 1", "rt"), 1U);
    EXPECT_GE(anira_test::count_records(collector, "a Static tensor is one block", "rt"), 1U);
#endif
}

TEST(AbiHandlerStatic, ARefusalIsLoggedOncePerKind) {
    RecordCollector collector;
    const GateRig rig;
    ASSERT_TRUE(rig.ready());
    StaticValues memory{};
    const anira_tensor wrong = whole_f32(memory.data(), {1, 6});
    for (int k = 0; k < 3; ++k) {
        EXPECT_EQ(anira_handler_set_static_input(rig.get(), 1, &wrong),
                  ANIRA_ERROR_INVALID_ARGUMENT);
    }
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "does not fit", "rt"), 1U)
        << "one latched record, the later refusals of the kind are counted";
#endif
}

TEST(AbiHandlerStatic, ASingleFormOnAStaticSlotIsRefused) {
    const GateRig rig;
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    StaticValues memory = values_of(80.0F);
    const anira_tensor whole = whole_f32(memory.data(), k_static_shape);
    size_t delivered = k_unset;
    const auto expect_refused = [&](anira_status status, const char* what) {
        EXPECT_EQ(status, ANIRA_ERROR_INVALID_ARGUMENT) << what;
        EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT) << what;
        anira_handler_reset(h);
    };
    expect_refused(anira_handler_process(h, &whole, 1, &whole, 1, &delivered), "process");
    EXPECT_EQ(delivered, 0U);
    // Each side of the two-slot form is refused on its own: the Streamed slot 0 on the other
    // side does not excuse a Static slot.
    expect_refused(anira_handler_process(h, &whole, 1, &whole, 0, nullptr),
                   "process, a Static in_slot beside a Streamed out_slot");
    expect_refused(anira_handler_process(h, &whole, 0, &whole, 1, nullptr),
                   "process, a Static out_slot beside a Streamed in_slot");
    expect_refused(anira_handler_push_data(h, &whole, 1), "push_data");
    expect_refused(anira_handler_pop_data(h, &whole, 1, &delivered), "pop_data");
    expect_refused(anira_handler_process_wait(h, &whole, 2, &whole, 2, &delivered, 0.0),
                   "process_wait on the second Static slot");
    expect_refused(anira_handler_pop_data_wait(h, &whole, 1, &delivered, 0.0), "pop_data_wait");
    EXPECT_EQ(memory, values_of(80.0F));
    const StaticValues zeros{};
    EXPECT_EQ(stored_output(h), zeros) << "a refused single form set nothing";
}

// ---- the _multi forms --------------------------------------------------------------------------

struct Block {
    std::array<float, static_cast<size_t>(2) * k_hop> m_stream_in{};
    std::array<float, static_cast<size_t>(2) * k_hop> m_stream_out{};
    StaticValues m_values_in{};
    StaticValues m_values_out{};
    std::array<float, 4> m_four_in{};
    std::array<float, 4> m_four_out{};

    explicit Block(size_t k) {
        for (size_t i = 0; i < m_stream_in.size(); ++i) {
            m_stream_in.at(i) = static_cast<float>((k * 100) + i + 1);
        }
        m_stream_out.fill(k_untouched);
        m_values_in = values_of(static_cast<float>(1000 * (k + 1)));
        m_values_out.fill(k_untouched);
        for (size_t i = 0; i < 4; ++i) { m_four_in.at(i) = static_cast<float>((k * 10) + i); }
        m_four_out.fill(k_untouched);
    }
    std::array<anira_tensor, 3> inputs() const {
        return {whole_f32(m_stream_in.data(), {2, k_hop}),
                whole_f32(m_values_in.data(), k_static_shape),
                whole_f32(m_four_in.data(), k_four_shape)};
    }
    std::array<anira_tensor, 3> outputs() {
        return {whole_f32(m_stream_out.data(), {2, k_hop}),
                whole_f32(m_values_out.data(), k_static_shape),
                whole_f32(m_four_out.data(), k_four_shape)};
    }
};

TEST(AbiHandlerStatic, AMultiCallEqualsTheThreeCallSequence) {
    GateRig multi;
    GateRig sequence;
    ASSERT_TRUE(multi.ready());
    ASSERT_TRUE(sequence.ready());
    for (size_t k = 0; k < 5; ++k) {
        SCOPED_TRACE("block " + std::to_string(k));
        Block a(k);
        const std::array<anira_tensor, 3> a_in = a.inputs();
        const std::array<anira_tensor, 3> a_out = a.outputs();
        std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
        const anira_status multi_status = anira_handler_process_multi(multi.get(),
                                                                      a_in.data(),
                                                                      3,
                                                                      a_out.data(),
                                                                      3,
                                                                      delivered.data());
        EXPECT_EQ(delivered[1], k_static_elements) << "a carried Static output: its element count";
        EXPECT_EQ(delivered[2], 4U);

        Block b(k);
        const std::array<anira_tensor, 3> b_in = b.inputs();
        const std::array<anira_tensor, 3> b_out = b.outputs();
        ASSERT_EQ(anira_handler_set_static_input(sequence.get(), 1, &b_in[1]), ANIRA_OK);
        ASSERT_EQ(anira_handler_set_static_input(sequence.get(), 2, &b_in[2]), ANIRA_OK);
        size_t streamed_count = k_unset;
        const anira_status sequence_status =
            anira_handler_process(sequence.get(), b_in.data(), 0, b_out.data(), 0, &streamed_count);
        ASSERT_EQ(anira_handler_get_static_output(sequence.get(), 1, &b_out[1]), ANIRA_OK);
        ASSERT_EQ(anira_handler_get_static_output(sequence.get(), 2, &b_out[2]), ANIRA_OK);

        EXPECT_EQ(multi_status, sequence_status);
        EXPECT_EQ(delivered[0], streamed_count);
        EXPECT_EQ(a.m_stream_out, b.m_stream_out);
        EXPECT_EQ(a.m_values_out, b.m_values_out);
        EXPECT_EQ(a.m_four_out, b.m_four_out);
        if (k > 0) {
            EXPECT_EQ(a.m_values_out, Block(k - 1).m_values_in) << "what the last inference saw";
        }
        multi.settle();
        sequence.settle();
    }
    // The split forms are the same sequence: the set in push_data_multi, the get in the pop.
    Block pushed(7);
    const std::array<anira_tensor, 3> in = pushed.inputs();
    const std::array<anira_tensor, 3> out = pushed.outputs();
    ASSERT_EQ(anira_handler_push_data_multi(multi.get(), in.data(), 3), ANIRA_OK);
    multi.settle();
    std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
    ASSERT_EQ(anira_handler_pop_data_multi(multi.get(), out.data(), 3, delivered.data()), ANIRA_OK);
    EXPECT_EQ(delivered[0], k_hop);
    EXPECT_EQ(delivered[1], k_static_elements);
    EXPECT_EQ(pushed.m_values_out, pushed.m_values_in);
    EXPECT_EQ(pushed.m_four_out, pushed.m_four_in);
}

TEST(AbiHandlerStatic, EveryElementIsValidatedBeforeAnythingIsSetOrPushed) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    run_through(rig, values_of(90.0F));
    Block block(1);
    const std::array<anira_tensor, 3> in = block.inputs();
    std::array<anira_tensor, 3> out = block.outputs();
    out[2].dtype = ANIRA_DTYPE_I16;  // the last element of the call is the wrong one
    std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
    EXPECT_EQ(anira_handler_process_multi(rig.get(), in.data(), 3, out.data(), 3, delivered.data()),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(delivered, (std::array<size_t, 3>{0, 0, 0}));
    EXPECT_EQ(block.m_values_out[0], k_untouched) << "nothing was written";
    // Nothing was set either: the next inference still sees the earlier value.
    anira_handler_reset(rig.get());
    ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
    rig.settle();
    EXPECT_EQ(stored_output(rig.get()), values_of(90.0F));
    // A planar Static element is refused like a planar tensor of the entry.
    std::array<anira_tensor, 3> planar_in = block.inputs();
    planar_in[1].flags |= static_cast<uint32_t>(ANIRA_TENSOR_PLANAR);
    EXPECT_EQ(anira_handler_push_data_multi(rig.get(), planar_in.data(), 3),
              ANIRA_ERROR_NOT_SUPPORTED);
}

TEST(AbiHandlerStatic, AnElementWithAnExtentOfZeroLeavesItsSlotOut) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    run_through(rig, values_of(100.0F));
    // Three spellings of "left out": rank 1, the flat {1, 0}, and the spec's rank with one
    // extent 0; none names memory.
    const std::array<std::vector<int64_t>, 3> spellings{{{0}, {1, 0}, {1, 0, 3}}};
    for (const std::vector<int64_t>& shape : spellings) {
        Block block(2);
        std::array<anira_tensor, 3> in = block.inputs();
        std::array<anira_tensor, 3> out = block.outputs();
        in[1] = whole_f32(static_cast<float*>(nullptr), shape);
        out[1] = whole_f32(static_cast<float*>(nullptr), shape);
        out[2] = whole_f32(static_cast<float*>(nullptr), shape);
        std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
        ASSERT_TRUE(ANIRA_SUCCEEDED(
            anira_handler_process_multi(rig.get(), in.data(), 3, out.data(), 3, delivered.data())));
        EXPECT_EQ(delivered[1], 0U) << "0 for a Static output left out";
        EXPECT_EQ(delivered[2], 0U);
        EXPECT_EQ(block.m_values_out[0], k_untouched);
        rig.settle();
        EXPECT_EQ(stored_output(rig.get()), values_of(100.0F)) << "the input was not set";
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);
}

// ---- a missed block ----------------------------------------------------------------------------

/// Two delivered blocks empty the ring while their inferences wait at the closed gate (the
/// latency is two hops); the third block starves. Returns the status of the third.
anira_status starve(GateRig& rig, Block& third, std::array<size_t, 3>& delivered) {
    for (size_t k = 0; k < 2; ++k) { EXPECT_EQ(rig.stream_block(), ANIRA_OK) << "block " << k; }
    const std::array<anira_tensor, 3> in = third.inputs();
    const std::array<anira_tensor, 3> out = third.outputs();
    return anira_handler_process_multi(rig.get(), in.data(), 3, out.data(), 3, delivered.data());
}

TEST(AbiHandlerStatic, TheStaticOutputOfAMissedBlockIsTheStoredValueUnderEveryPolicy) {
    for (const anira_miss_policy policy :
         {ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST, ANIRA_MISS_BYPASS}) {
        SCOPED_TRACE("policy " + std::to_string(static_cast<int>(policy)));
        GateRig rig(policy);
        ASSERT_TRUE(rig.ready());
        ASSERT_EQ(anira_handler_get_latency(rig.get(), 0), 2U * k_hop);
        run_through(rig, values_of(110.0F));
        anira_handler_reset(rig.get());  // a fresh stream, the store untouched
        Block third(3);
        std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
        ASSERT_EQ(starve(rig, third, delivered), ANIRA_MISSED);
        EXPECT_EQ(delivered[0], 0U) << "a missed block delivers 0 on a Streamed slot";
        EXPECT_EQ(delivered[1], k_static_elements) << "and the whole stored Static tensor";
        EXPECT_EQ(third.m_values_out, values_of(110.0F))
            << "ZEROS, HOLD_LAST and BYPASS define nothing for a Static output";
    }
}

struct MissSeen {
    int m_calls = 0;
    bool m_overwrite = false;
    StaticValues m_found{};  ///< Static output 1 as the function found it
};

/// Records what Static output 1 holds when the function is called, then either overwrites the
/// whole block with -1 or leaves the Static output alone.
anira_status ANIRA_CALL miss_fn(anira_handler* /*handler*/,
                                const anira_tensor* /*inputs*/,
                                uint32_t /*num_inputs*/,
                                const anira_tensor* outputs,
                                uint32_t num_outputs,
                                void* user_data) ANIRA_NONBLOCKING {
    auto* seen = static_cast<MissSeen*>(user_data);
    ++seen->m_calls;
    if (num_outputs < 2) { return ANIRA_ERROR_INTERNAL; }
    float* values = anira_tensor_data_f32(&outputs[1]);
    if (values == nullptr) { return ANIRA_ERROR_INTERNAL; }
    for (size_t i = 0; i < k_static_elements; ++i) { seen->m_found.at(i) = values[i]; }
    float* stream = anira_tensor_data_f32(outputs);
    const size_t samples = anira_tensor_num_elements(outputs);
    for (size_t i = 0; stream != nullptr && i < samples; ++i) { stream[i] = -1.0F; }
    if (seen->m_overwrite) {
        for (size_t i = 0; i < k_static_elements; ++i) { values[i] = -1.0F; }
    }
    return ANIRA_OK;
}

TEST(AbiHandlerStatic, TheMissFunctionFindsTheStoredValueAndHasTheLastWord) {
    for (const bool overwrite : {false, true}) {
        SCOPED_TRACE(overwrite ? "the function overwrites" : "the function leaves it");
        MissSeen seen;
        seen.m_overwrite = overwrite;
        GateRig rig(ANIRA_MISS_CALLBACK, &miss_fn, &seen);
        ASSERT_TRUE(rig.ready());
        run_through(rig, values_of(120.0F));
        anira_handler_reset(rig.get());
        ASSERT_EQ(seen.m_calls, 0);
        Block third(3);
        std::array<size_t, 3> delivered{k_unset, k_unset, k_unset};
        ASSERT_EQ(starve(rig, third, delivered), ANIRA_MISSED);
        ASSERT_EQ(seen.m_calls, 1);
        EXPECT_EQ(seen.m_found, values_of(120.0F)) << "filled with the stored value first";
        EXPECT_EQ(delivered[1], k_static_elements);
        StaticValues expected = values_of(120.0F);
        if (overwrite) { expected.fill(-1.0F); }
        EXPECT_EQ(third.m_values_out, expected) << "the get step is skipped: the function's word";
        EXPECT_EQ(third.m_stream_out[0], -1.0F);
        EXPECT_EQ(stored_output(rig.get()), values_of(120.0F))
            << "what the function wrote never enters the store";
    }
}

// ---- a chunk completed as zeros ----------------------------------------------------------------

TEST(AbiHandlerStatic, AChunkCompletedAsZerosDoesNotOverwriteTheStore) {
    GateRig rig;
    ASSERT_TRUE(rig.ready());
    const auto block_with = [&](const StaticValues& values) {
        const anira_tensor tensor = whole_f32(values.data(), k_static_shape);
        ASSERT_EQ(anira_handler_set_static_input(rig.get(), 1, &tensor), ANIRA_OK);
        ASSERT_TRUE(ANIRA_SUCCEEDED(rig.stream_block()));
        rig.settle();
    };
    ASSERT_NO_FATAL_FAILURE(block_with(values_of(130.0F)));
    EXPECT_EQ(stored_output(rig.get()), values_of(130.0F));
    // The engine fails: the chunk delivers zeros on its stream, and captures nothing.
    rig.gate().m_fail.store(true);
    ASSERT_NO_FATAL_FAILURE(block_with(values_of(140.0F)));
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_ERROR_ENGINE);
    EXPECT_EQ(stored_output(rig.get()), values_of(130.0F)) << "what the model produced last";
    rig.gate().m_fail.store(false);

    ASSERT_NO_FATAL_FAILURE(block_with(values_of(150.0F)));
    EXPECT_EQ(stored_output(rig.get()), values_of(150.0F));
}

// ---- stores of another dtype, through a white-box handler --------------------------------------

template <typename T>
void typed_round_trip(anira_dtype dtype) {
    auto handler = std::make_unique<anira_handler>();
    const std::vector<int64_t> shape{2, 3};
    // What anira_handler_create builds: one port per tensor, sized once and the static port's
    // value constructed in place (it holds atomics and does not move), and the slot counts, the
    // lengths of the model config's two lists (one tensor per side here, slot 0).
    handler->m_input_ports = std::vector<anira::capi::Port>(1);
    handler->m_output_ports = std::vector<anira::capi::Port>(1);
    handler->m_input_ports[0].emplace<anira::capi::StaticPort>(shape, dtype);
    handler->m_output_ports[0].emplace<anira::capi::StaticPort>(shape, dtype);
    handler->m_num_inputs = 1;
    handler->m_num_outputs = 1;
    const std::array<T, 6> values{T{-3}, T{2}, T{32767}, T{-32768}, T{5}, T{6}};
    anira_tensor in{};
    anira_tensor_init_host(&in, const_cast<T*>(values.data()), dtype, 2, shape.data());
    ASSERT_EQ(anira_handler_set_static_input(handler.get(), 0, &in), ANIRA_OK);
    // What the chain does between the two stores, without a session: packed out, packed in.
    const anira::capi::StaticSlot* input_store =
        anira::capi::static_slot(handler->m_input_ports, 0);
    ASSERT_NE(input_store, nullptr);
    ASSERT_NE(anira::capi::static_slot(handler->m_output_ports, 0), nullptr);
    std::array<T, 6> model{};
    input_store->read_packed(model.data(), sizeof(model));
    EXPECT_EQ(model, values);
    anira::capi::static_slot(handler->m_output_ports, 0)->write_packed(model.data(), sizeof(model));
    std::array<T, 6> out{};
    anira_tensor out_tensor{};
    anira_tensor_init_host(&out_tensor, out.data(), dtype, 2, shape.data());
    ASSERT_EQ(anira_handler_get_static_output(handler.get(), 0, &out_tensor), ANIRA_OK);
    EXPECT_EQ(out, values);

    // Nothing converts: a float32 tensor of the same shape is another dtype.
    std::array<float, 6> floats{};
    const anira_tensor wrong = whole_f32(floats.data(), shape);
    EXPECT_EQ(anira_handler_set_static_input(handler.get(), 0, &wrong), ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_handler_rt_error(handler.get()), ANIRA_ERROR_CONFIG);
    // Memory that does not start on a multiple of the element size.
    anira_tensor misaligned = out_tensor;
    misaligned.byte_offset = 1;
    EXPECT_EQ(anira_handler_get_static_output(handler.get(), 0, &misaligned),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(out, values) << "a refused call writes nothing";
}

TEST(AbiHandlerStatic, AnInt32AndAnInt16StoreTravelTyped) {
    typed_round_trip<int32_t>(ANIRA_DTYPE_I32);
    typed_round_trip<int16_t>(ANIRA_DTYPE_I16);
}

// A writer that rewrites the whole tensor with one value per pass, a reader that spins: every
// tensor the reader takes holds one value. Under TSan this is also the proof that a read which
// overlaps a write is no data race (the values sit in atomics, the counter orders them).
TEST(AbiHandlerStatic, ASpinningReaderNeverSeesATornTensor) {
    constexpr size_t k_elements = 67;  // not a multiple of the store's word
    constexpr int32_t k_passes = 20000;
    const std::vector<int64_t> shape{static_cast<int64_t>(k_elements)};
    anira::capi::StaticSlot slot(shape, ANIRA_DTYPE_I32);
    std::atomic<bool> done{false};
    std::atomic<int> torn{0};
    std::thread reader([&] {
        std::array<int32_t, k_elements> seen{};
        anira_tensor tensor{};
        anira_tensor_init_host(&tensor, seen.data(), ANIRA_DTYPE_I32, 1, shape.data());
        while (!done.load()) {
            slot.read(tensor);
            for (const int32_t value : seen) {
                if (value != seen[0]) { torn.fetch_add(1); }
            }
        }
    });
    std::array<int32_t, k_elements> values{};
    anira_tensor tensor{};
    anira_tensor_init_host(&tensor, values.data(), ANIRA_DTYPE_I32, 1, shape.data());
    for (int32_t pass = 1; pass <= k_passes; ++pass) {
        values.fill(pass);
        slot.write(tensor);
    }
    done.store(true);
    reader.join();
    EXPECT_EQ(torn.load(), 0);
}

}  // namespace
