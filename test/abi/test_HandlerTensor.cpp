// The Hard entries over host tensors (anira/abi/handler.h under the bare names):
// anira_handler_process, push_data, pop_data, their _multi forms and the four _wait twins. A
// host block is one anira_tensor per slot, of the logical shape [channels, samples], over
// planar memory, one block read by strides (contiguous, interleaved) or one packed block.
//
// The tensor entries, the _f32 entries and the 2.x handler share one copy path, so a
// pass-through model cancels a symmetric stride bug and no single oracle would see it. The
// cases are therefore named comparisons: absolute expectations per description (every channel
// its own ramp, late by the reported latency), crossed descriptions on the two sides, in
// place, the multi forms with a Static slot, the single forms beside unused slots, the miss
// policies, and every refusal with its status, its latched record and the fixed check order.
// A descriptor is compared byte for byte around every call: anira never writes one.
//
// Determinism as in test_HandlerCopyOracle.cpp: the session's backend is a GateBackend that is
// closed during every nonblocking call; settle() opens it until every submitted inference is
// collected. A starved block is a call without the settle. No sleeps, no wall clock.

#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../support/copy_oracle.h"
#include "../support/log_record_collector.h"
#include "handler_support.h"

namespace {

namespace oracle = anira_test::oracle;
using anira::ModelConfig;
using anira::TensorSpec;
using anira_test::RecordCollector;

constexpr uint32_t k_hop = 8;      // the window of every streamed tensor, and the contract's block
constexpr double k_rate = 800.0;   // a hop lasts 10 ms, the explicit budget below is 1 ms
constexpr size_t k_unset = 77777;  // what a delivered count holds before its call
constexpr float k_untouched = -7.0F;  // what host memory holds before a call writes it

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

/// `channels` in, `channels` out.
ModelConfig pass_through(int64_t channels) {
    ModelConfig model;
    model.add_model_path(anira_test::k_custom, "custom-processor");
    model.input(streamed("in", channels));
    model.output(streamed("out", channels));
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

// ---- a host block ------------------------------------------------------------------------------

/// How a block's memory is described. Every description names the same logical [channels,
/// samples] block.
enum class Layout : uint8_t { Planar, Contiguous, Packed, Interleaved };

constexpr std::array<Layout, 4> k_layouts{Layout::Planar,
                                          Layout::Contiguous,
                                          Layout::Packed,
                                          Layout::Interleaved};

const char* layout_name(Layout layout) {
    switch (layout) {
        case Layout::Planar: return "planar";
        case Layout::Contiguous: return "contiguous";
        case Layout::Packed: return "packed";
        case Layout::Interleaved: return "interleaved";
    }
    return "?";
}

/// Float memory and the tensor that describes it, built the way a host builds one: through
/// anira_tensor_init_host or anira_tensor_init_host_planar, the strides assigned afterwards.
class Block {
public:
    Block(Layout layout, size_t channels, size_t samples, anira_dtype dtype = ANIRA_DTYPE_F32)
        : m_interleaved(layout == Layout::Interleaved)
        , m_channels(channels)
        , m_samples(samples)
        , m_data(channels * samples, k_untouched) {
        const std::array<int64_t, 2> shape{static_cast<int64_t>(channels),
                                           static_cast<int64_t>(samples)};
        if (layout == Layout::Planar) {
            for (size_t channel = 0; channel < channels; ++channel) {
                m_planes.push_back(m_data.data() + (channel * samples));
            }
            anira_tensor_init_host_planar(&m_tensor,
                                          static_cast<const void*>(m_planes.data()),
                                          static_cast<uint32_t>(channels),
                                          dtype,
                                          2,
                                          shape.data());
            return;
        }
        anira_tensor_init_host(&m_tensor, m_data.data(), dtype, 2, shape.data());
        if (layout == Layout::Contiguous) {
            m_tensor.strides[0] = static_cast<int64_t>(samples);
            m_tensor.strides[1] = 1;
        } else if (layout == Layout::Interleaved) {
            m_tensor.strides[0] = 1;
            m_tensor.strides[1] = static_cast<int64_t>(channels);
        }
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(Block&&) = delete;
    ~Block() = default;

    float& at(size_t channel, size_t sample) {
        return m_data[m_interleaved ? (sample * m_channels) + channel
                                    : (channel * m_samples) + sample];
    }
    float at(size_t channel, size_t sample) const {
        return m_data[m_interleaved ? (sample * m_channels) + channel
                                    : (channel * m_samples) + sample];
    }
    size_t channels() const { return m_channels; }
    size_t samples() const { return m_samples; }

    const anira_tensor& tensor() const { return m_tensor; }
    /// For the refusal cases, which break one field of a well-formed descriptor.
    anira_tensor& descriptor() { return m_tensor; }

private:
    bool m_interleaved;
    size_t m_channels;
    size_t m_samples;
    std::vector<float> m_data;
    std::vector<float*> m_planes;
    anira_tensor m_tensor{};
};

/// The bytes of a descriptor: what a call must leave as it found it.
std::array<unsigned char, sizeof(anira_tensor)> bytes_of(const anira_tensor& tensor) {
    std::array<unsigned char, sizeof(anira_tensor)> bytes{};
    std::memcpy(bytes.data(), &tensor, sizeof(anira_tensor));
    return bytes;
}

/// Sample `position` of channel `channel` of the input stream: every channel its own ramp.
float stream_value(size_t channel, size_t position) {
    return static_cast<float>((channel * 1000) + position + 1);
}

/// The output stream of a pass-through model: the input, late by the latency.
float expected_value(size_t channel, size_t position, size_t latency) {
    return position < latency ? 0.0F : stream_value(channel, position - latency);
}

// ---- the rig -----------------------------------------------------------------------------------

class Rig {
public:
    explicit Rig(const ModelConfig& model,
                 anira_miss_policy policy = ANIRA_MISS_ZEROS,
                 uint32_t threads = 2)
        : m_context(threads), m_handler(m_context, model, m_candidates) {
        // Blocks of 1 to k_hop samples.
        anira::ContractHandle contract =
            anira_test::explicit_contract(k_hop, k_rate, policy, 0.0, 1.0);
        contract.hard_geometry(1, k_hop, k_rate);
        const anira_status prepared = m_handler.prepare(contract);
        EXPECT_EQ(prepared, ANIRA_OK) << m_handler.m_err.message;
        if (prepared != ANIRA_OK) { return; }
        m_gate = std::make_unique<anira_test::GateBackend>(get()->m_inference_config);
        m_session = anira_test::session_of(get());
        if (m_session == nullptr) { return; }
        m_session->m_custom_processor = m_gate.get();  // attach_processor, the gate closed
        m_gate->m_open.store(false);
    }
    /// The gate opens and the handler is destroyed (which drains the in-flight work) before
    /// the gate dies: what anira_test::DestroyFirst does for a stack backend.
    ~Rig() {
        if (m_gate != nullptr) { m_gate->m_open.store(true); }
        m_handler.destroy();
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;
    Rig(Rig&&) = delete;
    Rig& operator=(Rig&&) = delete;

    bool ready() const { return m_session != nullptr; }
    anira_handler* get() const { return m_handler.m_handler; }
    anira::SessionElement& session() const { return *m_session; }
    size_t latency(uint32_t slot = 0) const { return anira_handler_get_latency(get(), slot); }

    /// Opens the gate until every submitted inference is collected.
    void settle() {
        oracle::settle(*m_gate, *m_session, [this] { anira_test::available(get()); });
    }
    /// A _wait twin waits for its own block, so the gate is open while it runs.
    void open_gate(bool open) { m_gate->m_open.store(open); }

private:
    anira_test::Context m_context;
    /// The custom rows only: the models name no engine.
    std::vector<anira_backend_id> m_candidates{{.struct_size = sizeof(anira_backend_id),
                                                .engine = ANIRA_ENGINE_NONE,
                                                .provider = ANIRA_PROVIDER_DEFAULT,
                                                .engine_id = nullptr}};
    anira_test::Handler m_handler;
    std::unique_ptr<anira_test::GateBackend> m_gate;
    std::shared_ptr<anira::SessionElement> m_session;
};

// ---- one stream through one form ---------------------------------------------------------------

/// The entry a stream runs through.
enum class Form : uint8_t {
    Process,
    ProcessMulti,
    PushPop,
    PushPopMulti,
    ProcessWait,
    PushPopWait
};

constexpr std::array<Form, 4> k_nonblocking_forms{Form::Process,
                                                  Form::ProcessMulti,
                                                  Form::PushPop,
                                                  Form::PushPopMulti};

const char* form_name(Form form) {
    switch (form) {
        case Form::Process: return "process";
        case Form::ProcessMulti: return "process_multi";
        case Form::PushPop: return "push_data + pop_data";
        case Form::PushPopMulti: return "push_data_multi + pop_data_multi";
        case Form::ProcessWait: return "process_wait";
        case Form::PushPopWait: return "push_data + pop_data_wait";
    }
    return "?";
}

struct Outcome {
    anira_status m_status = ANIRA_OK;
    size_t m_delivered = k_unset;
};

/// One block of a one-slot model through `form`. `in` and `out` may be the same tensor.
Outcome call(Rig& rig, Form form, const anira_tensor& in, const anira_tensor& out) {
    anira_handler* handler = rig.get();
    Outcome outcome;
    switch (form) {
        case Form::Process:
            outcome.m_status = anira_handler_process(handler, &in, &out, 0, &outcome.m_delivered);
            break;
        case Form::ProcessMulti:
            outcome.m_status =
                anira_handler_process_multi(handler, &in, 1, &out, 1, &outcome.m_delivered);
            break;
        case Form::PushPop:
            EXPECT_EQ(anira_handler_push_data(handler, &in, 0), ANIRA_OK);
            outcome.m_status = anira_handler_pop_data(handler, &out, 0, &outcome.m_delivered);
            break;
        case Form::PushPopMulti:
            EXPECT_EQ(anira_handler_push_data_multi(handler, &in, 1), ANIRA_OK);
            outcome.m_status = anira_handler_pop_data_multi(handler, &out, 1, &outcome.m_delivered);
            break;
        case Form::ProcessWait:
            rig.open_gate(true);
            outcome.m_status = anira_handler_process_wait(handler,
                                                          &in,
                                                          &out,
                                                          ANIRA_WAIT_FOREVER,
                                                          0,
                                                          &outcome.m_delivered);
            rig.open_gate(false);
            break;
        case Form::PushPopWait:
            rig.open_gate(true);
            EXPECT_EQ(anira_handler_push_data(handler, &in, 0), ANIRA_OK);
            outcome.m_status = anira_handler_pop_data_wait(handler,
                                                           &out,
                                                           ANIRA_WAIT_FOREVER,
                                                           0,
                                                           &outcome.m_delivered);
            rig.open_gate(false);
            break;
    }
    return outcome;
}

/// Fills `block` with the next `block.samples()` samples of the input stream.
void fill_input(Block& block, size_t position) {
    for (size_t channel = 0; channel < block.channels(); ++channel) {
        for (size_t i = 0; i < block.samples(); ++i) {
            block.at(channel, i) = stream_value(channel, position + i);
        }
    }
}

/// `calls` delivered blocks of `samples` through `form`, the input described as `in_layout`
/// and the output as `out_layout`: every channel comes back as its own ramp, late by the
/// latency, and both descriptors are left byte for byte as they were.
void expect_delayed_stream(size_t channels,
                           Layout in_layout,
                           Layout out_layout,
                           Form form,
                           size_t calls = 5,
                           size_t samples = k_hop) {
    const std::string what = std::string(form_name(form)) + ", " + std::to_string(channels) +
                             " channels, " + layout_name(in_layout) + " in, " +
                             layout_name(out_layout) + " out";
    SCOPED_TRACE(what);
    Rig rig(pass_through(static_cast<int64_t>(channels)));
    ASSERT_TRUE(rig.ready());
    const size_t latency = rig.latency();
    ASSERT_GE(latency, samples) << "the first block would starve";
    size_t position = 0;
    for (size_t k = 0; k < calls; ++k) {
        Block in(in_layout, channels, samples);
        Block out(out_layout, channels, samples);
        fill_input(in, position);
        const auto in_before = bytes_of(in.tensor());
        const auto out_before = bytes_of(out.tensor());
        const Outcome outcome = call(rig, form, in.tensor(), out.tensor());
        EXPECT_EQ(outcome.m_status, ANIRA_OK) << "block " << k;
        EXPECT_EQ(outcome.m_delivered, samples) << "block " << k;
        EXPECT_EQ(bytes_of(in.tensor()), in_before) << "the input descriptor was written";
        EXPECT_EQ(bytes_of(out.tensor()), out_before) << "the output descriptor was written";
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t i = 0; i < samples; ++i) {
                ASSERT_EQ(out.at(channel, i), expected_value(channel, position + i, latency))
                    << "block " << k << ", channel " << channel << ", sample " << i;
                ASSERT_EQ(in.at(channel, i), stream_value(channel, position + i))
                    << "the input memory was written";
            }
        }
        position += samples;
        rig.settle();
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);
}

// ---- absolute expectations per description -----------------------------------------------------

TEST(AbiHandlerTensor, EveryDescriptionDeliversEachChannelsRampLateByTheLatency) {
    for (const Form form : k_nonblocking_forms) {
        for (const Layout layout : k_layouts) { expect_delayed_stream(2, layout, layout, form); }
    }
}

TEST(AbiHandlerTensor, TheRuleHoldsForAnyChannelCount) {
    for (const size_t channels : {1U, 3U, 6U}) {
        for (const Layout layout : k_layouts) {
            expect_delayed_stream(channels, layout, layout, Form::Process, 4);
        }
    }
}

TEST(AbiHandlerTensor, TheTwoSidesOfACallMayBeDescribedDifferently) {
    for (const Layout in_layout : k_layouts) {
        for (const Layout out_layout : k_layouts) {
            if (in_layout == out_layout) { continue; }
            expect_delayed_stream(3, in_layout, out_layout, Form::Process, 4);
        }
    }
    expect_delayed_stream(2, Layout::Interleaved, Layout::Planar, Form::PushPopMulti, 4);
    expect_delayed_stream(2, Layout::Planar, Layout::Interleaved, Form::ProcessMulti, 4);
}

TEST(AbiHandlerTensor, ABlockBelowTheHopIsCarriedSampleBySample) {
    expect_delayed_stream(2, Layout::Interleaved, Layout::Interleaved, Form::Process, 9, 3);
    expect_delayed_stream(2, Layout::Planar, Layout::Contiguous, Form::PushPop, 9, 5);
}

TEST(AbiHandlerTensor, TheWaitTwinsDeliverTheSameStream) {
    for (const Form form : {Form::ProcessWait, Form::PushPopWait}) {
        expect_delayed_stream(2, Layout::Interleaved, Layout::Planar, form, 4);
        expect_delayed_stream(2, Layout::Packed, Layout::Interleaved, form, 4);
    }
}

// ---- in place ----------------------------------------------------------------------------------

TEST(AbiHandlerTensor, OneTensorOnBothSidesIsInPlace) {
    for (const Layout layout : k_layouts) {
        SCOPED_TRACE(layout_name(layout));
        Rig rig(pass_through(2));
        ASSERT_TRUE(rig.ready());
        const size_t latency = rig.latency();
        size_t position = 0;
        for (size_t k = 0; k < 4; ++k) {
            Block block(layout, 2, k_hop);
            fill_input(block, position);
            const auto before = bytes_of(block.tensor());
            size_t delivered = k_unset;
            ASSERT_EQ(
                anira_handler_process(rig.get(), &block.tensor(), &block.tensor(), 0, &delivered),
                ANIRA_OK);
            EXPECT_EQ(delivered, k_hop);
            EXPECT_EQ(bytes_of(block.tensor()), before);
            for (size_t channel = 0; channel < 2; ++channel) {
                for (size_t i = 0; i < k_hop; ++i) {
                    ASSERT_EQ(block.at(channel, i), expected_value(channel, position + i, latency))
                        << "block " << k << ", channel " << channel << ", sample " << i;
                }
            }
            position += k_hop;
            rig.settle();
        }
    }
}

// ---- the multi forms with a Static slot, the single forms beside unused slots ------------------

/// The Static input values of the multi model, the same in every call.
constexpr std::array<float, 3> k_static_in{0.5F, 1.5F, 2.5F};

void fill_static(Block& block) {
    for (size_t i = 0; i < k_static_in.size(); ++i) { block.at(0, i) = k_static_in.at(i); }
}

TEST(AbiHandlerTensor, ProcessMultiCarriesStaticSlotsAndReportsTheClampedCount) {
    Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    const size_t latency = rig.latency(0);
    size_t position = 0;
    for (size_t k = 0; k < 4; ++k) {
        Block stream_in(Layout::Interleaved, 3, k_hop);
        Block values_in(Layout::Packed, 1, 3);
        Block stream_out(Layout::Planar, 3, k_hop);
        Block values_out(Layout::Packed, 1, 5);  // two values more than the slot holds
        fill_input(stream_in, position);
        fill_static(values_in);
        const std::array<anira_tensor, 2> inputs{stream_in.tensor(), values_in.tensor()};
        const std::array<anira_tensor, 2> outputs{stream_out.tensor(), values_out.tensor()};
        const auto inputs_before = std::array{bytes_of(inputs[0]), bytes_of(inputs[1])};
        const auto outputs_before = std::array{bytes_of(outputs[0]), bytes_of(outputs[1])};
        std::array<size_t, 2> delivered{k_unset, k_unset};
        // The delivered array is nullable: every second call runs without one.
        ASSERT_EQ(anira_handler_process_multi(rig.get(),
                                              inputs.data(),
                                              2,
                                              outputs.data(),
                                              2,
                                              k % 2 == 0 ? delivered.data() : nullptr),
                  ANIRA_OK);
        if (k % 2 == 0) {
            EXPECT_EQ(delivered[0], k_hop);
            EXPECT_EQ(delivered[1], 3U) << "a Static output reports its clamped count";
        }
        EXPECT_EQ(bytes_of(inputs[0]), inputs_before[0]);
        EXPECT_EQ(bytes_of(inputs[1]), inputs_before[1]);
        EXPECT_EQ(bytes_of(outputs[0]), outputs_before[0]);
        EXPECT_EQ(bytes_of(outputs[1]), outputs_before[1]);
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < k_hop; ++i) {
                ASSERT_EQ(stream_out.at(channel, i), expected_value(channel, position + i, latency))
                    << "block " << k << ", channel " << channel << ", sample " << i;
            }
        }
        if (k > 0) {  // an inference has completed: the model passed the values through
            for (size_t i = 0; i < 3; ++i) { EXPECT_EQ(values_out.at(0, i), k_static_in.at(i)); }
        }
        EXPECT_EQ(values_out.at(0, 3), k_untouched) << "written past the slot's value count";
        EXPECT_EQ(values_out.at(0, 4), k_untouched);
        position += k_hop;
        rig.settle();
    }
}

TEST(AbiHandlerTensor, ASingleTensorFormLeavesTheOtherSlotsEmpty) {
    Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    anira_handler* handler = rig.get();
    const size_t latency = rig.latency(0);
    size_t position = 0;
    for (size_t k = 0; k < 3; ++k) {
        // Slot 1 first, then slot 0: each call carries one slot, the other stays empty.
        Block values_in(Layout::Packed, 1, 3);
        fill_static(values_in);
        ASSERT_EQ(anira_handler_push_data(handler, &values_in.tensor(), 1), ANIRA_OK);
        Block stream_in(Layout::Interleaved, 3, k_hop);
        fill_input(stream_in, position);
        ASSERT_EQ(anira_handler_push_data(handler, &stream_in.tensor(), 0), ANIRA_OK);
        for (const anira_tensor& slot : handler->m_input_tensors) {
            EXPECT_EQ(slot.shape[1], 0) << "a staged slot is an empty tensor again";
        }

        const Block values_out(Layout::Packed, 1, 5);
        size_t delivered = k_unset;
        ASSERT_EQ(anira_handler_pop_data(handler, &values_out.tensor(), 1, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, 3U);
        Block stream_out(Layout::Contiguous, 3, k_hop);
        delivered = k_unset;
        ASSERT_EQ(anira_handler_pop_data(handler, &stream_out.tensor(), 0, &delivered), ANIRA_OK);
        EXPECT_EQ(delivered, k_hop);
        for (const anira_tensor& slot : handler->m_output_tensors) {
            EXPECT_EQ(slot.shape[1], 0) << "a staged slot is an empty tensor again";
        }
        for (size_t channel = 0; channel < 3; ++channel) {
            for (size_t i = 0; i < k_hop; ++i) {
                ASSERT_EQ(stream_out.at(channel, i), expected_value(channel, position + i, latency))
                    << "block " << k << ", channel " << channel << ", sample " << i;
            }
        }
        position += k_hop;
        rig.settle();
    }
    EXPECT_EQ(anira_handler_rt_error(handler), ANIRA_OK);
}

TEST(AbiHandlerTensor, AnEmptyTensorWithoutMemoryLeavesItsSlotOut) {
    const Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    // The empty tensor of a slot: rank 2, {channels, 0}, the slot's dtype, no memory; with
    // and without the planar flag.
    const std::array<int64_t, 2> no_values{1, 0};
    const std::array<int64_t, 2> no_samples{3, 0};
    std::array<anira_tensor, 2> inputs{};
    std::array<anira_tensor, 2> outputs{};
    anira_tensor_init_host(inputs.data(), nullptr, ANIRA_DTYPE_F32, 2, no_samples.data());
    anira_tensor_init_host(&inputs[1], nullptr, ANIRA_DTYPE_F32, 2, no_values.data());
    anira_tensor_init_host_planar(outputs.data(),
                                  nullptr,
                                  3,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  no_samples.data());
    anira_tensor_init_host(&outputs[1], nullptr, ANIRA_DTYPE_F32, 2, no_values.data());
    std::array<size_t, 2> delivered{k_unset, k_unset};
    EXPECT_EQ(anira_handler_process_multi(rig.get(),
                                          inputs.data(),
                                          2,
                                          outputs.data(),
                                          2,
                                          delivered.data()),
              ANIRA_OK);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_EQ(delivered[1], 0U);
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);

    // One carried slot beside an empty one.
    const Block stream_out(Layout::Interleaved, 3, k_hop);
    outputs[0] = stream_out.tensor();
    EXPECT_EQ(anira_handler_pop_data_multi(rig.get(), outputs.data(), 2, delivered.data()),
              ANIRA_OK);
    EXPECT_EQ(delivered[0], k_hop);
    EXPECT_EQ(delivered[1], 0U);
}

// ---- the miss policies -------------------------------------------------------------------------

/// Three blocks: a delivered one, a delivered one whose inference stays at the gate, and the
/// starved one, which holds what the policy says.
void expect_missed_block(anira_miss_policy policy, Layout in_layout, Layout out_layout, Form form) {
    SCOPED_TRACE(std::string(form_name(form)) + ", " + layout_name(in_layout) + " in, " +
                 layout_name(out_layout) + " out, policy " + std::to_string(policy));
    Rig rig(pass_through(2), policy);
    ASSERT_TRUE(rig.ready());
    const size_t latency = rig.latency();
    ASSERT_LT(latency, 2U * k_hop) << "the third block would not starve";
    std::vector<std::vector<float>> last_delivered(2, std::vector<float>(k_hop, 0.0F));
    size_t position = 0;
    for (size_t k = 0; k < 3; ++k) {
        Block in(in_layout, 2, k_hop);
        Block out(out_layout, 2, k_hop);
        fill_input(in, position);
        const Outcome outcome = call(rig, form, in.tensor(), out.tensor());
        if (k < 2) {
            ASSERT_EQ(outcome.m_status, ANIRA_OK);
            ASSERT_EQ(outcome.m_delivered, k_hop);
            for (size_t channel = 0; channel < 2; ++channel) {
                for (size_t i = 0; i < k_hop; ++i) {
                    last_delivered[channel][i] = out.at(channel, i);
                    ASSERT_EQ(out.at(channel, i), expected_value(channel, position + i, latency));
                }
            }
            if (k == 0) { rig.settle(); }  // block 1 stays at the gate, block 2 starves
        } else {
            EXPECT_EQ(outcome.m_status, ANIRA_MISSED);
            EXPECT_EQ(outcome.m_delivered, 0U);
            const bool has_input = form == Form::Process || form == Form::ProcessMulti;
            for (size_t channel = 0; channel < 2; ++channel) {
                for (size_t i = 0; i < k_hop; ++i) {
                    float expected = 0.0F;
                    if (policy == ANIRA_MISS_HOLD_LAST) { expected = last_delivered[channel][i]; }
                    if (policy == ANIRA_MISS_BYPASS && has_input) {
                        expected = stream_value(channel, position + i);
                    }
                    ASSERT_EQ(out.at(channel, i), expected)
                        << "channel " << channel << ", sample " << i;
                }
            }
        }
        position += k_hop;
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK) << "a miss is not recorded";
}

TEST(AbiHandlerTensor, AMissedBlockFollowsThePolicyUnderEveryDescription) {
    for (const anira_miss_policy policy :
         {ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST, ANIRA_MISS_BYPASS}) {
        expect_missed_block(policy, Layout::Interleaved, Layout::Interleaved, Form::Process);
        expect_missed_block(policy, Layout::Interleaved, Layout::Planar, Form::ProcessMulti);
        expect_missed_block(policy, Layout::Planar, Layout::Interleaved, Form::Process);
        expect_missed_block(policy, Layout::Contiguous, Layout::Interleaved, Form::PushPop);
        expect_missed_block(policy, Layout::Packed, Layout::Planar, Form::PushPopMulti);
    }
}

TEST(AbiHandlerTensor, ABypassMissLeavesAnInPlaceTensorWhereItIs) {
    for (const Layout layout : k_layouts) {
        SCOPED_TRACE(layout_name(layout));
        Rig rig(pass_through(2), ANIRA_MISS_BYPASS);
        ASSERT_TRUE(rig.ready());
        ASSERT_LT(rig.latency(), 2U * k_hop) << "the third block would not starve";
        size_t position = 0;
        for (size_t k = 0; k < 3; ++k) {
            Block block(layout, 2, k_hop);
            fill_input(block, position);
            size_t delivered = k_unset;
            const anira_status status =
                anira_handler_process(rig.get(), &block.tensor(), &block.tensor(), 0, &delivered);
            if (k == 0) { rig.settle(); }
            if (k == 2) {
                EXPECT_EQ(status, ANIRA_MISSED);
                EXPECT_EQ(delivered, 0U);
                for (size_t channel = 0; channel < 2; ++channel) {
                    for (size_t i = 0; i < k_hop; ++i) {
                        ASSERT_EQ(block.at(channel, i), stream_value(channel, position + i));
                    }
                }
            } else {
                EXPECT_EQ(status, ANIRA_OK);
            }
            position += k_hop;
        }
    }
}

// ---- the refusals ------------------------------------------------------------------------------

TEST(AbiHandlerTensor, ANullHandlerIsRefusedAndTheCountsAreZeroed) {
    const Block block(Layout::Planar, 2, k_hop);
    const anira_tensor* tensor = &block.tensor();
    size_t delivered = k_unset;
    std::array<size_t, 2> counts{k_unset, k_unset};
    EXPECT_EQ(anira_handler_process(nullptr, tensor, tensor, 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_process_multi(nullptr, tensor, 1, tensor, 2, counts.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(counts[0], 0U);
    EXPECT_EQ(counts[1], 0U) << "a multi form zeroes the num_outputs counts the caller announced";
    EXPECT_EQ(anira_handler_push_data(nullptr, tensor, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_push_data_multi(nullptr, tensor, 1), ANIRA_ERROR_INVALID_ARGUMENT);
    delivered = k_unset;
    EXPECT_EQ(anira_handler_pop_data(nullptr, tensor, 0, &delivered), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    counts = {k_unset, k_unset};
    EXPECT_EQ(anira_handler_pop_data_multi(nullptr, tensor, 1, counts.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(counts[0], 0U);
    EXPECT_EQ(counts[1], k_unset) << "one count was announced";
    delivered = k_unset;
    EXPECT_EQ(anira_handler_process_wait(nullptr, tensor, tensor, 0.0, 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_process_multi_wait(nullptr, tensor, 1, tensor, 1, nullptr, 0.0),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_wait(nullptr, tensor, 0.0, 0, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi_wait(nullptr, tensor, 1, nullptr, 0.0),
              ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiHandlerTensor, UnpreparedEntriesRecordNotPrepared) {
    const anira_test::Context context;
    anira_drain_log();
    RecordCollector collector;
    const ModelConfig model = pass_through(2);
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_NONE,
                                                    .provider = ANIRA_PROVIDER_DEFAULT,
                                                    .engine_id = nullptr}};
    const anira_test::Handler handler(context, model, candidates);
    anira_handler* h = handler.m_handler;
    ASSERT_NE(h, nullptr);
    const Block block(Layout::Interleaved, 2, k_hop);
    const anira_tensor* tensor = &block.tensor();
    size_t delivered = k_unset;
    EXPECT_EQ(anira_handler_process(h, tensor, tensor, 0, &delivered), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_process_multi(h, tensor, 1, tensor, 1, nullptr),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data(h, tensor, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data_multi(h, tensor, 1), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data(h, tensor, 0, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data_multi(h, tensor, 1, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_process_wait(h, tensor, tensor, ANIRA_WAIT_FOREVER, 0, nullptr),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(
        anira_handler_process_multi_wait(h, tensor, 1, tensor, 1, nullptr, ANIRA_WAIT_FOREVER),
        ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data_wait(h, tensor, ANIRA_WAIT_FOREVER, 0, nullptr),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data_multi_wait(h, tensor, 1, nullptr, ANIRA_WAIT_FOREVER),
              ANIRA_ERROR_NOT_PREPARED);
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "handler not prepared", "rt"), 1U)
        << "the kind is latched: the later refusals are suppressed";
    EXPECT_EQ(anira_test::find_record(collector, "handler not prepared", "rt").m_message,
              "anira_handler_process: handler not prepared");
#endif
}

/// What one refusal case expects.
struct Refusal {
    const char* m_what;
    anira_status m_status;
    void (*m_break)(anira_tensor& tensor);
    bool m_output_only = false;  ///< the rule holds for an output tensor only
};

// Every way decision 4 names in which a descriptor is malformed, one field broken at a time
// on an interleaved stereo block of k_hop samples (strides {1, 2}).
const std::array<Refusal, 13> k_refusals{{
    {.m_what = "a zeroed tensor",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor = anira_tensor{}; }},
    {.m_what = "rank 1",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.ndim = 1; }},
    {.m_what = "rank 3",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.ndim = 3; }},
    {.m_what = "a device domain",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.domain = ANIRA_DOMAIN_CUDA; }},
    {.m_what = "an unknown flag bit",
     .m_status = ANIRA_ERROR_NOT_SUPPORTED,
     .m_break = [](anira_tensor& tensor) { tensor.flags |= 0x100U; }},
    {.m_what = "a read-only output",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.flags |= ANIRA_TENSOR_READ_ONLY; },
     .m_output_only = true},
    {.m_what = "shape[0] other than the slot's channels",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.shape[0] = 1; }},
    {.m_what = "a negative shape[1]",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.shape[1] = -1; }},
    {.m_what = "another dtype",
     .m_status = ANIRA_ERROR_CONFIG,
     .m_break = [](anira_tensor& tensor) { tensor.dtype = ANIRA_DTYPE_I16; }},
    {.m_what = "NULL memory",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.handle.host.ptr = nullptr; }},
    {.m_what = "a negative stride",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.strides[1] = -2; }},
    {.m_what = "a stride of 0 over more than one sample",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.strides[1] = 0; }},
    {.m_what = "a run that does not start on an element",
     .m_status = ANIRA_ERROR_INVALID_ARGUMENT,
     .m_break = [](anira_tensor& tensor) { tensor.byte_offset = 1; }},
}};

TEST(AbiHandlerTensor, EveryMalformedTensorIsRefusedBeforeAnythingIsPushed) {
    const Rig rig(pass_through(2));
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    for (const Refusal& refusal : k_refusals) {
        SCOPED_TRACE(refusal.m_what);
        Block good(Layout::Interleaved, 2, k_hop);
        Block bad(Layout::Interleaved, 2, k_hop);
        fill_input(good, 0);
        fill_input(bad, 0);
        refusal.m_break(bad.descriptor());
        size_t delivered = k_unset;
        std::array<size_t, 1> counts{k_unset};

        if (!refusal.m_output_only) {
            anira_handler_reset(h);
            EXPECT_EQ(anira_handler_process(h, &bad.tensor(), &good.tensor(), 0, &delivered),
                      refusal.m_status);
            EXPECT_EQ(delivered, 0U);
            EXPECT_EQ(anira_handler_rt_error(h), refusal.m_status);
            EXPECT_EQ(anira_handler_push_data(h, &bad.tensor(), 0), refusal.m_status);
            EXPECT_EQ(anira_handler_push_data_multi(h, &bad.tensor(), 1), refusal.m_status);
            EXPECT_EQ(
                anira_handler_process_multi(h, &bad.tensor(), 1, &good.tensor(), 1, counts.data()),
                refusal.m_status);
            EXPECT_EQ(counts[0], 0U);
        }
        anira_handler_reset(h);
        delivered = k_unset;
        EXPECT_EQ(anira_handler_process(h, &good.tensor(), &bad.tensor(), 0, &delivered),
                  refusal.m_status)
            << "the output is validated before the input is pushed";
        EXPECT_EQ(delivered, 0U);
        EXPECT_EQ(anira_handler_rt_error(h), refusal.m_status);
        delivered = k_unset;
        EXPECT_EQ(anira_handler_pop_data(h, &bad.tensor(), 0, &delivered), refusal.m_status);
        EXPECT_EQ(delivered, 0U);
        counts = {k_unset};
        EXPECT_EQ(anira_handler_pop_data_multi(h, &bad.tensor(), 1, counts.data()),
                  refusal.m_status);
        EXPECT_EQ(counts[0], 0U);
        EXPECT_EQ(anira_handler_process_wait(h, &good.tensor(), &bad.tensor(), 0.0, 0, nullptr),
                  refusal.m_status);
        EXPECT_EQ(anira_handler_pop_data_wait(h, &bad.tensor(), 0.0, 0, nullptr), refusal.m_status);

        EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U)
            << "a refused call pushed samples";
        EXPECT_EQ(rig.session().m_receive_buffer[0].get_available_samples(0), rig.latency())
            << "a refused call popped samples";
        for (size_t i = 0; i < k_hop; ++i) {
            ASSERT_EQ(good.at(0, i), stream_value(0, i)) << "a refused call wrote host memory";
        }
    }
}

TEST(AbiHandlerTensor, APlanarTensorIsCheckedPlaneByPlane) {
    const Rig rig(pass_through(2));
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    const Block good(Layout::Planar, 2, k_hop);
    std::array<float, k_hop> samples{};
    std::array<float*, 2> planes{samples.data(), nullptr};
    const std::array<int64_t, 2> shape{2, k_hop};
    anira_tensor tensor;
    anira_tensor_init_host_planar(&tensor,
                                  static_cast<const void*>(planes.data()),
                                  2,
                                  ANIRA_DTYPE_F32,
                                  2,
                                  shape.data());
    EXPECT_EQ(anira_handler_push_data(h, &tensor, 0), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a NULL plane";
    planes[1] = samples.data();
    tensor.handle.planes.count = 1;
    EXPECT_EQ(anira_handler_push_data(h, &tensor, 0), ANIRA_ERROR_INVALID_ARGUMENT)
        << "a plane count other than shape[0]";
    tensor.handle.planes.count = 2;
    tensor.handle.planes.ptrs = nullptr;
    EXPECT_EQ(anira_handler_push_data(h, &tensor, 0), ANIRA_ERROR_INVALID_ARGUMENT)
        << "no plane array";
    // A planar tensor on a device domain is refused by the domain, before the flag is read.
    anira_tensor device = good.tensor();
    device.domain = ANIRA_DOMAIN_CUDA;
    device.flags |= 0x100U;
    anira_handler_reset(h);
    EXPECT_EQ(anira_handler_push_data(h, &device, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
}

TEST(AbiHandlerTensor, TheCheckOrderIsPinnedByTensorsWrongInTwoWays) {
    const Rig rig(pass_through(2));
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    struct Pair {
        const char* m_what;
        void (*m_break)(anira_tensor& tensor);
        anira_status m_status;
    };
    // rank, domain, flags, shape, dtype, memory and strides: the earlier one is reported.
    const std::array<Pair, 5> pairs{{
        {.m_what = "rank before flags",
         .m_break =
             [](anira_tensor& tensor) {
                 tensor.ndim = 1;
                 tensor.flags |= 0x100U;
             },
         .m_status = ANIRA_ERROR_INVALID_ARGUMENT},
        {.m_what = "domain before flags",
         .m_break =
             [](anira_tensor& tensor) {
                 tensor.domain = ANIRA_DOMAIN_GL_BUFFER;
                 tensor.flags |= 0x100U;
             },
         .m_status = ANIRA_ERROR_INVALID_ARGUMENT},
        {.m_what = "flags before shape",
         .m_break =
             [](anira_tensor& tensor) {
                 tensor.flags |= 0x100U;
                 tensor.shape[0] = 7;
             },
         .m_status = ANIRA_ERROR_NOT_SUPPORTED},
        {.m_what = "shape before dtype",
         .m_break =
             [](anira_tensor& tensor) {
                 tensor.shape[0] = 7;
                 tensor.dtype = ANIRA_DTYPE_I16;
             },
         .m_status = ANIRA_ERROR_INVALID_ARGUMENT},
        {.m_what = "dtype before memory",
         .m_break =
             [](anira_tensor& tensor) {
                 tensor.dtype = ANIRA_DTYPE_I16;
                 tensor.handle.host.ptr = nullptr;
             },
         .m_status = ANIRA_ERROR_CONFIG},
    }};
    for (const Pair& pair : pairs) {
        SCOPED_TRACE(pair.m_what);
        Block block(Layout::Interleaved, 2, k_hop);
        pair.m_break(block.descriptor());
        EXPECT_EQ(anira_handler_push_data(h, &block.tensor(), 0), pair.m_status);
    }
    // In a call the input is checked before the output, and a multi form walks its slots in
    // order: the first refusal is the one reported.
    const Block wrong_dtype(Layout::Interleaved, 2, k_hop, ANIRA_DTYPE_I16);
    Block wrong_rank(Layout::Interleaved, 2, k_hop);
    wrong_rank.descriptor().ndim = 1;
    EXPECT_EQ(anira_handler_process(h, &wrong_dtype.tensor(), &wrong_rank.tensor(), 0, nullptr),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_handler_process(h, &wrong_rank.tensor(), &wrong_dtype.tensor(), 0, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiHandlerTensor, AZeroedTensorInOneSlotRefusesTheWholeMultiCall) {
    const Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    Block stream_in(Layout::Interleaved, 3, k_hop);
    Block stream_out(Layout::Interleaved, 3, k_hop);
    const Block values_out(Layout::Packed, 1, 3);
    fill_input(stream_in, 0);
    // What a refused void factory leaves in slot 1: a zeroed record.
    const std::array<anira_tensor, 2> inputs{stream_in.tensor(), anira_tensor{}};
    const std::array<anira_tensor, 2> outputs{stream_out.tensor(), values_out.tensor()};
    std::array<size_t, 2> delivered{k_unset, k_unset};
    EXPECT_EQ(anira_handler_process_multi(h, inputs.data(), 2, outputs.data(), 2, delivered.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_EQ(delivered[1], 0U);
    EXPECT_EQ(anira_handler_push_data_multi(h, inputs.data(), 2), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U)
        << "slot 0 was pushed although slot 1 is malformed";
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(stream_out.at(0, i), k_untouched); }
    // A zeroed output slot: nothing is pushed either.
    const std::array<anira_tensor, 2> good_inputs{stream_in.tensor(), values_out.tensor()};
    const std::array<anira_tensor, 2> bad_outputs{stream_out.tensor(), anira_tensor{}};
    EXPECT_EQ(anira_handler_process_multi(h, good_inputs.data(), 2, bad_outputs.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
}

TEST(AbiHandlerTensor, TheArgumentsAreRefused) {
    const Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    Block stream(Layout::Interleaved, 3, k_hop);
    const Block values(Layout::Packed, 1, 3);
    fill_input(stream, 0);
    const std::array<anira_tensor, 2> tensors{stream.tensor(), values.tensor()};
    size_t delivered = k_unset;
    std::array<size_t, 2> counts{k_unset, k_unset};

    // A NULL tensor, an index out of range.
    EXPECT_EQ(anira_handler_process(h, nullptr, tensors.data(), 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_process(h, tensors.data(), nullptr, 0, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process(h, tensors.data(), tensors.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_push_data(h, nullptr, 0), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_push_data(h, tensors.data(), 2), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data(h, nullptr, 0, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data(h, tensors.data(), 2, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_wait(h, tensors.data(), 0.0, 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);

    // The multi forms take exactly one tensor per slot, Static slots included.
    EXPECT_EQ(anira_handler_process_multi(h, tensors.data(), 1, tensors.data(), 2, counts.data()),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(counts[0], 0U);
    EXPECT_EQ(counts[1], 0U);
    EXPECT_EQ(anira_handler_process_multi(h, tensors.data(), 2, tensors.data(), 1, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process_multi(h, tensors.data(), 3, tensors.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process_multi(h, nullptr, 2, tensors.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_push_data_multi(h, tensors.data(), 1), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_push_data_multi(h, nullptr, 2), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi(h, tensors.data(), 3, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi(h, nullptr, 2, nullptr), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(
        anira_handler_process_multi_wait(h, tensors.data(), 2, tensors.data(), 1, nullptr, 0.0),
        ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_pop_data_multi_wait(h, tensors.data(), 1, nullptr, 0.0),
              ANIRA_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
}

TEST(AbiHandlerTensor, ARefusalIsLoggedOncePerKindAndNamesTheSlot) {
    const anira_test::Context context(2, ANIRA_WAIT_SPIN_BACKOFF, ANIRA_LOG_DEBUG);
    anira_drain_log();
    RecordCollector collector;
    const ModelConfig model = multi_model();
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_NONE,
                                                    .provider = ANIRA_PROVIDER_DEFAULT,
                                                    .engine_id = nullptr}};
    anira_test::Handler handler(context, model, candidates);
    const anira::ContractHandle contract =
        anira_test::explicit_contract(k_hop, k_rate, ANIRA_MISS_ZEROS, 0.0, 1.0);
    ASSERT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    anira_handler* h = handler.m_handler;

    const Block stream(Layout::Interleaved, 3, k_hop);
    const Block values(Layout::Packed, 1, 3, ANIRA_DTYPE_I16);
    Block flagged(Layout::Interleaved, 3, k_hop);
    flagged.descriptor().flags |= 0x100U;
    Block wrong_rank(Layout::Interleaved, 3, k_hop);
    wrong_rank.descriptor().ndim = 3;
    for (int i = 0; i < 3; ++i) {
        const std::array<anira_tensor, 2> inputs{stream.tensor(), values.tensor()};
        EXPECT_EQ(anira_handler_push_data_multi(h, inputs.data(), 2), ANIRA_ERROR_CONFIG);
        EXPECT_EQ(anira_handler_pop_data(h, &flagged.tensor(), 0, nullptr),
                  ANIRA_ERROR_NOT_SUPPORTED);
        EXPECT_EQ(anira_handler_push_data(h, &wrong_rank.tensor(), 0),
                  ANIRA_ERROR_INVALID_ARGUMENT);
    }
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_ARGUMENT) << "last wins";
    anira_drain_log();
#ifdef ENABLE_LOGGING
    EXPECT_EQ(anira_test::count_records(collector, "nothing converts", "rt"), 1U);
    EXPECT_EQ(anira_test::find_record(collector, "nothing converts", "rt").m_message,
              "anira_handler_push_data_multi: the tensor of input slot 1 has dtype " +
                  std::to_string(ANIRA_DTYPE_I16) + ", the slot's is " +
                  std::to_string(ANIRA_DTYPE_F32) + "; nothing converts");
    EXPECT_EQ(anira_test::count_records(collector, "does not know", "rt"), 1U);
    EXPECT_NE(anira_test::find_record(collector, "does not know", "rt")
                  .m_message.find("anira_handler_pop_data: the tensor of output slot 0"),
              std::string::npos);
    EXPECT_EQ(anira_test::count_records(collector, "is malformed", "rt"), 1U);
    const RecordCollector::Record malformed =
        anira_test::find_record(collector, "is malformed", "rt");
    EXPECT_NE(malformed.m_message.find("anira_handler_push_data: the tensor of input slot 0 is "
                                       "malformed (rank 3,"),
              std::string::npos)
        << malformed.m_message;
    EXPECT_EQ(malformed.m_flags, ANIRA_LOG_RECORD_REALTIME | ANIRA_LOG_RECORD_CONTRACT_VIOLATION);
    EXPECT_EQ(malformed.m_group, "anira.capi");
#endif
}

// ---- the _wait twins without an inference thread -----------------------------------------------

TEST(AbiHandlerTensor, AWaitTwinWithoutAThreadRunsTheStemAndRefuses) {
    const Rig rig(multi_model(), ANIRA_MISS_ZEROS, /*threads=*/0);
    ASSERT_TRUE(rig.ready());
    anira_handler* h = rig.get();
    ASSERT_GE(rig.latency(0), 2U * k_hop);
    Block stream_in(Layout::Interleaved, 3, k_hop);
    Block values_in(Layout::Packed, 1, 3);
    Block stream_out(Layout::Planar, 3, k_hop);
    const Block values_out(Layout::Packed, 1, 5);
    fill_input(stream_in, 0);
    fill_static(values_in);
    const std::array<anira_tensor, 2> inputs{stream_in.tensor(), values_in.tensor()};
    const std::array<anira_tensor, 2> outputs{stream_out.tensor(), values_out.tensor()};

    // The stem delivers the latency's zeros; the counts are the stem's, the status the refusal.
    size_t delivered = k_unset;
    EXPECT_EQ(anira_handler_process_wait(h,
                                         inputs.data(),
                                         outputs.data(),
                                         ANIRA_WAIT_FOREVER,
                                         0,
                                         &delivered),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(delivered, k_hop);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_INVALID_STATE);
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(stream_out.at(1, i), 0.0F); }

    std::array<size_t, 2> counts{k_unset, k_unset};
    EXPECT_EQ(anira_handler_pop_data_multi_wait(h, outputs.data(), 2, counts.data(), 5.0),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(counts[0], k_hop);
    EXPECT_EQ(counts[1], 3U);

    // The ring is empty now: the stem misses, and the counts say so.
    delivered = k_unset;
    EXPECT_EQ(anira_handler_pop_data_wait(h, outputs.data(), ANIRA_WAIT_CONTRACT, 0, &delivered),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(delivered, 0U);
    counts = {k_unset, k_unset};
    EXPECT_EQ(anira_handler_process_multi_wait(h,
                                               inputs.data(),
                                               2,
                                               outputs.data(),
                                               2,
                                               counts.data(),
                                               ANIRA_WAIT_CONTRACT),
              ANIRA_ERROR_INVALID_STATE);
    EXPECT_EQ(counts[0], 0U);
    EXPECT_EQ(counts[1], 0U);
}

}  // namespace
