// The Hard entries over host tensors (anira/abi/handler.h under the bare names):
// anira_handler_process, push_data, pop_data, their _multi forms and the four _wait twins. A
// host block is one anira_tensor per slot, of the logical shape [channels, samples], over
// planar memory, one block read by strides (contiguous, interleaved) or one packed block.
//
// The tensor entries and the 2.x handler share one copy path, so a
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

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/InferenceHandler.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/log.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../../extras/models/model_files.h"
#include "../support/copy_oracle.h"
#include "../support/inference_config_eq.h"
#include "../support/log_record_collector.h"
#include "float_face.h"
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
                 uint32_t threads = 2,
                 anira_miss_fn miss_fn = nullptr,
                 void* miss_user_data = nullptr)
        : m_context(threads), m_handler(m_context, model, m_candidates) {
        // Blocks of 1 to k_hop samples.
        anira::ContractHandle contract =
            anira_test::explicit_contract(k_hop, k_rate, policy, 0.0, 1.0);
        contract.hard_geometry(1, k_hop, k_rate);
        contract.hard_miss_fn(miss_fn, miss_user_data);
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
            outcome.m_status =
                anira_handler_process(handler, &in, 0, &out, 0, &outcome.m_delivered);
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
                                                          0,
                                                          &out,
                                                          0,
                                                          ANIRA_WAIT_FOREVER,
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
            ASSERT_EQ(anira_handler_process(rig.get(),
                                            &block.tensor(),
                                            0,
                                            &block.tensor(),
                                            0,
                                            &delivered),
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

// A Static element of a multi form is the whole tensor in the spec's shape ([1, 3] here), what
// the two Static entries take: the model passes the values through, and the count of a carried
// Static output is its element count.
TEST(AbiHandlerTensor, ProcessMultiCarriesStaticSlotsAsWholeTensors) {
    Rig rig(multi_model());
    ASSERT_TRUE(rig.ready());
    const size_t latency = rig.latency(0);
    size_t position = 0;
    for (size_t k = 0; k < 4; ++k) {
        Block stream_in(Layout::Interleaved, 3, k_hop);
        Block values_in(Layout::Packed, 1, 3);
        Block stream_out(Layout::Planar, 3, k_hop);
        Block values_out(Layout::Packed, 1, 3);
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
            EXPECT_EQ(delivered[1], 3U) << "a Static output reports its element count";
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
        // Zeros until an inference has completed, then what the model passed through.
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(values_out.at(0, i), k > 0 ? k_static_in.at(i) : 0.0F) << "block " << k;
        }
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
        // The Static slot through its own entries, the Streamed slot through the single forms:
        // each of those carries one slot, the other stays empty.
        Block values_in(Layout::Packed, 1, 3);
        fill_static(values_in);
        ASSERT_EQ(anira_handler_set_static_input(handler, 1, &values_in.tensor()), ANIRA_OK);
        Block stream_in(Layout::Interleaved, 3, k_hop);
        fill_input(stream_in, position);
        ASSERT_EQ(anira_handler_push_data(handler, &stream_in.tensor(), 0), ANIRA_OK);
        for (const anira_tensor& slot : handler->m_input_tensors) {
            EXPECT_EQ(slot.shape[1], 0) << "a staged slot is an empty tensor again";
        }

        Block stream_out(Layout::Contiguous, 3, k_hop);
        size_t delivered = k_unset;
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
        // The inference the push submitted is still at the gate: the store holds what the
        // model produced last (nothing yet in block 0).
        const Block values_out(Layout::Packed, 1, 3);
        ASSERT_EQ(anira_handler_get_static_output(handler, 1, &values_out.tensor()), ANIRA_OK);
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(values_out.at(0, i), k > 0 ? k_static_in.at(i) : 0.0F) << "block " << k;
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
            const anira_status status = anira_handler_process(rig.get(),
                                                              &block.tensor(),
                                                              0,
                                                              &block.tensor(),
                                                              0,
                                                              &delivered);
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
    EXPECT_EQ(anira_handler_process(nullptr, tensor, 0, tensor, 0, &delivered),
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
    EXPECT_EQ(anira_handler_process_wait(nullptr, tensor, 0, tensor, 0, 0.0, &delivered),
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
    EXPECT_EQ(anira_handler_process(h, tensor, 0, tensor, 0, &delivered), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_rt_error(h), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_process_multi(h, tensor, 1, tensor, 1, nullptr),
              ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data(h, tensor, 0), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_push_data_multi(h, tensor, 1), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data(h, tensor, 0, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_pop_data_multi(h, tensor, 1, nullptr), ANIRA_ERROR_NOT_PREPARED);
    EXPECT_EQ(anira_handler_process_wait(h, tensor, 0, tensor, 0, ANIRA_WAIT_FOREVER, nullptr),
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
            EXPECT_EQ(anira_handler_process(h, &bad.tensor(), 0, &good.tensor(), 0, &delivered),
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
        EXPECT_EQ(anira_handler_process(h, &good.tensor(), 0, &bad.tensor(), 0, &delivered),
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
        EXPECT_EQ(anira_handler_process_wait(h, &good.tensor(), 0, &bad.tensor(), 0, 0.0, nullptr),
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
    EXPECT_EQ(anira_handler_process(h, &wrong_dtype.tensor(), 0, &wrong_rank.tensor(), 0, nullptr),
              ANIRA_ERROR_CONFIG);
    EXPECT_EQ(anira_handler_process(h, &wrong_rank.tensor(), 0, &wrong_dtype.tensor(), 0, nullptr),
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

    // A NULL tensor, a slot out of range: each side of the two-slot form has its own range, so
    // a good slot on one side never excuses a bad one on the other.
    EXPECT_EQ(anira_handler_process(h, nullptr, 0, tensors.data(), 0, &delivered),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(anira_handler_process(h, tensors.data(), 0, nullptr, 0, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process(h, tensors.data(), 2, tensors.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(anira_handler_process(h, tensors.data(), 2, tensors.data(), 0, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "in_slot out of range beside a good out_slot";
    EXPECT_EQ(anira_handler_process(h, tensors.data(), 0, tensors.data(), 2, nullptr),
              ANIRA_ERROR_INVALID_ARGUMENT)
        << "out_slot out of range beside a good in_slot";
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
              "anira_handler_push_data_multi: the tensor of Static input slot 1 'values_in' has "
              "dtype " +
                  std::to_string(ANIRA_DTYPE_I16) + ", the spec's is " +
                  std::to_string(ANIRA_DTYPE_F32) + "; nothing converts");
    EXPECT_EQ(anira_test::count_records(collector, "does not know", "rt"), 1U);
    EXPECT_NE(anira_test::find_record(collector, "does not know", "rt")
                  .m_message.find("anira_handler_pop_data: the tensor of output slot 0 'out'"),
              std::string::npos);
    EXPECT_EQ(anira_test::count_records(collector, "is malformed", "rt"), 1U);
    const RecordCollector::Record malformed =
        anira_test::find_record(collector, "is malformed", "rt");
    EXPECT_NE(malformed.m_message.find("anira_handler_push_data: the tensor of input slot 0 'in' "
                                       "is malformed (rank 3,"),
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
    const Block values_out(Layout::Packed, 1, 3);
    fill_input(stream_in, 0);
    fill_static(values_in);
    const std::array<anira_tensor, 2> inputs{stream_in.tensor(), values_in.tensor()};
    const std::array<anira_tensor, 2> outputs{stream_out.tensor(), values_out.tensor()};

    // The stem delivers the latency's zeros; the counts are the stem's, the status the refusal.
    size_t delivered = k_unset;
    EXPECT_EQ(anira_handler_process_wait(h,
                                         inputs.data(),
                                         0,
                                         outputs.data(),
                                         0,
                                         ANIRA_WAIT_FOREVER,
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
    EXPECT_EQ(counts[1], 3U) << "the sequence ran: the Static output is the stored value, whole";
}

// ---- ANIRA_MISS_CALLBACK -----------------------------------------------------------------------
// The backup function runs on the driver thread inside an ANIRA_NONBLOCKING entry, so it is
// declared ANIRA_NONBLOCKING, holds no gtest assertion (a failing EXPECT allocates, and under
// RealtimeSanitizer that aborts the process in place of failing the test), allocates nothing
// and writes only into storage its fixture owns. Every EXPECT runs after the call. The state
// is a member of the fixture, fresh for every test.

constexpr size_t k_samples = k_hop;  // k_hop as a sample count
constexpr float k_backup = 0.25F;    // what the backup function writes
constexpr size_t k_recorded = 4;     // the slots per side the state has room for

struct MissState {
    int m_calls = 0;
    anira_status m_return = ANIRA_OK;
    anira_handler* m_handler = nullptr;
    const anira_tensor* m_inputs = nullptr;
    const anira_tensor* m_outputs = nullptr;
    uint32_t m_num_inputs = 0;
    uint32_t m_num_outputs = 0;
    std::array<anira_tensor, k_recorded> m_input_copies{};
    std::array<anira_tensor, k_recorded> m_output_copies{};
    float m_first_input = 0.0F;  ///< sample 0 of the last channel of input 0, when it has samples
    float m_last_input = 0.0F;   ///< its last sample
    std::thread::id m_thread;
};

/// Element (channel, sample) of a float32 host tensor, by its description; NULL when the
/// tensor names no such memory. Callback-safe entries only.
float* element_of(const anira_tensor& tensor, size_t channel, size_t sample) ANIRA_NONBLOCKING {
    const bool planar = (tensor.flags & static_cast<uint32_t>(ANIRA_TENSOR_PLANAR)) != 0U;
    const bool packed = tensor.strides[0] == 0 && tensor.strides[1] == 0;
    const int64_t step = tensor.strides[1] != 0 ? tensor.strides[1] : 1;
    const auto index = static_cast<int64_t>(sample) * step;
    if (planar) {
        auto* plane = static_cast<float*>(
            anira_tensor_plane(&tensor, static_cast<uint32_t>(channel), ANIRA_DTYPE_F32));
        return plane != nullptr ? plane + index : nullptr;
    }
    float* base = anira_tensor_data_f32(&tensor);
    const int64_t channel_stride = packed ? tensor.shape[1] : tensor.strides[0];
    return base != nullptr ? base + (static_cast<int64_t>(channel) * channel_stride) + index
                           : nullptr;
}

anira_status ANIRA_CALL backup(anira_handler* handler,
                               const anira_tensor* inputs,
                               uint32_t num_inputs,
                               const anira_tensor* outputs,
                               uint32_t num_outputs,
                               void* user_data) ANIRA_NONBLOCKING {
    auto* state = static_cast<MissState*>(user_data);
    ++state->m_calls;
    state->m_handler = handler;
    state->m_inputs = inputs;
    state->m_outputs = outputs;
    state->m_num_inputs = num_inputs;
    state->m_num_outputs = num_outputs;
    state->m_thread = std::this_thread::get_id();
    for (uint32_t i = 0; i < num_inputs && i < k_recorded; ++i) {
        state->m_input_copies.at(i) = inputs[i];
    }
    for (uint32_t i = 0; i < num_outputs && i < k_recorded; ++i) {
        state->m_output_copies.at(i) = outputs[i];
    }
    if (num_inputs > 0 && inputs[0].shape[1] > 0) {
        const auto last_channel = static_cast<size_t>(inputs[0].shape[0] - 1);
        const auto last_sample = static_cast<size_t>(inputs[0].shape[1] - 1);
        const float* first = element_of(inputs[0], last_channel, 0);
        const float* last = element_of(inputs[0], last_channel, last_sample);
        if (first != nullptr && last != nullptr) {
            state->m_first_input = *first;
            state->m_last_input = *last;
        }
    }
    // Every requested output: k_backup + the slot, in every channel.
    for (uint32_t slot = 0; slot < num_outputs; ++slot) {
        const auto channels = static_cast<size_t>(outputs[slot].shape[0]);
        const auto samples = static_cast<size_t>(outputs[slot].shape[1]);
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t i = 0; i < samples; ++i) {
                float* out = element_of(outputs[slot], channel, i);
                if (out == nullptr) { return ANIRA_ERROR_INTERNAL; }
                *out = k_backup + static_cast<float>(slot);
            }
        }
    }
    return state->m_return;
}

class AbiHandlerTensorMiss : public testing::Test {
protected:
    /// Two delivered blocks, the second left at the gate, so that the next block starves.
    static void starve(Rig& rig, size_t channels) {
        ASSERT_LT(rig.latency(), 2U * k_samples) << "the third block would not starve";
        for (size_t k = 0; k < 2; ++k) {
            Block in(Layout::Planar, channels, k_hop);
            const Block out(Layout::Planar, channels, k_hop);
            fill_input(in, k * k_samples);
            ASSERT_EQ(anira_handler_process(rig.get(), &in.tensor(), 0, &out.tensor(), 0, nullptr),
                      ANIRA_OK);
            if (k == 0) { rig.settle(); }
        }
    }

    MissState m_state;
};

TEST_F(AbiHandlerTensorMiss, TheBackupFunctionFillsTheWholeBlockOncePerMiss) {
    Rig rig(pass_through(2), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    starve(rig, 2);
    EXPECT_EQ(m_state.m_calls, 0) << "a delivered block does not call it";
    for (int miss = 1; miss <= 2; ++miss) {
        Block in(Layout::Interleaved, 2, k_hop);
        const Block out(Layout::Interleaved, 2, k_hop);
        fill_input(in, 2 * k_samples);
        const auto in_before = bytes_of(in.tensor());
        const auto out_before = bytes_of(out.tensor());
        size_t delivered = k_unset;
        EXPECT_EQ(anira_handler_process(rig.get(), &in.tensor(), 0, &out.tensor(), 0, &delivered),
                  ANIRA_MISSED)
            << "the host filled the block, and it still counts as missed";
        EXPECT_EQ(delivered, 0U);
        EXPECT_EQ(m_state.m_calls, miss) << "once per missed block";
        EXPECT_EQ(m_state.m_handler, rig.get());
        EXPECT_EQ(m_state.m_thread, std::this_thread::get_id()) << "on the driving thread";
        EXPECT_EQ(m_state.m_num_inputs, 1U);
        EXPECT_EQ(m_state.m_num_outputs, 1U);
        EXPECT_EQ(m_state.m_inputs, &in.tensor()) << "the caller's own descriptor";
        EXPECT_EQ(m_state.m_outputs, &out.tensor());
        EXPECT_EQ(m_state.m_first_input, stream_value(1, 2 * k_samples)) << "the pushed block";
        EXPECT_EQ(m_state.m_last_input, stream_value(1, (3 * k_samples) - 1));
        EXPECT_EQ(bytes_of(in.tensor()), in_before);
        EXPECT_EQ(bytes_of(out.tensor()), out_before);
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(out.at(channel, i), k_backup); }
        }
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK) << "a miss is not recorded";
}

TEST_F(AbiHandlerTensorMiss, AnInPlaceBlockIsTheInputAndTheOutputOfTheFunction) {
    Rig rig(pass_through(2), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    starve(rig, 2);
    Block block(Layout::Interleaved, 2, k_hop);
    fill_input(block, 2 * k_samples);
    EXPECT_EQ(anira_handler_process(rig.get(), &block.tensor(), 0, &block.tensor(), 0, nullptr),
              ANIRA_MISSED);
    EXPECT_EQ(m_state.m_calls, 1);
    EXPECT_EQ(m_state.m_inputs, m_state.m_outputs);
    EXPECT_EQ(m_state.m_first_input, stream_value(1, 2 * k_samples))
        << "the input was still intact";
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(block.at(1, i), k_backup); }
}

TEST_F(AbiHandlerTensorMiss, AFailingStatusZeroFills) {
    m_state.m_return = ANIRA_ERROR_NOT_SUPPORTED;
    Rig rig(pass_through(2), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    starve(rig, 2);
    Block in(Layout::Planar, 2, k_hop);
    const Block out(Layout::Interleaved, 2, k_hop);
    fill_input(in, 2 * k_samples);
    size_t delivered = k_unset;
    EXPECT_EQ(anira_handler_process(rig.get(), &in.tensor(), 0, &out.tensor(), 0, &delivered),
              ANIRA_MISSED);
    EXPECT_EQ(delivered, 0U);
    EXPECT_EQ(m_state.m_calls, 1);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < k_hop; ++i) {
            ASSERT_EQ(out.at(channel, i), 0.0F) << "what the function wrote is overwritten";
        }
    }
    EXPECT_EQ(anira_handler_rt_error(rig.get()), ANIRA_OK);
}

TEST_F(AbiHandlerTensorMiss, APopPassesEmptyInputs) {
    Rig rig(pass_through(2), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    starve(rig, 2);
    const Block out(Layout::Contiguous, 2, k_hop);
    size_t delivered = k_unset;
    EXPECT_EQ(anira_handler_pop_data(rig.get(), &out.tensor(), 0, &delivered), ANIRA_MISSED);
    EXPECT_EQ(delivered, 0U);
    ASSERT_EQ(m_state.m_calls, 1);
    EXPECT_EQ(m_state.m_num_inputs, 1U);
    const anira_tensor& input = m_state.m_input_copies[0];
    EXPECT_EQ(input.ndim, 2U);
    EXPECT_EQ(input.shape[0], 2);
    EXPECT_EQ(input.shape[1], 0) << "an empty tensor: a pop has no input block";
    EXPECT_EQ(input.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(m_state.m_outputs, &out.tensor());
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(out.at(0, i), k_backup); }
}

TEST_F(AbiHandlerTensorMiss, AMultiFormHandsOverTheCallersArrays) {
    const Rig rig(multi_model(), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    ASSERT_EQ(rig.latency(0), 2U * k_samples);
    // Latency 16: two delivered blocks empty the ring while their inferences wait.
    for (size_t k = 0; k < 2; ++k) {
        Block in(Layout::Planar, 3, k_hop);
        const Block out(Layout::Planar, 3, k_hop);
        fill_input(in, k * k_samples);
        ASSERT_EQ(anira_handler_process(rig.get(), &in.tensor(), 0, &out.tensor(), 0, nullptr),
                  ANIRA_OK);
    }
    Block stream_in(Layout::Interleaved, 3, k_hop);
    Block values_in(Layout::Packed, 1, 3);
    const Block stream_out(Layout::Planar, 3, k_hop);
    const Block values_out(Layout::Packed, 1, 3);
    fill_input(stream_in, 2 * k_samples);
    fill_static(values_in);
    const std::array<anira_tensor, 2> inputs{stream_in.tensor(), values_in.tensor()};
    const std::array<anira_tensor, 2> outputs{stream_out.tensor(), values_out.tensor()};
    std::array<size_t, 2> delivered{k_unset, k_unset};
    EXPECT_EQ(anira_handler_process_multi(rig.get(),
                                          inputs.data(),
                                          2,
                                          outputs.data(),
                                          2,
                                          delivered.data()),
              ANIRA_MISSED);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_EQ(delivered[1], 3U) << "a carried Static output is whole on a miss too";
    ASSERT_EQ(m_state.m_calls, 1);
    EXPECT_EQ(m_state.m_inputs, inputs.data());
    EXPECT_EQ(m_state.m_outputs, outputs.data());
    EXPECT_EQ(m_state.m_num_inputs, 2U);
    EXPECT_EQ(m_state.m_num_outputs, 2U);
    EXPECT_EQ(m_state.m_first_input, stream_value(2, 2 * k_samples));
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(stream_out.at(2, i), k_backup); }
    for (size_t i = 0; i < 3; ++i) { ASSERT_EQ(values_out.at(0, i), k_backup + 1.0F); }

    // A single-tensor form on the same handler: the handler's array, the caller's descriptor
    // in its slot and an empty tensor beside it.
    const Block single_out(Layout::Interleaved, 3, k_hop);
    EXPECT_EQ(
        anira_handler_process(rig.get(), &stream_in.tensor(), 0, &single_out.tensor(), 0, nullptr),
        ANIRA_MISSED);
    ASSERT_EQ(m_state.m_calls, 2);
    EXPECT_EQ(m_state.m_inputs, rig.get()->m_input_tensors.data());
    EXPECT_EQ(m_state.m_outputs, rig.get()->m_output_tensors.data());
    EXPECT_EQ(bytes_of(m_state.m_input_copies[0]), bytes_of(stream_in.tensor()));
    EXPECT_EQ(bytes_of(m_state.m_output_copies[0]), bytes_of(single_out.tensor()));
    EXPECT_EQ(m_state.m_input_copies[1].shape[1], 0) << "the slot the call did not carry";
    EXPECT_EQ(m_state.m_input_copies[1].shape[0], 1);
    EXPECT_EQ(m_state.m_output_copies[1].shape[1], 0);
    for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(single_out.at(1, i), k_backup); }
}

/// The empty tensor anira_handler_prepare builds for a float32 slot: rank 2, {channels, 0},
/// host memory, no pointer.
anira_tensor empty_slot(int64_t channels) {
    const std::array<int64_t, 2> shape{channels, 0};
    anira_tensor tensor{};
    anira_tensor_init_host(&tensor, nullptr, ANIRA_DTYPE_F32, 2, shape.data());
    return tensor;
}

// The handler's arrays serve every single-tensor call. A slot a call staged is the empty tensor
// of prepare again when the call returns, so the function never finds the pointers of an
// earlier call, which may name dead memory by then, in a slot the missed call did not carry.
TEST_F(AbiHandlerTensorMiss, ASlotTheCallDidNotCarryNamesNoMemoryOfAnEarlierCall) {
    const Rig rig(multi_model(), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    ASSERT_EQ(rig.latency(0), 2U * k_samples);
    anira_handler* handler = rig.get();
    const auto empty_values = bytes_of(empty_slot(1));
    {
        // Slot 1 is Static: a single-tensor form refuses it (its single form is the Static
        // entry), so nothing of this descriptor is ever staged, whatever it carries: the planar
        // flag, a plane array, a byte offset, a stride. The memory and the plane array die with
        // this scope.
        std::array<float, 4> memory{k_untouched, k_static_in[0], k_static_in[1], k_static_in[2]};
        std::array<float*, 1> planes{memory.data()};
        const std::array<int64_t, 2> shape{1, 3};
        anira_tensor values{};
        anira_tensor_init_host_planar(&values,
                                      static_cast<const void*>(planes.data()),
                                      1,
                                      ANIRA_DTYPE_F32,
                                      2,
                                      shape.data());
        values.byte_offset = sizeof(float);
        values.strides[1] = 1;
        ASSERT_EQ(anira_handler_push_data(handler, &values, 1), ANIRA_ERROR_INVALID_ARGUMENT);
        ASSERT_EQ(anira_handler_pop_data(handler, &values, 1, nullptr),
                  ANIRA_ERROR_INVALID_ARGUMENT);
        EXPECT_EQ(m_state.m_calls, 0);
    }
    EXPECT_EQ(bytes_of(handler->m_input_tensors[1]), empty_values) << "never staged";
    EXPECT_EQ(bytes_of(handler->m_output_tensors[1]), empty_values);

    // Latency 16: two delivered blocks of slot 0 empty the ring, the third one starves.
    for (size_t k = 0; k < 3; ++k) {
        Block in(Layout::Planar, 3, k_hop);
        const Block out(Layout::Planar, 3, k_hop);
        fill_input(in, k * k_samples);
        ASSERT_EQ(anira_handler_process(handler, &in.tensor(), 0, &out.tensor(), 0, nullptr),
                  k < 2 ? ANIRA_OK : ANIRA_MISSED);
    }
    ASSERT_EQ(m_state.m_calls, 1);
    ASSERT_EQ(m_state.m_inputs, handler->m_input_tensors.data());
    ASSERT_EQ(m_state.m_outputs, handler->m_output_tensors.data());
    EXPECT_GT(m_state.m_input_copies[0].shape[1], 0) << "the slot the call carried";
    // What the function was handed in the other slot, copied while it ran.
    for (const anira_tensor* slot : {&m_state.m_input_copies[1], &m_state.m_output_copies[1]}) {
        EXPECT_EQ(slot->shape[0], 1);
        EXPECT_EQ(slot->shape[1], 0);
        EXPECT_EQ(slot->handle.raw[0], 0U) << "a pointer of the earlier call";
        EXPECT_EQ(slot->handle.raw[1], 0U);
        EXPECT_EQ(slot->handle.raw[2], 0U);
        EXPECT_EQ(slot->byte_offset, 0U);
        EXPECT_EQ(bytes_of(*slot), empty_values);
    }
    // The carried slot is un-staged the same way once the missed call has returned.
    EXPECT_EQ(bytes_of(handler->m_input_tensors[0]), bytes_of(empty_slot(3)));
    EXPECT_EQ(bytes_of(handler->m_output_tensors[0]), bytes_of(empty_slot(3)));
}

// A float host: channel pointers presented as planar float32 tensors through an
// anira::PlanarFloatAdapter (anira_test::FloatFace), the way the 2.x anira::InferenceHandler
// presents its own. The miss function is handed the caller's arrays as they are, and a slot
// the missed call did not carry names no memory: the adapter nulls the planes of a slot it
// leaves out, so the pointers of an earlier call never reach the function.
TEST_F(AbiHandlerTensorMiss, AFloatHostsPlanarTensorsReachTheMissFunctionAsTheyAre) {
    const Rig rig(multi_model(), ANIRA_MISS_CALLBACK, 2, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    ASSERT_EQ(rig.latency(0), 2U * k_samples);
    anira_test::FloatFace face(rig.get());
    std::array<std::vector<float>, 3> in_data;
    std::array<std::vector<float>, 3> out_data;
    std::array<const float*, 3> in_channels{};
    std::array<float*, 3> out_channels{};
    for (size_t channel = 0; channel < 3; ++channel) {
        in_data.at(channel).assign(k_hop, static_cast<float>(channel + 1));
        out_data.at(channel).assign(k_hop, k_untouched);
        in_channels.at(channel) = in_data.at(channel).data();
        out_channels.at(channel) = out_data.at(channel).data();
    }
    // The Static slots are left out: a request of 0 with a NULL pointer.
    const std::array<const float* const*, 2> in{in_channels.data(), nullptr};
    const std::array<float* const*, 2> out{out_channels.data(), nullptr};
    const std::array<size_t, 2> num_in{k_hop, 0};
    const std::array<size_t, 2> num_out{k_hop, 0};
    {
        // An earlier call carries the Static slots (whole tensors, which is how the face
        // presents a carried Static slot) over memory that dies with this scope.
        std::array<float, 3> values{k_static_in};
        const std::array<const float*, 1> value_in{values.data()};
        const std::array<float*, 1> value_out{values.data()};
        const std::array<const float* const*, 2> static_in{nullptr, value_in.data()};
        const std::array<float* const*, 2> static_out{nullptr, value_out.data()};
        const std::array<size_t, 2> static_count{0, 3};
        ASSERT_EQ(face.process_multi(static_in.data(),
                                     static_count.data(),
                                     static_out.data(),
                                     static_count.data(),
                                     nullptr),
                  ANIRA_OK);
        EXPECT_EQ(m_state.m_calls, 0);
    }
    anira_status status = ANIRA_OK;
    std::array<size_t, 2> delivered{k_unset, k_unset};
    for (int k = 0; k < 3; ++k) {
        status = face.process_multi(in.data(),
                                    num_in.data(),
                                    out.data(),
                                    num_out.data(),
                                    delivered.data());
    }
    EXPECT_EQ(status, ANIRA_MISSED);
    EXPECT_EQ(delivered[0], 0U) << "a missed block delivers 0";
    EXPECT_EQ(delivered[1], 0U);
    ASSERT_EQ(m_state.m_calls, 1);
    EXPECT_EQ(m_state.m_num_inputs, 2U);
    EXPECT_EQ(m_state.m_num_outputs, 2U);
    // The arrays the entry was handed, as they are: the caller's own, here the face's.
    EXPECT_EQ(m_state.m_inputs, face.multi_inputs());
    EXPECT_EQ(m_state.m_outputs, face.multi_outputs());
    const anira_tensor& input = m_state.m_input_copies[0];
    EXPECT_EQ(
        input.flags,
        static_cast<uint32_t>(ANIRA_TENSOR_PLANAR) | static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY));
    EXPECT_EQ(input.dtype, ANIRA_DTYPE_F32);
    EXPECT_EQ(input.shape[0], 3);
    EXPECT_EQ(input.shape[1], static_cast<int64_t>(k_hop));
    EXPECT_EQ(anira_tensor_plane(&input, 2, ANIRA_DTYPE_F32), in_data[2].data());
    EXPECT_EQ(m_state.m_first_input, 3.0F);
    EXPECT_EQ(m_state.m_input_copies[1].shape[1], 0) << "the slot the call did not carry";
    const anira_tensor& output = m_state.m_output_copies[0];
    EXPECT_EQ(output.flags, static_cast<uint32_t>(ANIRA_TENSOR_PLANAR));
    EXPECT_EQ(anira_tensor_plane(&output, 1, ANIRA_DTYPE_F32), out_data[1].data());
    EXPECT_EQ(m_state.m_output_copies[1].shape[1], 0);
    // The slots the missed call did not carry name no memory: not the dead array of the
    // earlier call that carried them.
    EXPECT_EQ(anira_tensor_plane(&m_state.m_input_copies[1], 0, ANIRA_DTYPE_F32), nullptr);
    EXPECT_EQ(anira_tensor_plane(&m_state.m_output_copies[1], 0, ANIRA_DTYPE_F32), nullptr);
    for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(out_data.at(channel).at(i), k_backup); }
    }
}

TEST_F(AbiHandlerTensorMiss, AWaitTwinWithoutAThreadCallsItAndStillRefuses) {
    const Rig rig(pass_through(2), ANIRA_MISS_CALLBACK, /*threads=*/0, &backup, &m_state);
    ASSERT_TRUE(rig.ready());
    ASSERT_LT(rig.latency(), 2U * k_samples);
    size_t delivered = k_unset;
    for (int k = 0; k < 2; ++k) {
        Block in(Layout::Interleaved, 2, k_hop);
        const Block out(Layout::Planar, 2, k_hop);
        fill_input(in, static_cast<size_t>(k) * k_samples);
        delivered = k_unset;
        EXPECT_EQ(anira_handler_process_wait(rig.get(),
                                             &in.tensor(),
                                             0,
                                             &out.tensor(),
                                             0,
                                             ANIRA_WAIT_FOREVER,
                                             &delivered),
                  ANIRA_ERROR_INVALID_STATE);
        if (k == 0) {
            EXPECT_EQ(delivered, k_hop);
            EXPECT_EQ(m_state.m_calls, 0);
        } else {
            EXPECT_EQ(delivered, 0U);
            EXPECT_EQ(m_state.m_calls, 1) << "the stem missed and the function ran";
            for (size_t i = 0; i < k_hop; ++i) { ASSERT_EQ(out.at(0, i), k_backup); }
        }
    }
}

TEST(AbiHandlerTensor, TheCallbackPolicyNeedsItsFunctionAtPrepare) {
    const anira_test::Context context;
    const ModelConfig model = pass_through(2);
    const std::vector<anira_backend_id> candidates{{.struct_size = sizeof(anira_backend_id),
                                                    .engine = ANIRA_ENGINE_NONE,
                                                    .provider = ANIRA_PROVIDER_DEFAULT,
                                                    .engine_id = nullptr}};
    anira_test::Handler handler(context, model, candidates);
    // A parsed contract names the policy; a file cannot carry the function.
    anira::ContractHandle contract = anira::ContractHandle::from_json(
        R"({"hard": {"budget": {"ms": 1.0}, "warmup": {"fixed": 0}, "on_miss": "callback"}})");
    contract.hard_geometry(1, k_hop, k_rate);
    anira_error err = ANIRA_ERROR_INIT;
    EXPECT_EQ(handler.prepare(contract, &err), ANIRA_ERROR_CONFIG);
    EXPECT_NE(std::string(err.message).find("on_miss"), std::string::npos) << err.message;
    EXPECT_NE(std::string(err.message).find("anira_contract_hard_set_miss_fn"), std::string::npos);
    EXPECT_EQ(anira_handler_plan_report(handler.m_handler), nullptr) << "left unprepared";

    MissState state;
    contract.hard_miss_fn(&backup, &state);
    EXPECT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
    EXPECT_EQ(handler.m_handler->m_miss_fn, &backup);
    EXPECT_EQ(handler.m_handler->m_miss_user_data, &state);
    // The function without the policy is legal and unused.
    contract.hard_on_miss(ANIRA_MISS_ZEROS);
    EXPECT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
}

// ---- the named comparisons beside the absolute expectations ------------------------------------

// A white-box look at the send ring: after anira_handler_push_data under any description,
// channel c of the ring holds channel c's ramp. 5 samples stay below the hop, so nothing
// consumes them.
TEST(AbiHandlerTensor, APushLandsEveryChannelInItsRing) {
    constexpr size_t k_count = 5;
    for (const size_t channels : {2U, 3U, 6U}) {
        for (const Layout layout : k_layouts) {
            SCOPED_TRACE(std::to_string(channels) + " channels, " + layout_name(layout));
            const Rig rig(pass_through(static_cast<int64_t>(channels)));
            ASSERT_TRUE(rig.ready());
            Block block(layout, channels, k_count);
            fill_input(block, 0);
            ASSERT_EQ(anira_handler_push_data(rig.get(), &block.tensor(), 0), ANIRA_OK);
            for (size_t channel = 0; channel < channels; ++channel) {
                ASSERT_EQ(rig.session().m_send_buffer[0].get_available_samples(channel), k_count);
                std::array<float, k_count> ring{};
                rig.session().m_send_buffer[0].pop_block(channel, ring.data(), k_count);
                for (size_t i = 0; i < k_count; ++i) {
                    EXPECT_EQ(ring.at(i), stream_value(channel, i))
                        << "channel " << channel << ", sample " << i;
                }
            }
        }
    }
}

// Two descriptions of one stream, stated as a consistency check: the same script over planar
// blocks (one pointer per channel, what a float host holds) and over interleaved blocks gives
// the same statuses, the same counts and the same samples, missed blocks included. One
// description after the other: a closed gate holds its inference thread, so two rigs at once
// would starve each other's settle.
struct FaceBlock {
    anira_status m_status = ANIRA_OK;
    size_t m_delivered = k_unset;
    std::vector<float> m_samples;  ///< channel after channel
};

/// Blocks 0 and 1 are delivered, 2 and 3 starve (block 1 stays at the gate), 4 and 5 are
/// delivered again after the catch-up.
std::vector<FaceBlock> run_face(Layout layout, anira_miss_policy policy) {
    std::vector<FaceBlock> blocks;
    Rig rig(pass_through(2), policy);
    if (!rig.ready() || rig.latency() >= 2 * k_samples) {
        ADD_FAILURE() << "the script needs a latency below two blocks";
        return blocks;
    }
    for (size_t k = 0; k < 6; ++k) {
        Block in(layout, 2, k_hop);
        Block out(layout, 2, k_hop);
        fill_input(in, k * k_hop);
        FaceBlock block;
        block.m_status =
            anira_handler_process(rig.get(), &in.tensor(), 0, &out.tensor(), 0, &block.m_delivered);
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t i = 0; i < k_hop; ++i) { block.m_samples.push_back(out.at(channel, i)); }
        }
        blocks.push_back(std::move(block));
        if (k == 0 || k >= 3) { rig.settle(); }
    }
    return blocks;
}

void expect_faces_agree(anira_miss_policy policy) {
    SCOPED_TRACE("policy " + std::to_string(policy));
    const std::vector<FaceBlock> planar = run_face(Layout::Planar, policy);
    const std::vector<FaceBlock> interleaved = run_face(Layout::Interleaved, policy);
    ASSERT_EQ(planar.size(), 6U);
    ASSERT_EQ(interleaved.size(), 6U);
    for (size_t k = 0; k < 6; ++k) {
        EXPECT_EQ(interleaved[k].m_status, planar[k].m_status) << "block " << k;
        EXPECT_EQ(interleaved[k].m_status, k == 2 || k == 3 ? ANIRA_MISSED : ANIRA_OK)
            << "block " << k;
        EXPECT_EQ(interleaved[k].m_delivered, planar[k].m_delivered) << "block " << k;
        EXPECT_EQ(interleaved[k].m_samples, planar[k].m_samples) << "block " << k;
    }
}

TEST(AbiHandlerTensor, APlanarAndAnInterleavedBlockAgreeBitForBitAcrossMisses) {
    for (const anira_miss_policy policy :
         {ANIRA_MISS_ZEROS, ANIRA_MISS_HOLD_LAST, ANIRA_MISS_BYPASS}) {
        expect_faces_agree(policy);
    }
}

// The tensor entries against the still-public 2.x anira::InferenceHandler in one binary
// (test_Handler's oracle), on the bundled stereo gain model: every plan this build has an
// engine for, and the engine-free custom row. The C side takes one interleaved block and the
// Static gain as [1, 1] through anira_handler_process_multi; the 2.x side takes channel
// pointers. Bit-equal per channel after de-interleaving. Through a real engine this covers two
// channels only; 3 and 6 channels run through the engine-free pass-through above.
TEST(AbiHandlerTensor, AnInterleavedBlockMatchesTheTwoPointXHandlerOnEveryPlan) {
    using anira_test::k_block;
    const anira_test::Context context;
    const std::vector<anira_backend_id> candidates = anira_test::custom_candidates();
    uint32_t num_plans = 1;
    for (uint32_t plan = 0; plan < num_plans; ++plan) {
        SCOPED_TRACE("plan " + std::to_string(plan));
        anira_test::Handler c(context, anira_test::stereo_gain_with_custom(), candidates);
        anira::InferenceConfig config_2x =
            anira_test::bridged_2x(k_stereo_gain_model_json, k_stereo_gain_contract_json, true);
        anira::PrePostProcessor pp(config_2x);
        anira::CoreConfig core_config(2, anira::WaitStrategy::SpinBackoff, anira::LogLevel::Error);
        core_config.m_log.m_drain = anira::LogDrain::Manual;
        anira::InferenceHandler v2(pp, config_2x, core_config);
        ASSERT_EQ(c.prepare(anira_test::file_contract(k_stereo_gain_contract_json, k_block)),
                  ANIRA_OK)
            << c.m_err.message;
        anira_handler* h = c.m_handler;
        v2.prepare(
            anira::HostConfig(static_cast<float>(k_block), static_cast<float>(anira_test::k_rate)));
        anira_test::expect_inference_config_eq(h->m_inference_config, config_2x);
        num_plans = anira_plan_report_num_plans(anira_handler_plan_report(h));
        ASSERT_EQ(anira_handler_set_plan(h, plan), ANIRA_OK);
        v2.set_inference_backend(h->m_plans.at(plan).m_backend);
        ASSERT_EQ(anira_handler_get_latency(h, 0), v2.get_latency(0));

        for (size_t k = 1; k <= 8; ++k) {
            const std::vector<float> left = anira_test::ramp(k);
            std::vector<float> right = anira_test::ramp(k);
            for (float& sample : right) { sample = -sample; }  // L != R
            Block c_in(Layout::Interleaved, 2, k_block);
            const Block c_out(Layout::Interleaved, 2, k_block);
            for (size_t i = 0; i < k_block; ++i) {
                c_in.at(0, i) = left.at(i);
                c_in.at(1, i) = right.at(i);
            }
            // The Static gain travels whole, in the spec's shape: [1].
            const float gain_in = 1.0F;
            float gain_out = k_untouched;
            const std::array<anira_tensor, 2> inputs{c_in.tensor(),
                                                     anira_test::whole_f32(&gain_in, {1})};
            const std::array<anira_tensor, 2> outputs{c_out.tensor(),
                                                      anira_test::whole_f32(&gain_out, {1})};
            std::array<size_t, 2> delivered{k_unset, k_unset};
            const size_t prev_c = anira_test::available(h);
            ASSERT_EQ(anira_handler_process_multi(h,
                                                  inputs.data(),
                                                  2,
                                                  outputs.data(),
                                                  2,
                                                  delivered.data()),
                      ANIRA_OK)
                << "block " << k;
            EXPECT_EQ(delivered[0], k_block);
            EXPECT_EQ(delivered[1], 1U);
            anira_test::wait_for_block(h, prev_c);

            std::vector<float> v_left(k_block, k_untouched);
            std::vector<float> v_right(k_block, k_untouched);
            const float v_gain_in = 1.0F;
            float v_gain_out = k_untouched;
            const std::array<const float*, 2> v_in_ch{left.data(), right.data()};
            const std::array<const float*, 1> v_gain_ch{&v_gain_in};
            const std::array<const float* const*, 2> v_in{v_in_ch.data(), v_gain_ch.data()};
            std::array<size_t, 2> v_num_in{k_block, 1};
            const std::array<float*, 2> v_out_ch{v_left.data(), v_right.data()};
            const std::array<float*, 1> v_gout_ch{&v_gain_out};
            const std::array<float* const*, 2> v_outs{v_out_ch.data(), v_gout_ch.data()};
            std::array<size_t, 2> v_num_out{k_block, 1};
            const size_t prev_v = v2.get_available_samples(0);
            const size_t n_v =
                v2.process(v_in.data(), v_num_in.data(), v_outs.data(), v_num_out.data())[0];
            anira_test::wait_for_block(v2, prev_v);
            ASSERT_EQ(n_v, k_block) << "block " << k;
            // The Static output is written on every delivered block, and that is all this case
            // says about it. It is the latest completed value (PrePostProcessor stores it after
            // the ring push of its inference), so whether a call already sees the result of the
            // block it pushed is a race that each side runs on its own: compared with each other
            // the two sides differed once in about 200 local runs, the C side holding the
            // result of block 4 and the 2.x side that of block 3, with every sample of the
            // stream bit-equal. What the value is differs per engine, too. The values of a
            // Static output are compared exactly where the gate makes the order certain:
            // ProcessMultiCarriesStaticSlotsAsWholeTensors, and test_HandlerStatic.cpp.
            EXPECT_NE(gain_out, k_untouched) << "block " << k;
            EXPECT_NE(v_gain_out, k_untouched) << "block " << k;
            for (size_t i = 0; i < k_block; ++i) {
                ASSERT_EQ(c_out.at(0, i), v_left.at(i)) << "block " << k << ", left " << i;
                ASSERT_EQ(c_out.at(1, i), v_right.at(i)) << "block " << k << ", right " << i;
            }
        }
    }
}

}  // namespace
