// The tensor stems of InferenceManager, driven directly with what the float face
// (anira::PlanarFloatAdapter, src/scheduler/PlanarFloatAdapter.h) cannot express: one block read
// by strides (interleaved for 1, 2, 3 and 6 channels, contiguous,
// packed), planar memory with a byte offset and with a stride inside each plane, the two
// descriptions crossed between input and output, the same tensor on both sides, empty and
// zeroed tensors, a ring that is not float32, and a tensor whose dtype is not its slot's.
//
// Two kinds of proof. (1) The scenarios of test_CopyPathOracle.cpp are replayed through the
// tensor stems under every description and compared with the transcripts that the float core
// recorded before it was rewritten (copy_path_oracle_golden.h): an interleaved or an offset
// planar block has to give, float for float, what the float*** call of the time gave over
// separate channel pointers, under the three miss policies too. This file never records: it has its
// own comparison, which ignores ANIRA_COPY_ORACLE_RECORD. (2) Absolute expectations where the
// recording has no word: channel counts other than 2 and 3, the send ring's content after an
// interleaved push, an int16 ring, the refusals.
//
// test/support/copy_oracle.h says what a transcript is and why a scenario is a pure function
// of its calls (the clocked gate).

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>
#include <anira/utils/RingBuffer.h>
#include <anira/utils/RtLatch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "../support/copy_oracle.h"
#include "copy_path_oracle_golden.h"
#include "gtest/gtest.h"
#include "scheduler/PlanarFloatAdapter.h"
#include "scheduler/TensorRun.h"

using namespace anira;

namespace {

namespace oracle = anira_test::oracle;

constexpr size_t k_hop = 8;  // the window of every streamed tensor, and the host block

// ---- the models (those of test_CopyPathOracle.cpp, for any channel count) ----------------------

InferenceConfig make_config(TensorShape shape, ProcessingSpec spec, float blocking_ratio = 0.F) {
    return InferenceConfig(
        std::vector<ModelData>{ModelData("placeholder", InferenceBackend::CUSTOM)},
        std::vector<TensorShape>{std::move(shape)},
        std::move(spec),
        1.F,    // max_inference_time, in ms
        0,      // warm_up
        false,  // session_exclusive_processor
        blocking_ratio,
        2);  // num_parallel_processors
}

/// `channels` channels in, as many out: the gate passes the input through.
InferenceConfig pass_through_config(size_t channels, float blocking_ratio = 0.F) {
    const auto count = static_cast<int64_t>(channels);
    return make_config(TensorShape({{1, count, k_hop}}, {{1, count, k_hop}}),
                       ProcessingSpec({channels}, {channels}),
                       blocking_ratio);
}

InferenceConfig multi_config() {
    return make_config(TensorShape({{1, 3, k_hop}, {1, 3}}, {{1, 3, k_hop}, {1, 3}}),
                       ProcessingSpec({3, 1}, {3, 1}, {k_hop, 0}, {k_hop, 0}));
}

InferenceConfig mono_to_stereo_config() {
    return make_config(TensorShape({{1, 1, k_hop}}, {{1, 2, k_hop}}), ProcessingSpec({1}, {2}));
}

InferenceConfig generator_config() {
    return make_config(TensorShape({{1, 4}}, {{1, 1, k_hop}}),
                       ProcessingSpec({1}, {1}, {0}, {k_hop}));
}

HostConfig host_config() {
    return {static_cast<float>(k_hop), 800.F, true};
}

// ---- host memory under a description -----------------------------------------------------------

/// How a block lies in memory and how its tensor says so.
enum class Layout {
    Planar,         ///< one run per channel, ANIRA_TENSOR_PLANAR, all-zero strides
    PlanarOffset,   ///< planar, the samples k_lead elements into each plane (byte_offset)
    PlanarStrided,  ///< planar, every other element of each plane (strides[1] == 2)
    Contiguous,     ///< one block, channel after channel, strides {run length, 1}
    Interleaved,    ///< one block, frame after frame, strides {1, channels}
    Packed          ///< one block of exactly channels x count elements, all-zero strides
};

const char* name_of(Layout layout) {
    switch (layout) {
        case Layout::Planar: return "planar";
        case Layout::PlanarOffset: return "planar+offset";
        case Layout::PlanarStrided: return "planar, stride 2";
        case Layout::Contiguous: return "contiguous";
        case Layout::Interleaved: return "interleaved";
        case Layout::Packed: return "packed";
    }
    return "?";
}

constexpr size_t k_lead = 3;          // Layout::PlanarOffset: elements before the first sample
constexpr size_t k_plane_stride = 2;  // Layout::PlanarStrided: elements between two samples

/// The host memory of one slot: `length` samples per channel and oracle::k_guard more that a
/// call must leave alone (Layout::Packed has no room between its channels: its guard follows
/// the block). Every element is `fill` until something writes it. at() addresses a sample by
/// channel and index whatever the layout; slack_untouched() says that no element that is not
/// a sample (the lead of a plane, the elements between strided samples, a packed block's
/// tail) was written.
template <typename T>
class HostBlock {
public:
    HostBlock(Layout layout, size_t channel_count, size_t length, T fill)
        : m_layout(layout)
        , m_channels(channel_count)
        , m_run_length(layout == Layout::Packed ? length : length + oracle::k_guard)
        , m_fill(fill) {
        const bool planar = layout == Layout::Planar || layout == Layout::PlanarOffset ||
                            layout == Layout::PlanarStrided;
        size_t elements = channel_count * m_run_length;
        if (layout == Layout::Planar) { elements = m_run_length; }
        if (layout == Layout::PlanarOffset) { elements = k_lead + m_run_length; }
        if (layout == Layout::PlanarStrided) { elements = k_plane_stride * m_run_length; }
        if (layout == Layout::Packed) { elements += oracle::k_guard; }
        m_storage.assign(planar ? channel_count : 1, std::vector<T>(elements, fill));
        m_is_sample.assign(m_storage.size(), std::vector<bool>(elements, false));
        for (std::vector<T>& plane : m_storage) { m_planes.push_back(plane.data()); }
        for (size_t channel = 0; channel < channel_count; ++channel) {
            for (size_t index = 0; index < m_run_length; ++index) {
                const auto [plane, element] = locate(channel, index);
                m_is_sample[plane][element] = true;
            }
        }
    }
    HostBlock(const HostBlock&) = delete;
    HostBlock& operator=(const HostBlock&) = delete;
    HostBlock(HostBlock&&) = default;
    HostBlock& operator=(HostBlock&&) = default;

    size_t channels() const { return m_channels; }
    size_t run_length() const { return m_run_length; }

    T& at(size_t channel, size_t index) {
        const auto [plane, element] = locate(channel, index);
        return m_storage[plane][element];
    }

    /// The samples of one channel in order, the guard included.
    std::vector<T> run(size_t channel) const {
        std::vector<T> samples;
        for (size_t index = 0; index < m_run_length; ++index) {
            const auto [plane, element] = locate(channel, index);
            samples.push_back(m_storage[plane][element]);
        }
        return samples;
    }

    bool slack_untouched() const {
        for (size_t plane = 0; plane < m_storage.size(); ++plane) {
            for (size_t element = 0; element < m_storage[plane].size(); ++element) {
                if (!m_is_sample[plane][element] && m_storage[plane][element] != m_fill) {
                    return false;
                }
            }
        }
        return true;
    }

    /// One pointer per channel (Layout::Planar only): what a float*** call takes.
    T** planes() { return m_planes.data(); }

    /// The tensor of the first `count` samples per channel, built the way a host builds it:
    /// a factory, then the strides and the offset assigned on the record.
    anira_tensor tensor(size_t count, anira_dtype dtype, bool read_only) {
        const std::array<int64_t, 2> shape{static_cast<int64_t>(m_channels),
                                           static_cast<int64_t>(count)};
        anira_tensor record{};
        switch (m_layout) {
            case Layout::Planar:
            case Layout::PlanarOffset:
            case Layout::PlanarStrided:
                anira_tensor_init_host_planar(&record,
                                              static_cast<const void*>(m_planes.data()),
                                              static_cast<uint32_t>(m_channels),
                                              dtype,
                                              2,
                                              shape.data());
                break;
            case Layout::Contiguous:
            case Layout::Interleaved:
            case Layout::Packed:
                anira_tensor_init_host(&record, m_storage[0].data(), dtype, 2, shape.data());
                break;
        }
        if (m_layout == Layout::PlanarOffset) { record.byte_offset = k_lead * sizeof(T); }
        if (m_layout == Layout::PlanarStrided) {
            record.strides[1] = static_cast<int64_t>(k_plane_stride);
        }
        if (m_layout == Layout::Contiguous) {
            record.strides[0] = static_cast<int64_t>(m_run_length);
            record.strides[1] = 1;
        }
        if (m_layout == Layout::Interleaved) {
            record.strides[0] = 1;
            record.strides[1] = static_cast<int64_t>(m_channels);
        }
        if (m_layout == Layout::Packed) {
            // All-zero strides read the block packed: the channel stride is shape[1].
            EXPECT_EQ(count, m_run_length) << "a packed block describes all of its samples";
        }
        if (read_only) { record.flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY); }
        return record;
    }

private:
    std::pair<size_t, size_t> locate(size_t channel, size_t index) const {
        switch (m_layout) {
            case Layout::Planar: return {channel, index};
            case Layout::PlanarOffset: return {channel, k_lead + index};
            case Layout::PlanarStrided: return {channel, k_plane_stride * index};
            case Layout::Contiguous:
            case Layout::Packed: return {0, (channel * m_run_length) + index};
            case Layout::Interleaved: return {0, (index * m_channels) + channel};
        }
        return {0, 0};
    }

    Layout m_layout;
    size_t m_channels;
    size_t m_run_length;
    T m_fill;
    std::vector<std::vector<T>> m_storage;
    std::vector<std::vector<bool>> m_is_sample;
    std::vector<T*> m_planes;
};

/// An empty tensor in the spelling a call uses for a slot it does not carry: rank 2, the
/// slot's channel count, no samples, NULL memory.
anira_tensor empty_tensor(size_t channels, anira_dtype dtype = ANIRA_DTYPE_F32) {
    const std::array<int64_t, 2> shape{static_cast<int64_t>(channels), 0};
    anira_tensor tensor{};
    anira_tensor_init_host(&tensor, nullptr, dtype, 2, shape.data());
    return tensor;
}

/// Every field of a host tensor, one by one: the record has padding, so its bytes do not
/// compare.
bool same_record(const anira_tensor& a, const anira_tensor& b) {
    return a.domain == b.domain && a.dtype == b.dtype && a.ndim == b.ndim && a.flags == b.flags &&
           std::equal(std::begin(a.shape), std::end(a.shape), std::begin(b.shape)) &&
           std::equal(std::begin(a.strides), std::end(a.strides), std::begin(b.strides)) &&
           a.byte_offset == b.byte_offset &&
           std::equal(std::begin(a.handle.raw), std::end(a.handle.raw), std::begin(b.handle.raw)) &&
           a.manager_ctx == b.manager_ctx && a.release == b.release &&
           a.acquire.kind == b.acquire.kind && a.acquire.flags == b.acquire.flags;
}

// ---- the rig -----------------------------------------------------------------------------------

enum class Stem { Process, ProcessNowait, ProcessWait, PopData, PopDataUntil, PopDataWait };

const char* name_of(Stem stem) {
    switch (stem) {
        case Stem::Process: return "process";
        case Stem::ProcessNowait: return "process_nowait";
        case Stem::ProcessWait: return "process_wait";
        case Stem::PopData: return "pop_data";
        case Stem::PopDataUntil: return "pop_data_until";
        case Stem::PopDataWait: return "pop_data_wait";
    }
    return "?";
}

const char* name_of(Core::WaitOutcome outcome) {
    switch (outcome) {
        case Core::WaitOutcome::Done: return "Done";
        case Core::WaitOutcome::Deadline: return "Deadline";
        case Core::WaitOutcome::NoThread: return "NoThread";
    }
    return "?";
}

struct CallOptions {
    bool m_in_place = false;     ///< output slot 0 is the memory of input slot 0
    bool m_null_unused = false;  ///< a slot whose count is 0 is handed over without memory
    bool m_settle = true;        ///< false leaves the call's inferences at the gate
};

constexpr CallOptions k_plain{.m_in_place = false, .m_null_unused = false, .m_settle = true};
constexpr CallOptions k_in_place{.m_in_place = true, .m_null_unused = false, .m_settle = true};
constexpr CallOptions k_null_unused{.m_in_place = false, .m_null_unused = true, .m_settle = true};
constexpr CallOptions k_gate_closed{.m_in_place = false, .m_null_unused = false, .m_settle = false};
constexpr CallOptions k_in_place_gate_closed{.m_in_place = true,
                                             .m_null_unused = false,
                                             .m_settle = false};

enum class Mode { Clocked, Waiting };
enum class Model { PassThrough, FanOut, ParamRamp };

/// How a rig hands its blocks to the tensor stems.
enum class Face {
    FloatAdapter,  ///< separate channel pointers, presented by a PlanarFloatAdapter the way
                   ///< anira::InferenceHandler presents its own
    Tensor         ///< hand-built tensors over blocks of the rig's two layouts
};

struct Layouts {
    Layout m_in = Layout::Planar;   ///< every input block; the shared block of an in-place call
    Layout m_out = Layout::Planar;  ///< every output block
};

std::unique_ptr<anira_test::GateBackend> make_gate(Model model, InferenceConfig& config) {
    switch (model) {
        case Model::FanOut: return std::make_unique<oracle::FanOutGate>(config);
        case Model::ParamRamp: return std::make_unique<oracle::ParamRampGate>(config);
        case Model::PassThrough: break;
    }
    return std::make_unique<anira_test::GateBackend>(config);
}

/// The ManagerRig of test_CopyPathOracle.cpp over either face: the same calls, the same
/// transcript, stem for stem (the 2.x dispatcher and the deadline form of pop_data have their
/// tensor forms). A tensor stem returns the manager's own array of delivered counts, so on the
/// tensor face "returned=out" says that an array came back, and "out=" prints it; on the float
/// face both are read off the adapter's deliver_counts().
class Rig {
public:
    Rig(InferenceConfig config,
        anira_miss_policy policy,
        Face face,
        Layouts layouts = {},
        Mode mode = Mode::Clocked,
        Model model = Model::PassThrough,
        const RingDtypes& ring_dtypes = {})
        : m_config(std::move(config))
        , m_pp_processor(m_config)
        , m_gate(make_gate(model, m_config))
        , m_manager(m_pp_processor, m_config, m_gate.get(), CoreConfig(2))
        , m_face(face)
        , m_layouts(layouts)
        , m_mode(mode)
        , m_positions(m_config.get_tensor_input_shape().size(), 0) {
        m_manager.set_miss_policy(policy);
        m_manager.prepare(host_config(), CustomLatencies{}, ring_dtypes);
        m_adapter.prepare(m_config);
        for (const std::shared_ptr<SessionElement>& candidate : Core::get_sessions()) {
            if (candidate->m_session_id == m_manager.get_session_id()) { m_session = candidate; }
        }
        if (m_session == nullptr) {
            ADD_FAILURE() << "the manager's session is not registered";
            return;
        }
        if (m_mode == Mode::Waiting) {
            m_manager.set_non_realtime(true);
        } else {
            m_gate->m_open.store(false);
        }
        std::string line = "latency=[";
        for (size_t i = 0; i < m_manager.get_latency().size(); ++i) {
            line += (i > 0 ? "," : "") + std::to_string(m_manager.get_latency()[i]);
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
    ~Rig() { m_gate->m_open.store(true); }  // before the manager releases the session
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    InferenceManager& manager() { return m_manager; }
    PlanarFloatAdapter& adapter() { return m_adapter; }
    SessionElement& session() { return *m_session; }
    bool ready() const { return m_session != nullptr; }
    const oracle::Transcript& transcript() const { return m_transcript; }

    /// A process or a pop stem; a pop takes no input counts.
    void call(Stem stem,
              const std::vector<size_t>& in_counts,
              const std::vector<size_t>& out_counts,
              CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const bool pops =
            stem == Stem::PopData || stem == Stem::PopDataUntil || stem == Stem::PopDataWait;
        const size_t call_number = ++m_calls;
        const size_t shared_length = options.m_in_place ? out_counts[0] : 0;
        std::vector<HostBlock<float>> inputs = make_inputs(in_counts, shared_length, call_number);
        std::vector<std::vector<std::vector<float>>> inputs_before;
        for (const HostBlock<float>& block : inputs) {
            inputs_before.emplace_back();
            for (size_t channel = 0; channel < block.channels(); ++channel) {
                inputs_before.back().push_back(block.run(channel));
            }
        }
        std::vector<HostBlock<float>> outputs;
        outputs.reserve(out_counts.size());
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            outputs.emplace_back(out_layout(),
                                 m_config.get_postprocess_output_channels()[slot],
                                 out_counts[slot],
                                 oracle::k_untouched);
        }
        std::vector<bool> out_unused;
        out_unused.reserve(out_counts.size());
        for (const size_t count : out_counts) {
            out_unused.push_back(options.m_null_unused && count == 0);
        }

        std::vector<size_t> num_in = in_counts;
        std::vector<size_t> num_out = out_counts;
        Core::WaitOutcome outcome = Core::WaitOutcome::Done;
        bool returned = false;
        if (m_face == Face::FloatAdapter) {
            std::vector<const float* const*> in_planes;
            for (size_t slot = 0; slot < in_counts.size(); ++slot) {
                const bool unused = options.m_null_unused && in_counts[slot] == 0;
                in_planes.push_back(unused ? nullptr : inputs[slot].planes());
            }
            std::vector<float* const*> out_planes;
            for (size_t slot = 0; slot < out_counts.size(); ++slot) {
                const bool shared = options.m_in_place && slot == 0;
                float* const* planes = shared ? inputs[0].planes() : outputs[slot].planes();
                out_planes.push_back(out_unused[slot] ? nullptr : planes);
            }
            const size_t* counts =
                call_floats(stem, in_planes, in_counts, out_planes, num_out, outcome);
            returned = counts == num_out.data();
            // The caller's input counts are const to the adapter; what the stem was handed
            // is shape[1] of the tensors it presented, read back after the call.
            for (size_t slot = 0; !pops && slot < num_in.size(); ++slot) {
                num_in[slot] = static_cast<size_t>(m_adapter.inputs()[slot].shape[1]);
            }
        } else {
            std::vector<anira_tensor> in_tensors;
            for (size_t slot = 0; slot < in_counts.size(); ++slot) {
                const bool unused = options.m_null_unused && in_counts[slot] == 0;
                const bool shared = options.m_in_place && slot == 0;
                in_tensors.push_back(
                    unused ? empty_tensor(inputs[slot].channels())
                           : inputs[slot].tensor(in_counts[slot], ANIRA_DTYPE_F32, !shared));
            }
            std::vector<anira_tensor> out_tensors;
            for (size_t slot = 0; slot < out_counts.size(); ++slot) {
                const bool shared = options.m_in_place && slot == 0;
                HostBlock<float>& block = shared ? inputs[0] : outputs[slot];
                out_tensors.push_back(out_unused[slot]
                                          ? empty_tensor(outputs[slot].channels())
                                          : block.tensor(out_counts[slot], ANIRA_DTYPE_F32, false));
            }
            const std::vector<anira_tensor> in_before = in_tensors;
            const std::vector<anira_tensor> out_before = out_tensors;
            const size_t* counts = call_tensors(stem, in_tensors, out_tensors, outcome);
            returned = counts != nullptr;
            if (counts != nullptr) { std::copy_n(counts, num_out.size(), num_out.begin()); }
            // A descriptor is never written, an output's included.
            for (size_t slot = 0; slot < in_tensors.size(); ++slot) {
                EXPECT_TRUE(same_record(in_tensors[slot], in_before[slot])) << "input " << slot;
                num_in[slot] = static_cast<size_t>(in_tensors[slot].shape[1]);
            }
            for (size_t slot = 0; slot < out_tensors.size(); ++slot) {
                EXPECT_TRUE(same_record(out_tensors[slot], out_before[slot])) << "output " << slot;
            }
        }

        std::string header = "#" + std::to_string(call_number) + " " + name_of(stem);
        if (!pops) { header += " in=" + oracle::format_counts(in_counts); }
        header += " out=" + oracle::format_counts(out_counts);
        if (options.m_in_place) { header += " in-place"; }
        if (options.m_null_unused) { header += " null-unused"; }
        header += " ->";
        if (!pops) { header += " in=" + oracle::format_counts(num_in); }
        header += " out=" + oracle::format_counts(num_out);
        header += returned ? " returned=out" : " returned=other";
        header += std::string(" missed=") + (m_manager.last_block_missed() ? "1" : "0");
        if (stem == Stem::ProcessWait || stem == Stem::PopDataWait) {
            header += std::string(" outcome=") + name_of(outcome);
        }
        m_transcript.add(header + " " + oracle::format_rings(*m_session));
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            if (out_unused[slot]) { continue; }
            const bool shared = options.m_in_place && slot == 0;
            add_block(shared ? "io0" : "out" + std::to_string(slot),
                      shared ? inputs[0] : outputs[slot],
                      out_counts[slot]);
            EXPECT_TRUE(outputs[slot].slack_untouched())
                << "call " << call_number << ": output " << slot;
        }
        for (size_t slot = 0; slot < inputs.size(); ++slot) {
            EXPECT_TRUE(inputs[slot].slack_untouched())
                << "call " << call_number << ": input " << slot;
            if (options.m_in_place && slot == 0) { continue; }
            for (size_t channel = 0; channel < inputs[slot].channels(); ++channel) {
                EXPECT_EQ(inputs[slot].run(channel), inputs_before[slot][channel])
                    << "call " << call_number << ": input " << slot << ", channel " << channel;
            }
        }
        if (options.m_settle) { settle(); }
    }

    void push(const std::vector<size_t>& in_counts, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call_number = ++m_calls;
        std::vector<HostBlock<float>> inputs = make_inputs(in_counts, 0, call_number);
        // The counts the stem was handed, read back from shape[1] after the call.
        std::vector<size_t> num_in;
        if (m_face == Face::FloatAdapter) {
            std::vector<const float* const*> in_planes;
            for (size_t slot = 0; slot < in_counts.size(); ++slot) {
                const bool unused = options.m_null_unused && in_counts[slot] == 0;
                in_planes.push_back(unused ? nullptr : inputs[slot].planes());
            }
            const anira_tensor* presented =
                m_adapter.present_inputs(in_planes.data(), in_counts.data());
            m_manager.push_data(presented);
            for (size_t slot = 0; slot < in_counts.size(); ++slot) {
                num_in.push_back(static_cast<size_t>(presented[slot].shape[1]));
            }
        } else {
            std::vector<anira_tensor> in_tensors;
            for (size_t slot = 0; slot < in_counts.size(); ++slot) {
                const bool unused = options.m_null_unused && in_counts[slot] == 0;
                in_tensors.push_back(
                    unused ? empty_tensor(inputs[slot].channels())
                           : inputs[slot].tensor(in_counts[slot], ANIRA_DTYPE_F32, true));
            }
            m_manager.push_data(in_tensors.data());
            for (const anira_tensor& tensor : in_tensors) {
                num_in.push_back(static_cast<size_t>(tensor.shape[1]));
            }
        }
        std::string header =
            "#" + std::to_string(call_number) + " push_data in=" + oracle::format_counts(in_counts);
        if (options.m_null_unused) { header += " null-unused"; }
        m_transcript.add(header + " -> in=" + oracle::format_counts(num_in) + " " +
                         oracle::format_rings(*m_session));
        for (const HostBlock<float>& block : inputs) { EXPECT_TRUE(block.slack_untouched()); }
        if (options.m_settle) { settle(); }
    }

    void reset() {
        if (m_session == nullptr) { return; }
        m_manager.reset();
        m_transcript.add("#" + std::to_string(++m_calls) + " reset " +
                         oracle::format_rings(*m_session));
    }

    /// Opens the gate until every submitted inference is collected. In Mode::Waiting the one
    /// collection waits by itself.
    void settle() {
        if (m_session == nullptr) { return; }
        if (m_mode == Mode::Clocked) {
            oracle::settle(*m_gate, *m_session, [this] { m_manager.get_available_samples(0, 0); });
        } else {
            m_manager.get_available_samples(0, 0);
        }
        m_transcript.add("  settled: " + oracle::format_rings(*m_session));
    }

private:
    Layout in_layout() const { return m_face == Face::Tensor ? m_layouts.m_in : Layout::Planar; }
    Layout out_layout() const { return m_face == Face::Tensor ? m_layouts.m_out : Layout::Planar; }

    /// The ramps of oracle::make_inputs, laid out as the rig's input layout says.
    std::vector<HostBlock<float>> make_inputs(const std::vector<size_t>& in_counts,
                                              size_t slot0_length,
                                              size_t call_number) {
        const std::vector<oracle::SlotBlock> ramps =
            oracle::make_inputs(m_config, m_positions, in_counts, slot0_length, call_number);
        std::vector<HostBlock<float>> inputs;
        inputs.reserve(ramps.size());
        for (const oracle::SlotBlock& ramp : ramps) {
            const size_t length = ramp.run(0).size() - oracle::k_guard;
            inputs.emplace_back(in_layout(), ramp.channels(), length, oracle::k_untouched);
            for (size_t channel = 0; channel < ramp.channels(); ++channel) {
                for (size_t index = 0; index < ramp.run(channel).size(); ++index) {
                    inputs.back().at(channel, index) = ramp.run(channel)[index];
                }
            }
        }
        return inputs;
    }

    void add_block(const std::string& label, const HostBlock<float>& block, size_t count) {
        for (size_t channel = 0; channel < block.channels(); ++channel) {
            m_transcript.add("  " + label + ".c" + std::to_string(channel) + ":" +
                             oracle::format_run(block.run(channel), count));
        }
    }

    /// The float face: the adapter presents the channel pointers (a pop presents no input),
    /// the tensor stem runs on its tensors, deliver_counts() writes the counts into num_out
    /// and returns that array.
    const size_t* call_floats(Stem stem,
                              const std::vector<const float* const*>& in_planes,
                              const std::vector<size_t>& num_in,
                              const std::vector<float* const*>& out_planes,
                              std::vector<size_t>& num_out,
                              Core::WaitOutcome& outcome) {
        const bool pops =
            stem == Stem::PopData || stem == Stem::PopDataUntil || stem == Stem::PopDataWait;
        const anira_tensor* inputs =
            pops ? nullptr : m_adapter.present_inputs(in_planes.data(), num_in.data());
        const anira_tensor* outputs = m_adapter.present_outputs(out_planes.data(), num_out.data());
        const size_t* delivered = call_stem(stem, inputs, outputs, outcome);
        return delivered == nullptr ? nullptr : m_adapter.deliver_counts(delivered, num_out.data());
    }

    const size_t* call_tensors(Stem stem,
                               const std::vector<anira_tensor>& inputs,
                               const std::vector<anira_tensor>& outputs,
                               Core::WaitOutcome& outcome) {
        return call_stem(stem, inputs.data(), outputs.data(), outcome);
    }

    /// The tensor stem of each name, over the arrays of either face.
    const size_t* call_stem(Stem stem,
                            const anira_tensor* inputs,
                            const anira_tensor* outputs,
                            Core::WaitOutcome& outcome) {
        constexpr auto k_forever = std::chrono::steady_clock::duration::max();
        switch (stem) {
            case Stem::Process: return m_manager.process(inputs, outputs);
            case Stem::ProcessNowait: return m_manager.process_nowait(inputs, outputs);
            case Stem::ProcessWait:
                return m_manager.process_wait(inputs, outputs, k_forever, outcome);
            case Stem::PopData: return m_manager.pop_data(outputs);
            case Stem::PopDataUntil:
                return m_manager.pop_data(outputs, std::chrono::steady_clock::time_point::max());
            case Stem::PopDataWait: return m_manager.pop_data_wait(outputs, k_forever, outcome);
        }
        return nullptr;
    }

    InferenceConfig m_config;
    PrePostProcessor m_pp_processor;
    std::unique_ptr<anira_test::GateBackend> m_gate;
    InferenceManager m_manager;
    PlanarFloatAdapter m_adapter;  ///< Face::FloatAdapter: sized beside the manager's prepare
    Face m_face;
    Layouts m_layouts;
    Mode m_mode;
    std::shared_ptr<SessionElement> m_session;
    std::vector<size_t> m_positions;  ///< per input slot, the samples pushed so far
    size_t m_calls = 0;
    oracle::Transcript m_transcript;
};

// ---- the scenarios of test_CopyPathOracle.cpp --------------------------------------------------
// Call for call the scripts the golden transcripts were recorded with; that file says what
// each one covers.

void block_sizes_script(Rig& rig) {
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {3}, {3});
    rig.call(Stem::ProcessNowait, {3}, {3});
    rig.call(Stem::ProcessNowait, {3}, {3});
    rig.call(Stem::ProcessNowait, {0}, {0});
    rig.call(Stem::ProcessNowait, {13}, {13});
    rig.call(Stem::ProcessNowait, {8}, {8}, k_in_place);
    rig.call(Stem::ProcessNowait, {3}, {3}, k_in_place);
    rig.call(Stem::ProcessNowait, {13}, {13}, k_in_place);
    rig.call(Stem::ProcessNowait, {0}, {0}, k_in_place);
    rig.call(Stem::Process, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {40});
    rig.call(Stem::ProcessNowait, {8}, {40}, k_in_place);
    rig.call(Stem::ProcessNowait, {8}, {3});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.reset();
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {20}, {13});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
}

void starved_blocks_script(Rig& rig) {
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {13}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {8}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {3}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {8}, k_in_place_gate_closed);
    rig.call(Stem::ProcessNowait, {5}, {3}, k_gate_closed);
    rig.call(Stem::PopData, {}, {8}, k_gate_closed);
    rig.settle();
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.reset();
    rig.call(Stem::PopData, {}, {20});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
}

void push_pop_script(Rig& rig) {
    rig.push({8});
    rig.push({3});
    rig.push({5});
    rig.call(Stem::PopData, {}, {5});
    rig.call(Stem::PopData, {}, {11});
    rig.call(Stem::PopData, {}, {0});
    rig.call(Stem::PopData, {}, {30});
    rig.push({8});
    rig.call(Stem::PopData, {}, {8});
    rig.push({8}, k_gate_closed);
    rig.call(Stem::PopDataUntil, {}, {8});
    rig.call(Stem::PopData, {}, {13});
}

void multi_slot_script(Rig& rig) {
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 3});
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 3});
    rig.call(Stem::ProcessNowait, {8, 0}, {8, 0}, k_null_unused);
    rig.call(Stem::ProcessNowait, {0, 3}, {0, 3}, k_null_unused);
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 5});
    rig.call(Stem::ProcessNowait, {8, 2}, {8, 2}, k_in_place);
    rig.push({8, 3});
    rig.push({0, 3}, k_null_unused);
    rig.call(Stem::PopData, {}, {8, 3});
    rig.call(Stem::PopData, {}, {0, 3}, k_null_unused);
    rig.call(Stem::PopData, {}, {8, 0}, k_null_unused);
    rig.call(Stem::ProcessNowait, {8, 3}, {40, 5});
    rig.call(Stem::ProcessNowait, {8, 3}, {40, 2}, k_in_place);
    rig.call(Stem::PopData, {}, {40, 3});
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 3});
    rig.reset();
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 3});
    rig.call(Stem::ProcessNowait, {8, 3}, {8, 3});
}

void mono_to_stereo_script(Rig& rig) {
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {13}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {8}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {3}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {3}, k_gate_closed);
    rig.settle();
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
}

void generator_script(Rig& rig) {
    rig.push({4});
    rig.call(Stem::PopData, {}, {8});
    rig.call(Stem::PopData, {}, {8});
    rig.call(Stem::PopData, {}, {3});
    rig.call(Stem::ProcessNowait, {4}, {13});
    rig.call(Stem::ProcessNowait, {2}, {8});
    rig.call(Stem::PopData, {}, {20});
    rig.call(Stem::PopData, {}, {8});
    rig.reset();
    rig.call(Stem::PopData, {}, {8});
    rig.call(Stem::PopData, {}, {8});
}

void waiting_stems_script(Rig& rig) {
    rig.call(Stem::Process, {8}, {8});
    rig.call(Stem::Process, {8}, {8});
    rig.call(Stem::ProcessWait, {8}, {8});
    rig.call(Stem::ProcessWait, {2}, {2}, k_in_place);
    rig.call(Stem::ProcessNowait, {13}, {13});
    rig.push({8});
    rig.call(Stem::PopDataUntil, {}, {8});
    rig.push({8});
    rig.call(Stem::PopDataWait, {}, {8});
    rig.call(Stem::PopData, {}, {3});
    rig.call(Stem::ProcessWait, {8}, {40});
    rig.call(Stem::Process, {8}, {40}, k_in_place);
    rig.call(Stem::PopDataWait, {}, {40});
    rig.call(Stem::PopDataUntil, {}, {40});
    rig.call(Stem::ProcessWait, {8}, {8});
    rig.call(Stem::Process, {8}, {8});
}

/// Every description a transcript can be compared under (Layout::Packed has no guard between
/// its channels and is tested by value below), alone and crossed between input and output.
constexpr std::array<Layouts, 9> k_descriptions{{
    {.m_in = Layout::Planar, .m_out = Layout::Planar},
    {.m_in = Layout::Interleaved, .m_out = Layout::Interleaved},
    {.m_in = Layout::Contiguous, .m_out = Layout::Contiguous},
    {.m_in = Layout::PlanarOffset, .m_out = Layout::PlanarOffset},
    {.m_in = Layout::PlanarStrided, .m_out = Layout::PlanarStrided},
    {.m_in = Layout::Interleaved, .m_out = Layout::Planar},
    {.m_in = Layout::Planar, .m_out = Layout::Interleaved},
    {.m_in = Layout::Contiguous, .m_out = Layout::Interleaved},
    {.m_in = Layout::Interleaved, .m_out = Layout::PlanarStrided},
}};

/// Line by line against a transcript, reporting the call a difference happened in. Unlike
/// oracle::expect_golden it never records.
void expect_transcript(const std::string& what,
                       const std::vector<std::string_view>& expected,
                       const oracle::Transcript& transcript) {
    const std::vector<std::string>& actual = transcript.lines();
    EXPECT_EQ(actual.size(), expected.size()) << what << ": the number of transcript lines";
    constexpr size_t k_max_reports = 4;
    size_t reports = 0;
    std::string_view call = "(before the first call)";
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        if (!expected[i].empty() && expected[i].front() == '#') { call = expected[i]; }
        if (actual[i] == expected[i]) { continue; }
        ADD_FAILURE() << what << ", line " << i << ", in the call\n    " << call
                      << "\n  expected: " << expected[i] << "\n  now:      " << actual[i];
        if (++reports == k_max_reports) { break; }
    }
}

using Script = void (*)(Rig&);

/// The script through the tensor stems under every description, against the recording.
void expect_recording(const InferenceConfig& config,
                      anira_miss_policy policy,
                      Script script,
                      std::string_view golden,
                      Mode mode = Mode::Clocked,
                      Model model = Model::PassThrough) {
    const std::vector<std::string_view> expected = oracle::split_lines(golden);
    for (const Layouts& layouts : k_descriptions) {
        Rig rig(config, policy, Face::Tensor, layouts, mode, model);
        script(rig);
        expect_transcript(
            std::string(name_of(layouts.m_in)) + " in, " + name_of(layouts.m_out) + " out",
            expected,
            rig.transcript());
    }
}

// ---- the fixtures of the tests by value --------------------------------------------------------

/// Sample `position` of channel `channel` of the pass-through's output: the input ramp, late
/// by the latency, zeros before it.
float delayed_value(size_t channel, size_t position, size_t latency) {
    return position < latency ? 0.F : oracle::stream_value(0, channel, position - latency);
}

/// A block of `count` samples per channel that continues the input ramp at `position`.
HostBlock<float> ramp_block(Layout layout, size_t channels, size_t count, size_t position) {
    HostBlock<float> block(layout, channels, count, oracle::k_untouched);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t i = 0; i < count; ++i) {
            block.at(channel, i) = oracle::stream_value(0, channel, position + i);
        }
    }
    return block;
}

constexpr std::array<size_t, 4> k_channel_counts{1, 2, 3, 6};

// The stems as a driver thread calls them. RealtimeSanitizer looks at what runs inside a
// nonblocking function, and a planar float32 block, which is all the float adapter presents,
// only ever reaches the unit-stride paths; these bring the strided ones into its view. No gtest
// assertion in here: a failing one allocates.
const size_t* process_on_the_driver_thread(InferenceManager& manager,
                                           const anira_tensor* inputs,
                                           const anira_tensor* outputs) ANIRA_NONBLOCKING {
    return manager.process_nowait(inputs, outputs);
}

void push_on_the_driver_thread(InferenceManager& manager,
                               const anira_tensor* inputs) ANIRA_NONBLOCKING {
    manager.push_data(inputs);
}

const size_t* pop_on_the_driver_thread(InferenceManager& manager,
                                       const anira_tensor* outputs) ANIRA_NONBLOCKING {
    return manager.pop_data(outputs);
}

int16_t i16_value(size_t channel, size_t position) {
    return static_cast<int16_t>((channel * 1000) + position + 1);
}

constexpr int16_t k_untouched_i16 = -7;

RingDtypes i16_rings() {
    return RingDtypes{.m_inputs = {ANIRA_DTYPE_I16}, .m_outputs = {ANIRA_DTYPE_I16}};
}

}  // namespace

// ---- (1) the recording, through the tensor stems, under every description ----------------------

TEST(TensorStems, StereoBlockSizesReproduceTheRecording) {
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_ZEROS,
                     block_sizes_script,
                     oracle::k_manager_block_sizes_zeros);
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_HOLD_LAST,
                     block_sizes_script,
                     oracle::k_manager_block_sizes_hold_last);
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_BYPASS,
                     block_sizes_script,
                     oracle::k_manager_block_sizes_bypass);
}

// HOLD_LAST and BYPASS through tensors: the held block captured from and delivered into an
// interleaved, an offset and a strided block, the BYPASS copy across descriptions (interleaved
// in, planar out and the reverse) and its in-place skip, request above the hold capacity.
TEST(TensorStems, StereoStarvedBlocksReproduceTheRecording) {
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_ZEROS,
                     starved_blocks_script,
                     oracle::k_manager_starved_zeros);
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_HOLD_LAST,
                     starved_blocks_script,
                     oracle::k_manager_starved_hold_last);
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_BYPASS,
                     starved_blocks_script,
                     oracle::k_manager_starved_bypass);
}

TEST(TensorStems, StereoPushPopReproducesTheRecording) {
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_ZEROS,
                     push_pop_script,
                     oracle::k_manager_push_pop_zeros);
    expect_recording(pass_through_config(2),
                     ANIRA_MISS_HOLD_LAST,
                     push_pop_script,
                     oracle::k_manager_push_pop_hold_last);
}

// Three channels beside a Static slot on either side: a Static tensor is [1, values] under any
// description, a slot the call does not carry is an empty tensor without memory.
TEST(TensorStems, MultiSlotWithStaticTensorsReproducesTheRecording) {
    expect_recording(multi_config(),
                     ANIRA_MISS_ZEROS,
                     multi_slot_script,
                     oracle::k_manager_multi_zeros);
    expect_recording(multi_config(),
                     ANIRA_MISS_HOLD_LAST,
                     multi_slot_script,
                     oracle::k_manager_multi_hold_last);
    expect_recording(multi_config(),
                     ANIRA_MISS_BYPASS,
                     multi_slot_script,
                     oracle::k_manager_multi_bypass);
}

TEST(TensorStems, MonoToStereoBypassReproducesTheRecording) {
    expect_recording(mono_to_stereo_config(),
                     ANIRA_MISS_BYPASS,
                     mono_to_stereo_script,
                     oracle::k_manager_mono_to_stereo_bypass,
                     Mode::Clocked,
                     Model::FanOut);
}

TEST(TensorStems, GeneratorReproducesTheRecording) {
    expect_recording(generator_config(),
                     ANIRA_MISS_BYPASS,
                     generator_script,
                     oracle::k_manager_generator,
                     Mode::Clocked,
                     Model::ParamRamp);
}

// process_wait and pop_data_wait over tensors, and the two 2.x forms: the dispatcher (process
// with contract_wait_budget under the blocking ratio) and the deadline form of pop_data.
TEST(TensorStems, WaitingStemsReproduceTheRecording) {
    expect_recording(pass_through_config(2, 0.5F),
                     ANIRA_MISS_BYPASS,
                     waiting_stems_script,
                     oracle::k_manager_waiting_stems,
                     Mode::Waiting);
}

// ---- (2) the float adapter against the tensor stems, for every channel count -------------------

// The same scenario through the float adapter (channel pointers presented as planar float32
// tensors) and through hand-built tensors gives the same transcript, for 1, 2, 3 and 6
// channels, under the three policies.
TEST(TensorStems, TheFloatAdapterAndTheTensorStemsAgreeForEveryChannelCount) {
    constexpr std::array<anira_miss_policy, 3> k_policies{ANIRA_MISS_ZEROS,
                                                          ANIRA_MISS_HOLD_LAST,
                                                          ANIRA_MISS_BYPASS};
    constexpr std::array<Script, 2> k_scripts{block_sizes_script, starved_blocks_script};
    constexpr std::array<Layouts, 3> k_pairs{{
        {.m_in = Layout::Planar, .m_out = Layout::Planar},
        {.m_in = Layout::Interleaved, .m_out = Layout::Interleaved},
        {.m_in = Layout::Interleaved, .m_out = Layout::PlanarOffset},
    }};
    for (const size_t channels : k_channel_counts) {
        for (const anira_miss_policy policy : k_policies) {
            for (const Script script : k_scripts) {
                Rig floats(pass_through_config(channels), policy, Face::FloatAdapter);
                script(floats);
                std::vector<std::string_view> expected;
                for (const std::string& line : floats.transcript().lines()) {
                    expected.emplace_back(line);
                }
                for (const Layouts& layouts : k_pairs) {
                    Rig tensors(pass_through_config(channels), policy, Face::Tensor, layouts);
                    script(tensors);
                    expect_transcript(std::to_string(channels) + " channels, " +
                                          name_of(layouts.m_in) + " in, " + name_of(layouts.m_out) +
                                          " out",
                                      expected,
                                      tensors.transcript());
                }
            }
        }
    }
}

// ---- (3) by value -------------------------------------------------------------------------------

// A white-box look at the send ring: after a push under any description, channel c of the
// ring holds channel c's ramp. 5 samples stay below the hop, so nothing consumes them.
TEST(TensorStems, APushLandsEveryChannelInItsRingUnderEveryDescription) {
    constexpr size_t k_count = 5;
    constexpr std::array<Layout, 6> k_layouts{Layout::Planar,
                                              Layout::PlanarOffset,
                                              Layout::PlanarStrided,
                                              Layout::Contiguous,
                                              Layout::Interleaved,
                                              Layout::Packed};
    for (const size_t channels : k_channel_counts) {
        for (const Layout layout : k_layouts) {
            Rig rig(pass_through_config(channels), ANIRA_MISS_ZEROS, Face::Tensor);
            ASSERT_TRUE(rig.ready());
            HostBlock<float> block = ramp_block(layout, channels, k_count, 0);
            const anira_tensor input = block.tensor(k_count, ANIRA_DTYPE_F32, true);
            rig.manager().push_data(&input);
            for (size_t channel = 0; channel < channels; ++channel) {
                ASSERT_EQ(rig.session().m_send_buffer[0].get_available_samples(channel), k_count);
                std::array<float, k_count> ring{};
                rig.session().m_send_buffer[0].pop_block(channel, ring.data(), k_count);
                for (size_t i = 0; i < k_count; ++i) {
                    EXPECT_EQ(ring[i], oracle::stream_value(0, channel, i))
                        << channels << " channels, " << name_of(layout) << ", channel " << channel
                        << ", sample " << i;
                }
            }
            EXPECT_TRUE(block.slack_untouched());
        }
    }
}

// Absolute expectations end to end: every channel of the pass-through comes back late by the
// reported latency, for 1, 2, 3 and 6 channels and for the descriptions crossed between input
// and output, the packed block (all-zero strides) among them.
TEST(TensorStems, EveryChannelComesBackLateByTheLatency) {
    constexpr std::array<Layouts, 6> k_pairs{{
        {.m_in = Layout::Interleaved, .m_out = Layout::Interleaved},
        {.m_in = Layout::Interleaved, .m_out = Layout::Planar},
        {.m_in = Layout::Planar, .m_out = Layout::Interleaved},
        {.m_in = Layout::Packed, .m_out = Layout::Packed},
        {.m_in = Layout::Packed, .m_out = Layout::Interleaved},
        {.m_in = Layout::Interleaved, .m_out = Layout::Packed},
    }};
    constexpr size_t k_blocks = 6;
    for (const size_t channels : k_channel_counts) {
        for (const Layouts& layouts : k_pairs) {
            Rig rig(pass_through_config(channels), ANIRA_MISS_ZEROS, Face::Tensor);
            ASSERT_TRUE(rig.ready());
            const size_t latency = rig.manager().get_latency()[0];
            for (size_t block_index = 0; block_index < k_blocks; ++block_index) {
                const size_t position = block_index * k_hop;
                HostBlock<float> in = ramp_block(layouts.m_in, channels, k_hop, position);
                HostBlock<float> out(layouts.m_out, channels, k_hop, oracle::k_untouched);
                const anira_tensor input = in.tensor(k_hop, ANIRA_DTYPE_F32, true);
                const anira_tensor output = out.tensor(k_hop, ANIRA_DTYPE_F32, false);
                const size_t* delivered = rig.manager().process_nowait(&input, &output);
                ASSERT_NE(delivered, nullptr);
                EXPECT_EQ(delivered[0], k_hop);
                EXPECT_FALSE(rig.manager().last_block_missed());
                for (size_t channel = 0; channel < channels; ++channel) {
                    for (size_t i = 0; i < k_hop; ++i) {
                        EXPECT_EQ(out.at(channel, i), delayed_value(channel, position + i, latency))
                            << channels << " channels, " << name_of(layouts.m_in) << " in, "
                            << name_of(layouts.m_out) << " out, channel " << channel
                            << ", position " << position + i;
                    }
                }
                EXPECT_TRUE(out.slack_untouched());
                EXPECT_TRUE(in.slack_untouched());
                rig.settle();
            }
        }
    }
}

// In place is the same tensor on both sides: one record handed over as the input and as the
// output. Delivered, the block is overwritten with the late stream; on a BYPASS miss the
// samples stay where they are, for an interleaved block and for a planar one.
TEST(TensorStems, TheSameTensorOnBothSidesIsInPlace) {
    for (const Layout layout : {Layout::Interleaved, Layout::Planar, Layout::Packed}) {
        Rig rig(pass_through_config(2), ANIRA_MISS_BYPASS, Face::Tensor);
        ASSERT_TRUE(rig.ready());
        const size_t latency = rig.manager().get_latency()[0];
        // Blocks 0 to 2 are delivered; the inferences of blocks 2 and 3 stay at the gate, so
        // block 3 finds 7 samples for a request of 8.
        for (size_t block_index = 0; block_index < 4; ++block_index) {
            const size_t position = block_index * k_hop;
            HostBlock<float> block = ramp_block(layout, 2, k_hop, position);
            const anira_tensor tensor = block.tensor(k_hop, ANIRA_DTYPE_F32, false);
            const size_t* delivered = rig.manager().process_nowait(&tensor, &tensor);
            ASSERT_NE(delivered, nullptr);
            const bool missed = block_index == 3;
            EXPECT_EQ(rig.manager().last_block_missed(), missed) << name_of(layout);
            EXPECT_EQ(delivered[0], missed ? 0U : k_hop) << name_of(layout);
            for (size_t channel = 0; channel < 2; ++channel) {
                for (size_t i = 0; i < k_hop; ++i) {
                    EXPECT_EQ(block.at(channel, i),
                              missed ? oracle::stream_value(0, channel, position + i)
                                     : delayed_value(channel, position + i, latency))
                        << name_of(layout) << ", block " << block_index << ", channel " << channel
                        << ", sample " << i;
                }
            }
            EXPECT_TRUE(block.slack_untouched());
            if (block_index < 2) { rig.settle(); }
        }
    }
}

// Six channels, interleaved in and strided planes out, called from a nonblocking function: the
// delivered path, then a starved block under HOLD_LAST (the held block goes out through the
// strided planes it was captured from) and under BYPASS (interleaved memory to strided
// planes), then the split calls. Allocating, locking or waiting anywhere below the stems
// fails this under RealtimeSanitizer.
TEST(TensorStems, TheStridedPathsAreLegalOnTheDriverThread) {
    constexpr size_t k_channels = 6;
    for (const anira_miss_policy policy : {ANIRA_MISS_HOLD_LAST, ANIRA_MISS_BYPASS}) {
        Rig rig(pass_through_config(k_channels), policy, Face::Tensor);
        ASSERT_TRUE(rig.ready());
        const size_t latency = rig.manager().get_latency()[0];
        // Blocks 0 to 2 are delivered; the inferences of blocks 2 and 3 stay at the gate, so
        // block 3 finds 7 samples for a request of 8.
        for (size_t block_index = 0; block_index < 4; ++block_index) {
            const size_t position = block_index * k_hop;
            HostBlock<float> in = ramp_block(Layout::Interleaved, k_channels, k_hop, position);
            HostBlock<float> out(Layout::PlanarStrided, k_channels, k_hop, oracle::k_untouched);
            const anira_tensor input = in.tensor(k_hop, ANIRA_DTYPE_F32, true);
            const anira_tensor output = out.tensor(k_hop, ANIRA_DTYPE_F32, false);
            const size_t* delivered = process_on_the_driver_thread(rig.manager(), &input, &output);
            ASSERT_NE(delivered, nullptr);
            const bool missed = block_index == 3;
            EXPECT_EQ(rig.manager().last_block_missed(), missed);
            EXPECT_EQ(delivered[0], missed ? 0U : k_hop);
            for (size_t channel = 0; channel < k_channels; ++channel) {
                for (size_t i = 0; i < k_hop; ++i) {
                    float expected = delayed_value(channel, position + i, latency);
                    if (missed) {
                        expected = policy == ANIRA_MISS_BYPASS
                                       ? oracle::stream_value(0, channel, position + i)
                                       : delayed_value(channel, (2 * k_hop) + i, latency);
                    }
                    EXPECT_EQ(out.at(channel, i), expected)
                        << "policy " << policy << ", block " << block_index << ", channel "
                        << channel << ", sample " << i;
                }
            }
            EXPECT_TRUE(out.slack_untouched());
            if (block_index < 2) { rig.settle(); }
        }
    }

    Rig rig(pass_through_config(k_channels), ANIRA_MISS_ZEROS, Face::Tensor);
    ASSERT_TRUE(rig.ready());
    HostBlock<float> in = ramp_block(Layout::Interleaved, k_channels, k_hop, 0);
    HostBlock<float> out(Layout::Interleaved, k_channels, k_hop, oracle::k_untouched);
    const anira_tensor input = in.tensor(k_hop, ANIRA_DTYPE_F32, true);
    const anira_tensor output = out.tensor(k_hop, ANIRA_DTYPE_F32, false);
    push_on_the_driver_thread(rig.manager(), &input);
    rig.settle();
    const size_t* delivered = pop_on_the_driver_thread(rig.manager(), &output);
    ASSERT_NE(delivered, nullptr);
    EXPECT_EQ(delivered[0], k_hop);
    for (size_t channel = 0; channel < k_channels; ++channel) {
        for (size_t i = 0; i < k_hop; ++i) { EXPECT_EQ(out.at(channel, i), 0.F) << "the latency"; }
    }
}

// A slot whose shape[1] is 0 is not touched and its memory is not read: an empty tensor
// without memory, and a zeroed record (what a refused factory leaves), beside a carried slot.
TEST(TensorStems, EmptyAndZeroedTensorsAreNotTouched) {
    Rig rig(multi_config(), ANIRA_MISS_ZEROS, Face::Tensor);
    ASSERT_TRUE(rig.ready());
    const std::array<anira_tensor, 2> nothing_in{anira_tensor{}, empty_tensor(1)};
    const std::array<anira_tensor, 2> nothing_out{empty_tensor(3), anira_tensor{}};
    const size_t* delivered = rig.manager().process_nowait(nothing_in.data(), nothing_out.data());
    ASSERT_NE(delivered, nullptr);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_EQ(delivered[1], 0U);
    EXPECT_FALSE(rig.manager().last_block_missed());
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);

    // The stream slot carried, the Static slot zeroed on both sides.
    HostBlock<float> in = ramp_block(Layout::Interleaved, 3, 5, 0);
    HostBlock<float> out(Layout::Interleaved, 3, 5, oracle::k_untouched);
    const std::array<anira_tensor, 2> inputs{in.tensor(5, ANIRA_DTYPE_F32, true), anira_tensor{}};
    const std::array<anira_tensor, 2> outputs{out.tensor(5, ANIRA_DTYPE_F32, false),
                                              anira_tensor{}};
    delivered = rig.manager().process_nowait(inputs.data(), outputs.data());
    EXPECT_EQ(delivered[0], 5U);  // the latency's zeros
    EXPECT_EQ(delivered[1], 0U);
    for (size_t channel = 0; channel < 3; ++channel) {
        EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(channel), 5U);
        for (size_t i = 0; i < 5; ++i) { EXPECT_EQ(out.at(channel, i), 0.F); }
    }
    EXPECT_TRUE(out.slack_untouched());
}

// A push of three host blocks at once, interleaved, six channels: the send ring (15) keeps the
// tail, one hop of it is consumed by the inference the push submits, and nothing is read or
// written past the block (the sanitizer legs run this).
TEST(TensorStems, AnOversizedInterleavedPushKeepsTheTail) {
    constexpr size_t k_channels = 6;
    constexpr size_t k_count = 3 * k_hop;
    Rig rig(pass_through_config(k_channels), ANIRA_MISS_ZEROS, Face::Tensor);
    ASSERT_TRUE(rig.ready());
    const size_t capacity = rig.session().m_send_buffer[0].get_num_samples();
    ASSERT_LT(capacity, k_count);
    ASSERT_GT(capacity, k_hop);
    HostBlock<float> block = ramp_block(Layout::Interleaved, k_channels, k_count, 0);
    const anira_tensor input = block.tensor(k_count, ANIRA_DTYPE_F32, true);
    rig.manager().push_data(&input);
    const size_t left = capacity - k_hop;
    for (size_t channel = 0; channel < k_channels; ++channel) {
        ASSERT_EQ(rig.session().m_send_buffer[0].get_available_samples(channel), left);
        std::vector<float> ring(left, 0.F);
        rig.session().m_send_buffer[0].pop_block(channel, ring.data(), left);
        for (size_t i = 0; i < left; ++i) {
            EXPECT_EQ(ring[i], oracle::stream_value(0, channel, k_count - left + i))
                << "channel " << channel << ", sample " << i;
        }
    }
    EXPECT_TRUE(block.slack_untouched());
    rig.settle();
}

// ---- a ring that is not float32 ----------------------------------------------------------------
// A session whose rings are int16 can be prepared through InferenceManager::prepare(RingDtypes),
// but no backend can run it before the pre- and post-processors read typed rings: they use the
// float face of the ring, which does nothing on an int16 ring. So the int16 copy is tested at
// the ring boundary, with no inference in play: pushes below the hop (nothing is submitted)
// read back from the send ring, and pops of samples this test puts into the receive ring.

TEST(TensorStems, Int16TensorsCopyIntoAndOutOfInt16Rings) {
    for (const Layout layout : {Layout::Interleaved, Layout::PlanarOffset, Layout::Packed}) {
        Rig rig(pass_through_config(2),
                ANIRA_MISS_HOLD_LAST,
                Face::Tensor,
                {},
                Mode::Clocked,
                Model::PassThrough,
                i16_rings());
        ASSERT_TRUE(rig.ready());
        RingBuffer& send = rig.session().m_send_buffer[0];
        RingBuffer& receive = rig.session().m_receive_buffer[0];
        ASSERT_EQ(send.dtype(), ANIRA_DTYPE_I16);
        ASSERT_EQ(receive.dtype(), ANIRA_DTYPE_I16);

        // Host -> ring.
        constexpr size_t k_count = 5;
        HostBlock<int16_t> in(layout, 2, k_count, k_untouched_i16);
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t i = 0; i < k_count; ++i) { in.at(channel, i) = i16_value(channel, i); }
        }
        const anira_tensor input = in.tensor(k_count, ANIRA_DTYPE_I16, true);
        rig.manager().push_data(&input);
        for (size_t channel = 0; channel < 2; ++channel) {
            ASSERT_EQ(send.get_available_samples(channel), k_count) << name_of(layout);
            std::array<int16_t, k_count> ring{};
            ASSERT_EQ(send.pop_block(channel, ring.data(), ANIRA_DTYPE_I16, k_count), k_count);
            for (size_t i = 0; i < k_count; ++i) {
                EXPECT_EQ(ring[i], i16_value(channel, i)) << name_of(layout);
            }
        }

        // Ring -> host: the latency's zeros, then eight samples per channel put into the ring
        // here, delivered; then a starved request above what is held: the held block and
        // zeros, in int16.
        const size_t latency = rig.manager().get_latency()[0];
        HostBlock<int16_t> primed(layout, 2, latency, k_untouched_i16);
        const anira_tensor primed_tensor = primed.tensor(latency, ANIRA_DTYPE_I16, false);
        const size_t* delivered = rig.manager().pop_data(&primed_tensor);
        ASSERT_EQ(delivered[0], latency);
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t i = 0; i < latency; ++i) { EXPECT_EQ(primed.at(channel, i), 0); }
        }
        for (size_t channel = 0; channel < 2; ++channel) {
            std::array<int16_t, k_hop> samples{};
            for (size_t i = 0; i < k_hop; ++i) { samples[i] = i16_value(channel, 100 + i); }
            ASSERT_EQ(receive.push_block(channel, samples.data(), ANIRA_DTYPE_I16, k_hop), k_hop);
        }
        HostBlock<int16_t> out(layout, 2, k_hop, k_untouched_i16);
        const anira_tensor output = out.tensor(k_hop, ANIRA_DTYPE_I16, false);
        delivered = rig.manager().pop_data(&output);
        EXPECT_EQ(delivered[0], k_hop);
        EXPECT_FALSE(rig.manager().last_block_missed());
        constexpr size_t k_request = k_hop + 3;
        HostBlock<int16_t> held(layout, 2, k_request, k_untouched_i16);
        const anira_tensor held_tensor = held.tensor(k_request, ANIRA_DTYPE_I16, false);
        delivered = rig.manager().pop_data(&held_tensor);
        EXPECT_EQ(delivered[0], 0U);
        EXPECT_TRUE(rig.manager().last_block_missed());
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t i = 0; i < k_request; ++i) {
                if (i < k_hop) { EXPECT_EQ(out.at(channel, i), i16_value(channel, 100 + i)); }
                EXPECT_EQ(held.at(channel, i), i < k_hop ? i16_value(channel, 100 + i) : 0)
                    << name_of(layout) << ", channel " << channel << ", sample " << i;
            }
        }
        EXPECT_TRUE(in.slack_untouched());
        EXPECT_TRUE(out.slack_untouched());
        EXPECT_TRUE(held.slack_untouched());
    }
}

// BYPASS is host memory to host memory at the ring's element size: an int16 planar input
// passed through to an int16 interleaved output on a starved block.
TEST(TensorStems, Int16BypassCopiesAcrossDescriptions) {
    Rig rig(pass_through_config(2),
            ANIRA_MISS_BYPASS,
            Face::Tensor,
            {},
            Mode::Clocked,
            Model::PassThrough,
            i16_rings());
    ASSERT_TRUE(rig.ready());
    const size_t latency = rig.manager().get_latency()[0];
    HostBlock<int16_t> primed(Layout::Planar, 2, latency, k_untouched_i16);
    const anira_tensor primed_tensor = primed.tensor(latency, ANIRA_DTYPE_I16, false);
    ASSERT_EQ(rig.manager().pop_data(&primed_tensor)[0], latency);

    constexpr size_t k_pushed = 5;
    constexpr size_t k_request = 7;
    HostBlock<int16_t> in(Layout::Planar, 2, k_pushed, k_untouched_i16);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < k_pushed; ++i) { in.at(channel, i) = i16_value(channel, i); }
    }
    HostBlock<int16_t> out(Layout::Interleaved, 2, k_request, k_untouched_i16);
    const anira_tensor input = in.tensor(k_pushed, ANIRA_DTYPE_I16, true);
    const anira_tensor output = out.tensor(k_request, ANIRA_DTYPE_I16, false);
    const size_t* delivered = rig.manager().process_nowait(&input, &output);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_TRUE(rig.manager().last_block_missed());
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < k_request; ++i) {
            EXPECT_EQ(out.at(channel, i), i < k_pushed ? i16_value(channel, i) : 0)
                << "channel " << channel << ", sample " << i;
        }
    }
    EXPECT_TRUE(out.slack_untouched());
}

// ---- the dtype, the one thing the core checks --------------------------------------------------

// An int16 tensor on a float32 slot: the input is not pushed, the output is zero-filled at the
// tensor's own element size (two bytes per sample, nothing beyond), is not popped and reports
// 0; the site records it once.
TEST(TensorStems, ATensorOfAnotherDtypeIsNotCopied) {
    Rig rig(pass_through_config(2), ANIRA_MISS_ZEROS, Face::Tensor);
    ASSERT_TRUE(rig.ready());
    const RtLatch& site = detail::rt_site(RtSite::TensorDtypeMismatch);
    ASSERT_EQ(site.m_latched.load(), 0U) << "every prepare re-arms the sites";

    HostBlock<int16_t> in(Layout::Interleaved, 2, k_hop, k_untouched_i16);
    HostBlock<int16_t> out(Layout::Interleaved, 2, k_hop, k_untouched_i16);
    const anira_tensor input = in.tensor(k_hop, ANIRA_DTYPE_I16, true);
    const anira_tensor output = out.tensor(k_hop, ANIRA_DTYPE_I16, false);
    const size_t available = rig.session().m_receive_buffer[0].get_available_samples(0);
    const size_t* delivered = rig.manager().process_nowait(&input, &output);
    EXPECT_EQ(delivered[0], 0U);
    EXPECT_FALSE(rig.manager().last_block_missed());
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
    EXPECT_EQ(rig.session().m_receive_buffer[0].get_available_samples(0), available);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < k_hop; ++i) { EXPECT_EQ(out.at(channel, i), 0); }
        for (size_t i = k_hop; i < out.run_length(); ++i) {
            EXPECT_EQ(out.at(channel, i), k_untouched_i16) << "the guard";
        }
    }
    EXPECT_EQ(site.m_latched.load(), 1U);
    EXPECT_EQ(site.m_suppressed.load(), 1U) << "the input's record, then the output's";
}

// The float adapter on a session whose rings are int16: its tensors are float32, so nothing
// is pushed and nothing popped, the output is zeros and the count 0. Before the rewrite the
// float face of the ring did nothing there and the call reported a delivered block over
// memory it had not written.
TEST(TensorStems, TheFloatAdapterOnAnInt16RingCopiesNothing) {
    Rig rig(pass_through_config(2),
            ANIRA_MISS_ZEROS,
            Face::FloatAdapter,
            {},
            Mode::Clocked,
            Model::PassThrough,
            i16_rings());
    ASSERT_TRUE(rig.ready());
    const RtLatch& site = detail::rt_site(RtSite::TensorDtypeMismatch);
    HostBlock<float> in = ramp_block(Layout::Planar, 2, k_hop, 0);
    HostBlock<float> out(Layout::Planar, 2, k_hop, oracle::k_untouched);
    const std::array<const float* const*, 1> in_planes{in.planes()};
    const std::array<float* const*, 1> out_planes{out.planes()};
    const std::array<size_t, 1> num_in{k_hop};
    std::array<size_t, 1> num_out{k_hop};
    const size_t available = rig.session().m_receive_buffer[0].get_available_samples(0);
    const anira_tensor* inputs = rig.adapter().present_inputs(in_planes.data(), num_in.data());
    const anira_tensor* outputs = rig.adapter().present_outputs(out_planes.data(), num_out.data());
    const size_t* delivered =
        rig.adapter().deliver_counts(rig.manager().process_nowait(inputs, outputs), num_out.data());
    EXPECT_EQ(delivered, num_out.data());
    EXPECT_EQ(num_out[0], 0U);
    EXPECT_EQ(inputs[0].shape[1], static_cast<int64_t>(k_hop)) << "a descriptor is never written";
    EXPECT_EQ(rig.session().m_send_buffer[0].get_available_samples(0), 0U);
    EXPECT_EQ(rig.session().m_receive_buffer[0].get_available_samples(0), available);
    for (size_t channel = 0; channel < 2; ++channel) {
        for (size_t i = 0; i < k_hop; ++i) { EXPECT_EQ(out.at(channel, i), 0.F); }
    }
    EXPECT_TRUE(out.slack_untouched());
    EXPECT_EQ(site.m_latched.load(), 1U);
}

// The adapter alone, without a manager. A slot a call does not carry is an empty tensor whose
// planes are NULL, whatever an earlier call stored there, and the caller's channel array of
// that slot is not read (it may be NULL). A miss function (ANIRA_MISS_CALLBACK) is handed
// these arrays as they are, so a pointer of an earlier call, whose memory may be gone by then,
// must not survive in a slot the present call left out. Both present forms, both sides; each
// slot is first shown to hold the caller's pointer, so a NULL afterwards is the adapter's doing.
TEST(TensorStems, TheFloatAdapterNullsThePlanesOfASlotACallDidNotCarry) {
    const InferenceConfig config = multi_config();  // a 3-channel stream and 3 Static values
    PlanarFloatAdapter adapter;
    adapter.prepare(config);

    std::array<std::vector<float>, 3> stream_in;
    std::array<std::vector<float>, 3> stream_out;
    std::array<const float*, 3> stream_in_channels{};
    std::array<float*, 3> stream_out_channels{};
    for (size_t channel = 0; channel < 3; ++channel) {
        stream_in.at(channel).assign(k_hop, 1.F);
        stream_out.at(channel).assign(k_hop, 0.F);
        stream_in_channels.at(channel) = stream_in.at(channel).data();
        stream_out_channels.at(channel) = stream_out.at(channel).data();
    }
    const std::array<float, 3> values_in{1.F, 2.F, 3.F};
    std::array<float, 3> values_out{};
    const std::array<const float*, 1> values_in_channels{values_in.data()};
    const std::array<float*, 1> values_out_channels{values_out.data()};

    const std::array<const float* const*, 2> in_both{stream_in_channels.data(),
                                                     values_in_channels.data()};
    const std::array<float* const*, 2> out_both{stream_out_channels.data(),
                                                values_out_channels.data()};
    const std::array<size_t, 2> count_both{k_hop, 3};
    const auto plane = [](const anira_tensor& tensor, uint32_t index) {
        return anira_tensor_plane(&tensor, index, ANIRA_DTYPE_F32);
    };
    const auto carry_both = [&] {
        const anira_tensor* inputs = adapter.present_inputs(in_both.data(), count_both.data());
        const anira_tensor* outputs = adapter.present_outputs(out_both.data(), count_both.data());
        // The control: every slot names the caller's memory after a call that carried it.
        EXPECT_EQ(plane(inputs[0], 2), stream_in[2].data());
        EXPECT_EQ(plane(inputs[1], 0), values_in.data());
        EXPECT_EQ(plane(outputs[0], 2), stream_out[2].data());
        EXPECT_EQ(plane(outputs[1], 0), values_out.data());
    };

    // The multi forms: the Static slot left out, its channel array NULL and not read.
    carry_both();
    {
        const std::array<const float* const*, 2> in_stream{stream_in_channels.data(), nullptr};
        const std::array<float* const*, 2> out_stream{stream_out_channels.data(), nullptr};
        const std::array<size_t, 2> count_stream{k_hop, 0};
        const anira_tensor* inputs = adapter.present_inputs(in_stream.data(), count_stream.data());
        const anira_tensor* outputs =
            adapter.present_outputs(out_stream.data(), count_stream.data());
        EXPECT_EQ(inputs[1].shape[1], 0);
        EXPECT_EQ(outputs[1].shape[1], 0);
        EXPECT_EQ(plane(inputs[1], 0), nullptr) << "a pointer of the earlier call";
        EXPECT_EQ(plane(outputs[1], 0), nullptr) << "a pointer of the earlier call";
        EXPECT_EQ(inputs[0].shape[1], static_cast<int64_t>(k_hop)) << "the carried slot";
        EXPECT_EQ(plane(inputs[0], 2), stream_in[2].data()) << "the carried slot";
        EXPECT_EQ(plane(outputs[0], 2), stream_out[2].data()) << "the carried slot";
    }

    // The single forms: slot 1 carried, every plane of slot 0 left out.
    carry_both();
    {
        const anira_tensor* inputs = adapter.present_input(1, values_in_channels.data(), 3);
        const anira_tensor* outputs = adapter.present_output(1, values_out_channels.data(), 3);
        EXPECT_EQ(inputs[0].shape[1], 0);
        EXPECT_EQ(outputs[0].shape[1], 0);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            EXPECT_EQ(plane(inputs[0], channel), nullptr) << "input channel " << channel;
            EXPECT_EQ(plane(outputs[0], channel), nullptr) << "output channel " << channel;
        }
        EXPECT_EQ(inputs[1].shape[1], 3) << "the carried slot";
        EXPECT_EQ(plane(inputs[1], 0), values_in.data()) << "the carried slot";
        EXPECT_EQ(plane(outputs[1], 0), values_out.data()) << "the carried slot";
    }
}

// ---- the run helper and the validator, on int16 data -------------------------------------------

TEST(TensorRun, ChannelRunResolvesEveryDescription) {
    constexpr size_t k_channels = 3;
    constexpr size_t k_count = 4;
    for (const Layout layout : {Layout::Planar,
                                Layout::PlanarOffset,
                                Layout::PlanarStrided,
                                Layout::Contiguous,
                                Layout::Interleaved,
                                Layout::Packed}) {
        HostBlock<int16_t> block(layout, k_channels, k_count, k_untouched_i16);
        const anira_tensor tensor = block.tensor(k_count, ANIRA_DTYPE_I16, false);
        for (size_t channel = 0; channel < k_channels; ++channel) {
            const tensor_run::Run run = tensor_run::channel_run(tensor, channel, sizeof(int16_t));
            for (size_t i = 0; i < k_count; ++i) {
                EXPECT_EQ(tensor_run::sample_of(run, i, sizeof(int16_t)),
                          static_cast<void*>(&block.at(channel, i)))
                    << name_of(layout) << ", channel " << channel << ", sample " << i;
            }
        }
    }
    EXPECT_EQ(tensor_run::dtype_size(ANIRA_DTYPE_I16), 2U);
    EXPECT_EQ(tensor_run::dtype_size(ANIRA_DTYPE_F32), 4U);
    EXPECT_EQ(tensor_run::dtype_size(ANIRA_DTYPE_F64), 8U);
    EXPECT_EQ(tensor_run::dtype_size(ANIRA_DTYPE_BOOL8), 1U);
    EXPECT_EQ(tensor_run::dtype_size(0), 0U);
}

TEST(TensorRun, CopyRunAndZeroRunMoveStridedInt16) {
    // Channel 1 of an interleaved stereo block into every third element of a run, and back.
    const std::array<int16_t, 8> interleaved{1, 101, 2, 102, 3, 103, 4, 104};
    std::array<int16_t, 12> strided{};
    strided.fill(k_untouched_i16);
    tensor_run::copy_run(strided.data(), 3, interleaved.data() + 1, 2, 4, sizeof(int16_t));
    EXPECT_EQ(strided,
              (std::array<int16_t, 12>{101, -7, -7, 102, -7, -7, 103, -7, -7, 104, -7, -7}));
    std::array<int16_t, 4> packed{};
    tensor_run::copy_run(packed.data(), 1, strided.data(), 3, 4, sizeof(int16_t));
    EXPECT_EQ(packed, (std::array<int16_t, 4>{101, 102, 103, 104}));
    // Two unit steps are a memmove: an overlapping move is legal.
    std::array<int16_t, 6> overlapping{1, 2, 3, 4, 5, 6};
    tensor_run::copy_run(overlapping.data() + 1, 1, overlapping.data(), 1, 5, sizeof(int16_t));
    EXPECT_EQ(overlapping, (std::array<int16_t, 6>{1, 1, 2, 3, 4, 5}));

    tensor_run::zero_run(strided.data(), 3, 3, sizeof(int16_t));
    EXPECT_EQ(strided, (std::array<int16_t, 12>{0, -7, -7, 0, -7, -7, 0, -7, -7, 104, -7, -7}));
    tensor_run::zero_run(packed.data() + 1, 1, 2, sizeof(int16_t));
    EXPECT_EQ(packed, (std::array<int16_t, 4>{101, 0, 0, 104}));
    tensor_run::zero_run(packed.data(), 1, 0, sizeof(int16_t));
    EXPECT_EQ(packed[0], 101);
}

// What a caller of the tensor stems checks first: every refusal with its status, and the
// order of the checks pinned by tensors that are wrong in two ways.
TEST(TensorRun, CheckHostTensorRefusesAMalformedDescriptor) {
    constexpr uint32_t k_channels = 2;
    constexpr size_t k_count = 4;
    HostBlock<int16_t> planar_block(Layout::Planar, k_channels, k_count, k_untouched_i16);
    HostBlock<int16_t> interleaved_block(Layout::Interleaved, k_channels, k_count, k_untouched_i16);
    const anira_tensor planar = planar_block.tensor(k_count, ANIRA_DTYPE_I16, false);
    const anira_tensor interleaved = interleaved_block.tensor(k_count, ANIRA_DTYPE_I16, false);
    const auto check = [](const anira_tensor& tensor, bool output = true) {
        return tensor_run::check_host_tensor(tensor, ANIRA_DTYPE_I16, k_channels, output);
    };
    EXPECT_EQ(check(planar), ANIRA_OK);
    EXPECT_EQ(check(interleaved), ANIRA_OK);
    EXPECT_EQ(check(empty_tensor(k_channels, ANIRA_DTYPE_I16)), ANIRA_OK) << "NULL memory";

    // The dtype: nothing converts.
    EXPECT_EQ(tensor_run::check_host_tensor(planar, ANIRA_DTYPE_F32, k_channels, true),
              ANIRA_ERROR_CONFIG);
    // A zeroed record, which is what a refused factory leaves.
    EXPECT_EQ(check(anira_tensor{}), ANIRA_ERROR_INVALID_ARGUMENT);

    anira_tensor wrong = interleaved;
    wrong.ndim = 3;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "the rank";
    wrong = interleaved;
    wrong.shape[1] = -1;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a negative extent";
    wrong = interleaved;
    wrong.shape[0] = 3;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "the channel count";
    wrong = planar;
    wrong.domain = ANIRA_DOMAIN_CUDA;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "planar on a device domain";
    wrong = planar;
    wrong.domain = ANIRA_DOMAIN_HOST_PINNED;
    EXPECT_EQ(check(wrong), ANIRA_OK) << "page-locked planes";
    wrong = planar;
    wrong.handle.planes.count = 3;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "count != C";
    wrong = planar;
    wrong.handle.planes.ptrs = nullptr;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "no plane array";
    std::array<void*, 2> one_null{planar_block.planes()[0], nullptr};
    wrong = planar;
    wrong.handle.planes.ptrs = one_null.data();
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a NULL plane";
    wrong = interleaved;
    wrong.handle.host.ptr = nullptr;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "NULL memory with samples";
    wrong = interleaved;
    wrong.strides[1] = -2;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a negative stride";
    wrong = interleaved;
    wrong.strides[1] = 0;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a stride of 0 over 4 samples";
    wrong = interleaved;
    wrong.strides[0] = 0;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "two channels on one run";
    wrong = interleaved;
    wrong.byte_offset = 1;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a run between two elements";
    wrong = interleaved;
    wrong.flags |= static_cast<uint32_t>(ANIRA_TENSOR_READ_ONLY);
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "a read-only output";
    EXPECT_EQ(check(wrong, false), ANIRA_OK) << "a read-only input";
    wrong = interleaved;
    wrong.flags |= 0x100U;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_NOT_SUPPORTED) << "an unknown flag";

    // The order: the rank before the domain, the domain before the flags, the flags before
    // the shape, the shape before the dtype, the dtype before the memory.
    wrong = interleaved;
    wrong.flags |= 0x100U;
    wrong.domain = ANIRA_DOMAIN_CUDA;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "the domain before the flags";
    wrong = interleaved;
    wrong.flags |= 0x100U;
    wrong.shape[0] = 3;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_NOT_SUPPORTED) << "the flags before the shape";
    wrong = interleaved;
    wrong.shape[0] = 3;
    wrong.dtype = ANIRA_DTYPE_F32;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_INVALID_ARGUMENT) << "the shape before the dtype";
    wrong = interleaved;
    wrong.dtype = ANIRA_DTYPE_F32;
    wrong.handle.host.ptr = nullptr;
    EXPECT_EQ(check(wrong), ANIRA_ERROR_CONFIG) << "the dtype before the memory";
    // The memory of an empty tensor is not looked at.
    wrong = interleaved;
    wrong.shape[1] = 0;
    wrong.handle.host.ptr = nullptr;
    wrong.strides[1] = -2;
    EXPECT_EQ(check(wrong), ANIRA_OK);
}
