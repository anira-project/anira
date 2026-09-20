// The recording of the host<->ring copy path under the float*** functions of InferenceManager:
// process_input, process_output and the three miss policies, driven through every public
// float*** function over engine-free sessions, and compared with the transcripts of
// copy_path_oracle_golden.h. The golden transcripts were written by the float copy core as it
// stood before it was rewritten over anira_tensor, so a core that passes reproduces that one
// bit for bit: the delivered counts, the miss flag, every float a call wrote and every float
// it left alone. test/support/copy_oracle.h says what a transcript is and why a scenario is
// a pure function of its calls; test/abi/test_HandlerCopyOracle.cpp is the twin over the _f32
// Hard entries.

#include <anira/CoreConfig.h>
#include <anira/InferenceConfig.h>
#include <anira/PrePostProcessor.h>
#include <anira/abi/enums.h>
#include <anira/scheduler/Core.h>
#include <anira/scheduler/InferenceManager.h>
#include <anira/scheduler/SessionElement.h>
#include <anira/utils/HostConfig.h>
#include <anira/utils/InferenceBackend.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../abi/handler_support.h"
#include "../support/copy_oracle.h"
#include "copy_path_oracle_golden.h"
#include "gtest/gtest.h"

using namespace anira;

namespace {

namespace oracle = anira_test::oracle;

constexpr size_t k_hop = 8;  // the window of every streamed tensor, and the host block

// ---- the models --------------------------------------------------------------------------------
// Engine-free: the session's custom backend is the gate. The stereo and the multi model run
// BackendBase::process through it (tensor i of the output is tensor i of the input); the two
// others have a model of their own (Model below), because BackendBase delivers zeros when the
// two sides differ in channels or samples.

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

/// Two channels in, two channels out.
InferenceConfig stereo_config(float blocking_ratio = 0.F) {
    return make_config(TensorShape({{1, 2, k_hop}}, {{1, 2, k_hop}}),
                       ProcessingSpec({2}, {2}),
                       blocking_ratio);
}

/// A three-channel stream and three Static values on either side.
InferenceConfig multi_config() {
    return make_config(TensorShape({{1, 3, k_hop}, {1, 3}}, {{1, 3, k_hop}, {1, 3}}),
                       ProcessingSpec({3, 1}, {3, 1}, {k_hop, 0}, {k_hop, 0}));
}

/// One channel in, two out (oracle::FanOutGate): a BYPASS miss has one input channel for two
/// output channels.
InferenceConfig mono_to_stereo_config() {
    return make_config(TensorShape({{1, 1, k_hop}}, {{1, 2, k_hop}}), ProcessingSpec({1}, {2}));
}

/// Four Static values in, one streamed channel out (oracle::ParamRampGate): the session is
/// driven by its pops.
InferenceConfig generator_config() {
    return make_config(TensorShape({{1, 4}}, {{1, 1, k_hop}}),
                       ProcessingSpec({1}, {1}, {0}, {k_hop}));
}

/// Blocks of up to k_hop samples at 800 Hz: a hop lasts 10 ms against 1 ms of inference, so
/// the latency is small and a transcript stays short.
HostConfig host_config() {
    return {static_cast<float>(k_hop), 800.F, true};
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
    bool m_null_unused = false;  ///< a slot whose count is 0 is handed over as NULL
    bool m_settle = true;        ///< false leaves the call's inferences at the gate
};

constexpr CallOptions k_plain{.m_in_place = false, .m_null_unused = false, .m_settle = true};
constexpr CallOptions k_in_place{.m_in_place = true, .m_null_unused = false, .m_settle = true};
constexpr CallOptions k_null_unused{.m_in_place = false, .m_null_unused = true, .m_settle = true};
constexpr CallOptions k_gate_closed{.m_in_place = false, .m_null_unused = false, .m_settle = false};
constexpr CallOptions k_in_place_gate_closed{.m_in_place = true,
                                             .m_null_unused = false,
                                             .m_settle = false};

/// How the rig keeps a scenario deterministic (test/support/copy_oracle.h).
enum class Mode {
    Clocked,  ///< the gate is closed during every call and opened by the settle between two
    Waiting   ///< the gate stays open and the session is non-real-time: every collection
              ///< waits for every submitted inference, so the waiting stems can be recorded
};

/// What the session's custom backend computes behind the gate.
enum class Model { PassThrough, FanOut, ParamRamp };

std::unique_ptr<anira_test::GateBackend> make_gate(Model model, InferenceConfig& config) {
    switch (model) {
        case Model::FanOut: return std::make_unique<oracle::FanOutGate>(config);
        case Model::ParamRamp: return std::make_unique<oracle::ParamRampGate>(config);
        case Model::PassThrough: break;
    }
    return std::make_unique<anira_test::GateBackend>(config);
}

class ManagerRig {
public:
    ManagerRig(InferenceConfig config,
               anira_miss_policy policy,
               Mode mode = Mode::Clocked,
               Model model = Model::PassThrough)
        : m_config(std::move(config))
        , m_pp_processor(m_config)
        , m_gate(make_gate(model, m_config))
        , m_manager(m_pp_processor, m_config, m_gate.get(), CoreConfig(2))
        , m_mode(mode)
        , m_positions(m_config.get_tensor_input_shape().size(), 0) {
        m_manager.set_miss_policy(policy);
        m_manager.prepare(host_config());
        for (const std::shared_ptr<SessionElement>& session : Core::get_sessions()) {
            if (session->m_session_id == m_manager.get_session_id()) { m_session = session; }
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
    ~ManagerRig() { m_gate->m_open.store(true); }  // before the manager releases the session
    ManagerRig(const ManagerRig&) = delete;
    ManagerRig& operator=(const ManagerRig&) = delete;

    /// A process or a pop stem; a pop takes no input counts.
    void call(Stem stem,
              const std::vector<size_t>& in_counts,
              const std::vector<size_t>& out_counts,
              CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const bool pops =
            stem == Stem::PopData || stem == Stem::PopDataUntil || stem == Stem::PopDataWait;
        const size_t call = ++m_calls;
        const size_t shared_length = options.m_in_place ? out_counts[0] : 0;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, shared_length, call);
        const oracle::Snapshot inputs_before = oracle::snapshot(inputs);
        std::vector<const float* const*> in_planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            in_planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        std::vector<oracle::SlotBlock> outputs;
        outputs.reserve(out_counts.size());
        std::vector<float* const*> out_planes;
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            outputs.emplace_back(m_config.get_postprocess_output_channels()[slot],
                                 out_counts[slot]);
            const bool unused = options.m_null_unused && out_counts[slot] == 0;
            const bool shared = options.m_in_place && slot == 0;
            float* const* planes = shared ? inputs[0].planes() : outputs[slot].planes();
            out_planes.push_back(unused ? nullptr : planes);
        }
        std::vector<size_t> num_in = in_counts;
        std::vector<size_t> num_out = out_counts;
        Core::WaitOutcome outcome = Core::WaitOutcome::Done;
        const size_t* returned = nullptr;
        switch (stem) {
            case Stem::Process:
                returned = m_manager.process(in_planes.data(),
                                             num_in.data(),
                                             out_planes.data(),
                                             num_out.data());
                break;
            case Stem::ProcessNowait:
                returned = m_manager.process_nowait(in_planes.data(),
                                                    num_in.data(),
                                                    out_planes.data(),
                                                    num_out.data());
                break;
            case Stem::ProcessWait:
                returned = m_manager.process_wait(in_planes.data(),
                                                  num_in.data(),
                                                  out_planes.data(),
                                                  num_out.data(),
                                                  std::chrono::steady_clock::duration::max(),
                                                  outcome);
                break;
            case Stem::PopData:
                returned = m_manager.pop_data(out_planes.data(), num_out.data());
                break;
            case Stem::PopDataUntil:
                returned = m_manager.pop_data(out_planes.data(),
                                              num_out.data(),
                                              std::chrono::steady_clock::time_point::max());
                break;
            case Stem::PopDataWait:
                returned = m_manager.pop_data_wait(out_planes.data(),
                                                   num_out.data(),
                                                   std::chrono::steady_clock::duration::max(),
                                                   outcome);
                break;
        }

        std::string header = "#" + std::to_string(call) + " " + name_of(stem);
        if (!pops) { header += " in=" + oracle::format_counts(in_counts); }
        header += " out=" + oracle::format_counts(out_counts);
        if (options.m_in_place) { header += " in-place"; }
        if (options.m_null_unused) { header += " null-unused"; }
        header += " ->";
        if (!pops) { header += " in=" + oracle::format_counts(num_in); }
        header += " out=" + oracle::format_counts(num_out);
        header += returned == num_out.data() ? " returned=out" : " returned=other";
        header += std::string(" missed=") + (m_manager.last_block_missed() ? "1" : "0");
        if (stem == Stem::ProcessWait || stem == Stem::PopDataWait) {
            header += std::string(" outcome=") + name_of(outcome);
        }
        m_transcript.add(header + " " + oracle::format_rings(*m_session));
        for (size_t slot = 0; slot < out_counts.size(); ++slot) {
            if (out_planes[slot] == nullptr) { continue; }
            if (options.m_in_place && slot == 0) {
                m_transcript.add_block("io0", inputs[0], out_counts[0]);
            } else {
                m_transcript.add_block("out" + std::to_string(slot),
                                       outputs[slot],
                                       out_counts[slot]);
            }
        }
        oracle::expect_inputs_untouched(inputs, inputs_before, options.m_in_place, call);
        if (options.m_settle) { settle(); }
    }

    void push(const std::vector<size_t>& in_counts, CallOptions options = k_plain) {
        if (m_session == nullptr) { return; }
        const size_t call = ++m_calls;
        std::vector<oracle::SlotBlock> inputs =
            oracle::make_inputs(m_config, m_positions, in_counts, 0, call);
        const oracle::Snapshot inputs_before = oracle::snapshot(inputs);
        std::vector<const float* const*> in_planes;
        for (size_t slot = 0; slot < in_counts.size(); ++slot) {
            const bool unused = options.m_null_unused && in_counts[slot] == 0;
            in_planes.push_back(unused ? nullptr : inputs[slot].planes());
        }
        std::vector<size_t> num_in = in_counts;
        m_manager.push_data(in_planes.data(), num_in.data());
        std::string header =
            "#" + std::to_string(call) + " push_data in=" + oracle::format_counts(in_counts);
        if (options.m_null_unused) { header += " null-unused"; }
        m_transcript.add(header + " -> in=" + oracle::format_counts(num_in) + " " +
                         oracle::format_rings(*m_session));
        oracle::expect_inputs_untouched(inputs, inputs_before, false, call);
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

    const oracle::Transcript& transcript() const { return m_transcript; }

private:
    InferenceConfig m_config;
    PrePostProcessor m_pp_processor;
    std::unique_ptr<anira_test::GateBackend> m_gate;
    InferenceManager m_manager;
    Mode m_mode;
    std::shared_ptr<SessionElement> m_session;
    std::vector<size_t> m_positions;  ///< per input slot, the samples pushed so far
    size_t m_calls = 0;
    oracle::Transcript m_transcript;
};

// ---- the scenarios -----------------------------------------------------------------------------
// The figures in the comments are those of the stereo model under host_config(): a latency
// of 15 samples, a send ring of 15, a receive ring of 47, 4 inference structs, a hold
// capacity of 8 (the host block). The golden transcripts print them.

/// Blocks below, at and above the hop, a block of 0 samples, separate memory and in place,
/// the 2.x dispatcher, two calls whose counts differ (a request the ring cannot serve, then a
/// request below the pushed count), the catch-up after the misses, a reset in the middle of
/// the stream, and a push larger than the send ring, of which the ring keeps the tail.
void run_block_sizes(anira_miss_policy policy, const char* name, std::string_view golden) {
    ManagerRig rig(stereo_config(), policy);
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
    rig.call(Stem::Process, {8}, {8});  // blocking_ratio 0: the dispatcher takes the nowait stem
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
    oracle::expect_golden(name, golden, rig.transcript());
}

/// The starved block: the gate stays closed from call 3 on, so the ring runs dry while four
/// inferences wait (one per struct, none refused). A delivered block above the hold capacity,
/// then misses at a request equal to the hold capacity, above it, in place, below the pushed
/// count and on a pop; the gate opens, the catch-up discards the late blocks; a reset, after
/// which HOLD_LAST holds nothing.
void run_starved_blocks(anira_miss_policy policy, const char* name, std::string_view golden) {
    ManagerRig rig(stereo_config(), policy);
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
    oracle::expect_golden(name, golden, rig.transcript());
}

/// The split calls: pushes and pops of unequal sizes, a pop of 0 samples, a pop the ring
/// cannot serve, the deadline form of pop_data on a session without a blocking ratio (it
/// collects nothing and pops what is there).
void run_push_pop(anira_miss_policy policy, const char* name, std::string_view golden) {
    ManagerRig rig(stereo_config(), policy);
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
    oracle::expect_golden(name, golden, rig.transcript());
}

/// Two slots on either side, one of them Static: both at once, one at a time with the other
/// handed over as NULL, a Static request above and below the value count, the split calls, a
/// starved request on both outputs at once, the catch-up and a reset.
void run_multi_slot(anira_miss_policy policy, const char* name, std::string_view golden) {
    ManagerRig rig(multi_config(), policy);
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
    oracle::expect_golden(name, golden, rig.transcript());
}

}  // namespace

TEST(CopyPathOracle, StereoBlockSizes) {
    run_block_sizes(ANIRA_MISS_ZEROS,
                    "k_manager_block_sizes_zeros",
                    oracle::k_manager_block_sizes_zeros);
    run_block_sizes(ANIRA_MISS_HOLD_LAST,
                    "k_manager_block_sizes_hold_last",
                    oracle::k_manager_block_sizes_hold_last);
    run_block_sizes(ANIRA_MISS_BYPASS,
                    "k_manager_block_sizes_bypass",
                    oracle::k_manager_block_sizes_bypass);
}

TEST(CopyPathOracle, StereoStarvedBlocks) {
    run_starved_blocks(ANIRA_MISS_ZEROS,
                       "k_manager_starved_zeros",
                       oracle::k_manager_starved_zeros);
    run_starved_blocks(ANIRA_MISS_HOLD_LAST,
                       "k_manager_starved_hold_last",
                       oracle::k_manager_starved_hold_last);
    run_starved_blocks(ANIRA_MISS_BYPASS,
                       "k_manager_starved_bypass",
                       oracle::k_manager_starved_bypass);
}

TEST(CopyPathOracle, StereoPushPop) {
    run_push_pop(ANIRA_MISS_ZEROS, "k_manager_push_pop_zeros", oracle::k_manager_push_pop_zeros);
    run_push_pop(ANIRA_MISS_HOLD_LAST,
                 "k_manager_push_pop_hold_last",
                 oracle::k_manager_push_pop_hold_last);
}

TEST(CopyPathOracle, MultiSlotWithStaticTensors) {
    run_multi_slot(ANIRA_MISS_ZEROS, "k_manager_multi_zeros", oracle::k_manager_multi_zeros);
    run_multi_slot(ANIRA_MISS_HOLD_LAST,
                   "k_manager_multi_hold_last",
                   oracle::k_manager_multi_hold_last);
    run_multi_slot(ANIRA_MISS_BYPASS, "k_manager_multi_bypass", oracle::k_manager_multi_bypass);
}

// One input channel for two output channels: a BYPASS miss copies the channel the input has
// and zero-fills the other, for equal counts, a longer request and a shorter one.
TEST(CopyPathOracle, MonoToStereoBypass) {
    ManagerRig rig(mono_to_stereo_config(), ANIRA_MISS_BYPASS, Mode::Clocked, Model::FanOut);
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {13}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {8}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {3}, {13}, k_gate_closed);
    rig.call(Stem::ProcessNowait, {8}, {3}, k_gate_closed);
    rig.settle();
    rig.call(Stem::ProcessNowait, {8}, {8});
    rig.call(Stem::ProcessNowait, {8}, {8});
    oracle::expect_golden("k_manager_mono_to_stereo_bypass",
                          oracle::k_manager_mono_to_stereo_bypass,
                          rig.transcript());
}

// A generator: the Static input is pushed, the pops drive the inferences (request_output),
// and a process call does both. BYPASS has no anchored input here and zero-fills.
TEST(CopyPathOracle, GeneratorPulledByItsPops) {
    ManagerRig rig(generator_config(), ANIRA_MISS_BYPASS, Mode::Clocked, Model::ParamRamp);
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
    oracle::expect_golden("k_manager_generator", oracle::k_manager_generator, rig.transcript());
}

// The waiting stems, on a session with a blocking ratio and in non-real-time mode, where
// every collection waits for every submitted inference whatever the budget: the 2.x
// dispatcher (which takes process_wait here), process_wait, the deadline form of pop_data,
// pop_data_wait and the non-waiting stems beside them, a request none of them can serve.
TEST(CopyPathOracle, WaitingStems) {
    ManagerRig rig(stereo_config(0.5F), ANIRA_MISS_BYPASS, Mode::Waiting);
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
    oracle::expect_golden("k_manager_waiting_stems",
                          oracle::k_manager_waiting_stems,
                          rig.transcript());
}
