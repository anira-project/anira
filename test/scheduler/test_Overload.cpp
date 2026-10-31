#include <anira/abi/engine.h>
#include <anira/abi/enums.h>
#include <anira/abi/export.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/scheduler/SessionElement.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../abi/float_face.h"
#include "../abi/handler_support.h"
#include "gtest/gtest.h"

// An overload, made deterministic: an engine behind a gate that holds every inference for a
// span of calls (a spike: every slot of the session ends up busy), a sample-index ramp in, and
// the output checked sample by sample. Outside the spike every call waits until every claimed
// slot has completed, and the spike ends between two calls, so no inference completes while a
// call runs and the run does not depend on timing. The engine doubles its input, so an
// inferred sample, the dry input and a fill tell apart. What the scheduler promises:
//  - a hop that finds no free slot is dropped and costs exactly one hop of the miss policy's
//    fill at its own stream position: zeros, the dropped input hop itself (BYPASS), the head of
//    the last delivered block (HOLD_LAST), or what the host's function made of the hop
//    (CALLBACK); rt_error records ANIRA_ERROR_CAPACITY;
//  - a block whose inference is late is a miss (ANIRA_MISSED, count 0, the policy's block
//    fill), and the late result is discarded;
//  - every other sample is the model's output at exactly the reported latency, the window
//    starts stay on the hop grid, and the output is back at its latency once the late chunks
//    are collected; a late sample is never delivered.

namespace {

// The model: twice its input. Records the first sample of every window it ran, the stream
// position of the hop (the host writes every sample's own index).
class DoublingGate : public anira_test::GateEngine {
public:
    anira_status run(const anira_engine_ctx& ctx) noexcept override {
        const float* in = anira_tensor_data_f32(&ctx.inputs[0]);
        float* out = anira_tensor_data_f32(&ctx.outputs[0]);
        if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        const size_t n = anira_test::elements_of(ctx.outputs[0]);
        for (size_t i = 0; i < n; ++i) { out[i] = 2.0F * in[i]; }
        const std::lock_guard<std::mutex> lock(m_mutex);
        m_positions.push_back(static_cast<int64_t>(in[0]));
        return ANIRA_OK;
    }

    std::vector<int64_t> positions() {
        const std::lock_guard<std::mutex> lock(m_mutex);
        return m_positions;
    }

private:
    std::mutex m_mutex;
    std::vector<int64_t> m_positions;
};

// ANIRA_MISS_CALLBACK: every output sample the negative of its input sample, for a missed block
// (the call's own input) and for a dropped hop (the hop's input) alike; counts both shapes.
std::atomic<int> g_block_calls{0};
std::atomic<int> g_hop_calls{0};
std::atomic<int64_t> g_hop_samples{0};

anira_status ANIRA_CALL negate(anira_handler* /*handler*/,
                               const anira_tensor* inputs,
                               uint32_t num_inputs,
                               const anira_tensor* outputs,
                               uint32_t num_outputs,
                               void* /*user_data*/) ANIRA_NONBLOCKING {
    if (num_inputs != 1 || num_outputs != 1) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const int64_t samples = outputs[0].shape[1];
    if (inputs[0].shape[1] < samples) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    // A hop is the manager's packed [1, hop] tensor; a block is the host's planar one.
    const bool hop = (outputs[0].flags & ANIRA_TENSOR_PLANAR) == 0;
    (hop ? g_hop_calls : g_block_calls).fetch_add(1);
    if (hop) { g_hop_samples.store(samples); }
    const auto first = [hop](const anira_tensor& tensor) {
        return static_cast<float*>(hop ? anira_tensor_data(&tensor, ANIRA_DTYPE_F32)
                                       : anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_F32));
    };
    const float* in = first(inputs[0]);
    float* out = first(outputs[0]);
    if (in == nullptr || out == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    for (int64_t i = 0; i < samples; ++i) { out[i] = -in[i]; }
    return ANIRA_OK;
}

struct Case {
    std::string m_name;
    double m_host_block_min;
    uint32_t m_host_block_max;
    std::vector<uint32_t> m_blocks;  ///< the host's blocks, cycled
    int64_t m_hop = 480;
    size_t m_spike_start = 20;
    size_t m_spike_calls = 8;
    size_t m_calls = 60;
    bool m_stateful = false;
    anira_miss_policy m_policy = ANIRA_MISS_ZEROS;
};

struct Call {
    uint32_t m_block = 0;
    int64_t m_first = 0;  ///< the stream position of its first sample
    anira_status m_status = ANIRA_OK;
    size_t m_delivered = 0;
    std::vector<float> m_out;
};

struct Recording {
    int64_t m_latency = 0;
    std::vector<Call> m_calls;
    std::vector<int64_t> m_positions;  ///< the window starts the engine ran, in run order
    anira_status m_rt_error = ANIRA_OK;
};

// Every slot of the session completed (free, or done and waiting for its collection).
bool settled(const anira::SessionElement& session) {
    return std::ranges::all_of(session.m_inference_queue, [](const auto& chunk) {
        return chunk->m_free.load() || chunk->m_done_atomic.load();
    });
}

Recording record(const Case& c) {
    Recording result;
    anira::ModelConfig model = anira_test::custom_model(2);
    model.input(anira_test::stream_tensor("in", 1, c.m_hop, c.m_hop));
    model.output(anira_test::stream_tensor("out", 1, c.m_hop, c.m_hop));
    if (c.m_stateful) { model.state(ANIRA_MODEL_STATEFUL); }
    DoublingGate engine;
    {
        const anira_test::Context context(2);
        anira_test::Handler handler(context, model, {}, {}, engine.engine());
        const anira_test::DestroyFirst guard(handler, &engine);
        anira::ContractHandle contract =
            anira_test::explicit_contract(c.m_host_block_max, 48000.0, c.m_policy, 0.0, 5.0);
        contract.hard_host_blocks(c.m_host_block_min, c.m_host_block_max, 48000.0);
        if (c.m_policy == ANIRA_MISS_CALLBACK) { contract.hard_miss_fn(&negate, nullptr); }
        EXPECT_EQ(handler.prepare(contract), ANIRA_OK) << handler.m_err.message;
        anira_handler* h = handler.m_handler;
        result.m_latency = static_cast<int64_t>(anira_handler_get_latency(h, 0));
        const std::shared_ptr<anira::SessionElement> session = anira_test::session_of(h);
        if (session == nullptr) { return result; }
        std::vector<float> buffer(c.m_host_block_max);
        int64_t pushed = 0;
        for (size_t k = 0; k < c.m_calls; ++k) {
            const uint32_t block = c.m_blocks[k % c.m_blocks.size()];
            if (k == c.m_spike_start) { engine.m_open.store(false); }
            for (uint32_t i = 0; i < block; ++i) { buffer[i] = static_cast<float>(pushed + i); }
            const std::array<float*, 1> channels{buffer.data()};
            const anira_tensor io = anira_test::planar_f32(channels.data(), 1, block);
            Call call{.m_block = block, .m_first = pushed};
            call.m_status = anira_handler_process(h, &io, 0, &io, 0, &call.m_delivered);
            call.m_out.assign(buffer.begin(), buffer.begin() + block);
            result.m_calls.push_back(std::move(call));
            pushed += block;
            // The spike ends between two calls: no inference completes while a call runs.
            if (k + 1 == c.m_spike_start + c.m_spike_calls) { engine.m_open.store(true); }
            if (engine.m_open.load()) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                while (!settled(*session) && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                EXPECT_TRUE(settled(*session)) << "call " << k << ": the slots did not settle";
            }
        }
        result.m_rt_error = anira_handler_rt_error(h);
    }
    result.m_positions = engine.positions();
    return result;
}

// What the run delivered against the promise above. Returns the dropped hops.
size_t expect_the_overload_contract(const Case& c, const Recording& r) {
    SCOPED_TRACE(c.m_name);
    // The window starts: on the hop grid, each at most once, in stream order on a stateful
    // model (one inference at a time, in submission order).
    std::set<int64_t> inferred;
    size_t off_grid = 0;
    for (size_t k = 0; k < r.m_positions.size(); ++k) {
        const int64_t start = r.m_positions[k];
        if (start % c.m_hop != 0 && off_grid++ == 0) {
            ADD_FAILURE() << "window " << k << " starts at " << start << ", off the hop grid";
        }
        EXPECT_TRUE(inferred.insert(start).second) << "window " << start << " ran twice";
        if (c.m_stateful && k > 0) {
            EXPECT_GT(start, r.m_positions[k - 1]) << "window " << k << " ran out of order";
        }
    }
    EXPECT_EQ(off_grid, 0U) << "windows off the hop grid";
    const int64_t pushed = r.m_calls.back().m_first + r.m_calls.back().m_block;
    size_t dropped = 0;
    for (int64_t start = 0; start + c.m_hop <= pushed; start += c.m_hop) {
        dropped += inferred.contains(start) ? 0 : 1;
    }
    EXPECT_EQ(r.m_rt_error, dropped > 0 ? ANIRA_ERROR_CAPACITY : ANIRA_OK)
        << "a dropped hop records ANIRA_ERROR_CAPACITY";

    // The call during which a hop was dropped: the one whose push completed it.
    const auto dropping_call = [&r](int64_t hop_end) {
        for (size_t k = 0; k < r.m_calls.size(); ++k) {
            if (r.m_calls[k].m_first + r.m_calls[k].m_block >= hop_end) { return k; }
        }
        return r.m_calls.size();
    };
    // HOLD_LAST: the block the output delivered last before call k.
    const auto held_before = [&r](size_t k) -> const std::vector<float>* {
        for (size_t j = k; j-- > 0;) {
            if (r.m_calls[j].m_status != ANIRA_MISSED) { return &r.m_calls[j].m_out; }
        }
        return nullptr;
    };
    // The value of output sample n delivered on time.
    const auto on_time = [&](int64_t n) {
        const int64_t input = n - r.m_latency;
        if (input < 0) { return 0.0F; }
        const int64_t start = (input / c.m_hop) * c.m_hop;
        if (inferred.contains(start)) { return 2.0F * static_cast<float>(input); }
        switch (c.m_policy) {
            case ANIRA_MISS_BYPASS: return static_cast<float>(input);
            case ANIRA_MISS_CALLBACK: return -static_cast<float>(input);
            case ANIRA_MISS_HOLD_LAST: {
                const std::vector<float>* held = held_before(dropping_call(start + c.m_hop));
                const auto offset = static_cast<size_t>(input - start);
                return held != nullptr && offset < held->size() ? (*held)[offset] : 0.0F;
            }
            default: return 0.0F;
        }
    };
    size_t wrong = 0;
    for (size_t k = 0; k < r.m_calls.size(); ++k) {
        const Call& call = r.m_calls[k];
        if (call.m_status == ANIRA_MISSED) {
            EXPECT_EQ(call.m_delivered, 0U) << "call " << k;
            const std::vector<float>* held = held_before(k);
            for (uint32_t i = 0; i < call.m_block; ++i) {
                float fill = 0.0F;
                const auto input = static_cast<float>(call.m_first + i);
                if (c.m_policy == ANIRA_MISS_BYPASS) { fill = input; }
                if (c.m_policy == ANIRA_MISS_CALLBACK) { fill = -input; }
                if (c.m_policy == ANIRA_MISS_HOLD_LAST && held != nullptr && i < held->size()) {
                    fill = (*held)[i];
                }
                if (call.m_out[i] != fill && wrong++ == 0) {
                    ADD_FAILURE() << "call " << k << " missed, sample " << i << ": "
                                  << call.m_out[i] << ", the policy's fill " << fill;
                }
            }
            continue;
        }
        EXPECT_EQ(call.m_status, ANIRA_OK) << "call " << k;
        EXPECT_EQ(call.m_delivered, call.m_block) << "call " << k;
        for (uint32_t i = 0; i < call.m_block; ++i) {
            const float expected = on_time(call.m_first + i);
            if (call.m_out[i] != expected && wrong++ == 0) {
                ADD_FAILURE() << "call " << k << ", sample " << i << " (output " << call.m_first + i
                              << "): " << call.m_out[i] << ", expected " << expected;
            }
        }
    }
    EXPECT_EQ(wrong, 0U) << "samples neither the stream at its latency nor the policy's fill";
    // Recovered: the last ten calls deliver the model's output at its latency.
    for (size_t k = r.m_calls.size() - 10; k < r.m_calls.size(); ++k) {
        EXPECT_EQ(r.m_calls[k].m_status, ANIRA_OK) << "call " << k << " after the recovery";
        const int64_t input = r.m_calls[k].m_first - r.m_latency;
        EXPECT_TRUE(inferred.contains((input / c.m_hop) * c.m_hop))
            << "call " << k << " after the recovery plays a dropped hop";
    }
    return dropped;
}

std::vector<Case> host_block_cases() {
    return {
        Case{.m_name = "block 1024 > hop 480",
             .m_host_block_min = 1024,
             .m_host_block_max = 1024,
             .m_blocks = {1024}},
        Case{.m_name = "block 480 = hop 480",
             .m_host_block_min = 480,
             .m_host_block_max = 480,
             .m_blocks = {480}},
        Case{.m_name = "block 256 < hop 480",
             .m_host_block_min = 256,
             .m_host_block_max = 256,
             .m_blocks = {256},
             .m_calls = 80},
        Case{.m_name = "range 1..1024, blocks of 1024",
             .m_host_block_min = 1,
             .m_host_block_max = 1024,
             .m_blocks = {1024}},
        Case{.m_name = "range 1..1024, varying blocks",
             .m_host_block_min = 1,
             .m_host_block_max = 1024,
             .m_blocks = {1024, 37, 1, 1024, 512, 1, 1, 1023, 64, 32, 1024, 3},
             .m_spike_calls = 24,
             .m_calls = 90},
    };
}

void expect_every_host_block_case(anira_miss_policy policy, const char* word) {
    for (Case c : host_block_cases()) {
        c.m_policy = policy;
        c.m_name += std::string(", ") + word;
        const Recording r = record(c);
        ASSERT_FALSE(r.m_calls.empty()) << c.m_name;
        const size_t dropped = expect_the_overload_contract(c, r);
        EXPECT_GT(dropped, 0U) << c.m_name << ": the spike dropped no hop";
        EXPECT_TRUE(
            std::ranges::any_of(r.m_calls,
                                [](const Call& call) { return call.m_status == ANIRA_MISSED; }))
            << c.m_name << ": the spike missed no block";
    }
}

}  // namespace

// One spike in every host block range the Hard contract allows: a block above the hop (several hops
// a call), at the hop, below it (a hop every few calls) and the 1..host_block_max range, at its
// maximum and with varying blocks. Every one drops hops and misses blocks, and the output holds to
// the promise above, under each miss policy.
TEST(Overload, ASpikeCostsTheDroppedHopsUnderZeros) {
    expect_every_host_block_case(ANIRA_MISS_ZEROS, "zeros");
}

TEST(Overload, ASpikeCostsTheDroppedHopsUnderBypass) {
    expect_every_host_block_case(ANIRA_MISS_BYPASS, "bypass");
}

TEST(Overload, ASpikeCostsTheDroppedHopsUnderHoldLast) {
    expect_every_host_block_case(ANIRA_MISS_HOLD_LAST, "hold_last");
}

// The host's function fills a missed block (the host's planar tensors) and a dropped hop (the
// manager's packed [channels, hop] tensors over the hop) alike, on the driver thread.
TEST(Overload, ASpikeCostsTheDroppedHopsUnderACallback) {
    g_block_calls.store(0);
    g_hop_calls.store(0);
    expect_every_host_block_case(ANIRA_MISS_CALLBACK, "callback");
    EXPECT_GT(g_block_calls.load(), 0) << "no missed block reached the function";
    EXPECT_GT(g_hop_calls.load(), 0) << "no dropped hop reached the function";
    EXPECT_EQ(g_hop_samples.load(), 480) << "a dropped hop is one hop of the output";
}

// A stateful model: its inferences run one at a time in stream order, across the drop too. A
// dropped hop never reaches the model, so its state continues from the last hop it ran, as if
// the input had jumped by the dropped hops.
TEST(Overload, AStatefulModelRunsItsWindowsInOrderAcrossADrop) {
    Case c = host_block_cases().front();
    c.m_name += ", stateful";
    c.m_stateful = true;
    const Recording r = record(c);
    ASSERT_FALSE(r.m_calls.empty());
    EXPECT_GT(expect_the_overload_contract(c, r), 0U);
}
